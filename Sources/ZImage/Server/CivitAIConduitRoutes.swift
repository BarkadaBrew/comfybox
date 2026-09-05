// CivitAIConduitRoutes.swift — request/response plumbing for
// GET /v1/civitai/search, POST /v1/civitai/harvest, GET /v1/civitai/repo (#234).
//
// Kept as pure, network-free types and functions so they're unit-testable in
// isolation from WarmServer (which owns the actual route dispatch + the
// server's resolved CivitAI API key — see the civitai*Route methods added to
// WarmServer.swift). Only CivitAIHarvestRunner.run touches the network, and
// it's a thin wrapper around CivitAIClient.searchModels() + the extraction
// function below.
//
// `request.queryParameters` (HTTPRequest.swift) percent-decodes values as of
// comfybox#380 — but HTTP request lines still can't carry a literal space, so
// callers must percent-encode it either way, and the decode is now handled
// once, centrally, before these types ever see the dict (this file no longer
// decodes on its own — doing so on top of an already-decoded value risked a
// double-decode). `CivitAISortPeriodParsing` still accepts snake_case/kebab-
// case/any-case forms instead of CivitAI's raw sort/period values
// ("most_downloaded", "Most Downloaded", "MOST-DOWNLOADED", …) and falls back
// to the default on anything unrecognized — used for both the GET query
// string and the POST JSON body, so the two routes behave identically; this
// leniency is about the wire format callers choose to send, independent of
// percent-decoding.

import Foundation

// MARK: - Sort/period parsing

enum CivitAISortPeriodParsing {
  static func parseSort(_ raw: String?) -> CivitAIClient.SortOrder {
    guard let raw, !raw.isEmpty else { return .mostDownloaded }
    let normalized = normalize(raw)
    return CivitAIClient.SortOrder.allCases.first { normalize($0.rawValue) == normalized } ?? .mostDownloaded
  }

  static func parsePeriod(_ raw: String?) -> CivitAIClient.Period {
    guard let raw, !raw.isEmpty else { return .allTime }
    let normalized = normalize(raw)
    return CivitAIClient.Period.allCases.first {
      normalize($0.rawValue) == normalized || $0.rawValue.lowercased() == raw.lowercased()
    } ?? .allTime
  }

  private static func normalize(_ s: String) -> String {
    s.replacingOccurrences(of: "_", with: " ")
      .replacingOccurrences(of: "-", with: " ")
      .lowercased()
  }
}

// MARK: - Site allowlist (adversarial review P1-1)

/// The ONE shared place the `site` parameter is turned into a base URL, for
/// BOTH the search and harvest routes (and anything added later).
///
/// Why strict: `CivitAIClient` attaches the resolved Bearer API key to every
/// request it makes. The previous code interpolated caller input straight
/// into `URL(string: "https://\(site)")`, so `site=attacker.com` (or
/// `site=evil.example/%2e%2e`) would ship the CivitAI key to an arbitrary
/// host — SSRF plus credential exfiltration. Only the two known CivitAI
/// hosts are ever allowed; anything else maps to `nil`, which WarmServer's
/// route handlers turn into HTTP 400 before a client is even constructed.
enum CivitAIHostAllowlist {
  /// Exactly the two hosts the conduit may talk to. civitai.red is the same
  /// API, NSFW-default mirror (see CivitAIBrowserView.CivitAISource).
  static let allowedSites: [String] = ["civitai.com", "civitai.red"]

  /// `nil` unless `site` (case-insensitively, whitespace-trimmed) is exactly
  /// one of `allowedSites`.
  static func baseURL(forSite site: String) -> URL? {
    let normalized = site.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard allowedSites.contains(normalized) else { return nil }
    return URL(string: "https://\(normalized)")
  }

  /// The HTTP 400 body both routes return for a disallowed site.
  static func rejectionMessage(forSite site: String) -> String {
    "Invalid site \"\(site)\": allowed values are \(allowedSites.joined(separator: ", "))"
  }
}

// MARK: - GET /v1/civitai/search

/// Parsed query parameters for `GET /v1/civitai/search`.
struct CivitAISearchQuery: Equatable {
  var query: String
  var types: [String]
  var baseModel: String?
  var sort: CivitAIClient.SortOrder
  var period: CivitAIClient.Period
  var nsfw: Bool
  var cursor: String?
  var limit: Int
  /// CivitAI host to query — "civitai.com" (default) or "civitai.red"
  /// (same API, NSFW-default mirror — see CivitAIBrowserView.CivitAISource).
  var site: String

  init(queryParameters params: [String: String]) {
    // comfybox#380: `params` is `request.queryParameters`, already
    // percent-decoded centrally — do not decode again here (see the file
    // header comment).
    query = params["query"] ?? ""
    types = (params["types"] ?? "")
      .split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    baseModel = Self.nonEmpty(params["base_model"])
    sort = CivitAISortPeriodParsing.parseSort(params["sort"])
    period = CivitAISortPeriodParsing.parsePeriod(params["period"])
    nsfw = (params["nsfw"] ?? "false").lowercased() == "true"
    cursor = Self.nonEmpty(params["cursor"])
    limit = params["limit"].flatMap(Int.init) ?? 24
    site = Self.nonEmpty(params["site"]) ?? "civitai.com"
  }

  /// `nil` for any site outside `CivitAIHostAllowlist` — the route returns
  /// HTTP 400 rather than falling back, so a typo'd/hostile site is never
  /// silently redirected (P1-1).
  var validatedBaseURL: URL? { CivitAIHostAllowlist.baseURL(forSite: site) }

  private static func nonEmpty(_ s: String?) -> String? {
    guard let s, !s.isEmpty else { return nil }
    return s
  }
}

// MARK: - POST /v1/civitai/harvest

/// JSON body for `POST /v1/civitai/harvest`. Every field is optional/has a
/// default — an empty `{}` body is valid and harvests the default
/// most-downloaded browse. Decode with `keyDecodingStrategy =
/// .convertFromSnakeCase` (WarmServer's `decode(_:from:)` convention) so
/// `base_model` on the wire lands on `baseModel` here.
struct CivitAIHarvestRequestBody: Decodable, Equatable {
  var query: String = ""
  var types: [String] = []
  var baseModel: String?
  var sort: String?
  var period: String?
  var nsfw: Bool = false
  /// Total number of MODELS to scan across pages (not a per-page size).
  /// Server-side, `CivitAIHarvestRunner` clamps this to
  /// `CivitAIHarvestRunner.maxModelsPerHarvest` (200) per harvest call —
  /// asking for more silently harvests 200 (P1-2a).
  var limit: Int = 24
  var site: String = "civitai.com"

  private enum CodingKeys: String, CodingKey { case query, types, baseModel, sort, period, nsfw, limit, site }

  init() {}

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    query = (try? c.decodeIfPresent(String.self, forKey: .query)) ?? ""
    types = (try? c.decodeIfPresent([String].self, forKey: .types)) ?? []
    baseModel = (try? c.decodeIfPresent(String.self, forKey: .baseModel)).flatMap { $0 }
    sort = (try? c.decodeIfPresent(String.self, forKey: .sort)).flatMap { $0 }
    period = (try? c.decodeIfPresent(String.self, forKey: .period)).flatMap { $0 }
    nsfw = (try? c.decodeIfPresent(Bool.self, forKey: .nsfw)) ?? false
    limit = (try? c.decodeIfPresent(Int.self, forKey: .limit)) ?? 24
    site = (try? c.decodeIfPresent(String.self, forKey: .site)) ?? "civitai.com"
  }

  /// Same allowlist enforcement as `CivitAISearchQuery.validatedBaseURL` —
  /// both routes go through `CivitAIHostAllowlist`, the single shared
  /// enforcement point (P1-1).
  var validatedBaseURL: URL? { CivitAIHostAllowlist.baseURL(forSite: site) }
}

// MARK: - GET /v1/civitai/repo

/// Parsed query parameters for `GET /v1/civitai/repo`.
struct CivitAIRepoQuery: Equatable {
  /// Result cap when the caller doesn't pass `limit` (P2: the route used to
  /// return the whole store per request).
  static let defaultLimit = 100
  /// Hard ceiling — `limit` above this clamps down to it.
  static let maxLimit = 500

  var baseModel: String?
  var act: String?
  var tag: String?
  var keyword: String?
  /// Max entries to return: default 100, clamped to 1...500.
  var limit: Int

  init(queryParameters params: [String: String]) {
    // comfybox#380: `params` is `request.queryParameters`, already
    // percent-decoded centrally — do not decode again here.
    baseModel = Self.nonEmpty(params["base_model"])
    act = Self.nonEmpty(params["act"])
    tag = Self.nonEmpty(params["tag"])
    keyword = Self.nonEmpty(params["keyword"])
    limit = min(max(params["limit"].flatMap(Int.init) ?? Self.defaultLimit, 1), Self.maxLimit)
  }

  private static func nonEmpty(_ s: String?) -> String? {
    guard let s, !s.isEmpty else { return nil }
    return s
  }
}

// MARK: - Extraction (pure — unit-tested against a canned fixture, #234)

/// Builds `PromptRepositoryEntry` values out of a decoded `CivitAIModel` —
/// one entry per model version. No network access.
enum CivitAIHarvestExtractor {
  /// Harvest one entry per model version: trainedWords + a truncated,
  /// HTML-stripped description excerpt + inferred act taxonomy.
  ///
  /// `rawPrompt` is deliberately left `nil` here — this harvester does not
  /// call `CivitAIClient.images(modelVersionId:)`. Ground-truth finding
  /// (2026-07-15): `meta.prompt` on images is gated even with a valid API
  /// key, so building the harvest around it would make results unreliable
  /// AND cost one extra HTTP round trip per model version. `trainedWords` +
  /// description text are the reliable signal; `rawPrompt` stays as a field
  /// on `PromptRepositoryEntry` for if/when the gating opens up.
  static func entries(from model: CivitAIModel, harvestedAt: Date = Date()) -> [PromptRepositoryEntry] {
    let act = CivitAITaxonomy.inferAct(modelType: model.type, name: model.name, tags: model.tags)
    return model.modelVersions.map { version in
      PromptRepositoryEntry(
        sourceModelId: model.id,
        sourceVersionId: version.id,
        modelName: model.name,
        baseModel: version.baseModel,
        actTaxonomy: act,
        trainedWords: version.trainedWords,
        descriptionExcerpt: stripHTML(version.description),
        rawPrompt: nil,
        tags: model.tags,
        harvestedAt: harvestedAt
      )
    }
  }

  /// Minimal HTML de-tagging for CivitAI's rich-text version descriptions —
  /// good enough for a readable excerpt, not a general HTML parser.
  static func stripHTML(_ html: String?) -> String? {
    guard let html, !html.isEmpty else { return nil }
    var result = ""
    result.reserveCapacity(html.count)
    var inTag = false
    for ch in html {
      if ch == "<" { inTag = true }
      else if ch == ">" { inTag = false }
      else if !inTag { result.append(ch) }
    }
    let decoded = result
      .replacingOccurrences(of: "&nbsp;", with: " ")
      .replacingOccurrences(of: "&amp;", with: "&")
      .replacingOccurrences(of: "&#39;", with: "'")
      .replacingOccurrences(of: "&quot;", with: "\"")
    let trimmed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
  }
}

// MARK: - Harvest runner (paging loop is unit-tested via the fetchPage/upsert
// seams — CivitAIHarvestRunnerTests; only the thin `run(client:request:)`
// overload touches the network)

enum CivitAIHarvestRunner {
  struct Summary: Encodable {
    let modelsScanned: Int
    let versionsHarvested: Int
    let created: Int
    let updated: Int
    /// True when the harvest stopped early — time budget exhausted or the
    /// surrounding task was cancelled. The counts above reflect only what
    /// was actually fetched AND persisted (upserts happen per page), so a
    /// truncated summary is still accurate, just smaller than asked for
    /// (P1-2c).
    let truncated: Bool
  }

  /// Politeness delay between paged upstream requests during a harvest — the
  /// ground-facts note "don't hammer Civitai" made explicit as a fixed
  /// inter-page sleep, decision flagged for review (500ms; not specified by
  /// the issue).
  static let interPageDelayNanoseconds: UInt64 = 500_000_000

  /// Server-side cap on MODELS scanned per harvest call (P1-2a). Documented
  /// in the `civitai_prompts` MCP tool description and the harvest body's
  /// `limit` field; requests above it are clamped, not rejected.
  static let maxModelsPerHarvest = 200

  /// Soft wall-clock budget per harvest call (P1-2c). Checked between pages;
  /// on expiry the harvest returns what it has (already persisted) with
  /// `truncated: true` instead of paging on forever.
  static let timeBudgetSeconds: TimeInterval = 60

  /// `request.limit` → the number of models this run will actually scan.
  static func clampedLimit(_ requested: Int) -> Int {
    min(max(requested, 1), maxModelsPerHarvest)
  }

  /// Production entry point: search + paginate up to
  /// `clampedLimit(request.limit)` models via a real `CivitAIClient`,
  /// upserting into `PromptRepositoryStore` one page at a time.
  static func run(client: CivitAIClient, request: CivitAIHarvestRequestBody) async throws -> Summary {
    try await run(
      request: request,
      fetchPage: { cursor, pageLimit in
        try await client.searchModels(
          query: request.query,
          types: request.types,
          baseModel: request.baseModel,
          sort: CivitAISortPeriodParsing.parseSort(request.sort),
          period: CivitAISortPeriodParsing.parsePeriod(request.period),
          nsfw: request.nsfw,
          cursor: cursor,
          limit: pageLimit)
      })
  }

  /// The paging loop, with the network (`fetchPage`) and persistence
  /// (`upsert`) injected so it is unit-testable without either. Entries are
  /// upserted PER PAGE (P1-2b) — peak memory is one page's worth of entries,
  /// never the whole multi-page harvest, and a failure/timeout partway
  /// through keeps every page already persisted.
  static func run(
    request: CivitAIHarvestRequestBody,
    timeBudget: TimeInterval = timeBudgetSeconds,
    interPageDelay: UInt64 = interPageDelayNanoseconds,
    fetchPage: (_ cursor: String?, _ pageLimit: Int) async throws -> CivitAIModelsPage,
    upsert: ([PromptRepositoryEntry]) -> PromptRepositoryStore.UpsertResult = { PromptRepositoryStore.upsert($0) }
  ) async throws -> Summary {
    let start = Date()
    var modelsScanned = 0
    var versionsHarvested = 0
    var created = 0
    var updated = 0
    var truncated = false
    var cursor: String?
    var remaining = clampedLimit(request.limit)
    var firstPage = true

    while remaining > 0 {
      // Budget/cancellation check INSIDE the loop, before each page (P1-2c):
      // whatever pages already ran are persisted, so stopping here loses
      // nothing — the summary just reports truncated: true.
      if Task.isCancelled || Date().timeIntervalSince(start) >= timeBudget {
        truncated = true
        break
      }
      if !firstPage {
        try? await Task.sleep(nanoseconds: interPageDelay)
      }
      firstPage = false

      let pageLimit = min(remaining, 100)
      let page = try await fetchPage(cursor, pageLimit)

      guard !page.items.isEmpty else { break }
      var pageEntries: [PromptRepositoryEntry] = []
      for model in page.items {
        modelsScanned += 1
        pageEntries.append(contentsOf: CivitAIHarvestExtractor.entries(from: model))
      }
      let result = upsert(pageEntries)
      versionsHarvested += pageEntries.count
      created += result.created
      updated += result.updated
      remaining -= page.items.count

      guard let next = page.nextCursor, !next.isEmpty else { break }
      cursor = next
    }

    return Summary(
      modelsScanned: modelsScanned, versionsHarvested: versionsHarvested,
      created: created, updated: updated, truncated: truncated)
  }
}

// MARK: - Response DTOs

struct CivitAISearchResultVersion: Encodable {
  let id: Int
  let name: String
  let baseModel: String
  let trainedWords: [String]

  init(_ v: CivitAIModelVersion) {
    id = v.id
    name = v.name
    baseModel = v.baseModel
    trainedWords = v.trainedWords
  }
}

struct CivitAISearchResultModel: Encodable {
  let id: Int
  let name: String
  let type: String
  let nsfw: Bool
  let tags: [String]
  let downloadCount: Int
  let thumbsUpCount: Int
  let actTaxonomy: String?
  let versions: [CivitAISearchResultVersion]

  init(_ m: CivitAIModel) {
    id = m.id
    name = m.name
    type = m.type
    nsfw = m.nsfw
    tags = m.tags
    downloadCount = m.downloadCount
    thumbsUpCount = m.thumbsUpCount
    actTaxonomy = CivitAITaxonomy.inferAct(modelType: m.type, name: m.name, tags: m.tags)
    versions = m.modelVersions.map(CivitAISearchResultVersion.init)
  }
}

struct CivitAISearchResponse: Encodable {
  let models: [CivitAISearchResultModel]
  let count: Int
  let nextCursor: String?
}

struct CivitAIRepoResponse: Encodable {
  let entries: [PromptRepositoryEntry]
  let count: Int
}
