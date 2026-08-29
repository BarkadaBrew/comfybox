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
// `request.queryParameters` (HTTPRequest.swift) does not percent-decode
// values, and HTTP request lines can't carry literal spaces — so query
// params never spell CivitAI's raw sort/period values ("Most Downloaded")
// exactly. `CivitAISortPeriodParsing` accepts snake_case/kebab-case/any-case
// forms instead ("most_downloaded", "Most Downloaded", "MOST-DOWNLOADED", …)
// and falls back to the default on anything unrecognized — used for both the
// GET query string and the POST JSON body, so the two routes behave
// identically.

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
    query = CivitAIQueryDecoding.decode(params["query"]) ?? ""
    types = (CivitAIQueryDecoding.decode(params["types"]) ?? "")
      .split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    baseModel = Self.nonEmpty(CivitAIQueryDecoding.decode(params["base_model"]))
    sort = CivitAISortPeriodParsing.parseSort(params["sort"])
    period = CivitAISortPeriodParsing.parsePeriod(params["period"])
    nsfw = (params["nsfw"] ?? "false").lowercased() == "true"
    cursor = Self.nonEmpty(CivitAIQueryDecoding.decode(params["cursor"]))
    limit = params["limit"].flatMap(Int.init) ?? 24
    site = Self.nonEmpty(CivitAIQueryDecoding.decode(params["site"])) ?? "civitai.com"
  }

  var baseURL: URL { URL(string: "https://\(site)") ?? URL(string: "https://civitai.com")! }

  private static func nonEmpty(_ s: String?) -> String? {
    guard let s, !s.isEmpty else { return nil }
    return s
  }
}

/// `HTTPRequest.queryParameters` (WarmServer.swift) splits the raw query
/// string on `&`/`=` WITHOUT percent-decoding — fine for the simple
/// identifiers/paths other routes pass, but civitai_search/civitai_prompts
/// (MCPToolExecutor.swift) build query strings from free text (e.g. a
/// multi-word search query) and DO percent-encode them, since an
/// unencoded space can't survive an HTTP request line at all. Decode here,
/// at the one place that needs it, rather than changing shared
/// query-parsing behavior every other route already depends on.
enum CivitAIQueryDecoding {
  static func decode(_ raw: String?) -> String? {
    guard let raw else { return nil }
    return raw.removingPercentEncoding ?? raw
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

  var resolvedBaseURL: URL { URL(string: "https://\(site)") ?? URL(string: "https://civitai.com")! }
}

// MARK: - GET /v1/civitai/repo

/// Parsed query parameters for `GET /v1/civitai/repo`.
struct CivitAIRepoQuery: Equatable {
  var baseModel: String?
  var act: String?
  var tag: String?
  var keyword: String?

  init(queryParameters params: [String: String]) {
    baseModel = Self.nonEmpty(CivitAIQueryDecoding.decode(params["base_model"]))
    act = Self.nonEmpty(CivitAIQueryDecoding.decode(params["act"]))
    tag = Self.nonEmpty(CivitAIQueryDecoding.decode(params["tag"]))
    keyword = Self.nonEmpty(CivitAIQueryDecoding.decode(params["keyword"]))
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

// MARK: - Harvest runner (network — NOT unit tested; see CivitAIHarvestExtractionTests)

enum CivitAIHarvestRunner {
  struct Summary: Encodable {
    let modelsScanned: Int
    let versionsHarvested: Int
    let created: Int
    let updated: Int
  }

  /// Politeness delay between paged upstream requests during a harvest — the
  /// ground-facts note "don't hammer Civitai" made explicit as a fixed
  /// inter-page sleep, decision flagged for review (500ms; not specified by
  /// the issue).
  static let interPageDelayNanoseconds: UInt64 = 500_000_000

  /// Search + paginate up to `request.limit` MODELS, extract prompt-repo
  /// entries from every version of every model scanned, and upsert them all
  /// into `PromptRepositoryStore`.
  static func run(client: CivitAIClient, request: CivitAIHarvestRequestBody) async throws -> Summary {
    var modelsScanned = 0
    var allEntries: [PromptRepositoryEntry] = []
    var cursor: String?
    var remaining = max(request.limit, 1)
    var firstPage = true

    while remaining > 0 {
      if !firstPage {
        try? await Task.sleep(nanoseconds: interPageDelayNanoseconds)
      }
      firstPage = false

      let pageLimit = min(remaining, 100)
      let page = try await client.searchModels(
        query: request.query,
        types: request.types,
        baseModel: request.baseModel,
        sort: CivitAISortPeriodParsing.parseSort(request.sort),
        period: CivitAISortPeriodParsing.parsePeriod(request.period),
        nsfw: request.nsfw,
        cursor: cursor,
        limit: pageLimit)

      guard !page.items.isEmpty else { break }
      for model in page.items {
        modelsScanned += 1
        allEntries.append(contentsOf: entries(from: model))
      }
      remaining -= page.items.count

      guard let next = page.nextCursor, !next.isEmpty else { break }
      cursor = next
    }

    let result = PromptRepositoryStore.upsert(allEntries)
    return Summary(
      modelsScanned: modelsScanned, versionsHarvested: allEntries.count,
      created: result.created, updated: result.updated)
  }

  private static func entries(from model: CivitAIModel) -> [PromptRepositoryEntry] {
    CivitAIHarvestExtractor.entries(from: model)
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
