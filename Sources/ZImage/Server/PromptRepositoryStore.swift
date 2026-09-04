// PromptRepositoryStore.swift — durable JSON store of CivitAI-harvested prompt material (#234).
//
// Backs GET/POST /v1/civitai/{search,harvest,repo}: a local, queryable
// repository of prompt-adjacent signal pulled from CivitAI model versions —
// trained words, a truncated excerpt of the version description, and (best
// effort, often absent — see CivitAIConduitRoutes.swift) a raw sampled
// prompt — so Kira and other agents can draw on real, model-specific
// vocabulary via the MCP `civitai_prompts` tool instead of guessing trigger
// words.
//
// Mirrors QueuePersistence.swift's on-disk convention: JSON at
// `~/.comfybox/prompt-repository.json`, overridable via COMFYBOX_STATE_DIR
// (the same override QueueStateStore, the pause sentinel and everything else
// under ~/.comfybox honour), written with Foundation's `.atomic` option
// (temp file + rename under the hood, so a crash mid-write can't leave a
// truncated/corrupt file).

import Foundation
import Logging

/// One harvested prompt-repository entry, keyed by the CivitAI
/// (modelId, versionId) pair it was harvested from.
public struct PromptRepositoryEntry: Codable, Sendable, Equatable {
  /// Stable identity — literally "\(sourceModelId)-\(sourceVersionId)", so
  /// "upsert by composite key" (sourceModelId, sourceVersionId) IS identity
  /// equality on `id` (see `PromptRepositoryStore.upsert`).
  public let id: String
  public let sourceModelId: Int
  public let sourceVersionId: Int
  public var modelName: String
  public var baseModel: String
  /// Best-effort category inferred from the model's type/tags/name — see
  /// `CivitAITaxonomy`. Optional: absent when nothing matched.
  public var actTaxonomy: String?
  public var trainedWords: [String]
  /// CivitAI's HTML-stripped version description, capped to
  /// `descriptionExcerptLimit` characters.
  public var descriptionExcerpt: String?
  /// Sampled prompt text, when CivitAI's `meta.prompt` gate happens to be
  /// open for this version. Deliberately left nil by the current harvester
  /// (see CivitAIHarvestExtractor) — kept as a field so persisted entries
  /// don't need a schema migration if the gating changes later.
  public var rawPrompt: String?
  public var tags: [String]
  public var harvestedAt: Date

  /// Cap applied to `descriptionExcerpt` before it is stored.
  public static let descriptionExcerptLimit = 500

  public init(
    sourceModelId: Int, sourceVersionId: Int, modelName: String, baseModel: String,
    actTaxonomy: String? = nil, trainedWords: [String] = [], descriptionExcerpt: String? = nil,
    rawPrompt: String? = nil, tags: [String] = [], harvestedAt: Date = Date()
  ) {
    self.id = "\(sourceModelId)-\(sourceVersionId)"
    self.sourceModelId = sourceModelId
    self.sourceVersionId = sourceVersionId
    self.modelName = modelName
    self.baseModel = baseModel
    self.actTaxonomy = actTaxonomy
    self.trainedWords = trainedWords
    self.descriptionExcerpt = PromptRepositoryEntry.truncatedDescription(descriptionExcerpt)
    self.rawPrompt = rawPrompt
    self.tags = tags
    self.harvestedAt = harvestedAt
  }

  /// Truncate a description to `descriptionExcerptLimit` characters. `nil`
  /// and empty strings both stay `nil` (no empty-string excerpts stored).
  public static func truncatedDescription(_ text: String?) -> String? {
    guard let text, !text.isEmpty else { return nil }
    if text.count <= descriptionExcerptLimit { return text }
    return String(text.prefix(descriptionExcerptLimit))
  }
}

/// On-disk snapshot: a flat array of entries. Keeps the file diff-friendly
/// and mirrors `PersistedQueueState`'s array-of-jobs shape.
struct PromptRepositoryState: Codable, Sendable {
  var entries: [PromptRepositoryEntry] = []
}

/// Reads/writes `~/.comfybox/prompt-repository.json`.
///
/// Concurrency (adversarial review P2): every load→merge→save is serialized
/// under one `NSLock` — the same idiom PresetStore / NearlineLibrary /
/// VideoGeneratorHolder use for their shared mutable state. Two concurrent
/// harvests used to be able to interleave load→merge→save and silently drop
/// each other's whole batch; with the lock (and per-page upserts in
/// CivitAIHarvestRunner) every batch lands. All critical sections are
/// synchronous — no awaits are ever held across the lock.
public enum PromptRepositoryStore {
  /// Result of an upsert batch.
  public struct UpsertResult: Sendable, Equatable {
    public var created: Int
    public var updated: Int
  }

  /// Cap on stored entries (adversarial review P2: the file used to grow
  /// forever). When an upsert would overflow, the oldest-`harvestedAt`
  /// entries are evicted down to the cap — mirroring the job-pruner
  /// convention in WarmServer (`pruneCompleted`) of bounding every
  /// long-running server's stores. Tests inject a smaller cap via
  /// `upsert(_:cap:)`.
  public static let defaultMaxEntries = 5000

  private static let lock = NSLock()
  private static let logger = Logger(label: "comfybox.prompt-repository")

  /// `~/.comfybox`, or `COMFYBOX_STATE_DIR` when set — same resolution as
  /// `QueueStateStore.stateDirectory`. Kept as its own computed property
  /// (not shared) so this file's persistence contract is provable in
  /// isolation, exactly the way QueuePersistence.swift's is.
  public static var stateDirectory: URL {
    if let override = ProcessInfo.processInfo.environment["COMFYBOX_STATE_DIR"], !override.isEmpty {
      let dir = URL(fileURLWithPath: (override as NSString).expandingTildeInPath, isDirectory: true)
      try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
      return dir
    }
    let dir = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".comfybox", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  public static var path: URL { stateDirectory.appendingPathComponent("prompt-repository.json") }

  public static func loadAll() -> [PromptRepositoryEntry] {
    lock.lock()
    defer { lock.unlock() }
    return load().entries
  }

  /// Upsert a batch of freshly-harvested entries into the store, keyed by
  /// `id` (== the (sourceModelId, sourceVersionId) composite). Re-harvesting
  /// the same model version REPLACES its stored entry (fresher
  /// trainedWords/description/tags win) rather than duplicating it.
  ///
  /// The whole load→merge→save runs under `lock`, so concurrent upserts
  /// (e.g. two harvests' per-page batches) serialize instead of clobbering
  /// each other. If the merged store exceeds `cap` (default
  /// `defaultMaxEntries`), the oldest-`harvestedAt` entries are evicted down
  /// to the cap, with one log line per evicting upsert.
  @discardableResult
  public static func upsert(
    _ incoming: [PromptRepositoryEntry], cap: Int = defaultMaxEntries
  ) -> UpsertResult {
    guard !incoming.isEmpty else { return UpsertResult(created: 0, updated: 0) }
    lock.lock()
    defer { lock.unlock() }

    var state = load()
    // Last-writer-wins on BOTH sides (adversarial review P2):
    // `Dictionary(uniqueKeysWithValues:)` TRAPS on a duplicate key, so a
    // persisted file that ever picked up two entries with the same id would
    // crash the server on its next harvest. `load()` already self-heals the
    // file's contents; dedupe the incoming batch the same way so a page that
    // carries the same model version twice can't trap either.
    var byId = Dictionary(state.entries.map { ($0.id, $0) }, uniquingKeysWith: { _, b in b })
    let incomingDeduped = dedupedLastWins(incoming)
    var created = 0
    var updated = 0
    for entry in incomingDeduped {
      if byId.updateValue(entry, forKey: entry.id) != nil {
        updated += 1
      } else {
        created += 1
      }
    }
    // Stable order: existing entries first (updated in place, same order),
    // then newly-created ones in harvest order.
    let existingIds = state.entries.map(\.id)
    var seen = Set<String>()
    var merged: [PromptRepositoryEntry] = []
    for id in existingIds {
      if let e = byId[id], seen.insert(id).inserted { merged.append(e) }
    }
    for entry in incomingDeduped where !seen.contains(entry.id) {
      merged.append(entry)
      seen.insert(entry.id)
    }
    // Cap the store: evict oldest-harvestedAt first down to `cap` (P2).
    if merged.count > cap {
      let evictCount = merged.count - cap
      let evictIds = Set(
        merged.sorted { $0.harvestedAt < $1.harvestedAt }.prefix(evictCount).map(\.id))
      merged.removeAll { evictIds.contains($0.id) }
      logger.info(
        "Prompt repository over cap — evicted \(evictCount) oldest entries (cap \(cap))")
    }
    state.entries = merged
    save(state)
    return UpsertResult(created: created, updated: updated)
  }

  /// Filter the stored repository. Every filter is optional and
  /// case-insensitive; `keyword` matches against modelName, trainedWords,
  /// descriptionExcerpt and tags. `limit`, when set, caps the number of
  /// results returned (the /v1/civitai/repo route passes
  /// `CivitAIRepoQuery.limit`: default 100, max 500 — P2).
  public static func query(
    baseModel: String? = nil, act: String? = nil, tag: String? = nil, keyword: String? = nil,
    limit: Int? = nil
  ) -> [PromptRepositoryEntry] {
    var results = loadAll()
    if let baseModel, !baseModel.isEmpty {
      results = results.filter { $0.baseModel.caseInsensitiveCompare(baseModel) == .orderedSame }
    }
    if let act, !act.isEmpty {
      results = results.filter { ($0.actTaxonomy ?? "").caseInsensitiveCompare(act) == .orderedSame }
    }
    if let tag, !tag.isEmpty {
      results = results.filter { entry in entry.tags.contains { $0.caseInsensitiveCompare(tag) == .orderedSame } }
    }
    if let keyword, !keyword.isEmpty {
      let needle = keyword.lowercased()
      results = results.filter { entry in
        entry.modelName.lowercased().contains(needle)
          || entry.trainedWords.contains { $0.lowercased().contains(needle) }
          || (entry.descriptionExcerpt ?? "").lowercased().contains(needle)
          || entry.tags.contains { $0.lowercased().contains(needle) }
      }
    }
    if let limit, limit >= 0, results.count > limit {
      results = Array(results.prefix(limit))
    }
    return results
  }

  // MARK: - Deduplication (adversarial review P2)

  /// Order-preserving, last-writer-wins dedupe: each id keeps its FIRST
  /// position but its LAST value. Built on
  /// `Dictionary(_:uniquingKeysWith:)` — never the trapping
  /// `Dictionary(uniqueKeysWithValues:)` — so duplicate ids (a hand-edited
  /// or corrupt persisted file, or a page that repeats a model version)
  /// self-heal instead of crashing the server.
  static func dedupedLastWins(_ entries: [PromptRepositoryEntry]) -> [PromptRepositoryEntry] {
    let byId = Dictionary(entries.map { ($0.id, $0) }, uniquingKeysWith: { _, b in b })
    guard byId.count < entries.count else { return entries }
    var seen = Set<String>()
    var result: [PromptRepositoryEntry] = []
    result.reserveCapacity(byId.count)
    for entry in entries where seen.insert(entry.id).inserted {
      result.append(byId[entry.id]!)
    }
    return result
  }

  // MARK: - Private

  /// Callers must hold `lock` (or be inside a caller that does).
  private static func load() -> PromptRepositoryState {
    guard let data = try? Data(contentsOf: path) else { return PromptRepositoryState() }
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    var state = (try? decoder.decode(PromptRepositoryState.self, from: data)) ?? PromptRepositoryState()
    // Self-heal a file with duplicate ids on the way in (last entry wins) —
    // the next save persists the deduped form.
    state.entries = dedupedLastWins(state.entries)
    return state
  }

  private static func save(_ state: PromptRepositoryState) {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    guard let data = try? encoder.encode(state) else { return }
    // Foundation's `.atomic` write option writes to a temp file in the same
    // directory then renames over the destination — same approach as
    // QueueStateStore.save.
    try? data.write(to: path, options: .atomic)
  }
}
