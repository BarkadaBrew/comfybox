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

/// Reads/writes `~/.comfybox/prompt-repository.json`. Not actor-isolated:
/// callers are WarmServer route handlers, which run one request at a time
/// per connection; two harvests racing the same file is a known, accepted
/// narrow window (last-write-wins on save) — identical to QueueStateStore's
/// contract.
public enum PromptRepositoryStore {
  /// Result of an upsert batch.
  public struct UpsertResult: Sendable, Equatable {
    public var created: Int
    public var updated: Int
  }

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
    load().entries
  }

  /// Upsert a batch of freshly-harvested entries into the store, keyed by
  /// `id` (== the (sourceModelId, sourceVersionId) composite). Re-harvesting
  /// the same model version REPLACES its stored entry (fresher
  /// trainedWords/description/tags win) rather than duplicating it.
  @discardableResult
  public static func upsert(_ incoming: [PromptRepositoryEntry]) -> UpsertResult {
    guard !incoming.isEmpty else { return UpsertResult(created: 0, updated: 0) }
    var state = load()
    var byId = Dictionary(uniqueKeysWithValues: state.entries.map { ($0.id, $0) })
    var created = 0
    var updated = 0
    for entry in incoming {
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
      if let e = byId[id] { merged.append(e); seen.insert(id) }
    }
    for entry in incoming where !seen.contains(entry.id) {
      merged.append(entry)
      seen.insert(entry.id)
    }
    state.entries = merged
    save(state)
    return UpsertResult(created: created, updated: updated)
  }

  /// Filter the stored repository. Every filter is optional and
  /// case-insensitive; `keyword` matches against modelName, trainedWords,
  /// descriptionExcerpt and tags.
  public static func query(
    baseModel: String? = nil, act: String? = nil, tag: String? = nil, keyword: String? = nil
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
    return results
  }

  // MARK: - Private

  private static func load() -> PromptRepositoryState {
    guard let data = try? Data(contentsOf: path) else { return PromptRepositoryState() }
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return (try? decoder.decode(PromptRepositoryState.self, from: data)) ?? PromptRepositoryState()
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
