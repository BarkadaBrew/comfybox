// NearlineLibrary.swift — Models/LoRAs on attached storage, staged on demand
//
// Attached volumes (default: the Seagate archive) hold the deep catalog;
// the internal disk holds a working set. Items are staged (copied) to local
// storage when needed and evicted least-recently-used when the staging
// budget is exceeded. Only files THIS library staged are ever evicted —
// user-placed files in the cache directories are untouchable.
//
// State (~/.comfybox/nearline.json) records the configured roots, the
// staging budget, and per-item staging bookkeeping. Guarded by a lock —
// same house style as PresetStore.

import Foundation
import Logging

public struct NearlineItem: Codable, Equatable, Sendable {
  public var name: String        // filename, the stable identity
  public var path: String        // source path on attached storage
  public var sizeMB: Double
  public var kind: String        // "model" | "lora" (size heuristic)
  public var stagedPath: String?
  public var stagedAt: Date?
  public var lastUsedAt: Date?

  /// #273: pin so the eviction planner (``NearlineLibrary/planEviction``)
  /// never selects this item, and ``NearlineLibrary/setAnchored`` stages it
  /// in synchronously when set. Additive; absent (or malformed) in
  /// nearline.json ⇒ false, never a load failure — same tolerant-decode
  /// posture as `LoRALibraryEntry.compatibilitySource` (#353).
  public var anchored: Bool

  /// #273 fix round 1 (I3): true while a `stage()` copy for this item is in
  /// flight, so a concurrent `item(named:)`/`list()` caller can see the copy
  /// is in progress rather than either the pre- or post-copy state. Runtime
  /// only — deliberately NOT persisted (see the custom `encode(to:)` below):
  /// trusting a stale `true` across a crash/restart would wedge the item as
  /// "staging" forever, since the in-memory coordination that would ever
  /// clear it is gone.
  public var staging: Bool = false

  public var staged: Bool { stagedPath != nil }

  public init(
    name: String, path: String, sizeMB: Double, kind: String,
    stagedPath: String? = nil, stagedAt: Date? = nil, lastUsedAt: Date? = nil,
    anchored: Bool = false
  ) {
    self.name = name
    self.path = path
    self.sizeMB = sizeMB
    self.kind = kind
    self.stagedPath = stagedPath
    self.stagedAt = stagedAt
    self.lastUsedAt = lastUsedAt
    self.anchored = anchored
    self.staging = false
  }

  private enum CodingKeys: String, CodingKey {
    case name, path, sizeMB, kind, stagedPath, stagedAt, lastUsedAt, anchored
    // `staging` intentionally excluded — see the doc comment on the property.
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    name = try c.decode(String.self, forKey: .name)
    path = try c.decode(String.self, forKey: .path)
    sizeMB = try c.decode(Double.self, forKey: .sizeMB)
    kind = try c.decode(String.self, forKey: .kind)
    stagedPath = try c.decodeIfPresent(String.self, forKey: .stagedPath)
    stagedAt = try c.decodeIfPresent(Date.self, forKey: .stagedAt)
    lastUsedAt = try c.decodeIfPresent(Date.self, forKey: .lastUsedAt)
    // Additive field (#273): absent (pre-existing nearline.json entries) or
    // unrecognized ⇒ false, never a load failure.
    anchored = (try? c.decodeIfPresent(Bool.self, forKey: .anchored)).flatMap { $0 } ?? false
    staging = false
  }

  /// Written out explicitly (rather than relying on synthesis) so `staging`
  /// — deliberately absent from `CodingKeys` — can never accidentally leak
  /// into nearline.json.
  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(name, forKey: .name)
    try c.encode(path, forKey: .path)
    try c.encode(sizeMB, forKey: .sizeMB)
    try c.encode(kind, forKey: .kind)
    try c.encodeIfPresent(stagedPath, forKey: .stagedPath)
    try c.encodeIfPresent(stagedAt, forKey: .stagedAt)
    try c.encodeIfPresent(lastUsedAt, forKey: .lastUsedAt)
    try c.encode(anchored, forKey: .anchored)
  }
}

public final class NearlineLibrary: @unchecked Sendable {
  public struct Configuration: Codable, Equatable, Sendable {
    public var roots: [String]
    public var cacheLimitGB: Double

    public static let `default` = Configuration(
      roots: ["/Volumes/Seagate 22T/MacMigrate/Models"],
      cacheLimitGB: 200
    )
  }

  private struct State: Codable {
    var config: Configuration = .default
    var items: [NearlineItem] = []
  }

  /// Files at or above this size are considered checkpoints, not LoRAs.
  static let modelSizeThresholdMB: Double = 1_500

  private let lock = NSLock()
  private var state = State()
  private let statePath: URL
  private let loraCacheDir: URL
  private let modelCacheDir: URL
  private let logger: Logger

  /// #273 fix round 1 (I3): one entry per name currently being copied by
  /// `stage()`. A second `stage()` call for the same name waits on the
  /// group (released once by the copier's single `leave()`) instead of
  /// racing a second multi-GB copy; `stage()` for a *different* name is
  /// never blocked by this, because the copy itself runs with `lock`
  /// released.
  private var stagingGroups: [String: DispatchGroup] = [:]
  private var stagingResults: [String: Result<String, Error>] = [:]

  /// #273 fix round 2 (M): number of `stage()` calls currently blocked on
  /// `stagingGroups[name]`'s group, waiting to read `stagingResults[name]`.
  /// Lets the copier (or the last waiter) know it's safe to prune the
  /// `Result` entry — otherwise it would sit in `stagingResults` for the
  /// life of the process once nobody remains to read it.
  private var stagingWaiterCounts: [String: Int] = [:]

  /// Test-only seam (#273 fix round 1, I3): when set, called with the item
  /// about to be copied, from inside the (lock-free) copy step, so a test
  /// can simulate a slow/blocking copier for ONE name — e.g. to prove
  /// `stage()` for a different name does not block — without needing a
  /// genuinely multi-GB file. Takes the item (not just the name) so a test
  /// can filter by name; a hook that delays unconditionally would also
  /// block every other concurrent `stage()` call sharing this instance.
  /// Never set outside tests.
  var testCopyDelayHook: ((NearlineItem) -> Void)?

  /// Test-only seam (#273 fix round 2, M): called synchronously, under
  /// `lock`, right after a `stage()` call registers itself as a waiter on
  /// someone else's in-flight copy for `name` — lets a test learn
  /// deterministically that the waiter has reached the "wait for the
  /// in-flight group" branch, instead of guessing with `Thread.sleep`.
  /// Never set outside tests.
  var testWaiterRegisteredHook: ((String) -> Void)?

  /// Test-only seam (#273 fix round 2, M): the number of `Result` entries
  /// currently held in `stagingResults` — a test can assert this drops back
  /// to 0 once every waiter (if any) has read its result, proving the dict
  /// doesn't leak an entry per name for the life of the process.
  var stagingResultsCountForTesting: Int {
    lock.lock(); defer { lock.unlock() }
    return stagingResults.count
  }

  /// `~/.comfybox/nearline.json`.
  public static func defaultStatePath() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
      .appendingPathComponent(".comfybox/nearline.json")
  }

  public init(
    statePath: URL = NearlineLibrary.defaultStatePath(),
    loraCacheDir: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".comfybox/loras", isDirectory: true),
    modelCacheDir: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".comfybox/nearline-models", isDirectory: true),
    logger: Logger = Logger(label: "comfybox.nearline")
  ) {
    self.statePath = statePath
    self.loraCacheDir = loraCacheDir
    self.modelCacheDir = modelCacheDir
    self.logger = logger
    if let data = try? Data(contentsOf: statePath) {
      let decoder = JSONDecoder()
      decoder.dateDecodingStrategy = .iso8601
      if let loaded = try? decoder.decode(State.self, from: data) {
        state = loaded
      }
    }
  }

  // MARK: - Introspection

  public var configuration: Configuration {
    lock.lock(); defer { lock.unlock() }
    return state.config
  }

  public func list() -> [NearlineItem] {
    lock.lock(); defer { lock.unlock() }
    return state.items
  }

  public func item(named name: String) -> NearlineItem? {
    lock.lock(); defer { lock.unlock() }
    return state.items.first { $0.name == name }
  }

  public func updateConfiguration(_ config: Configuration) {
    lock.lock(); defer { lock.unlock() }
    state.config = config
    persistLocked()
  }

  // MARK: - Scan

  /// Walk the configured roots for *.safetensors, preserving staging
  /// bookkeeping for items that persist across scans. Returns item count.
  @discardableResult
  public func scan() -> Int {
    let fm = FileManager.default
    lock.lock(); defer { lock.unlock() }

    var found: [NearlineItem] = []
    var seen = Set<String>()
    for root in state.config.roots {
      let rootURL = URL(fileURLWithPath: root, isDirectory: true)
      guard fm.fileExists(atPath: root),
            let enumerator = fm.enumerator(
              at: rootURL, includingPropertiesForKeys: [.fileSizeKey],
              options: [.skipsHiddenFiles])
      else { continue }

      for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        let name = url.lastPathComponent
        guard !seen.contains(name) else { continue }  // first root wins on collision
        seen.insert(name)
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let sizeMB = Double(size) / 1_048_576
        var item = NearlineItem(
          name: name,
          path: url.path,
          sizeMB: sizeMB,
          kind: sizeMB >= Self.modelSizeThresholdMB ? "model" : "lora"
        )
        // Keep staging + anchor bookkeeping across rescans (verify the copy
        // exists before trusting a stale staged path).
        if let existing = state.items.first(where: { $0.name == name }) {
          if let stagedPath = existing.stagedPath, fm.fileExists(atPath: stagedPath) {
            item.stagedPath = stagedPath
            item.stagedAt = existing.stagedAt
            item.lastUsedAt = existing.lastUsedAt
          }
          item.anchored = existing.anchored
        }
        found.append(item)
      }
    }
    state.items = found.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    persistLocked()
    return state.items.count
  }

  // MARK: - Stage / evict

  /// Copy an item to local storage (LoRAs into the LoRA library so existing
  /// flows just see them; checkpoints into the nearline model cache). Evicts
  /// least-recently-used staged items first if the budget would overflow,
  /// or throws `NearlineError.insufficientCapacity` if that still isn't
  /// enough (anchored items are never evicted — see `planEviction`).
  /// Returns the local path.
  ///
  /// #273 fix round 1 (I3): the actual multi-GB copy runs with `lock`
  /// released — only the bookkeeping (index lookup, capacity check,
  /// publishing the result) is done under lock — so a concurrent
  /// `item(named:)`/`list()`/`stage()` call for a *different* name is never
  /// blocked by an in-flight copy. A second `stage()` call for the *same*
  /// name waits for the in-flight copy and returns its result rather than
  /// racing a second copy of the same file.
  @discardableResult
  public func stage(name: String) throws -> String {
    lock.lock()

    guard let index = state.items.firstIndex(where: { $0.name == name }) else {
      lock.unlock()
      throw NearlineError.unknownItem(name)
    }

    // Already staged and present — just touch it. No copy needed.
    if let stagedPath = state.items[index].stagedPath, FileManager.default.fileExists(atPath: stagedPath) {
      state.items[index].lastUsedAt = Date()
      persistLocked()
      lock.unlock()
      return stagedPath
    }

    // Another caller is already staging this exact name — wait for it
    // instead of racing a second copy.
    if let group = stagingGroups[name] {
      stagingWaiterCounts[name, default: 0] += 1
      testWaiterRegisteredHook?(name)
      lock.unlock()
      group.wait()
      lock.lock()
      let result = stagingResults[name]
      // #273 fix round 2 (M): the last waiter prunes its own entry so
      // `stagingResults` doesn't hold one Result per name forever; also
      // touch lastUsedAt on success — piggybacking on someone else's
      // in-flight copy is still using the item, for LRU purposes.
      let remaining = (stagingWaiterCounts[name] ?? 1) - 1
      if remaining <= 0 {
        stagingWaiterCounts.removeValue(forKey: name)
        stagingResults.removeValue(forKey: name)
      } else {
        stagingWaiterCounts[name] = remaining
      }
      if case .success = result, let idx = state.items.firstIndex(where: { $0.name == name }) {
        state.items[idx].lastUsedAt = Date()
        persistLocked()
      }
      lock.unlock()
      switch result {
      case .success(let path): return path
      case .failure(let error): throw error
      case nil: throw NearlineError.unknownItem(name)
      }
    }

    let item = state.items[index]
    guard FileManager.default.fileExists(atPath: item.path) else {
      lock.unlock()
      throw NearlineError.sourceMissing(item.path)
    }

    // Reserve capacity (evict LRU non-anchored items, or throw) — cheap
    // bookkeeping only, still under lock; no file I/O for the incoming item
    // itself happens here.
    do {
      try ensureCapacityLocked(forIncomingMB: item.sizeMB)
    } catch {
      lock.unlock()
      throw error
    }

    let group = DispatchGroup()
    group.enter()
    stagingGroups[name] = group
    stagingResults.removeValue(forKey: name)
    state.items[index].staging = true
    lock.unlock()

    let result = copyToCache(item: item)

    lock.lock()
    stagingResults[name] = result
    stagingGroups.removeValue(forKey: name)
    // #273 fix round 2 (M): if nobody is waiting to read it, prune the
    // Result immediately — the common case (no concurrent duplicate
    // stage() calls) would otherwise leak one entry per staged name for
    // the life of the process. A waiter that IS currently registered
    // prunes it itself after reading it (see the waiting branch above).
    if (stagingWaiterCounts[name] ?? 0) == 0 {
      stagingResults.removeValue(forKey: name)
    }
    if let idx2 = state.items.firstIndex(where: { $0.name == name }) {
      state.items[idx2].staging = false
      if case .success(let path) = result {
        state.items[idx2].stagedPath = path
        state.items[idx2].stagedAt = Date()
        state.items[idx2].lastUsedAt = Date()
      }
      persistLocked()
    }
    lock.unlock()
    group.leave()

    switch result {
    case .success(let path):
      logger.info("Nearline: staged \(name) (\(Int(item.sizeMB)) MB) → \(path)")
      return path
    case .failure(let error):
      throw error
    }
  }

  /// The actual file copy — deliberately free of `self.lock` so it can run
  /// concurrently with unrelated catalog reads/writes (#273 fix round 1,
  /// I3). Pure w.r.t. `state`: takes the item by value, returns a `Result`.
  private func copyToCache(item: NearlineItem) -> Result<String, Error> {
    testCopyDelayHook?(item)
    do {
      let directory = item.kind == "model" ? modelCacheDir : loraCacheDir
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      let destination = directory.appendingPathComponent(item.name)
      if FileManager.default.fileExists(atPath: destination.path) {
        try FileManager.default.removeItem(at: destination)
      }
      // Copy via a temp name so a crash mid-copy never leaves a plausible file.
      let temp = directory.appendingPathComponent(".\(item.name).staging")
      try? FileManager.default.removeItem(at: temp)
      try FileManager.default.copyItem(atPath: item.path, toPath: temp.path)
      try FileManager.default.moveItem(at: temp, to: destination)
      return .success(destination.path)
    } catch {
      return .failure(error)
    }
  }

  /// #273: pin (or unpin) an item so ``planEviction`` never selects it.
  /// Pinning ("anchor") synchronously stages the item in first if it is not
  /// already staged locally — anchoring means "always resident on internal
  /// storage," not merely "don't evict once staged." Un-anchoring only
  /// clears the flag; it does not itself evict (a later staging pass may).
  ///
  /// Throws `NearlineError.unknownItem` if `name` is not in the catalog, or
  /// whatever `stage(name:)` throws if the synchronous pull-in fails — a
  /// missing source volume (`.sourceMissing`) or a staging budget that
  /// can't fit it even after evicting everything evictable
  /// (`.insufficientCapacity`).
  ///
  /// #273 fix round 1 (ruling M): when anchoring requires a stage (the item
  /// isn't already local), `anchored` is persisted only AFTER `stage(name:)`
  /// succeeds — nearline.json must never claim an item is anchored while
  /// staging it actually failed. Un-anchoring, and anchoring an
  /// already-staged item, need no copy and persist immediately.
  @discardableResult
  public func setAnchored(name: String, anchored: Bool) throws -> NearlineItem {
    lock.lock()
    guard let index = state.items.firstIndex(where: { $0.name == name }) else {
      lock.unlock()
      throw NearlineError.unknownItem(name)
    }
    let needsStage = anchored && state.items[index].stagedPath == nil
    guard needsStage else {
      state.items[index].anchored = anchored
      persistLocked()
      let result = state.items[index]
      lock.unlock()
      return result
    }
    lock.unlock()

    // stage(name:) takes the lock itself — must not be called while held.
    // If this throws (insufficient capacity, missing source), `anchored`
    // is never set, matching ruling M.
    _ = try stage(name: name)

    lock.lock()
    guard let index2 = state.items.firstIndex(where: { $0.name == name }) else {
      lock.unlock()
      throw NearlineError.unknownItem(name)
    }
    state.items[index2].anchored = true
    persistLocked()
    let result = state.items[index2]
    lock.unlock()
    return result
  }

  /// Remove a staged copy (the attached-storage original is untouched).
  ///
  /// #273 fix round 2 (N2): refuses to evict an anchored item —
  /// anchoring means "always resident on internal storage," so silently
  /// removing the staged copy out from under it would leave `anchored: true`
  /// pointing at a file that no longer exists. Un-anchor first
  /// (``setAnchored``), then evict.
  @discardableResult
  public func evict(name: String) throws -> Bool {
    lock.lock(); defer { lock.unlock() }
    if let item = state.items.first(where: { $0.name == name }), item.anchored {
      throw NearlineError.anchored(name)
    }
    return evictLocked(name: name)
  }

  private func evictLocked(name: String) -> Bool {
    guard let index = state.items.firstIndex(where: { $0.name == name }),
          let stagedPath = state.items[index].stagedPath
    else { return false }
    try? FileManager.default.removeItem(atPath: stagedPath)
    state.items[index].stagedPath = nil
    state.items[index].stagedAt = nil
    persistLocked()
    logger.info("Nearline: evicted \(name)")
    return true
  }

  /// Total MB of currently staged copies.
  public var stagedMB: Double {
    lock.lock(); defer { lock.unlock() }
    return stagedMBLocked
  }

  private var stagedMBLocked: Double {
    state.items.filter(\.staged).reduce(0) { $0 + $1.sizeMB }
  }

  /// #273: pure eviction planner — given the currently staged items, the
  /// staging budget (MB), and an incoming item's size, returns (in eviction
  /// order) the staged items that must be freed to make room. Anchored
  /// items are never selected: anchoring pins an asset to internal storage
  /// regardless of LRU pressure. A free function (no lock, no I/O) so it is
  /// directly unit-testable.
  static func planEviction(
    stagedItems: [NearlineItem], limitMB: Double, incomingMB: Double
  ) -> [NearlineItem] {
    var used = stagedItems.reduce(0.0) { $0 + $1.sizeMB }
    guard used + incomingMB > limitMB else { return [] }

    let lruOrder = stagedItems
      .filter { !$0.anchored }
      .sorted { ($0.lastUsedAt ?? .distantPast) < ($1.lastUsedAt ?? .distantPast) }

    var toEvict: [NearlineItem] = []
    for candidate in lruOrder {
      guard used + incomingMB > limitMB else { break }
      used -= candidate.sizeMB
      toEvict.append(candidate)
    }
    return toEvict
  }

  /// Evict LRU (non-anchored) staged items until the incoming file fits the
  /// budget. #273 fix round 1 (C2): if eviction still can't free enough
  /// room — every remaining staged byte is anchored, or the incoming file
  /// is simply bigger than the whole budget — throws
  /// `NearlineError.insufficientCapacity` instead of letting `stage()`
  /// silently copy past the budget onto a volume that may be nearly full.
  ///
  /// #273 fix round 2 (N3): bytes reserved by OTHER in-flight `stage()`
  /// copies (marked `staging` under this same lock, before it's released
  /// for the copy) count as already-used too. Without this, two concurrent
  /// `stage()` calls for different names can each pass this guard against
  /// the same free space — neither sees the other's incoming file — and
  /// both copies land even though combined they overflow the budget. Since
  /// this method only ever runs under `lock`, and a `staging` flag is set
  /// (still under `lock`) before that lock is ever released for the actual
  /// copy, a second caller's check here is guaranteed to observe the
  /// first's reservation.
  private func ensureCapacityLocked(forIncomingMB incoming: Double) throws {
    let limitMB = state.config.cacheLimitGB * 1024
    let staged = state.items.filter(\.staged)
    for candidate in Self.planEviction(stagedItems: staged, limitMB: limitMB, incomingMB: incoming) {
      _ = evictLocked(name: candidate.name)
    }

    let usedAfter = state.items.filter(\.staged).reduce(0.0) { $0 + $1.sizeMB }
    let reservedByOthersMB = state.items.filter(\.staging).reduce(0.0) { $0 + $1.sizeMB }
    let freeMB = limitMB - usedAfter - reservedByOthersMB
    guard incoming > freeMB else { return }

    let anchoredMB = state.items.filter { $0.staged && $0.anchored }.reduce(0.0) { $0 + $1.sizeMB }
    throw NearlineError.insufficientCapacity(needMB: incoming, freeMB: max(freeMB, 0), anchoredMB: anchoredMB)
  }

  private func persistLocked() {
    let dir = statePath.deletingLastPathComponent()
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    if let data = try? encoder.encode(state) {
      try? data.write(to: statePath, options: .atomic)
    }
  }
}

public enum NearlineError: Error, LocalizedError, Equatable {
  case unknownItem(String)
  case sourceMissing(String)
  /// #273 fix round 1 (C2): `stage()` could not free enough room even after
  /// evicting every evictable (non-anchored) staged item.
  case insufficientCapacity(needMB: Double, freeMB: Double, anchoredMB: Double)
  /// #273 fix round 2 (N2): `evict(name:)` was called on an item that is
  /// currently anchored — un-anchor it first.
  case anchored(String)

  public var errorDescription: String? {
    switch self {
    case .unknownItem(let name):
      return "Nearline item not in catalog: \(name) (rescan?)"
    case .sourceMissing(let path):
      return "Nearline source missing (volume unmounted?): \(path)"
    case .insufficientCapacity(let need, let free, let anchored):
      return "Nearline staging budget exceeded: need \(Int(need)) MB, only \(Int(free)) MB free"
        + " (\(Int(anchored)) MB pinned by anchored items)"
    case .anchored(let name):
      return "Cannot evict anchored item: \(name) — un-anchor it first"
    }
  }
}
