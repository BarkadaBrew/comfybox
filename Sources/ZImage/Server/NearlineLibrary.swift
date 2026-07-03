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

  public var staged: Bool { stagedPath != nil }
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
        // Keep staging bookkeeping across rescans (verify the copy exists).
        if let existing = state.items.first(where: { $0.name == name }),
           let stagedPath = existing.stagedPath,
           fm.fileExists(atPath: stagedPath) {
          item.stagedPath = stagedPath
          item.stagedAt = existing.stagedAt
          item.lastUsedAt = existing.lastUsedAt
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
  /// least-recently-used staged items first if the budget would overflow.
  /// Returns the local path.
  @discardableResult
  public func stage(name: String) throws -> String {
    lock.lock(); defer { lock.unlock() }
    guard let index = state.items.firstIndex(where: { $0.name == name }) else {
      throw NearlineError.unknownItem(name)
    }
    var item = state.items[index]

    // Already staged and present — just touch it.
    if let stagedPath = item.stagedPath, FileManager.default.fileExists(atPath: stagedPath) {
      item.lastUsedAt = Date()
      state.items[index] = item
      persistLocked()
      return stagedPath
    }

    guard FileManager.default.fileExists(atPath: item.path) else {
      throw NearlineError.sourceMissing(item.path)
    }

    ensureCapacityLocked(forIncomingMB: item.sizeMB)

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

    item.stagedPath = destination.path
    item.stagedAt = Date()
    item.lastUsedAt = Date()
    state.items[index] = item
    persistLocked()
    logger.info("Nearline: staged \(name) (\(Int(item.sizeMB)) MB) → \(destination.path)")
    return destination.path
  }

  /// Remove a staged copy (the attached-storage original is untouched).
  @discardableResult
  public func evict(name: String) -> Bool {
    lock.lock(); defer { lock.unlock() }
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

  /// Evict LRU staged items until the incoming file fits the budget.
  private func ensureCapacityLocked(forIncomingMB incoming: Double) {
    let limitMB = state.config.cacheLimitGB * 1024
    var used = stagedMBLocked
    guard used + incoming > limitMB else { return }

    let lruOrder = state.items
      .filter(\.staged)
      .sorted { ($0.lastUsedAt ?? .distantPast) < ($1.lastUsedAt ?? .distantPast) }
    for candidate in lruOrder {
      guard used + incoming > limitMB else { break }
      used -= candidate.sizeMB
      _ = evictLocked(name: candidate.name)
    }
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

public enum NearlineError: Error, LocalizedError {
  case unknownItem(String)
  case sourceMissing(String)

  public var errorDescription: String? {
    switch self {
    case .unknownItem(let name):
      return "Nearline item not in catalog: \(name) (rescan?)"
    case .sourceMissing(let path):
      return "Nearline source missing (volume unmounted?): \(path)"
    }
  }
}
