import Foundation

/// One required or optional filesystem asset in the LTX-2 local-video
/// preflight (the weights dir, the Gemma text-encoder dir, or the optional
/// learned upsampler file). Every field a human needs to explain a `false`
/// verdict is carried explicitly, rather than collapsing to a bare boolean —
/// "not ready" with no reason is not something Bree or Kira can act on.
public struct LocalVideoAssetStatus: Sendable, Equatable {
  public let name: String
  public let required: Bool
  public let configured: Bool
  public let path: String?
  public let exists: Bool
  public let readable: Bool
  public let valid: Bool
  public let error: String?

  public var json: [String: Any] {
    [
      "name": name, "required": required, "configured": configured,
      "path": (path as Any?) ?? NSNull(), "exists": exists, "readable": readable,
      "valid": valid, "error": (error as Any?) ?? NSNull(),
    ]
  }
}

/// A point-in-time verdict on whether LOCAL LTX-2 video generation can run —
/// disk existence + safetensors integrity only, no model load, no network
/// resolution. `/health` must stay cheap and truthful even when the machine
/// is offline, so this is deliberately the whole check.
///
/// `compute` is a pure function of its inputs (no shared state, no caching)
/// so it is directly unit-testable; `LocalVideoReadinessMonitor` is the
/// stateful wrapper that runs it off the request path.
public struct LocalVideoReadiness: Sendable, Equatable {
  public let ready: Bool
  public let reason: String?
  public let checkedAt: Date?
  public let requiredAssets: [LocalVideoAssetStatus]
  public let optionalAssets: [LocalVideoAssetStatus]

  /// The state before the monitor's first tick has completed — a brief
  /// startup window (the first computation itself runs on a background task,
  /// per #298 review finding 3, rather than blocking server startup).
  public static let unchecked = LocalVideoReadiness(
    ready: false, reason: "not_checked_yet", checkedAt: nil,
    requiredAssets: [], optionalAssets: [])

  public var json: [String: Any] {
    [
      "ready": ready,
      "reason": (reason as Any?) ?? NSNull(),
      "checked_at": checkedAt.map { ISO8601DateFormatter().string(from: $0) } as Any? ?? NSNull(),
      "required_assets": requiredAssets.map { $0.json },
      "optional_assets": optionalAssets.map { $0.json },
    ]
  }

  /// Disk-only LTX-2 preflight. `requiredFiles` on a directory asset accepts
  /// literal filenames plus the sentinel "safetensors", meaning: at least one
  /// top-level `.safetensors` shard must exist AND every shard found must
  /// pass `SafetensorsIntegrity.check` — a truncated/corrupt shard fails
  /// readiness with `truncated:<file>` even though the filename looks right
  /// (the exact trap intent.md warns about). The learned upsampler is
  /// optional because the built-in default is a single-stage core render.
  public static func compute(weightsPath: String?, gemmaPath: String?, upsamplerPath: String?) -> LocalVideoReadiness {
    let weights = directoryAsset(name: "ltx2_weights", path: weightsPath, requiredFiles: ["safetensors"])
    let gemma = directoryAsset(
      name: "gemma_text_encoder", path: gemmaPath,
      requiredFiles: ["config.json", "tokenizer.json", "safetensors"])
    let upsampler = optionalFileAsset(name: "ltx2_upsampler", path: upsamplerPath)
    let required = [weights, gemma]
    let ready = required.allSatisfy { $0.valid }
    return LocalVideoReadiness(
      ready: ready,
      reason: ready ? nil : required.first(where: { !$0.valid })?.error,
      checkedAt: Date(),
      requiredAssets: required,
      optionalAssets: [upsampler])
  }

  static func expandedPath(_ path: String?) -> String? {
    guard let path, !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
    return URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL.path
  }

  private static func directoryAsset(name: String, path: String?, requiredFiles: [String]) -> LocalVideoAssetStatus {
    guard let resolved = expandedPath(path) else {
      return LocalVideoAssetStatus(
        name: name, required: true, configured: false,
        path: nil, exists: false, readable: false, valid: false, error: "not_configured")
    }
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: resolved, isDirectory: &isDirectory)
    let readable = exists && FileManager.default.isReadableFile(atPath: resolved)
    var error: String?
    if !exists {
      error = "path_not_found"
    } else if !isDirectory.boolValue {
      error = "not_a_directory"
    } else if !readable {
      error = "path_not_readable"
    } else {
      let entries = (try? FileManager.default.contentsOfDirectory(atPath: resolved)) ?? []
      for requirement in requiredFiles {
        if requirement == "safetensors" {
          let shardNames = entries.filter { $0.hasSuffix(".safetensors") }.sorted()
          if shardNames.isEmpty {
            error = "missing_safetensors"
            break
          }
          if let corrupt = firstIntegrityFailure(directory: resolved, fileNames: shardNames) {
            error = corrupt
            break
          }
        } else if !entries.contains(requirement) {
          error = "missing_\(requirement)"
          break
        }
      }
    }
    return LocalVideoAssetStatus(
      name: name, required: true, configured: true,
      path: resolved, exists: exists, readable: readable, valid: error == nil, error: error)
  }

  private static func optionalFileAsset(name: String, path: String?) -> LocalVideoAssetStatus {
    guard let resolved = expandedPath(path) else {
      return LocalVideoAssetStatus(
        name: name, required: false, configured: false,
        path: nil, exists: false, readable: false, valid: true, error: nil)
    }
    let exists = FileManager.default.fileExists(atPath: resolved)
    let readable = exists && FileManager.default.isReadableFile(atPath: resolved)
    var error: String?
    if !exists {
      error = "path_not_found"
    } else if !readable {
      error = "path_not_readable"
    } else if let corrupt = firstIntegrityFailure(directory: nil, fileNames: [resolved]) {
      error = corrupt
    }
    return LocalVideoAssetStatus(
      name: name, required: false, configured: true,
      path: resolved, exists: exists, readable: readable, valid: error == nil, error: error)
  }

  /// Runs `SafetensorsIntegrity.check` over each file, in order, and returns
  /// the first failure's reason (already formatted as `truncated:<file>`).
  /// `directory` is prefixed onto each name when checking shards found by a
  /// directory listing; pass `nil` (with an already-absolute path) for a
  /// single standalone file like the optional upsampler.
  private static func firstIntegrityFailure(directory: String?, fileNames: [String]) -> String? {
    for fileName in fileNames {
      let url = directory.map { URL(fileURLWithPath: $0).appendingPathComponent(fileName) }
        ?? URL(fileURLWithPath: fileName)
      if case .invalid(let reason) = SafetensorsIntegrity.check(url: url) {
        return reason
      }
    }
    return nil
  }
}

/// Lock-protected, periodically refreshed cache of `LocalVideoReadiness`.
/// `/health` reads only `current()` — it must never touch the filesystem
/// itself (#298 review finding 3: synchronous disk I/O — `contentsOfDirectory`,
/// per-shard header parses — ran on every `/health` poll). All filesystem
/// access is confined to `refresh()`, which runs once at startup (on a
/// background task, not blocking server start) and every 60s thereafter.
///
/// `ltx2WeightsPath`/`ltx2GemmaPath` are set once on `WarmServerConfiguration`
/// (a `let` for the life of the process) — there is no live "config change"
/// to react to today. If a future release makes those paths mutable at
/// runtime, that setter should call `refresh()` again.
final class LocalVideoReadinessMonitor: @unchecked Sendable {
  private let lock = NSLock()
  private var snapshot = LocalVideoReadiness.unchecked
  private var loopTask: Task<Void, Never>?

  private let weightsPath: String?
  private let gemmaPath: String?
  private let upsamplerPath: String?

  init(weightsPath: String?, gemmaPath: String?, upsamplerPath: String?) {
    self.weightsPath = weightsPath
    self.gemmaPath = gemmaPath
    self.upsamplerPath = upsamplerPath
  }

  deinit { loopTask?.cancel() }

  func current() -> LocalVideoReadiness {
    lock.lock(); defer { lock.unlock() }
    return snapshot
  }

  /// Idempotent — a second call while the loop is already running is a no-op.
  func start() {
    guard loopTask == nil else { return }
    loopTask = Task { [weak self] in
      while !Task.isCancelled {
        guard let self else { return }
        self.refresh()
        try? await Task.sleep(nanoseconds: 60_000_000_000)
      }
    }
  }

  func stop() {
    loopTask?.cancel()
    loopTask = nil
  }

  private func refresh() {
    let computed = LocalVideoReadiness.compute(
      weightsPath: weightsPath, gemmaPath: gemmaPath, upsamplerPath: upsamplerPath)
    lock.lock(); snapshot = computed; lock.unlock()
  }
}
