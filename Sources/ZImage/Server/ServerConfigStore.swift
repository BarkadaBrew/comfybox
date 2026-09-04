// ServerConfigStore.swift — Lock-serialized, in-memory authoritative store over
// ~/.comfybox/config.json (FDD-ui-api-parity §3.3, D3; comfybox#300 Phase 3).
//
// GET/PUT/PATCH /v1/config all route through this store instead of touching disk
// directly on every request, so:
//   - reads are a lock-guarded in-memory copy, not a disk read (Phase-0-compatible:
//     no actor hop, no I/O on the request path);
//   - PUT (full replace) and PATCH (RFC 7386 JSON Merge Patch) serialize under one
//     NSLock — the PromptRepositoryStore idiom (chosen over an actor precisely
//     because FDD §3.1 makes actor hops the enemy on this server);
//   - a PATCH's merge happens INSIDE the lock against the CURRENT in-memory
//     document, so two concurrent patches to different pointers cannot conflict —
//     there is no retry loop because there is nothing to retry.
//
// `If-Match` is advisory (FDD §3.3, v2 REVISED): honoured when present and stale
// (409), never required — no current caller sends it. `PUT` without it still
// succeeds; the route handler adds a deprecation `Warning` header instead.

import Foundation
import CryptoKit

/// Everything a caller needs after a read or a write: the resolved document plus
/// the ETag it can send back as `If-Match` on a later write.
public struct ServerConfigSnapshot: Sendable {
  public let config: ComfyBoxServerConfig
  public let etag: String
}

/// Errors the store can raise. Route handlers map `.etagMismatch` to `409` and
/// everything else to `400`.
public enum ServerConfigStoreError: Error, CustomStringConvertible, Equatable {
  case etagMismatch(current: String)
  case invalidPatch(String)
  case validation(String)

  public var description: String {
    switch self {
    case .etagMismatch(let current):
      return "If-Match precondition failed (current ETag is \(current))"
    case .invalidPatch(let message):
      return message
    case .validation(let message):
      return message
    }
  }
}

/// Lock-serialized, in-memory authoritative store over `~/.comfybox/config.json`.
public final class ServerConfigStore: @unchecked Sendable {
  /// The server's one store, over the real config path. Route handlers use this;
  /// tests construct their own instances over temp paths instead.
  public static let shared = ServerConfigStore()

  private let lock = NSLock()
  private let path: URL
  private let fileManager: FileManager
  private let auditLog: AuditLog?
  private var document: ComfyBoxServerConfig
  private var etag: String

  /// - Parameters:
  ///   - path: where the document lives (defaults to `~/.comfybox/config.json`).
  ///   - coffeeShopProviders/coffeeShopConfig: the pre-existing coffee-shop migration
  ///     sources, forwarded to `ComfyBoxServerConfig.loadOrMigrate` unchanged.
  ///   - auditLog: where migrated values are logged (`config.migrate.*`). `nil`
  ///     silences logging (handy for tests that don't want a stray file).
  ///
  /// IMPORTANT: this does NOT run the renderDefaults/videoDefaults migration
  /// (see ``runFirstRunDefaultsMigrationIfNeeded()``) — only the pre-existing
  /// coffee-shop `loadOrMigrate`. Merely constructing a store (including via
  /// `.shared`, e.g. from a unit test that exercises `makePipelineRequest`
  /// without any test isolation) must never WRITE to `~/.comfybox/config.json`
  /// — the K-FIX-1 lesson (`ComfyBoxStateDirectoryIsolation.swift`) applied to
  /// this store. Migration is instead an explicit, one-time call made from
  /// `comfybox serve`'s startup path — the one place that is never a test.
  public init(
    path: URL = ComfyBoxServerConfig.defaultPath(),
    coffeeShopProviders: URL = ComfyBoxServerConfig.coffeeShopProvidersPath(),
    coffeeShopConfig: URL = ComfyBoxServerConfig.coffeeShopConfigPath(),
    fileManager: FileManager = .default,
    auditLog: AuditLog? = AuditLog()
  ) {
    self.path = path
    self.fileManager = fileManager
    self.auditLog = auditLog

    let loaded = ComfyBoxServerConfig.loadOrMigrate(
      at: path, coffeeShopProviders: coffeeShopProviders, coffeeShopConfig: coffeeShopConfig,
      fileManager: fileManager)
    self.document = loaded
    self.etag = Self.computeETag(loaded)
  }

  /// Run the renderDefaults/videoDefaults first-run migration (FDD §3.3) if
  /// either block is currently empty, persisting the result. A no-op (no
  /// disk write) once both blocks are populated — safe to call more than
  /// once, e.g. defensively at every server start.
  ///
  /// Call this EXACTLY from `comfybox serve`'s startup — never from a shared
  /// library init path, so a test that merely reads `renderDefaults(family:)`
  /// (which resolves correctly against an unmigrated/empty document — that IS
  /// the bit-identical invariant) can never trigger a write to a real
  /// machine's `~/.comfybox/config.json`.
  @discardableResult
  public func runFirstRunDefaultsMigrationIfNeeded() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    var mutable = document
    let migrated = Self.migrateRenderAndVideoDefaults(into: &mutable, auditLog: auditLog)
    guard migrated else { return false }
    try? mutable.save(to: path, fileManager: fileManager)
    document = mutable
    etag = Self.computeETag(mutable)
    return true
  }

  // MARK: - Read

  public func current() -> ServerConfigSnapshot {
    lock.lock()
    defer { lock.unlock() }
    return ServerConfigSnapshot(config: document, etag: etag)
  }

  // MARK: - Write: full replace (PUT)

  /// Full-document replace. `ifMatch`, when non-nil, must equal the CURRENT ETag or
  /// this throws `.etagMismatch` — the route handler turns that into `409`. `nil`
  /// proceeds unconditionally (the route handler adds the deprecation `Warning`).
  @discardableResult
  public func replace(with newDocument: ComfyBoxServerConfig, ifMatch: String?) throws -> ServerConfigSnapshot {
    lock.lock()
    defer { lock.unlock() }
    if let ifMatch, ifMatch != etag {
      throw ServerConfigStoreError.etagMismatch(current: etag)
    }
    try Self.validate(newDocument)
    try newDocument.save(to: path, fileManager: fileManager)
    document = newDocument
    etag = Self.computeETag(newDocument)
    return ServerConfigSnapshot(config: document, etag: etag)
  }

  // MARK: - Write: RFC 7386 JSON Merge Patch (PATCH)

  /// Apply an RFC 7386 JSON Merge Patch to the CURRENT document, inside the lock.
  /// Two concurrent patches to different top-level/nested keys cannot conflict:
  /// each call reads `document`, merges, validates, saves and republishes
  /// atomically under the same lock the other is waiting on — no retry loop.
  @discardableResult
  public func applyMergePatch(_ patch: [String: Any], ifMatch: String?) throws -> ServerConfigSnapshot {
    lock.lock()
    defer { lock.unlock() }
    if let ifMatch, ifMatch != etag {
      throw ServerConfigStoreError.etagMismatch(current: etag)
    }
    let currentObject = try Self.jsonObject(from: document)
    let mergedObject = JSONMergePatch.apply(patch: patch, to: currentObject)
    let mergedData: Data
    do {
      mergedData = try JSONSerialization.data(withJSONObject: mergedObject)
    } catch {
      throw ServerConfigStoreError.invalidPatch("Merge result is not valid JSON: \(error.localizedDescription)")
    }
    let merged: ComfyBoxServerConfig
    do {
      merged = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: mergedData)
    } catch {
      throw ServerConfigStoreError.invalidPatch("Merged document failed to decode: \(error.localizedDescription)")
    }
    try Self.validate(merged)
    try merged.save(to: path, fileManager: fileManager)
    document = merged
    etag = Self.computeETag(merged)
    return ServerConfigSnapshot(config: document, etag: etag)
  }

  // MARK: - Family-aware defaults (FDD §3.3 hot-apply)

  /// Config-layer render-parameter overrides for one engine family — read fresh
  /// (lock, no disk I/O) on every generate call. Sits ABOVE the engine's hardcoded
  /// fallback and BELOW request/preset in the resolution order; an unmigrated/empty
  /// config resolves to all-nil fields, so `payload.x ?? resolved.x ?? <engine
  /// constant>` is unchanged from today's behavior.
  public func renderDefaults(family: String) -> RenderDefaultValues {
    lock.lock()
    defer { lock.unlock() }
    return document.renderDefaults.resolved(family: family)
  }

  /// Config-layer video (Motion tab) defaults — same resolution posture as
  /// ``renderDefaults(family:)``. `family` defaults to `"ltx2"`, the only video
  /// engine today; `byFamily` exists for shape-parity with renderDefaults.
  public func videoDefaults(family: String = "ltx2") -> VideoDefaultValues {
    lock.lock()
    defer { lock.unlock() }
    return document.videoDefaults.resolved(family: family)
  }

  /// Resolution + memory caps for `ImageMemoryPreflight` (issue #22) — same
  /// lock-read, no-disk-I/O posture as `renderDefaults(family:)`.
  public func imageMemoryCaps() -> ImageMemoryCapsConfig {
    lock.lock()
    defer { lock.unlock() }
    return document.imageMemoryCaps
  }

  // MARK: - Validation

  /// Rejects a document with structurally-invalid render/video defaults — never
  /// NaN/infinite guidance, never non-positive width/height/steps/frames. Named in
  /// the thrown error so a `400` response can point at the exact field.
  static func validate(_ config: ComfyBoxServerConfig) throws {
    func checkRender(_ values: RenderDefaultValues, path: String) throws {
      if let width = values.width, width <= 0 {
        throw ServerConfigStoreError.validation("\(path).width must be > 0 (got \(width))")
      }
      if let height = values.height, height <= 0 {
        throw ServerConfigStoreError.validation("\(path).height must be > 0 (got \(height))")
      }
      if let steps = values.steps, steps <= 0 {
        throw ServerConfigStoreError.validation("\(path).steps must be > 0 (got \(steps))")
      }
      if let guidance = values.guidance, !guidance.isFinite {
        throw ServerConfigStoreError.validation("\(path).guidance must be a finite number (got \(guidance))")
      }
    }
    try checkRender(config.renderDefaults.default, path: "renderDefaults.default")
    for (family, values) in config.renderDefaults.byFamily {
      try checkRender(values, path: "renderDefaults.byFamily.\(family)")
    }

    func checkVideo(_ values: VideoDefaultValues, path: String) throws {
      if let width = values.width, width <= 0 {
        throw ServerConfigStoreError.validation("\(path).width must be > 0 (got \(width))")
      }
      if let height = values.height, height <= 0 {
        throw ServerConfigStoreError.validation("\(path).height must be > 0 (got \(height))")
      }
      if let frames = values.frames, frames <= 0 {
        throw ServerConfigStoreError.validation("\(path).frames must be > 0 (got \(frames))")
      }
    }
    try checkVideo(config.videoDefaults.default, path: "videoDefaults.default")
    for (family, values) in config.videoDefaults.byFamily {
      try checkVideo(values, path: "videoDefaults.byFamily.\(family)")
    }

    // #22: image memory/resolution caps — never non-positive, headroom fraction
    // must stay a finite value in [0, 1) (1.0 would demand ALL of `available`
    // stay free, refusing every render outright). I6 (PR #363 review): both
    // caps are also BOUNDED above — `ImageMemoryPreflight.estimateBytes` is
    // overflow-safe regardless, but an absurd cap (e.g. maxPixels = Int.max)
    // would make the resolution-cap gate itself meaningless, defeating its
    // whole "no probing needed" purpose. `maxImageMemoryCapsLongEdge`/
    // `maxImageMemoryCapsPixels` are themselves generous — well above any
    // resolution DyPE is remotely usable at today — this only rejects
    // configuration nonsense, not real requests.
    let caps = config.imageMemoryCaps
    if caps.maxLongEdge <= 0 {
      throw ServerConfigStoreError.validation("imageMemoryCaps.maxLongEdge must be > 0 (got \(caps.maxLongEdge))")
    }
    if caps.maxLongEdge > Self.maxImageMemoryCapsLongEdge {
      throw ServerConfigStoreError.validation(
        "imageMemoryCaps.maxLongEdge must be <= \(Self.maxImageMemoryCapsLongEdge) (got \(caps.maxLongEdge))")
    }
    if caps.maxPixels <= 0 {
      throw ServerConfigStoreError.validation("imageMemoryCaps.maxPixels must be > 0 (got \(caps.maxPixels))")
    }
    if caps.maxPixels > Self.maxImageMemoryCapsPixels {
      throw ServerConfigStoreError.validation(
        "imageMemoryCaps.maxPixels must be <= \(Self.maxImageMemoryCapsPixels) (got \(caps.maxPixels))")
    }
    if !caps.minAvailableHeadroomFraction.isFinite
      || caps.minAvailableHeadroomFraction < 0 || caps.minAvailableHeadroomFraction >= 1 {
      throw ServerConfigStoreError.validation(
        "imageMemoryCaps.minAvailableHeadroomFraction must be in [0, 1) (got \(caps.minAvailableHeadroomFraction))")
    }
  }

  /// I6 (PR #363 review): upper bound on `imageMemoryCaps.maxLongEdge` — well
  /// above any resolution DyPE is usable at today (issue #22's own table
  /// tops out at 2048px on a 128GB machine), but still small enough that
  /// `width*height` in `ImageMemoryPreflight.decideResolution` (plain `Int`
  /// arithmetic, checked before the overflow-safe `UInt64` estimate path)
  /// cannot itself overflow: `16384 * 16384` is `2^28`, comfortably inside
  /// `Int64`.
  static let maxImageMemoryCapsLongEdge = 16_384
  /// I6: upper bound on `imageMemoryCaps.maxPixels` — `2^28` (268,435,456),
  /// i.e. exactly `maxImageMemoryCapsLongEdge²`.
  static let maxImageMemoryCapsPixels = 1 << 28

  // MARK: - First-run migration (FDD §3.3 "The migration, inverted")

  /// The five dispatchable warm-server families (`WarmModelFamily.allCases`
  /// mirrored as raw strings so this file doesn't need to import that internal
  /// enum). `ControlSurfaceParityTests` cross-checks this list against the real
  /// enum so the two can't silently drift.
  static let engineFamilies = ["flux1", "flux2", "fibo", "chroma", "krea2"]

  /// Engine-constant seed values, ported 1:1 from the hardcoded fallback at each
  /// family's generate call site (`WarmServer.swift`, verified 2026-08-30):
  ///   - flux1 (base Z-Image, `GeneratePayload.makePipelineRequest`):
  ///     `ZImageModelMetadata.recommended{Width,Height,InferenceSteps,
  ///     GuidanceScale}` = 1024×1024, 9 steps, 0.0 guidance (Turbo).
  ///   - fibo (`runFiboGenerate`): 1024×1024, 30 steps, 4.0 guidance.
  ///   - chroma (`renderChroma`): 1024×1024, 28 steps, 0.0 guidance.
  ///   - flux2, krea2: width/height ONLY. Their steps/guidance are not fixed
  ///     engine constants — flux2's depend on which checkpoint is loaded
  ///     (`Flux2Pipeline.defaultSteps`/`isDistilled`: base 50/3.5, distilled
  ///     4/1.0) and krea2's on the physical variant (`Krea2Variant.defaultSteps`/
  ///     `defaultGuidance`: turbo 9/1.0, raw 30/1.0). Seeding a snapshot value
  ///     for either would FREEZE a number that must keep adapting to the active
  ///     checkpoint, so those two fields are deliberately left nil — the runtime
  ///     engine constant keeps applying exactly as it did before migration.
  static func engineSeed(family: String) -> RenderDefaultValues {
    switch family {
    case "flux1": return RenderDefaultValues(width: 1024, height: 1024, steps: 9, guidance: 0.0)
    case "flux2": return RenderDefaultValues(width: 1024, height: 1024)
    case "fibo": return RenderDefaultValues(width: 1024, height: 1024, steps: 30, guidance: 4.0)
    case "chroma": return RenderDefaultValues(width: 1024, height: 1024, steps: 28, guidance: 0.0)
    case "krea2": return RenderDefaultValues(width: 1024, height: 1024)
    default: return RenderDefaultValues()
    }
  }

  /// The one video engine family today. `byFamily` keying matches
  /// `renderDefaults`' shape so discovery (Phase 4) treats both uniformly.
  static let videoEngineFamilies = ["ltx2"]

  /// Video engine-constant seed, ported 1:1 from the LTX-2 prep path's
  /// hardcoded fallbacks (`WarmServer.swift` `prepareLocalVideo`:
  /// `?? 704`/`?? 448` dims, `?? 97` frames — same constants echoed by
  /// `POST /v1/video/config/effective`).
  static func videoEngineSeed(family: String) -> VideoDefaultValues {
    switch family {
    case "ltx2": return VideoDefaultValues(width: 704, height: 448, frames: 97)
    default: return VideoDefaultValues()
    }
  }

  /// First-run migration (FDD §3.3, engine-neutrality corrected per the
  /// 2026-08-30 adversarial review F1): if `renderDefaults`/`videoDefaults`
  /// is empty, seed it from the ENGINE's own current fallbacks — freezing
  /// TODAY's numbers into an editable document; resolution is unaffected,
  /// since every seed equals what the engine already does.
  ///
  /// Desktop values are deliberately NOT imported — for video either. The
  /// v2 FDD text carved out `videoWidth/Height/Frames` as "genuinely read"
  /// (by `MotionView.applyDefaults`), but importing them here would push one
  /// operator's Motion-tab UI numbers onto EVERY headless LTX caller that
  /// omits dims — the exact anti-pattern §3.3's inverted-migration rule
  /// exists to prevent, on THE production video family. Desktop values stay
  /// Desktop-local (MotionView keeps reading them for its own prefill);
  /// server-side `videoDefaults` starts at the engine constants and changes
  /// only via an explicit `PUT`/`PATCH /v1/config`.
  ///
  /// Each seeded value is logged to the audit log (`config.migrate.*`).
  /// Returns whether anything changed, so the caller knows whether to persist.
  @discardableResult
  static func migrateRenderAndVideoDefaults(
    into config: inout ComfyBoxServerConfig,
    auditLog: AuditLog?
  ) -> Bool {
    var changed = false

    if config.renderDefaults.isEmpty {
      var byFamily: [String: RenderDefaultValues] = [:]
      for family in engineFamilies {
        let seed = engineSeed(family: family)
        guard !seed.isEmpty else { continue }
        byFamily[family] = seed
        auditLog?.append(
          kind: "config.migrate.renderDefaults",
          message: "Seeded renderDefaults.byFamily.\(family) from the engine's current fallback",
          metadata: [
            "family": family,
            "width": seed.width.map(String.init) ?? "",
            "height": seed.height.map(String.init) ?? "",
            "steps": seed.steps.map(String.init) ?? "",
            "guidance": seed.guidance.map { String($0) } ?? "",
          ]
        )
      }
      config.renderDefaults = RenderDefaultsConfig(byFamily: byFamily)
      changed = true
    }

    if config.videoDefaults.isEmpty {
      var byFamily: [String: VideoDefaultValues] = [:]
      for family in videoEngineFamilies {
        let seed = videoEngineSeed(family: family)
        guard !seed.isEmpty else { continue }
        byFamily[family] = seed
        auditLog?.append(
          kind: "config.migrate.videoDefaults",
          message: "Seeded videoDefaults.byFamily.\(family) from the engine's current fallback",
          metadata: [
            "family": family,
            "width": seed.width.map(String.init) ?? "",
            "height": seed.height.map(String.init) ?? "",
            "frames": seed.frames.map(String.init) ?? "",
          ]
        )
      }
      config.videoDefaults = VideoDefaultsConfig(byFamily: byFamily)
      changed = true
    }

    return changed
  }

  // MARK: - ETag

  /// Strong ETag over the sorted-keys JSON encoding — any field change produces a
  /// different tag. Advisory only (§3.3): no caller is required to send it.
  static func computeETag(_ config: ComfyBoxServerConfig) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = (try? encoder.encode(config)) ?? Data()
    let digest = SHA256.hash(data: data)
    let hex = digest.map { String(format: "%02x", $0) }.joined()
    return "\"\(hex)\""
  }

  static func jsonObject(from config: ComfyBoxServerConfig) throws -> [String: Any] {
    let data = try JSONEncoder().encode(config)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw ServerConfigStoreError.invalidPatch("Current document did not encode as a JSON object")
    }
    return object
  }
}

// MARK: - RFC 7386 JSON Merge Patch

/// A minimal, dependency-free RFC 7386 (JSON Merge Patch) implementation over the
/// `Any` trees `JSONSerialization` produces/consumes. Pure and independently
/// testable (`JSONMergePatchTests`).
enum JSONMergePatch {
  /// Apply `patch` to `target` per RFC 7386 §2:
  ///   - if `patch` is not a JSON object, it REPLACES `target` wholesale;
  ///   - otherwise, each key in `patch` either deletes the matching key in
  ///     `target` (`null`) or recursively merges (anything else) — keys `target`
  ///     has that `patch` doesn't mention are left untouched.
  static func apply(patch: Any, to target: Any?) -> Any {
    guard let patchObject = patch as? [String: Any] else {
      return patch
    }
    var result = (target as? [String: Any]) ?? [:]
    for (key, value) in patchObject {
      if value is NSNull {
        result.removeValue(forKey: key)
      } else {
        result[key] = apply(patch: value, to: result[key])
      }
    }
    return result
  }
}
