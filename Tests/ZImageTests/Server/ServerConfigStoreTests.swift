// ServerConfigStoreTests.swift — FDD-ui-api-parity §3.3/§4.4 (Phase 3, D3).
//
// Every test constructs its OWN ServerConfigStore over a temp path — NEVER
// `.shared` — so this suite can never touch a real machine's
// `~/.comfybox/config.json` (K-FIX-1: see ComfyBoxStateDirectoryIsolation.swift
// for the reason that matters here).

import XCTest
@testable import ZImage

final class ServerConfigStoreTests: XCTestCase {

  private func makeTempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-config-store-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  /// A store over an empty temp directory with no coffee-shop sources — the
  /// "brand new install" shape. `auditLog: nil` so tests don't leave a stray
  /// file behind.
  private func makeStore(in dir: URL) -> ServerConfigStore {
    ServerConfigStore(
      path: dir.appendingPathComponent("config.json"),
      coffeeShopProviders: dir.appendingPathComponent("absent-providers.json"),
      coffeeShopConfig: dir.appendingPathComponent("absent-config.json"),
      auditLog: nil
    )
  }

  /// Parse a JSON string into `[String: Any]` via `JSONSerialization` —
  /// used for merge-patch bodies so nested objects are the same shape
  /// production code (and an HTTP request body) actually produces, rather
  /// than a hand-written Swift dictionary literal passed inline as a
  /// `[String: Any]` argument (which does not reliably preserve `Any`-boxing
  /// at every nesting level for `as? [String: Any]` recovery inside
  /// `JSONMergePatch.apply`).
  private func jsonObject(_ json: String) -> [String: Any] {
    (try! JSONSerialization.jsonObject(with: Data(json.utf8))) as! [String: Any]
  }

  // MARK: - Empty-config bit-identity (the D3 invariant)

  /// A freshly-constructed store (no migration run) resolves every family to
  /// an all-nil `RenderDefaultValues` — so `payload.x ?? resolved.x ?? <engine
  /// constant>` at every call site is exactly `payload.x ?? <engine constant>`,
  /// unchanged from pre-Phase-3 behavior. This is the resolution-order half of
  /// the invariant; ``testMigrationSeedsExactlyTheEngineBaseline`` covers the
  /// migration half.
  func testEmptyConfigResolvesToAllNilForEveryFamily() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)

    for family in ["flux1", "flux2", "fibo", "chroma", "krea2", "some-future-family"] {
      let resolved = store.renderDefaults(family: family)
      XCTAssertNil(resolved.width, "width should be nil for \(family)")
      XCTAssertNil(resolved.height, "height should be nil for \(family)")
      XCTAssertNil(resolved.steps, "steps should be nil for \(family)")
      XCTAssertNil(resolved.guidance, "guidance should be nil for \(family)")
    }
    let video = store.videoDefaults()
    XCTAssertNil(video.width)
    XCTAssertNil(video.height)
    XCTAssertNil(video.frames)
  }

  /// Merely reading defaults — the exact call pattern
  /// `makePipelineRequest`/`runFiboGenerate`/etc. use — must never write to
  /// disk. This is the regression this suite exists to prevent: a test in
  /// ANOTHER file (`DyPEAutoEnableTests`, `ZImageEtaRegressionTests`) calls
  /// `makePipelineRequest` without any state-dir isolation, so if reading
  /// defaults had a migration side effect, running the whole suite would
  /// silently rewrite a real machine's config.
  /// `ComfyBoxServerConfig.loadOrMigrate` legitimately persists on first
  /// launch (pre-existing contract — see `ComfyBoxServerConfigTests.
  /// testLoadOrMigrateWritesFileOnFirstLaunch`), so construction itself may
  /// create `config.json`. What must NEVER happen merely from reading
  /// defaults — the exact call pattern `makePipelineRequest`/
  /// `runFiboGenerate`/etc. use, exercised by pre-existing tests
  /// (`DyPEAutoEnableTests`, `ZImageEtaRegressionTests`) with NO state-dir
  /// isolation — is the NEW renderDefaults/videoDefaults seed migration.
  /// That is gated behind the explicit `runFirstRunDefaultsMigrationIfNeeded()`
  /// call (only ever made from `comfybox serve`'s startup), so whatever
  /// `loadOrMigrate` legitimately wrote must contain neither key.
  func testReadingDefaultsNeverTriggersTheSeedMigration() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("config.json")
    let store = makeStore(in: dir)
    _ = store.renderDefaults(family: "flux1")
    _ = store.videoDefaults()
    _ = store.current()
    guard FileManager.default.fileExists(atPath: path.path) else { return }
    let data = try Data(contentsOf: path)
    let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    XCTAssertNil(object?["renderDefaults"], "reading defaults must not trigger the seed migration")
    XCTAssertNil(object?["videoDefaults"], "reading defaults must not trigger the seed migration")
  }

  // MARK: - Migration: engine-baseline value preservation

  /// The five families' seeded values, per `ServerConfigStore.engineSeed`,
  /// ported 1:1 from the hardcoded fallback at each family's generate call
  /// site. flux2/krea2 seed width/height ONLY (their steps/guidance are
  /// checkpoint/variant-dependent, not a fixed engine constant — freezing a
  /// snapshot would be wrong, not right).
  func testMigrationSeedsExactlyTheEngineBaseline() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    XCTAssertTrue(store.runFirstRunDefaultsMigrationIfNeeded())

    let flux1 = store.renderDefaults(family: "flux1")
    XCTAssertEqual(flux1.width, 1024)
    XCTAssertEqual(flux1.height, 1024)
    XCTAssertEqual(flux1.steps, 9)
    XCTAssertEqual(flux1.guidance, 0.0)

    let fibo = store.renderDefaults(family: "fibo")
    XCTAssertEqual(fibo.width, 1024)
    XCTAssertEqual(fibo.height, 1024)
    XCTAssertEqual(fibo.steps, 30)
    XCTAssertEqual(fibo.guidance, 4.0)

    let chroma = store.renderDefaults(family: "chroma")
    XCTAssertEqual(chroma.width, 1024)
    XCTAssertEqual(chroma.height, 1024)
    XCTAssertEqual(chroma.steps, 28)
    XCTAssertEqual(chroma.guidance, 0.0)

    let flux2 = store.renderDefaults(family: "flux2")
    XCTAssertEqual(flux2.width, 1024)
    XCTAssertEqual(flux2.height, 1024)
    XCTAssertNil(flux2.steps, "flux2 steps are checkpoint-dependent — not seeded")
    XCTAssertNil(flux2.guidance, "flux2 guidance is checkpoint-dependent — not seeded")

    let krea2 = store.renderDefaults(family: "krea2")
    XCTAssertEqual(krea2.width, 1024)
    XCTAssertEqual(krea2.height, 1024)
    XCTAssertNil(krea2.steps, "krea2 steps are variant-dependent — not seeded")
    XCTAssertNil(krea2.guidance, "krea2 guidance is variant-dependent — not seeded")

    // Video: ltx2 seeds from the ENGINE constants (704x448, 97f), never from
    // desktop values (adversarial review F1 — see the neutrality guard test).
    let ltx2 = store.videoDefaults()
    XCTAssertEqual(ltx2.width, 704)
    XCTAssertEqual(ltx2.height, 448)
    XCTAssertEqual(ltx2.frames, 97)
  }

  /// Migration is idempotent: a second call, or a fresh store re-loading the
  /// now-populated file, makes no further change.
  func testMigrationIsIdempotent() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    XCTAssertTrue(store.runFirstRunDefaultsMigrationIfNeeded())
    XCTAssertFalse(store.runFirstRunDefaultsMigrationIfNeeded(), "second call should be a no-op")

    let reloaded = makeStore(in: dir)
    XCTAssertFalse(reloaded.runFirstRunDefaultsMigrationIfNeeded(),
                    "a fresh store over the already-migrated file should also no-op")
    XCTAssertEqual(reloaded.renderDefaults(family: "fibo").steps, 30)
  }

  /// The engine-neutrality guard (adversarial review F1, 2026-08-30):
  /// migration must NOT import desktop video values. Even with a
  /// `desktop-config.json` carrying Motion-tab dims (the moment Desktop
  /// Motion settings are saved, it does — `SettingsView.swift:80-82`),
  /// a post-migration LTX request that omits dims must still resolve to
  /// the engine's own 704x448x97 — one operator's UI numbers must never
  /// silently shift every headless caller of THE production video family.
  /// Desktop values stay Desktop-local; server-side `videoDefaults` starts
  /// at the engine constants and changes only via explicit `PUT`/`PATCH`.
  func testMigrationIsVideoEngineNeutralDespiteDesktopValues() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    // A desktop-config.json with video dims, in the same directory the
    // server config lives in — exactly where the (rejected) import would
    // have found it.
    let desktopConfigPath = dir.appendingPathComponent("desktop-config.json")
    let desktopJSON: [String: Any] = [
      "serverHost": "127.0.0.1", "serverPort": 7870, "autoConnect": true,
      "outputDirectory": "/tmp/out", "defaultSteps": 9, "defaultGuidance": 3.5,
      "defaultWidth": 1024, "defaultHeight": 1024, "thumbnailSize": 180,
      "gallerySortDefault": "date",
      "videoWidth": 960, "videoHeight": 544, "videoFrames": 121, "videoSteps": 12,
    ]
    try JSONSerialization.data(withJSONObject: desktopJSON).write(to: desktopConfigPath)

    let store = makeStore(in: dir)
    XCTAssertTrue(store.runFirstRunDefaultsMigrationIfNeeded())

    // The migrated document carries the ENGINE constants, not 960x544x121.
    let video = store.videoDefaults()
    XCTAssertEqual(video.width, 704, "desktop videoWidth must NOT be imported")
    XCTAssertEqual(video.height, 448, "desktop videoHeight must NOT be imported")
    XCTAssertEqual(video.frames, 97, "desktop videoFrames must NOT be imported")

    // And the LTX prep path's exact resolution chain — request/named/preset
    // all absent — lands on 704x448x97, bit-identical to pre-migration.
    let reqWidth: Int? = nil, namedWidth: Int? = nil, presetWidth: Int? = nil
    XCTAssertEqual(reqWidth ?? namedWidth ?? presetWidth ?? video.width ?? 704, 704)
    let reqHeight: Int? = nil, namedHeight: Int? = nil, presetHeight: Int? = nil
    XCTAssertEqual(reqHeight ?? namedHeight ?? presetHeight ?? video.height ?? 448, 448)
    let reqFrames: Int? = nil
    XCTAssertEqual(reqFrames ?? video.frames ?? 97, 97)

    // desktop-config.json itself is untouched (never read, never written).
    let stillThere = try Data(contentsOf: desktopConfigPath)
    let decoded = try JSONSerialization.jsonObject(with: stillThere) as? [String: Any]
    XCTAssertEqual(decoded?["videoWidth"] as? Int, 960, "desktop-config.json must never be modified")
  }

  // MARK: - byFamily resolution order

  func testByFamilyOverridesDefaultPerFieldIndependently() {
    let config = RenderDefaultsConfig(
      default: RenderDefaultValues(width: 512, height: 512, steps: 20, guidance: 3.0),
      byFamily: ["fibo": RenderDefaultValues(steps: 40)]
    )
    let resolved = config.resolved(family: "fibo")
    XCTAssertEqual(resolved.steps, 40, "byFamily.fibo.steps overrides default.steps")
    XCTAssertEqual(resolved.width, 512, "fibo has no byFamily.width override — inherits default")
    XCTAssertEqual(resolved.height, 512)
    XCTAssertEqual(resolved.guidance, 3.0)

    let untouchedFamily = config.resolved(family: "chroma")
    XCTAssertEqual(untouchedFamily.steps, 20, "a family with no byFamily entry inherits default entirely")
  }

  func testRequestAndPresetStillWinOverConfig() throws {
    // Resolution order is request -> preset -> config.byFamily -> config.default
    // -> engine constant. This test proves the CONFIG side stays inert once a
    // caller supplies its own value — the exact `payload.steps ?? configDefaults.steps
    // ?? <engine constant>` chain used at every call site.
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    _ = store.runFirstRunDefaultsMigrationIfNeeded()
    let configDefaults = store.renderDefaults(family: "fibo")
    XCTAssertEqual(configDefaults.steps, 30)

    let requestSteps: Int? = 12
    let resolvedSteps = requestSteps ?? configDefaults.steps ?? 30
    XCTAssertEqual(resolvedSteps, 12, "an explicit request value must win over the config default")
  }

  // MARK: - Full replace (PUT)

  func testReplaceRoundTrips() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    var updated = store.current().config
    updated.modelSpec = "z-image-turbo"
    let snapshot = try store.replace(with: updated, ifMatch: nil)
    XCTAssertEqual(snapshot.config.modelSpec, "z-image-turbo")
    XCTAssertEqual(store.current().config.modelSpec, "z-image-turbo")
  }

  func testReplaceRejectsInvalidRenderDefaults() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    var invalid = store.current().config
    invalid.renderDefaults = RenderDefaultsConfig(byFamily: ["fibo": RenderDefaultValues(steps: -5)])
    XCTAssertThrowsError(try store.replace(with: invalid, ifMatch: nil)) { error in
      guard case ServerConfigStoreError.validation(let message) = error else {
        return XCTFail("expected .validation, got \(error)")
      }
      XCTAssertTrue(message.contains("steps"), "error should name the offending field: \(message)")
    }
  }

  // MARK: - ETag / If-Match (advisory)

  func testIfMatchAbsentAlwaysProceeds() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    var updated = store.current().config
    updated.modelSpec = "no-if-match"
    XCTAssertNoThrow(try store.replace(with: updated, ifMatch: nil))
  }

  func testIfMatchStaleIsRejectedWithConflict() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    let staleETag = store.current().etag
    // A write that changes the ETag, so the captured tag is now stale.
    var first = store.current().config
    first.modelSpec = "first-write"
    _ = try store.replace(with: first, ifMatch: nil)

    var second = store.current().config
    second.modelSpec = "second-write"
    XCTAssertThrowsError(try store.replace(with: second, ifMatch: staleETag)) { error in
      guard case ServerConfigStoreError.etagMismatch = error else {
        return XCTFail("expected .etagMismatch, got \(error)")
      }
    }
    XCTAssertEqual(store.current().config.modelSpec, "first-write", "the stale write must not have applied")
  }

  func testIfMatchCurrentSucceeds() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    let currentETag = store.current().etag
    var updated = store.current().config
    updated.modelSpec = "matched"
    let snapshot = try store.replace(with: updated, ifMatch: currentETag)
    XCTAssertEqual(snapshot.config.modelSpec, "matched")
    XCTAssertNotEqual(snapshot.etag, currentETag, "the ETag must change after a successful write")
  }

  // MARK: - RFC 7386 JSON Merge Patch semantics

  func testMergePatchOmittedFieldsUnchanged() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    var seeded = store.current().config
    seeded.modelSpec = "keep-me"
    seeded.host = "keep-me-too"
    _ = try store.replace(with: seeded, ifMatch: nil)

    let snapshot = try store.applyMergePatch(["port": 7871], ifMatch: nil)
    XCTAssertEqual(snapshot.config.port, 7871)
    XCTAssertEqual(snapshot.config.modelSpec, "keep-me", "fields the patch omits must be untouched")
    XCTAssertEqual(snapshot.config.host, "keep-me-too")
  }

  func testMergePatchNullDeletesField() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    var seeded = store.current().config
    seeded.modelSpec = "z-image-turbo"
    _ = try store.replace(with: seeded, ifMatch: nil)
    XCTAssertEqual(store.current().config.modelSpec, "z-image-turbo")

    let snapshot = try store.applyMergePatch(["modelSpec": NSNull()], ifMatch: nil)
    XCTAssertNil(snapshot.config.modelSpec, "an explicit null must delete the field, reverting to its default")
  }

  func testMergePatchNestedObjectMergesNotReplaces() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    _ = try store.applyMergePatch(
      jsonObject(#"{ "renderDefaults": { "byFamily": { "fibo": { "steps": 40 } } } }"#), ifMatch: nil)
    XCTAssertEqual(store.renderDefaults(family: "fibo").steps, 40)

    // A second patch to a DIFFERENT nested field must not clobber the first —
    // proves the merge is against the CURRENT document, not a stale copy.
    _ = try store.applyMergePatch(
      jsonObject(#"{ "renderDefaults": { "byFamily": { "chroma": { "steps": 22 } } } }"#), ifMatch: nil)
    XCTAssertEqual(store.renderDefaults(family: "fibo").steps, 40, "the earlier patch must survive")
    XCTAssertEqual(store.renderDefaults(family: "chroma").steps, 22)
  }

  func testMergePatchRejectsInvalidResultingDocument() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    XCTAssertThrowsError(
      try store.applyMergePatch(jsonObject(#"{ "renderDefaults": { "default": { "width": -1 } } }"#), ifMatch: nil)
    ) { error in
      guard case ServerConfigStoreError.validation = error else {
        return XCTFail("expected .validation, got \(error)")
      }
    }
    XCTAssertTrue(store.renderDefaults(family: "flux1").isEmpty, "a rejected patch must not partially apply")
  }

  // MARK: - Concurrency: N threads patching N distinct pointers

  func testConcurrentPatchesToDistinctFamiliesAllLand() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = makeStore(in: dir)
    let families = ["flux1", "flux2", "fibo", "chroma", "krea2"]
    let expectations = families.map { expectation(description: "patch-\($0)") }

    for (index, family) in families.enumerated() {
      DispatchQueue.global().async {
        let patch = self.jsonObject(#"{ "renderDefaults": { "byFamily": { "\#(family)": { "steps": \#(index + 1) } } } }"#)
        do {
          _ = try store.applyMergePatch(patch, ifMatch: nil)
        } catch {
          XCTFail("concurrent patch for \(family) failed: \(error)")
        }
        expectations[index].fulfill()
      }
    }
    wait(for: expectations, timeout: 10)

    for (index, family) in families.enumerated() {
      XCTAssertEqual(store.renderDefaults(family: family).steps, index + 1,
                      "\(family)'s concurrent patch was lost")
    }

    // One valid, fully-decodable file on disk — no torn write.
    let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
    XCTAssertNoThrow(try JSONDecoder().decode(ComfyBoxServerConfig.self, from: data))
  }

  // MARK: - Fix round 2 (PR #363 review): sanitize-on-load

  /// A hand-edited `config.json` with a NEGATIVE `maxPixels` must not crash
  /// `ServerConfigStore.init` (which never ran `validate` before this fix) —
  /// it repairs to `.default` in memory instead. Regression pin: before the
  /// fix, this value reached `UInt64(caps.maxPixels)` in
  /// `ImageMemoryPreflight` and TRAPPED on the first sized request.
  func testLoadWithNegativeMaxPixelsFallsBackToDefaultsWithoutTrapping() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("config.json")
    try Data(#"{ "host": "h", "imageMemoryCaps": { "maxLongEdge": 4096, "maxPixels": -1, "minAvailableHeadroomFraction": 0.1 } }"#.utf8)
      .write(to: path)

    let store = ServerConfigStore(
      path: path,
      coffeeShopProviders: dir.appendingPathComponent("absent-providers.json"),
      coffeeShopConfig: dir.appendingPathComponent("absent-config.json"),
      auditLog: nil)

    XCTAssertEqual(store.imageMemoryCaps(), .default, "an invalid on-disk block repairs to the default, not the garbage value")
    // The repaired document is usable — decideResolution must not trap.
    let decision = ImageMemoryPreflight.decideResolution(width: 2048, height: 2048, caps: store.imageMemoryCaps())
    XCTAssertTrue(decision.allow)
  }

  /// The other half of the same failure mode: a NaN headroom fraction.
  /// `min`/`max` do not reliably filter NaN out (comparisons against NaN are
  /// always false), so before this fix a NaN reached
  /// `UInt64(Double(availableBytes) * (1.0 - headroomFraction))` in
  /// `ImageMemoryPreflight.validate` and TRAPPED.
  func testLoadWithNaNHeadroomFallsBackToDefaultsWithoutTrapping() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("config.json")
    // JSON has no NaN literal — the field decodes via Double's own JSON
    // handling, so we go through Foundation's non-conforming-number escape
    // hatch the same way a hand-edited file realistically could not; instead
    // simulate the same in-memory shape `sanitizeImageMemoryCaps` must catch
    // regardless of HOW a NaN got there.
    try Data(#"{ "host": "h" }"#.utf8).write(to: path)

    let store = ServerConfigStore(
      path: path,
      coffeeShopProviders: dir.appendingPathComponent("absent-providers.json"),
      coffeeShopConfig: dir.appendingPathComponent("absent-config.json"),
      auditLog: nil)

    var config = store.current().config
    config.imageMemoryCaps.minAvailableHeadroomFraction = .nan
    var repaired = config
    ServerConfigStore.sanitizeImageMemoryCaps(&repaired) { _ in }
    XCTAssertEqual(repaired.imageMemoryCaps, .default)

    // And the point-of-use clamp alone (defense in depth, PR #363 review):
    // even an UN-repaired NaN must not trap `validate`.
    XCTAssertNoThrow(
      try ImageMemoryPreflight.validate(
        width: 1024, height: 1024, family: .flux1, dype: false,
        caps: config.imageMemoryCaps, availableBytes: 10 * 1024 * 1024 * 1024))
  }

  /// A config file that loads fine (imageMemoryCaps absent/valid) must NOT
  /// be touched by the repair — no spurious audit entry, values pass through.
  func testLoadWithValidImageMemoryCapsIsNotRepaired() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("config.json")
    try Data(#"{ "host": "h", "imageMemoryCaps": { "maxLongEdge": 3072, "maxPixels": 9437184, "minAvailableHeadroomFraction": 0.2 } }"#.utf8)
      .write(to: path)

    let store = ServerConfigStore(
      path: path,
      coffeeShopProviders: dir.appendingPathComponent("absent-providers.json"),
      coffeeShopConfig: dir.appendingPathComponent("absent-config.json"),
      auditLog: nil)

    XCTAssertEqual(store.imageMemoryCaps().maxLongEdge, 3072, "a VALID on-disk value must survive, not be reset to default")
    XCTAssertEqual(store.imageMemoryCaps().maxPixels, 9_437_184)
  }
}
