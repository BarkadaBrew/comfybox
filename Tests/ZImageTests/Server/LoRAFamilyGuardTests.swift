import Foundation
import XCTest

@testable import ZImage

/// #402 (coffeeshop-server #1681, the Luxe_Sensual incident): wiring
/// coverage for the cross-family LoRA guard at the three enforcement points
/// the ruling named — `POST /v1/lora/swap`, per-request `loras[]` on
/// `/v1/generate` / `/v1/generate/async` (via the shared
/// `decodedGeneratePayload` choke point), and `POST /v1/presets` (covered in
/// `PresetStoreLoRAFamilyGuardTests`). Exercises the pure static functions
/// and `WarmServer.errorResponse(for:)` directly — no listening server, no
/// weights, per `WarmServerRejectionTests`' own pattern.
final class LoRAFamilyGuardTests: XCTestCase {

  private func bodyString(_ response: HTTPResponse) -> String {
    String(decoding: response.body, as: UTF8.self)
  }

  /// Minimal, valid `LoRALibraryEntry` declaring `compat` — mirrors
  /// `LoRALibraryEntryCodingTests.entryJSON`'s required-fields shape.
  private func makeEntry(id: String = "test-lora", compat: [String]) -> LoRALibraryEntry {
    let json = """
    {
      "id": "\(id)",
      "filename": "\(id).safetensors",
      "relative_path": "\(id).safetensors",
      "size_bytes": 1024,
      "model_compatibility": [\(compat.map { "\"\($0)\"" }.joined(separator: ","))],
      "format": "lora",
      "rank": 64,
      "key_count": 10,
      "layer_targets": ["attention"],
      "triggerwords": [],
      "recommended_scale": 1.0,
      "scale_range": [0.0, 2.0],
      "tags": [],
      "category": "uncategorized",
      "notes": "",
      "date_added": "2026-09-04",
      "quarantined": false
    }
    """
    let decoder = JSONDecoder()
    return try! decoder.decode(LoRALibraryEntry.self, from: Data(json.utf8))
  }

  // MARK: - WarmModelFamily.loraCompatibilityFamily (#393)

  func testWarmModelFamilyMapsToCanonicalGroups() {
    XCTAssertEqual(WarmModelFamily.flux1.loraCompatibilityFamily, "z-image")
    XCTAssertEqual(WarmModelFamily.flux2.loraCompatibilityFamily, "flux2-klein")
    XCTAssertEqual(WarmModelFamily.fibo.loraCompatibilityFamily, "fibo")
    XCTAssertEqual(WarmModelFamily.chroma.loraCompatibilityFamily, "chroma")
    XCTAssertEqual(WarmModelFamily.krea2.loraCompatibilityFamily, "krea2")
  }

  // MARK: - WarmServer.validateLoRAFamilyCompatibility (shared by swap + generate)

  func testMismatchThrowsNamingLoRAAndBothFamilies() {
    let ltxEntry = makeEntry(id: "video-lora", compat: ["ltx"])
    XCTAssertThrowsError(
      try WarmServer.validateLoRAFamilyCompatibility(
        entries: [LoRAEntry(path: "video-lora.safetensors", scale: 1.0)],
        targetFamily: "z-image",
        lookup: { _ in ltxEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(let path, let libraryId, let declaredTags, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(path, "video-lora.safetensors")
      XCTAssertEqual(libraryId, "video-lora")
      XCTAssertEqual(declaredTags, ["ltx"])
      XCTAssertEqual(families, ["ltx"])
      XCTAssertEqual(target, "z-image")

      // Ruling 5: the 400 must name the full path, the raw declared tag,
      // the normalized group, and the target family.
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      let body = bodyString(response)
      XCTAssertTrue(body.contains("video-lora.safetensors"), body)
      XCTAssertTrue(body.contains("video-lora"), body)
      XCTAssertTrue(body.contains("ltx"), body)
      XCTAssertTrue(body.contains("z-image"), body)
    }
  }

  func testUnknownLoRANeverThrowsOnlyLogsWarning() throws {
    var logged: [String] = []
    try WarmServer.validateLoRAFamilyCompatibility(
      entries: [LoRAEntry(path: "never-scanned.safetensors", scale: 1.0)],
      targetFamily: "z-image",
      lookup: { _ in nil },
      log: { logged.append($0) })
    XCTAssertTrue(logged.contains { $0.contains("never-scanned.safetensors") })
  }

  func testCompatibleLoRAPassesSilently() throws {
    let entry = makeEntry(id: "z-lora", compat: ["z-image"])
    var logged: [String] = []
    try WarmServer.validateLoRAFamilyCompatibility(
      entries: [LoRAEntry(path: "z-lora.safetensors", scale: 1.0)],
      targetFamily: "z-image",
      lookup: { _ in entry },
      log: { logged.append($0) })
    XCTAssertTrue(logged.isEmpty)
  }

  // MARK: - /v1/generate & /v1/generate/async (decodedGeneratePayload choke point)

  private func makePresetStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-lora-family-guard-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  /// A video (ltx) LoRA sent on a bare (default flux1/z-image) generate
  /// request — the Luxe_Sensual incident, reproduced through the real
  /// decode choke point both `/v1/generate` and `/v1/generate/async` share.
  func testDecodedGeneratePayloadRejectsVideoLoRAOnImageRequest() throws {
    let store = try makePresetStore()
    let ltxEntry = makeEntry(id: "sulphur-video", compat: ["ltx"])
    XCTAssertThrowsError(
      try WarmServer.decodedGeneratePayload(
        from: Data(#"{"prompt":"x","loras":[{"path":"sulphur-video.safetensors"}]}"#.utf8),
        store: store, configuration: configuration,
        loraLookup: { _ in ltxEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(let path, _, let declaredTags, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(path, "sulphur-video.safetensors")
      XCTAssertEqual(declaredTags, ["ltx"])
      XCTAssertEqual(families, ["ltx"])
      XCTAssertEqual(target, "z-image")
    }
  }

  /// An image (z-image) LoRA sent on an explicit `model` that resolves to
  /// krea2 — the inverse direction, and comfybox#393's flux1/krea2 case via
  /// the SAME family resolution `ImageMemoryPreflight.resolvedFamily` uses.
  func testDecodedGeneratePayloadRejectsImageLoRAOnKrea2Model() throws {
    let store = try makePresetStore()
    let zImageEntry = makeEntry(id: "portrait-style", compat: ["z-image"])
    XCTAssertThrowsError(
      try WarmServer.decodedGeneratePayload(
        from: Data(#"{"prompt":"x","model":"krea2","loras":[{"path":"portrait-style.safetensors"}]}"#.utf8),
        store: store, configuration: configuration,
        loraLookup: { _ in zImageEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(let path, _, let declaredTags, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(path, "portrait-style.safetensors")
      XCTAssertEqual(declaredTags, ["z-image"])
      XCTAssertEqual(families, ["z-image"])
      XCTAssertEqual(target, "krea2")
    }
  }

  /// #22's `gateSubmission: false` (crash-recovery replay) must skip this
  /// gate too, same as the memory preflight beside it — a job already
  /// accepted must never be re-refused by a gate that did not exist (or
  /// disagreed) when it was submitted.
  func testGateSubmissionFalseSkipsTheGuardEntirely() throws {
    let store = try makePresetStore()
    let ltxEntry = makeEntry(id: "sulphur-video-2", compat: ["ltx"])
    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","loras":[{"path":"sulphur-video-2.safetensors"}]}"#.utf8),
      store: store, configuration: configuration, gateSubmission: false,
      loraLookup: { _ in ltxEntry })
    XCTAssertEqual(payload.loras?.first?.path, "sulphur-video-2.safetensors")
  }

  /// Additivity (#402 ruling 3): a request with a NORMAL, compatible krea2
  /// LoRA stack against an explicit krea2 `model` renders byte-identically —
  /// same fields, no throw — before and after this guard exists. Pinned
  /// against the resolver's own output rather than a hand-typed literal, so
  /// a future accidental behavior change on this exact shape fails loudly.
  func testExistingCompatibleKrea2RequestIsByteIdenticalPin() throws {
    let store = try makePresetStore()
    let accelEntry = makeEntry(id: "krea2_turbo_distill_r256", compat: ["krea2"])
    let body = Data(
      #"""
      {"prompt":"a portrait","model":"krea2","width":1024,"height":1024,
       "loras":[{"path":"krea2_turbo_distill_r256.safetensors","scale":0.6,"role":"accel"}]}
      """#.utf8)
    let payload = try WarmServer.decodedGeneratePayload(
      from: body, store: store, configuration: configuration,
      loraLookup: { _ in accelEntry })
    // Pin: the fields this guard could plausibly disturb, unchanged.
    XCTAssertEqual(payload.model, "krea2")
    XCTAssertEqual(payload.loras?.count, 1)
    XCTAssertEqual(payload.loras?.first?.path, "krea2_turbo_distill_r256.safetensors")
    XCTAssertEqual(payload.loras?.first?.scale, 0.6)
    XCTAssertEqual(payload.loras?.first?.role, "accel")
    XCTAssertNotNil(payload.memoryEstimateBytes, "the memory preflight beside this guard still ran")
  }

  /// A LoRA the library has never scanned (no lookup match) must never block
  /// a request that was accepted before this guard existed — the default
  /// `loraLookup` (used when the route's real library has nothing for this
  /// filename) resolves to "unknown", which is always allowed.
  func testUnscannedLoRAOnGenerateNeverBlocks() throws {
    let store = try makePresetStore()
    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","loras":[{"path":"never-scanned.safetensors"}]}"#.utf8),
      store: store, configuration: configuration)
    XCTAssertEqual(payload.loras?.first?.path, "never-scanned.safetensors")
  }

  // MARK: - #402 fix round 1 (Critical 3): the LTX-2 video path, target "ltx"
  //
  // `prepareLocalVideo` (the choke point both `/v1/video/generate` and
  // `/v1/video/generate/async` share) requires configured LTX-2 weights
  // paths and is not reachable without them — AGENT-RULES forbids loading
  // weights from this worktree. The insertion point calls the SAME shared
  // `WarmServer.validateLoRAFamilyCompatibility` exercised throughout this
  // file with `targetFamily: "ltx"`; these two tests pin that exact call
  // shape so the video wiring's behavior is covered by the same evidence as
  // the image side, without a live generator.
  func testVideoTargetRejectsImageLoRA() {
    let entry = makeEntry(id: "portrait-style-2", compat: ["z-image"])
    XCTAssertThrowsError(
      try WarmServer.validateLoRAFamilyCompatibility(
        entries: [LoRAEntry(path: "portrait-style-2.safetensors", scale: 1.0)],
        targetFamily: "ltx", lookup: { _ in entry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(_, _, let declaredTags, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(declaredTags, ["z-image"])
      XCTAssertEqual(families, ["z-image"])
      XCTAssertEqual(target, "ltx")
    }
  }

  func testVideoTargetAllowsCompatibleAndUnscannedLoRAs() throws {
    let ltxEntry = makeEntry(id: "ltx-act-lora", compat: ["ltx"])
    var logged: [String] = []
    try WarmServer.validateLoRAFamilyCompatibility(
      entries: [
        LoRAEntry(path: "ltx-act-lora.safetensors", scale: 1.0),
        LoRAEntry(path: "never-scanned-video.safetensors", scale: 1.0),
      ],
      targetFamily: "ltx", lookup: { name in name.contains("ltx-act-lora") ? ltxEntry : nil },
      log: { logged.append($0) })
    XCTAssertTrue(logged.contains { $0.contains("never-scanned-video.safetensors") })
  }

  // MARK: - #402 fix round 1/2 (Critical 1): POST /v1/lora/swap

  /// Cold swap: no declared family (checkpoint_family/model absent) and
  /// `hasLoadedModel` is false — nothing has EVER been loaded on this
  /// coordinator. A krea2-tagged LoRA (a REAL library entry, e.g.
  /// Filipina_Pinay_Women.safetensors) must be allowed with a warning,
  /// never a 400. This is the exact production pattern SwapResidencyRestore
  /// (30735df) established: Kira's stack is swapped BEFORE krea2 becomes
  /// resident.
  func testColdSwapWithNoFamilyAndNoModelEverLoadedIsAllowedWithWarning() throws {
    let krea2Entry = makeEntry(id: "Filipina_Pinay_Women", compat: ["krea2"])
    var logged: [String] = []
    XCTAssertNoThrow(
      try WarmServer.validateLoRASwapCompatibility(
        entries: [LoRAEntry(path: "Filipina_Pinay_Women.safetensors", scale: 1.0)],
        checkpointFamily: nil, model: nil, residentFamily: .flux1, hasLoadedModel: false,
        lookup: { _ in krea2Entry }, log: { logged.append($0) }))
    XCTAssertTrue(
      logged.contains { $0.contains("Filipina_Pinay_Women.safetensors") && $0.contains("ever been") })
  }

  /// Swap declares its OWN target family (checkpoint_family "raw-accel" →
  /// krea2) even on a cold coordinator: an ltx-tagged LoRA is rejected
  /// against that DECLARED family, not silently skipped.
  func testSwapWithDeclaredKrea2FamilyRejectsLtxTaggedLoRA() {
    let ltxEntry = makeEntry(id: "sulphur-video-3", compat: ["ltx"])
    XCTAssertThrowsError(
      try WarmServer.validateLoRASwapCompatibility(
        entries: [LoRAEntry(path: "sulphur-video-3.safetensors", scale: 1.0)],
        checkpointFamily: "raw-accel", model: nil, residentFamily: .flux1, hasLoadedModel: false,
        lookup: { _ in ltxEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(_, _, _, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(families, ["ltx"])
      XCTAssertEqual(target, "krea2")
    }
  }

  /// Once krea2 is genuinely resident (`hasLoadedModel: true`), the swap
  /// route trusts it even with no declared family — a z-image-tagged LoRA
  /// is rejected.
  func testSwapAfterKrea2LoadedRejectsZImageTaggedLoRA() {
    let zImageEntry = makeEntry(id: "portrait-style-3", compat: ["z-image"])
    XCTAssertThrowsError(
      try WarmServer.validateLoRASwapCompatibility(
        entries: [LoRAEntry(path: "portrait-style-3.safetensors", scale: 1.0)],
        checkpointFamily: nil, model: nil, residentFamily: .krea2, hasLoadedModel: true,
        lookup: { _ in zImageEntry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(_, _, _, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(families, ["z-image"])
      XCTAssertEqual(target, "krea2")
    }
  }

  /// Fix round 2 (Critical 1 re-review): a GENUINELY resident Z-Image
  /// pipeline (`residentFamily: .flux1, hasLoadedModel: true` — NOT the
  /// cold/never-loaded case, which shares the same `.flux1` value) must
  /// still gate a krea2-tagged LoRA. Round 1's `residentFamily == .flux1 ?
  /// nil` collapsed this into the cold case and left the guard silently
  /// inert here — 89 of 135 live library entries are z-image-tagged.
  func testSwapWithZImageGenuinelyLoadedRejectsKrea2TaggedLoRA() {
    let krea2Entry = makeEntry(id: "some-krea2-lora", compat: ["krea2"])
    XCTAssertThrowsError(
      try WarmServer.validateLoRASwapCompatibility(
        entries: [LoRAEntry(path: "some-krea2-lora.safetensors", scale: 1.0)],
        checkpointFamily: nil, model: nil, residentFamily: .flux1, hasLoadedModel: true,
        lookup: { _ in krea2Entry })
    ) { error in
      guard case WarmServerError.loraFamilyMismatch(_, _, _, let families, let target) = error else {
        return XCTFail("expected .loraFamilyMismatch, got \(error)")
      }
      XCTAssertEqual(families, ["krea2"])
      XCTAssertEqual(target, "z-image")
    }
  }

  /// `loraSwapTargetFamily` in isolation, for every input combination the
  /// route can produce — including the round-2 fix's core distinction:
  /// `.flux1` resident with `hasLoadedModel: true` resolves to "z-image"
  /// (trusted), while the SAME `.flux1` with `hasLoadedModel: false`
  /// resolves to nil (skip).
  func testLoraSwapTargetFamilyResolution() {
    XCTAssertNil(
      WarmServer.loraSwapTargetFamily(
        checkpointFamily: nil, model: nil, residentFamily: .flux1, hasLoadedModel: false))
    XCTAssertEqual(
      WarmServer.loraSwapTargetFamily(
        checkpointFamily: nil, model: nil, residentFamily: .flux1, hasLoadedModel: true), "z-image")
    XCTAssertEqual(
      WarmServer.loraSwapTargetFamily(
        checkpointFamily: nil, model: nil, residentFamily: .krea2, hasLoadedModel: true), "krea2")
    XCTAssertEqual(
      WarmServer.loraSwapTargetFamily(
        checkpointFamily: "raw-accel", model: nil, residentFamily: .flux1, hasLoadedModel: false), "krea2")
    XCTAssertEqual(
      WarmServer.loraSwapTargetFamily(
        checkpointFamily: nil, model: "krea2", residentFamily: .flux1, hasLoadedModel: false), "krea2")
    // A declared family wins even when the resident family disagrees.
    XCTAssertEqual(
      WarmServer.loraSwapTargetFamily(
        checkpointFamily: "raw-accel", model: nil, residentFamily: .flux2, hasLoadedModel: true), "krea2")
  }

  // MARK: - #402 fix round 1/2 (Critical 2): flux1/Flux-derived ambiguity, wired

  /// The scanner-ambiguity exception (`LoRACompatibility.checkFamily`) flows
  /// through the shared guard unchanged: a "flux1"-tagged LoRA against a
  /// krea2 target warns instead of throwing.
  func testFlux1TaggedLoRAOnKrea2TargetWarnsInsteadOfThrowing() throws {
    let entry = makeEntry(id: "ambiguous-flux1", compat: ["flux1"])
    var logged: [String] = []
    XCTAssertNoThrow(
      try WarmServer.validateLoRAFamilyCompatibility(
        entries: [LoRAEntry(path: "ambiguous-flux1.safetensors", scale: 1.0)],
        targetFamily: "krea2", lookup: { _ in entry }, log: { logged.append($0) }))
    XCTAssertTrue(logged.contains { $0.contains("ambiguous-flux1.safetensors") && $0.contains("ambiguous") })
  }

  /// Fix round 2: the real case named in review — `fk-adrianoanal.safetensors`
  /// was scanned (pre-fix) as `["flux1"]` but is actually Flux 2 Klein. A
  /// klein-9b target must warn, not 400, until it's rescanned.
  func testFlux1TaggedLoRAOnKlein9bTargetWarnsInsteadOfThrowing() throws {
    let entry = makeEntry(id: "fk-adrianoanal", compat: ["flux1"])
    var logged: [String] = []
    XCTAssertNoThrow(
      try WarmServer.validateLoRAFamilyCompatibility(
        entries: [LoRAEntry(path: "fk-adrianoanal.safetensors", scale: 1.0)],
        targetFamily: "flux2-klein", lookup: { _ in entry }, log: { logged.append($0) }))
    XCTAssertTrue(logged.contains { $0.contains("fk-adrianoanal.safetensors") && $0.contains("ambiguous") })
  }

  /// Fix round 2: the other real case named in review — `chroma-unlocked-v47…`
  /// carries `ss_base_model_version: "flux1"` (an explicit marker, so still
  /// confidently "flux1" post-fix) but Chroma is itself Flux-derived. A
  /// chroma target must warn, not 400.
  func testFlux1TaggedLoRAOnChromaTargetWarnsInsteadOfThrowing() throws {
    let entry = makeEntry(id: "chroma-unlocked-v47", compat: ["flux1"])
    var logged: [String] = []
    XCTAssertNoThrow(
      try WarmServer.validateLoRAFamilyCompatibility(
        entries: [LoRAEntry(path: "chroma-unlocked-v47.safetensors", scale: 1.0)],
        targetFamily: "chroma", lookup: { _ in entry }, log: { logged.append($0) }))
    XCTAssertTrue(logged.contains { $0.contains("chroma-unlocked-v47.safetensors") && $0.contains("ambiguous") })
  }

  /// The two architectures Flux genuinely shares nothing with — z-image and
  /// ltx — are UNAFFECTED by the broadened carve-out: a bare "flux1" tag is
  /// still a confident, real mismatch against either.
  func testFlux1TaggedLoRAStillRejectsAgainstZImageAndLtx() {
    for target in ["z-image", "ltx"] {
      let entry = makeEntry(id: "still-fatal-\(target)", compat: ["flux1"])
      XCTAssertThrowsError(
        try WarmServer.validateLoRAFamilyCompatibility(
          entries: [LoRAEntry(path: "still-fatal-\(target).safetensors", scale: 1.0)],
          targetFamily: target, lookup: { _ in entry }),
        "flux1 must still reject against \(target)")
    }
  }
}
