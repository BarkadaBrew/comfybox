import XCTest
@testable import ZImage

/// #286 — `POST /v1/generate {"preset": name}` did not apply the LoRA stack
/// `POST /v1/presets/resolve` reports for that same name.
///
/// The image path treated `preset` as a provenance LABEL only
/// (`WarmServer.GeneratePayload.preset`), and the only code that set the
/// resident stack for a request was `if let loraEntries = payload.loras`. So a
/// preset-by-name render used whatever adapters the warm pipeline happened to
/// hold: an earlier `/v1/lora/swap`'s stack, an earlier job's per-job override,
/// or none at all after a restart — and reported `success: true` either way.
///
/// These pin the decision that closes it. Production shapes throughout: the
/// presets are the live `~/.comfybox/presets.json` entries this was reported
/// against (`krea-kira-avocado`, `krea-kira`, `kira-video-avocado`).
final class PresetLoRAStackTests: XCTestCase {

  // MARK: Fixtures — the live presets, as `/v1/presets/resolve` returns them

  /// The preset in the bug report: krea2 raw-stock, kroma declared OFF, five
  /// content LoRAs.
  private func kreaKiraAvocado() -> ResolvedPreset {
    ResolvedPreset(preset: ImagePreset(
      id: "krea-kira-avocado", name: "Kira Avocado", mediaKind: "image",
      model: "krea2-raw",
      loras: [
        LoraReference(filename: "snofs_krea_v1_3D.safetensors", scale: 0.8),
        LoraReference(filename: "Girly_Tiana.safetensors", scale: 0.6),
        LoraReference(filename: "LARP_v0-5.safetensors", scale: 1.5),
        LoraReference(filename: "snofs_photoSlider_000000200.safetensors", scale: 1.25),
        LoraReference(filename: "Krea2_TextFusion_Refusal_Reduction.safetensors", scale: 1.0),
      ],
      checkpointFamily: "raw-stock",
      kroma: KromaPolicy(strength: 0)))
  }

  /// Kira's standard lane: kroma ON at 0.6 with a named file, accel + polish.
  private func kreaKira() -> ResolvedPreset {
    ResolvedPreset(preset: ImagePreset(
      id: "krea-kira", name: "Kira", mediaKind: "image",
      model: "krea2-raw",
      loras: [
        LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel"),
        LoraReference(filename: "RealisticSnapshotKrea2.safetensors", scale: 0.4),
      ],
      checkpointFamily: "raw-accel",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")))
  }

  // MARK: The bug: a named preset must supply the stack

  func testNamedPresetSuppliesItsResolvedStack() {
    let decision = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "krea-kira-avocado",
      lookup: .resolved(kreaKiraAvocado()))

    guard case .apply(let id, let loras) = decision else {
      return XCTFail("preset-by-name must apply its resolved stack, got \(decision)")
    }
    XCTAssertEqual(id, "krea-kira-avocado")
    XCTAssertEqual(
      loras.map(\.filename),
      ["snofs_krea_v1_3D.safetensors", "Girly_Tiana.safetensors", "LARP_v0-5.safetensors",
       "snofs_photoSlider_000000200.safetensors", "Krea2_TextFusion_Refusal_Reduction.safetensors"])
    XCTAssertEqual(loras.map(\.scale), [0.8, 0.6, 1.5, 1.25, 1.0])
  }

  /// The render must NEVER silently keep the resident stack when a preset was
  /// named — that is the whole defect (0 LoRAs after a restart, 2 stale ones
  /// before it, `success: true` for both).
  func testNamedPresetIsNeverUnchanged() {
    for lookup: PresetLoRAStack.Lookup in [
      .resolved(kreaKiraAvocado()), .resolved(kreaKira()), .notFound,
      .invalid(reason: "missing kroma"),
    ] {
      let decision = PresetLoRAStack.decide(
        requestHasLoras: false, presetId: "krea-kira-avocado", lookup: lookup)
      XCTAssertNotEqual(
        decision, .unchanged,
        "a named preset must never leave the resident stack in place (lookup: \(lookup))")
    }
  }

  // MARK: D14 — kroma is a first-class field, prepended, never a `loras[]` row

  func testKromaIsPrependedAtItsDeclaredStrength() {
    let decision = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "krea-kira", lookup: .resolved(kreaKira()))

    guard case .apply(_, let loras) = decision else {
      return XCTFail("expected .apply, got \(decision)")
    }
    XCTAssertEqual(loras.count, 3)
    XCTAssertEqual(loras[0].filename, "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(loras[0].scale, 0.6)
    XCTAssertEqual(loras[0].role, "kroma")
    XCTAssertEqual(loras[1].filename, "krea2_turbo_distill_r256.safetensors")
    XCTAssertEqual(loras[1].role, "accel")
  }

  func testKromaStrengthZeroContributesNothing() {
    let decision = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "krea-kira-avocado",
      lookup: .resolved(kreaKiraAvocado()))
    guard case .apply(_, let loras) = decision else {
      return XCTFail("expected .apply, got \(decision)")
    }
    XCTAssertFalse(loras.contains { $0.role == "kroma" })
  }

  /// The engine has no family→default-kroma-file table (that policy lives in
  /// the client layer, FDD §3.17). Declared-on with no file is unreproducible,
  /// so it refuses rather than rendering a stack that is missing its kroma.
  func testKromaOnWithNoFileRefuses() {
    let preset = ResolvedPreset(preset: ImagePreset(
      id: "kroma-no-file", name: "x", mediaKind: "image", model: "krea2-raw",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0.6)))

    let decision = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "kroma-no-file", lookup: .resolved(preset))
    guard case .refuse(let message) = decision else {
      return XCTFail("expected .refuse, got \(decision)")
    }
    XCTAssertTrue(message.contains("kroma.file"), message)
  }

  /// The bypass `.diff` adapter is a preset dial with no engine application
  /// path — dropping it silently is the same class of defect as #286 itself.
  func testDeclaredBypassRefuses() {
    let preset = ResolvedPreset(preset: ImagePreset(
      id: "bypass-on", name: "x", mediaKind: "image", model: "krea2-raw",
      loras: [], checkpointFamily: "raw-stock", kroma: KromaPolicy(strength: 0),
      bypass: BypassPolicy(strength: 2.0)))

    guard case .refuse(let message) = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "bypass-on", lookup: .resolved(preset))
    else { return XCTFail("expected .refuse") }
    XCTAssertTrue(message.contains("bypass"), message)
  }

  // MARK: Fail loud, never quietly

  func testUnknownPresetRefuses() {
    guard case .refuse(let message) = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "krea-kira-typo", lookup: .notFound)
    else { return XCTFail("an unresolvable preset must refuse, not render on residency") }
    XCTAssertTrue(message.contains("krea-kira-typo"), message)
  }

  func testInvalidPresetRefusesTheSameWayResolveDoes() {
    guard case .refuse(let message) = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "krea-broken",
      lookup: .invalid(reason: "must declare \"kroma\""))
    else { return XCTFail("expected .refuse") }
    XCTAssertTrue(message.contains("krea-broken"), message)
    XCTAssertTrue(message.contains("kroma"), message)
  }

  func testVideoPresetOnTheImagePathRefuses() {
    let video = ResolvedPreset(preset: ImagePreset(
      id: "kira-video-avocado", name: "Kira video", mediaKind: "video",
      loras: [LoraReference(filename: "ltx-2.3-i2v-t2v-video-reasoning-lora-vbvr.safetensors", scale: 1)]))

    guard case .refuse(let message) = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "kira-video-avocado", lookup: .resolved(video))
    else { return XCTFail("a video preset must not push LTX adapters at the image pipeline") }
    XCTAssertTrue(message.contains("video"), message)
  }

  func testNamedPresetWithNoLookupRefuses() {
    guard case .refuse = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "krea-kira", lookup: nil)
    else { return XCTFail("an unresolved lookup must refuse, never fall through to residency") }
  }

  // MARK: No-regression — the shapes that already worked keep working

  /// Explicit `loras` bound correctly every time (confirmed in the report) and
  /// must stay untouched: the request wins, `preset` stays a label, and an id
  /// the engine does not know is harmless because nothing is read from it.
  func testExplicitLorasWinAndKeepPresetAsALabel() {
    XCTAssertEqual(
      PresetLoRAStack.decide(
        requestHasLoras: true, presetId: "krea-kira-avocado",
        lookup: .resolved(kreaKiraAvocado())),
      .requestExplicit)
    XCTAssertEqual(
      PresetLoRAStack.decide(requestHasLoras: true, presetId: "some-daemon-label", lookup: .notFound),
      .requestExplicit)
    XCTAssertEqual(
      PresetLoRAStack.decide(requestHasLoras: true, presetId: nil, lookup: nil),
      .requestExplicit)
  }

  /// A bare `/v1/generate` with neither field is the swap-first client's shape
  /// (`/v1/lora/swap` then generate) — it must still render on the stack the
  /// swap just installed.
  func testNoPresetNoLorasLeavesResidencyAlone() {
    XCTAssertEqual(
      PresetLoRAStack.decide(requestHasLoras: false, presetId: nil, lookup: nil), .unchanged)
    XCTAssertEqual(
      PresetLoRAStack.decide(requestHasLoras: false, presetId: "", lookup: nil), .unchanged)
    XCTAssertEqual(
      PresetLoRAStack.decide(requestHasLoras: false, presetId: "   ", lookup: nil), .unchanged)
  }

  /// A preset that resolves to an empty stack applies an EMPTY stack — it does
  /// not mean "leave whatever is loaded". `krea-bree` is exactly this shape.
  func testPresetWithNoLorasAppliesAnEmptyStack() {
    let bare = ResolvedPreset(preset: ImagePreset(
      id: "krea-bree", name: "Bree", mediaKind: "image", model: "kroma-v0.2-turbo",
      kroma: KromaPolicy(strength: 0)))
    XCTAssertEqual(
      PresetLoRAStack.decide(requestHasLoras: false, presetId: "krea-bree", lookup: .resolved(bare)),
      .apply(presetId: "krea-bree", loras: []))
  }

  // MARK: Parity with /v1/presets/resolve

  /// The decision must consume exactly what `/v1/presets/resolve` publishes —
  /// same store, same `resolve`, no parallel computation. Driven through a real
  /// `PresetStore` to prove the two agree.
  func testAgreesWithPresetStoreResolve() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("preset-lora-stack-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let store = PresetStore(
      path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
    let preset = ImagePreset(
      id: "krea-kira", name: "Kira", mediaKind: "image", model: "krea2-raw",
      loras: [
        LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel"),
        LoraReference(filename: "RealisticSnapshotKrea2.safetensors", scale: 0.4),
      ],
      checkpointFamily: "raw-accel",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors"))
    _ = try store.upsert(preset)

    let resolved = try store.resolve("krea-kira")
    guard case .apply(_, let loras) = PresetLoRAStack.decide(
      requestHasLoras: false, presetId: "krea-kira", lookup: .resolved(resolved))
    else { return XCTFail("expected .apply") }

    // Every LoRA `/v1/presets/resolve` reports is in the applied stack, at the
    // scale it reported, plus the structured kroma the same response declares.
    let appliedPairs: [String: Double] = Dictionary(
      uniqueKeysWithValues: loras.map { ($0.filename, $0.scale) })
    for reference in resolved.loras {
      XCTAssertEqual(
        appliedPairs[reference.filename], reference.scale,
        "resolve reported \(reference.filename) but the applied stack disagrees")
    }
    XCTAssertEqual(loras.count, resolved.loras.count + 1)
  }

  // MARK: The verification hook — `applied_loras` on the wire

  /// #286's other half: the daemon could not tell a wrong stack from a right
  /// one because the response said nothing about adapters for any family but
  /// Krea 2. `applied_loras` is the additive field it can diff against
  /// `/v1/presets/resolve` — snake_case, names + scales, no existing field
  /// renamed.
  func testAppliedLorasIsOnTheWireAndAdditive() throws {
    let response = GenerateResponse(
      success: true, outputPath: "/tmp/x.png", durationMs: 1234,
      appliedLoras: [
        LoRAState(.local("/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors", scale: 0.6)),
        LoRAState(.local("/loras/krea2_turbo_distill_r256.safetensors", scale: 0.6)),
      ])

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try encoder.encode(response)) as? [String: Any])

    // Every pre-#286 field keeps its name.
    XCTAssertEqual(object["success"] as? Bool, true)
    XCTAssertEqual(object["output_path"] as? String, "/tmp/x.png")
    XCTAssertEqual(object["duration_ms"] as? Int, 1234)

    let applied = try XCTUnwrap(object["applied_loras"] as? [[String: Any]])
    XCTAssertEqual(applied.count, 2)
    XCTAssertEqual(
      applied.first?["source"] as? String,
      "/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(applied.first?["scale"] as? Double, 0.6)
  }

  // MARK: The seam — `/v1/generate`'s own decode, end to end

  /// Mirrors `WarmServer.decode(_:from:)`.
  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  /// THE regression: the exact body Kira's daemon posts. Before the fix
  /// `loras` came out of the decode path nil, nothing was applied, and the
  /// render used whatever the warm pipeline still held.
  func testGenerateBodyWithOnlyAPresetGetsTheResolvedStack() throws {
    let payload = try decode(#"{"prompt":"a portrait","preset":"krea-kira-avocado"}"#)
    XCTAssertNil(payload.loras, "precondition: the wire body carries no `loras`")

    let expanded = try GeneratePayload.expandingPresetLoRAs(payload) { id in
      XCTAssertEqual(id, "krea-kira-avocado")
      return .resolved(self.kreaKiraAvocado())
    }

    let applied = try XCTUnwrap(expanded.loras)
    XCTAssertEqual(
      applied.map(\.path),
      ["snofs_krea_v1_3D.safetensors", "Girly_Tiana.safetensors", "LARP_v0-5.safetensors",
       "snofs_photoSlider_000000200.safetensors", "Krea2_TextFusion_Refusal_Reduction.safetensors"])
    XCTAssertEqual(applied.map(\.scale), [0.8, 0.6, 1.5, 1.25, 1.0])
    // `preset` is still the provenance label it always was.
    XCTAssertEqual(expanded.preset, "krea-kira-avocado")
  }

  func testGenerateBodyWithPresetAndKromaCarriesTheKromaRole() throws {
    let payload = try decode(#"{"prompt":"a portrait","preset":"krea-kira"}"#)
    let expanded = try GeneratePayload.expandingPresetLoRAs(payload) { _ in .resolved(self.kreaKira()) }
    let applied = try XCTUnwrap(expanded.loras)
    XCTAssertEqual(applied.first?.path, "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(applied.first?.role, "kroma")
    XCTAssertEqual(applied.first?.scale, 0.6)
  }

  func testGenerateBodyWithAnUnknownPresetThrowsInsteadOfRendering() throws {
    let payload = try decode(#"{"prompt":"a portrait","preset":"krea-kira-typo"}"#)
    XCTAssertThrowsError(
      try GeneratePayload.expandingPresetLoRAs(payload) { _ in .notFound }
    ) { error in
      XCTAssertTrue(
        "\(error)".contains("krea-kira-typo"),
        "the refusal must name the preset, got: \(error)")
    }
  }

  /// Explicit `loras` still win and are left exactly as sent, even when the
  /// request also names a preset the engine could have expanded.
  func testGenerateBodyWithExplicitLorasIsUntouched() throws {
    let payload = try decode(
      #"{"prompt":"x","preset":"krea-kira","loras":[{"path":"purelens_krea2.safetensors","scale":1.0}]}"#)
    let expanded = try GeneratePayload.expandingPresetLoRAs(payload) { _ in
      XCTFail("a request with explicit `loras` must not consult the preset store")
      return .notFound
    }
    XCTAssertEqual(expanded.loras?.map(\.path), ["purelens_krea2.safetensors"])
  }

  /// A bare body — the swap-first client's shape — is untouched.
  func testGenerateBodyWithNeitherFieldIsUntouched() throws {
    let payload = try decode(#"{"prompt":"x"}"#)
    let expanded = try GeneratePayload.expandingPresetLoRAs(payload) { _ in
      XCTFail("no preset named — the store must not be consulted")
      return .notFound
    }
    XCTAssertNil(expanded.loras)
  }

  /// A family with no LoRA path at all reports the key ABSENT, never an empty
  /// array — "engine has no stack here" must not read as "rendered bare".
  func testAbsentAppliedLorasIsNotAnEmptyArray() throws {
    let response = GenerateResponse(success: true, outputPath: "/tmp/x.png", durationMs: 1)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try encoder.encode(response)) as? [String: Any])
    XCTAssertNil(object["applied_loras"])
  }
}
