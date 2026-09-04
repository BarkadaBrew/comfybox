import XCTest
@testable import ZImage

/// #286 review round 1, I6 — the route, not just the decision.
///
/// Round 1's tests all exercised the pure helper: removing the
/// `decodedGeneratePayload` expansion call would have left every one of them
/// green. These drive `WarmServer.expandGeneratePayload` — the exact function
/// the `/v1/generate` and `/v1/generate/async` handlers call — over a real
/// `PresetStore`, from the wire body inward, and
/// `WarmServer.rawBody(_:expandedWith:)`, which is what the crash-recovery
/// snapshot stores.
final class GeneratePresetRouteTests: XCTestCase {

  // MARK: Harness

  private func makeStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-generate-preset-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  /// Mirrors `WarmServer.decode(_:from:)` — the generate routes decode with
  /// `.convertFromSnakeCase`.
  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  /// The production expansion, over an explicit store.
  private func expand(
    _ json: String, store: PresetStore,
    stageNearline: ([LoRAEntry]) -> [LoRAEntry] = { $0 }
  ) throws -> GeneratePayload {
    try WarmServer.expandGeneratePayload(
      try decode(json), store: store, stageNearline: stageNearline)
  }

  /// Kira's live lane: krea2-raw, kroma 0.6 with a named file, accel + polish.
  @discardableResult
  private func seedKreaKira(_ store: PresetStore, steps: Int? = nil) throws -> ImagePreset {
    try store.upsert(ImagePreset(
      id: "krea-kira", name: "Kira", mediaKind: "image", model: "krea2-raw", steps: steps,
      loras: [
        LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel"),
        LoraReference(filename: "RealisticSnapshotKrea2.safetensors", scale: 0.4),
      ],
      checkpointFamily: "raw-accel",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")))
  }

  // MARK: I6 — preset-only request (THE regression)

  /// The exact body Kira's daemon posts. Before the fix, `loras` came out of
  /// the route nil, nothing was applied, and the render used whatever the warm
  /// pipeline still held — on whatever base was active.
  func testPresetOnlyRequestGetsTheStackAndTheModel() throws {
    let store = try makeStore()
    try seedKreaKira(store)

    let payload = try expand(#"{"prompt":"a portrait","preset":"krea-kira"}"#, store: store)

    let loras = try XCTUnwrap(payload.loras)
    XCTAssertEqual(loras.map(\.path), [
      "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
      "krea2_turbo_distill_r256.safetensors",
      "RealisticSnapshotKrea2.safetensors",
    ])
    XCTAssertEqual(loras.map { $0.scale ?? -1 }, [0.6, 0.6, 0.4])
    XCTAssertEqual(loras.first?.role, "kroma")
    XCTAssertEqual(loras[1].role, "accel")
    // C1: the preset's base travels with its adapters.
    XCTAssertEqual(payload.model, "krea2-raw")
    XCTAssertNil(payload.presetUnresolved)
    XCTAssertNil(payload.presetStackMismatch)
    // `preset` is still the provenance label it always was.
    XCTAssertEqual(payload.preset, "krea-kira")
  }

  /// The stack the route builds must be the stack `/v1/presets/resolve`
  /// publishes for the same id, through the same store.
  func testRouteStackMatchesPresetsResolve() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let resolved = try store.resolve("krea-kira")
    let payload = try expand(#"{"prompt":"x","preset":"krea-kira"}"#, store: store)

    let applied = try XCTUnwrap(payload.loras)
    for reference in resolved.loras {
      let match = applied.first { $0.path == reference.filename }
      XCTAssertEqual(
        match?.scale, Float(reference.scale),
        "resolve reported \(reference.filename)@\(reference.scale); the route disagrees")
    }
    XCTAssertEqual(applied.count, resolved.loras.count + 1, "plus the structured kroma")
    XCTAssertEqual(payload.model, resolved.model)
  }

  func testDeclaredStepsTravelOnlyWhenTheRequestOmitsThem() throws {
    let store = try makeStore()
    try seedKreaKira(store, steps: 52)
    XCTAssertEqual(try expand(#"{"prompt":"x","preset":"krea-kira"}"#, store: store).steps, 52)
    XCTAssertEqual(try expand(#"{"prompt":"x","preset":"krea-kira","steps":9}"#, store: store).steps, 9)
  }

  // MARK: I6 — preset + explicit loras

  /// Explicit `loras` keep their precedence, and a disagreement is REPORTED.
  /// This is the real async production shape: the client sends `preset` plus a
  /// flat list that has already dropped the structured kroma.
  func testPresetPlusExplicitLorasKeepsExplicitAndFlagsTheMismatch() throws {
    let store = try makeStore()
    try seedKreaKira(store)

    let payload = try expand(#"""
      {"prompt":"x","preset":"krea-kira","loras":[
        {"path":"krea2_turbo_distill_r256.safetensors","scale":0.6},
        {"path":"RealisticSnapshotKrea2.safetensors","scale":0.4}]}
      """#, store: store)

    XCTAssertEqual(
      payload.loras?.map(\.path),
      ["krea2_turbo_distill_r256.safetensors", "RealisticSnapshotKrea2.safetensors"],
      "the explicit list must be left exactly as sent")
    XCTAssertEqual(payload.presetStackMismatch, true, "kroma is missing from it — say so")
    // The base still travels, which is what stops the explicit adapters landing
    // on whatever model happens to be active.
    XCTAssertEqual(payload.model, "krea2-raw")
  }

  func testPresetPlusMatchingExplicitLorasRaisesNoFlag() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let payload = try expand(#"""
      {"prompt":"x","preset":"krea-kira","loras":[
        {"path":"/Volumes/Bolt/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors","scale":0.6},
        {"path":"/Volumes/Bolt/loras/krea2_turbo_distill_r256.safetensors","scale":0.6},
        {"path":"/Volumes/Bolt/loras/RealisticSnapshotKrea2.safetensors","scale":0.4}]}
      """#, store: store)
    XCTAssertNil(payload.presetStackMismatch, "same adapters by name and scale — no disagreement")
  }

  // MARK: I6 — unknown preset (C2: no 400)

  /// The pre-#286 contract: an unknown `preset` is harmless provenance. It must
  /// stay that way — the render proceeds untouched — and the response says so.
  func testUnknownPresetIsALabelNotAnError() throws {
    let store = try makeStore()
    let payload = try expand(#"{"prompt":"x","preset":"some-daemon-label"}"#, store: store)
    XCTAssertNil(payload.loras, "nothing applied — exactly as before #286")
    XCTAssertNil(payload.model)
    XCTAssertNil(payload.steps)
    XCTAssertEqual(payload.presetUnresolved, "some-daemon-label")
    XCTAssertEqual(payload.preset, "some-daemon-label")
  }

  func testUnknownPresetWithExplicitLorasLeavesThemAlone() throws {
    let store = try makeStore()
    let payload = try expand(
      #"{"prompt":"x","preset":"gone","loras":[{"path":"purelens_krea2.safetensors","scale":1.0}]}"#,
      store: store)
    XCTAssertEqual(payload.loras?.map(\.path), ["purelens_krea2.safetensors"])
    XCTAssertEqual(payload.presetUnresolved, "gone")
    XCTAssertNil(payload.presetStackMismatch, "nothing to compare against")
  }

  // MARK: I6 — model mismatch is a 409

  func testContradictingModelIsA409NamingAllThree() throws {
    let store = try makeStore()
    try seedKreaKira(store)

    XCTAssertThrowsError(
      try expand(#"{"prompt":"x","preset":"krea-kira","model":"z-image-turbo"}"#, store: store)
    ) { error in
      guard let warm = error as? WarmServerError,
            case .presetModelConflict(let preset, let presetModel, let requestModel) = warm
      else { return XCTFail("expected .presetModelConflict, got \(error)") }
      XCTAssertEqual(preset, "krea-kira")
      XCTAssertEqual(presetModel, "krea2-raw")
      XCTAssertEqual(requestModel, "z-image-turbo")

      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 409, "a contradiction is a conflict, not a malformed request")
      let text = String(decoding: response.body, as: UTF8.self)
      for needle in ["krea-kira", "krea2-raw", "z-image-turbo"] {
        XCTAssertTrue(text.contains(needle), "409 body must name \(needle): \(text)")
      }
    }
  }

  /// An alias and the directory it resolves to are the SAME model — a spelling
  /// difference must not 409 a valid request.
  func testEquivalentModelSpellingIsNotAConflict() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let resolvedPath = WarmServer.parseModelSpec(from: "krea2-raw")
    let json = "{\"prompt\":\"x\",\"preset\":\"krea-kira\",\"model\":\"\(resolvedPath)\"}"
    let payload = try expand(json, store: store)
    XCTAssertEqual(payload.model, resolvedPath, "the request's own spelling stands")
    XCTAssertEqual(payload.loras?.count, 3)
  }

  // MARK: I3 — nearline staging

  /// A preset may name an adapter that lives only on nearline storage.
  /// `/v1/lora/swap` stages those; the expanded stack must be staged the same
  /// way, with the semantic `role` surviving the rewrite.
  func testExpandedStackGoesThroughNearlineStaging() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    var staged: [String] = []
    let payload = try expand(#"{"prompt":"x","preset":"krea-kira"}"#, store: store) { entries in
      staged = entries.map(\.path)
      return entries.map { LoRAEntry(path: "/nearline/\($0.path)", scale: $0.scale, role: $0.role) }
    }
    XCTAssertEqual(staged.count, 3, "the whole expanded stack is offered for staging")
    XCTAssertEqual(payload.loras?.map(\.path).first, "/nearline/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(payload.loras?.first?.role, "kroma", "staging must not drop the slot")
  }

  func testNearlineStagingIsNotConsultedWhenNothingWasExpanded() throws {
    let store = try makeStore()
    var called = false
    _ = try expand(#"{"prompt":"x"}"#, store: store) { entries in
      called = true
      return entries
    }
    XCTAssertFalse(called)
  }

  // MARK: I5 — the persisted body carries the ACCEPTED stack

  /// A crash-recovery replay must repeat the stack the job was accepted with.
  /// Replaying the ORIGINAL body would re-resolve the preset against whatever
  /// the store says at replay time, so editing or deleting a preset after
  /// acceptance could change or invalidate a queued render.
  func testPersistedBodyCarriesTheExpandedStackAndReplaysItAfterAPresetEdit() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let original = Data(#"{"prompt":"x","preset":"krea-kira"}"#.utf8)
    let accepted = try WarmServer.expandGeneratePayload(
      try JSONDecoder.snakeCase.decode(GeneratePayload.self, from: original), store: store)
    let persisted = WarmServer.rawBody(original, expandedWith: accepted)

    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: persisted) as? [String: Any])
    XCTAssertEqual((object["loras"] as? [[String: Any]])?.count, 3)
    XCTAssertEqual(object["model"] as? String, "krea2-raw")
    XCTAssertEqual(object["preset"] as? String, "krea-kira", "provenance survives")

    // Now the preset changes underneath the queued job — the replay must be
    // unmoved, because the persisted body carries explicit `loras`.
    _ = try store.upsert(ImagePreset(
      id: "krea-kira", name: "Kira", mediaKind: "image", model: "krea2-raw",
      loras: [LoraReference(filename: "something_else.safetensors", scale: 1.0)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))

    let replayed = try WarmServer.expandGeneratePayload(
      try JSONDecoder.snakeCase.decode(GeneratePayload.self, from: persisted), store: store)
    XCTAssertEqual(replayed.loras?.map(\.path), [
      "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
      "krea2_turbo_distill_r256.safetensors",
      "RealisticSnapshotKrea2.safetensors",
    ], "the replay must repeat the ACCEPTED stack, not re-resolve the edited preset")
    XCTAssertEqual(replayed.presetStackMismatch, true, "and it says the preset has since diverged")
  }

  /// A deleted preset must not invalidate a job that was already accepted.
  func testReplayAfterThePresetIsDeletedStillCarriesTheAcceptedStack() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let original = Data(#"{"prompt":"x","preset":"krea-kira"}"#.utf8)
    let accepted = try WarmServer.expandGeneratePayload(
      try JSONDecoder.snakeCase.decode(GeneratePayload.self, from: original), store: store)
    let persisted = WarmServer.rawBody(original, expandedWith: accepted)
    try store.delete("krea-kira")

    let replayed = try WarmServer.expandGeneratePayload(
      try JSONDecoder.snakeCase.decode(GeneratePayload.self, from: persisted), store: store)
    XCTAssertEqual(replayed.loras?.count, 3)
    XCTAssertEqual(replayed.presetUnresolved, "krea-kira", "and the replay says the preset is gone")
  }

  /// Nothing to merge ⇒ the original bytes, untouched.
  func testUnexpandedBodyIsPersistedVerbatim() throws {
    let store = try makeStore()
    let original = Data(#"{"prompt":"x"}"#.utf8)
    let payload = try WarmServer.expandGeneratePayload(
      try JSONDecoder.snakeCase.decode(GeneratePayload.self, from: original), store: store)
    XCTAssertEqual(WarmServer.rawBody(original, expandedWith: payload), original)
  }

  /// A request that already named its own `loras`/`model` keeps them — the
  /// merge never overwrites what the caller sent.
  func testPersistedBodyNeverOverwritesTheCallersOwnFields() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let original = Data(#"""
      {"prompt":"x","preset":"krea-kira","model":"krea2-raw",
       "loras":[{"path":"only.safetensors","scale":1}]}
      """#.utf8)
    let payload = try WarmServer.expandGeneratePayload(
      try JSONDecoder.snakeCase.decode(GeneratePayload.self, from: original), store: store)
    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: WarmServer.rawBody(original, expandedWith: payload))
        as? [String: Any])
    XCTAssertEqual((object["loras"] as? [[String: Any]])?.count, 1)
    XCTAssertEqual(object["model"] as? String, "krea2-raw")
  }
}

extension JSONDecoder {
  /// The generate routes' decoder.
  static var snakeCase: JSONDecoder {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return d
  }
}
