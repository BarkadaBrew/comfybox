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

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  /// THE route function.
  ///
  /// Round 2 (I6): `WarmServer.decodedGeneratePayload(from:store:configuration:…)`
  /// is the whole decode the `/v1/generate` and `/v1/generate/async` handlers
  /// run — parse, init-image, output-path containment, recipe names AND the
  /// #286 preset expansion, in one function. The instance method of the same
  /// name is a one-line forward to it, so there is no separate "expansion call
  /// at the decode site" that could be deleted while the decode survived: any
  /// such deletion fails these tests.
  ///
  /// `loraExists` defaults to "everything resolves" so a test does not need
  /// files on disk; the missing-LoRA tests below override it.
  private func expand(
    _ json: String, store: PresetStore,
    stageNearline: ([LoRAEntry]) -> [LoRAEntry] = { $0 },
    loraExists: @escaping (LoRAEntry) -> Bool = { _ in true }
  ) throws -> GeneratePayload {
    try WarmServer.decodedGeneratePayload(
      from: Data(json.utf8), store: store, configuration: configuration,
      stageNearline: stageNearline, loraExists: loraExists)
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

    // Todd 2026-09-04: kroma has no special semantics — `decide` applies
    // `loras[]` exactly as `seedKreaKira`'s `store.upsert` migrated it:
    // the two declared entries, THEN the structured kroma folded on at the
    // end (`ImagePreset.migratingKromaDeprecation` appends, never prepends).
    let loras = try XCTUnwrap(payload.loras)
    XCTAssertEqual(loras.map(\.path), [
      "krea2_turbo_distill_r256.safetensors",
      "RealisticSnapshotKrea2.safetensors",
      "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
    ])
    XCTAssertEqual(loras.map { $0.scale ?? -1 }, [0.6, 0.4, 0.6])
    XCTAssertEqual(loras[0].role, "accel")
    XCTAssertEqual(loras[2].role, "kroma")
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
    // Todd 2026-09-04: `resolve()` already reflects the migrated stack (the
    // structured kroma is folded into `loras[]` on save) — no separate
    // prepend at the route, so the counts now agree exactly.
    XCTAssertEqual(applied.count, resolved.loras.count)
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
    // Todd 2026-09-04: order must match the migrated declared order —
    // turbo, snapshot, THEN kroma appended (no prepend) — `isSameStack`
    // compares positionally.
    let payload = try expand(#"""
      {"prompt":"x","preset":"krea-kira","loras":[
        {"path":"/Volumes/Bolt/loras/krea2_turbo_distill_r256.safetensors","scale":0.6},
        {"path":"/Volumes/Bolt/loras/RealisticSnapshotKrea2.safetensors","scale":0.4},
        {"path":"/Volumes/Bolt/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors","scale":0.6}]}
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
    // Todd 2026-09-04: kroma is appended (migrated), not prepended, so it is
    // the LAST entry of the declared+migrated stack, not the first.
    XCTAssertEqual(payload.loras?.map(\.path).last, "/nearline/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(payload.loras?.last?.role, "kroma", "staging must not drop the slot")
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
    let accepted = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration, loraExists: { _ in true })
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

    let replayed = try WarmServer.decodedGeneratePayload(
      from: persisted, store: store, configuration: configuration, loraExists: { _ in true })
    XCTAssertEqual(replayed.loras?.map(\.path), [
      "krea2_turbo_distill_r256.safetensors",
      "RealisticSnapshotKrea2.safetensors",
      "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
    ], "the replay must repeat the ACCEPTED stack, not re-resolve the edited preset")
    XCTAssertEqual(replayed.presetStackMismatch, true, "and it says the preset has since diverged")
  }

  /// A deleted preset must not invalidate a job that was already accepted.
  func testReplayAfterThePresetIsDeletedStillCarriesTheAcceptedStack() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let original = Data(#"{"prompt":"x","preset":"krea-kira"}"#.utf8)
    let accepted = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration, loraExists: { _ in true })
    let persisted = WarmServer.rawBody(original, expandedWith: accepted)
    try store.delete("krea-kira")

    let replayed = try WarmServer.decodedGeneratePayload(
      from: persisted, store: store, configuration: configuration, loraExists: { _ in true })
    XCTAssertEqual(replayed.loras?.count, 3)
    XCTAssertEqual(replayed.presetUnresolved, "krea-kira", "and the replay says the preset is gone")
  }

  // MARK: Round 2, finding 1 — the engine/provider gate at the route

  /// The seeded default. Before this gate, `POST /v1/generate {"preset":
  /// "schnell-hq"}` expanded `model: "schnell"` into a `poolLoad` of a model
  /// this engine cannot serve — a failed render where the label used to be
  /// harmless.
  func testSeededSchnellHQPresetRendersAsALabel() throws {
    let store = try makeStore()
    _ = try store.upsert(try XCTUnwrap(PresetStore.defaultPresets.first { $0.id == "schnell-hq" }))

    let payload = try expand(#"{"prompt":"x","preset":"schnell-hq"}"#, store: store)
    XCTAssertNil(payload.model, "nothing must reach poolLoad")
    XCTAssertNil(payload.loras)
    XCTAssertNil(payload.steps)
    XCTAssertEqual(payload.presetUnresolved, "schnell-hq")
    XCTAssertEqual(payload.presetUnresolvedReason, "engine:mflux")
  }

  /// A store seeded with BOTH defaults: the engine's own preset still expands.
  func testSeededZImageChatStillExpandsFromTheSameStore() throws {
    let store = try makeStore()
    for preset in PresetStore.defaultPresets { _ = try store.upsert(preset) }

    let payload = try expand(#"{"prompt":"x","preset":"zimage-chat"}"#, store: store)
    XCTAssertEqual(payload.model, "z-image-turbo")
    XCTAssertEqual(payload.steps, 8)
    // Round 2, finding 6: an EMPTY preset stack is a declaration — it clears
    // the resident adapters. `zimage-chat` is exactly that shape.
    XCTAssertEqual(payload.loras?.isEmpty, true)
    XCTAssertNil(payload.presetUnresolved)
  }

  // MARK: Round 2, finding 2 — a preset with no model of its own

  func testDesktopPresetWithOnlyCustomModelPathRendersAsALabel() throws {
    let store = try makeStore()
    _ = try store.upsert(ImagePreset(
      id: "788B45BC", name: "Desktop preset",
      customModelPath: "/Users/todd/LocalModels/krea2-raw",
      loras: [LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel")],
      kroma: KromaPolicy(strength: 0)))

    let payload = try expand(#"{"prompt":"x","preset":"788B45BC"}"#, store: store)
    XCTAssertNil(payload.loras, "its stack must not land on whatever base is resident")
    XCTAssertEqual(payload.presetUnresolvedReason, "no_model")
  }

  /// Round 3: naming a base on the request is not enough. The 26 desktop-saved
  /// presets declare neither `model` nor `checkpoint_family`, so the engine
  /// cannot tell whether their stack belongs on the requested base — and
  /// "cannot tell" is not "yes". They need one of the two fields to expand
  /// (desktop follow-up: save `checkpoint_family` at creation).
  func testDesktopPresetStaysALabelEvenWhenTheRequestNamesTheBase() throws {
    let store = try makeStore()
    _ = try store.upsert(ImagePreset(
      id: "788B45BC", name: "Desktop preset",
      customModelPath: "/Users/todd/LocalModels/krea2-raw",
      loras: [LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel")],
      kroma: KromaPolicy(strength: 0)))

    let payload = try expand(
      #"{"prompt":"x","preset":"788B45BC","model":"krea2-raw"}"#, store: store)
    XCTAssertNil(payload.loras, "unknowable family — its stack must not ride the request's base")
    XCTAssertEqual(payload.model, "krea2-raw", "the request's own base is untouched")
    XCTAssertEqual(payload.presetUnresolvedReason, "no_model")
  }

  /// The same preset with a `checkpoint_family` declared DOES expand — which is
  /// what the desktop follow-up needs to write.
  func testDesktopPresetExpandsOnceItDeclaresACheckpointFamily() throws {
    let store = try makeStore()
    _ = try store.upsert(ImagePreset(
      id: "788B45BC", name: "Desktop preset",
      customModelPath: "/Users/todd/LocalModels/krea2-raw",
      loras: [LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel")],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))

    let payload = try expand(
      #"{"prompt":"x","preset":"788B45BC","model":"krea2-raw"}"#, store: store)
    XCTAssertEqual(payload.loras?.map(\.path), ["krea2_turbo_distill_r256.safetensors"])
    XCTAssertEqual(payload.model, "krea2-raw", "the request's own base stands")
    XCTAssertNil(payload.presetUnresolved)
  }

  // MARK: Round 2, finding 3 — a preset naming a LoRA that is not on disk

  /// This used to become a 400 at DEQUEUE (`LoRAEntry.makeConfiguration`
  /// throws on an unresolvable bare filename), turning a provenance label into
  /// a failed render for every caller of that preset. Resolved at decode
  /// instead, while the request can still fall back.
  func testPresetNamingAMissingLoRARendersAsALabel() throws {
    let store = try makeStore()
    try seedKreaKira(store)

    let payload = try expand(#"{"prompt":"x","preset":"krea-kira"}"#, store: store) { entry in
      entry.path != "RealisticSnapshotKrea2.safetensors"
    }
    XCTAssertNil(payload.loras)
    XCTAssertNil(payload.model)
    XCTAssertEqual(payload.presetUnresolved, "krea-kira")
    XCTAssertEqual(payload.presetUnresolvedReason, "missing_lora:RealisticSnapshotKrea2.safetensors")
  }

  /// The check runs AFTER nearline staging — an adapter that is merely
  /// archived is staged, not refused.
  func testNearlineStagedLoRAIsNotReportedMissing() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    let payload = try expand(
      #"{"prompt":"x","preset":"krea-kira"}"#, store: store,
      stageNearline: { entries in
        entries.map { LoRAEntry(path: "/staged/\($0.path)", scale: $0.scale, role: $0.role) }
      },
      loraExists: { $0.path.hasPrefix("/staged/") })
    XCTAssertNil(payload.presetUnresolved)
    XCTAssertEqual(payload.loras?.count, 3)
  }

  /// An EXPLICIT `loras` entry that does not resolve keeps its long-standing
  /// 400 at dequeue — that contract is not this ticket's to change.
  func testExplicitLorasAreNotSubjectToTheMissingCheck() throws {
    let store = try makeStore()
    let payload = try expand(
      #"{"prompt":"x","loras":[{"path":"nope.safetensors","scale":1}]}"#,
      store: store, loraExists: { _ in false })
    XCTAssertEqual(payload.loras?.map(\.path), ["nope.safetensors"])
    XCTAssertNil(payload.presetUnresolved)
  }

  /// Round 3, minor 2: the missing-LoRA rule must hold for EVERY source form.
  /// `LoRAEntry.makeConfiguration()` only searches for a bare filename — an
  /// absolute, `~` or relative path used to be taken at its word, so the rule
  /// held for one form and not the others. Driven through the PRODUCTION
  /// `loraExists` (`WarmServer.loRASourceExists`), not a stub.
  func testPresetNamingAnAbsolutePathThatIsNotOnDiskRendersAsALabel() throws {
    let store = try makeStore()
    let absent = "/Volumes/definitely-not-mounted-\(UUID().uuidString)/ghost.safetensors"
    _ = try store.upsert(ImagePreset(
      id: "abs-missing", name: "x", mediaKind: "image", model: "krea2-raw",
      loras: [LoraReference(filename: absent, scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))

    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","preset":"abs-missing"}"#.utf8),
      store: store, configuration: configuration)
    XCTAssertNil(payload.loras)
    XCTAssertNil(payload.model, "nothing must reach poolLoad either")
    XCTAssertEqual(payload.presetUnresolvedReason, "missing_lora:ghost.safetensors")
  }

  /// A `~`-prefixed source is expanded before the stat, and a real file passes.
  func testPresetNamingATildePathThatExistsExpands() throws {
    let store = try makeStore()
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-lora-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    let file = dir.appendingPathComponent("present.safetensors")
    try Data("weights".utf8).write(to: file)

    _ = try store.upsert(ImagePreset(
      id: "abs-present", name: "x", mediaKind: "image", model: "krea2-raw",
      loras: [LoraReference(filename: file.path, scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))

    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","preset":"abs-present"}"#.utf8),
      store: store, configuration: configuration)
    XCTAssertEqual(payload.loras?.map(\.path), [file.path])
    XCTAssertNil(payload.presetUnresolved)
  }

  /// The production check itself, on each source form.
  func testProductionLoRAExistenceCheckCoversEverySourceForm() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-lora-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    let file = dir.appendingPathComponent("present.safetensors")
    try Data("weights".utf8).write(to: file)

    XCTAssertTrue(WarmServer.loRASourceExists(LoRAEntry(path: file.path, scale: 1)))
    XCTAssertFalse(WarmServer.loRASourceExists(
      LoRAEntry(path: dir.appendingPathComponent("absent.safetensors").path, scale: 1)))
    XCTAssertFalse(WarmServer.loRASourceExists(
      LoRAEntry(path: "~/definitely-absent-\(UUID().uuidString).safetensors", scale: 1)))
    XCTAssertFalse(WarmServer.loRASourceExists(
      LoRAEntry(path: "nobody-has-this-\(UUID().uuidString).safetensors", scale: 1)))
    // A HuggingFace reference is fetched at load time — nothing to stat.
    XCTAssertTrue(WarmServer.loRASourceExists(LoRAEntry(path: "org/some-lora-repo", scale: 1)))
  }

  // MARK: I6 — the decode's other halves still run

  /// The expansion lives INSIDE the route's decode, so the decode's own
  /// validations must still fire around it.
  func testOutputPathContainmentStillAppliesToAPresetRequest() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    XCTAssertThrowsError(
      try expand(
        #"{"prompt":"x","preset":"krea-kira","output_path":"/etc/passwd.png"}"#, store: store))
  }

  func testUnknownSamplerStillFailsBeforeExpansion() throws {
    let store = try makeStore()
    try seedKreaKira(store)
    XCTAssertThrowsError(
      try expand(#"{"prompt":"x","preset":"krea-kira","scheduler":"not-a-sampler"}"#, store: store))
  }

  /// Nothing to merge ⇒ the original bytes, untouched.
  func testUnexpandedBodyIsPersistedVerbatim() throws {
    let store = try makeStore()
    let original = Data(#"{"prompt":"x"}"#.utf8)
    let payload = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration, loraExists: { _ in true })
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
    let payload = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration, loraExists: { _ in true })
    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: WarmServer.rawBody(original, expandedWith: payload))
        as? [String: Any])
    XCTAssertEqual((object["loras"] as? [[String: Any]])?.count, 1)
    XCTAssertEqual(object["model"] as? String, "krea2-raw")
  }
}
