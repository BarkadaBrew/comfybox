// StylePackFamilyGateTests.swift — comfybox#399, PR #407 review rulings 1–3
//
// Three silences the first cut of the merge left open, closed and pinned:
//
//   1. THE FAMILY GATE. The StylePack pass lives in `runKrea2Generate` and
//      nowhere else, while `runGenerate` dispatches five families. A `style`
//      on chroma/fibo/flux1/flux2 was accepted at the decode, carried on the
//      payload, and then silently skipped — which is precisely what
//      `FamilyRecipeMatrix` exists to prevent for samplers and schedules.
//   2. THE INVALID PRESET. `store.lookup` returns the preset AND its validity
//      flag (WP-E20 / AC-44c); the resolution read `.0` and threw the flag
//      away, so a look could be adopted off a document the engine had already
//      refused to trust.
//   3. PROVENANCE. A styled render and an unstyled one wrote identical
//      metadata, so the file could not say which look its pixels carry.

import XCTest

@testable import ZImage

final class StylePackFamilyGateTests: XCTestCase {

  // MARK: Harness

  private func makeStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-stylegate-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  private func decode(_ json: String, store: PresetStore) throws -> GeneratePayload {
    try WarmServer.decodedGeneratePayload(
      from: Data(json.utf8), store: store, configuration: configuration,
      loraExists: { _ in true })
  }

  // MARK: 1 — the family gate (ruling 1)

  /// The matrix is the authority, and it says exactly one family runs the
  /// pass today. If a second family grows one, this list moves in the same
  /// commit as the flag — which is the point of keeping it in the table.
  func testOnlyKrea2DeclaresTheStylePackPass() {
    XCTAssertEqual(FamilyRecipeMatrix.stylePackFamilyNames(), ["krea2"])
    for family in WarmModelFamily.allCases {
      XCTAssertEqual(
        FamilyRecipeMatrix.capability(for: family).appliesStylePack,
        family == .krea2,
        "\(family.rawValue) disagrees with the dispatch arm that actually applies the pass")
    }
  }

  /// Every family, every registered style: krea2 passes, the other four are a
  /// 400 naming the field, the value, the family AND the family that can.
  func testEveryNonKrea2FamilyRefusesEveryStyle() {
    for family in WarmModelFamily.allCases {
      for name in StylePack.knownNames {
        let error = FamilyRecipeMatrix.validateStyle(name, family: family)
        if family == .krea2 {
          XCTAssertNil(error, "krea2 runs the pass; '\(name)' must not be refused")
          continue
        }
        guard case .unsupportedRecipeField(let field, let value, let fam, let reason)? = error else {
          return XCTFail("\(family.rawValue) + '\(name)': expected a 400, got \(String(describing: error))")
        }
        XCTAssertEqual(field, "style")
        XCTAssertEqual(value, name)
        XCTAssertEqual(fam, family.rawValue)
        XCTAssertTrue(reason.contains("krea2"), "the refusal must name where the look CAN run")
        XCTAssertEqual(WarmServer.errorResponse(for: error!).status, 400)
      }
    }
  }

  /// Absence is untouched on every family — this gate must not change any
  /// request that asked for no look.
  func testNoStyleIsAcceptedOnEveryFamily() {
    for family in WarmModelFamily.allCases {
      XCTAssertNil(FamilyRecipeMatrix.validateStyle(nil, family: family))
      XCTAssertNil(FamilyRecipeMatrix.validateStyle("", family: family))
      XCTAssertNil(FamilyRecipeMatrix.validateStyle("   ", family: family))
    }
  }

  /// `GeneratePayload.styleGate` is the seam `runGenerate` calls, one line
  /// below `validateFamilyRecipe`, on the single path every submission takes
  /// (`/v1/generate`, `/v1/generate/async`, crash-recovery replay, preemption
  /// and the ComfyUI bridge all reach the render via `enqueueGenerate` →
  /// `runGenerate`). It reads the payload, so it also catches the boolean
  /// alias on a payload built in-process that never went through the decode.
  func testStyleGateReadsBothTheResolvedNameAndTheLegacyAlias() throws {
    let store = try makeStore()
    let styled = try decode(#"{"prompt":"x","style":"trix-bw"}"#, store: store)
    XCTAssertNil(GeneratePayload.styleGate(styled, family: .krea2))
    XCTAssertNotNil(GeneratePayload.styleGate(styled, family: .chroma))
    XCTAssertNotNil(GeneratePayload.styleGate(styled, family: .flux1))
    XCTAssertNotNil(GeneratePayload.styleGate(styled, family: .flux2))
    XCTAssertNotNil(GeneratePayload.styleGate(styled, family: .fibo))

    // The bridge shape: `phone_look` alone, never collapsed onto `style`.
    var aliasOnly = try decode(#"{"prompt":"x"}"#, store: store)
    aliasOnly.phoneLook = true
    aliasOnly.style = nil
    XCTAssertNil(GeneratePayload.styleGate(aliasOnly, family: .krea2))
    guard case .unsupportedRecipeField(_, let value, _, _)? =
      GeneratePayload.styleGate(aliasOnly, family: .fibo)
    else { return XCTFail("phone_look alone must be gated too") }
    XCTAssertEqual(value, StylePack.phone.rawValue)

    let plain = try decode(#"{"prompt":"x"}"#, store: store)
    for family in WarmModelFamily.allCases {
      XCTAssertNil(GeneratePayload.styleGate(plain, family: family))
    }
  }

  // MARK: 2 — an INVALID preset contributes no look (ruling 2)

  /// A store loaded from a `presets.json` whose entry DECODES but fails
  /// ``PresetStore/validate`` — `revalidate()` runs at load and flags it
  /// (WP-E20 / AC-44c). This is how an invalid preset really arises: `upsert`
  /// refuses one, so it can only come off disk.
  private func storeWithInvalidStyledPreset(style: String) throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-stylegate-invalid-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("presets.json")
    // `steps: -1` is refused by `PresetStore.validate`. (One word, so it means
    // the same thing whichever spelling the file uses: `presets.json` is
    // camelCase — `decodeEntries` uses a PLAIN JSONDecoder — while the
    // `POST /v1/presets` route decodes `.convertFromSnakeCase`.)
    let json = """
    [{"id":"broken","name":"Broken","mediaKind":"image","model":"krea2-raw",
      "loras":[],"checkpointFamily":"raw-accel","steps":-1,"style":"\(style)"}]
    """
    try Data(json.utf8).write(to: path)
    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertNotNil(store.lookup("broken").invalidReason, "fixture precondition: flagged invalid")
    XCTAssertEqual(store.lookup("broken").preset?.style, style, "fixture precondition: it declares a look")
    return store
  }

  /// `PresetStore.lookup` returns `(preset, invalidReason)`. A preset flagged
  /// invalid at load contributes nothing — not its stack, not its model, and
  /// not its look: the document itself is what the engine refused to trust.
  ///
  /// But it must not be DROPPED either (review r2, ruling 1). `PresetLoRAStack`'s
  /// "an unexpandable preset is a label, never a 400" rule is about the
  /// RECIPE — an unknown preset id was harmless provenance for the daemon's
  /// whole life. A `style` is the caller asking for a visible change to the
  /// pixels, so when the ONLY look on the request came off a preset the
  /// engine will not use, it is a 400 naming the preset, the look and why.
  func testAStyleDeclaredOnlyByAnInvalidPresetIsA400() throws {
    let store = try storeWithInvalidStyledPreset(style: "trix-bw")
    XCTAssertThrowsError(try decode(#"{"prompt":"x","preset":"broken"}"#, store: store)) { error in
      guard case WarmServerError.styleFromUnusablePreset(
        let preset, let style, let code, let reason) = error
      else { return XCTFail("expected .styleFromUnusablePreset, got \(error)") }
      XCTAssertEqual(preset, "broken")
      XCTAssertEqual(style, "trix-bw")
      XCTAssertEqual(code, "invalid_preset")
      XCTAssertFalse(reason.isEmpty, "the refusal must carry the store's own reason")
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      let body = String(data: response.body, encoding: .utf8) ?? ""
      XCTAssertTrue(body.contains("trix-bw"), "the 400 must name the look")
      XCTAssertTrue(body.contains("broken"), "the 400 must name the preset")
      XCTAssertTrue(body.contains("invalid_preset"), "the 400 must name the reason")
    }
  }

  /// The scope of that rule: an invalid preset that declares NO look is
  /// untouched — still a label, still never a 400, still reported through
  /// `preset_unresolved_reason`. The refusal is about a dropped LOOK, not
  /// about presets.
  func testAnInvalidPresetWithNoStyleKeepsTheLabelBehaviour() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-stylegate-plain-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("presets.json")
    let json = """
    [{"id":"broken","name":"Broken","mediaKind":"image","model":"krea2-raw",
      "loras":[],"checkpointFamily":"raw-accel","steps":-1}]
    """
    try Data(json.utf8).write(to: path)
    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertNotNil(store.lookup("broken").invalidReason, "precondition")
    XCTAssertNil(store.lookup("broken").preset?.style, "precondition: no look declared")

    let payload = try decode(#"{"prompt":"x","preset":"broken"}"#, store: store)
    XCTAssertEqual(payload.presetUnresolvedReason, "invalid_preset")
    XCTAssertNil(payload.style)
  }

  /// The same preset shape, VALID: the look is adopted — so the test above
  /// pins the validity flag and not an unrelated failure to resolve.
  func testTheSamePresetLendsItsStyleWhenValid() throws {
    let store = try makeStore()
    var preset = ImagePreset(
      id: "broken", name: "Broken", mediaKind: "image", model: "krea2-raw", loras: [],
      checkpointFamily: "raw-accel")
    preset.style = "trix-bw"
    try store.upsert(preset)
    XCTAssertNil(store.lookup("broken").invalidReason, "precondition")

    let payload = try decode(#"{"prompt":"x","preset":"broken"}"#, store: store)
    XCTAssertEqual(payload.style, "trix-bw")
  }

  /// An explicit request style still wins over an invalid preset — the
  /// request never depends on the store being healthy.
  func testRequestStyleSurvivesAnInvalidPreset() throws {
    let store = try storeWithInvalidStyledPreset(style: "trix-bw")
    let payload = try decode(#"{"prompt":"x","preset":"broken","style":"hp5-soft"}"#, store: store)
    XCTAssertEqual(payload.style, "hp5-soft")
  }

  /// The invalid preset's own `phone_look` alias takes the same route — the
  /// shim must not be a way around either the validity flag or the refusal.
  func testInvalidPresetPhoneLookAliasIsRefusedToo() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-stylegate-invalid-alias-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("presets.json")
    let json = """
    [{"id":"broken","name":"Broken","mediaKind":"image","model":"krea2-raw",
      "loras":[],"checkpointFamily":"raw-accel","steps":-1,"phoneLook":true}]
    """
    try Data(json.utf8).write(to: path)
    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertNotNil(store.lookup("broken").invalidReason, "precondition")
    XCTAssertEqual(store.lookup("broken").preset?.phoneLook, true, "precondition")

    XCTAssertThrowsError(try decode(#"{"prompt":"x","preset":"broken"}"#, store: store)) { error in
      guard case WarmServerError.styleFromUnusablePreset(_, let style, let code, _) = error
      else { return XCTFail("expected .styleFromUnusablePreset, got \(error)") }
      XCTAssertEqual(style, StylePack.phone.rawValue, "the alias resolves before it is refused")
      XCTAssertEqual(code, "invalid_preset")
    }
  }

  // MARK: 3 — provenance (ruling 3)

  /// `applied.style` names the look the save path applied. Two renders that
  /// differ ONLY by style produce different records; an unstyled render's
  /// record is byte-identical to the pre-#399 one (the key is absent, not
  /// null).
  func testAppliedRecordCarriesTheStyleAndOnlyWhenOneRan() throws {
    let trace = RenderRecipeFixture.trace(steps: 8)
    var styledInputs = RenderRecipeFixture.inputs(trace: trace)
    styledInputs.style = "hp5-soft"
    let styled = RenderRecipe.krea2(styledInputs)
    let plain = RenderRecipe.krea2(RenderRecipeFixture.inputs(trace: trace))

    XCTAssertEqual(styled.style, "hp5-soft")
    XCTAssertNil(plain.style)
    XCTAssertNotEqual(styled, plain, "the record must distinguish the two renders")

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.sortedKeys]
    let styledJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: try encoder.encode(styled)) as? [String: Any])
    let plainJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: try encoder.encode(plain)) as? [String: Any])
    XCTAssertEqual(styledJSON["style"] as? String, "hp5-soft")
    XCTAssertNil(plainJSON["style"], "an unstyled record must not gain a null key")
  }

  /// The PNG sidecar, which is what a gallery reads. Two renders differing
  /// only by style must differ in the embedded parameter JSON — and the
  /// top-level `style` is written even when the provenance record itself was
  /// REFUSED (`"applied": null`), because the pixels still carry the look.
  func testPNGMetadataDiffersOnlyByStyleAndSurvivesARefusedRecord() throws {
    let record = RenderRecipeFixture.recipe(steps: 8)

    func metadata(style: String?, slot: AppliedRecordSlot) -> QwenImageIO.ImageMetadata {
      QwenImageIO.ImageMetadata.generation(
        prompt: "a portrait", seed: 44821, steps: 8, guidance: 1.0,
        width: 1024, height: 1024, model: "krea2-raw",
        appliedSlot: slot, style: style)
    }

    let plain = metadata(style: nil, slot: AppliedRecordSlot(record: record))
    let styled = metadata(style: "trix-bw", slot: AppliedRecordSlot(record: record))
    XCTAssertNotEqual(
      styled.parametersJSON, plain.parametersJSON,
      "two renders differing only by style wrote identical metadata")
    XCTAssertTrue(try XCTUnwrap(styled.parametersJSON).contains("\"style\":\"trix-bw\""))
    XCTAssertFalse(try XCTUnwrap(plain.parametersJSON).contains("\"style\""))

    // Record refused (C4's `"applied": null`) — the look must still be recorded.
    let refused = metadata(style: "trix-bw", slot: AppliedRecordSlot(record: nil))
    let json = try XCTUnwrap(refused.parametersJSON)
    XCTAssertTrue(json.contains("\"applied\":null"))
    XCTAssertTrue(json.contains("\"style\":\"trix-bw\""))
  }

  /// Unstyled metadata carries NO `style` key, and its top-level key set is
  /// exactly the pre-#399 one.
  ///
  /// (Review r2, ruling 4: this test used to compare `generation(…)` to
  /// `generation(…, style: nil)` and call the result "byte-identical to the
  /// pre-#399 call" — a comparison of the new API with itself, which pins the
  /// default-argument behaviour and nothing more. The checked-in key list
  /// below is the actual fixture: `style` is the only key #399 can add, so a
  /// set that matches this literal is the set main produced for these
  /// inputs.)
  func testUnstyledMetadataCarriesNoStyleKeyAndThePre399KeySet() throws {
    let record = RenderRecipeFixture.recipe(steps: 8)
    let unstyled = QwenImageIO.ImageMetadata.generation(
      prompt: "a portrait", seed: 44821, steps: 8, guidance: 1.0,
      width: 1024, height: 1024, model: "krea2-raw",
      appliedSlot: AppliedRecordSlot(record: record))
    let json = try XCTUnwrap(unstyled.parametersJSON)
    let object = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    XCTAssertEqual(
      Set(object.keys),
      ["applied", "guidance", "height", "model", "prompt", "seed", "steps", "width"],
      "an unstyled render's metadata keys must be the pre-#399 set")
    XCTAssertNil(object["style"])

    // The default argument is the no-look path — deleting `style: nil` from a
    // call site must not change what it writes.
    let explicitNil = QwenImageIO.ImageMetadata.generation(
      prompt: "a portrait", seed: 44821, steps: 8, guidance: 1.0,
      width: 1024, height: 1024, model: "krea2-raw",
      appliedSlot: AppliedRecordSlot(record: record), style: nil)
    XCTAssertEqual(unstyled.parametersJSON, explicitNil.parametersJSON)
  }

  // MARK: 4 — one shared resolution, and the bridge's batch loop (r2, 2 & 3)

  /// The gate and the save-path apply must read the SAME resolution. They did
  /// not: `styleGate` resolved the `phone_look` alias while the apply site
  /// read `style` alone, so a payload carrying `phone_look: true` and no
  /// `style` was refused on chroma/fibo/flux and rendered UNSTYLED on krea2.
  /// `effectiveStyleName` is now the one answer both ask for.
  func testGateAndApplyReadTheSameResolution() throws {
    let store = try makeStore()
    var aliasOnly = try decode(#"{"prompt":"x"}"#, store: store)
    aliasOnly.phoneLook = true
    aliasOnly.style = nil

    // What the save path resolves — the same call `runKrea2Generate` makes.
    XCTAssertEqual(StylePack.resolved(aliasOnly.effectiveStyleName), .phone)
    // What the gate resolves.
    XCTAssertNil(GeneratePayload.styleGate(aliasOnly, family: .krea2))
    XCTAssertNotNil(GeneratePayload.styleGate(aliasOnly, family: .fibo))

    // …and every shape agrees between the two.
    for (styleField, alias) in [
      (String?.none, Bool?.none), (nil, true), (nil, false),
      ("trix-bw", nil), ("trix-bw", true), ("hp5-soft", false),
    ] as [(String?, Bool?)] {
      var p = try decode(#"{"prompt":"x"}"#, store: store)
      p.style = styleField
      p.phoneLook = alias
      let applied = StylePack.resolved(p.effectiveStyleName)?.rawValue
      let gated = GeneratePayload.styleGate(p, family: .fibo) != nil
      XCTAssertEqual(
        applied != nil, gated,
        "gate and apply disagree for style=\(styleField as Any) phone_look=\(alias as Any)")
    }
  }

  /// The ComfyUI bridge's batch loop derives a per-item payload. It used to
  /// REBUILD `GeneratePayload` field by field and silently dropped every
  /// field the rebuild did not mention — `style`/`phone_look` were the third
  /// instance of that bug in this feature alone. `batchItem` copies and
  /// overrides ONE field, so there is nothing left to forget.
  func testBridgeBatchItemCarriesEveryFieldAndOverridesOnlyTheSeed() throws {
    let store = try makeStore()
    var preset = ImagePreset(
      id: "krea-kira", name: "Kira", mediaKind: "image", model: "krea2-raw", steps: 12,
      loras: [LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel")],
      checkpointFamily: "raw-accel")
    preset.style = "hp5-soft"
    try store.upsert(preset)
    var payload = try decode(#"""
      {"prompt":"a portrait","negative_prompt":"blurry","preset":"krea-kira",
       "width":1024,"height":1536,"seed":1,"content_mode":"apple","source":"comfyui",
       "scheduler":"res_2s","sigma_schedule":"beta57","shift":1.15}
      """#, store: store)
    payload.phoneLook = true

    let item = GeneratePayload.batchItem(payload, seed: 99)

    XCTAssertEqual(item.seed, 99, "the seed is the one thing that changes")
    XCTAssertEqual(item.style, "hp5-soft", "the resolved look must survive the batch loop")
    XCTAssertEqual(item.phoneLook, true)
    XCTAssertEqual(item.effectiveStyleName, payload.effectiveStyleName)
    // …and the rest of the accepted request, which the old rebuild also lost.
    XCTAssertEqual(item.prompt, payload.prompt)
    XCTAssertEqual(item.negativePrompt, payload.negativePrompt)
    XCTAssertEqual(item.preset, payload.preset)
    XCTAssertEqual(item.model, payload.model)
    XCTAssertEqual(item.steps, payload.steps)
    XCTAssertEqual(item.guidance, payload.guidance)
    XCTAssertEqual(item.shift, payload.shift)
    XCTAssertEqual(item.scheduler, payload.scheduler)
    XCTAssertEqual(item.sigmaSchedule, payload.sigmaSchedule)
    XCTAssertEqual(item.contentMode, payload.contentMode)
    XCTAssertEqual(item.source, payload.source)
    XCTAssertEqual(item.width, payload.width)
    XCTAssertEqual(item.height, payload.height)
    XCTAssertEqual(item.loras?.map(\.path), payload.loras?.map(\.path))
    XCTAssertEqual(item.presetStackApplied, payload.presetStackApplied)
  }
}
