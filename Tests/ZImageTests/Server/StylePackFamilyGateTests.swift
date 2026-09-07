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
  /// not its look. Reading only `.0` (the first cut of this merge) adopted a
  /// look off a document the engine had already refused to trust.
  func testInvalidPresetDoesNotContributeItsStyle() throws {
    let store = try storeWithInvalidStyledPreset(style: "trix-bw")
    let payload = try decode(#"{"prompt":"x","preset":"broken"}"#, store: store)
    XCTAssertEqual(payload.presetUnresolvedReason, "invalid_preset", "precondition")
    XCTAssertNil(payload.style, "an invalid preset must not lend its look either")
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

  /// The invalid preset's own `phone_look` alias is refused for the same
  /// reason — the shim must not be a way around the validity flag.
  func testInvalidPresetPhoneLookAliasIsAlsoIgnored() throws {
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

    let payload = try decode(#"{"prompt":"x","preset":"broken"}"#, store: store)
    XCTAssertNil(payload.style)
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

  /// …and the no-look case is unchanged end to end: passing `style: nil`
  /// produces the exact bytes the pre-#399 call site produced.
  func testUnstyledMetadataIsByteIdenticalToThePre399Call() {
    let record = RenderRecipeFixture.recipe(steps: 8)
    let before = QwenImageIO.ImageMetadata.generation(
      prompt: "a portrait", seed: 44821, steps: 8, guidance: 1.0,
      width: 1024, height: 1024, model: "krea2-raw",
      appliedSlot: AppliedRecordSlot(record: record))
    let after = QwenImageIO.ImageMetadata.generation(
      prompt: "a portrait", seed: 44821, steps: 8, guidance: 1.0,
      width: 1024, height: 1024, model: "krea2-raw",
      appliedSlot: AppliedRecordSlot(record: record), style: nil)
    XCTAssertEqual(before.parametersJSON, after.parametersJSON)
  }
}
