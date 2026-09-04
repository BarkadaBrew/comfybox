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
/// Review round 1 added two rules that shape these tests:
///
/// - a preset is expanded as a WHOLE — its `model` travels with its `loras`,
///   because adapters applied to the wrong base render wrong and still look
///   successful (and an explicit `model` that contradicts the preset is a 409);
/// - a preset the engine cannot expand is a LABEL, exactly as before, and says
///   so through `preset_unresolved`. Never a 400: an unknown preset id was
///   harmless provenance for the daemon's whole life, and the daemon contract
///   is production.
///
/// Production shapes throughout: the presets are the live
/// `~/.comfybox/presets.json` entries this was reported against.
final class PresetLoRAStackTests: XCTestCase {

  // MARK: Fixtures — the live presets, as `/v1/presets/resolve` returns them

  /// The preset in the bug report: krea2 raw-stock, kroma declared OFF, five
  /// content LoRAs.
  private func kreaKiraAvocado(steps: Int? = nil, guidance: Double? = nil) -> PresetLoRAStack.Lookup {
    lookup(ImagePreset(
      id: "krea-kira-avocado", name: "Kira Avocado", mediaKind: "image",
      model: "krea2-raw", steps: steps, guidance: guidance,
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
  private func kreaKira() -> PresetLoRAStack.Lookup {
    lookup(ImagePreset(
      id: "krea-kira", name: "Kira", mediaKind: "image",
      model: "krea2-raw",
      loras: [
        LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel"),
        LoraReference(filename: "RealisticSnapshotKrea2.safetensors", scale: 0.4),
      ],
      checkpointFamily: "raw-accel",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")))
  }

  private func lookup(_ preset: ImagePreset) -> PresetLoRAStack.Lookup {
    .resolved(ResolvedPreset(preset: preset), declared: preset)
  }

  /// The `unresolved` reason code, or nil when the preset expanded.
  private func reason(_ decision: PresetLoRAStack) -> String? {
    guard case .apply(let e) = decision else { return nil }
    return e.unresolved?.code
  }

  private func expansion(_ decision: PresetLoRAStack, _ file: StaticString = #filePath, _ line: UInt = #line)
    throws -> PresetExpansion
  {
    guard case .apply(let e) = decision else {
      XCTFail("expected .apply, got \(decision)", file: file, line: line)
      throw XCTSkip("not .apply")
    }
    return e
  }

  // MARK: The bug: a named preset must supply the stack

  func testNamedPresetSuppliesItsResolvedStack() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira-avocado", lookup: kreaKiraAvocado(), requestLoras: nil))

    XCTAssertEqual(e.presetId, "krea-kira-avocado")
    XCTAssertNil(e.unresolved)
    XCTAssertEqual(
      e.loras?.map(\.filename),
      ["snofs_krea_v1_3D.safetensors", "Girly_Tiana.safetensors", "LARP_v0-5.safetensors",
       "snofs_photoSlider_000000200.safetensors", "Krea2_TextFusion_Refusal_Reduction.safetensors"])
    XCTAssertEqual(e.loras?.map(\.scale), [0.8, 0.6, 1.5, 1.25, 1.0])
  }

  /// The render must NEVER silently keep the resident stack when a preset was
  /// named without saying so — that is the whole defect (0 LoRAs after a
  /// restart, 2 stale ones before it, `success: true` for both). Either the
  /// stack is expanded, or `unresolved` names the preset.
  func testNamedPresetEitherExpandsOrSaysWhyNot() throws {
    let lookups: [PresetLoRAStack.Lookup] = [
      kreaKiraAvocado(), kreaKira(), .notFound, .invalid(reason: "missing kroma"),
    ]
    for lookup in lookups {
      let e = try expansion(PresetLoRAStack.decide(
        presetId: "krea-kira-avocado", lookup: lookup, requestLoras: nil))
      XCTAssertTrue(
        e.loras != nil || e.unresolved != nil,
        "a named preset must either supply a stack or report why it could not (lookup: \(lookup))")
    }
  }

  // MARK: C1 — the preset's model travels with its LoRAs

  /// Round 1's critical finding: expanding only the adapters lets a preset's
  /// LoRAs be applied to whatever base is active, which renders wrong AND
  /// reports success.
  func testPresetModelIsAdoptedWithTheStack() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira", lookup: kreaKira(), requestLoras: nil))
    XCTAssertEqual(e.model, "krea2-raw")
    XCTAssertEqual(e.loras?.count, 3)
  }

  /// An explicit `model` that agrees with the preset is not a conflict, and the
  /// request's spelling is left alone — an alias and the directory it names are
  /// the same model.
  func testMatchingExplicitModelIsNotAConflict() throws {
    let normalize: (String) -> String = { $0 == "krea2-raw" ? "/models/krea2-raw" : $0 }
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira", lookup: kreaKira(), requestLoras: nil,
      requestModel: "/models/krea2-raw", normalizeModelSpec: normalize))
    XCTAssertNil(e.model, "the request already named the base; nothing to adopt")
    XCTAssertEqual(e.loras?.count, 3)
  }

  func testContradictingExplicitModelIsAConflict() {
    let decision = PresetLoRAStack.decide(
      presetId: "krea-kira", lookup: kreaKira(), requestLoras: nil,
      requestModel: "z-image-turbo")
    XCTAssertEqual(
      decision,
      .modelConflict(preset: "krea-kira", presetModel: "krea2-raw", requestModel: "z-image-turbo"))
  }

  // MARK: Round 2, finding 2 — a preset with no model of its own

  /// A preset that names no model must NOT hand its adapters to whatever base
  /// is resident — that is the original #286 defect wearing a different hat.
  func testPresetWithNoModelAndNoRequestModelIsUnresolvable() throws {
    let bare = lookup(ImagePreset(
      id: "no-model", name: "x", mediaKind: "image",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)]))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "no-model", lookup: bare, requestLoras: nil))
    XCTAssertNil(e.loras)
    XCTAssertNil(e.model)
    XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "no_model")
  }

  /// `custom_model_path` is stored for the desktop app and never read by the
  /// engine — it does not make a preset loadable. This is the shape of all 26
  /// desktop-saved presets in the live store.
  func testCustomModelPathAloneIsStillNoModel() throws {
    let desktop = lookup(ImagePreset(
      id: "788B45BC", name: "Desktop preset", model: nil,
      customModelPath: "/Users/todd/LocalModels/krea2-raw",
      loras: [LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel")],
      kroma: KromaPolicy(strength: 0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "788B45BC", lookup: desktop, requestLoras: nil))
    XCTAssertNil(e.loras)
    let reason = try XCTUnwrap(e.unresolved)
    XCTAssertEqual(reason.code, "no_model")
    XCTAssertTrue(reason.message.contains("custom_model_path"), reason.message)
  }

  /// Round 3: a request model is NOT permission on its own. With neither
  /// `model` nor `checkpoint_family` the preset's family is unknowable, and
  /// unknowable is not a match — a krea2 stack pushed onto Z-Image binds zero
  /// layers and only warns, which is #286's silent-wrong-look defect again.
  func testPresetWithNeitherModelNorFamilyIsUnresolvableEvenWithARequestModel() throws {
    let bare = lookup(ImagePreset(
      id: "no-model", name: "x", mediaKind: "image",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)]))
    for requested in ["z-image-turbo", "krea2-raw", "something-nobody-classifies"] {
      let e = try expansion(PresetLoRAStack.decide(
        presetId: "no-model", lookup: bare, requestLoras: nil, requestModel: requested))
      XCTAssertNil(e.loras, "no family to check against '\(requested)' — must not expand")
      XCTAssertNil(e.model)
      XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "no_model")
    }
  }

  /// The UNLESS, as it now stands: the preset's family is KNOWN and the
  /// requested base's family is KNOWN and they agree. Then the stack expands
  /// and the request's own model stands.
  func testPresetWithNoModelExpandsWhenBothFamiliesAreKnownAndAgree() throws {
    let krea2 = lookup(ImagePreset(
      id: "krea2-loras", name: "x", mediaKind: "image",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea2-loras", lookup: krea2, requestLoras: nil, requestModel: "krea2-raw"))
    XCTAssertNil(e.model, "the request's own base stands")
    XCTAssertEqual(e.loras?.map(\.filename), ["a.safetensors"])
    XCTAssertNil(e.unresolved)
  }

  /// A base the engine does not classify is also unknowable — same rule.
  func testPresetWithNoModelRefusesAnUnclassifiableRequestModel() throws {
    let krea2 = lookup(ImagePreset(
      id: "krea2-loras", name: "x", mediaKind: "image",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea2-loras", lookup: krea2, requestLoras: nil,
      requestModel: "/Volumes/Bolt/some-unclassified-checkpoint"))
    XCTAssertNil(e.loras)
    XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "no_model")
  }

  /// …but where the preset's family IS knowable it still has to agree: krea2
  /// adapters on a Z-Image base bind zero layers and only warn.
  func testPresetWithNoModelRefusesAFamilyMismatch() throws {
    let krea2Stack = lookup(ImagePreset(
      id: "krea2-loras", name: "x", mediaKind: "image",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea2-loras", lookup: krea2Stack, requestLoras: nil,
      requestModel: "z-image-turbo"))
    XCTAssertNil(e.loras)
    let reason = try XCTUnwrap(e.unresolved)
    XCTAssertEqual(reason.code, "no_model")
    XCTAssertTrue(reason.message.contains("z-image-turbo"), reason.message)
  }

  // MARK: Round 3, minor 3 — `~` in a model spec

  /// `~/LocalModels/krea2-raw` and its expansion are the SAME base; a 409
  /// between them would refuse a valid request.
  func testTildeAndExpandedModelSpecsAreNotAConflict() throws {
    let home = NSHomeDirectory()
    let preset = lookup(ImagePreset(
      id: "tilde", name: "x", mediaKind: "image", model: "~/LocalModels/krea2-raw",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "tilde", lookup: preset, requestLoras: nil,
      requestModel: "\(home)/LocalModels/krea2-raw"))
    XCTAssertNil(e.unresolved)
    XCTAssertNil(e.model, "the request already named the base")
    XCTAssertEqual(e.loras?.count, 1)
  }

  func testTildeSpecsThatDifferAreStillAConflict() {
    let preset = lookup(ImagePreset(
      id: "tilde", name: "x", mediaKind: "image", model: "~/LocalModels/krea2-raw",
      loras: [], checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))
    guard case .modelConflict = PresetLoRAStack.decide(
      presetId: "tilde", lookup: preset, requestLoras: nil,
      requestModel: "~/LocalModels/somewhere-else")
    else { return XCTFail("different directories must still conflict") }
  }

  // MARK: Round 2, finding 1 — the engine/provider gate

  /// The seeded default. `engine: "mflux"`, `model: "schnell"` — expanding it
  /// turns a harmless label into a `poolLoad` of a model this engine cannot
  /// serve.
  func testSeededSchnellHQPresetIsUnresolvable() throws {
    let schnell = try XCTUnwrap(PresetStore.defaultPresets.first { $0.id == "schnell-hq" })
    XCTAssertEqual(schnell.engine, "mflux", "precondition: the seed still declares mflux")

    let e = try expansion(PresetLoRAStack.decide(
      presetId: "schnell-hq", lookup: lookup(schnell), requestLoras: nil))
    XCTAssertNil(e.loras)
    XCTAssertNil(e.model, "nothing must reach poolLoad")
    XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "engine:mflux")
  }

  /// The other seeded default is this engine's own and must still expand.
  func testSeededZImageChatPresetExpands() throws {
    let chat = try XCTUnwrap(PresetStore.defaultPresets.first { $0.id == "zimage-chat" })
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "zimage-chat", lookup: lookup(chat), requestLoras: nil))
    XCTAssertNil(e.unresolved)
    XCTAssertEqual(e.model, "z-image-turbo")
    XCTAssertEqual(e.loras, [], "an empty preset stack CLEARS the resident one — it is a declaration")
    XCTAssertEqual(e.steps, 8)
  }

  func testRemoteProviderIsUnresolvable() throws {
    let remote = lookup(ImagePreset(
      id: "replicate-lane", name: "x", mediaKind: "image", provider: "replicate",
      engine: "zimage", model: "krea2-raw",
      loras: [LoraReference(filename: "a.safetensors", scale: 1)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "replicate-lane", lookup: remote, requestLoras: nil))
    XCTAssertNil(e.loras)
    XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "provider:replicate")
  }

  /// An OMITTED engine is not a declaration. `ResolvedPreset` fills it from
  /// `PresetDefaults`, whose default is literally "mflux" — reading the gate
  /// off the resolved view would refuse every preset that simply leaves the
  /// field out, which is 26 of the presets in the live store.
  func testOmittedEngineIsNotAGate() throws {
    XCTAssertEqual(
      PresetDefaults.standard.engine, "mflux",
      "precondition: the resolved default really is mflux, which is why the gate reads `declared`")
    let preset = ImagePreset(
      id: "no-engine", name: "x", mediaKind: "image", model: "krea2-raw",
      loras: [LoraReference(filename: "a.safetensors", scale: 1)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0))
    XCTAssertEqual(ResolvedPreset(preset: preset).engine, "mflux", "precondition")

    let e = try expansion(PresetLoRAStack.decide(
      presetId: "no-engine", lookup: lookup(preset), requestLoras: nil))
    XCTAssertNil(e.unresolved)
    XCTAssertEqual(e.model, "krea2-raw")
  }

  /// The gate must never fire before the request/preset model contradiction is
  /// even considered — an unresolvable preset contributes no model, so it can
  /// never 409.
  func testANonLocalEnginePresetNeverConflictsOnModel() {
    let schnell = PresetStore.defaultPresets.first { $0.id == "schnell-hq" }!
    XCTAssertEqual(
      reason(PresetLoRAStack.decide(
        presetId: "schnell-hq", lookup: lookup(schnell), requestLoras: nil,
        requestModel: "krea2-raw")),
      "engine:mflux")
  }

  // MARK: C1 — declared steps/guidance, and only declared

  func testDeclaredStepsAndGuidanceAreAdoptedWhenTheRequestOmitsThem() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira-avocado", lookup: kreaKiraAvocado(steps: 52, guidance: 3.5),
      requestLoras: nil))
    XCTAssertEqual(e.steps, 52)
    XCTAssertEqual(e.guidance, 3.5)
  }

  func testRequestStepsAndGuidanceWin() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira-avocado", lookup: kreaKiraAvocado(steps: 52, guidance: 3.5),
      requestLoras: nil, requestSteps: 9, requestGuidance: 1.0))
    XCTAssertNil(e.steps)
    XCTAssertNil(e.guidance)
  }

  /// `ResolvedPreset.steps` falls back to `PresetDefaults.standard.steps` — **4**.
  /// Adopting that would drop a 52-step raw-stock render to 4 steps under the
  /// preset's own name, so only a DECLARED value is ever taken.
  func testUndeclaredStepsAreNotAdoptedFromPresetDefaults() throws {
    let undeclared = kreaKiraAvocado()
    guard case .resolved(let resolved, _) = undeclared else { return XCTFail("fixture") }
    XCTAssertEqual(resolved.steps, PresetDefaults.standard.steps, "precondition: resolve defaults it")

    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira-avocado", lookup: undeclared, requestLoras: nil))
    XCTAssertNil(e.steps, "an undeclared steps must never arrive as the store's default of 4")
    XCTAssertNil(e.guidance)
  }

  // MARK: D14 — kroma is a first-class field, prepended, never a `loras[]` row

  func testKromaIsPrependedAtItsDeclaredStrength() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira", lookup: kreaKira(), requestLoras: nil))
    let loras = try XCTUnwrap(e.loras)
    XCTAssertEqual(loras.count, 3)
    XCTAssertEqual(loras[0].filename, "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(loras[0].scale, 0.6)
    XCTAssertEqual(loras[0].role, "kroma")
    XCTAssertEqual(loras[1].filename, "krea2_turbo_distill_r256.safetensors")
    XCTAssertEqual(loras[1].role, "accel")
  }

  func testKromaStrengthZeroContributesNothing() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira-avocado", lookup: kreaKiraAvocado(), requestLoras: nil))
    XCTAssertFalse(e.loras?.contains { $0.role == "kroma" } ?? true)
  }

  // MARK: C2 — unexpandable presets stay LABELS, and say so

  /// The pre-#286 contract: an unknown `preset` is harmless provenance. Round 1
  /// ruled that a 400 here is a silent contract break, so the render behaves as
  /// before and the response carries `preset_unresolved`.
  func testUnknownPresetIsALabelAndIsReported() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira-typo", lookup: .notFound, requestLoras: nil))
    XCTAssertNil(e.loras, "nothing is applied — the request renders exactly as it did before #286")
    XCTAssertNil(e.model)
    XCTAssertNil(e.steps)
    let reason = try XCTUnwrap(e.unresolved)
    XCTAssertEqual(reason.code, "unknown_preset")
    XCTAssertTrue(reason.message.contains("krea-kira-typo"), reason.message)
  }

  func testInvalidPresetIsALabelAndIsReported() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-broken", lookup: .invalid(reason: "must declare \"kroma\""),
      requestLoras: nil))
    XCTAssertNil(e.loras)
    let reason = try XCTUnwrap(e.unresolved)
    XCTAssertEqual(reason.code, "invalid_preset")
    XCTAssertTrue(reason.message.contains("krea-broken"), reason.message)
    XCTAssertTrue(reason.message.contains("kroma"), reason.message)
  }

  /// A video preset on the image path would push LTX adapters at a Krea 2
  /// pipeline. Reported, not applied.
  func testVideoPresetIsALabelAndIsReported() throws {
    let video = lookup(ImagePreset(
      id: "kira-video-avocado", name: "Kira video", mediaKind: "video",
      loras: [LoraReference(filename: "ltx-2.3-i2v-t2v-video-reasoning-lora-vbvr.safetensors", scale: 1)]))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "kira-video-avocado", lookup: video, requestLoras: nil))
    XCTAssertNil(e.loras)
    XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "media_kind:video")
  }

  /// The engine has no family→default-kroma-file table (that policy lives in
  /// the client layer, FDD §3.17). Declared-on with no file is unreproducible,
  /// so nothing is applied rather than a stack that is missing its kroma.
  func testKromaOnWithNoFileIsALabelAndIsReported() throws {
    let preset = lookup(ImagePreset(
      id: "kroma-no-file", name: "x", mediaKind: "image", model: "krea2-raw",
      loras: [LoraReference(filename: "a.safetensors", scale: 0.5)],
      checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0.6)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "kroma-no-file", lookup: preset, requestLoras: nil))
    XCTAssertNil(e.loras)
    XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "kroma_file_missing")
  }

  /// The bypass `.diff` adapter is a preset dial with no engine application
  /// path — applying a stack that quietly drops it is the same class of defect
  /// as #286 itself.
  func testDeclaredBypassIsALabelAndIsReported() throws {
    let preset = lookup(ImagePreset(
      id: "bypass-on", name: "x", mediaKind: "image", model: "krea2-raw",
      loras: [], checkpointFamily: "raw-stock", kroma: KromaPolicy(strength: 0),
      bypass: BypassPolicy(strength: 2.0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "bypass-on", lookup: preset, requestLoras: nil))
    XCTAssertNil(e.loras)
    XCTAssertEqual(try XCTUnwrap(e.unresolved).code, "bypass_declared")
  }

  func testUnresolvedNeverConflictsOnModel() throws {
    // A preset that could not be expanded contributes no model either, so it
    // can never 409 a request that named its own base.
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "gone", lookup: .notFound, requestLoras: nil, requestModel: "z-image-turbo"))
    XCTAssertNotNil(e.unresolved)
    XCTAssertNil(e.model)
  }

  // MARK: I1 — explicit `loras` win, and a disagreement is reported

  func testExplicitLorasWinAndMatchingStackRaisesNoFlag() throws {
    guard case .resolved(let resolved, _) = kreaKira() else { return XCTFail("fixture") }
    var same: [LoraReference] = [
      LoraReference(filename: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors", scale: 0.6),
    ]
    same.append(contentsOf: resolved.loras)

    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira", lookup: kreaKira(), requestLoras: same))
    XCTAssertNil(e.loras, "explicit loras stand — the engine does not overwrite them")
    XCTAssertFalse(e.stackMismatch)
    XCTAssertEqual(e.model, "krea2-raw", "the preset's base still travels")
  }

  /// The real async production path: the client sends `preset` AND a FLAT
  /// `loras` list that has already dropped the structured kroma. Explicit still
  /// wins (existing precedence), but the response now says the two disagreed.
  func testFlatClientStackMissingKromaRaisesTheMismatchFlag() throws {
    let flat = [
      LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6),
      LoraReference(filename: "RealisticSnapshotKrea2.safetensors", scale: 0.4),
    ]
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira", lookup: kreaKira(), requestLoras: flat))
    XCTAssertNil(e.loras)
    XCTAssertTrue(e.stackMismatch, "kroma is missing from the client's flat list — say so")
  }

  func testExplicitEmptyListAgainstANonEmptyPresetIsAMismatch() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira", lookup: kreaKira(), requestLoras: []))
    XCTAssertNil(e.loras)
    XCTAssertTrue(e.stackMismatch)
  }

  /// An absolute path and a bare filename name the same adapter.
  func testStackComparisonIgnoresPathForm() {
    XCTAssertTrue(PresetLoRAStack.isSameStack(
      [LoraReference(filename: "/Volumes/Bolt/loras/a.safetensors", scale: 0.6)],
      [LoraReference(filename: "a.safetensors", scale: 0.6)]))
    XCTAssertFalse(PresetLoRAStack.isSameStack(
      [LoraReference(filename: "a.safetensors", scale: 0.6)],
      [LoraReference(filename: "a.safetensors", scale: 0.8)]))
    XCTAssertFalse(PresetLoRAStack.isSameStack(
      [LoraReference(filename: "a.safetensors", scale: 0.6)], []))
  }

  // MARK: No-regression

  /// A bare `/v1/generate` with no preset is the swap-first client's shape
  /// (`/v1/lora/swap` then generate) — it must still render on the stack the
  /// swap just installed.
  func testNoPresetIsUnchanged() {
    for id in [nil, "", "   "] as [String?] {
      XCTAssertEqual(
        PresetLoRAStack.decide(presetId: id, lookup: nil, requestLoras: nil), .unchanged)
      XCTAssertEqual(
        PresetLoRAStack.decide(presetId: id, lookup: nil, requestLoras: []), .unchanged)
    }
  }

  /// A preset that resolves to an empty stack applies an EMPTY stack — it does
  /// not mean "leave whatever is loaded". `krea-bree` is exactly this shape.
  func testPresetWithNoLorasAppliesAnEmptyStack() throws {
    let bare = lookup(ImagePreset(
      id: "krea-bree", name: "Bree", mediaKind: "image", model: "kroma-v0.2-turbo",
      kroma: KromaPolicy(strength: 0)))
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-bree", lookup: bare, requestLoras: nil))
    XCTAssertEqual(e.loras, [])
    XCTAssertEqual(e.model, "kroma-v0.2-turbo")
  }

  // MARK: Parity with /v1/presets/resolve

  /// The decision must consume exactly what `/v1/presets/resolve` publishes —
  /// same store, same read, no parallel computation. Driven through a real
  /// `PresetStore`.
  func testAgreesWithPresetStoreResolve() throws {
    let store = try makeStore()
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
    let (declared, invalidReason) = store.lookup("krea-kira")
    XCTAssertNil(invalidReason)
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira",
      lookup: .resolved(resolved, declared: try XCTUnwrap(declared)),
      requestLoras: nil))

    let loras = try XCTUnwrap(e.loras)
    let appliedPairs: [String: Double] = Dictionary(
      uniqueKeysWithValues: loras.map { ($0.filename, $0.scale) })
    for reference in resolved.loras {
      XCTAssertEqual(
        appliedPairs[reference.filename], reference.scale,
        "resolve reported \(reference.filename) but the applied stack disagrees")
    }
    XCTAssertEqual(loras.count, resolved.loras.count + 1)
    XCTAssertEqual(e.model, resolved.model)
  }

  private func makeStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("preset-lora-stack-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  // MARK: The verification hook — `applied_loras` on the wire

  /// #286's other half: the daemon could not tell a wrong stack from a right
  /// one because the response said nothing about adapters for any family but
  /// Krea 2. `applied_loras` is the additive field it can diff against
  /// `/v1/presets/resolve` — snake_case, name + path + scale + role, no
  /// existing field renamed.
  func testAppliedLorasIsOnTheWireAndAdditive() throws {
    var kroma = LoRAConfiguration.local(
      "/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors", scale: 0.6)
    kroma.role = "kroma"
    let response = GenerateResponse(
      success: true, outputPath: "/tmp/x.png", durationMs: 1234,
      appliedLoras: [kroma, .local("/loras/krea2_turbo_distill_r256.safetensors", scale: 0.6)]
        .map(LoRAState.init))

    let object = try encodeToObject(response)
    // Every pre-#286 field keeps its name.
    XCTAssertEqual(object["success"] as? Bool, true)
    XCTAssertEqual(object["output_path"] as? String, "/tmp/x.png")
    XCTAssertEqual(object["duration_ms"] as? Int, 1234)

    let applied = try XCTUnwrap(object["applied_loras"] as? [[String: Any]])
    XCTAssertEqual(applied.count, 2)
    let first = try XCTUnwrap(applied.first)
    XCTAssertEqual(first["name"] as? String, "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(first["path"] as? String, "/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(first["source"] as? String, "/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors")
    XCTAssertEqual(first["scale"] as? Double, 0.6)
    XCTAssertEqual(first["role"] as? String, "kroma")
  }

  /// A family with no LoRA path at all reports the key ABSENT, never an empty
  /// array — "engine has no stack here" must not read as "rendered bare". The
  /// two preset flags are absent the same way when nothing happened.
  func testAbsentFieldsAreAbsentNotEmpty() throws {
    let object = try encodeToObject(
      GenerateResponse(success: true, outputPath: "/tmp/x.png", durationMs: 1))
    XCTAssertNil(object["applied_loras"])
    XCTAssertNil(object["preset_unresolved"])
    XCTAssertNil(object["preset_unresolved_reason"])
    XCTAssertNil(object["preset_stack_mismatch"])
  }

  func testPresetFlagsAreOnTheWire() throws {
    let object = try encodeToObject(GenerateResponse(
      success: true, outputPath: "/tmp/x.png", durationMs: 1,
      presetUnresolved: "schnell-hq", presetUnresolvedReason: "engine:mflux",
      presetStackMismatch: true))
    XCTAssertEqual(object["preset_unresolved"] as? String, "schnell-hq")
    XCTAssertEqual(object["preset_unresolved_reason"] as? String, "engine:mflux")
    XCTAssertEqual(object["preset_stack_mismatch"] as? Bool, true)
  }

  /// `LoRAState` gained `name`/`path`/`role` — a persisted pre-#286 job status
  /// must still decode.
  func testLoRAStateDecodesPre286JSON() throws {
    let json = Data(#"{"source":"/loras/a.safetensors","scale":0.6}"#.utf8)
    let state = try JSONDecoder().decode(LoRAState.self, from: json)
    XCTAssertEqual(state.source, "/loras/a.safetensors")
    XCTAssertEqual(state.path, "/loras/a.safetensors")
    XCTAssertEqual(state.name, "a.safetensors")
    XCTAssertNil(state.role)
  }

  private func encodeToObject<T: Encodable>(_ value: T) throws -> [String: Any] {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return try XCTUnwrap(
      JSONSerialization.jsonObject(with: try encoder.encode(value)) as? [String: Any])
  }
}
