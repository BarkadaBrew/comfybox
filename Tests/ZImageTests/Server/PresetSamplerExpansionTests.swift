// PresetSamplerExpansionTests.swift — comfybox#419 (scope B, engine)
//
// A preset has carried a full sampler recipe since WP-E20 — `sampler` (and the
// older `scheduler`), `sigma_schedule`, `eta`, `bongmath`, `stage2`, and the
// ClownsharK dials `noise_type` / `noise_alpha` / `implicit_steps` / `c2` /
// `projector_scale` — and `PresetStore.validate` has always checked the names
// and ranges. But `expandingPreset` applied only `model`/`loras`/`steps`/
// `guidance`/`vae`/`shift`, so `POST /v1/generate {"preset": "krea-kira"}`
// rendered euler over the family's default grid while the preset said
// `res_2s + beta57`, and reported success. Only the daemon, which reads
// presets.json itself and sends every field explicitly, ever got the recipe.
//
// These pin the expansion under the rule every other declared field follows
// (`PresetShiftExpansionTests`): DECLARED only, per field, and only where the
// request said nothing — and, new here, that the family gates at dispatch run
// on the EXPANDED values, so a preset can never smuggle a combination past a
// 400 an explicit request would get.

import XCTest

@testable import ZImage

final class PresetSamplerExpansionTests: XCTestCase {

  // MARK: Fixtures

  /// A Krea 2 preset that declares EVERY recipe field.
  private func fullRecipePreset(
    id: String = "krea-clown", sampler: String? = "res_2s", scheduler: String? = nil,
    sigmaSchedule: String? = "beta57", eta: Double? = 0.5, bongmath: Bool? = true,
    stage2: PresetStage? = PresetStage(
      sampler: "res_3s", sigmaSchedule: "bong_tangent", steps: 6, denoise: 0.4, eta: 0.3),
    noiseType: String? = "fractal", noiseAlpha: Double? = 0.3, implicitSteps: Int? = 2,
    c2: Double? = 0.4, projectorScale: Double? = 1.2, shift: Double? = 1.15
  ) -> ImagePreset {
    ImagePreset(
      id: id, name: id, mediaKind: "image", model: "krea2-raw", steps: 12, guidance: 1.0,
      projectorScale: projectorScale, noiseType: noiseType, noiseAlpha: noiseAlpha,
      implicitSteps: implicitSteps, c2: c2,
      loras: [LoraReference(filename: "Girly_Tiana.safetensors", scale: 0.6)],
      scheduler: scheduler, checkpointFamily: "raw-accel",
      sampler: sampler, sigmaSchedule: sigmaSchedule, shift: shift, eta: eta,
      bongmath: bongmath, stage2: stage2)
  }

  private func lookup(_ preset: ImagePreset) -> PresetLoRAStack.Lookup {
    .resolved(ResolvedPreset(preset: preset), declared: preset)
  }

  private func expansion(_ decision: PresetLoRAStack) throws -> PresetExpansion {
    guard case .apply(let e) = decision else {
      XCTFail("expected .apply, got \(decision)")
      throw NSError(domain: "test", code: 1)
    }
    XCTAssertNil(e.unresolved, "expected an expanded preset, got \(e.unresolved!.code)")
    return e
  }

  /// `preset` is engine-set-from-the-wire only, so the seam tests decode the
  /// body a client actually posts — snake_case, through the route's decoder.
  private func decode(_ json: String) throws -> GeneratePayload {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  private func expand(_ json: String, _ preset: ImagePreset) throws -> GeneratePayload {
    try GeneratePayload.expandingPreset(try decode(json)) { _ in self.lookup(preset) }
  }

  private static let everyKey = [
    "scheduler", "sigma_schedule", "eta", "bongmath", "stage2",
    "noise_type", "noise_alpha", "implicit_steps", "c2", "projector_scale",
  ]

  // MARK: - `decide`: each field, declared and adopted

  func testEveryDeclaredRecipeFieldIsAdoptedWhenTheRequestOmitsIt() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-clown", lookup: lookup(fullRecipePreset()), requestLoras: nil))
    XCTAssertEqual(e.sampler, "res_2s")
    XCTAssertEqual(e.sigmaSchedule, "beta57")
    XCTAssertEqual(e.eta, 0.5)
    XCTAssertEqual(e.bongmath, true)
    XCTAssertEqual(e.stage2, PresetStage(
      sampler: "res_3s", sigmaSchedule: "bong_tangent", steps: 6, denoise: 0.4, eta: 0.3))
    XCTAssertEqual(e.noiseType, "fractal")
    XCTAssertEqual(e.noiseAlpha, 0.3)
    XCTAssertEqual(e.implicitSteps, 2)
    XCTAssertEqual(e.c2, 0.4)
    XCTAssertEqual(e.projectorScale, 1.2)
    // …and the fields that were already expanded are unaffected.
    XCTAssertEqual(e.model, "krea2-raw")
    XCTAssertEqual(e.steps, 12)
    XCTAssertEqual(e.loras?.count, 1)
    // #154 stands: a krea2 preset's `shift` is `mu` and is never adopted.
    XCTAssertNil(e.shift)
  }

  /// Request > preset, PER FIELD: naming one field on the request suppresses
  /// only that field's adoption; the others still come from the preset.
  func testRequestWinsPerField() throws {
    typealias Req = PresetLoRAStack.RequestRecipe
    let cases: [(String, Req, (PresetExpansion) -> Bool)] = [
      ("sampler", Req(sampler: "euler"), { $0.sampler == nil }),
      ("sigma_schedule", Req(sigmaSchedule: "karras"), { $0.sigmaSchedule == nil }),
      ("eta", Req(eta: 0), { $0.eta == nil }),
      ("bongmath", Req(bongmath: false), { $0.bongmath == nil }),
      ("stage2", Req(stage2Declared: true), { $0.stage2 == nil }),
      ("noise_type", Req(noiseType: "gaussian"), { $0.noiseType == nil }),
      ("noise_alpha", Req(noiseAlpha: 0), { $0.noiseAlpha == nil }),
      ("implicit_steps", Req(implicitSteps: 0), { $0.implicitSteps == nil }),
      ("c2", Req(c2: 0.5), { $0.c2 == nil }),
      ("projector_scale", Req(projectorScale: 1.0), { $0.projectorScale == nil }),
    ]
    for (field, request, suppressed) in cases {
      let e = try expansion(PresetLoRAStack.decide(
        presetId: "krea-clown", lookup: lookup(fullRecipePreset()), requestLoras: nil,
        requestRecipe: request))
      XCTAssertTrue(suppressed(e), "\(field): the request named it, so the preset's is not adopted")
      let adopted = [
        e.sampler != nil, e.sigmaSchedule != nil, e.eta != nil, e.bongmath != nil,
        e.stage2 != nil, e.noiseType != nil, e.noiseAlpha != nil, e.implicitSteps != nil,
        e.c2 != nil, e.projectorScale != nil,
      ].filter { $0 }.count
      if field == "sampler" {
        // B2: the request's `euler` makes the preset's eta 0.5 and bongmath
        // undefined — left off and recorded, so seven of the other nine
        // come through.
        XCTAssertEqual(adopted, 7, "\(field): eta + bongmath are skipped under the request's euler")
        XCTAssertEqual(e.skipped, [
          "eta (non-RES4LYF sampler 'euler')", "bongmath (non-RES4LYF sampler 'euler')",
        ])
      } else {
        XCTAssertEqual(adopted, 9, "\(field): exactly the other nine fields still come from the preset")
        XCTAssertEqual(e.skipped, [], field)
      }
    }
  }

  /// An explicit request value that EQUALS the preset's is still the
  /// request's — nothing is adopted, nothing is recorded as preset-sourced.
  func testMatchingExplicitValueIsStillTheRequests() throws {
    let out = try expand(
      #"{"prompt":"a portrait","preset":"krea-clown","scheduler":"res_2s","eta":0.5}"#,
      fullRecipePreset())
    XCTAssertEqual(out.scheduler, "res_2s")
    XCTAssertEqual(out.eta, 0.5)
    XCTAssertEqual(
      out.presetRecipeApplied,
      ["sigma_schedule", "bongmath", "stage2", "noise_type", "noise_alpha", "implicit_steps",
       "c2", "projector_scale"],
      "a field the request sent is never listed as preset-sourced, even when the values agree")
  }

  // MARK: - Legacy `scheduler`

  /// The daemon still reads the older `scheduler` key; a preset that carries
  /// only that spelling still names a sampler.
  func testLegacySchedulerOnlyPresetNamesTheSampler() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "legacy",
      lookup: lookup(fullRecipePreset(id: "legacy", sampler: nil, scheduler: "dpmpp_2m")),
      requestLoras: nil))
    XCTAssertEqual(e.sampler, "dpmpp_2m")
  }

  func testSamplerWinsOverLegacySchedulerWhenBothAreDeclared() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "both",
      lookup: lookup(fullRecipePreset(id: "both", sampler: "res_2s", scheduler: "euler")),
      requestLoras: nil))
    XCTAssertEqual(e.sampler, "res_2s")
  }

  func testBlankNamesAreNotDeclarations() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "blank",
      lookup: lookup(fullRecipePreset(id: "blank", sampler: " ", scheduler: "", sigmaSchedule: "")),
      requestLoras: nil))
    XCTAssertNil(e.sampler)
    XCTAssertNil(e.sigmaSchedule)
  }

  // MARK: - Nothing declared

  func testPresetDeclaringNoRecipeContributesNothing() throws {
    let bare = ImagePreset(
      id: "bare", name: "bare", mediaKind: "image", model: "krea2-raw", steps: 8,
      loras: [], checkpointFamily: "raw-accel")
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "bare", lookup: lookup(bare), requestLoras: nil))
    XCTAssertNil(e.sampler); XCTAssertNil(e.sigmaSchedule); XCTAssertNil(e.eta)
    XCTAssertNil(e.bongmath); XCTAssertNil(e.stage2); XCTAssertNil(e.noiseType)
    XCTAssertNil(e.noiseAlpha); XCTAssertNil(e.implicitSteps); XCTAssertNil(e.c2)
    XCTAssertNil(e.projectorScale)

    let out = try expand(#"{"prompt":"a portrait","preset":"bare"}"#, bare)
    XCTAssertNil(out.scheduler); XCTAssertNil(out.sigmaSchedule); XCTAssertNil(out.eta)
    XCTAssertNil(out.bongmath); XCTAssertNil(out.stage2); XCTAssertNil(out.noiseType)
    XCTAssertNil(out.noiseAlpha); XCTAssertNil(out.implicitSteps); XCTAssertNil(out.c2)
    XCTAssertNil(out.projectorScale)
    XCTAssertNil(out.presetRecipeApplied, "no key when the preset contributed no recipe")
    XCTAssertEqual(out.model, "krea2-raw", "the rest of the expansion is untouched")
  }

  /// An all-nil `stage2: {}` is not a declaration — it neither expands nor
  /// trips the incomplete-stage refusal.
  func testEmptyStage2ObjectIsTreatedAsUndeclared() throws {
    let preset = fullRecipePreset(stage2: PresetStage())
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-clown", lookup: lookup(preset), requestLoras: nil))
    XCTAssertNil(e.stage2)
    let out = try expand(#"{"prompt":"a portrait","preset":"krea-clown"}"#, preset)
    XCTAssertNil(out.stage2)
    XCTAssertFalse(out.presetRecipeApplied?.contains("stage2") ?? false)
  }

  func testNoPresetLeavesTheRecipeUntouched() throws {
    let payload = GeneratePayload(
      prompt: "a portrait", scheduler: "res_2s", sigmaSchedule: "beta57", eta: 0.5)
    let out = try GeneratePayload.expandingPreset(payload) { _ in
      XCTFail("no preset named — the store must not be consulted")
      return .notFound
    }
    XCTAssertEqual(out.scheduler, "res_2s")
    XCTAssertEqual(out.sigmaSchedule, "beta57")
    XCTAssertEqual(out.eta, 0.5)
    XCTAssertNil(out.presetRecipeApplied)
  }

  /// A preset the engine cannot expand stays a LABEL — its recipe is not
  /// applied any more than its LoRA stack is.
  func testUnexpandablePresetContributesNoRecipe() throws {
    let out = try GeneratePayload.expandingPreset(
      try decode(#"{"prompt":"a portrait","preset":"nope"}"#)) { _ in .notFound }
    XCTAssertNil(out.scheduler)
    XCTAssertNil(out.stage2)
    XCTAssertNil(out.presetRecipeApplied)
    XCTAssertEqual(out.presetUnresolved, "nope")
  }

  // MARK: - The `/v1/generate` seam

  func testExpandingPresetPutsTheWholeRecipeOnThePayloadAndRecordsIt() throws {
    let out = try expand(#"{"prompt":"a portrait","preset":"krea-clown"}"#, fullRecipePreset())
    XCTAssertEqual(out.scheduler, "res_2s")
    XCTAssertEqual(out.sigmaSchedule, "beta57")
    XCTAssertEqual(out.eta, 0.5)
    XCTAssertEqual(out.bongmath, true)
    XCTAssertEqual(out.stage2, Stage2Payload(
      steps: 6, denoise: 0.4, scheduler: "res_3s", sigmaSchedule: "bong_tangent", eta: 0.3))
    XCTAssertEqual(out.noiseType, "fractal")
    XCTAssertEqual(out.noiseAlpha, 0.3)
    XCTAssertEqual(out.implicitSteps, 2)
    XCTAssertEqual(out.c2, 0.4)
    XCTAssertEqual(out.projectorScale, 1.2)
    XCTAssertEqual(out.presetRecipeApplied, Self.everyKey)
    XCTAssertEqual(PresetExpansion.recipeWireKeys, Self.everyKey)
    XCTAssertNil(out.shift, "#154: krea2 shift stays request-only")
  }

  func testExpandingPresetLeavesEveryExplicitRequestFieldAlone() throws {
    let body = #"""
      {"prompt":"a portrait","preset":"krea-clown",
       "sampler":"euler","sigma_schedule":"karras","eta":0,"bongmath":false,
       "stage2":{"steps":3,"denoise":0.2},
       "noise_type":"gaussian","noise_alpha":0,"implicit_steps":0,"c2":0.5,"projector_scale":1}
      """#
    let out = try expand(body, fullRecipePreset())
    XCTAssertEqual(out.scheduler, "euler")
    XCTAssertEqual(out.sigmaSchedule, "karras")
    XCTAssertEqual(out.eta, 0)
    XCTAssertEqual(out.bongmath, false)
    XCTAssertEqual(out.stage2, Stage2Payload(steps: 3, denoise: 0.2))
    XCTAssertEqual(out.noiseType, "gaussian")
    XCTAssertEqual(out.noiseAlpha, 0)
    XCTAssertEqual(out.implicitSteps, 0)
    XCTAssertEqual(out.c2, 0.5)
    XCTAssertEqual(out.projectorScale, 1)
    XCTAssertNil(out.presetRecipeApplied, "the request named everything — nothing is preset-sourced")
  }

  /// The daemon's shape: `preset` PLUS every sampler field spelled out. The
  /// preset's recipe must be invisible to such a request — this is the
  /// behaviour-preservation guarantee for the production caller.
  func testDaemonStyleRequestWithPresetAndExplicitRecipeIsUnchanged() throws {
    let explicit = #"""
      {"prompt":"a portrait","scheduler":"res_2s","sigma_schedule":"beta57","eta":0.5,
       "bongmath":true,"steps":12,"guidance":1}
      """#
    let withPreset = #"""
      {"prompt":"a portrait","preset":"krea-clown","scheduler":"res_2s","sigma_schedule":"beta57",
       "eta":0.5,"bongmath":true,"steps":12,"guidance":1,
       "stage2":{"steps":3,"denoise":0.2},"noise_type":"gaussian","noise_alpha":0,
       "implicit_steps":0,"c2":0.5,"projector_scale":1}
      """#
    // A preset whose recipe DISAGREES with the request on every field.
    let preset = fullRecipePreset(
      sampler: "res_3s", sigmaSchedule: "bong_tangent", eta: 0.9, bongmath: false,
      noiseType: "pyramid", noiseAlpha: 0.7, implicitSteps: 4, c2: 0.3, projectorScale: 2.0)
    let a = try decode(explicit)
    let b = try expand(withPreset, preset)
    XCTAssertEqual(try a.krea2RecipeFields(), try b.krea2RecipeFields())
    XCTAssertEqual(b.stage2, Stage2Payload(steps: 3, denoise: 0.2))
    XCTAssertEqual(try b.validatedNoiseType(), .gaussian)
    XCTAssertEqual(try b.validatedImplicitSteps(), 0)
    XCTAssertEqual(try b.validatedC2(), 0.5)
    XCTAssertEqual(try b.validatedProjectorScale(), 1)
    XCTAssertNil(b.presetRecipeApplied)
  }

  // MARK: - Stage 2 partial → 400

  func testStage2MissingDenoiseIsRefusedNamingThePreset() throws {
    let preset = fullRecipePreset(stage2: PresetStage(sampler: "res_3s", steps: 6))
    XCTAssertThrowsError(try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)) { error in
      guard let warm = error as? WarmServerError,
            case .presetRecipeInvalid(let id, let field, let reason) = warm
      else { return XCTFail("expected .presetRecipeInvalid, got \(error)") }
      XCTAssertEqual(id, "krea-clown")
      XCTAssertEqual(field, "stage2")
      XCTAssertTrue(reason.contains("without denoise"), reason)
      XCTAssertFalse(reason.contains("steps and denoise"), reason)
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      let text = String(decoding: response.body, as: UTF8.self)
      XCTAssertTrue(text.contains("krea-clown") && text.contains("stage2"), text)
    }
  }

  func testStage2MissingStepsIsRefused() throws {
    let preset = fullRecipePreset(stage2: PresetStage(denoise: 0.4))
    XCTAssertThrowsError(try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)) { error in
      guard let warm = error as? WarmServerError,
            case .presetRecipeInvalid(_, let field, let reason) = warm
      else { return XCTFail("expected .presetRecipeInvalid, got \(error)") }
      XCTAssertEqual(field, "stage2")
      XCTAssertTrue(reason.contains("without steps"), reason)
    }
  }

  func testStage2MissingBothButDeclaringASamplerIsRefusedNamingBoth() throws {
    let preset = fullRecipePreset(stage2: PresetStage(sampler: "res_3s"))
    XCTAssertThrowsError(try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)) { error in
      guard let warm = error as? WarmServerError,
            case .presetRecipeInvalid(_, _, let reason) = warm
      else { return XCTFail("expected .presetRecipeInvalid, got \(error)") }
      XCTAssertTrue(reason.contains("without steps and denoise"), reason)
    }
  }

  /// The request's own `stage2` wins as a WHOLE object, so a partial preset
  /// stage is never even looked at — no 400 where the caller sent its own.
  func testRequestStage2SuppressesAPartialPresetStage() throws {
    let preset = fullRecipePreset(stage2: PresetStage(steps: 6))
    let out = try expand(
      #"{"prompt":"x","preset":"krea-clown","stage2":{"steps":3,"denoise":0.2}}"#, preset)
    XCTAssertEqual(out.stage2, Stage2Payload(steps: 3, denoise: 0.2))
  }

  // MARK: - Unresolvable preset names → 400 naming the preset

  /// `PresetStore.validate` checks `sampler` but not the legacy `scheduler`,
  /// so a stored preset CAN carry a name the engine does not resolve. It is a
  /// 400 naming the preset, never euler by coercion.
  func testUnknownPresetSamplerNameIs400NamingThePreset() throws {
    let preset = fullRecipePreset(sampler: nil, scheduler: "not-a-sampler")
    XCTAssertThrowsError(try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)) { error in
      guard let warm = error as? WarmServerError,
            case .presetRecipeInvalid(let id, let field, let reason) = warm
      else { return XCTFail("expected .presetRecipeInvalid, got \(error)") }
      XCTAssertEqual(id, "krea-clown")
      XCTAssertEqual(field, "sampler")
      XCTAssertTrue(reason.contains("not-a-sampler"), reason)
      XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
    }
  }

  func testUnknownPresetStage2ScheduleIs400() throws {
    let preset = fullRecipePreset(
      stage2: PresetStage(sigmaSchedule: "no-such-grid", steps: 6, denoise: 0.4))
    XCTAssertThrowsError(try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)) { error in
      guard let warm = error as? WarmServerError,
            case .presetRecipeInvalid(_, let field, _) = warm
      else { return XCTFail("expected .presetRecipeInvalid, got \(error)") }
      XCTAssertEqual(field, "stage2.sigma_schedule")
    }
  }

  // MARK: - Family gates run on the EXPANDED values

  private func krea2Gate(_ payload: GeneratePayload) -> WarmServerError? {
    do {
      try payload.validateKrea2TierGates(try payload.validateRecipeNames())
      return nil
    } catch { return error as? WarmServerError }
  }

  // MARK: PR #420 review B2 — the daemon's #1797 rule, engine-side

  /// A preset declaring `euler + eta 0.5` on Krea 2, with nothing on the
  /// request: the preset is the only layer that asked for the eta, so it is
  /// left OFF and recorded — the render goes ahead single-layer euler, it is
  /// NOT a 400 (the daemon does the same), and it is not silent.
  func testPresetEulerPlusEtaAloneRendersWithTheEtaSkippedAndRecorded() throws {
    let preset = fullRecipePreset(
      sampler: "euler", sigmaSchedule: nil, eta: 0.5, bongmath: nil, stage2: nil,
      noiseType: nil, noiseAlpha: nil, implicitSteps: nil, c2: nil, projectorScale: nil)
    let out = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)
    XCTAssertEqual(out.scheduler, "euler")
    XCTAssertNil(out.eta, "the preset's eta is not adopted under euler")
    XCTAssertEqual(out.presetRecipeApplied, ["scheduler"])
    XCTAssertEqual(out.presetRecipeSkipped, ["eta (non-RES4LYF sampler 'euler')"])
    XCTAssertNil(krea2Gate(out), "no eta on the payload ⇒ the eta gate has nothing to refuse")
    XCTAssertEqual(try out.krea2RecipeFields().eta, 0)
  }

  /// Preset `res_2s + eta 0.5`, request overrides the sampler to `euler`:
  /// the EFFECTIVE sampler is the request's, the preset's eta is undefined
  /// against it, so it is left off and recorded; the render proceeds.
  func testRequestSamplerOverrideToEulerSkipsThePresetEta() throws {
    let preset = fullRecipePreset(
      sampler: "res_2s", sigmaSchedule: "beta57", eta: 0.5, bongmath: nil, stage2: nil,
      noiseType: nil, noiseAlpha: nil, implicitSteps: nil, c2: nil, projectorScale: nil)
    let out = try expand(#"{"prompt":"x","preset":"krea-clown","sampler":"euler"}"#, preset)
    XCTAssertEqual(out.scheduler, "euler")
    XCTAssertNil(out.eta)
    XCTAssertEqual(out.sigmaSchedule, "beta57")
    XCTAssertEqual(out.presetRecipeApplied, ["sigma_schedule"])
    XCTAssertEqual(out.presetRecipeSkipped, ["eta (non-RES4LYF sampler 'euler')"])
    XCTAssertNil(krea2Gate(out))
    // Alias / prefix normalisation: a RES4LYF spelling the resolver accepts
    // keeps the eta.
    let kept = try expand(#"{"prompt":"x","preset":"krea-clown","sampler":"res_2s"}"#, preset)
    XCTAssertEqual(kept.eta, 0.5)
    XCTAssertNil(kept.presetRecipeSkipped)
  }

  /// A REQUEST-sourced eta on euler is untouched: still the existing 400.
  func testRequestSourcedEtaOnEulerIsStillRefused() throws {
    let explicit = try decode(#"{"prompt":"x","scheduler":"euler","eta":0.5}"#)
    guard case .unsupportedRecipeField(let field, let value, let family, _)? = krea2Gate(explicit)
    else { return XCTFail("a request eta on euler must hit the eta gate") }
    XCTAssertEqual(field, "eta"); XCTAssertEqual(value, "0.5"); XCTAssertEqual(family, "krea2")
    // …also when a preset is named that would have kept it (the request's
    // eta is the request's, whatever the preset says).
    let withPreset = try expand(
      #"{"prompt":"x","preset":"krea-clown","scheduler":"euler","eta":0.5}"#,
      fullRecipePreset(bongmath: nil, stage2: nil))
    XCTAssertEqual(withPreset.eta, 0.5)
    XCTAssertNil(withPreset.presetRecipeSkipped)
    XCTAssertNotNil(krea2Gate(withPreset))
  }

  /// A zero eta is adopted whatever the sampler — it asks for nothing.
  func testZeroPresetEtaIsAdoptedUnderEuler() throws {
    let preset = fullRecipePreset(sampler: "euler", eta: 0, bongmath: nil, stage2: nil)
    let out = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)
    XCTAssertEqual(out.eta, 0)
    XCTAssertNil(out.presetRecipeSkipped)
  }

  /// The rule is Krea 2's: on Z-Image `eta` is a shipped DDIM η and a preset's
  /// `euler + eta` is adopted as declared.
  func testZImagePresetEtaIsAdoptedUnderEuler() throws {
    let preset = ImagePreset(
      id: "zeta", name: "Zeta", mediaKind: "image", model: "z-image-zeta-chroma", loras: [],
      checkpointFamily: "zimage-base", sampler: "euler", eta: 0.5)
    let out = try expand(#"{"prompt":"x","preset":"zeta"}"#, preset)
    XCTAssertEqual(out.eta, 0.5)
    XCTAssertNil(out.presetRecipeSkipped)
  }

  /// Same rule for `stage2.eta` against the EFFECTIVE stage-2 sampler: the
  /// stage's own, else the render's (the `stage2Gate` fallback).
  func testPresetStage2EtaIsSkippedAgainstTheEffectiveStage2Sampler() throws {
    // Stage names euler itself → its eta is undefined → stage adopted, eta off.
    let own = fullRecipePreset(
      stage2: PresetStage(sampler: "euler", steps: 6, denoise: 0.4, eta: 0.3))
    let a = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, own)
    XCTAssertEqual(a.stage2, Stage2Payload(steps: 6, denoise: 0.4, scheduler: "euler"))
    XCTAssertEqual(a.presetRecipeSkipped, ["stage2.eta (non-RES4LYF sampler 'euler')"])
    XCTAssertTrue(a.presetRecipeApplied?.contains("stage2") ?? false)

    // Stage names no sampler → falls back to the render's res_2s → eta kept.
    let inherit = fullRecipePreset(bongmath: nil, stage2: PresetStage(steps: 6, denoise: 0.4, eta: 0.3))
    let b = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, inherit)
    XCTAssertEqual(b.stage2?.eta, 0.3)
    XCTAssertNil(b.presetRecipeSkipped)
    XCTAssertNil(GeneratePayload.stage2Gate(b, family: .krea2))

    // …and the render's sampler is the REQUEST's when it sent one.
    let c = try expand(#"{"prompt":"x","preset":"krea-clown","sampler":"euler"}"#, inherit)
    XCTAssertNil(c.stage2?.eta)
    XCTAssertEqual(c.presetRecipeSkipped, [
      "eta (non-RES4LYF sampler 'euler')", "stage2.eta (non-RES4LYF sampler 'euler')",
    ])
    XCTAssertNil(GeneratePayload.stage2Gate(c, family: .krea2))
  }

  // MARK: PR #420 review B1 — the explicit stage-2 OFF switch

  func testDetailPassFalseSwitchesThePresetStageOffAndRecordsIt() throws {
    let out = try expand(#"{"prompt":"x","preset":"krea-clown","detail_pass":false}"#, fullRecipePreset())
    XCTAssertNil(out.stage2, "single stage")
    XCTAssertEqual(out.detailPass, false)
    XCTAssertEqual(out.presetRecipeSkipped, ["stage2 (detail_pass=false)"])
    XCTAssertFalse(out.presetRecipeApplied?.contains("stage2") ?? false)
    XCTAssertEqual(out.scheduler, "res_2s", "the rest of the recipe is unaffected")
    XCTAssertNil(GeneratePayload.detailPassGate(out))
    XCTAssertNil(GeneratePayload.stage2Gate(out, family: .krea2))
    XCTAssertNil(GeneratePayload.stage2Gate(out, family: .flux1), "detail_pass=false is fine everywhere")
  }

  func testStage2NullSwitchesThePresetStageOffAndRecordsIt() throws {
    let out = try expand(#"{"prompt":"x","preset":"krea-clown","stage2":null}"#, fullRecipePreset())
    XCTAssertNil(out.stage2)
    XCTAssertTrue(out.stage2Null)
    XCTAssertEqual(out.stage2ExplicitlyOff, "stage2=null")
    XCTAssertEqual(out.presetRecipeSkipped, ["stage2 (stage2=null)"])
    XCTAssertNil(GeneratePayload.detailPassGate(out))
    XCTAssertNil(GeneratePayload.stage2Gate(out, family: .krea2))
  }

  /// Omission is NOT off: `{preset}` alone still adopts the stage (unchanged).
  func testOmittedStage2StillAdoptsThePresetStage() throws {
    let out = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, fullRecipePreset())
    XCTAssertNotNil(out.stage2)
    XCTAssertFalse(out.stage2Null)
    XCTAssertNil(out.stage2ExplicitlyOff)
    XCTAssertNil(out.presetRecipeSkipped)
  }

  /// The off switch records nothing when the preset declares no stage — there
  /// was nothing to skip.
  func testOffSwitchAgainstAPresetWithoutStage2RecordsNothing() throws {
    let out = try expand(
      #"{"prompt":"x","preset":"krea-clown","detail_pass":false}"#, fullRecipePreset(stage2: nil))
    XCTAssertNil(out.stage2)
    XCTAssertNil(out.presetRecipeSkipped)
  }

  func testDetailPassTrueAdoptsThePresetStage() throws {
    let out = try expand(#"{"prompt":"x","preset":"krea-clown","detail_pass":true}"#, fullRecipePreset())
    XCTAssertEqual(out.stage2?.steps, 6)
    XCTAssertNil(GeneratePayload.detailPassGate(out))
  }

  func testDetailPassTrueAgainstAPresetWithoutStage2Is400NamingThePreset() throws {
    XCTAssertThrowsError(
      try expand(#"{"prompt":"x","preset":"krea-clown","detail_pass":true}"#, fullRecipePreset(stage2: nil))
    ) { error in
      guard case .presetRecipeInvalid(let id, let field, let reason)? = error as? WarmServerError
      else { return XCTFail("expected .presetRecipeInvalid, got \(error)") }
      XCTAssertEqual(id, "krea-clown")
      XCTAssertEqual(field, "stage2")
      XCTAssertTrue(reason.contains("declares no stage2"), reason)
      XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
    }
  }

  func testDetailPassTrueBesideStage2NullIsAContradiction() throws {
    let out = try expand(
      #"{"prompt":"x","preset":"krea-clown","detail_pass":true,"stage2":null}"#, fullRecipePreset())
    guard case .mutuallyExclusive? = GeneratePayload.detailPassGate(out)
    else { return XCTFail("detail_pass=true + stage2=null must be refused") }
  }

  func testDetailPassFalseBesideARequestStage2IsAContradiction() throws {
    let payload = try decode(#"{"prompt":"x","detail_pass":false,"stage2":{"steps":3,"denoise":0.2}}"#)
    guard case .mutuallyExclusive? = GeneratePayload.detailPassGate(payload)
    else { return XCTFail("detail_pass=false + a stage2 object must be refused") }
    XCTAssertNotNil(GeneratePayload.stage2Gate(payload, family: .krea2))
  }

  /// No preset, `detail_pass: true`, no stage: the original AC-68a 400.
  func testDetailPassTrueWithNothingToExpandIsTheOriginal400() throws {
    let payload = try decode(#"{"prompt":"x","detail_pass":true}"#)
    guard case .unsupportedRecipeField(let field, _, _, _)? = GeneratePayload.detailPassGate(payload)
    else { return XCTFail("bare detail_pass=true must still be refused") }
    XCTAssertEqual(field, "detail_pass")
  }

  /// Replay: the off switch is carried faithfully — `detail_pass: false` /
  /// `stage2: null` stay in the body and no stage is written.
  func testRewrittenReplayBodyKeepsTheOffSwitch() throws {
    for body in [
      #"{"prompt":"x","preset":"krea-clown","detail_pass":false}"#,
      #"{"prompt":"x","preset":"krea-clown","stage2":null}"#,
    ] {
      let original = Data(body.utf8)
      let payload = try expand(body, fullRecipePreset())
      let object = try XCTUnwrap(
        try JSONSerialization.jsonObject(with: WarmServer.rawBody(original, expandedWith: payload))
          as? [String: Any])
      if body.contains("detail_pass") {
        XCTAssertEqual(object["detail_pass"] as? Bool, false, body)
        XCTAssertNil(object["stage2"], body)
      } else {
        XCTAssertTrue(object["stage2"] is NSNull, "stage2=null must survive: \(body)")
      }
      XCTAssertEqual(object["scheduler"] as? String, "res_2s", "the rest still replays: \(body)")
      // The replay decodes to the same OFF decision.
      let replayed = try decode(String(decoding: WarmServer.rawBody(original, expandedWith: payload), as: UTF8.self))
      XCTAssertNil(replayed.stage2, body)
      XCTAssertNotNil(replayed.stage2ExplicitlyOff, body)
    }
  }

  // MARK: PR #420 review nit — a JSON null is ABSENT for the replay merge

  /// `"eta": null` decodes as no eta, so the preset's eta applies; the replay
  /// body must carry that eta rather than leave the null in place.
  func testExplicitNullIsAbsentForTheReplayMerge() throws {
    let body = #"{"prompt":"x","preset":"krea-clown","eta":null,"steps":null}"#
    let payload = try expand(body, fullRecipePreset())
    XCTAssertEqual(payload.eta, 0.5)
    XCTAssertEqual(payload.steps, 12)
    let object = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: WarmServer.rawBody(Data(body.utf8), expandedWith: payload))
        as? [String: Any])
    XCTAssertEqual((object["eta"] as? NSNumber)?.floatValue, 0.5)
    XCTAssertEqual(object["steps"] as? Int, 12, "the pre-existing keys take the same rule")
  }

  /// B2, bongmath twin: a preset `euler + bongmath: true` alone renders
  /// single-layer with bongmath absent and the skip on the record — not a 400.
  func testPresetBongmathOnEulerAloneIsSkippedAndRecorded() throws {
    let preset = fullRecipePreset(sampler: "euler", eta: nil, bongmath: true, stage2: nil)
    let out = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)
    XCTAssertEqual(out.scheduler, "euler")
    XCTAssertNil(out.bongmath)
    XCTAssertNil(out.stage2)
    XCTAssertEqual(out.presetRecipeSkipped, ["bongmath (non-RES4LYF sampler 'euler')"])
    XCTAssertFalse(out.presetRecipeApplied?.contains("bongmath") ?? false)
    XCTAssertNil(krea2Gate(out))
    XCTAssertEqual(try out.krea2RecipeFields().bongmath, false)
  }

  /// Preset `res_2s + bongmath: true`, request overrides the sampler to
  /// `euler`: skipped against the effective sampler; `res_2s` keeps it.
  func testRequestSamplerOverrideToEulerSkipsThePresetBongmath() throws {
    let preset = fullRecipePreset(sampler: "res_2s", eta: nil, bongmath: true, stage2: nil)
    let out = try expand(#"{"prompt":"x","preset":"krea-clown","sampler":"euler"}"#, preset)
    XCTAssertNil(out.bongmath)
    XCTAssertEqual(out.presetRecipeSkipped, ["bongmath (non-RES4LYF sampler 'euler')"])
    XCTAssertNil(krea2Gate(out))
    let kept = try expand(#"{"prompt":"x","preset":"krea-clown","sampler":"res_2s"}"#, preset)
    XCTAssertEqual(kept.bongmath, true)
    XCTAssertNil(kept.presetRecipeSkipped)
  }

  /// A REQUEST-sourced `bongmath: true` on euler is untouched: still the 400.
  func testRequestSourcedBongmathOnEulerIsStillRefused() throws {
    let explicit = try decode(#"{"prompt":"x","scheduler":"euler","bongmath":true}"#)
    guard case .unsupportedRecipeField(let field, _, _, _)? = krea2Gate(explicit)
    else { return XCTFail("a request bongmath on euler must hit the bongmath gate") }
    XCTAssertEqual(field, "bongmath")
    let withPreset = try expand(
      #"{"prompt":"x","preset":"krea-clown","scheduler":"euler","bongmath":true}"#,
      fullRecipePreset(eta: nil, stage2: nil))
    XCTAssertEqual(withPreset.bongmath, true)
    XCTAssertNil(withPreset.presetRecipeSkipped)
    XCTAssertNotNil(krea2Gate(withPreset))
  }

  /// `bongmath: false` from a preset asks for nothing and is adopted as-is.
  func testFalsePresetBongmathIsAdoptedUnderEuler() throws {
    let preset = fullRecipePreset(sampler: "euler", eta: nil, bongmath: false, stage2: nil)
    let out = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)
    XCTAssertEqual(out.bongmath, false)
    XCTAssertNil(out.presetRecipeSkipped)
  }

  /// The family capability matrix reads the expanded names too: a preset's
  /// `res_3s` (an N-row tableau, krea2-only) is refused on the Z-Image family
  /// by name, and its `stage2` is refused on any non-krea2 family.
  func testExpandedNamesAndStage2GoThroughTheFamilyMatrixAndStage2Gate() throws {
    let out = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, fullRecipePreset(sampler: "res_3s"))
    let names = try out.validateRecipeNames()
    guard case .unsupportedSampler(let name, let family, _)? =
      GeneratePayload.validateFamilyRecipe(names, family: .flux1)
    else { return XCTFail("res_3s from a preset must be refused on flux1 by name") }
    XCTAssertEqual(name, "res_3s")
    XCTAssertEqual(family, "flux1")
    XCTAssertNil(GeneratePayload.validateFamilyRecipe(names, family: .krea2))

    guard case .unsupportedRecipeField(let field, _, _, _)? =
      GeneratePayload.stage2Gate(out, family: .flux1)
    else { return XCTFail("a preset-sourced stage2 must be refused on flux1") }
    XCTAssertEqual(field, "stage2")
  }

  /// And the stage-2 T3 stub: a preset stage that says `bongmath: true` is
  /// the same "not implemented" 400 the wire gets — passed through, not
  /// stripped.
  func testPresetStage2BongmathReachesTheExistingUnimplementedGate() throws {
    let preset = fullRecipePreset(
      stage2: PresetStage(sampler: "res_2s", steps: 6, denoise: 0.4, bongmath: true))
    let out = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)
    guard case .unsupportedRecipeField(let field, _, _, _)? =
      GeneratePayload.stage2Gate(out, family: .krea2)
    else { return XCTFail("stage2.bongmath from a preset must hit the T3 gate") }
    XCTAssertEqual(field, "stage2.bongmath")
  }

  // MARK: - Identity with the explicit request

  /// `res_2s + eta 0.5 + beta57 (+ shift 1.15)` from a preset resolves to the
  /// SAME recipe the pipeline runs as the explicit request — with the one
  /// documented exception: on Krea 2 the preset's `shift` is `mu` and is not
  /// adopted (#154, `PresetShiftExpansionTests`), so the explicit request
  /// carries 1.15 and the preset render keeps the resolution-dependent mu.
  func testPresetRecipeLandsIdenticallyToTheExplicitRequestOnKrea2() throws {
    let preset = fullRecipePreset(
      sampler: "res_2s", sigmaSchedule: "beta57", eta: 0.5, bongmath: nil, stage2: nil,
      noiseType: nil, noiseAlpha: nil, implicitSteps: nil, c2: nil, projectorScale: nil,
      shift: 1.15)
    let fromPreset = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, preset)
    let explicit = try decode(
      #"{"prompt":"x","scheduler":"res_2s","sigma_schedule":"beta57","eta":0.5,"shift":1.15}"#)

    let a = try fromPreset.krea2RecipeFields()
    let b = try explicit.krea2RecipeFields()
    XCTAssertEqual(a.sampler, b.sampler)
    XCTAssertEqual(a.sampler, .res2s)
    XCTAssertEqual(a.sigmaSchedule, b.sigmaSchedule)
    XCTAssertEqual(a.sigmaSchedule, .beta57)
    XCTAssertEqual(a.eta, b.eta)
    XCTAssertEqual(a.bongmath, b.bongmath)
    XCTAssertEqual(a.samplerRequested, b.samplerRequested)
    XCTAssertEqual(a.sigmaScheduleRequested, b.sigmaScheduleRequested)
    XCTAssertEqual(b.shift, 1.15)
    XCTAssertNil(a.shift, "#154: a krea2 preset's shift is never adopted — request-only")
    XCTAssertEqual(fromPreset.presetRecipeApplied, ["scheduler", "sigma_schedule", "eta"])
    // Both pass the same krea2 gates.
    XCTAssertNoThrow(try fromPreset.validateKrea2TierGates(try fromPreset.validateRecipeNames()))
    XCTAssertNoThrow(try explicit.validateKrea2TierGates(try explicit.validateRecipeNames()))
  }

  /// On the Z-Image family the shift IS adopted (#154), so the identity holds
  /// for the whole recipe including the shift.
  func testPresetRecipeLandsIdenticallyOnZImageIncludingShift() throws {
    let preset = ImagePreset(
      id: "zeta-chroma", name: "Zeta Chroma", mediaKind: "image",
      model: "z-image-zeta-chroma", steps: 28, guidance: 5.0, loras: [],
      checkpointFamily: "zimage-base",
      sampler: "euler", sigmaSchedule: "simple", shift: 3.0, eta: 0)
    let fromPreset = try expand(#"{"prompt":"x","preset":"zeta-chroma"}"#, preset)
    let explicit = try decode(
      #"{"prompt":"x","scheduler":"euler","sigma_schedule":"simple","shift":3.0,"eta":0}"#)
    XCTAssertEqual(try fromPreset.validateRecipeNames(), try explicit.validateRecipeNames())
    XCTAssertEqual(fromPreset.shift, explicit.shift)
    XCTAssertEqual(fromPreset.eta, explicit.eta)
    XCTAssertNil(GeneratePayload.validateShift(fromPreset.shift, family: .flux1))
    XCTAssertEqual(fromPreset.presetRecipeApplied, ["scheduler", "sigma_schedule", "eta"])
  }

  /// The stage-2 fields the pipeline runs are identical from either source.
  func testPresetStage2ResolvesIdenticallyToTheExplicitStage2() throws {
    let fromPreset = try expand(#"{"prompt":"x","preset":"krea-clown"}"#, fullRecipePreset())
    let explicit = try decode(#"""
      {"prompt":"x","stage2":{"steps":6,"denoise":0.4,"scheduler":"res_3s",
       "sigma_schedule":"bong_tangent","eta":0.3}}
      """#)
    XCTAssertEqual(try fromPreset.krea2Stage2Fields(), try explicit.krea2Stage2Fields())
    XCTAssertNotNil(try fromPreset.krea2Stage2Fields())
    XCTAssertNil(GeneratePayload.stage2Gate(fromPreset, family: .krea2))
  }

  // MARK: - Crash-recovery replay body

  /// A preset-owned recipe must survive a replay: the rewritten body carries
  /// every field under the wire's snake_case keys, so the replayed job renders
  /// the recipe it was accepted with (and, carrying them explicitly, takes
  /// the request-wins branch and resolves nothing again).
  func testRewrittenReplayBodyCarriesThePresetRecipe() throws {
    let original = Data(#"{"prompt":"a portrait","preset":"krea-clown"}"#.utf8)
    let payload = try GeneratePayload.expandingPreset(try decode(String(decoding: original, as: UTF8.self))) {
      _ in self.lookup(self.fullRecipePreset())
    }
    let rewritten = WarmServer.rawBody(original, expandedWith: payload)
    let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: rewritten) as? [String: Any])
    XCTAssertEqual(object["scheduler"] as? String, "res_2s")
    XCTAssertEqual(object["sigma_schedule"] as? String, "beta57")
    XCTAssertEqual((object["eta"] as? NSNumber)?.floatValue, 0.5)
    XCTAssertEqual(object["bongmath"] as? Bool, true)
    XCTAssertEqual(object["noise_type"] as? String, "fractal")
    XCTAssertEqual((object["noise_alpha"] as? NSNumber)?.floatValue, 0.3)
    XCTAssertEqual(object["implicit_steps"] as? Int, 2)
    XCTAssertEqual((object["c2"] as? NSNumber)?.floatValue, 0.4)
    XCTAssertEqual((object["projector_scale"] as? NSNumber)?.floatValue, 1.2)
    let stage = try XCTUnwrap(object["stage2"] as? [String: Any])
    XCTAssertEqual(stage["steps"] as? Int, 6)
    XCTAssertEqual((stage["denoise"] as? NSNumber)?.doubleValue, 0.4)
    XCTAssertEqual(stage["scheduler"] as? String, "res_3s")
    XCTAssertEqual(stage["sigma_schedule"] as? String, "bong_tangent")
    XCTAssertEqual((stage["eta"] as? NSNumber)?.floatValue, 0.3)
    XCTAssertNil(stage["bongmath"])
    XCTAssertNil(object["shift"], "#154: no krea2 shift on the replay body either")

    // The replayed body decodes to the SAME recipe, now request-owned.
    let replayed = try decode(String(decoding: rewritten, as: UTF8.self))
    XCTAssertEqual(try replayed.krea2RecipeFields(), try payload.krea2RecipeFields())
    XCTAssertEqual(try replayed.krea2Stage2Fields(), try payload.krea2Stage2Fields())
    XCTAssertEqual(replayed.noiseType, payload.noiseType)
    XCTAssertEqual(replayed.implicitSteps, payload.implicitSteps)
    XCTAssertEqual(replayed.c2, payload.c2)
    XCTAssertEqual(replayed.projectorScale, payload.projectorScale)
    let again = try GeneratePayload.expandingPreset(replayed) { _ in self.lookup(self.fullRecipePreset()) }
    XCTAssertNil(again.presetRecipeApplied, "a replay carries the recipe explicitly, so nothing is re-adopted")
  }

  /// A request that named its own fields — under EITHER accepted spelling —
  /// is not rewritten: the body already says what it wants.
  func testRewrittenReplayBodyKeepsExplicitFieldsUnderBothSpellings() throws {
    let original = Data(#"""
      {"prompt":"a portrait","preset":"krea-clown","sampler":"euler","sigmaSchedule":"karras",
       "stage2":{"steps":3,"denoise":0.2},"projectorScale":1}
      """#.utf8)
    let payload = try GeneratePayload.expandingPreset(try decode(String(decoding: original, as: UTF8.self))) {
      _ in self.lookup(self.fullRecipePreset())
    }
    let object = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: WarmServer.rawBody(original, expandedWith: payload))
        as? [String: Any])
    XCTAssertEqual(object["sampler"] as? String, "euler")
    XCTAssertNil(object["scheduler"], "the request spelled it `sampler`; no second key is added")
    XCTAssertEqual(object["sigmaSchedule"] as? String, "karras")
    XCTAssertNil(object["sigma_schedule"])
    XCTAssertEqual((object["stage2"] as? [String: Any])?["steps"] as? Int, 3)
    XCTAssertEqual((object["projectorScale"] as? NSNumber)?.floatValue, 1)
    XCTAssertNil(object["projector_scale"])
    // The preset-sourced remainder IS written…
    XCTAssertEqual(object["noise_type"] as? String, "fractal")
    // …except eta and bongmath, which B2 left off under the request's `euler`
    // — so the replay body carries neither, and the skips are on the record.
    XCTAssertNil(object["eta"])
    XCTAssertNil(object["bongmath"])
    XCTAssertEqual(payload.presetRecipeSkipped, [
      "eta (non-RES4LYF sampler 'euler')", "bongmath (non-RES4LYF sampler 'euler')",
    ])
  }

  // MARK: - `preset_recipe_applied` on the response

  func testPresetRecipeAppliedIsAbsentByDefaultAndSnakeCasedWhenSet() throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase

    let quiet = GenerateResponse(success: true, outputPath: "/tmp/a.png", durationMs: 10)
    let quietJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: encoder.encode(quiet)) as? [String: Any])
    XCTAssertNil(quietJSON["preset_recipe_applied"])
    XCTAssertNil(quietJSON["preset_recipe_skipped"])

    let sourced = GenerateResponse(
      success: true, outputPath: "/tmp/a.png", durationMs: 10,
      presetRecipeApplied: ["scheduler", "sigma_schedule"],
      presetRecipeSkipped: ["eta (non-RES4LYF sampler 'euler')"])
    let sourcedJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: encoder.encode(sourced)) as? [String: Any])
    XCTAssertEqual(sourcedJSON["preset_recipe_applied"] as? [String], ["scheduler", "sigma_schedule"])
    XCTAssertEqual(sourcedJSON["preset_recipe_skipped"] as? [String], ["eta (non-RES4LYF sampler 'euler')"])
  }

  func testImageJobStatusCarriesPresetRecipeAppliedAndDecodesWithoutIt() throws {
    let status = ImageJobStatus(
      jobId: "j-1", status: .succeeded, source: "api", outputPath: "/tmp/a.png",
      durationMs: 10, error: nil, elapsedMs: 12, preemptRefused: nil, etaSec: nil,
      presetRecipeApplied: ["eta"], presetRecipeSkipped: ["stage2 (detail_pass=false)"])
    XCTAssertEqual(status.presetRecipeApplied, ["eta"])
    XCTAssertEqual(status.presetRecipeSkipped, ["stage2 (detail_pass=false)"])
    let legacy = #"""
    {"jobId":"j-1","status":"succeeded","source":"api","outputPath":"/tmp/a.png",
     "durationMs":10,"elapsedMs":12}
    """#.data(using: .utf8)!
    let decoded = try JSONDecoder().decode(ImageJobStatus.self, from: legacy)
    XCTAssertNil(decoded.presetRecipeApplied)
    XCTAssertNil(decoded.presetRecipeSkipped)
  }
}
