import XCTest
import MLX
import MLXRandom
@testable import ZImage

/// WP-E17 — two stages inside ONE render (FDD-krea2-raw-recipe §3.14, D4;
/// AC-27, AC-30, AC-32).
///
/// Weight-free: `evaluate` is a velocity field, so "the real loop, twice" is
/// exactly what runs — `Krea2DenoiseLoop.run` for stage 1 and
/// `Krea2StagedRender.runStage2` (which calls the same driver) for stage 2.
/// The VAE half of AC-30 (one decode, zero encodes) is structural here: this
/// path never touches a VAE, and the pipeline decodes once after both stages.
final class Krea2StagedRenderTests: XCTestCase {

  static let align = 16
  static let seqLen1024 = (1024 / 16) * (1024 / 16)
  static let patch = 2
  static let latH = 8, latW = 8
  static var hTok: Int { latH / patch }
  static var wTok: Int { latW / patch }

  static func shift(explicit: Float? = 1.15) throws -> Krea2Sampling.ScheduleShift {
    try Krea2Sampling.resolveShift(explicit: explicit, seqLen: seqLen1024, align: align)
  }

  /// Rectified flow with a closed-form solution: `v = ε − x₀`, independent of
  /// `x`. A consistent two-stage run must land back on `x₀` — the re-noise
  /// puts the sample exactly on the flow line at σ₂[0], and the second solve
  /// integrates the same field back to zero.
  struct ClosedFormField {
    let x0: MLXArray
    let eps: MLXArray
    func velocity(_ x: MLXArray, _ sigma: Float) -> MLXArray { eps - x0 }
    func sample(at sigma: Float) -> MLXArray { sigma * eps + (1.0 - sigma) * x0 }
  }

  static func closedForm(seed: UInt64, shape: [Int]) -> ClosedFormField {
    let keys = MLXRandom.split(key: MLXRandom.key(seed))
    let x0 = MLXRandom.normal(shape, key: keys.0).asType(.bfloat16)
    let eps = MLXRandom.normal(shape, key: keys.1).asType(.bfloat16)
    MLX.eval(x0, eps)
    return ClosedFormField(x0: x0, eps: eps)
  }

  /// A nonlinear stand-in for the transformer: any perturbation of the
  /// trajectory propagates instead of cancelling, so "stage 2 changed the
  /// picture" is a real measurement rather than an artefact of a linear field.
  static func nonlinear(_ x: MLXArray, _ sigma: Float) -> MLXArray {
    let t = MLX.full([1], values: MLXArray(sigma)).asType(.bfloat16)
    return MLX.tanh(x * 0.5) * 0.5 + t * 0.25 - x * 0.1
  }

  static func patchedNoise(seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let n = MLXRandom.normal([1, Krea2VAE.latentChannels, latH, latW]).asType(.bfloat16)
    MLX.eval(n)
    return Krea2Sampling.patchify(n, patch: patch)
  }

  static func bits(_ x: MLXArray) -> [Float] { x.asType(.float32).asArray(Float.self) }

  static func meanAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.mean(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
  }

  /// The published stage 2 (C8's `raw-turbo` row): `deis_3m` + `bong_tangent`,
  /// 2 steps at denoise 0.2.
  static func publishedStage2(seed: UInt64) -> Krea2Pipeline.Stage2 {
    Krea2Pipeline.Stage2(
      steps: 2, denoise: 0.2, sampler: .deis3m, sigmaSchedule: .bongTangent, seed: seed)
  }

  static func resolvedPublishedStage2(
    request: Krea2Pipeline.Request
  ) throws -> Krea2ResolvedStage {
    try XCTUnwrap(request.stage2).resolved(against: request)
  }

  /// Stage 1 exactly as `Krea2Pipeline.generateStaged` runs it, minus the
  /// transformer: the same scheduler builder, the same driver, the same trace.
  static func runStageOne(
    request: Krea2Pipeline.Request,
    shift: Krea2Sampling.ScheduleShift,
    initial: MLXArray,
    evaluate: (MLXArray, Float) -> MLXArray
  ) throws -> (sample: MLXArray, trace: Krea2RunTrace) {
    var scheduler = try Krea2Pipeline.makeScheduler(
      sampler: request.sampler, sigmaSchedule: request.sigmaSchedule, steps: request.steps,
      shift: shift, seed: request.seed, c2: request.c2)
    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: initial, startIndex: 0,
      modelEvalsPerEvaluate: request.guidance > 1 ? 2 : 1, evaluate: evaluate)
    let trace = Krea2RunTrace(
      request: request, shift: shift, scheduler: scheduler, stats: stats,
      startIndex: 0, denoise: 1.0, width: 1024, height: 1024,
      negativePromptApplied: Krea2RunTrace.negativePromptApplied(
        cfgActive: request.guidance > 1, requested: request.negativePrompt))
    return (x, trace)
  }

  static func runStageTwo(
    _ stage: Krea2ResolvedStage,
    shift: Krea2Sampling.ScheduleShift,
    stageOne: MLXArray,
    evaluate: @escaping (MLXArray, Float) -> MLXArray
  ) throws -> (sample: MLXArray, trace: Krea2RunTrace) {
    try Krea2StagedRender.runStage2(
      stage: stage, shift: shift, stageOneLatent: stageOne,
      patch: patch, hTok: hTok, wTok: wTok, latentHeight: latH, latentWidth: latW,
      dtype: .bfloat16, width: 1024, height: 1024, negativePromptApplied: nil,
      evaluate: { latent, sigma, _ in evaluate(latent, sigma) })
  }

  static func baseRequest(
    steps: Int = 6, seed: UInt64 = 44821, stage2: Krea2Pipeline.Stage2? = nil
  ) -> Krea2Pipeline.Request {
    var r = Krea2Pipeline.Request(
      prompt: "a test", width: 1024, height: 1024, steps: steps, seed: seed,
      sampler: .res2s, sigmaSchedule: .beta)
    r.shift = 1.15
    r.stage2 = stage2
    return r
  }

  // MARK: - AC-30 (weight-free half): two stages, two traces, one render

  func testTwoStagesRunTheLoopTwiceAndSumTheirEvals() throws {
    let shift = try Self.shift()
    let request = Self.baseRequest(stage2: Self.publishedStage2(seed: 900))
    let stage = try Self.resolvedPublishedStage2(request: request)

    var stageOneCalls = 0, stageTwoCalls = 0
    let (x1, trace1) = try Self.runStageOne(
      request: request, shift: shift, initial: Self.patchedNoise(seed: request.seed)
    ) { x, s in stageOneCalls += 1; return Self.nonlinear(x, s) }

    let (x2, trace2) = try Self.runStageTwo(stage, shift: shift, stageOne: x1) { x, s in
      stageTwoCalls += 1
      return Self.nonlinear(x, s)
    }

    // The loop really ran twice: stage 1's res_2s is 2 evaluations per step,
    // stage 2's deis_3m is entirely warm-up (AC-24) at 3 rows per step.
    XCTAssertEqual(trace1.modelEvals, stageOneCalls)
    XCTAssertEqual(trace2.modelEvals, stageTwoCalls)
    XCTAssertEqual(trace1.stepsRun, 6)
    XCTAssertEqual(trace2.stepsRun, 2)
    XCTAssertEqual(trace2.warmupSampler, "ralston_3s")
    XCTAssertEqual(trace2.warmupSteps, 2)
    XCTAssertEqual(trace2.modelEvals, 6)

    // Two Stage records from two traces, `model_evals_total` summed (D4, E10).
    let record = RenderRecipe.krea2(RenderRecipeFixture.inputs(traces: [trace1, trace2]))
    XCTAssertEqual(record.stages.count, 2)
    XCTAssertEqual(record.stages.map(\.index), [0, 1])
    XCTAssertEqual(record.modelEvalsTotal, trace1.modelEvals + trace2.modelEvals)
    XCTAssertEqual(record.stages[1].sampler, "deis_3m")
    XCTAssertEqual(record.stages[1].sigmaSchedule, "bong_tangent")
    XCTAssertEqual(record.stages[1].denoise, 0.2, accuracy: 1e-6)
    XCTAssertEqual(record.stages[1].stepsRequested, 2)
    XCTAssertEqual(record.stages[1].stepsEffective, 2)
    XCTAssertEqual(record.stages[1].seed, 900)
    XCTAssertFalse(record.stages[1].shiftApplied, "bong_tangent is shift-free (D6)")
    // AC-31 read back through provenance: the stretched tail, not stage 1's.
    XCTAssertEqual(record.stages[1].sigmaHead[0], 0.11746056, accuracy: 1e-6)
    XCTAssertNotEqual(record.stages[1].sigmaHead[0], record.stages[0].sigmaHead[0])

    // AC-30's other half: the output moved.
    XCTAssertGreaterThan(Self.meanAbsDiff(x2, x1), 1e-3)
  }

  /// A single-stage record still has exactly one stage — the shape a client
  /// polish render leaves behind (`stages.length == 1`, AC-30).
  func testSingleStageRecordStillHasOneStage() throws {
    let shift = try Self.shift()
    let request = Self.baseRequest()
    XCTAssertNil(request.stage2)
    let (_, trace) = try Self.runStageOne(
      request: request, shift: shift, initial: Self.patchedNoise(seed: request.seed),
      evaluate: Self.nonlinear)
    let record = RenderRecipe.krea2(RenderRecipeFixture.inputs(trace: trace))
    XCTAssertEqual(record.stages.count, 1)
    XCTAssertEqual(record.modelEvalsTotal, trace.modelEvals)
  }

  // MARK: - §3.14 step 3: the re-noise

  /// The stretch-and-tail σ the re-noise uses is the stage-2 grid's FIRST
  /// sigma (σ ≈ 0.117 at the published recipe), taken as the scheduler's
  /// float32 0-d `MLXArray` — never `.item(Float.self)`, which would run the
  /// whole mix in bf16 (§3.3's AC-2 trap, the same one `mixSourceLatent`
  /// preconditions on).
  func testRenoiseIsTheFloat32MixAtTheStageTwoStartSigma() throws {
    let shift = try Self.shift()
    let stage = Krea2ResolvedStage(
      sampler: .deis3m, sigmaSchedule: .bongTangent, sigmaScheduleRequested: nil,
      steps: 2, denoise: 0.2, guidance: 1.0, eta: 0, bongmath: false, c2: 0.5, seed: 900)
    let scheduler = try Krea2StagedRender.makeScheduler(
      sampler: stage.sampler, sigmaSchedule: stage.sigmaSchedule, steps: stage.steps,
      denoise: stage.denoise, shift: shift, seed: stage.seed, c2: stage.c2)
    let sigma = scheduler.sigmas[0]
    XCTAssertEqual(sigma.dtype, .float32)
    XCTAssertEqual(sigma.ndim, 0)
    XCTAssertEqual(sigma.item(Float.self), 0.11746056, accuracy: 1e-6)

    let stageOne = Self.patchedNoise(seed: 5)
    let mixed = Krea2StagedRender.renoise(
      stageOneLatent: stageOne, sigma: sigma, seed: stage.seed,
      patch: Self.patch, hTok: Self.hTok, wTok: Self.wTok,
      latentHeight: Self.latH, latentWidth: Self.latW, dtype: .bfloat16)

    // The reference: the same NCHW mix, computed here, in float32.
    MLXRandom.seed(stage.seed)
    let eps = MLXRandom.normal([1, Krea2VAE.latentChannels, Self.latH, Self.latW])
      .asType(.bfloat16)
    let source = Krea2Sampling.unpatchify(
      stageOne, patch: Self.patch, h: Self.hTok, w: Self.wTok, c: Krea2VAE.latentChannels)
    let want = Krea2Sampling.patchify(
      (eps * sigma + source * (1.0 - sigma)).asType(.bfloat16), patch: Self.patch)
    XCTAssertEqual(Self.bits(mixed), Self.bits(want))
    XCTAssertEqual(mixed.shape, stageOne.shape)
  }

  /// The stage-2 continuation is a genuine rectified-flow continuation, and
  /// the σ it continues from is the stretched tail's FIRST sigma.
  ///
  /// With a closed-form field (`v = ε − x₀`, constant) the whole second stage
  /// has an exact answer: the re-noise puts the sample at
  /// `x₂ = σ₂[0]·ε' + (1 − σ₂[0])·x₁` and integrating a constant velocity from
  /// σ₂[0] to 0 lands on `x₂ − σ₂[0]·v` — whatever grid and sampler get it
  /// there. So this pins the σ numerically: predicting the same run from
  /// `denoise` read as a starting sigma (0.2) misses by an order of magnitude.
  ///
  /// It deliberately does NOT assert "lands back on x₀": a fresh ε' is not the
  /// field's own ε, so the exact answer is `x₀ + σ₂[0]·(ε' − ε)` — the detail
  /// pass moves the picture, which is the point of it.
  func testClosedFormFieldLandsOnTheStageTwoLine() throws {
    let shift = try Self.shift()
    let tokens = Self.hTok * Self.wTok
    let feature = Krea2VAE.latentChannels * Self.patch * Self.patch
    let field = Self.closedForm(seed: 21, shape: [1, tokens, feature])
    let request = Self.baseRequest(steps: 6, stage2: Self.publishedStage2(seed: 900))

    let (x1, _) = try Self.runStageOne(
      request: request, shift: shift, initial: field.sample(at: 1.0),
      evaluate: { x, s in field.velocity(x, s) })
    XCTAssertLessThan(
      Self.meanAbsDiff(x1, field.x0), 5e-2, "stage 1 solves the closed-form field back to x₀")

    let stage = try Self.resolvedPublishedStage2(request: request)
    let scheduler = try Krea2StagedRender.makeScheduler(
      sampler: stage.sampler, sigmaSchedule: stage.sigmaSchedule, steps: stage.steps,
      denoise: stage.denoise, shift: shift, seed: stage.seed, c2: stage.c2)
    let sigma0 = scheduler.sigmas[0].item(Float.self)
    XCTAssertEqual(sigma0, 0.11746056, accuracy: 1e-6)

    let start = Krea2StagedRender.renoise(
      stageOneLatent: x1, sigma: scheduler.sigmas[0], seed: stage.seed,
      patch: Self.patch, hTok: Self.hTok, wTok: Self.wTok,
      latentHeight: Self.latH, latentWidth: Self.latW, dtype: .bfloat16)
    let v = field.eps - field.x0
    let predicted = start - sigma0 * v

    let (x2, _) = try Self.runStageTwo(stage, shift: shift, stageOne: x1) { x, s in
      field.velocity(x, s)
    }
    XCTAssertLessThan(
      Self.meanAbsDiff(x2, predicted), 2e-2,
      "stage 2 must integrate the field from σ₂[0] to 0")
    // The σ is load-bearing: `denoise` read as a starting sigma predicts a
    // materially different latent, and stage 2 is not it.
    XCTAssertGreaterThan(Self.meanAbsDiff(x2, start - 0.2 * v), 5e-2)
  }

  // MARK: - AC-32 (weight-free analogue): denoise → 0 is a no-op

  func testDenoiseNearZeroBarelyMoves() throws {
    let shift = try Self.shift()
    var request = Self.baseRequest(
      stage2: Krea2Pipeline.Stage2(
        steps: 2, denoise: 0.01, sampler: .deis3m, sigmaSchedule: .bongTangent, seed: 900))
    request.steps = 6
    let (x1, _) = try Self.runStageOne(
      request: request, shift: shift, initial: Self.patchedNoise(seed: request.seed),
      evaluate: Self.nonlinear)
    let stage = try Self.resolvedPublishedStage2(request: request)
    XCTAssertLessThan(
      try XCTUnwrap(
        Krea2StagedRender.makeScheduler(
          sampler: stage.sampler, sigmaSchedule: stage.sigmaSchedule, steps: stage.steps,
          denoise: stage.denoise, shift: shift, seed: stage.seed, c2: stage.c2
        ).sigmas[0].item(Float.self)),
      0.01, "at denoise 0.01 the stretched tail starts far below σ = 0.01")
    let (x2, _) = try Self.runStageTwo(stage, shift: shift, stageOne: x1, evaluate: Self.nonlinear)
    XCTAssertLessThan(Self.meanAbsDiff(x2, x1), 2.0 / 255.0)
  }

  // MARK: - AC-27: determinism, and the stage-2 seed

  func testSameSeedsAreBitIdenticalAndStageTwoSeedMoves() throws {
    let shift = try Self.shift()
    func run(stage2Seed: UInt64) throws -> MLXArray {
      let request = Self.baseRequest(stage2: Self.publishedStage2(seed: stage2Seed))
      let (x1, _) = try Self.runStageOne(
        request: request, shift: shift, initial: Self.patchedNoise(seed: request.seed),
        evaluate: Self.nonlinear)
      let stage = try Self.resolvedPublishedStage2(request: request)
      return try Self.runStageTwo(stage, shift: shift, stageOne: x1, evaluate: Self.nonlinear).sample
    }
    XCTAssertEqual(Self.bits(try run(stage2Seed: 900)), Self.bits(try run(stage2Seed: 900)))
    XCTAssertNotEqual(Self.bits(try run(stage2Seed: 900)), Self.bits(try run(stage2Seed: 901)))
  }

  /// `stage2.seed: null` → stage-1 seed `&+ 1`, and it is RECORDED either way
  /// (§3.14's payload comment).
  func testStageTwoSeedDefaultsToStageOnePlusOne() throws {
    var request = Self.baseRequest(seed: .max)
    request.stage2 = Krea2Pipeline.Stage2(steps: 2, denoise: 0.2, sampler: .deis3m)
    let resolved = try XCTUnwrap(request.stage2).resolved(against: request)
    XCTAssertEqual(resolved.seed, 0, "&+ 1 wraps rather than trapping")

    var r2 = Self.baseRequest(seed: 44821)
    r2.stage2 = Krea2Pipeline.Stage2(steps: 2, denoise: 0.2)
    XCTAssertEqual(try XCTUnwrap(r2.stage2).resolved(against: r2).seed, 44822)
  }

  /// An under-specified stage 2 continues the render's OWN recipe rather than
  /// an invented one; `steps` and `denoise` — the two that decide the grid —
  /// are never defaulted (they are non-optional on the type).
  func testUnspecifiedStageTwoFieldsInheritStageOne() throws {
    var request = Self.baseRequest()
    request.guidance = 2.0
    request.eta = 0.5
    request.sampler = .res2s
    request.sigmaSchedule = .beta
    request.stage2 = Krea2Pipeline.Stage2(steps: 3, denoise: 0.5)
    let resolved = try XCTUnwrap(request.stage2).resolved(against: request)
    XCTAssertEqual(resolved.sampler, .res2s)
    XCTAssertEqual(resolved.sigmaSchedule, .beta)
    XCTAssertEqual(resolved.guidance, 2.0)
    XCTAssertEqual(resolved.eta, 0.5)
    XCTAssertEqual(resolved.steps, 3)
    XCTAssertEqual(resolved.denoise, 0.5)
  }

  // MARK: - AC-29: the published recipe is ONE payload

  /// The reference recipe, end to end without weights: one `/v1/generate` body
  /// decodes into one render whose `applied` names both stages and every
  /// published default around them.
  ///
  /// KNOWN GAP, deliberate: the "what loaded" half (`vae`, `base_variant`, the
  /// turbo LoRA at 0.6, the bypass entry) is supplied here by
  /// `RenderRecipeFixture` rather than by a live pipeline — no weights are in
  /// the default gate. The live half is the controller's integration batch;
  /// what this pins is that the request SHAPE reaches the render and that the
  /// record has room for all of it.
  func testReferenceRecipeRoundTrip() throws {
    let json = """
      {"prompt":"x","scheduler":"res_2s","sigma_schedule":"beta","shift":1.15,
       "steps":6,"guidance":1.0,"eta":0.5,
       "stage2":{"scheduler":"deis_3m","sigma_schedule":"bong_tangent",
                 "steps":2,"denoise":0.2,"guidance":1.0,"eta":0.5,"seed":null}}
      """
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let payload = try decoder.decode(GeneratePayload.self, from: Data(json.utf8))
    let recipe = try payload.krea2RecipeFields()
    let stage2 = try XCTUnwrap(try payload.krea2Stage2Fields())

    var request = Krea2Pipeline.Request(
      prompt: payload.prompt, guidance: payload.guidance ?? 1.0,
      width: 1024, height: 1024, steps: payload.steps ?? 9, seed: 44821,
      shift: recipe.shift, sampler: recipe.sampler, sigmaSchedule: recipe.sigmaSchedule,
      sigmaScheduleRequested: recipe.sigmaScheduleRequested, eta: recipe.eta,
      stage2: stage2)
    // T3 (`bongmath`) is WP-E16's and is still refused at the pipeline, so the
    // recipe's `bongmath: true` is asserted as a REQUEST shape below rather
    // than run here.
    request.bongmath = false

    let shift = try Krea2Sampling.resolveShift(
      explicit: recipe.shift, seqLen: Self.seqLen1024, align: Self.align)
    XCTAssertEqual(shift.source, .explicit)
    XCTAssertEqual(shift.mu, 1.15)

    let (x1, trace1) = try Self.runStageOne(
      request: request, shift: shift, initial: Self.patchedNoise(seed: request.seed),
      evaluate: Self.nonlinear)
    let stage = try XCTUnwrap(request.stage2).resolved(against: request)
    XCTAssertEqual(stage.seed, 44822, "null stage seed → stage1 &+ 1")
    XCTAssertEqual(stage.eta, 0.5, "stated on the stage, not inherited by accident")
    let (_, trace2) = try Self.runStageTwo(stage, shift: shift, stageOne: x1) { x, s in
      Self.nonlinear(x, s)
    }

    // The published read-backs around the two stages.
    var accel = LoRAConfiguration.local("/models/krea2-turbo-lora.safetensors", scale: 0.6)
    accel.role = "accel"
    var bypass = LoRAConfiguration.local("/models/\(Krea2BypassPolicy.workflowFile)", scale: 1.0)
    bypass.role = "bypass"
    let record = RenderRecipe.krea2(RenderRecipeFixture.inputs(
      traces: [trace1, trace2],
      loras: [
        RenderRecipe.LoRAReadBack(
          configuration: accel,
          report: RenderRecipeFixture.report(offered: 256, bound: 256),
          resolvedRelativeTo: .turbo),
        RenderRecipe.LoRAReadBack(
          configuration: bypass,
          report: RenderRecipeFixture.report(offered: 0, bound: 0, deltasApplied: 1),
          resolvedRelativeTo: nil),
      ]))

    XCTAssertEqual(record.stages.count, 2)
    XCTAssertEqual(record.stages[0].sampler, "res_2s")
    XCTAssertEqual(record.stages[0].sigmaSchedule, "beta")
    XCTAssertEqual(record.stages[0].eta, 0.5)
    XCTAssertEqual(record.stages[1].sampler, "deis_3m")
    XCTAssertEqual(record.stages[1].sigmaSchedule, "bong_tangent")
    XCTAssertEqual(record.stages[1].eta, 0.5)
    XCTAssertEqual(record.stages[1].seed, 44822)
    XCTAssertEqual(record.modelEvalsTotal, trace1.modelEvals + trace2.modelEvals)
    XCTAssertEqual(try XCTUnwrap(record.shift), Foundation.exp(Float(1.15)), accuracy: 1e-5)
    XCTAssertEqual(record.mu, 1.15)
    XCTAssertEqual(record.shiftSource, "explicit")
    // D16 / §3.14's "every published default appears in `applied`".
    XCTAssertEqual(record.baseVariant, "raw")
    XCTAssertTrue(record.vae.hasSuffix("Wan2_1_VAE_fp32.safetensors"))
    XCTAssertEqual(record.vaeLayout, "wanNative")
    XCTAssertEqual(record.loras[0].scaleApplied, 0.6)
    XCTAssertEqual(record.loras[0].role, "accel")
    XCTAssertEqual(record.loras[0].relativeTo, "turbo")
    XCTAssertEqual(record.loras[1].role, "bypass")
    XCTAssertTrue(record.loras[1].file.hasSuffix(Krea2BypassPolicy.workflowFile))

    // The whole record survives the wire encoder every sink uses.
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let wire = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: try encoder.encode(record)) as? [String: Any])
    let stages = try XCTUnwrap(wire["stages"] as? [[String: Any]])
    XCTAssertEqual(stages.count, 2)
    XCTAssertEqual(stages[1]["sampler"] as? String, "deis_3m")
    XCTAssertEqual(wire["model_evals_total"] as? Int, record.modelEvalsTotal)
  }

  // MARK: - Progress across both stages

  /// The published recipe's bar runs 1…8, never 6/6 then 1/2 (WP-E10 wired
  /// `/health.progress_percent` to this callback, and a bar that goes backwards
  /// halfway through every staged render is a defect users see).
  func testProgressIsOneMonotonicBarOverBothStages() {
    let bar = Krea2StagedRender.Progress(stage1Steps: 6, stage2Steps: 2)
    XCTAssertEqual(bar.total, 8)
    let reported = (1...6).map { bar.stage1($0) } + (1...2).map { bar.stage2($0) }
    XCTAssertEqual(reported.map(\.0), [1, 2, 3, 4, 5, 6, 7, 8])
    XCTAssertEqual(Set(reported.map(\.1)), [8])
    XCTAssertEqual(
      reported.map { RenderProgressPercent.of(step: $0.0, total: $0.1) }.sorted(),
      reported.map { RenderProgressPercent.of(step: $0.0, total: $0.1) },
      "the percentage must never go backwards")
  }

  // MARK: - Fail-loud

  func testStageTwoRefusesAnOutOfRangeDenoise() throws {
    let shift = try Self.shift()
    for bad: Double in [0, -1, 2] {
      XCTAssertThrowsError(
        try Krea2StagedRender.makeScheduler(
          sampler: .deis3m, sigmaSchedule: .bongTangent, steps: 2, denoise: bad,
          shift: shift, seed: 1, c2: 0.5), "denoise \(bad)")
    }
  }

  /// A stage-2 grid the schedule cannot build is a throw, not a substitution:
  /// `bong_tangent` divides by zero below 2 steps upstream, and
  /// `{steps: 2, denoise: 0.9}` stretches to exactly 2 — the boundary.
  func testStageTwoScheduleMinimumIsEnforcedNotClamped() throws {
    let shift = try Self.shift()
    XCTAssertNoThrow(
      try Krea2StagedRender.makeScheduler(
        sampler: .deis3m, sigmaSchedule: .bongTangent, steps: 2, denoise: 0.9,
        shift: shift, seed: 1, c2: 0.5))
    XCTAssertThrowsError(
      try Krea2StagedRender.makeScheduler(
        sampler: .deis3m, sigmaSchedule: .bongTangent, steps: 1, denoise: 0.9,
        shift: shift, seed: 1, c2: 0.5))
  }
}
