import XCTest
import MLX
@testable import ZImage

/// WP-E3 — `Krea2RunTrace` (FDD-krea2-raw-recipe §3.3, §3.10).
///
/// The trace is what `generateWithRecipe` / `generateImg2ImgWithRecipe` HAND
/// the caller; WP-E10 maps it into `RenderRecipe.Stage`. Weight-free: every
/// input (request, resolved shift, scheduler, loop stats) is constructible
/// without a model, which is the point of keeping the trace a plain value.
final class Krea2RunTraceTests: XCTestCase {

  static let align = 16
  static let seqLen1024 = (1024 / 16) * (1024 / 16)

  static func shift(explicit: Float? = nil, seqLen: Int = seqLen1024) throws
    -> Krea2Sampling.ScheduleShift
  {
    try Krea2Sampling.resolveShift(explicit: explicit, seqLen: seqLen, align: align)
  }

  static func stats(stepsRun: Int, rowsAtStart: Int = 1, modelEvals: Int)
    -> Krea2DenoiseLoop.Stats
  {
    Krea2DenoiseLoop.Stats(
      stepsRun: stepsRun, rowsAtStart: rowsAtStart,
      evaluateCalls: stepsRun * rowsAtStart, modelEvals: modelEvals)
  }

  // MARK: - Text-to-image

  func testT2ITraceReportsTheGridTheLoopWalked() throws {
    let shift = try Self.shift()
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 44821, c2: 0.5)
    let request = Krea2Pipeline.Request(prompt: "x", steps: 9, seed: 44821)
    let trace = Krea2RunTrace(
      request: request, shift: shift, scheduler: scheduler,
      stats: Self.stats(stepsRun: 9, modelEvals: 9),
      startIndex: 0, denoise: 1.0, width: 1024, height: 1024,
      negativePromptApplied: Krea2RunTrace.negativePromptApplied(
        cfgActive: request.guidance > 1.0, requested: request.negativePrompt))

    XCTAssertEqual(trace.sampler, .euler)
    XCTAssertEqual(trace.sigmaSchedule, .krea2)
    XCTAssertNil(trace.sigmaScheduleRequested)
    XCTAssertEqual(trace.sigmas, scheduler.sigmas.asArray(Float.self))
    XCTAssertEqual(trace.sigmas.count, 10)
    XCTAssertEqual(trace.sigmas.last, 0.0)
    XCTAssertEqual(trace.stepsRequested, 9)
    XCTAssertEqual(trace.stepsEffective, 9)
    XCTAssertEqual(trace.stepsRun, 9)
    XCTAssertEqual(trace.modelEvals, 9)
    XCTAssertEqual(trace.startIndex, 0)
    XCTAssertEqual(trace.denoise, 1.0)
    XCTAssertEqual(trace.guidance, 1.0)
    XCTAssertEqual(trace.eta, 0)
    XCTAssertNil(trace.sde, "eta 0 renders carry no SDE block (WP-E15)")
    XCTAssertFalse(trace.bongmath)
    XCTAssertNil(trace.bong, "bongmath-off renders carry no T3 block (WP-E16)")
    XCTAssertEqual(trace.seed, 44821)
    XCTAssertEqual(trace.width, 1024)
    XCTAssertEqual(trace.height, 1024)
  }

  /// WP-E15 (§3.13, scope 4): a render that ran the T2 SDE records what it ran
  /// with — including the two stream seeds, which are the only part of the SDE
  /// the request does not state and which are what makes AC-27's "same
  /// payload, same output" checkable from the record.
  func testT2ITraceRecordsTheSDEWhenEtaRan() throws {
    let shift = try Self.shift()
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .res2s, sigmaSchedule: .beta, steps: 6, shift: shift, seed: 4242, c2: 0.5)
    var request = Krea2Pipeline.Request(prompt: "x", steps: 6, seed: 4242)
    request.sampler = .res2s
    request.sigmaSchedule = .beta
    request.eta = 0.5
    let trace = Krea2RunTrace(
      request: request, shift: shift, scheduler: scheduler,
      stats: Self.stats(stepsRun: 6, rowsAtStart: 2, modelEvals: 12),
      startIndex: 0, denoise: 1.0, width: 1024, height: 1024,
      negativePromptApplied: nil)

    XCTAssertEqual(trace.eta, 0.5)
    let sde = try XCTUnwrap(trace.sde)
    XCTAssertEqual(sde.eta, 0.5)
    XCTAssertEqual(sde.etaSubstep, 0.5, "ClownsharKSampler sets eta_substep = eta")
    XCTAssertEqual(sde.noiseMode, "hard")
    XCTAssertEqual(sde.sNoise, 1.0)
    XCTAssertEqual(sde.stepNoiseSeed, 4243, "seed + 1, as SharkSampler derives it")
    XCTAssertEqual(sde.substepNoiseSeed, 4243 + 10_000, "+ MAX_STEPS")
    // The record and the injector agree by construction, not by comment.
    XCTAssertEqual(
      sde.stepNoiseSeed, RES4LYFSDENoiseInjector.stepNoiseSeed(stageSeed: request.seed))
  }

  /// D3 as amended by A.1: `mu` is the shift, the recorded `shift` is `e^mu`,
  /// and the source says which of the two ways it was arrived at.
  func testShiftSourceAndEffectiveShift() throws {
    let dynamic = try Self.shift()
    let explicit = try Self.shift(explicit: 1.15)
    for (shift, expectedSource) in [(dynamic, "dynamic"), (explicit, "explicit")] {
      let scheduler = try Krea2Pipeline.makeScheduler(
        sampler: .euler, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 1, c2: 0.5)
      let request = Krea2Pipeline.Request(
        prompt: "x", steps: 9, shift: shift.source == .explicit ? 1.15 : nil)
      let trace = Krea2RunTrace(
        request: request,
        shift: shift, scheduler: scheduler, stats: Self.stats(stepsRun: 9, modelEvals: 9),
        startIndex: 0, denoise: 1.0, width: 1024, height: 1024,
        negativePromptApplied: Krea2RunTrace.negativePromptApplied(
          cfgActive: request.guidance > 1.0, requested: request.negativePrompt))
      XCTAssertEqual(trace.shiftSource, expectedSource)
      XCTAssertEqual(trace.mu, shift.mu)
      XCTAssertEqual(trace.shift, Foundation.exp(shift.mu), accuracy: 1e-6)
    }
    XCTAssertEqual(explicit.mu, 1.15, "A.1: shift IS mu, not log(shift)")
  }

  /// The width/height on the trace are the POST-round-up numbers the render
  /// actually used, not the request's (`Krea2Pipeline.generate` rounds to the
  /// VAE scale × patch = 16 before anything else).
  func testTraceCarriesRoundedGeometry() throws {
    let shift = try Self.shift(seqLen: (1040 / 16) * (1024 / 16))
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 1, c2: 0.5)
    let request = Krea2Pipeline.Request(prompt: "x", width: 1030, height: 1024, steps: 9)
    let rounded = Krea2Sampling.roundUp(request.width, multiple: 16)
    XCTAssertEqual(rounded, 1040)
    let trace = Krea2RunTrace(
      request: request, shift: shift, scheduler: scheduler,
      stats: Self.stats(stepsRun: 9, modelEvals: 9),
      startIndex: 0, denoise: 1.0, width: rounded, height: 1024,
      negativePromptApplied: Krea2RunTrace.negativePromptApplied(
        cfgActive: request.guidance > 1.0, requested: request.negativePrompt))
    XCTAssertEqual(trace.width, 1040)
  }

  // MARK: - The alias is visible, never silent (D22)

  func testRequestedScheduleNameIsRecordedOnlyWhenItIsAnAlias() throws {
    let shift = try Self.shift()
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .flow, steps: 9, shift: shift, seed: 1, c2: 0.5)
    func trace(requested: String?) -> Krea2RunTrace {
      let request = Krea2Pipeline.Request(
        prompt: "x", steps: 9, sigmaSchedule: .flow, sigmaScheduleRequested: requested)
      return Krea2RunTrace(
        request: request,
        shift: shift, scheduler: scheduler, stats: Self.stats(stepsRun: 9, modelEvals: 9),
        startIndex: 0, denoise: 1.0, width: 1024, height: 1024,
        negativePromptApplied: Krea2RunTrace.negativePromptApplied(
          cfgActive: request.guidance > 1.0, requested: request.negativePrompt))
    }
    // Krita's default style sends "normal", which resolves to flow.
    XCTAssertEqual(trace(requested: "normal").sigmaScheduleRequested, "normal")
    // The resolved kind's own spelling is not worth recording twice.
    XCTAssertNil(trace(requested: "flow").sigmaScheduleRequested)
    XCTAssertNil(trace(requested: nil).sigmaScheduleRequested)
  }

  // MARK: - Steps: requested vs the grid's own count (D5, AC-22)

  /// `beta`/`beta57` de-duplicate colliding table indices, so the grid can be
  /// shorter than the request. The trace reports BOTH — the request's number
  /// and the count the loop actually walked — rather than conflating them.
  ///
  /// Pinned with the 1000-entry DiscreteFlow table, where the first collision
  /// is at `beta57(97) → 96` (`BetaScheduleComfyParityTests`). On Krea 2's own
  /// 10 000-entry Flux table the two agree at every production budget, so a
  /// krea2-config case could not tell a conflation from a correct mapping.
  func testStepsEffectiveIsTheGridsCountNotTheRequests() throws {
    let config = ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: 1.0, useDynamicShifting: false)
    let scheduler = try SchedulerFactory.create(
      kind: .euler, sigmaSchedule: .beta57, numInferenceSteps: 97, config: config)
    XCTAssertEqual(scheduler.numInferenceSteps, 96, "precondition: this grid de-dups")

    let request = Krea2Pipeline.Request(prompt: "x", steps: 97, sigmaSchedule: .beta57)
    let trace = Krea2RunTrace(
      request: request,
      shift: try Self.shift(), scheduler: scheduler,
      stats: Self.stats(stepsRun: 96, modelEvals: 96),
      startIndex: 0, denoise: 1.0, width: 1024, height: 1024,
      negativePromptApplied: Krea2RunTrace.negativePromptApplied(
        cfgActive: request.guidance > 1.0, requested: request.negativePrompt))
    XCTAssertEqual(trace.stepsRequested, 97)
    XCTAssertEqual(trace.stepsEffective, 96)
    XCTAssertEqual(trace.stepsRun, 96)
    XCTAssertEqual(trace.sigmas.count, 97)
  }

  // MARK: - img2img

  /// The img2img trace records where the partial denoise started and how much
  /// of the schedule it therefore ran.
  func testImg2ImgTraceRecordsStartIndexAndDenoise() throws {
    let shift = try Self.shift()
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 7, c2: 0.5)
    // The pipeline's own arithmetic at strength 0.3.
    let denoise = 1.0 - Float(0.3)
    let startIndex = max(0, 9 - Int((Double(9) * Double(denoise)).rounded(.up)))
    XCTAssertEqual(startIndex, 2)

    let request = Krea2Pipeline.Img2ImgRequest(
      prompt: "x", sourceImage: MLX.zeros([1, 64, 64, 3]), steps: 9, seed: 7, strength: 0.3)
    let trace = Krea2RunTrace(
      request: request, shift: shift, scheduler: scheduler,
      stats: Self.stats(stepsRun: 9 - startIndex, modelEvals: 9 - startIndex),
      startIndex: startIndex, denoise: denoise, width: 1024, height: 1024,
      negativePromptApplied: Krea2RunTrace.negativePromptApplied(
        cfgActive: request.guidance > 1.0, requested: request.negativePrompt))

    XCTAssertEqual(trace.startIndex, 2)
    XCTAssertEqual(trace.denoise, denoise)
    XCTAssertEqual(trace.stepsRequested, 9)
    XCTAssertEqual(trace.stepsEffective, 9)
    XCTAssertEqual(trace.stepsRun, 7)
    XCTAssertEqual(trace.modelEvals, 7)
    XCTAssertEqual(trace.seed, 7)
  }

  /// §3.3's cost line: a `res_2s` + CFG render is 4x a plain euler one, and
  /// the trace carries that number so it is reported rather than discovered.
  ///
  /// The number itself comes from `Stats.modelEvals`, which the loop COUNTS.
  /// `stepsRun × rowsAtStart × cfg` happens to equal it here only because
  /// `res_2s` takes 2 rows on every step; `Krea2DenoiseLoopTests`'
  /// `testRowsMayChangeMidRunAndModelEvalsCountsTheActualCalls` pins the case
  /// where it does not.
  func testTraceCarriesTheMultiplicativeEvalCost() throws {
    let shift = try Self.shift()
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .res2s, sigmaSchedule: .krea2, steps: 6, shift: shift, seed: 1, c2: 0.5)
    let request = Krea2Pipeline.Request(prompt: "x", guidance: 2.0, steps: 6, sampler: .res2s)
    let trace = Krea2RunTrace(
      request: request,
      shift: shift, scheduler: scheduler,
      stats: Self.stats(stepsRun: 6, rowsAtStart: 2, modelEvals: 6 * 2 * 2),
      startIndex: 0, denoise: 1.0, width: 1024, height: 1024,
      negativePromptApplied: Krea2RunTrace.negativePromptApplied(
        cfgActive: request.guidance > 1.0, requested: request.negativePrompt))
    XCTAssertEqual(trace.modelEvals, 24)
    XCTAssertEqual(trace.guidance, 2.0)
    XCTAssertEqual(trace.sampler, .res2s)
  }

  /// WP-E16 (§3.13, scope 1/4): a render that ran the T3 fixed point records
  /// what it DID — the rounds, the rebases, the steps it ran on, the rows
  /// upstream's guards refused, and the extra model evaluations it made.
  ///
  /// The counts are the hook's own, so the record cannot claim a fixed point
  /// that never ran: `bongmath: true` in the request is a request, and
  /// upstream's guards refuse two of a 6-step `res_2s` grid's steps outright.
  func testTraceRecordsWhatTheT3FixedPointDid() throws {
    let hook = RES4LYFBongMath(sigmaMin: 3.1575115281157196e-4, sigmaMax: 1.0)
    XCTAssertNil(
      Krea2RunTrace.BongMathParameters.forRun(nil), "no hook, no block")

    var grid: any ZImageScheduler = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T3")
    let m = trace.manifest
    let stepNoise = try m.steps.compactMap { $0.noiseStep }.map { try trace.tensor($0) }
    let substepNoise = try m.steps.flatMap { $0.substeps }.compactMap { $0.noise }
      .map { try trace.tensor($0) }
    _ = try Krea2DenoiseLoop.run(
      scheduler: &grid, initialSample: try trace.tensor(m.xInit),
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: RES4LYFSDENoiseInjector(
        eta: m.recipe.eta, etaSubstep: m.recipe.etaSubstep, sNoise: 1.0, sNoiseSubstep: 1.0,
        sigmaMax: m.sigmaMax,
        stepNoise: RES4LYFEtaSDEParityTests.RecordedNoiseStream(
          name: "step", tensors: stepNoise),
        substepNoise: RES4LYFEtaSDEParityTests.RecordedNoiseStream(
          name: "substep", tensors: substepNoise)),
      bongmath: hook)

    let record = try XCTUnwrap(Krea2RunTrace.BongMathParameters.forRun(hook))
    XCTAssertEqual(record.iterations, 100, "upstream's `for i in range(100)`")
    XCTAssertEqual(record.rebases, 4)
    XCTAssertEqual(record.steps, [0, 1, 2, 3], "steps 4 and 5 are refused by h ≥ σ_max/2")
    XCTAssertEqual(record.refusals, 2)
    XCTAssertEqual(record.extraModelEvals, 0, "the fixed point is algebraic")
  }
}
