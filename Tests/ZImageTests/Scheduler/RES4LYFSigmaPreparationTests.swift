import Foundation
import MLX
import XCTest

@testable import ZImage

/// Task S-FIX-1 — RES4LYF's `prepare_sigmas` and the model-free `σ_min → 0`
/// conversion, on the PRODUCTION Krea 2 path.
///
/// Two halves:
///   * the pure grid rewrite, against upstream's three branches
///     (`rk_noise_sampler_beta.py:916`) and its step-count rule
///     (`rk_sampler_beta.py:480`, `num_steps = len(sigmas) − 2`);
///   * the containment proofs — that the default `euler` + `krea2` path and the
///     Z-Image `res_2s` path are BYTE-for-byte where they were.
final class RES4LYFSigmaPreparationTests: XCTestCase {

  /// Krea 2's registered `ModelSamplingFlux(shift = 1.15)` σ_min: 3.1575e-4
  /// (`comfy_sigmas.json` → `model_samplings.flux.sigma_min`, Addendum A.1).
  static let fluxSigmaMin = SigmaSchedule.fluxSigmaTable(
    shift: 1.15, tableSize: Krea2Sampling.fluxTableSize)[0]

  // MARK: - The grid rewrite, branch by branch

  func testSigmaMinMatchesTheOracleTable() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let samplings = try XCTUnwrap(fixture["model_samplings"] as? [String: Any])
    let flux = try XCTUnwrap(samplings["flux"] as? [String: Any])
    XCTAssertEqual(Double(Self.fluxSigmaMin), try XCTUnwrap(flux["sigma_min"] as? Double), accuracy: 1e-12)
  }

  /// The common branch: the penultimate sigma is far above σ_min, so σ_min is
  /// INSERTED before the trailing zero. The grid gains an entry, the zero is
  /// dropped as a sentinel, and the solver step count is UNCHANGED — which is
  /// how upstream runs the six steps a six-step request asked for.
  func testInsertBranchKeepsTheRequestedStepCount() {
    // `beta` at 6 steps under the Flux table, from `comfy_sigmas.json`.
    let published: [Float] = [
      1.0, 0.9690954089164734, 0.8925824165344238, 0.7595839500427246,
      0.545648992061615, 0.24154026806354523, 0.0,
    ]
    let prepared = RES4LYFSigmaPreparation.prepare(
      published: published, sigmaMin: Self.fluxSigmaMin)

    XCTAssertEqual(prepared.numInferenceSteps, 6, "6 published steps stay 6 solver steps")
    XCTAssertEqual(prepared.solverSigmas.count, published.count)
    XCTAssertEqual(Array(prepared.solverSigmas.prefix(6)), Array(published.prefix(6)),
      "everything above the tail is untouched")
    XCTAssertEqual(prepared.solverSigmas.last, Self.fluxSigmaMin)
    XCTAssertEqual(prepared.finalConversionSigma, Self.fluxSigmaMin)
    XCTAssertEqual(prepared.walkedGrid, prepared.solverSigmas + [0.0])
    XCTAssertNotEqual(prepared.solverSigmas.last, 0.0, "the solver never steps onto the sentinel")
  }

  /// `if sigmas[-2] < SIGMA_MIN: sigmas[-2] = SIGMA_MIN` — a REPLACE, so the
  /// grid keeps its length and therefore LOSES a solver step relative to the
  /// published schedule. That is upstream's arithmetic, not an approximation
  /// of it.
  func testReplaceBranchWhenThePenultimateIsBelowSigmaMin() {
    let published: [Float] = [1.0, 0.5, 0.2, 1e-6, 0.0]
    let prepared = RES4LYFSigmaPreparation.prepare(
      published: published, sigmaMin: Self.fluxSigmaMin)

    XCTAssertEqual(prepared.solverSigmas, [1.0, 0.5, 0.2, Self.fluxSigmaMin])
    XCTAssertEqual(prepared.numInferenceSteps, 3, "4 published steps become 3 (len − 2)")
    XCTAssertEqual(prepared.finalConversionSigma, Self.fluxSigmaMin)
  }

  /// `elif (sigmas[-2] − SIGMA_MIN).abs() > 1e-4` — a penultimate already
  /// within the tolerance is left EXACTLY as it is, and because upstream's tail
  /// guard is `sigmas[-2] == NS.sigma_min` there is then no conversion either.
  func testAlreadyCloseEnoughBranchInsertsNothingAndConvertsNothing() {
    let nearby = Self.fluxSigmaMin + 5e-5  // inside the 1e-4 window
    let published: [Float] = [1.0, 0.5, 0.2, nearby, 0.0]
    let prepared = RES4LYFSigmaPreparation.prepare(
      published: published, sigmaMin: Self.fluxSigmaMin)

    XCTAssertEqual(prepared.solverSigmas, [1.0, 0.5, 0.2, nearby])
    XCTAssertNil(prepared.finalConversionSigma, "upstream's tail needs sigmas[-2] == σ_min exactly")
    XCTAssertEqual(prepared.walkedGrid, prepared.solverSigmas, "nothing lands on 0")
  }

  /// Just outside the window: back to the insert branch.
  func testJustOutsideTheToleranceInsertsSigmaMin() {
    let outside = Self.fluxSigmaMin + 2e-4
    let prepared = RES4LYFSigmaPreparation.prepare(
      published: [1.0, 0.5, outside, 0.0], sigmaMin: Self.fluxSigmaMin)
    XCTAssertEqual(prepared.solverSigmas, [1.0, 0.5, outside, Self.fluxSigmaMin])
    XCTAssertEqual(prepared.finalConversionSigma, Self.fluxSigmaMin)
  }

  /// `consecutive_duplicate_mask` (`rk_noise_sampler_beta.py:912`), which is
  /// what keeps a repeated sigma from producing `h = 0`.
  func testConsecutiveDuplicatesAreDropped() {
    let prepared = RES4LYFSigmaPreparation.prepare(
      published: [1.0, 1.0, 0.5, 0.5, 0.2, 0.0], sigmaMin: Self.fluxSigmaMin)
    XCTAssertEqual(prepared.solverSigmas, [1.0, 0.5, 0.2, Self.fluxSigmaMin])
  }

  /// No trailing zero: upstream prepares nothing and runs no tail.
  func testGridWithoutATrailingZeroIsLeftAlone() {
    let published: [Float] = [1.0, 0.5, 0.2]
    let prepared = RES4LYFSigmaPreparation.prepare(
      published: published, sigmaMin: Self.fluxSigmaMin)
    XCTAssertEqual(prepared.solverSigmas, published)
    XCTAssertNil(prepared.finalConversionSigma)
    XCTAssertEqual(prepared.numInferenceSteps, 2, "len − 1 when there is no sentinel")
  }

  /// Degenerate: dropping the zero would leave no step to take. Every scheduler
  /// here preconditions on at least one step, so the schedule is left as it is
  /// rather than handed on as an empty solver.
  func testDegenerateOneStepBelowSigmaMinIsNotPrepared() {
    let prepared = RES4LYFSigmaPreparation.prepare(
      published: [1e-6, 0.0], sigmaMin: Self.fluxSigmaMin)
    XCTAssertEqual(prepared.solverSigmas, [1e-6, 0.0])
    XCTAssertNil(prepared.finalConversionSigma)
  }

  // MARK: - The two real schedules that already end at σ_min

  /// `karras` and `exponential` both take their floor FROM the model table's
  /// σ_min, so their penultimate sigma lands within a float32 ulp of it — well
  /// inside upstream's 1e-4 window. Neither ever gets an insertion, so both
  /// run `len(sigmas) − 2` steps: one fewer than the request, exactly as
  /// RES4LYF runs them.
  ///
  /// Which side of the ulp they land on then decides the tail, and upstream's
  /// two branches disagree on purpose: BELOW σ_min is a replace (the grid ends
  /// on σ_min exactly, so the tail fires); at or just above it is left alone
  /// (the tail's `sigmas[-2] == NS.sigma_min` guard does not fire). Both
  /// outcomes occur across ordinary `mu` values, so this pins the RULE and not
  /// one side of a rounding coin flip.
  func testKarrasAndExponentialAreNeverGivenAnInsertedSigmaMin() throws {
    let config = Krea2Sampling.schedulerConfig()
    for mu: Float in [1.15, 0.8977, 0.5] {
      let sigmaMin = SigmaSchedule.fluxSigmaTable(
        shift: mu, tableSize: Krea2Sampling.fluxTableSize)[0]
      for schedule in [SigmaScheduleKind.karras, .exponential] {
        let what = "\(schedule.rawValue) @ mu \(mu)"
        let published = try SchedulerFactory.resolveSigmas(
          schedule: schedule, numSteps: 9, config: config, mu: mu)
        XCTAssertEqual(published.count, 10, what)
        XCTAssertEqual(published.last, 0.0, what)
        XCTAssertEqual(
          Double(published[8]), Double(sigmaMin),
          accuracy: Double(RES4LYFSigmaPreparation.sigmaMinMatchTolerance),
          "\(what): the floor already sits inside upstream's window")

        let scheduler = try SchedulerFactory.create(
          kind: .res2s, sigmaSchedule: schedule, numInferenceSteps: 9,
          config: config, mu: mu, res4lyfSigmaPreparation: true)
        XCTAssertEqual(scheduler.numInferenceSteps, 8, "\(what): upstream's len − 2, no insertion")

        let last = try XCTUnwrap(scheduler.sigmas.asArray(Float.self).last)
        if published[8] < sigmaMin {
          XCTAssertEqual(last, sigmaMin, "\(what): replaced, so it ends ON σ_min")
          XCTAssertEqual(scheduler.finalConversionSigma, sigmaMin, "\(what): and the tail fires")
        } else {
          XCTAssertEqual(last, published[8], "\(what): left exactly as published")
          XCTAssertNil(
            scheduler.finalConversionSigma,
            "\(what): upstream's tail guard is an exact `== σ_min`")
        }
      }
    }
  }

  // MARK: - Which samplers this engages for

  func testOnlyTheRES4LYFPortsDeclareTheFamily() {
    XCTAssertTrue(SchedulerKind.res2s.isRES4LYFFamily)
    XCTAssertTrue(SchedulerKind.res3s.isRES4LYFFamily)
    XCTAssertTrue(SchedulerKind.ralston2s.isRES4LYFFamily)
    XCTAssertTrue(SchedulerKind.ralston3s.isRES4LYFFamily)
    XCTAssertTrue(SchedulerKind.ralston4s.isRES4LYFFamily)
    // `.deis` is the k-diffusion multistep port today; WP-E14 replaces it.
    XCTAssertFalse(SchedulerKind.deis.isRES4LYFFamily)
    for kind in [SchedulerKind.euler, .heun, .dpmplusplus2m, .dpmplusplus2sa, .ddim] {
      XCTAssertFalse(kind.isRES4LYFFamily, "\(kind.rawValue)")
    }
  }

  // MARK: - Containment: the default path does not move

  /// AC-1/AC-2's grid half. `euler` is not in the RES4LYF family, so asking the
  /// factory for preparation is a NO-OP: the same grid, element for element,
  /// and no conversion sigma for the loop to apply.
  func testDefaultEulerKrea2GridIsIdenticalWithPreparationOn() throws {
    let config = Krea2Sampling.schedulerConfig()
    for steps in [4, 6, 9, 12, 20] {
      for mu: Float in [0.5, 1.15, 0.87] {
        let off = try SchedulerFactory.create(
          kind: .euler, sigmaSchedule: .krea2, numInferenceSteps: steps,
          config: config, mu: mu, res4lyfSigmaPreparation: false)
        let on = try SchedulerFactory.create(
          kind: .euler, sigmaSchedule: .krea2, numInferenceSteps: steps,
          config: config, mu: mu, res4lyfSigmaPreparation: true)

        XCTAssertEqual(
          on.sigmas.asArray(Float.self), off.sigmas.asArray(Float.self),
          "steps \(steps) mu \(mu): the default grid moved")
        XCTAssertEqual(on.numInferenceSteps, steps)
        XCTAssertEqual(on.sigmas.asArray(Float.self).last, 0.0, "the krea2 warp's own 0 sentinel")
        XCTAssertNil(on.finalConversionSigma, "euler takes no RES4LYF tail")
        XCTAssertNil(off.finalConversionSigma)
      }
    }
  }

  /// …and the same through the real call site, which passes preparation `true`.
  func testKrea2MakeSchedulerDefaultPathHasNoConversion() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: 4096, align: 16)
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 0, c2: 0.5)
    XCTAssertNil(scheduler.finalConversionSigma)
    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(
      scheduler.sigmas.asArray(Float.self),
      SigmaSchedule.krea2(numSteps: 9, mu: shift.mu),
      "the default grid IS `SigmaSchedule.krea2`, unprepared")
  }

  /// Task item 5: the Z-Image pipelines pass no preparation flag, so their
  /// `res_2s` grid is exactly what it was — the gap is real there, and is
  /// ticketed rather than fixed here.
  func testZImageRES2sGridIsUnchangedBecauseItDoesNotOptIn() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let asZImageBuildsIt = try SchedulerFactory.create(
      kind: .res2s, sigmaSchedule: .karras, numInferenceSteps: 8, config: config)
    let grid = asZImageBuildsIt.sigmas.asArray(Float.self)

    XCTAssertEqual(grid.count, 9)
    XCTAssertEqual(grid.last, 0.0, "still solving straight through the 0 sentinel")
    XCTAssertNil(asZImageBuildsIt.finalConversionSigma, "and taking no model-free tail")
    XCTAssertEqual(
      grid,
      try SchedulerFactory.resolveSigmas(
        schedule: .karras, numSteps: 8, config: config, mu: nil),
      "identical to the published schedule")
  }

  // MARK: - The conversion is model-free

  /// The `σ_min → 0` conversion must cost no `evaluate` call, and must not be
  /// counted as a step. Driven with a scripted field so the counts are exact.
  func testConversionCostsNoModelEvaluationAndIsNotAStep() throws {
    var scheduler = try SchedulerFactory.create(
      kind: .res2s, sigmaSchedule: .beta, numInferenceSteps: 6,
      config: Krea2Sampling.schedulerConfig(), mu: 1.15, c2: 0.5,
      res4lyfSigmaPreparation: true)
    XCTAssertEqual(scheduler.finalConversionSigma, Self.fluxSigmaMin)

    var sigmasSeen: [Float] = []
    let x0 = MLXArray.ones([1, 4, 4]).asType(.float32) * 0.3
    let (_, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: x0, modelEvalsPerEvaluate: 2
    ) { latent, sigma in
      sigmasSeen.append(sigma)
      return RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    XCTAssertEqual(stats.stepsRun, 6, "the conversion is not a seventh step")
    XCTAssertEqual(stats.evaluateCalls, 12, "2 rows × 6 steps, and nothing for the tail")
    XCTAssertEqual(stats.modelEvals, 24, "CFG doubles the 12; the tail still adds none")
    XCTAssertEqual(sigmasSeen.count, 12)
    XCTAssertEqual(stats.finalConversionSigma, Self.fluxSigmaMin)
    // Nothing was ever evaluated at 0 — or below σ_min.
    XCTAssertEqual(sigmasSeen.filter { $0 <= 0 }.count, 0)
    XCTAssertGreaterThanOrEqual(sigmasSeen.min() ?? 0, Self.fluxSigmaMin)
  }

  // MARK: - Provenance reports the grid as actually walked

  /// `Krea2RunTrace` must describe the run, not the published schedule: the
  /// solver sigmas ending on σ_min, plus the 0 the model-free conversion landed
  /// on, and the conversion sigma itself.
  func testRunTraceRecordsThePreparedGridAndTheConversion() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: 4096, align: 16)
    var scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .res2s, sigmaSchedule: .beta, steps: 6, shift: shift, seed: 7, c2: 0.5)

    let x0 = MLXArray.ones([1, 4, 4]).asType(.float32) * 0.3
    let (_, stats) = Krea2DenoiseLoop.run(scheduler: &scheduler, initialSample: x0) {
      latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    let request = Krea2Pipeline.Request(
      prompt: "x", steps: 6, seed: 7, sampler: .res2s, sigmaSchedule: .beta)
    let trace = Krea2RunTrace(
      request: request, shift: shift, scheduler: scheduler, stats: stats,
      startIndex: 0, denoise: 1.0, width: 1024, height: 1024)

    XCTAssertEqual(trace.finalConversionSigma, Self.fluxSigmaMin)
    XCTAssertEqual(trace.stepsRun, 6)
    XCTAssertEqual(trace.stepsEffective, 6)
    XCTAssertEqual(trace.modelEvals, 6 * 2, "2 rows × 6 steps; the conversion adds none")
    // The walked grid: 7 solver sigmas ending on σ_min, then the conversion's 0.
    XCTAssertEqual(trace.sigmas.count, 8)
    XCTAssertEqual(trace.sigmas.dropLast(), scheduler.sigmas.asArray(Float.self)[...])
    XCTAssertEqual(trace.sigmaTail[1], Self.fluxSigmaMin)
    XCTAssertEqual(trace.sigmaTail[2], 0.0, "the conversion's landing point is in the record")
    XCTAssertEqual(trace.sigmaTail[0], 0.24154027, accuracy: 1e-6, "beta's last published sigma")
    XCTAssertEqual(trace.sigmaHead[0], 1.0)
    XCTAssertEqual(trace.sigmaHead[1], 0.9690954, accuracy: 1e-6)
  }

  /// A `.flux` model sampling has no σ_min without its `mu`, and `1e-8` is not
  /// a substitute — the factory refuses rather than silently floors.
  func testPreparationWithoutMuOnAFluxFamilyIsRefused() {
    XCTAssertThrowsError(
      try SchedulerFactory.create(
        kind: .res2s, sigmaSchedule: .bongTangent, numInferenceSteps: 6,
        config: Krea2Sampling.schedulerConfig(), mu: nil,
        res4lyfSigmaPreparation: true)
    ) { error in
      XCTAssertEqual(error as? SchedulerFactoryError, .missingMu(.bongTangent))
    }
  }
}
