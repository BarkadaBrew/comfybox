import XCTest
@testable import ZImage

/// WP-E17 — the stage-2 sigma grid (FDD-krea2-raw-recipe §3.14 step 2, AC-31).
///
/// Weight-free and GPU-free: the whole claim is arithmetic over the published
/// schedules, so it runs in the default gate.
///
/// The claim under test is that stage 2's grid is a **stretch-and-tail**, not a
/// truncation of stage 1's: RES4LYF's `get_sigmas`
/// (`res4lyf_sigmas.py:1397-1429`, truncation at `:1402`) does
/// `total = int(steps/denoise)` → build the schedule at `total` →
/// `sigmas[-(steps+1):]`. So `{steps: 2, denoise: 0.2, bong_tangent}` starts at
/// σ ≈ 0.117, not at 0.2 and not at stage 1's ninth sigma.
final class Krea2StagedSigmaTests: XCTestCase {

  static let align = 16
  static let seqLen1024 = (1024 / 16) * (1024 / 16)

  static func shift() throws -> Krea2Sampling.ScheduleShift {
    try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: seqLen1024, align: align)
  }

  // MARK: - AC-31: the stretched tail

  /// The published recipe's stage 2, to the value: `bong_tangent(10)[-3:]`.
  ///
  /// Pinned against upstream source (`res4lyf_sigmas.py:1402` truncation +
  /// `:4065-4098` bong_tangent, pure-Python re-run 2026-08-22), not against
  /// this port's own output — and explicitly NOT `[0.2, 0.1, 0.0]`, which is
  /// what reading `denoise` as "start at σ = denoise" would produce.
  func testStretchAndTailAtThePublishedRecipe() throws {
    let sigmas = try Krea2StagedRender.publishedSigmas(
      schedule: .bongTangent, steps: 2, denoise: 0.2,
      config: Krea2Sampling.schedulerConfig(), mu: try Self.shift().mu)

    XCTAssertEqual(sigmas.count, 3)
    let expected: [Float] = [0.11746056, 0.043265149, 0.0]
    for (got, want) in zip(sigmas, expected) {
      XCTAssertEqual(got, want, accuracy: 1e-6)
    }
    // Not the naive readings.
    XCTAssertNotEqual(sigmas[0], 0.2)
    XCTAssertNotEqual(sigmas[0], 0.1)
  }

  /// The tail IS the tail of the `total`-step grid, for every parametrised
  /// pair — the relationship, not just the one pinned triple.
  func testTailIsTheSuffixOfTheStretchedGrid() throws {
    let config = Krea2Sampling.schedulerConfig()
    let mu = try Self.shift().mu
    for schedule in [SigmaScheduleKind.bongTangent, .krea2, .flow] {
      for steps in 2...8 {
        for d in 1...9 {
          let denoise = Double(d) / 10.0
          let total = try Krea2StagedRender.stretchedStepCount(steps: steps, denoise: denoise)
          let full = try SchedulerFactory.resolveSigmas(
            schedule: schedule, numSteps: total, config: config, mu: mu)
          let tail = try Krea2StagedRender.publishedSigmas(
            schedule: schedule, steps: steps, denoise: denoise, config: config, mu: mu)
          XCTAssertEqual(
            tail, Array(full.suffix(steps + 1)),
            "\(schedule.rawValue) steps \(steps) denoise \(denoise)")
        }
      }
    }
  }

  /// `denoise == 1.0` is the identity: the stage runs the whole schedule.
  func testDenoiseOneIsTheWholeGrid() throws {
    let config = Krea2Sampling.schedulerConfig()
    let mu = try Self.shift().mu
    for steps in 2...8 {
      let tail = try Krea2StagedRender.publishedSigmas(
        schedule: .bongTangent, steps: steps, denoise: 1.0, config: config, mu: mu)
      let full = try SchedulerFactory.resolveSigmas(
        schedule: .bongTangent, numSteps: steps, config: config, mu: mu)
      XCTAssertEqual(tail, full, "steps \(steps)")
    }
  }

  // MARK: - AC-31: the arithmetic type is load-bearing

  /// `total = int(steps/denoise)` over the AC's parametrised grid, against
  /// values dumped from the Python source (`python3 -c 'int(steps/denoise)'`,
  /// 2026-08-22). Not re-derived here — a table.
  func testStretchedStepCountMatchesPython() throws {
    // (steps, denoise×10, total)
    let dumped: [(Int, Int, Int)] = [
      (2,1,20),(2,2,10),(2,3,6),(2,4,5),(2,5,4),(2,6,3),(2,7,2),(2,8,2),(2,9,2),
      (3,1,30),(3,2,15),(3,3,10),(3,4,7),(3,5,6),(3,6,5),(3,7,4),(3,8,3),(3,9,3),
      (4,1,40),(4,2,20),(4,3,13),(4,4,10),(4,5,8),(4,6,6),(4,7,5),(4,8,5),(4,9,4),
      (5,1,50),(5,2,25),(5,3,16),(5,4,12),(5,5,10),(5,6,8),(5,7,7),(5,8,6),(5,9,5),
      (6,1,60),(6,2,30),(6,3,20),(6,4,15),(6,5,12),(6,6,10),(6,7,8),(6,8,7),(6,9,6),
      (7,1,70),(7,2,35),(7,3,23),(7,4,17),(7,5,14),(7,6,11),(7,7,10),(7,8,8),(7,9,7),
      (8,1,80),(8,2,40),(8,3,26),(8,4,20),(8,5,16),(8,6,13),(8,7,11),(8,8,10),(8,9,8),
    ]
    for (steps, d10, total) in dumped {
      XCTAssertEqual(
        try Krea2StagedRender.stretchedStepCount(steps: steps, denoise: Double(d10) / 10.0),
        total, "steps \(steps) denoise 0.\(d10)")
    }
  }

  /// The cases that make the `Double` decision load-bearing rather than
  /// stylistic: `Float(9)/Float(0.3)` lands on the other side of the integer
  /// from Python's `9/0.3`, so a `Float` division would silently select a
  /// 29-step grid where upstream builds 30 — a different tail, a different
  /// picture, no error anywhere.
  func testFloatDivisionWouldSelectADifferentTail() throws {
    let divergent: [(steps: Int, denoise: Double, python: Int, float32: Int)] = [
      (9, 0.3, 30, 29), (9, 0.6, 15, 14), (9, 0.15, 60, 59), (7, 0.28, 24, 25),
    ]
    for c in divergent {
      XCTAssertEqual(
        try Krea2StagedRender.stretchedStepCount(steps: c.steps, denoise: c.denoise),
        c.python, "steps \(c.steps) denoise \(c.denoise)")
      // The trap, spelled out: this is what the same expression in Float gives.
      let f = Int((Float(c.steps) / Float(c.denoise)).rounded(.towardZero))
      XCTAssertEqual(f, c.float32, "the Float trap moved — re-check §3.14")
      XCTAssertNotEqual(f, c.python)
    }
  }

  // MARK: - AC-28 / §3.14: `denoise <= 0` is a hard error

  func testDenoiseOutOfRangeIsRefused() {
    for bad: Double in [0, -0.1, 1.5, .infinity] {
      XCTAssertThrowsError(
        try Krea2StagedRender.stretchedStepCount(steps: 2, denoise: bad), "denoise \(bad)"
      ) { error in
        XCTAssertEqual(error as? Krea2StageError, .invalidDenoise(bad))
      }
    }
    // NaN never equals itself, so it cannot be compared through Equatable —
    // assert the case instead of the payload.
    XCTAssertThrowsError(try Krea2StagedRender.stretchedStepCount(steps: 2, denoise: .nan)) {
      guard case .invalidDenoise(let v)? = $0 as? Krea2StageError else {
        return XCTFail("expected invalidDenoise, got \($0)")
      }
      XCTAssertTrue(v.isNaN)
    }
  }

  func testStepsMustBePositive() {
    XCTAssertThrowsError(try Krea2StagedRender.stretchedStepCount(steps: 0, denoise: 0.2)) {
      XCTAssertEqual($0 as? Krea2StageError, .invalidSteps(0))
    }
  }

  // MARK: - The prepared grid is stage 2's OWN

  /// Stage 2 runs its own `prepare_sigmas`: the solver ends on the model's
  /// `sigma_min` and the model-free `σ_min → 0` conversion finishes it —
  /// the same contract stage 1 gets, applied to the stretched tail.
  func testStageTwoGetsItsOwnPreparedGridAndTail() throws {
    let shift = try Self.shift()
    let scheduler = try Krea2StagedRender.makeScheduler(
      sampler: .deis3m, sigmaSchedule: .bongTangent, steps: 2, denoise: 0.2,
      shift: shift, seed: 7, c2: 0.5)

    let sigmaMin = try SchedulerFactory.modelSigmaMin(
      schedule: .bongTangent, config: shift.config, mu: shift.mu)
    let walked = scheduler.sigmas.asArray(Float.self)
    XCTAssertEqual(scheduler.numInferenceSteps, 2)
    XCTAssertEqual(walked.count, 3)
    XCTAssertEqual(walked[0], 0.11746056, accuracy: 1e-6)
    XCTAssertEqual(walked[1], 0.043265149, accuracy: 1e-6)
    XCTAssertEqual(walked[2], sigmaMin)
    XCTAssertEqual(scheduler.finalConversionSigma, sigmaMin)
  }

  /// A non-RES4LYF sampler gets no preparation and no tail — the same rule the
  /// factory applies to stage 1.
  func testNonRES4LYFStageTwoKeepsTheSentinelZero() throws {
    let scheduler = try Krea2StagedRender.makeScheduler(
      sampler: .euler, sigmaSchedule: .bongTangent, steps: 2, denoise: 0.2,
      shift: try Self.shift(), seed: 7, c2: 0.5)
    XCTAssertNil(scheduler.finalConversionSigma)
    XCTAssertEqual(scheduler.sigmas.asArray(Float.self).last, 0.0)
  }

  // MARK: - The construction seam agrees with `SchedulerFactory.create`

  /// `Krea2StagedRender` builds its scheduler from an explicit grid because the
  /// factory only builds from a step count (§3.14's stretch-and-tail has no
  /// step count that produces it). At `denoise == 1.0` the two constructions
  /// describe the same run, and this pins that they agree — so a change to
  /// `SchedulerFactory.create` that this seam does not follow is a test
  /// failure rather than a silently divergent stage 2.
  func testSeamAgreesWithFactoryAtDenoiseOne() throws {
    let shift = try Self.shift()
    for kind in SchedulerKind.allCases {
      for schedule in [SigmaScheduleKind.bongTangent, .krea2, .beta] {
        let steps = 6
        let viaFactory = try Krea2Pipeline.makeScheduler(
          sampler: kind, sigmaSchedule: schedule, steps: steps, shift: shift, seed: 11, c2: 0.5)
        let viaSeam = try Krea2StagedRender.makeScheduler(
          sampler: kind, sigmaSchedule: schedule, steps: steps, denoise: 1.0,
          shift: shift, seed: 11, c2: 0.5)
        let label = "\(kind.rawValue) + \(schedule.rawValue)"
        XCTAssertEqual(
          viaSeam.sigmas.asArray(Float.self), viaFactory.sigmas.asArray(Float.self), label)
        XCTAssertEqual(viaSeam.numInferenceSteps, viaFactory.numInferenceSteps, label)
        XCTAssertEqual(viaSeam.finalConversionSigma, viaFactory.finalConversionSigma, label)
        XCTAssertEqual(
          String(describing: type(of: viaSeam)), String(describing: type(of: viaFactory)), label)
        XCTAssertEqual(viaSeam.modelOutputConvention, viaFactory.modelOutputConvention, label)
        XCTAssertEqual(
          viaSeam.requiresIntermediateEvaluation, viaFactory.requiresIntermediateEvaluation, label)
      }
    }
  }
}
