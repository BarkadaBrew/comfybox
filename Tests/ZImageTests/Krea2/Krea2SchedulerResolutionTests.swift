import XCTest
import MLX
@testable import ZImage

/// WP-E3 — how a `Krea2Pipeline.Request` becomes a `ZImageScheduler`
/// (FDD-krea2-raw-recipe §3.3, D3/A.1, D11, D18, D23). Weight-free:
/// `makeScheduler` and `validateTiers` are pure.
final class Krea2SchedulerResolutionTests: XCTestCase {

  static let align = 16
  static let x1 = Float((256 / align) * (256 / align))
  static let x2 = Float((1280 / align) * (1280 / align))

  // MARK: - The default scheduler IS the pre-change grid (AC-3 through the pipeline seam)

  func testDefaultSchedulerIsEulerOnThePreChangeGrid() throws {
    for seqLen in [256, 1024, 4096, 6400, 9216] {
      for steps in [4, 6, 9, 12, 20, 52] {
        let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: seqLen, align: Self.align)
        let scheduler = try Krea2Pipeline.makeScheduler(
          sampler: .euler, sigmaSchedule: .krea2, steps: steps, shift: shift, seed: 0, c2: 0.5)
        XCTAssertTrue(scheduler is FlowMatchEulerScheduler, "seqLen \(seqLen) steps \(steps)")
        XCTAssertEqual(scheduler.numInferenceSteps, steps)
        XCTAssertEqual(scheduler.modelOutputConvention, .velocity)
        let preChange = Krea2Sampling.timesteps(seqLen: seqLen, steps: steps, x1: Self.x1, x2: Self.x2)
        XCTAssertEqual(scheduler.sigmas.asArray(Float.self), preChange, "seqLen \(seqLen) steps \(steps)")
        XCTAssertEqual(scheduler.sigmas.dtype, .float32)
      }
    }
  }

  /// An explicit shift moves the grid (A.1: mu = shift), through the same seam.
  func testExplicitShiftReachesTheScheduler() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: 4096, align: Self.align)
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 0, c2: 0.5)
    XCTAssertEqual(scheduler.sigmas.asArray(Float.self), SigmaSchedule.krea2(numSteps: 9, mu: 1.15))
  }

  // MARK: - D11: `flow` stays legal on Krea 2 and is a different grid

  func testFlowIsLegalAndDiffersFromKrea2() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: 4096, align: Self.align)
    let flow = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .flow, steps: 9, shift: shift, seed: 0, c2: 0.5)
    let krea2 = try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 0, c2: 0.5)
    let f = flow.sigmas.asArray(Float.self), k = krea2.sigmas.asArray(Float.self)
    XCTAssertEqual(f.count, k.count)
    XCTAssertEqual(f[0], 1.0); XCTAssertEqual(k[0], 1.0)
    XCTAssertNotEqual(f, k, "flow is the shifted 1→1/1000 grid, not the native 1→0 warp")
    XCTAssertNotEqual(f[8], k[8], "the penultimate sigma is where they differ most")
  }

  // MARK: - Every sampler resolves on every schedule the factory offers for the family

  /// Every sampler × schedule resolves, and each ends where its family ends.
  ///
  /// The RES4LYF ports (`res_2s`, `res_3s`, `ralston_*`) come back on a
  /// `prepare_sigmas`-prepared grid (S-FIX-1): the trailing `0.0` is a sentinel
  /// they never step onto, so their last SOLVER sigma is the active
  /// `ModelSamplingFlux` table's σ_min and the zero is reached by
  /// `Krea2DenoiseLoop`'s model-free conversion instead. Everything else still
  /// carries the schedule's own trailing zero.
  func testEverySamplerAndScheduleResolves() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: 4096, align: Self.align)
    let sigmaMin = SigmaSchedule.fluxSigmaTable(
      shift: shift.mu, tableSize: Krea2Sampling.fluxTableSize)[0]

    for sampler in SchedulerKind.allCases {
      for schedule in SigmaScheduleKind.allCases {
        let what = "\(sampler.rawValue)/\(schedule.rawValue)"
        let scheduler = try Krea2Pipeline.makeScheduler(
          sampler: sampler, sigmaSchedule: schedule, steps: 9, shift: shift, seed: 7, c2: 0.5)
        let sigmas = scheduler.sigmas.asArray(Float.self)
        XCTAssertEqual(sigmas.first, 1.0, what)

        if sampler.isRES4LYFFamily {
          XCTAssertNotEqual(sigmas.last, 0.0, "\(what): RES4LYF never solves onto the 0 sentinel")
          XCTAssertEqual(
            Double(try XCTUnwrap(sigmas.last)), Double(sigmaMin),
            accuracy: Double(RES4LYFSigmaPreparation.sigmaMinMatchTolerance),
            "\(what): the last solver sigma is the model's σ_min")
          // Upstream guards the model-free tail on `sigmas[-2] == NS.sigma_min`
          // EXACTLY. `karras` ramps down to σ_min itself and hits it; the
          // `exponential` log/exp round trip misses by an ulp and therefore
          // finishes at that sigma with no conversion, exactly as upstream does.
          XCTAssertEqual(
            scheduler.finalConversionSigma, sigmas.last == sigmaMin ? sigmaMin : nil, what)
          // …and only an INSERT keeps the requested step count. `karras` and
          // `exponential` already sit inside the 1e-4 window, so nothing is
          // inserted and the count is upstream's `len(sigmas) − 2`.
          let inserts = ![SigmaScheduleKind.karras, .exponential].contains(schedule)
          XCTAssertEqual(scheduler.numInferenceSteps, inserts ? 9 : 8, "\(what): step count")
        } else {
          XCTAssertEqual(sigmas.last, 0.0, what)
          XCTAssertNil(scheduler.finalConversionSigma, what)
          XCTAssertEqual(scheduler.numInferenceSteps, 9, "\(what): step count is unmoved")
        }
      }
    }
  }

  func testRES2sCarriesC2() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: 4096, align: Self.align)
    let scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .res2s, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 0, c2: 0.3)
    XCTAssertEqual((scheduler as? RES2sScheduler)?.c2, 0.3)
  }

  // MARK: - D18: unimplemented tiers fail loud at the pipeline, never downgrade

  /// WP-E15 landed T2, so `eta` is no longer a TIER refusal — but it is still
  /// never ignored: a non-zero `eta` is honoured on a RES4LYF sampler and
  /// refused by name on any other (`etaUnsupportedSampler`). The two arms
  /// together are the D18 property; the tier gate is now bongmath's alone.
  func testEtaIsNoLongerATierRefusalButIsStillRefusedBySampler() {
    XCTAssertNoThrow(try Krea2Pipeline.validateTiers(eta: 0.5, bongmath: false))
    XCTAssertThrowsError(
      try Krea2Pipeline.makeSDEInjector(
        eta: 0.5, sampler: .euler, stageSeed: 0, layout: .channelsAtAxis1)
    ) { error in
      guard case Krea2ScheduleError.etaUnsupportedSampler(let sampler, let value) = error else {
        return XCTFail("\(error)")
      }
      XCTAssertEqual(sampler, "euler"); XCTAssertEqual(value, "0.5")
    }
    XCTAssertNotNil(
      try Krea2Pipeline.makeSDEInjector(
        eta: 0.5, sampler: .res2s, stageSeed: 0, layout: .channelsAtAxis1))
  }

  func testBongmathIsRefusedBeforeT3() {
    XCTAssertThrowsError(try Krea2Pipeline.validateTiers(eta: 0, bongmath: true)) { error in
      guard case Krea2ScheduleError.tierNotImplemented(let field, _, let tier) = error else {
        return XCTFail("\(error)")
      }
      XCTAssertEqual(field, "bongmath"); XCTAssertEqual(tier, "T3")
      XCTAssertTrue("\(error)".contains("WP-E16"), "\(error)")
    }
  }

  func testDefaultTiersPass() {
    XCTAssertNoThrow(try Krea2Pipeline.validateTiers(eta: 0, bongmath: false))
  }
}
