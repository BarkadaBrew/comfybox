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

  func testEverySamplerAndScheduleResolves() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: 4096, align: Self.align)
    for sampler in SchedulerKind.allCases {
      for schedule in SigmaScheduleKind.allCases {
        let scheduler = try Krea2Pipeline.makeScheduler(
          sampler: sampler, sigmaSchedule: schedule, steps: 9, shift: shift, seed: 7, c2: 0.5)
        XCTAssertEqual(scheduler.sigmas.asArray(Float.self).first, 1.0, "\(sampler.rawValue)/\(schedule.rawValue)")
        XCTAssertEqual(scheduler.sigmas.asArray(Float.self).last, 0.0, "\(sampler.rawValue)/\(schedule.rawValue)")
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

  func testEtaIsRefusedBeforeT2() {
    XCTAssertThrowsError(try Krea2Pipeline.validateTiers(eta: 0.5, bongmath: false)) { error in
      guard case Krea2ScheduleError.tierNotImplemented(let field, let value, let tier) = error else {
        return XCTFail("\(error)")
      }
      XCTAssertEqual(field, "eta"); XCTAssertEqual(value, "0.5"); XCTAssertEqual(tier, "T2")
      XCTAssertTrue("\(error)".contains("WP-E15"), "\(error)")
    }
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
