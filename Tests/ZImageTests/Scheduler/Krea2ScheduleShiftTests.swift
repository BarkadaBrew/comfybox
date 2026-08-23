import Foundation
import XCTest

@testable import ZImage

/// WP-E12 / WP-E12b — the explicit `shift` request field (FDD-krea2-raw-recipe
/// D3 as amended by Addendum A.1: **`shift` IS mu**).
///
/// `nil` (the default) is today's resolution-dependent `mu`, so every existing
/// render is unmoved. A non-nil value **is** `mu` — ComfyUI registers Krea 2 as
/// `ModelSamplingFlux(shift=1.15)`, whose `flux_time_shift(mu=shift, t)` is the
/// same function as `Krea2Sampling.timesteps` — so the effective linear shift
/// is `e^shift`, never `shift` itself. The warp-by-`mu` schedules (`.krea2`,
/// `.flow`) and the table-backed schedules (`beta`, `beta57`, `karras`,
/// `exponential`, all on the 10 000-entry Flux table under the Krea 2 family)
/// move with the same `mu`. `bong_tangent` is immune (D6). Provenance later
/// records `mu`, `shift` and `shift_source` from the same struct (WP-E10).
///
/// Weight-free.
final class Krea2ScheduleShiftTests: XCTestCase {

  static let align = 16
  static let seqLen1024 = (1024 / 16) * (1024 / 16)  // 4096 tokens

  // MARK: - nil → dynamic mu (unchanged behaviour)

  func testNilShiftIsDynamicMu() throws {
    let resolved = try Krea2Sampling.resolveShift(explicit: nil, seqLen: Self.seqLen1024, align: Self.align)
    XCTAssertEqual(resolved.source, .dynamic)
    XCTAssertEqual(resolved.mu, Krea2Sampling.mu(seqLen: Self.seqLen1024, align: Self.align))
    XCTAssertEqual(resolved.mu, 0.90625, "mu at 1024² is exactly 0.5 + 0.65·(4096−256)/(6400−256)")
    XCTAssertEqual(resolved.shift, exp(resolved.mu), accuracy: 1e-6, "effective shift is e^mu (~2.475 at 1024²)")
    XCTAssertEqual(resolved.shift, 2.475, accuracy: 5e-3)
    // The synthetic config is the Krea 2 family's ModelSamplingFlux (§3.1, A.1).
    XCTAssertEqual(resolved.config.modelSampling, .flux(tableSize: 10000))
    XCTAssertEqual(resolved.config.shift, 1.0, "config.shift is not where Krea 2's shift lives; mu is")
    XCTAssertEqual(resolved.config.numTrainTimesteps, 1000)
    XCTAssertTrue(resolved.config.useDynamicShifting)
  }

  /// The mu the loop used before this WP is byte-identical to the resolved one.
  func testDynamicResolutionMatchesTimestepsInlineMu() throws {
    let x1 = Float((256 / Self.align) * (256 / Self.align))
    let x2 = Float((1280 / Self.align) * (1280 / Self.align))
    for seqLen in [256, 1024, 4096, 6400, 9216] {
      let resolved = try Krea2Sampling.resolveShift(explicit: nil, seqLen: seqLen, align: Self.align)
      let inline = Krea2Sampling.timesteps(seqLen: seqLen, steps: 9, x1: x1, x2: x2)
      let viaResolved = Krea2Sampling.timesteps(seqLen: seqLen, steps: 9, x1: x1, x2: x2, mu: resolved.mu)
      XCTAssertEqual(inline, viaResolved, "seqLen \(seqLen)")
    }
  }

  // MARK: - explicit → mu = shift (Addendum A.1)

  /// `shift` on the wire means mu directly: `mu = shift`, effective linear
  /// shift `e^shift`. D3's `mu = log(shift)` is withdrawn.
  func testExplicitShiftIsMu() throws {
    let resolved = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: Self.seqLen1024, align: Self.align)
    XCTAssertEqual(resolved.source, .explicit)
    XCTAssertEqual(resolved.mu, 1.15, "mu = explicit, not log(explicit)")
    XCTAssertNotEqual(resolved.mu, log(Float(1.15)), "the withdrawn D3 mapping must not survive")
    XCTAssertEqual(resolved.shift, exp(Float(1.15)), accuracy: 1e-6, "effective linear shift is e^shift ≈ 3.158")
    XCTAssertEqual(resolved.shift, 3.1582, accuracy: 5e-4)
    XCTAssertNotEqual(resolved.mu, Krea2Sampling.mu(seqLen: Self.seqLen1024, align: Self.align))
    // The config is the same ModelSamplingFlux either way; only mu moved.
    XCTAssertEqual(resolved.config.modelSampling, .flux(tableSize: 10000))
    XCTAssertEqual(resolved.config.shift, 1.0)
  }

  /// `.krea2` under `mu = 1.15` is exactly ComfyUI's `ModelSamplingFlux(shift=1.15)`
  /// warp `flux_time_shift(1.15, t) = e^1.15 / (e^1.15 + 1/t − 1)` on the native
  /// `linspace(1 → 0)` grid — the two parameterisations are the same function,
  /// which is why `shift` is mu (A.1).
  func testExplicitShiftKrea2GridIsTheFluxTimeShift() throws {
    let resolved = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: Self.seqLen1024, align: Self.align)
    let steps = 9
    let grid = SigmaSchedule.krea2(numSteps: steps, mu: resolved.mu)
    XCTAssertEqual(grid.count, steps + 1)
    let expMu = exp(1.15)
    for i in 0..<steps {
      let t = 1.0 - Double(i) / Double(steps)
      let want = expMu / (expMu + (1.0 / t - 1.0))
      XCTAssertEqual(Double(grid[i]), want, accuracy: 1e-6, "i=\(i)")
    }
    XCTAssertEqual(grid[steps], 0.0)
    // And it is NOT the DiscreteFlow warp 1.15·t/(1+0.15·t) that D3 originally assumed.
    let discreteFlowAtHalf: Float = 1.15 * 0.5 / (1 + 0.15 * 0.5)
    XCTAssertNotEqual(grid[4], discreteFlowAtHalf, accuracy: 1e-3, "linear 1.15 ≠ mu 1.15")
  }

  /// Every model-consulting schedule moves under an explicit shift; bong_tangent does not.
  func testExplicitShiftThreadsIntoEveryModelConsultingSchedule() throws {
    let dynamic = try Krea2Sampling.resolveShift(explicit: nil, seqLen: Self.seqLen1024, align: Self.align)
    let explicit = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: Self.seqLen1024, align: Self.align)
    let steps = 9
    for schedule in SigmaScheduleKind.allCases {
      let a = try SchedulerFactory.resolveSigmas(
        schedule: schedule, numSteps: steps, config: dynamic.config, mu: dynamic.mu)
      let b = try SchedulerFactory.resolveSigmas(
        schedule: schedule, numSteps: steps, config: explicit.config, mu: explicit.mu)
      if schedule == .bongTangent {
        XCTAssertEqual(a, b, "bong_tangent is shift-free (D6)")
      } else {
        XCTAssertNotEqual(a, b, "\(schedule.rawValue) must move under an explicit shift")
      }
    }
  }

  /// `beta` under the Krea 2 family at `shift: 1.15` is the published workflow's
  /// stage-1 grid — the Flux table, not the 1000-entry DiscreteFlow table
  /// (A.1; the full AC-21 pins live in `BetaScheduleComfyParityTests`).
  func testExplicitShiftBetaIsTheFluxGrid() throws {
    let resolved = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: Self.seqLen1024, align: Self.align)
    let beta = try SchedulerFactory.resolveSigmas(
      schedule: .beta, numSteps: 6, config: resolved.config, mu: resolved.mu)
    XCTAssertEqual(beta.count, 7)
    XCTAssertEqual(beta[1], 0.969095, accuracy: 1e-5, "Flux-table σ₁ (A.1), not DiscreteFlow's 0.919919")
  }

  // MARK: - fail loud

  func testNonPositiveShiftIsRejected() {
    for bad: Float in [0, -1, -0.5] {
      XCTAssertThrowsError(
        try Krea2Sampling.resolveShift(explicit: bad, seqLen: Self.seqLen1024, align: Self.align)
      ) { error in
        XCTAssertEqual(error as? Krea2ScheduleError, .invalidShift(bad))
        XCTAssertTrue("\(error)".contains("shift"), "the error names the field")
      }
    }
    XCTAssertThrowsError(
      try Krea2Sampling.resolveShift(explicit: .nan, seqLen: Self.seqLen1024, align: Self.align))
  }
}
