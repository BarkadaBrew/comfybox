import Foundation
import XCTest

@testable import ZImage

/// WP-E12 — the explicit `shift` request field (FDD-krea2-raw-recipe D3).
///
/// `nil` (the default) is today's resolution-dependent `mu`, so every existing
/// render is unmoved. A non-nil value overrides `mu` with `log(shift)` for the
/// schedules that warp by `mu` (`.krea2`, `.flow`) **and** moves
/// `Krea2Sampling.schedulerConfig(shift:)` so the table-backed schedules
/// (`beta`, `beta57`, `karras`, `exponential`) move with it. `bong_tangent`
/// is immune (D6). Provenance later records `mu`, `shift` and
/// `shift_source` from the same struct (WP-E10).
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
    XCTAssertEqual(resolved.shift, exp(resolved.mu), accuracy: 1e-6, "effective shift is e^mu (~2.475 at 1024²)")
    XCTAssertEqual(resolved.shift, 2.475, accuracy: 5e-3)
    // The synthetic config carries shift 1.0 → non-flow bounds (0.001, 1.0) (§3.1).
    XCTAssertEqual(resolved.config.shift, 1.0)
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

  // MARK: - explicit → log(shift)

  func testExplicitShiftOverridesMuWithLog() throws {
    let resolved = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: Self.seqLen1024, align: Self.align)
    XCTAssertEqual(resolved.source, .explicit)
    XCTAssertEqual(resolved.shift, 1.15)
    XCTAssertEqual(resolved.mu, log(Float(1.15)), accuracy: 1e-7)
    XCTAssertEqual(resolved.config.shift, 1.15, "D3: the explicit shift moves the karras/beta bounds with it")
    XCTAssertNotEqual(resolved.mu, Krea2Sampling.mu(seqLen: Self.seqLen1024, align: Self.align))
  }

  /// `.krea2` under `mu = log(1.15)` is exactly ComfyUI's DiscreteFlow warp
  /// `1.15·t / (1 + 0.15·t)` on the native `linspace(1 → 0)` grid — the two
  /// parameterisations meet here, which is what makes D3's override the right seam.
  func testExplicitShiftKrea2GridIsTheDiscreteFlowWarp() throws {
    let resolved = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: Self.seqLen1024, align: Self.align)
    let steps = 9
    let grid = SigmaSchedule.krea2(numSteps: steps, mu: resolved.mu)
    XCTAssertEqual(grid.count, steps + 1)
    for i in 0...steps {
      let t = 1.0 - Float(i) / Float(steps)
      let want: Float = 1.15 * t / (1 + 0.15 * t)
      XCTAssertEqual(grid[i], want, accuracy: 1e-6, "i=\(i)")
    }
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
