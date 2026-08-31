import XCTest

@testable import ZImage

/// Transcription gate for `heun_2s` / `heun_3s`. The N-row numerics are shared
/// with `RalstonScheduler` (via `RES4LYFTableau`, already trace-parity tested),
/// so the only heun-specific risk is the Butcher tableau. These pin it verbatim
/// against RES4LYF `rk_coefficients_beta.py` (commit 26036f647…) and check the
/// classical RK consistency conditions (row-sums of `a` equal `c`; `b` sums to 1).
final class RES4LYFHeunSchedulerTests: XCTestCase {

  func testHeun2sTableauMatchesUpstream() {
    // `"heun_2s"`: a = [[], [1]], b = [1/2, 1/2], c = [0, 1]
    let t = RES4LYFHeunScheduler.tableau(for: .two)
    XCTAssertEqual(t.a, [[0, 0], [1, 0]])
    XCTAssertEqual(t.b, [0.5, 0.5])
    XCTAssertEqual(t.c, [0, 1])
  }

  func testHeun3sTableauMatchesUpstream() {
    // `"heun_3s"`: a = [[], [1/3], [0, 2/3]], b = [1/4, 0, 3/4], c = [0, 1/3, 2/3]
    let t = RES4LYFHeunScheduler.tableau(for: .three)
    XCTAssertEqual(t.a, [[0, 0, 0], [1.0 / 3.0, 0, 0], [0, 2.0 / 3.0, 0]])
    XCTAssertEqual(t.b, [0.25, 0, 0.75])
    XCTAssertEqual(t.c, [0, 1.0 / 3.0, 2.0 / 3.0])
  }

  /// Classical explicit-RK consistency: every `a` row sums to its `c` node, and
  /// the `b` weights sum to 1. A transcription slip in any coefficient breaks one.
  func testConsistencyConditions() {
    for stages in RES4LYFHeunScheduler.Stages.allCases {
      let t = RES4LYFHeunScheduler.tableau(for: stages)
      XCTAssertEqual(t.b.reduce(0, +), 1.0, accuracy: 1e-12, "\(stages.name): b-weights sum to 1")
      for (row, aRow) in t.a.enumerated() {
        XCTAssertEqual(
          aRow.reduce(0, +), t.c[row], accuracy: 1e-12,
          "\(stages.name): a-row \(row) sums to c[\(row)]")
      }
    }
  }

  /// The scheduler advertises the right shape: `rows == stage count`, linear
  /// frame, data-prediction consumer, and the RES4LYF names.
  func testShapeAndNames() {
    let sigmas: [Float] = [1.0, 0.6, 0.2, 0.0]
    let two = RES4LYFHeunScheduler(stages: .two, numInferenceSteps: 3, sigmaValues: sigmas)
    XCTAssertEqual(two.rows, 2)
    XCTAssertEqual(two.frame, .linear)
    XCTAssertEqual(two.modelOutputConvention, .dataPrediction)
    XCTAssertEqual(RES4LYFHeunScheduler.Stages.two.name, "heun_2s")
    XCTAssertEqual(RES4LYFHeunScheduler.Stages.three.name, "heun_3s")
  }
}
