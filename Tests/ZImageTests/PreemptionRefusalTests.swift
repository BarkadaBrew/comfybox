import XCTest
@testable import ZImage

/// #1479: unit tests for the pure preemption refusal guard. Inert without
/// telemetry samples on both sides (never refuses on a guess); refuses when
/// the projected remaining render time is less than the observed evict+reload
/// round trip, reporting that projected time as the ETA; allows preemption
/// when the render has plenty of steps left.
final class PreemptionRefusalTests: XCTestCase {
  func testInertWithoutTelemetry() {
    XCTAssertNil(preemptionRefusalETA(stepsRemaining: 1, meanStepSec: nil,
      remainingPhaseMeansSec: [], evictReloadRoundTripSec: 120))
    XCTAssertNil(preemptionRefusalETA(stepsRemaining: 1, meanStepSec: 30,
      remainingPhaseMeansSec: [], evictReloadRoundTripSec: nil))
  }

  func testRefusesNearlyFinishedRender() {
    // 2 steps * 10s + 40s decode = 60s remaining < 120s round trip -> refuse, ETA 60
    let eta = preemptionRefusalETA(stepsRemaining: 2, meanStepSec: 10,
      remainingPhaseMeansSec: [40], evictReloadRoundTripSec: 120)
    XCTAssertEqual(eta!, 60.0, accuracy: 0.001)
  }

  func testAllowsLongRemainingRender() {
    XCTAssertNil(preemptionRefusalETA(stepsRemaining: 40, meanStepSec: 12,
      remainingPhaseMeansSec: [40, 15], evictReloadRoundTripSec: 120))
  }
}
