import XCTest
@testable import ZImage

final class LTX2PhaseTelemetryTests: XCTestCase {
  func testPhaseMeanAndSamples() {
    let t = LTX2PhaseTelemetry()
    t.begin(.vaeDecode, nowMs: 1_000); t.end(.vaeDecode, nowMs: 31_000)   // 30s
    t.begin(.vaeDecode, nowMs: 40_000); t.end(.vaeDecode, nowMs: 60_000)  // 20s
    let v = t.view()
    XCTAssertEqual(v.phases["vaeDecode"]?.samples, 2)
    XCTAssertEqual(v.phases["vaeDecode"]!.meanSec, 25.0, accuracy: 0.001)
  }

  func testMaxUninterruptibleExcludesDenoisePhases() {
    let t = LTX2PhaseTelemetry()
    t.begin(.baseDenoise, nowMs: 0); t.end(.baseDenoise, nowMs: 600_000)  // 600s, must NOT count
    t.begin(.vaeDecode, nowMs: 0); t.end(.vaeDecode, nowMs: 45_000)       // 45s
    t.begin(.vocoder, nowMs: 0); t.end(.vocoder, nowMs: 12_000)           // 12s
    XCTAssertEqual(t.view().maxUninterruptibleSec!, 45.0, accuracy: 0.001)
  }

  func testNilUntilSampled() {
    let t = LTX2PhaseTelemetry()
    XCTAssertNil(t.view().maxUninterruptibleSec)
    XCTAssertNil(t.view().meanStepSec)
    t.begin(.baseDenoise, nowMs: 0)   // begun but not ended
    XCTAssertEqual(t.view().currentPhase, "baseDenoise")
    XCTAssertNil(t.view().maxUninterruptibleSec)
  }

  func testStepSamples() {
    let t = LTX2PhaseTelemetry()
    t.recordStep(seconds: 2.0); t.recordStep(seconds: 4.0)
    XCTAssertEqual(t.view().meanStepSec!, 3.0, accuracy: 0.001)
  }
}
