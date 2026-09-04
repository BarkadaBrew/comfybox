import XCTest
@testable import ZImage

/// comfybox#307 point 4: the two-stage refine used to skip past two gates —
/// "no upsampler loaded" and the volume gate — with no consistent signal
/// (one branch logged nothing at all; the other logged `.info` only). This
/// pins the pure decision `LTX2RefineGate.decide` extracts from both.
final class LTX2RefineGateTests: XCTestCase {

  func testNotRequestedWhenTwoStageIsFalse() {
    // Regardless of upsampler/volume — this is normal single-pass operation,
    // never a "skip".
    XCTAssertEqual(
      LTX2RefineGate.decide(twoStage: false, upsamplerLoaded: false, preVolume: 999_999, maxVolume: 1),
      .notRequested)
    XCTAssertEqual(
      LTX2RefineGate.decide(twoStage: false, upsamplerLoaded: true, preVolume: 1, maxVolume: 999_999),
      .notRequested)
  }

  func testRunsWhenRequestedUpsamplerLoadedAndUnderTheVolumeGate() {
    XCTAssertEqual(
      LTX2RefineGate.decide(twoStage: true, upsamplerLoaded: true, preVolume: 1000, maxVolume: 2000),
      .run)
  }

  func testRunsWhenExactlyAtTheVolumeGate() {
    // The production check is `preVolume > maxVolume` — equal must still run.
    XCTAssertEqual(
      LTX2RefineGate.decide(twoStage: true, upsamplerLoaded: true, preVolume: 2000, maxVolume: 2000),
      .run)
  }

  func testSkipsWithUpsamplerUnavailableReasonWhenNotLoaded() {
    guard case .skip(let reason) = LTX2RefineGate.decide(
      twoStage: true, upsamplerLoaded: false, preVolume: 0, maxVolume: 0)
    else {
      return XCTFail("expected .skip")
    }
    XCTAssertTrue(reason.contains("upsampler_unavailable"), reason)
  }

  func testSkipsWithVolumeGateReasonWhenOverBudget() {
    // comfybox#307's own repro numbers: refine_max_vol=26000, 12s/480p at
    // refine_scale=1.5 exceeds it.
    guard case .skip(let reason) = LTX2RefineGate.decide(
      twoStage: true, upsamplerLoaded: true, preVolume: 30_000, maxVolume: 26_000)
    else {
      return XCTFail("expected .skip")
    }
    XCTAssertTrue(reason.contains("volume_gate"), reason)
    XCTAssertTrue(reason.contains("30000"), reason)
    XCTAssertTrue(reason.contains("26000"), reason)
  }

  /// Upsampler-unavailable is checked BEFORE the volume gate — an
  /// unavailable upsampler is reported as such even if the volume happens to
  /// also be over budget, not conflated into one ambiguous reason.
  func testUpsamplerUnavailableTakesPrecedenceOverVolumeGate() {
    guard case .skip(let reason) = LTX2RefineGate.decide(
      twoStage: true, upsamplerLoaded: false, preVolume: 999_999, maxVolume: 1)
    else {
      return XCTFail("expected .skip")
    }
    XCTAssertTrue(reason.contains("upsampler_unavailable"), reason)
  }
}
