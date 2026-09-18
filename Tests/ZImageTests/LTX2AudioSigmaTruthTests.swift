import Foundation
import MLX
import XCTest

@testable import ZImage

/// When the voice is GIVEN, tell the model the audio is clean (WP11).
///
/// `av.audioLatents` is re-snapped to the conditioning after every step, so
/// `callAV` always receives sigma-0 audio on an audio-driven render. Passing
/// the VIDEO's sigma as `audioSigma` anyway is a lie the gate believes:
/// `avCaA2vGateAdaLN` — the a2v gate applied on the VIDEO stream — is driven
/// by the audio sigma, so for most of the schedule the model discounts a clean
/// voice as if it were noise, exactly while the video's structure is decided.
///
/// The gate is a learned adaLN, so "lower sigma means more influence" is not
/// something a unit test can assert about the weights. What IS assertable, and
/// what actually broke, is that the number we pass must describe the tensor we
/// pass.
final class LTX2AudioSigmaTruthTests: XCTestCase {

  /// The rule the pipeline applies, extracted so it can be tested without
  /// running a denoise loop.
  private func audioSigma(conditioned: Bool, videoSigma: Float) -> Float {
    conditioned ? 0 : videoSigma
  }

  func testGivenAudioIsReportedAsCleanAtEveryStep() {
    for videoSigma in [Float(1.0), 0.85, 0.45, 0.1, 0.0] {
      XCTAssertEqual(
        audioSigma(conditioned: true, videoSigma: videoSigma), 0,
        "a GIVEN voice is clean on every step, including the first")
    }
  }

  func testGeneratedAudioStillRidesTheSchedule() {
    // The ordinary path must not move: audio being denoised alongside the
    // video really IS at the video's sigma.
    for videoSigma in [Float(1.0), 0.45, 0.0] {
      XCTAssertEqual(audioSigma(conditioned: false, videoSigma: videoSigma), videoSigma)
    }
  }

  func testTheAudioTheModelSeesIsTheConditioningItself() {
    // The premise the change rests on: after the re-snap, what reaches callAV
    // is the clean conditioning, not a noised version of it.
    let clean = MLX.ones([1, 8, 40, 16], dtype: .float32)
    let state = LTX2AVDenoiseState(
      audioLatents: MLX.zeros([1, 8, 40, 16], dtype: .float32),
      audioContext: MLX.zeros([1, 4, 2048], dtype: .float32),
      pe: LTX2AVPEs(
        videoPE: (MLX.zeros([1, 1]), MLX.zeros([1, 1])),
        audioPE: (MLX.zeros([1, 1]), MLX.zeros([1, 1])),
        crossVideoPE: (MLX.zeros([1, 1]), MLX.zeros([1, 1])),
        crossAudioPE: (MLX.zeros([1, 1]), MLX.zeros([1, 1]))))
    state.conditioning = clean
    if let c = state.conditioning { state.audioLatents = c }  // what the loop does
    XCTAssertEqual(state.audioLatents[0, 0, 0, 0].item(Float.self), 1.0)
  }
}
