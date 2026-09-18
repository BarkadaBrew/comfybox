import Foundation
import MLX
import XCTest

@testable import ZImage

/// waveform -> mel, for audio-driven chunks (FDD §4.7, WP11).
///
/// The risk the FDD names is a wrong normalization making the video follow
/// noise, so these tests check the analysis against things that are true of a
/// correct mel: a tone lands in the bin its frequency belongs to, silence sits
/// at the floor, louder is larger, and the frame count matches the audio latent
/// rate exactly. The basis itself comes from the checkpoint, not from us.
final class LTX2MelAnalysisTests: XCTestCase {

  /// A synthetic stand-in for the shipped basis: a real DFT plus a triangular
  /// mel filterbank. Same shapes and same meaning, so the code under test is
  /// exercised exactly as it is in production.
  private func analysis() -> LTX2MelAnalysis {
    let n = LTX2MelAnalysis.fftSize
    let bins = n / 2 + 1
    var forward = [Float](repeating: 0, count: 514 * n)
    for k in 0..<bins {
      for t in 0..<n {
        let angle = 2 * Float.pi * Float(k) * Float(t) / Float(n)
        // A Hann window, as every reference analysis uses.
        let window = 0.5 - 0.5 * cos(2 * Float.pi * Float(t) / Float(n - 1))
        forward[k * n + t] = cos(angle) * window
        forward[(k + bins) * n + t] = -sin(angle) * window
      }
    }
    // 64 triangular filters spread over the 257 bins.
    var mel = [Float](repeating: 0, count: 64 * bins)
    let step = Float(bins - 2) / 65
    for m in 0..<64 {
      let centre = step * Float(m + 1)
      for k in 0..<bins {
        let distance = abs(Float(k) - centre)
        let value = max(0, 1 - distance / max(step, 1))
        mel[m * bins + k] = value
      }
    }
    return LTX2MelAnalysis(
      forwardBasis: MLXArray(forward, [514, n]), melBasis: MLXArray(mel, [64, bins]))
  }

  private func tone(hz: Float, seconds: Float, amplitude: Float = 0.5) -> MLXArray {
    let count = Int(seconds * Float(LTX2MelAnalysis.sampleRate))
    let samples = (0..<count).map { index -> Float in
      amplitude * sin(2 * .pi * hz * Float(index) / Float(LTX2MelAnalysis.sampleRate))
    }
    return MLXArray(samples + samples, [2, count])
  }

  // MARK: shape and rate

  func testMelShapeIsWhatTheAudioVAEExpects() {
    let mel = analysis().mel(waveform: tone(hz: 440, seconds: 1.0))
    XCTAssertEqual(mel.shape, [1, 2, 64, LTX2MelAnalysis.frameCount(samples: 16_000)])
  }

  func testFrameRateMatchesTheAudioLatentRate() {
    // The audio latent is 25 frames/s and the VAE compresses time 4x, so mel
    // must be 100 frames/s. One second is 101 frames (the centre-padded +1).
    XCTAssertEqual(LTX2MelAnalysis.frameCount(samples: 16_000), 101)
    XCTAssertEqual(LTX2MelAnalysis.frameCount(samples: 8_000), 51)
    XCTAssertEqual(LTX2MelAnalysis.sampleRate / LTX2MelAnalysis.hop, 100)
  }

  func testAMonoWaveformIsDuplicatedRatherThanRefused() {
    let count = 4_800
    let mono = MLXArray((0..<count).map { sin(Float($0) * 0.01) }, [1, count])
    let mel = analysis().mel(waveform: mono)
    XCTAssertEqual(mel.dim(1), 2)
    let left = mel[0, 0].asArray(Float.self)
    let right = mel[0, 1].asArray(Float.self)
    XCTAssertEqual(left, right)
  }

  func testAShortClipStillProducesFrames() {
    let mel = analysis().mel(waveform: tone(hz: 440, seconds: 0.01))
    XCTAssertGreaterThan(mel.dim(3), 0, "a tail shorter than a window is padded, not dropped")
  }

  // MARK: the analysis is actually analysing

  func testATonesEnergyLandsWhereItsFrequencyBelongs() {
    let mel = analysis().mel(waveform: tone(hz: 300, seconds: 0.5))
    let low = mel[0, 0, 0..<16, 0...].mean().item(Float.self)
    let high = mel[0, 0, 48..<64, 0...].mean().item(Float.self)
    XCTAssertGreaterThan(low, high + 1.0, "a 300 Hz tone is low-frequency energy, not high")
  }

  func testSilenceSitsAtTheLogFloor() {
    let silence = MLX.zeros([2, 16_000], dtype: .float32)
    let mel = analysis().mel(waveform: silence)
    let value = mel.mean().item(Float.self)
    XCTAssertEqual(value, log(LTX2MelAnalysis.logFloor), accuracy: 0.01)
  }

  func testLouderIsLarger() {
    let quiet = analysis().mel(waveform: tone(hz: 300, seconds: 0.3, amplitude: 0.05))
    let loud = analysis().mel(waveform: tone(hz: 300, seconds: 0.3, amplitude: 0.8))
    XCTAssertGreaterThan(loud.mean().item(Float.self), quiet.mean().item(Float.self))
  }

  func testEnergyFollowsTheSignalInTime() {
    // Half a second of silence, then half a second of tone: the second half of
    // the mel must be louder than the first.
    let count = 8_000
    let silence = [Float](repeating: 0, count: count)
    let sound = (0..<count).map { 0.6 * sin(2 * .pi * 300 * Float($0) / 16_000) }
    let waveform = MLXArray(silence + sound + silence + sound, [2, count * 2])
    let mel = analysis().mel(waveform: waveform)
    let frames = mel.dim(3)
    let first = mel[0, 0, 0..., 0..<(frames / 2 - 2)].mean().item(Float.self)
    let second = mel[0, 0, 0..., (frames / 2 + 2)...].mean().item(Float.self)
    XCTAssertGreaterThan(second, first + 1.0)
  }

  // MARK: loading

  func testLoadRefusesACheckpointWithoutTheBasis() {
    XCTAssertNil(LTX2MelAnalysis.load(weights: [:]))
    XCTAssertNil(
      LTX2MelAnalysis.load(weights: [
        "vocoder.mel_stft.stft_fn.forward_basis": MLX.zeros([514, 1, 512]),
      ]), "half the basis is not a basis")
  }

  func testLoadAcceptsTheShippedShapes() {
    let loaded = LTX2MelAnalysis.load(weights: [
      "vocoder.mel_stft.stft_fn.forward_basis": MLX.zeros([514, 1, 512]),
      "vocoder.mel_stft.mel_basis": MLX.zeros([64, 257]),
    ])
    XCTAssertNotNil(loaded)
  }

  func testLoadRefusesWrongShapesRatherThanProducingNoise() {
    XCTAssertNil(
      LTX2MelAnalysis.load(weights: [
        "vocoder.mel_stft.stft_fn.forward_basis": MLX.zeros([514, 1, 1024]),
        "vocoder.mel_stft.mel_basis": MLX.zeros([64, 257]),
      ]))
    XCTAssertNil(
      LTX2MelAnalysis.load(weights: [
        "vocoder.mel_stft.stft_fn.forward_basis": MLX.zeros([514, 1, 512]),
        "vocoder.mel_stft.mel_basis": MLX.zeros([128, 257]),
      ]), "128 mel bins is the vocoder's synthesis side, not this analysis")
  }

  // MARK: resampling

  func testResampleChangesLengthProportionallyAndKeepsTheSignal() {
    let source = (0..<48_000).map { sin(2 * .pi * 200 * Float($0) / 48_000) }
    let resampled = LTX2MelAnalysis.resample([source], from: 48_000)
    XCTAssertEqual(resampled[0].count, 16_000, accuracy: 2)
    // A 200 Hz tone stays a 200 Hz tone: count zero crossings.
    let crossings = zip(resampled[0], resampled[0].dropFirst())
      .filter { ($0 < 0) != ($1 < 0) }.count
    XCTAssertEqual(Double(crossings), 400, accuracy: 6)
  }

  func testResampleIsANoOpAtTheTargetRate() {
    let signal: [[Float]] = [[0.1, 0.2, 0.3]]
    XCTAssertEqual(LTX2MelAnalysis.resample(signal, from: 16_000), signal)
  }
}

private func XCTAssertEqual(
  _ lhs: Int, _ rhs: Int, accuracy: Int, file: StaticString = #filePath, line: UInt = #line
) {
  XCTAssertTrue(abs(lhs - rhs) <= accuracy, "\(lhs) is not within \(accuracy) of \(rhs)", file: file, line: line)
}
