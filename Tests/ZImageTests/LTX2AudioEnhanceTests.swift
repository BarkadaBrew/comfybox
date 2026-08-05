import Foundation
import MLX
import XCTest

@testable import ZImage

/// Task #26: in-engine mastering between codec decode and AAC mux.
/// The chain (validated as an ffmpeg prototype 2026-08-04 on the e2e set):
/// high-pass rumble cut -> 7.5kHz de-harsh dip -> RMS loudness raise with a
/// soft-knee ceiling. Raw model output is ~10dB quiet with BWE grain; these
/// tests pin the contract on synthetic signals.
final class LTX2AudioEnhanceTests: XCTestCase {

  private func sine(_ hz: Float, seconds: Float, rate: Float, amp: Float) -> [Float] {
    (0..<Int(seconds * rate)).map { amp * sin(2 * .pi * hz * Float($0) / rate) }
  }
  private func rms(_ x: [Float]) -> Float {
    sqrt(x.reduce(0) { $0 + $1 * $1 } / Float(x.count))
  }

  func testQuietTrackIsRaisedTowardTargetWithoutClipping() {
    // -26 dBFS-ish quiet voice-band tone (the measured raw-output regime).
    let quiet = sine(440, seconds: 1.0, rate: 48000, amp: 0.05)
    let stereo = MLXArray(quiet + quiet).reshaped([2, quiet.count])
    let out = LTX2AudioEnhance.process(stereo, sampleRate: 48000)
    let ch = out[0].asArray(Float.self)
    let gained = rms(ch) / rms(quiet)
    XCTAssertGreaterThan(gained, 2.0, "quiet content must be raised substantially (~+8dB or more)")
    XCTAssertLessThanOrEqual(ch.map(abs).max()!, 0.9, "ceiling holds — no digital clipping")
  }

  func testLoudTrackIsNotBoostedIntoTheCeiling() {
    let loud = sine(440, seconds: 1.0, rate: 48000, amp: 0.6)
    let stereo = MLXArray(loud + loud).reshaped([2, loud.count])
    let out = LTX2AudioEnhance.process(stereo, sampleRate: 48000)
    let ch = out[0].asArray(Float.self)
    XCTAssertLessThanOrEqual(ch.map(abs).max()!, 0.9, "ceiling holds for hot input")
    let gained = rms(ch) / rms(loud)
    XCTAssertLessThan(gained, 1.3, "already-loud content is not slammed upward")
  }

  /// Single-bin DFT magnitude at `hz` (Goertzel-style correlation).
  private func binMag(_ x: [Float], hz: Float, rate: Float) -> Float {
    var re: Float = 0, im: Float = 0
    for (i, v) in x.enumerated() {
      let ph = 2 * Float.pi * hz * Float(i) / rate
      re += v * cos(ph); im += v * sin(ph)
    }
    return sqrt(re * re + im * im)
  }

  func testRumbleAndHarshBandsAreAttenuatedRelativeToVoiceBand() {
    // One COMPOSITE signal — loudness normalization applies a single shared
    // gain, so relative band ratios isolate the filters.
    let rate: Float = 48000
    let n = 48000
    var comp = [Float](repeating: 0, count: n)
    for i in 0..<n {
      let t = Float(i) / rate
      comp[i] = 0.1 * (sin(2 * .pi * 440 * t) + sin(2 * .pi * 25 * t) + sin(2 * .pi * 7500 * t))
    }
    let st = MLXArray(comp + comp).reshaped([2, n])
    let out = LTX2AudioEnhance.process(st, sampleRate: 48000)[0].asArray(Float.self)
    let inRatio25 = binMag(comp, hz: 25, rate: rate) / binMag(comp, hz: 440, rate: rate)
    let outRatio25 = binMag(out, hz: 25, rate: rate) / binMag(out, hz: 440, rate: rate)
    let inRatio75 = binMag(comp, hz: 7500, rate: rate) / binMag(comp, hz: 440, rate: rate)
    let outRatio75 = binMag(out, hz: 7500, rate: rate) / binMag(out, hz: 440, rate: rate)
    XCTAssertLessThan(outRatio25 / inRatio25, 0.35, "sub-rumble cut hard relative to voice band")
    XCTAssertLessThan(outRatio75 / inRatio75, 0.95, "7.5kHz dipped relative to voice band")
  }

  func testShapePreservedAndDeterministic() {
    let x = MLXRandom.normal([2, 48000], key: MLXRandom.key(7)) * MLXArray(Float(0.05))
    let a = LTX2AudioEnhance.process(x, sampleRate: 48000)
    let b = LTX2AudioEnhance.process(x, sampleRate: 48000)
    XCTAssertEqual(a.shape, [2, 48000])
    XCTAssertEqual(MLX.abs(a - b).max().item(Float.self), 0, "pure function")
  }
}
