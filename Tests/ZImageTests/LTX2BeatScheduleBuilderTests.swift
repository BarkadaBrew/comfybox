// LTX2BeatScheduleBuilderTests.swift — pure-math coverage for the temporal
// beat scheduling bias builder (comfybox#310), independent of any
// transformer. Fixtures are hand-computed against the documented PromptRelay
// formula: cost = strength * relu(|pos - midpoint| - window)^2 / (2*sigma^2),
// added as a NEGATIVE bias.

import XCTest
import MLX
@testable import ZImage

final class LTX2BeatScheduleBuilderTests: XCTestCase {

  /// Independent (not-shared-with-production) reimplementation of the
  /// penalty, used to hand-compute fixtures below.
  private func expectedPenalty(pos: Float, midpoint: Float, window: Float, strength: Float) -> Float {
    let over = max(abs(pos - midpoint) - window, 0)
    guard over > 0 else { return 0 }
    let sigma: Float = 1.0 / Foundation.log(1.0 / 1e-3)
    return -(strength * over * over / (2 * sigma * sigma))
  }

  // MARK: (d) builder math vs. hand-computed 3-beat fixture

  func testVideoBiasMatchesHandComputedThreeBeatFixture() {
    // 8 frames, 1 token/frame (so query index == frame index), 6 text tokens
    // split 2/2/2 across three beats with distinct fracs/strengths.
    let frames = 8
    let tokensPerFrame = 1
    let textLen = 6

    let beats = [
      LTX2ResolvedBeat(tokenStart: 0, tokenEnd: 2, startFrac: 0.0, endFrac: 0.5, strength: 1.0),
      LTX2ResolvedBeat(tokenStart: 2, tokenEnd: 4, startFrac: 0.5, endFrac: 1.0, strength: 2.0),
      LTX2ResolvedBeat(tokenStart: 4, tokenEnd: 6, startFrac: 0.25, endFrac: 0.75, strength: 1.0),
    ]

    // Hand-derived geometry (frac -> frame, per the documented formula):
    //   beat1: midpoint=(0+0.5)*0.5*8=2.0, span=4.0, window=max(2-2,0)=0
    //   beat2: midpoint=(0.5+1.0)*0.5*8=6.0, span=4.0, window=0
    //   beat3: midpoint=(0.25+0.75)*0.5*8=4.0, span=4.0, window=0
    let geometry: [(midpoint: Float, window: Float, strength: Float, cols: Range<Int>)] = [
      (2.0, 0.0, 1.0, 0..<2),
      (6.0, 0.0, 2.0, 2..<4),
      (4.0, 0.0, 1.0, 4..<6),
    ]

    guard let bias = LTX2BeatScheduleBuilder.buildVideoBias(
      resolved: beats, frames: frames, tokensPerFrame: tokensPerFrame, textLen: textLen
    ) else {
      XCTFail("expected a non-nil bias for a non-empty beat schedule")
      return
    }
    XCTAssertEqual(bias.shape, [1, 1, frames * tokensPerFrame, textLen])

    var expected = [Float](repeating: 0, count: frames * textLen)
    for q in 0..<frames {
      for (midpoint, window, strength, cols) in geometry {
        let penalty = expectedPenalty(pos: Float(q), midpoint: midpoint, window: window, strength: strength)
        for k in cols { expected[q * textLen + k] = penalty }
      }
    }

    let actual = bias.reshaped([frames * textLen]).asArray(Float.self)
    for i in 0..<expected.count {
      XCTAssertEqual(actual[i], expected[i], accuracy: 1e-6, "mismatch at flat index \(i)")
    }

    // Sanity: the midpoint frame itself must be exactly zero (over == 0),
    // and it must NOT bleed into a different beat's columns.
    XCTAssertEqual(actual[2 * textLen + 0], 0, accuracy: 1e-6)
    XCTAssertEqual(actual[2 * textLen + 2], expectedPenalty(pos: 2, midpoint: 6, window: 0, strength: 2), accuracy: 1e-6)
  }

  // MARK: (e) audio seconds -> token mapping fixture

  func testAudioBiasMatchesHandComputedSecondsFixture() {
    // Same three beats/geometry as the video fixture (frame-space midpoint
    // is shared), but the QUERY axis is now real per-token seconds from the
    // audio patchifier instead of an integer frame index.
    let totalFrames = 8
    let fps: Float = 4.0  // duration = 8/4 = 2.0s
    let textLen = 6

    let beats = [
      LTX2ResolvedBeat(tokenStart: 0, tokenEnd: 2, startFrac: 0.0, endFrac: 0.5, strength: 1.0),
      LTX2ResolvedBeat(tokenStart: 2, tokenEnd: 4, startFrac: 0.5, endFrac: 1.0, strength: 2.0),
      LTX2ResolvedBeat(tokenStart: 4, tokenEnd: 6, startFrac: 0.25, endFrac: 0.75, strength: 1.0),
    ]
    // midpointSec = midpointFrame / fps; window stays 0 (same span math).
    let geometry: [(midpointSec: Float, windowSec: Float, strength: Float, cols: Range<Int>)] = [
      (2.0 / 4.0, 0.0, 1.0, 0..<2),
      (6.0 / 4.0, 0.0, 2.0, 2..<4),
      (4.0 / 4.0, 0.0, 1.0, 4..<6),
    ]

    // Real per-token seconds from the audio patchifier (5 latent frames),
    // exactly as `LTX2AudioPatchifier.patchify` derives them.
    let audioFrames = 5
    let starts = LTX2AudioPatchifier.latentTimesSeconds(from: 0, to: audioFrames)
    let ends = LTX2AudioPatchifier.latentTimesSeconds(from: 1, to: audioFrames + 1)
    let midSeconds = zip(starts, ends).map { ($0 + $1) / 2 }
    XCTAssertEqual(midSeconds.count, audioFrames)

    guard let bias = LTX2BeatScheduleBuilder.buildAudioBias(
      resolved: beats, totalFrames: totalFrames, fps: fps,
      audioTokenMidSeconds: midSeconds, textLen: textLen
    ) else {
      XCTFail("expected a non-nil audio bias for a non-empty beat schedule")
      return
    }
    XCTAssertEqual(bias.shape, [1, 1, audioFrames, textLen])

    var expected = [Float](repeating: 0, count: audioFrames * textLen)
    for q in 0..<audioFrames {
      for (midpointSec, windowSec, strength, cols) in geometry {
        let penalty = expectedPenalty(pos: midSeconds[q], midpoint: midpointSec, window: windowSec, strength: strength)
        for k in cols { expected[q * textLen + k] = penalty }
      }
    }

    let actual = bias.reshaped([audioFrames * textLen]).asArray(Float.self)
    for i in 0..<expected.count {
      XCTAssertEqual(actual[i], expected[i], accuracy: 1e-6, "mismatch at flat index \(i)")
    }

    // Every audio token here is well inside the first ~0.15s, far from all
    // three beats' midpoints (0.5s/1.5s/1.0s) — the bias must be strictly
    // non-zero everywhere (no accidental all-zero collapse from a units bug).
    XCTAssertTrue(actual.allSatisfy { $0 < 0 }, "expected every audio-token penalty to be non-zero and negative")
  }

  // MARK: absence -> nil (byte-identical to no-bias)

  func testEmptyScheduleBuildsNilBias() {
    XCTAssertNil(LTX2BeatScheduleBuilder.buildVideoBias(resolved: [], frames: 8, tokensPerFrame: 1, textLen: 6))
    XCTAssertNil(LTX2BeatScheduleBuilder.buildAudioBias(
      resolved: [], totalFrames: 8, fps: 4, audioTokenMidSeconds: [0.1, 0.2], textLen: 6))
  }
}
