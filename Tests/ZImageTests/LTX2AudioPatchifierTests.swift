import Foundation
import MLX
import XCTest

@testable import ZImage

/// Task #21 wire 1 (top-level): AudioPatchifier parity with the reference
/// (symmetric_patchifier.py, AudioPatchifier(1, start_end=True), defaults
/// hop=160 sr=16000 downsample=4 causal). Times are analytic:
///   start(t) = max(4t - 3, 0) * 160/16000
///   end(t)   = (4(t+1) - 3) * 160/16000
final class LTX2AudioPatchifierTests: XCTestCase {

  func testPatchifyFlattensChannelMajor() {
    // [B=1, C=2, T=3, F=4]; value = c*100 + t*10 + f encodes position.
    var vals = [Float]()
    for c in 0..<2 { for t in 0..<3 { for f in 0..<4 { vals.append(Float(c * 100 + t * 10 + f)) } } }
    let z = MLXArray(vals).reshaped([1, 2, 3, 4])

    let (tokens, _) = LTX2AudioPatchifier.patchify(z)
    XCTAssertEqual(tokens.shape, [1, 3, 8], "b c t f -> b t (c f)")
    // Token t=1 must be [c0f0..c0f3, c1f0..c1f3] at t=1: 10,11,12,13,110,111,112,113
    let tok1 = tokens[0, 1].asArray(Float.self)
    XCTAssertEqual(tok1, [10, 11, 12, 13, 110, 111, 112, 113], "(c f) is channel-major")
  }

  func testTimingsMatchCausalMelMapping() {
    let z = MLXArray.zeros([2, 8, 5, 16])  // B=2, T=5
    let (tokens, timings) = LTX2AudioPatchifier.patchify(z)
    XCTAssertEqual(tokens.shape, [2, 5, 128])
    XCTAssertEqual(timings.shape, [2, 1, 5, 2], "start_end stacked on last axis")

    let starts = timings[0, 0, 0..., 0].asArray(Float.self)
    let ends = timings[0, 0, 0..., 1].asArray(Float.self)
    let expectedStarts: [Float] = [0.0, 0.01, 0.05, 0.09, 0.13]
    let expectedEnds: [Float] = [0.01, 0.05, 0.09, 0.13, 0.17]
    for (got, want) in zip(starts, expectedStarts) {
      XCTAssertEqual(got, want, accuracy: 1e-6, "start times: max(4t-3,0)*0.01")
    }
    for (got, want) in zip(ends, expectedEnds) {
      XCTAssertEqual(got, want, accuracy: 1e-6, "end times: (4t+1)*0.01")
    }
  }

  func testUnpatchifyRoundTrips() {
    let z = MLXRandom.normal([1, 8, 6, 16])
    let (tokens, _) = LTX2AudioPatchifier.patchify(z)
    let back = LTX2AudioPatchifier.unpatchify(tokens, channels: 8, freq: 16)
    XCTAssertEqual(back.shape, [1, 8, 6, 16])
    let diff = MLX.abs(back - z).max().item(Float.self)
    XCTAssertEqual(diff, 0, "patchify/unpatchify must be an exact inverse")
  }
}
