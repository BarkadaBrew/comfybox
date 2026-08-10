// LTX2NAGTests.swift — Normalized Attention Guidance math.
//
// NAG (Normalized Attention Guidance, 2025) is the reference PinkCherry recipe's
// guidance mechanism: the author runs CFG 1.0 on BOTH passes and gets prompt
// adherence from NAG (scale 11.0, alpha 0.25, tau 2.5). ComfyBox had no NAG and
// compensated with cfg 3.5, which drives latent channels into saturation —
// measured 2026-08-02 as cyan chroma blotches over 9.3% of the frame.
//
// Unlike CFG (which extrapolates the final prediction), NAG extrapolates inside
// CROSS-ATTENTION and then renormalizes, so it cannot inflate activation
// magnitude without bound — that clamp is the whole point.
//
//   z_guid = z_pos * scale - z_neg * (scale - 1)
//   ratio  = ||z_guid|| / ||z_pos||            (per attention vector)
//   z_clamp = ratio > tau ? z_guid * (tau / ratio) : z_guid
//   z_out  = alpha * z_clamp + (1 - alpha) * z_pos

import XCTest
import MLX
@testable import ZImage

final class LTX2NAGTests: XCTestCase {

  func testIdentityWhenPositiveEqualsNegative() {
    // Extrapolating between identical tensors changes nothing, so the whole
    // operator must collapse to the identity regardless of scale/alpha/tau.
    let z = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 1, 4)
    let out = ltx2ApplyNAG(positive: z, negative: z, scale: 11.0, alpha: 0.25, tau: 2.5)
    MLX.eval(out)
    XCTAssertLessThan(MLX.abs(out - z).max().item(Float.self), 1e-5)
  }

  func testAlphaZeroReturnsPositiveUnchanged() {
    // alpha is the blend toward the guided vector; at 0 NAG must be a no-op.
    let pos = MLXArray(converting: [1.0, 0.0, 0.0, 0.0]).reshaped(1, 1, 4)
    let neg = MLXArray(converting: [0.0, 1.0, 0.0, 0.0]).reshaped(1, 1, 4)
    let out = ltx2ApplyNAG(positive: pos, negative: neg, scale: 11.0, alpha: 0.0, tau: 2.5)
    MLX.eval(out)
    XCTAssertLessThan(MLX.abs(out - pos).max().item(Float.self), 1e-5)
  }

  func testNormClampCapsGuidedMagnitude() {
    // The defining property: however far extrapolation pushes, the guided
    // vector's norm may not exceed tau x the positive vector's norm. This is
    // what CFG lacks and why high CFG saturates channels.
    let pos = MLXArray(converting: [1.0, 0.0]).reshaped(1, 1, 2)
    let neg = MLXArray(converting: [-1.0, 0.0]).reshaped(1, 1, 2)
    // z_guid = 1*11 - (-1)*10 = 21 -> ratio 21, far above tau
    let tau: Float = 2.5, alpha: Float = 1.0
    let out = ltx2ApplyNAG(positive: pos, negative: neg, scale: 11.0, alpha: alpha, tau: tau)
    MLX.eval(out)
    let outNorm = MLX.sqrt((out * out).sum()).item(Float.self)
    let posNorm = MLX.sqrt((pos * pos).sum()).item(Float.self)
    XCTAssertEqual(outNorm, tau * posNorm, accuracy: 1e-4,
                   "clamped output should sit exactly at tau x ||positive||")
  }

  func testUnclampedCaseMatchesHandComputedBlend() {
    // Below tau the guided vector passes through unclamped and blends by alpha.
    // pos=[2,0] neg=[1,0] scale=2 -> z_guid = 4 - 1 = 3; ratio 1.5 < tau 2.5.
    // alpha 0.5 -> 0.5*3 + 0.5*2 = 2.5
    let pos = MLXArray(converting: [2.0, 0.0]).reshaped(1, 1, 2)
    let neg = MLXArray(converting: [1.0, 0.0]).reshaped(1, 1, 2)
    let out = ltx2ApplyNAG(positive: pos, negative: neg, scale: 2.0, alpha: 0.5, tau: 2.5)
    MLX.eval(out)
    XCTAssertEqual(out[0, 0, 0].item(Float.self), 2.5, accuracy: 1e-4)
    XCTAssertEqual(out[0, 0, 1].item(Float.self), 0.0, accuracy: 1e-4)
  }

  func testNormsAreComputedPerVectorNotGlobally() {
    // Two tokens with very different magnitudes: clamping must be decided
    // independently per token. A global norm would let a large token's ratio
    // mask a small token's excursion — exactly the flaw in the global-std
    // guidance rescale this replaces.
    let pos = MLXArray(converting: [1.0, 0.0, 100.0, 0.0]).reshaped(1, 2, 2)
    let neg = MLXArray(converting: [-1.0, 0.0, 99.0, 0.0]).reshaped(1, 2, 2)
    let out = ltx2ApplyNAG(positive: pos, negative: neg, scale: 11.0, alpha: 1.0, tau: 2.5)
    MLX.eval(out)
    // token 0: z_guid = 21, ratio 21 -> clamped to 2.5 * 1 = 2.5
    XCTAssertEqual(out[0, 0, 0].item(Float.self), 2.5, accuracy: 1e-3)
    // token 1: z_guid = 1100 - 990 = 110, ratio 1.1 < tau -> unclamped
    XCTAssertEqual(out[0, 1, 0].item(Float.self), 110.0, accuracy: 1e-2)
  }
}
