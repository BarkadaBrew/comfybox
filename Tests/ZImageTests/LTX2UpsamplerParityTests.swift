// LTX2UpsamplerParityTests.swift — numerical parity of the Swift latent
// upsampler against the ComfyUI/PyTorch reference implementation.
//
// Gated on two env vars so CI without the assets skips cleanly:
//   LTX2_UPSAMPLER_PATH     — the ltx-2.3-spatial-upscaler-x2-1.1.safetensors
//   LTX2_UPSAMPLER_REF_TAPS — safetensors with x + t0..t6 reference taps
//     (produced by the torch probe; see qa/video/FINDINGS-2026-07-25.md)
//
// 2026-07-25 context: with GT-config stage 1, our decoded upsampler output
// scored sharp 18.5 / flicker 0.555 vs ComfyUI's 41.4 / 0.157 on the SAME
// stage-1 latent statistics — this test bisects which stage of the module
// diverges numerically.

import XCTest
import MLX
import MLXNN
@testable import ZImage

final class LTX2UpsamplerParityTests: XCTestCase {

  private func relErr(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = MLX.abs(a - b).max().item(Float.self)
    let scale = MLX.abs(b).max().item(Float.self)
    return diff / max(scale, 1e-6)
  }

  func testParityAgainstTorchReference() throws {
    let env = ProcessInfo.processInfo.environment
    guard let ckptPath = env["LTX2_UPSAMPLER_PATH"],
          let tapsPath = env["LTX2_UPSAMPLER_REF_TAPS"],
          FileManager.default.fileExists(atPath: ckptPath),
          FileManager.default.fileExists(atPath: tapsPath) else {
      throw XCTSkip("LTX2_UPSAMPLER_PATH / LTX2_UPSAMPLER_REF_TAPS not set")
    }

    // Load the upsampler exactly as LTX2VideoGenerator does.
    let up = LTX2LatentUpsampler()
    let w = try MLX.loadArrays(url: URL(fileURLWithPath: ckptPath))
    let remapped: [(String, MLXArray)] = w.map { (rawKey, v) in
      let key = rawKey.hasPrefix("upsampler.0.")
        ? "upsampler.conv." + rawKey.dropFirst("upsampler.0.".count)
        : rawKey
      if key.hasSuffix(".weight") {
        if v.ndim == 5 { return (key, v.transposed(0, 2, 3, 4, 1)) }
        if v.ndim == 4 { return (key, v.transposed(0, 2, 3, 1)) }
      }
      return (key, v)
    }
    try up.update(parameters: ModuleParameters.unflattened(remapped), verify: [.shapeMismatch])
    MLX.eval(up.parameters())

    let taps = try MLX.loadArrays(url: URL(fileURLWithPath: tapsPath))
    // Reference tensors are (B, C, F, H, W); module works channels-last.
    let x = taps["x"]!.asType(.float32)
    var h = x.transposed(0, 2, 3, 4, 1)  // -> (B, F, H, W, C)

    func check(_ name: String, _ ours: MLXArray) {
      let ref = taps[name]!.asType(.float32)
      let oursCF = ours.transposed(0, 4, 1, 2, 3)  // back to (B, C, F, H, W)
      let e = relErr(oursCF, ref)
      print("PARITY \(name): relErr \(e)")
      XCTAssertLessThan(e, 2e-3, "\(name) diverges from torch reference")
    }

    h = up.initialConv(h)
    check("t0_initial_conv", h)
    h = up.initialNorm(h)
    check("t1_initial_norm", h)
    h = silu(h)
    check("t2_initial_act", h)
    for block in up.resBlocks { h = block(h) }
    check("t3_resblocks", h)
    h = up.upsampler(h)
    check("t4_upsampler", h)
    for block in up.postResBlocks { h = block(h) }
    check("t5_postblocks", h)
    h = up.finalConv(h)
    check("t6_final", h)
  }
}
