// Krea2VAERoundTripTests.swift — verifies the newly-ported Krea2 VAE encoder
// by round-tripping a real image through encode() -> decode() and checking
// the result is recognizably close to the source, not noise. This must pass
// before any img2img logic is built on top of Krea2VAE.encode().

import XCTest
import MLX
@testable import ZImage

final class Krea2VAERoundTripTests: XCTestCase {

  func testEncodeDecodeRoundTripPreservesImage() throws {
    try skipIfNoGPU()

    let paths: Krea2ModelPaths
    do {
      paths = try Krea2ModelPaths.resolve()
    } catch {
      throw XCTSkip("Krea-2 weights not available locally: \(error)")
    }

    let vae = Krea2VAE()
    try Krea2WeightLoader.loadVAE(vae, from: paths.vaeFile)
    MLX.eval(vae.parameters())

    // Synthetic 128x128 RGB test pattern (color blocks + gradient), built
    // directly as an MLXArray so this test has no external file dependency.
    // NHWC in [-1, 1], matching Krea2VAE.encode's expected input range.
    let size = 128
    var pixels = [Float](repeating: 0, count: size * size * 3)
    for y in 0..<size {
      for x in 0..<size {
        let idx = (y * size + x) * 3
        let r = Float(x) / Float(size - 1)
        let g = Float(y) / Float(size - 1)
        let b: Float = (x / 16 + y / 16) % 2 == 0 ? 0.9 : 0.1
        pixels[idx] = r * 2 - 1
        pixels[idx + 1] = g * 2 - 1
        pixels[idx + 2] = b * 2 - 1
      }
    }
    let source = MLXArray(pixels, [1, size, size, 3])

    let latents = vae.encode(source)
    MLX.eval(latents)
    XCTAssertEqual(latents.shape, [1, size / Krea2VAE.spatialScale, size / Krea2VAE.spatialScale, Krea2VAE.latentChannels])
    XCTAssertFalse(MLX.any(MLX.isNaN(latents)).item(Bool.self), "Encoder produced NaN latents")

    let decoded = vae.decode(latents)  // (1, H, W, 3) in [0, 1]
    MLX.eval(decoded)
    XCTAssertEqual(decoded.shape, [1, size, size, 3])

    // Compare against the source re-expressed in [0,1] (matching decode's output range).
    let sourceZeroOne = (source + 1) * 0.5
    let diff = MLX.abs(decoded - sourceZeroOne)
    let meanAbsError = diff.mean().item(Float.self)

    // A correctly-wired VAE round-trip is lossy but should stay well below
    // the error a garbled/mismatched encoder would produce (which saturates
    // toward ~0.3-0.5+ mean abs error on a normalized [0,1] image — visually
    // indistinguishable from noise, exactly the LoKr-alpha failure mode this
    // session already hit once). 0.15 leaves headroom for legitimate VAE
    // blur/compression while still catching a fundamentally broken encoder.
    XCTAssertLessThan(meanAbsError, 0.15, "VAE round-trip mean abs error too high — encoder likely miswired")
  }

  private func skipIfNoGPU() throws {
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }
}
