import XCTest
import MLX
@testable import ZImage

/// WP-E9 — loading the Wan 2.1 FP32 file into `Krea2VAE` (FDD §3.9, AC-54 /
/// AC-55). Gated on the two real files; decodes a small fixed latent so the
/// weights are proven live, not just shape-compatible.
final class Krea2VAELayoutLoadTests: XCTestCase {

  private func fixedLatent() -> MLXArray {
    // Deterministic, mean-0-ish latent in the normalized space decode() expects.
    var values = [Float](repeating: 0, count: 8 * 8 * 16)
    for i in 0..<values.count {
      values[i] = sin(Float(i) * 0.37) * 0.8
    }
    return MLXArray(values, [1, 8, 8, 16])
  }

  private func decode(file: URL, layout: VAELayout) throws -> MLXArray {
    let vae = Krea2VAE()
    let used = try Krea2WeightLoader.loadVAE(vae, from: file, layout: layout)
    XCTAssertEqual(used, layout)
    MLX.eval(vae.parameters())
    let out = vae.decode(fixedLatent())
    MLX.eval(out)
    return out
  }

  /// AC-54: no `.shapeMismatch`; Qwen vs Wan decodes differ and are well-formed.
  func testWanLoadsAndDecodesDifferentlyFromQwen() throws {
    try Krea2VAEFixtures.requireBoth()
    let qwen = try decode(file: Krea2VAEFixtures.qwen, layout: .qwenDiffusers)
    let wan = try decode(file: Krea2VAEFixtures.wan, layout: .wanNative)
    XCTAssertEqual(qwen.shape, [1, 64, 64, 3])
    XCTAssertEqual(wan.shape, [1, 64, 64, 3])
    for (name, img) in [("qwen", qwen), ("wan", wan)] {
      XCTAssertFalse(MLX.any(MLX.isNaN(img)).item(Bool.self), "\(name) decode has NaN")
      XCTAssertGreaterThanOrEqual(img.min().item(Float.self), 0, "\(name) below 0")
      XCTAssertLessThanOrEqual(img.max().item(Float.self), 1, "\(name) above 1")
    }
    let maxDiff = MLX.abs(qwen - wan).max().item(Float.self)
    XCTAssertGreaterThan(maxDiff, 1e-3, "Qwen and Wan decodes are identical — the Wan weights did not land")
  }

  /// The loader sniffs the layout itself; a wrong explicit `layout:` is a
  /// fail-loud mismatch, never a silent reinterpretation.
  func testExplicitLayoutMustMatchTheFile() throws {
    try Krea2VAEFixtures.requireBoth()
    XCTAssertThrowsError(
      try Krea2WeightLoader.loadVAE(Krea2VAE(), from: Krea2VAEFixtures.wan, layout: .qwenDiffusers)
    ) { error in
      guard case Krea2VAEKeyMapError.layoutMismatch(let file, let requested, let detected) = error else {
        return XCTFail("expected layoutMismatch, got \(error)")
      }
      XCTAssertEqual(file, Krea2VAEFixtures.wan.path)
      XCTAssertEqual(requested, .qwenDiffusers)
      XCTAssertEqual(detected, .wanNative)
    }
  }

  /// AC-55: latentsMean/latentsStd equal the Qwen-Image vae/config.json
  /// values to 4 decimals and are unchanged across layouts (they are
  /// normalization constants of the latent space, not of the decoder file).
  func testLatentNormalization() throws {
    try Krea2VAEFixtures.requireBoth()
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Krea2VAEFixtures.qwenConfig.path))
    let json = try JSONSerialization.jsonObject(
      with: Data(contentsOf: Krea2VAEFixtures.qwenConfig)) as? [String: Any]
    let mean = try XCTUnwrap(json?["latents_mean"] as? [Double])
    let std = try XCTUnwrap(json?["latents_std"] as? [Double])
    XCTAssertEqual(mean.count, 16)
    XCTAssertEqual(std.count, 16)
    for i in 0..<16 {
      XCTAssertEqual(Double(Krea2VAE.latentsMean[i]), mean[i], accuracy: 5e-5, "latents_mean[\(i)]")
      XCTAssertEqual(Double(Krea2VAE.latentsStd[i]), std[i], accuracy: 5e-5, "latents_std[\(i)]")
    }
    // Unchanged across layouts: the constants are not a property of the file.
    XCTAssertEqual(Krea2VAE.latentNormalization(for: .wanNative).mean, Krea2VAE.latentsMean)
    XCTAssertEqual(Krea2VAE.latentNormalization(for: .wanNative).std, Krea2VAE.latentsStd)
    XCTAssertEqual(Krea2VAE.latentNormalization(for: .qwenDiffusers).mean, Krea2VAE.latentsMean)
    XCTAssertEqual(Krea2VAE.latentNormalization(for: .qwenDiffusers).std, Krea2VAE.latentsStd)
  }
}
