import XCTest
import Logging
import MLX
@testable import ZImage

/// Unit tests for the JoyAI-Echo 2D mel-spectrogram VAE (Phase 2).
///
/// The key fixture below is the **ground-truth** `audio_vae.*` tensor list
/// (name + shape) dumped from the real 46 GB monolith's manifest — all 102
/// tensors. The anti-noise test asserts every one lands on a real module
/// parameter of matching shape (no silent drops → no silent-silence decode).
final class LTX2AudioVAETests: XCTestCase {

  override func setUpWithError() throws {
    do {
      try MLX.withError {
        let probe = MLXArray([1 as Float, 2], [2]) + MLXArray([3 as Float, 4], [2])
        MLX.eval(probe)
      }
    } catch {
      throw XCTSkip("MLX evaluation is unavailable in this test runner: \(error)")
    }
  }

  /// Ground-truth (checkpoint-key, shape) fixture — all 102 audio_vae tensors.
  private static let audioVAEFixture: [(String, [Int])] = [
    ("audio_vae.decoder.conv_in.conv.bias", [512]),
    ("audio_vae.decoder.conv_in.conv.weight", [512,8,3,3]),
    ("audio_vae.decoder.conv_out.conv.bias", [2]),
    ("audio_vae.decoder.conv_out.conv.weight", [2,128,3,3]),
    ("audio_vae.decoder.mid.block_1.conv1.conv.bias", [512]),
    ("audio_vae.decoder.mid.block_1.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.mid.block_1.conv2.conv.bias", [512]),
    ("audio_vae.decoder.mid.block_1.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.mid.block_2.conv1.conv.bias", [512]),
    ("audio_vae.decoder.mid.block_2.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.mid.block_2.conv2.conv.bias", [512]),
    ("audio_vae.decoder.mid.block_2.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.up.0.block.0.conv1.conv.bias", [128]),
    ("audio_vae.decoder.up.0.block.0.conv1.conv.weight", [128,256,3,3]),
    ("audio_vae.decoder.up.0.block.0.conv2.conv.bias", [128]),
    ("audio_vae.decoder.up.0.block.0.conv2.conv.weight", [128,128,3,3]),
    ("audio_vae.decoder.up.0.block.0.nin_shortcut.conv.bias", [128]),
    ("audio_vae.decoder.up.0.block.0.nin_shortcut.conv.weight", [128,256,1,1]),
    ("audio_vae.decoder.up.0.block.1.conv1.conv.bias", [128]),
    ("audio_vae.decoder.up.0.block.1.conv1.conv.weight", [128,128,3,3]),
    ("audio_vae.decoder.up.0.block.1.conv2.conv.bias", [128]),
    ("audio_vae.decoder.up.0.block.1.conv2.conv.weight", [128,128,3,3]),
    ("audio_vae.decoder.up.0.block.2.conv1.conv.bias", [128]),
    ("audio_vae.decoder.up.0.block.2.conv1.conv.weight", [128,128,3,3]),
    ("audio_vae.decoder.up.0.block.2.conv2.conv.bias", [128]),
    ("audio_vae.decoder.up.0.block.2.conv2.conv.weight", [128,128,3,3]),
    ("audio_vae.decoder.up.1.block.0.conv1.conv.bias", [256]),
    ("audio_vae.decoder.up.1.block.0.conv1.conv.weight", [256,512,3,3]),
    ("audio_vae.decoder.up.1.block.0.conv2.conv.bias", [256]),
    ("audio_vae.decoder.up.1.block.0.conv2.conv.weight", [256,256,3,3]),
    ("audio_vae.decoder.up.1.block.0.nin_shortcut.conv.bias", [256]),
    ("audio_vae.decoder.up.1.block.0.nin_shortcut.conv.weight", [256,512,1,1]),
    ("audio_vae.decoder.up.1.block.1.conv1.conv.bias", [256]),
    ("audio_vae.decoder.up.1.block.1.conv1.conv.weight", [256,256,3,3]),
    ("audio_vae.decoder.up.1.block.1.conv2.conv.bias", [256]),
    ("audio_vae.decoder.up.1.block.1.conv2.conv.weight", [256,256,3,3]),
    ("audio_vae.decoder.up.1.block.2.conv1.conv.bias", [256]),
    ("audio_vae.decoder.up.1.block.2.conv1.conv.weight", [256,256,3,3]),
    ("audio_vae.decoder.up.1.block.2.conv2.conv.bias", [256]),
    ("audio_vae.decoder.up.1.block.2.conv2.conv.weight", [256,256,3,3]),
    ("audio_vae.decoder.up.1.upsample.conv.conv.bias", [256]),
    ("audio_vae.decoder.up.1.upsample.conv.conv.weight", [256,256,3,3]),
    ("audio_vae.decoder.up.2.block.0.conv1.conv.bias", [512]),
    ("audio_vae.decoder.up.2.block.0.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.up.2.block.0.conv2.conv.bias", [512]),
    ("audio_vae.decoder.up.2.block.0.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.up.2.block.1.conv1.conv.bias", [512]),
    ("audio_vae.decoder.up.2.block.1.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.up.2.block.1.conv2.conv.bias", [512]),
    ("audio_vae.decoder.up.2.block.1.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.up.2.block.2.conv1.conv.bias", [512]),
    ("audio_vae.decoder.up.2.block.2.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.up.2.block.2.conv2.conv.bias", [512]),
    ("audio_vae.decoder.up.2.block.2.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.decoder.up.2.upsample.conv.conv.bias", [512]),
    ("audio_vae.decoder.up.2.upsample.conv.conv.weight", [512,512,3,3]),
    ("audio_vae.encoder.conv_in.conv.bias", [128]),
    ("audio_vae.encoder.conv_in.conv.weight", [128,2,3,3]),
    ("audio_vae.encoder.conv_out.conv.bias", [16]),
    ("audio_vae.encoder.conv_out.conv.weight", [16,512,3,3]),
    ("audio_vae.encoder.down.0.block.0.conv1.conv.bias", [128]),
    ("audio_vae.encoder.down.0.block.0.conv1.conv.weight", [128,128,3,3]),
    ("audio_vae.encoder.down.0.block.0.conv2.conv.bias", [128]),
    ("audio_vae.encoder.down.0.block.0.conv2.conv.weight", [128,128,3,3]),
    ("audio_vae.encoder.down.0.block.1.conv1.conv.bias", [128]),
    ("audio_vae.encoder.down.0.block.1.conv1.conv.weight", [128,128,3,3]),
    ("audio_vae.encoder.down.0.block.1.conv2.conv.bias", [128]),
    ("audio_vae.encoder.down.0.block.1.conv2.conv.weight", [128,128,3,3]),
    ("audio_vae.encoder.down.0.downsample.conv.bias", [128]),
    ("audio_vae.encoder.down.0.downsample.conv.weight", [128,128,3,3]),
    ("audio_vae.encoder.down.1.block.0.conv1.conv.bias", [256]),
    ("audio_vae.encoder.down.1.block.0.conv1.conv.weight", [256,128,3,3]),
    ("audio_vae.encoder.down.1.block.0.conv2.conv.bias", [256]),
    ("audio_vae.encoder.down.1.block.0.conv2.conv.weight", [256,256,3,3]),
    ("audio_vae.encoder.down.1.block.0.nin_shortcut.conv.bias", [256]),
    ("audio_vae.encoder.down.1.block.0.nin_shortcut.conv.weight", [256,128,1,1]),
    ("audio_vae.encoder.down.1.block.1.conv1.conv.bias", [256]),
    ("audio_vae.encoder.down.1.block.1.conv1.conv.weight", [256,256,3,3]),
    ("audio_vae.encoder.down.1.block.1.conv2.conv.bias", [256]),
    ("audio_vae.encoder.down.1.block.1.conv2.conv.weight", [256,256,3,3]),
    ("audio_vae.encoder.down.1.downsample.conv.bias", [256]),
    ("audio_vae.encoder.down.1.downsample.conv.weight", [256,256,3,3]),
    ("audio_vae.encoder.down.2.block.0.conv1.conv.bias", [512]),
    ("audio_vae.encoder.down.2.block.0.conv1.conv.weight", [512,256,3,3]),
    ("audio_vae.encoder.down.2.block.0.conv2.conv.bias", [512]),
    ("audio_vae.encoder.down.2.block.0.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.encoder.down.2.block.0.nin_shortcut.conv.bias", [512]),
    ("audio_vae.encoder.down.2.block.0.nin_shortcut.conv.weight", [512,256,1,1]),
    ("audio_vae.encoder.down.2.block.1.conv1.conv.bias", [512]),
    ("audio_vae.encoder.down.2.block.1.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.encoder.down.2.block.1.conv2.conv.bias", [512]),
    ("audio_vae.encoder.down.2.block.1.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.encoder.mid.block_1.conv1.conv.bias", [512]),
    ("audio_vae.encoder.mid.block_1.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.encoder.mid.block_1.conv2.conv.bias", [512]),
    ("audio_vae.encoder.mid.block_1.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.encoder.mid.block_2.conv1.conv.bias", [512]),
    ("audio_vae.encoder.mid.block_2.conv1.conv.weight", [512,512,3,3]),
    ("audio_vae.encoder.mid.block_2.conv2.conv.bias", [512]),
    ("audio_vae.encoder.mid.block_2.conv2.conv.weight", [512,512,3,3]),
    ("audio_vae.per_channel_statistics.mean-of-means", [128]),
    ("audio_vae.per_channel_statistics.std-of-means", [128]),
  ]

  /// Build a synthetic monolith subset from the fixture (PyTorch OIHW layout).
  private func syntheticCheckpoint() -> [String: MLXArray] {
    var w: [String: MLXArray] = [:]
    for (k, shape) in Self.audioVAEFixture {
      w[k] = MLXArray.zeros(shape)
    }
    return w
  }

  func testFixtureHasAll102Tensors() {
    XCTAssertEqual(Self.audioVAEFixture.count, 102)
  }

  /// Module structure exactly covers the checkpoint: 102 params, no more/less.
  func testModuleParameterCountMatchesCheckpoint() {
    let vae = LTX2AudioVAE()
    let moduleKeys = vae.parameters().flattened().map { $0.0 }
    XCTAssertEqual(moduleKeys.count, 102,
      "module param count \(moduleKeys.count) != checkpoint tensor count 102")
  }

  /// Anti-noise guard: every remapped checkpoint key hits a real module param.
  func testEveryCheckpointKeyMapsToAModuleParameter() {
    let vae = LTX2AudioVAE()
    let moduleKeys = Set(vae.parameters().flattened().map { $0.0 })
    let remapped = LTX2AudioVAE.remapKeys(syntheticCheckpoint())
    XCTAssertEqual(remapped.count, 102, "remap dropped keys before matching")
    var matched = 0
    for (k, _) in remapped {
      if moduleKeys.contains(k) { matched += 1 }
      else { XCTFail("remapped key has no module param: \(k)") }
    }
    XCTAssertEqual(matched, 102, "not all checkpoint tensors matched")
  }

  /// Weights apply with shape verification on (proves the OIHW→OHWI transpose
  /// and every channel count is correct), and the guard reports 102/102.
  func testLoadWeightsAppliesWithShapeVerificationAndFullCoverage() throws {
    let vae = LTX2AudioVAE()
    let logger = Logger(label: "test.audiovae")
    let (matched, total) = try vae.loadWeightsFromTensors(
      tensors: syntheticCheckpoint(), logger: logger)
    XCTAssertEqual(total, 102)
    XCTAssertEqual(matched, 102)
  }

  /// The guard throws (not silently succeeds) when almost nothing matches.
  func testLoadThrowsOnNearZeroCoverage() {
    let vae = LTX2AudioVAE()
    let logger = Logger(label: "test.audiovae")
    let junk: [String: MLXArray] = ["audio_vae.encoder.bogus.conv.weight": MLXArray.zeros([4, 4, 3, 3])]
    XCTAssertThrowsError(try vae.loadWeightsFromTensors(tensors: junk, logger: logger))
  }

  /// Encode compresses 4× on both axes; decode restores the mel shape exactly.
  func testEncodeDecodeShapeRoundTrip() {
    let vae = LTX2AudioVAE()
    let mel = MLXArray.zeros([1, 2, 64, 16])       // (B, 2, F=64, T=16)
    let z = vae.encode(mel)
    MLX.eval(z)
    XCTAssertEqual(z.shape, [1, 8, 16, 4], "latent shape wrong: \(z.shape)")
    let out = vae.decode(z)
    MLX.eval(out)
    XCTAssertEqual(out.shape, [1, 2, 64, 16], "decoded mel shape wrong: \(out.shape)")
  }

  /// A non-trivial decode from random latents is finite and not all-zero
  /// (guards against a dead/zero conv stack).
  func testDecodeProducesFiniteNonTrivialOutput() throws {
    let vae = LTX2AudioVAE()
    // Give the convs unit weights so a nonzero latent propagates (zeros-init
    // module would output zeros regardless of input).
    let logger = Logger(label: "test.audiovae")
    // Load ones-ish weights via the real remap path to exercise it end to end.
    var ck: [String: MLXArray] = [:]
    for (k, shape) in Self.audioVAEFixture {
      ck[k] = k.hasSuffix(".weight") && shape.count == 4
        ? MLXArray.ones(shape) * MLXArray(Float(0.01))
        : MLXArray.zeros(shape)
    }
    _ = try vae.loadWeightsFromTensors(tensors: ck, logger: logger)
    let z = MLXArray.ones([1, 8, 16, 4])
    let out = vae.decode(z)
    MLX.eval(out)
    let maxAbs = MLX.max(MLX.abs(out)).item(Float.self)
    XCTAssertTrue(maxAbs.isFinite, "decode produced non-finite output (NaN/Inf)")
    XCTAssertGreaterThan(maxAbs, 0, "decode produced all-zero output (dead conv stack)")
  }
}
