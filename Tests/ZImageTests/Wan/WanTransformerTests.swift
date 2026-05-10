import XCTest
import MLX
import MLXRandom
import MLXFast
import MLXNN
@testable import ZImage

// MARK: - S3.1: Config + RMSNorm + Sinusoidal Embedding Tests

final class WanTransformerConfigTests: XCTestCase {

  func testI2VA14BPreset() {
    let config = WanTransformerConfig.i2vA14B
    XCTAssertEqual(config.dim, 5120)
    XCTAssertEqual(config.ffnDim, 13824)
    XCTAssertEqual(config.freqDim, 256)
    XCTAssertEqual(config.inDim, 36)
    XCTAssertEqual(config.outDim, 16)
    XCTAssertEqual(config.numHeads, 40)
    XCTAssertEqual(config.numLayers, 40)
    XCTAssertEqual(config.textLen, 512)
    XCTAssertEqual(config.textDim, 4096)
    XCTAssertEqual(config.patchSize.0, 1)
    XCTAssertEqual(config.patchSize.1, 2)
    XCTAssertEqual(config.patchSize.2, 2)
    XCTAssertEqual(config.eps, 1e-6)
    XCTAssertTrue(config.qkNorm)
    XCTAssertTrue(config.crossAttnNorm)
    XCTAssertEqual(config.windowSize.0, -1)
    XCTAssertEqual(config.windowSize.1, -1)
    XCTAssertEqual(config.modelType, "i2v")
  }

  func testHeadDimDerived() {
    let config = WanTransformerConfig.i2vA14B
    XCTAssertEqual(config.headDim, 128, "5120 / 40 = 128")
  }
}

final class WanRMSNormTests: XCTestCase {

  func testConstruction() {
    let norm = WanRMSNorm(dim: 64)
    XCTAssertEqual(norm.dim, 64)
    XCTAssertEqual(norm.weight.shape, [64])
  }

  func testOutputShape() {
    let norm = WanRMSNorm(dim: 32)
    let x = MLXRandom.normal([2, 8, 32]).asType(.float32)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [2, 8, 32])
  }

  func testRMSNormBehavior() {
    // After normalization with weight=1, x * rsqrt(mean(x^2) + eps) should
    // produce output where RMS ~ 1.0
    let norm = WanRMSNorm(dim: 64, eps: 1e-5)
    let x = MLXRandom.normal([1, 4, 64]).asType(.float32) * MLXArray(Float(5.0))
    let out = norm(x)
    eval(out)

    // Check RMS of output is approximately 1.0 for each position
    let rms = MLX.sqrt(MLX.mean(out * out, axis: -1))
    eval(rms)
    let rmsVal = rms[0, 0].item(Float.self)
    XCTAssertEqual(rmsVal, 1.0, accuracy: 0.1, "RMS after normalization should be ~1.0")
  }

  func testZeroInputSafe() {
    let norm = WanRMSNorm(dim: 16, eps: 1e-5)
    let x = MLXArray.zeros([1, 2, 16], type: Float.self)
    let out = norm(x)
    eval(out)
    // Should not produce NaN
    let sum = MLX.sum(out)
    eval(sum)
    XCTAssertFalse(sum.item(Float.self).isNaN, "RMSNorm should handle zero input")
  }
}

final class WanLayerNormTests: XCTestCase {

  func testNoAffineConstruction() {
    let norm = WanLayerNorm(dim: 64, elementwiseAffine: false)
    XCTAssertFalse(norm.elementwiseAffine)
    XCTAssertNil(norm.weight)
    XCTAssertNil(norm.bias)
  }

  func testAffineConstruction() {
    let norm = WanLayerNorm(dim: 64, elementwiseAffine: true)
    XCTAssertTrue(norm.elementwiseAffine)
    XCTAssertNotNil(norm.weight)
    XCTAssertNotNil(norm.bias)
    XCTAssertEqual(norm.weight!.shape, [64])
    XCTAssertEqual(norm.bias!.shape, [64])
  }

  func testOutputShapeNoAffine() {
    let norm = WanLayerNorm(dim: 32, elementwiseAffine: false)
    let x = MLXRandom.normal([2, 8, 32]).asType(.float32)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [2, 8, 32])
  }

  func testOutputShapeAffine() {
    let norm = WanLayerNorm(dim: 32, elementwiseAffine: true)
    let x = MLXRandom.normal([2, 8, 32]).asType(.float32)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [2, 8, 32])
  }

  func testNormalizationBehavior() {
    let norm = WanLayerNorm(dim: 64, elementwiseAffine: false)
    let x = MLXRandom.normal([1, 4, 64]).asType(.float32) * MLXArray(Float(10.0))
    let out = norm(x)
    eval(out)

    // Mean should be ~0, std should be ~1
    let mean = MLX.mean(out, axis: -1)
    let std = MLX.sqrt(MLX.mean(out * out, axis: -1))
    eval(mean, std)
    XCTAssertEqual(mean[0, 0].item(Float.self), 0.0, accuracy: 0.01)
    XCTAssertEqual(std[0, 0].item(Float.self), 1.0, accuracy: 0.1)
  }
}

final class SinusoidalEmbedding1DTests: XCTestCase {

  func testOutputShape() {
    let pos = MLXArray([Float(0), 1, 2, 3])
    let emb = sinusoidalEmbedding1D(dim: 256, position: pos)
    eval(emb)
    XCTAssertEqual(emb.shape, [4, 256])
  }

  func testOutputShapeFreqDim256() {
    let pos = MLXArray([Float(42)])
    let emb = sinusoidalEmbedding1D(dim: 256, position: pos)
    eval(emb)
    XCTAssertEqual(emb.shape, [1, 256])
  }

  func testValueRange() {
    // cos and sin values should be in [-1, 1]
    let pos = MLXRandom.normal([100]).asType(.float32) * MLXArray(Float(1000.0))
    let emb = sinusoidalEmbedding1D(dim: 256, position: pos)
    eval(emb)
    let maxVal = MLX.max(emb).item(Float.self)
    let minVal = MLX.min(emb).item(Float.self)
    XCTAssertLessThanOrEqual(maxVal, 1.001)
    XCTAssertGreaterThanOrEqual(minVal, -1.001)
  }

  func testZeroPositionCosIsOne() {
    // cos(0) = 1 for all frequencies
    let pos = MLXArray([Float(0)])
    let emb = sinusoidalEmbedding1D(dim: 8, position: pos)
    eval(emb)
    // First half (cos) should all be 1.0, second half (sin) should all be 0.0
    for i in 0..<4 {
      XCTAssertEqual(emb[0, i].item(Float.self), 1.0, accuracy: 1e-5,
                     "cos(0) should be 1.0 at index \(i)")
    }
    for i in 4..<8 {
      XCTAssertEqual(emb[0, i].item(Float.self), 0.0, accuracy: 1e-5,
                     "sin(0) should be 0.0 at index \(i)")
    }
  }

  func testDifferentPositionsDifferentEmbeddings() {
    let pos = MLXArray([Float(0), 1])
    let emb = sinusoidalEmbedding1D(dim: 64, position: pos)
    eval(emb)
    let diff = MLX.sum(MLX.abs(emb[0] - emb[1]))
    eval(diff)
    XCTAssertGreaterThan(diff.item(Float.self), 0.1,
                         "Different positions should produce different embeddings")
  }
}

// MARK: - S3.2: RoPE Tests

final class WanRoPETests: XCTestCase {

  func testFrequencyShape() {
    let freqs = WanRoPE.ropeParams(maxSeqLen: 1024, dim: 128)
    eval(freqs)
    // dim=128 -> 64 complex pairs -> stored as [1024, 64, 2] (real/imag)
    XCTAssertEqual(freqs.shape, [1024, 64, 2])
  }

  func testFrequencySplit() {
    // headDim = 128 -> 64 complex pairs
    // Split: [64 - 4*(64//6), 2*(64//6), 2*(64//6)]
    // = [64 - 4*10, 2*10, 2*10] = [24, 20, 20] = 64 total
    // But actually c = headDim/2 = 64
    // freqs split: [c - 2*(c//3), c//3, c//3] = [64 - 2*21, 21, 21] = [22, 21, 21]
    let headDim = 128
    let c = headDim / 2  // 64
    let split0 = c - 2 * (c / 3)  // 64 - 42 = 22
    let split1 = c / 3  // 21
    let split2 = c / 3  // 21
    XCTAssertEqual(split0 + split1 + split2, 64, "RoPE freq splits must sum to headDim/2")
    XCTAssertEqual(split0, 22)
    XCTAssertEqual(split1, 21)
    XCTAssertEqual(split2, 21)
  }

  func testRopeParamsGeneration() {
    // rope_params produces dim/2 complex frequencies for dim=44
    // d - 4*(d//6) where d=128 gives 128-4*21=44 for temporal
    let freqs = WanRoPE.ropeParams(maxSeqLen: 16, dim: 44)
    eval(freqs)
    XCTAssertEqual(freqs.shape, [16, 22, 2], "dim=44 -> 22 complex pairs -> [16, 22, 2]")
  }

  func testCombinedFrequencies() {
    let headDim = 128
    let freqs = WanRoPE.buildFrequencies(maxSeqLen: 1024, headDim: headDim)
    eval(freqs)
    // Total: 22 + 21 + 21 = 64 complex pairs
    XCTAssertEqual(freqs.shape, [1024, 64, 2])
  }

  func testRopeApplyPreservesShape() {
    let headDim = 128
    let numHeads = 4
    let seqLen = 8
    let freqs = WanRoPE.buildFrequencies(maxSeqLen: 1024, headDim: headDim)

    // x: [B, seqLen, numHeads, headDim]
    let x = MLXRandom.normal([1, seqLen, numHeads, headDim]).asType(.float32)
    let gridSizes: [[Int]] = [[2, 2, 2]]  // F=2, H=2, W=2 -> 8 tokens

    let out = WanRoPE.ropeApply(x, gridSizes: gridSizes, freqs: freqs)
    eval(out)
    XCTAssertEqual(out.shape, [1, seqLen, numHeads, headDim])
  }

  func testRopeApplyWithPadding() {
    let headDim = 128
    let numHeads = 2
    let seqLen = 16  // padded
    let freqs = WanRoPE.buildFrequencies(maxSeqLen: 1024, headDim: headDim)

    let x = MLXRandom.normal([1, seqLen, numHeads, headDim]).asType(.float32)
    let gridSizes: [[Int]] = [[2, 2, 2]]  // only 8 real tokens

    let out = WanRoPE.ropeApply(x, gridSizes: gridSizes, freqs: freqs)
    eval(out)
    XCTAssertEqual(out.shape, [1, seqLen, numHeads, headDim],
                   "Should preserve padded sequence length")
  }
}

// MARK: - S3.3: Attention Tests

final class WanSelfAttentionTests: XCTestCase {

  func testConstruction() {
    let attn = WanSelfAttention(dim: 64, numHeads: 4, qkNorm: true)
    XCTAssertEqual(attn.numHeads, 4)
    XCTAssertEqual(attn.headDim, 16)
    XCTAssertNotNil(attn.normQ)
    XCTAssertNotNil(attn.normK)
  }

  func testConstructionNoQKNorm() {
    let attn = WanSelfAttention(dim: 64, numHeads: 4, qkNorm: false)
    XCTAssertNil(attn.normQ)
    XCTAssertNil(attn.normK)
  }

  func testForwardShape() {
    let dim = 64
    let numHeads = 4
    let seqLen = 8
    let attn = WanSelfAttention(dim: dim, numHeads: numHeads, qkNorm: true)
    let freqs = WanRoPE.buildFrequencies(maxSeqLen: 1024, headDim: dim / numHeads)
    let x = MLXRandom.normal([1, seqLen, dim]).asType(.float32)
    let seqLens = [seqLen]
    let gridSizes = [[2, 2, 2]]  // 8 tokens

    let out = attn(x, seqLens: seqLens, gridSizes: gridSizes, freqs: freqs)
    eval(out)
    XCTAssertEqual(out.shape, [1, seqLen, dim])
  }

  func testHasBias() {
    let attn = WanSelfAttention(dim: 32, numHeads: 2)
    let params = attn.parameters().flattened()
    let keys = Set(params.map(\.0))
    XCTAssertTrue(keys.contains("q.bias"), "q should have bias")
    XCTAssertTrue(keys.contains("k.bias"), "k should have bias")
    XCTAssertTrue(keys.contains("v.bias"), "v should have bias")
    XCTAssertTrue(keys.contains("o.bias"), "o should have bias")
  }
}

final class WanCrossAttentionTests: XCTestCase {

  func testForwardShape() {
    let dim = 64
    let numHeads = 4
    let seqLen = 8
    let contextLen = 12
    let attn = WanCrossAttention(dim: dim, numHeads: numHeads, qkNorm: true)
    let x = MLXRandom.normal([1, seqLen, dim]).asType(.float32)
    let context = MLXRandom.normal([1, contextLen, dim]).asType(.float32)

    let out = attn(x, context: context, contextLens: nil)
    eval(out)
    XCTAssertEqual(out.shape, [1, seqLen, dim])
  }

  func testContextDimMatchesDim() {
    // Cross-attention: q from x, k/v from context, both dim
    let dim = 64
    let numHeads = 4
    let attn = WanCrossAttention(dim: dim, numHeads: numHeads, qkNorm: true)
    let x = MLXRandom.normal([1, 4, dim]).asType(.float32)
    let ctx = MLXRandom.normal([1, 8, dim]).asType(.float32)

    let out = attn(x, context: ctx, contextLens: nil)
    eval(out)
    XCTAssertEqual(out.shape, [1, 4, dim])
  }
}

// MARK: - S3.4: Transformer Block Tests

final class WanTransformerBlockTests: XCTestCase {

  func testConstruction() {
    let block = WanTransformerBlock(
      dim: 64, ffnDim: 128, numHeads: 4,
      qkNorm: true, crossAttnNorm: true
    )
    // norm3 should have weight + bias when crossAttnNorm=true
    let params = block.parameters().flattened()
    let keys = Set(params.map(\.0))
    XCTAssertTrue(keys.contains("norm3.weight"), "norm3 should have weight")
    XCTAssertTrue(keys.contains("norm3.bias"), "norm3 should have bias")
  }

  func testModulationShape() {
    let dim = 64
    let block = WanTransformerBlock(
      dim: dim, ffnDim: 128, numHeads: 4,
      qkNorm: true, crossAttnNorm: true
    )
    XCTAssertEqual(block.modulation.shape, [1, 6, dim])
  }

  func testForwardShape() {
    let dim = 64
    let numHeads = 4
    let seqLen = 8
    let textLen = 4
    let block = WanTransformerBlock(
      dim: dim, ffnDim: 128, numHeads: numHeads,
      qkNorm: true, crossAttnNorm: true
    )
    let freqs = WanRoPE.buildFrequencies(maxSeqLen: 1024, headDim: dim / numHeads)

    let x = MLXRandom.normal([1, seqLen, dim]).asType(.float32)
    // e: per-position modulation [B, seqLen, 6, dim]
    let e = MLXRandom.normal([1, seqLen, 6, dim]).asType(.float32)
    let context = MLXRandom.normal([1, textLen, dim]).asType(.float32)

    let out = block(
      x, e: e, seqLens: [seqLen], gridSizes: [[2, 2, 2]],
      freqs: freqs, context: context, contextLens: nil
    )
    eval(out)
    XCTAssertEqual(out.shape, [1, seqLen, dim])
  }

  func testNorm1HasNoLearnableParams() {
    let block = WanTransformerBlock(
      dim: 64, ffnDim: 128, numHeads: 4,
      qkNorm: true, crossAttnNorm: true
    )
    let params = block.parameters().flattened()
    let keys = Set(params.map(\.0))
    // norm1 is WanLayerNorm(elementwiseAffine=false) -> no weight/bias
    XCTAssertFalse(keys.contains("norm1.weight"), "norm1 should have no weight")
    XCTAssertFalse(keys.contains("norm1.bias"), "norm1 should have no bias")
  }

  func testNorm2HasNoLearnableParams() {
    let block = WanTransformerBlock(
      dim: 64, ffnDim: 128, numHeads: 4,
      qkNorm: true, crossAttnNorm: true
    )
    let params = block.parameters().flattened()
    let keys = Set(params.map(\.0))
    XCTAssertFalse(keys.contains("norm2.weight"), "norm2 should have no weight")
    XCTAssertFalse(keys.contains("norm2.bias"), "norm2 should have no bias")
  }

  func testFFNIsGELU() {
    // FFN should be Linear -> GELU -> Linear (2 projections, not 3)
    let block = WanTransformerBlock(
      dim: 64, ffnDim: 128, numHeads: 4,
      qkNorm: true, crossAttnNorm: true
    )
    let params = block.parameters().flattened()
    let ffnKeys = params.filter { $0.0.hasPrefix("ffn.") }.map(\.0)
    // Should have: ffn.layers.0.weight, ffn.layers.0.bias, ffn.layers.2.weight, ffn.layers.2.bias
    let weightKeys = ffnKeys.filter { $0.hasSuffix(".weight") }
    XCTAssertEqual(weightKeys.count, 2, "FFN should have exactly 2 weight matrices")
  }
}

// MARK: - S3.5: Head + Unpatchify Tests

final class WanHeadTests: XCTestCase {

  func testConstruction() {
    let head = WanHead(dim: 64, outDim: 16, patchSize: (1, 2, 2))
    XCTAssertEqual(head.modulation.shape, [1, 2, 64])
  }

  func testOutputDim() {
    let head = WanHead(dim: 64, outDim: 16, patchSize: (1, 2, 2))
    let x = MLXRandom.normal([1, 8, 64]).asType(.float32)
    let e = MLXRandom.normal([1, 8, 64]).asType(.float32)
    let out = head(x, e: e)
    eval(out)
    // out_dim * prod(patch_size) = 16 * 1 * 2 * 2 = 64
    XCTAssertEqual(out.shape, [1, 8, 64])
  }
}

final class WanUnpatchifyTests: XCTestCase {

  func testBasicUnpatchify() {
    let outDim = 16
    let patchSize = (1, 2, 2)
    let gridSizes: [[Int]] = [[2, 4, 4]]  // F_patches, H_patches, W_patches
    let seqLen = 2 * 4 * 4  // 32 tokens
    let patchFeatures = outDim * patchSize.0 * patchSize.1 * patchSize.2  // 64

    let x = MLXRandom.normal([1, seqLen, patchFeatures]).asType(.float32)
    let result = WanTransformer3D.unpatchify(
      x, gridSizes: gridSizes, patchSize: patchSize, outDim: outDim
    )

    XCTAssertEqual(result.count, 1)
    let out = result[0]
    eval(out)
    // Expected: [outDim, F*1, H*2, W*2] = [16, 2, 8, 8]
    XCTAssertEqual(out.shape, [16, 2, 8, 8])
  }
}

// MARK: - S3.6: Full Model Tests

final class WanTransformer3DTests: XCTestCase {

  /// Small config for unit testing (avoid 14B parameter allocation).
  private var testConfig: WanTransformerConfig {
    WanTransformerConfig(
      dim: 64,
      ffnDim: 128,
      freqDim: 32,
      inDim: 36,
      outDim: 16,
      numHeads: 4,
      numLayers: 2,
      textLen: 8,
      textDim: 32,
      patchSize: (1, 2, 2),
      eps: 1e-6,
      qkNorm: true,
      crossAttnNorm: true,
      windowSize: (-1, -1),
      modelType: "i2v"
    )
  }

  func testConstruction() {
    let _ = WanTransformer3D(config: testConfig)
  }

  func testForwardShape() {
    let config = testConfig
    let model = WanTransformer3D(config: config)

    // Input: [C_in, F, H, W] per sample
    // For I2V: C_in = in_dim = 36 (16 noise + 20 cond concatenated)
    let x = [MLXRandom.normal([36, 2, 8, 8]).asType(.float32)]
    let t = MLXArray([Float(0.5)])  // timestep
    let context = [MLXRandom.normal([4, 32]).asType(.float32)]  // text embeddings

    // After patch_embedding with stride (1,2,2): patches = [2, 4, 4] = 32 tokens
    let seqLen = 32

    let out = model.forward(x: x, t: t, context: context, seqLen: seqLen, y: nil)
    eval(out[0])
    // Output per sample: [out_dim, F, H/stride, W/stride]
    // With I2V, x is already cat([noise, cond]) so output channels = out_dim = 16
    // But spatial dims should match: [16, F*1, H_patches*2, W_patches*2]
    // Actually: F=2, H=8, W=8. Patch stride (1,2,2) -> grid (2,4,4).
    // Unpatchify: (2*1, 4*2, 4*2) = (2, 8, 8)
    XCTAssertEqual(out.count, 1)
    XCTAssertEqual(out[0].shape, [16, 2, 8, 8])
  }
}

// MARK: - S3.7: Weight Mapping Tests

final class WanTransformerWeightMappingTests: XCTestCase {

  func testExpectedKeyCount() {
    let keys = WanTransformerWeightMapping.expectedKeys()
    // 27 per block * 40 blocks + 15 global = 1095
    XCTAssertEqual(keys.count, 1095)
  }

  func testExpectedKeysContainBlockKeys() {
    let keys = Set(WanTransformerWeightMapping.expectedKeys())
    // Check first block
    XCTAssertTrue(keys.contains("blocks.0.self_attn.q.weight"))
    XCTAssertTrue(keys.contains("blocks.0.self_attn.q.bias"))
    XCTAssertTrue(keys.contains("blocks.0.self_attn.norm_q.weight"))
    XCTAssertTrue(keys.contains("blocks.0.cross_attn.k.weight"))
    XCTAssertTrue(keys.contains("blocks.0.ffn.0.weight"))
    XCTAssertTrue(keys.contains("blocks.0.ffn.0.bias"))
    XCTAssertTrue(keys.contains("blocks.0.ffn.2.weight"))
    XCTAssertTrue(keys.contains("blocks.0.ffn.2.bias"))
    XCTAssertTrue(keys.contains("blocks.0.norm3.weight"))
    XCTAssertTrue(keys.contains("blocks.0.norm3.bias"))
    XCTAssertTrue(keys.contains("blocks.0.modulation"))
    // Check last block
    XCTAssertTrue(keys.contains("blocks.39.self_attn.q.weight"))
    XCTAssertTrue(keys.contains("blocks.39.modulation"))
  }

  func testExpectedKeysContainGlobalKeys() {
    let keys = Set(WanTransformerWeightMapping.expectedKeys())
    XCTAssertTrue(keys.contains("patch_embedding.weight"))
    XCTAssertTrue(keys.contains("patch_embedding.bias"))
    XCTAssertTrue(keys.contains("text_embedding.0.weight"))
    XCTAssertTrue(keys.contains("text_embedding.0.bias"))
    XCTAssertTrue(keys.contains("text_embedding.2.weight"))
    XCTAssertTrue(keys.contains("text_embedding.2.bias"))
    XCTAssertTrue(keys.contains("time_embedding.0.weight"))
    XCTAssertTrue(keys.contains("time_embedding.0.bias"))
    XCTAssertTrue(keys.contains("time_embedding.2.weight"))
    XCTAssertTrue(keys.contains("time_embedding.2.bias"))
    XCTAssertTrue(keys.contains("time_projection.1.weight"))
    XCTAssertTrue(keys.contains("time_projection.1.bias"))
    XCTAssertTrue(keys.contains("head.head.weight"))
    XCTAssertTrue(keys.contains("head.head.bias"))
    XCTAssertTrue(keys.contains("head.modulation"))
  }

  func testExpectedShapes() {
    let shapes = WanTransformerWeightMapping.expectedShapes()
    XCTAssertEqual(shapes["blocks.0.self_attn.q.weight"], [5120, 5120])
    XCTAssertEqual(shapes["blocks.0.self_attn.q.bias"], [5120])
    XCTAssertEqual(shapes["blocks.0.ffn.0.weight"], [13824, 5120])
    XCTAssertEqual(shapes["blocks.0.ffn.2.weight"], [5120, 13824])
    XCTAssertEqual(shapes["blocks.0.modulation"], [1, 6, 5120])
    XCTAssertEqual(shapes["patch_embedding.weight"], [5120, 36, 1, 2, 2])
    XCTAssertEqual(shapes["text_embedding.0.weight"], [5120, 4096])
    XCTAssertEqual(shapes["time_projection.1.weight"], [30720, 5120])
    XCTAssertEqual(shapes["head.head.weight"], [64, 5120])
    XCTAssertEqual(shapes["head.modulation"], [1, 2, 5120])
  }

  func testValidateKeysComplete() {
    var weights: [String: MLXArray] = [:]
    for key in WanTransformerWeightMapping.expectedKeys() {
      weights[key] = MLXArray(Float(0))
    }
    let (missing, unexpected) = WanTransformerWeightMapping.validateKeys(weights)
    XCTAssertTrue(missing.isEmpty, "No keys should be missing: \(missing.prefix(5))")
    XCTAssertTrue(unexpected.isEmpty, "No keys should be unexpected")
  }

  func testValidateKeysMissingDetected() {
    let weights: [String: MLXArray] = [
      "patch_embedding.weight": MLXArray(Float(0))
    ]
    let (missing, _) = WanTransformerWeightMapping.validateKeys(weights)
    XCTAssertFalse(missing.isEmpty, "Should detect missing keys")
    XCTAssertTrue(missing.contains("blocks.0.self_attn.q.weight"))
  }
}
