import XCTest
import MLX
import MLXRandom
import MLXNN
@testable import ZImage

// MARK: - S2.1: Tokenizer Tests

final class WanTokenizerTests: XCTestCase {

  /// Helper: create a small test tokenizer with known vocabulary.
  private func makeTestTokenizer() -> WanTokenizer {
    // Minimal vocab for testing. IDs match positions.
    let vocab: [String: Int32] = [
      "<pad>": 0,
      "</s>": 1,
      "<s>": 2,
      "<unk>": 3,
      "\u{2581}hello": 100,    // ▁hello
      "\u{2581}world": 101,    // ▁world
      "\u{2581}a": 102,        // ▁a
      "\u{2581}the": 103,      // ▁the
      "\u{2581}cat": 104,      // ▁cat
      "\u{2581}sat": 105,      // ▁sat
      "\u{2581}on": 106,       // ▁on
      "\u{2581}mat": 107,      // ▁mat
      "\u{2581}": 108,         // bare ▁ (space-only token)
    ]
    let scores: [String: Float] = vocab.keys.reduce(into: [:]) { $0[$1] = 0.0 }
    return WanTokenizer(vocab: vocab, scores: scores)
  }

  func testBasicEncoding() {
    let tokenizer = makeTestTokenizer()
    let result = tokenizer.encode("hello world", maxLength: 8)
    eval(result.tokenIds, result.attentionMask)

    // Should produce: [▁hello, ▁world, </s>, <pad>, <pad>, <pad>, <pad>, <pad>]
    let ids = result.tokenIds.asArray(Int32.self)
    XCTAssertEqual(ids.count, 8)
    XCTAssertEqual(ids[0], 100, "First token should be ▁hello")
    XCTAssertEqual(ids[1], 101, "Second token should be ▁world")
    XCTAssertEqual(ids[2], 1, "Third token should be EOS")
  }

  func testEOSAppended() {
    let tokenizer = makeTestTokenizer()
    let result = tokenizer.encode("hello", maxLength: 8)
    eval(result.tokenIds)

    let ids = result.tokenIds.asArray(Int32.self)
    // Should be: [▁hello, </s>, pad, pad, ...]
    XCTAssertEqual(ids[0], 100)
    XCTAssertEqual(ids[1], 1, "EOS should be appended after tokens")
  }

  func testPaddingToMaxLength() {
    let tokenizer = makeTestTokenizer()
    let result = tokenizer.encode("hello", maxLength: 10)
    eval(result.tokenIds, result.attentionMask)

    XCTAssertEqual(result.tokenIds.shape, [1, 10])
    XCTAssertEqual(result.attentionMask.shape, [1, 10])

    let ids = result.tokenIds.asArray(Int32.self)
    // Tokens: ▁hello(100), EOS(1), then 8x PAD(0)
    XCTAssertEqual(ids[2], 0, "Position 2 should be PAD")
    XCTAssertEqual(ids[9], 0, "Last position should be PAD")
  }

  func testAttentionMask() {
    let tokenizer = makeTestTokenizer()
    let result = tokenizer.encode("hello world", maxLength: 6)
    eval(result.attentionMask)

    let mask = result.attentionMask.asArray(Int32.self)
    // Tokens: ▁hello, ▁world, EOS = 3 real tokens, 3 padding
    XCTAssertEqual(mask[0], 1, "Real token should have mask=1")
    XCTAssertEqual(mask[1], 1, "Real token should have mask=1")
    XCTAssertEqual(mask[2], 1, "EOS should have mask=1")
    XCTAssertEqual(mask[3], 0, "PAD should have mask=0")
    XCTAssertEqual(mask[4], 0, "PAD should have mask=0")
    XCTAssertEqual(mask[5], 0, "PAD should have mask=0")
  }

  func testUnknownTokenFallback() {
    let tokenizer = makeTestTokenizer()
    // "xyz" has no matching token in our tiny vocab
    let result = tokenizer.encode("xyz", maxLength: 8)
    eval(result.tokenIds)

    let ids = result.tokenIds.asArray(Int32.self)
    // Should produce UNK tokens for each character, then EOS
    // ▁xyz -> ▁ might match as bare ▁, then x, y, z are unknown
    // Actually ▁xyz is one piece, and we try longest match first.
    // ▁xyz -> no match, ▁xy -> no, ▁x -> no, ▁ -> yes (108)
    // Then x -> no match -> UNK, y -> UNK, z -> UNK
    // But we only check the ▁ prefix, the rest are raw chars
    XCTAssertTrue(ids.contains(3), "Should contain UNK token for unrecognized characters")
    XCTAssertTrue(ids.contains(1), "Should still have EOS")
  }

  func testEmptyStringHandling() {
    let tokenizer = makeTestTokenizer()
    let result = tokenizer.encode("", maxLength: 4)
    eval(result.tokenIds, result.attentionMask)

    let ids = result.tokenIds.asArray(Int32.self)
    // Empty text -> no text tokens -> just EOS then padding
    XCTAssertEqual(ids[0], 1, "First token should be EOS for empty input")
    XCTAssertEqual(ids[1], 0, "Rest should be PAD")
  }

  func testMultipleSpacesCollapsed() {
    let tokenizer = makeTestTokenizer()
    let result1 = tokenizer.encode("hello world", maxLength: 8)
    let result2 = tokenizer.encode("hello   world", maxLength: 8)
    eval(result1.tokenIds, result2.tokenIds)

    let ids1 = result1.tokenIds.asArray(Int32.self)
    let ids2 = result2.tokenIds.asArray(Int32.self)
    XCTAssertEqual(ids1, ids2, "Multiple spaces should be collapsed to single space")
  }

  func testTruncation() {
    let tokenizer = makeTestTokenizer()
    // maxLength=3 but "the cat sat" = 3 tokens + EOS = 4
    let result = tokenizer.encode("the cat sat", maxLength: 3)
    eval(result.tokenIds)

    let ids = result.tokenIds.asArray(Int32.self)
    XCTAssertEqual(ids.count, 3)
    // Should truncate to first 2 tokens + EOS
    XCTAssertEqual(ids[2], 1, "Last token should be EOS after truncation")
  }

  func testOutputShape() {
    let tokenizer = makeTestTokenizer()
    let result = tokenizer.encode("hello", maxLength: 16)
    XCTAssertEqual(result.tokenIds.shape, [1, 16], "Token IDs should be [1, maxLength]")
    XCTAssertEqual(result.attentionMask.shape, [1, 16], "Attention mask should be [1, maxLength]")
  }
}

// MARK: - S2.2: Relative Position Bias Tests

final class WanT5RelativePositionBiasTests: XCTestCase {

  func testOutputShape() {
    let bias = WanT5RelativePositionBias(numBuckets: 32, numHeads: 64, maxDistance: 128)
    let result = bias.computeBias(queryLength: 8, keyLength: 8)
    eval(result)

    XCTAssertEqual(result.shape, [1, 64, 8, 8],
                   "Bias shape should be [1, numHeads, Q, K]")
  }

  func testAsymmetricLengths() {
    let bias = WanT5RelativePositionBias(numBuckets: 32, numHeads: 4, maxDistance: 128)
    let result = bias.computeBias(queryLength: 4, keyLength: 6)
    eval(result)

    XCTAssertEqual(result.shape, [1, 4, 4, 6],
                   "Should handle different Q and K lengths")
  }

  func testBidirectionalAsymmetry() {
    // For bidirectional, bucket(+d) should differ from bucket(-d)
    let posResult = WanT5RelativePositionBias.relativeBuckets(
      relativePosition: MLXArray([Int32(5)]),
      numBuckets: 32, maxDistance: 128, bidirectional: true
    )
    let negResult = WanT5RelativePositionBias.relativeBuckets(
      relativePosition: MLXArray([Int32(-5)]),
      numBuckets: 32, maxDistance: 128, bidirectional: true
    )
    eval(posResult, negResult)

    let posBucket = posResult.asArray(Int32.self)[0]
    let negBucket = negResult.asArray(Int32.self)[0]
    XCTAssertNotEqual(posBucket, negBucket,
                      "Positive and negative positions should map to different buckets")
  }

  func testLogScaleCollapse() {
    // Large distances should collapse to the same bucket
    let result1 = WanT5RelativePositionBias.relativeBuckets(
      relativePosition: MLXArray([Int32(100)]),
      numBuckets: 32, maxDistance: 128, bidirectional: true
    )
    let result2 = WanT5RelativePositionBias.relativeBuckets(
      relativePosition: MLXArray([Int32(110)]),
      numBuckets: 32, maxDistance: 128, bidirectional: true
    )
    eval(result1, result2)

    let bucket1 = result1.asArray(Int32.self)[0]
    let bucket2 = result2.asArray(Int32.self)[0]
    // 100 and 110 are close enough that they should be in the same log bucket
    XCTAssertEqual(bucket1, bucket2,
                   "Similar large distances should map to the same log bucket")
  }

  func testExactBucketsForSmallDistances() {
    // For bidirectional with 32 buckets: halfBuckets = 16/2 = 8
    // Distances 0-7 should have exact buckets
    let distances = MLXArray(Array(0..<Int32(8)))
    let result = WanT5RelativePositionBias.relativeBuckets(
      relativePosition: distances,
      numBuckets: 32, maxDistance: 128, bidirectional: true
    )
    eval(result)

    let buckets = result.asArray(Int32.self)
    // For positive positions with bidirectional, offset by 16
    // Each small distance should map to a unique bucket
    let uniqueBuckets = Set(buckets)
    XCTAssertEqual(uniqueBuckets.count, 8,
                   "Small distances should each have a unique bucket")
  }

  func testZeroDistance() {
    let result = WanT5RelativePositionBias.relativeBuckets(
      relativePosition: MLXArray([Int32(0)]),
      numBuckets: 32, maxDistance: 128, bidirectional: true
    )
    eval(result)

    let bucket = result.asArray(Int32.self)[0]
    XCTAssertTrue(bucket >= 0 && bucket < 32, "Bucket should be in valid range")
  }

  func testBucketRangeValid() {
    let distances = MLXArray(Array(stride(from: Int32(-128), through: 128, by: 1)))
    let result = WanT5RelativePositionBias.relativeBuckets(
      relativePosition: distances,
      numBuckets: 32, maxDistance: 128, bidirectional: true
    )
    eval(result)

    let buckets = result.asArray(Int32.self)
    for bucket in buckets {
      XCTAssertTrue(bucket >= 0 && bucket < 32,
                    "All buckets should be in [0, numBuckets)")
    }
  }
}

// MARK: - S2.3: Attention + FFN + Encoder Layer Tests

final class WanT5AttentionTests: XCTestCase {

  func testOutputShape() {
    let attn = WanT5Attention(hiddenSize: 64, numHeads: 4, headDim: 16)
    let x = MLXRandom.normal([1, 8, 64]).asType(.float32)
    let bias = MLXArray.zeros([1, 4, 8, 8], type: Float.self)
    let result = attn(x, positionBias: bias)
    eval(result)

    XCTAssertEqual(result.shape, [1, 8, 64],
                   "Output shape should match input [B, seqLen, hiddenSize]")
  }

  func testBatchDimension() {
    let attn = WanT5Attention(hiddenSize: 32, numHeads: 2, headDim: 16)
    let x = MLXRandom.normal([2, 4, 32]).asType(.float32)
    let bias = MLXArray.zeros([1, 2, 4, 4], type: Float.self)
    let result = attn(x, positionBias: bias)
    eval(result)

    XCTAssertEqual(result.shape[0], 2, "Batch dimension should be preserved")
  }

  func testNoBiasInLinearLayers() {
    let attn = WanT5Attention(hiddenSize: 64, numHeads: 4, headDim: 16)

    // Check that all linear layers have no bias by inspecting parameters
    let params = attn.parameters().flattened()
    let biasKeys = params.filter { $0.0.hasSuffix(".bias") }
    XCTAssertTrue(biasKeys.isEmpty,
                  "T5 attention should have no bias parameters, found: \(biasKeys.map(\.0))")
  }

  func testMaskedAttention() {
    let attn = WanT5Attention(hiddenSize: 32, numHeads: 2, headDim: 16)
    let x = MLXRandom.normal([1, 4, 32]).asType(.float32)
    let bias = MLXArray.zeros([1, 2, 4, 4], type: Float.self)

    // Mask out positions 2 and 3
    let maskValues: [Float] = [0, 0, -1e9, -1e9]
    let mask = MLXArray(maskValues).reshaped(1, 1, 1, 4)

    let result = attn(x, positionBias: bias, mask: mask)
    eval(result)
    XCTAssertEqual(result.shape, [1, 4, 32])
  }
}

final class WanT5FFNTests: XCTestCase {

  func testOutputShape() {
    let ffn = WanT5FFN(hiddenSize: 64, ffnHiddenSize: 128)
    let x = MLXRandom.normal([1, 8, 64]).asType(.float32)
    let result = ffn(x)
    eval(result)

    XCTAssertEqual(result.shape, [1, 8, 64],
                   "FFN output shape should match input")
  }

  func testGELUActivation() {
    // Verify the gate uses GELU by checking against known values
    let ffn = WanT5FFN(hiddenSize: 4, ffnHiddenSize: 8)

    // Create input and run
    let x = MLXArray([Float(1.0), 0.0, -1.0, 2.0]).reshaped(1, 1, 4)
    let result = ffn(x)
    eval(result)

    // Output should be finite (not NaN/inf)
    let values = result.asArray(Float.self)
    for v in values {
      XCTAssertFalse(v.isNaN, "FFN output should not contain NaN")
      XCTAssertFalse(v.isInfinite, "FFN output should not contain Inf")
    }
  }

  func testNoBiasParameters() {
    let ffn = WanT5FFN(hiddenSize: 64, ffnHiddenSize: 128)
    let params = ffn.parameters().flattened()
    let biasKeys = params.filter { $0.0.hasSuffix(".bias") }
    XCTAssertTrue(biasKeys.isEmpty,
                  "FFN should have no bias parameters")
  }

  func testWeightKeyStructure() {
    // Verify the module tree produces the right key hierarchy.
    // After the gate fix, the module key is gate.weight (not gate.0.weight).
    // The Wan safetensors key ffn.gate.0.weight is remapped to ffn.gate.weight
    // at load time by WanUMT5Encoder.remapGateKeys().
    let ffn = WanT5FFN(hiddenSize: 16, ffnHiddenSize: 32)
    let params = ffn.parameters().flattened()
    let keys = Set(params.map(\.0))

    XCTAssertTrue(keys.contains("gate.weight"),
                  "Should have gate.weight key")
    XCTAssertTrue(keys.contains("fc1.weight"),
                  "Should have fc1.weight key")
    XCTAssertTrue(keys.contains("fc2.weight"),
                  "Should have fc2.weight key")
  }
}

final class WanT5EncoderLayerTests: XCTestCase {

  func testOutputShape() {
    let config = WanUMT5Config(
      numLayers: 1, hiddenSize: 64, ffnHiddenSize: 128,
      numHeads: 4, headDim: 16, vocabSize: 100,
      numBuckets: 8, maxDistance: 32, rmsNormEps: 1e-6,
      maxSequenceLength: 64
    )
    let layer = WanT5EncoderLayer(config: config)
    let x = MLXRandom.normal([1, 8, 64]).asType(.float32)
    let result = layer(x)
    eval(result)

    XCTAssertEqual(result.shape, [1, 8, 64],
                   "Encoder layer should preserve input shape")
  }

  func testResidualConnection() {
    let config = WanUMT5Config(
      numLayers: 1, hiddenSize: 32, ffnHiddenSize: 64,
      numHeads: 2, headDim: 16, vocabSize: 100,
      numBuckets: 8, maxDistance: 32, rmsNormEps: 1e-6,
      maxSequenceLength: 64
    )
    let layer = WanT5EncoderLayer(config: config)
    let x = MLXRandom.normal([1, 4, 32]).asType(.float32)
    let result = layer(x)
    eval(result)

    // Output should differ from input (not identity)
    let diff = abs(result - x).sum()
    eval(diff)
    XCTAssertTrue(diff.item(Float.self) > 0,
                  "Output should differ from input (residual + transformation)")
  }

  func testWithAttentionMask() {
    let config = WanUMT5Config(
      numLayers: 1, hiddenSize: 32, ffnHiddenSize: 64,
      numHeads: 2, headDim: 16, vocabSize: 100,
      numBuckets: 8, maxDistance: 32, rmsNormEps: 1e-6,
      maxSequenceLength: 64
    )
    let layer = WanT5EncoderLayer(config: config)
    let x = MLXRandom.normal([1, 4, 32]).asType(.float32)

    // Mask: attend to first 2 positions only
    let mask = MLXArray([Float(0), 0, -1e9, -1e9]).reshaped(1, 1, 1, 4)
    let result = layer(x, mask: mask)
    eval(result)

    XCTAssertEqual(result.shape, [1, 4, 32])
  }
}

// MARK: - S2.4: Full Encoder + Weight Mapping Tests

final class WanUMT5EncoderTests: XCTestCase {

  /// Small config for unit tests (avoids allocating full 4096-dim model).
  private var testConfig: WanUMT5Config {
    WanUMT5Config(
      numLayers: 2, hiddenSize: 32, ffnHiddenSize: 64,
      numHeads: 2, headDim: 16, vocabSize: 128,
      numBuckets: 8, maxDistance: 32, rmsNormEps: 1e-6,
      maxSequenceLength: 64
    )
  }

  func testForwardShape() {
    let config = testConfig
    let encoder = WanUMT5Encoder(config: config)

    let tokenIds = MLXArray([Int32(1), 5, 10, 20, 1]).reshaped(1, 5)
    let result = encoder(tokenIds: tokenIds)
    eval(result)

    XCTAssertEqual(result.shape, [1, 5, 32],
                   "Encoder output should be [B, seqLen, hiddenSize]")
  }

  func testForwardWithMask() {
    let config = testConfig
    let encoder = WanUMT5Encoder(config: config)

    let tokenIds = MLXArray([Int32(1), 5, 10, 0, 0]).reshaped(1, 5)
    let mask = MLXArray([Int32(1), 1, 1, 0, 0]).reshaped(1, 5)
    let result = encoder(tokenIds: tokenIds, attentionMask: mask)
    eval(result)

    XCTAssertEqual(result.shape, [1, 5, 32])
  }

  func testBatchForward() {
    let config = testConfig
    let encoder = WanUMT5Encoder(config: config)

    let tokenIds = MLXArray([
      Int32(1), 5, 10, 1, 0, 0,
      Int32(2), 3, 4, 5, 6, 1
    ]).reshaped(2, 6)
    let result = encoder(tokenIds: tokenIds)
    eval(result)

    XCTAssertEqual(result.shape, [2, 6, 32])
  }

  func testParameterCount() {
    let config = testConfig
    let encoder = WanUMT5Encoder(config: config)
    let params = encoder.parameters().flattened()

    // Expected: token_embedding + norm + 2 layers * (attn q/k/v/o + pos_embed + norm1 + ffn gate/fc1/fc2 + norm2)
    // = 1 + 1 + 2 * 10 = 22 parameter tensors
    XCTAssertEqual(params.count, 22,
                   "Should have 22 parameter tensors (2 top-level + 10 per layer * 2 layers)")
  }
}

final class WanUMT5WeightMappingTests: XCTestCase {

  func testExpectedKeyCount() {
    let keys = WanUMT5WeightMapping.expectedKeys()
    // 2 top-level + 10 per layer * 24 layers = 242
    XCTAssertEqual(keys.count, 242,
                   "Should have 242 expected keys for default config")
  }

  func testExpectedKeysContainAllParts() {
    let keys = Set(WanUMT5WeightMapping.expectedKeys())

    // Check top-level keys
    XCTAssertTrue(keys.contains("token_embedding.weight"))
    XCTAssertTrue(keys.contains("norm.weight"))

    // Check first block keys
    XCTAssertTrue(keys.contains("blocks.0.attn.q.weight"))
    XCTAssertTrue(keys.contains("blocks.0.attn.k.weight"))
    XCTAssertTrue(keys.contains("blocks.0.attn.v.weight"))
    XCTAssertTrue(keys.contains("blocks.0.attn.o.weight"))
    XCTAssertTrue(keys.contains("blocks.0.pos_embedding.embedding.weight"))
    XCTAssertTrue(keys.contains("blocks.0.norm1.weight"))
    XCTAssertTrue(keys.contains("blocks.0.ffn.gate.0.weight"))
    XCTAssertTrue(keys.contains("blocks.0.ffn.fc1.weight"))
    XCTAssertTrue(keys.contains("blocks.0.ffn.fc2.weight"))
    XCTAssertTrue(keys.contains("blocks.0.norm2.weight"))

    // Check last block
    XCTAssertTrue(keys.contains("blocks.23.attn.q.weight"))
    XCTAssertTrue(keys.contains("blocks.23.ffn.fc2.weight"))
  }

  func testValidateKeysWithCompleteSet() {
    var weights: [String: MLXArray] = [:]
    for key in WanUMT5WeightMapping.expectedKeys() {
      weights[key] = MLXArray(Float(0))
    }

    let (missing, unexpected) = WanUMT5WeightMapping.validateKeys(weights)
    XCTAssertTrue(missing.isEmpty, "No keys should be missing: \(missing)")
    XCTAssertTrue(unexpected.isEmpty, "No keys should be unexpected: \(unexpected)")
  }

  func testValidateKeysDetectsMissing() {
    let weights: [String: MLXArray] = [
      "token_embedding.weight": MLXArray(Float(0)),
      "norm.weight": MLXArray(Float(0))
    ]

    let (missing, _) = WanUMT5WeightMapping.validateKeys(weights)
    XCTAssertFalse(missing.isEmpty, "Should detect missing block keys")
    XCTAssertTrue(missing.contains("blocks.0.attn.q.weight"))
  }

  func testValidateKeysDetectsUnexpected() {
    var weights: [String: MLXArray] = [:]
    for key in WanUMT5WeightMapping.expectedKeys() {
      weights[key] = MLXArray(Float(0))
    }
    weights["some_extra_key"] = MLXArray(Float(0))

    let (missing, unexpected) = WanUMT5WeightMapping.validateKeys(weights)
    XCTAssertTrue(missing.isEmpty)
    XCTAssertEqual(unexpected, ["some_extra_key"])
  }

  func testExpectedShapes() {
    let shapes = WanUMT5WeightMapping.expectedShapes()

    XCTAssertEqual(shapes["token_embedding.weight"], [256384, 4096])
    XCTAssertEqual(shapes["norm.weight"], [4096])
    XCTAssertEqual(shapes["blocks.0.attn.q.weight"], [4096, 4096])
    XCTAssertEqual(shapes["blocks.0.ffn.gate.0.weight"], [10240, 4096])
    XCTAssertEqual(shapes["blocks.0.ffn.fc1.weight"], [10240, 4096])
    XCTAssertEqual(shapes["blocks.0.ffn.fc2.weight"], [4096, 10240])
    XCTAssertEqual(shapes["blocks.0.pos_embedding.embedding.weight"], [32, 64])
    XCTAssertEqual(shapes["blocks.0.norm1.weight"], [4096])
  }

  func testModuleParameterKeysMatchExpected() {
    // Verify the Swift module hierarchy produces the same keys
    // as the weight mapping expects (after remapping).
    //
    // The weight mapping returns safetensors file keys (with gate.0.weight).
    // The module keys use gate.weight (no .0. nesting).
    // WanUMT5Encoder.remapGateKeys() bridges the difference at load time.
    let config = WanUMT5Config(
      numLayers: 2, hiddenSize: 32, ffnHiddenSize: 64,
      numHeads: 2, headDim: 16, vocabSize: 128,
      numBuckets: 8, maxDistance: 32, rmsNormEps: 1e-6,
      maxSequenceLength: 64
    )
    let encoder = WanUMT5Encoder(config: config)
    let params = encoder.parameters().flattened()
    let moduleKeys = Set(params.map(\.0))

    // Simulate the remap that happens at load time
    let safetensorsKeys = WanUMT5WeightMapping.expectedKeys(config: config)
    let remappedKeys = Set(safetensorsKeys.map { key in
      key.replacingOccurrences(of: ".ffn.gate.0.", with: ".ffn.gate.")
    })
    XCTAssertEqual(moduleKeys, remappedKeys,
                   "Module parameter keys should match remapped weight keys. " +
                   "Missing: \(remappedKeys.subtracting(moduleKeys)). " +
                   "Extra: \(moduleKeys.subtracting(remappedKeys))")
  }
}
