import XCTest
import MLX
import MLXRandom
@testable import ZImage

// MARK: - LTX-2 Phase 1 & Phase 2 Architecture Smoke Tests
//
// These tests verify that all LTX-2 modules instantiate correctly and produce
// tensors of the expected shape. No real weights required -- random init is fine
// since we only care about layer connectivity and dimension math.

final class LTX2SmokeTest: XCTestCase {

  // -----------------------------------------------
  // MARK: 1 -- VAE Config
  // -----------------------------------------------

  func testVAEConfigDefault() {
    let cfg = LTX2VideoVAEConfig.default

    // Architecture constants
    XCTAssertEqual(cfg.inChannels, 3, "Input should be RGB")
    XCTAssertEqual(cfg.latentChannels, 128, "Latent channels should be 128")
    XCTAssertEqual(cfg.patchSize, 4, "Patch size should be 4")

    // Computed compression factors
    // Spatial: patchSize(4) * compressSpace(2) * compressAll(2) * compressAll(2) = 32
    XCTAssertEqual(cfg.spatialCompression, 32, "Spatial compression = 32x")
    // Temporal: compressTime(2) * compressAll(2) * compressAll(2) = 8
    XCTAssertEqual(cfg.temporalCompression, 8, "Temporal compression = 8x")

    // Block counts
    XCTAssertEqual(cfg.encoderBlocks.count, 9, "Encoder has 9 blocks")
    XCTAssertEqual(cfg.decoderBlocks.count, 7, "Decoder has 7 blocks (reversed)")

    // Timestep conditioning
    XCTAssertTrue(cfg.timestepConditioning, "Timestep conditioning should be on")
    XCTAssertEqual(cfg.decodeNoiseScale, 0.025, accuracy: 1e-6)
    XCTAssertEqual(cfg.decodeTimestep, 0.05, accuracy: 1e-6)
  }

  // -----------------------------------------------
  // MARK: 2 -- VAE Architecture (encode -> decode round-trip)
  // -----------------------------------------------

  func testVAEArchitectureRoundTrip() {
    let cfg = LTX2VideoVAEConfig.default
    let vae = LTX2VAE(config: cfg)

    // Verify module properties
    XCTAssertEqual(vae.latentChannels, 128)
    XCTAssertEqual(vae.spatialCompression, 32)
    XCTAssertEqual(vae.temporalCompression, 8)

    // Input: (B=1, C=3, F=1, H=64, W=64) -- single frame, 64x64 RGB
    // F must satisfy (F-1) % 8 == 0, so F=1 works.
    // H and W must be divisible by spatialCompression (32), so 64 works.
    let input = MLXRandom.normal([1, 3, 1, 64, 64])
    MLX.eval(input)

    // Encode: (1, 3, 1, 64, 64) -> (1, 128, 1, 2, 2)
    // H_lat = 64/32 = 2, W_lat = 64/32 = 2, F_lat = 1
    let latent = vae.encode(input)
    MLX.eval(latent)

    XCTAssertEqual(latent.ndim, 5, "Latent should be 5D")
    XCTAssertEqual(latent.dim(0), 1, "Batch dim should be 1")
    XCTAssertEqual(latent.dim(1), 128, "Latent channels should be 128")
    XCTAssertEqual(latent.dim(2), 1, "Temporal dim should be 1")
    XCTAssertEqual(latent.dim(3), 2, "H_lat = 64/32 = 2")
    XCTAssertEqual(latent.dim(4), 2, "W_lat = 64/32 = 2")

    // Decode: (1, 128, 1, 2, 2) -> (1, 3, 1, 64, 64)
    let decoded = vae.decode(latent)
    MLX.eval(decoded)

    XCTAssertEqual(decoded.ndim, 5, "Decoded should be 5D")
    XCTAssertEqual(decoded.dim(0), 1, "Batch dim preserved")
    XCTAssertEqual(decoded.dim(1), 3, "RGB channels restored")
    XCTAssertEqual(decoded.dim(2), 1, "Temporal dim restored")
    XCTAssertEqual(decoded.dim(3), 64, "Height restored")
    XCTAssertEqual(decoded.dim(4), 64, "Width restored")

    // Values should be finite (no NaN/Inf from uninitialized weights)
    let hasNaN = MLX.any(MLX.isNaN(decoded)).item(Bool.self)
    XCTAssertFalse(hasNaN, "Decoded output should not contain NaN")
  }

  // -----------------------------------------------
  // MARK: 3 -- Patchify / Unpatchify round-trip
  // -----------------------------------------------

  func testPatchifyRoundTrip() {
    // Input: (B=1, C=3, F=1, H=16, W=16), patchSize=4
    let input = MLXRandom.normal([1, 3, 1, 16, 16])
    MLX.eval(input)

    // Patchify: (1, 3, 1, 16, 16) -> (1, 3*4*4, 1, 4, 4) = (1, 48, 1, 4, 4)
    let patched = LTX2Patchify.patchify(input, patchSizeHW: 4, patchSizeT: 1)
    MLX.eval(patched)

    XCTAssertEqual(patched.dim(0), 1, "Batch preserved")
    XCTAssertEqual(patched.dim(1), 48, "Channels = 3 * 4 * 4 = 48")
    XCTAssertEqual(patched.dim(2), 1, "Temporal unchanged (patchSizeT=1)")
    XCTAssertEqual(patched.dim(3), 4, "H/4 = 4")
    XCTAssertEqual(patched.dim(4), 4, "W/4 = 4")

    // Unpatchify: (1, 48, 1, 4, 4) -> (1, 3, 1, 16, 16)
    let unpatched = LTX2Patchify.unpatchify(patched, patchSizeHW: 4, patchSizeT: 1)
    MLX.eval(unpatched)

    XCTAssertEqual(unpatched.dim(0), 1)
    XCTAssertEqual(unpatched.dim(1), 3, "RGB restored")
    XCTAssertEqual(unpatched.dim(2), 1, "Temporal restored")
    XCTAssertEqual(unpatched.dim(3), 16, "Height restored")
    XCTAssertEqual(unpatched.dim(4), 16, "Width restored")

    // Round-trip should be exact (just reshape/permute, no learned params)
    let diff = MLX.abs(input - unpatched)
    let maxDiff = MLX.max(diff).item(Float.self)
    XCTAssertEqual(maxDiff, 0.0, accuracy: 1e-6, "Patchify round-trip should be exact")
  }

  func testPatchifyWithTemporalPatch() {
    // Test with temporal patching too
    // Input: (B=1, C=3, F=2, H=8, W=8), patchSizeHW=4, patchSizeT=2
    let input = MLXRandom.normal([1, 3, 2, 8, 8])
    MLX.eval(input)

    let patched = LTX2Patchify.patchify(input, patchSizeHW: 4, patchSizeT: 2)
    MLX.eval(patched)

    // Channels: 3 * 4 * 4 * 2 = 96, F: 2/2 = 1, H: 8/4 = 2, W: 8/4 = 2
    XCTAssertEqual(patched.dim(1), 96, "Channels = 3*4*4*2")
    XCTAssertEqual(patched.dim(2), 1, "F/patchSizeT = 1")
    XCTAssertEqual(patched.dim(3), 2, "H/patchSizeHW = 2")
    XCTAssertEqual(patched.dim(4), 2, "W/patchSizeHW = 2")

    let unpatched = LTX2Patchify.unpatchify(patched, patchSizeHW: 4, patchSizeT: 2)
    MLX.eval(unpatched)

    XCTAssertEqual(unpatched.shape, input.shape, "Round-trip shape must match")
    let maxDiff = MLX.max(MLX.abs(input - unpatched)).item(Float.self)
    XCTAssertEqual(maxDiff, 0.0, accuracy: 1e-6, "Temporal patchify round-trip should be exact")
  }

  // -----------------------------------------------
  // MARK: 4 -- Text Encoder Config
  // -----------------------------------------------

  func testTextEncoderConfigDefault() {
    // Default is LTX-2.3 (hasPromptAdaLN = true)
    let cfg = LTX2TextEncoderConfig()

    XCTAssertTrue(cfg.hasPromptAdaLN, "Default should be LTX-2.3 with prompt adaln")
    XCTAssertEqual(cfg.hiddenDim, 3840, "Gemma 3 hidden dim")
    XCTAssertEqual(cfg.numLayers, 49, "48 layers + 1 embedding")
    XCTAssertEqual(cfg.audioDim, 2048)

    // Gemma config
    XCTAssertEqual(cfg.gemma.vocabSize, 262208)
    XCTAssertEqual(cfg.gemma.hiddenSize, 3840)
    XCTAssertEqual(cfg.gemma.numHiddenLayers, 48)
    XCTAssertEqual(cfg.gemma.numAttentionHeads, 16)
    XCTAssertEqual(cfg.gemma.numKeyValueHeads, 8)
    XCTAssertEqual(cfg.gemma.headDim, 256)
    XCTAssertEqual(cfg.gemma.numKeyValueGroups, 2, "GQA: 16/8 = 2 groups")
    XCTAssertEqual(cfg.gemma.totalHiddenStates, 49)

    // Feature extractor (V2 for LTX-2.3)
    XCTAssertEqual(cfg.featureExtractor.inputDim, 3840 * 49, "hiddenDim * numLayers")
    XCTAssertTrue(cfg.featureExtractor.useV2, "LTX-2.3 uses V2 feature extractor")
    XCTAssertEqual(cfg.featureExtractor.videoOutputDim, 4096)
    XCTAssertEqual(cfg.featureExtractor.audioOutputDim, 2048)

    // Video connector (LTX-2.3: deeper, wider)
    XCTAssertEqual(cfg.videoConnector.dim, 4096)
    XCTAssertEqual(cfg.videoConnector.numHeads, 32)
    XCTAssertEqual(cfg.videoConnector.headDim, 128)
    XCTAssertEqual(cfg.videoConnector.numLayers, 8)
    XCTAssertEqual(cfg.videoConnector.innerDim, 4096, "32 * 128 = 4096")
    XCTAssertTrue(cfg.videoConnector.hasGateLogits)

    // Audio connector (LTX-2.3: separate, narrower)
    XCTAssertEqual(cfg.audioConnector.dim, 2048)
    XCTAssertEqual(cfg.audioConnector.numHeads, 32)
    XCTAssertEqual(cfg.audioConnector.headDim, 64)
    XCTAssertEqual(cfg.audioConnector.numLayers, 8)
    XCTAssertEqual(cfg.audioConnector.innerDim, 2048, "32 * 64 = 2048")
  }

  func testTextEncoderConfigLTX2Original() {
    // LTX-2 original: hasPromptAdaLN = false
    let cfg = LTX2TextEncoderConfig(hasPromptAdaLN: false)

    XCTAssertFalse(cfg.hasPromptAdaLN)

    // Shared 3840-dim connector for both video and audio
    XCTAssertEqual(cfg.videoConnector.dim, 3840)
    XCTAssertEqual(cfg.videoConnector.numHeads, 30)
    XCTAssertEqual(cfg.videoConnector.numLayers, 2)
    XCTAssertFalse(cfg.videoConnector.hasGateLogits)

    // Feature extractor V1
    XCTAssertFalse(cfg.featureExtractor.useV2)
    XCTAssertEqual(cfg.featureExtractor.videoOutputDim, 3840)
  }

  // -----------------------------------------------
  // MARK: 5 -- Connector1D Architecture
  // -----------------------------------------------

  func testConnector1DForwardPass() {
    // Use a small config so the test runs fast with random weights
    let connectorCfg = LTX2ConnectorConfig(
      dim: 64,
      numHeads: 4,
      headDim: 16,
      numLayers: 2,
      numLearnableRegisters: 8,
      positionalEmbeddingTheta: 10000.0,
      positionalEmbeddingMaxPos: [1],
      hasGateLogits: false
    )

    let connector = LTX2Connector1D(config: connectorCfg)

    // Input: (B=1, S=16, dim=64) -- 16 tokens of 64-dim features
    let embeddings = MLXRandom.normal([1, 16, 64])
    MLX.eval(embeddings)

    // Forward without mask (no padding replacement)
    let (output, _) = connector(hiddenStates: embeddings, attentionMask: nil)
    MLX.eval(output)

    XCTAssertEqual(output.dim(0), 1, "Batch preserved")
    XCTAssertEqual(output.dim(1), 16, "Sequence length preserved")
    XCTAssertEqual(output.dim(2), 64, "Dimension preserved")

    let hasNaN = MLX.any(MLX.isNaN(output)).item(Bool.self)
    XCTAssertFalse(hasNaN, "Connector output should not contain NaN")
  }

  func testConnector1DWithGateLogits() {
    // LTX-2.3 style with gate logits
    let connectorCfg = LTX2ConnectorConfig(
      dim: 64,
      numHeads: 4,
      headDim: 16,
      numLayers: 2,
      numLearnableRegisters: 8,
      positionalEmbeddingTheta: 10000.0,
      positionalEmbeddingMaxPos: [4096],
      hasGateLogits: true
    )

    let connector = LTX2Connector1D(config: connectorCfg)

    let embeddings = MLXRandom.normal([1, 16, 64])
    MLX.eval(embeddings)

    let (output, _) = connector(hiddenStates: embeddings, attentionMask: nil)
    MLX.eval(output)

    XCTAssertEqual(output.shape, [1, 16, 64], "Shape preserved with gate logits")

    let hasNaN = MLX.any(MLX.isNaN(output)).item(Bool.self)
    XCTAssertFalse(hasNaN, "Gated connector output should not contain NaN")
  }

  func testConnector1DWithAttentionMask() {
    let connectorCfg = LTX2ConnectorConfig(
      dim: 64,
      numHeads: 4,
      headDim: 16,
      numLayers: 2,
      numLearnableRegisters: 8,
      positionalEmbeddingTheta: 10000.0,
      positionalEmbeddingMaxPos: [1],
      hasGateLogits: false
    )

    let connector = LTX2Connector1D(config: connectorCfg)

    // Input: (B=1, S=16, dim=64)
    let embeddings = MLXRandom.normal([1, 16, 64])
    // Attention mask: (B=1, 1, 1, S=16) -- first 8 tokens padded, last 8 valid
    let maskValues: [Float] = Array(repeating: -1e9, count: 8) + Array(repeating: 0, count: 8)
    let mask = MLXArray(maskValues, [1, 1, 1, 16])
    MLX.eval(embeddings, mask)

    let (output, newMask) = connector(hiddenStates: embeddings, attentionMask: mask)
    MLX.eval(output)

    XCTAssertEqual(output.dim(0), 1, "Batch preserved")
    XCTAssertEqual(output.dim(1), 16, "Sequence length preserved")
    XCTAssertEqual(output.dim(2), 64, "Dimension preserved")

    // After register replacement, mask should be all zeros
    if let m = newMask {
      MLX.eval(m)
      let maxVal = MLX.max(MLX.abs(m)).item(Float.self)
      XCTAssertEqual(maxVal, 0.0, accuracy: 1e-6, "Mask should be zeroed after register replacement")
    }
  }

  // -----------------------------------------------
  // MARK: 6 -- Per-Channel Statistics
  // -----------------------------------------------

  func testPerChannelStatisticsRoundTrip() {
    let stats = LTX2PerChannelStatistics(latentChannels: 128)

    // With default mean=0, std=1, normalize/unnormalize should be identity
    let x = MLXRandom.normal([1, 128, 1, 2, 2])
    MLX.eval(x)

    let normalized = stats.normalize(x)
    let restored = stats.unNormalize(normalized)
    MLX.eval(normalized, restored)

    let maxDiff = MLX.max(MLX.abs(x - restored)).item(Float.self)
    XCTAssertEqual(maxDiff, 0.0, accuracy: 1e-5, "Default stats should be identity")
  }

  // -----------------------------------------------
  // MARK: 7 -- Connector Attention (SPLIT RoPE)
  // -----------------------------------------------

  func testSplitRoPEPreservesShape() {
    // Verify SPLIT RoPE does not change tensor shape
    let batchSize = 1
    let numHeads = 4
    let seqLen = 8
    let headDim = 16

    let x = MLXRandom.normal([batchSize, numHeads, seqLen, headDim])
    let cos = MLXRandom.normal([1, numHeads, seqLen, headDim / 2])
    let sin = MLXRandom.normal([1, numHeads, seqLen, headDim / 2])
    MLX.eval(x, cos, sin)

    let rotated = LTX2ConnectorAttention.applySplitRoPE(x, cos: cos, sin: sin)
    MLX.eval(rotated)

    XCTAssertEqual(rotated.shape, x.shape, "SPLIT RoPE should preserve shape")
    let hasNaN = MLX.any(MLX.isNaN(rotated)).item(Bool.self)
    XCTAssertFalse(hasNaN, "SPLIT RoPE output should not contain NaN")
  }

  // -----------------------------------------------
  // MARK: 8 -- Connector Feed Forward
  // -----------------------------------------------

  func testConnectorFeedForwardShape() {
    let ff = LTX2ConnectorFeedForward(dim: 64, mult: 4)

    let input = MLXRandom.normal([1, 8, 64])
    MLX.eval(input)

    let output = ff(input)
    MLX.eval(output)

    XCTAssertEqual(output.shape, [1, 8, 64], "FFN should preserve shape")
  }

  // -----------------------------------------------
  // MARK: 9 -- VAE Encode with 4D input (single image)
  // -----------------------------------------------

  func testVAEEncodeHandles4DInput() {
    let vae = LTX2VAE(config: .default)

    // 4D input (no temporal dim): (B=1, C=3, H=64, W=64)
    let input = MLXRandom.normal([1, 3, 64, 64])
    MLX.eval(input)

    let latent = vae.encode(input)
    MLX.eval(latent)

    // Should auto-expand to 5D: (1, 128, 1, 2, 2)
    XCTAssertEqual(latent.ndim, 5)
    XCTAssertEqual(latent.dim(1), 128)
    XCTAssertEqual(latent.dim(2), 1, "Single frame temporal dim")
  }
}
