// LTX2TextEncoderConfig.swift — Configuration for LTX-2 text encoder components
// Phase 2 of the LTX-2 Swift/MLX port
//
// Defines configuration for all text encoder sub-components:
// - Gemma 3 12B language model (with Q4 quantization options)
// - Feature extractor (V1 or V2 depending on model version)
// - 1D Connector (transformer blocks that produce dual embeddings)
// - Text projection (PixArtAlpha-style for AdaLN conditioning)
//
// LTX-2 (original): 2-layer connector, shared feature extractor, 3840-dim
// LTX-2.3 (has_prompt_adaln): 8-layer connector, V2 feature extractor, dual heads

import Foundation

// MARK: - Gemma 3 Quantization Config

/// Quantization configuration for the Gemma 3 language model.
///
/// Q4 quantization reduces memory from ~24 GB to ~6 GB for the 12B model.
public struct LTX2GemmaQuantizationConfig: Codable, Sendable {
  /// Number of quantization bits. 4 for Q4, 8 for Q8.
  public let bits: Int

  /// Group size for quantization. Typical: 32 or 64.
  public let groupSize: Int

  public init(bits: Int = 4, groupSize: Int = 64) {
    self.bits = bits
    self.groupSize = groupSize
  }

  enum CodingKeys: String, CodingKey {
    case bits
    case groupSize = "group_size"
  }
}

// MARK: - Gemma 3 Config

/// Configuration for the Gemma 3 12B language model used as LTX-2's text encoder backbone.
///
/// Architecture (Gemma 3 12B):
///   vocab_size: 262144
///   hidden_size: 3840
///   num_hidden_layers: 48 (+ 1 embedding = 49 total hidden states)
///   num_attention_heads: 16
///   num_key_value_heads: 8 (GQA)
///   head_dim: 256
///   intermediate_size: 21504
///   rms_norm_eps: 1e-6
///   rope_theta: 1000000.0
///   sliding_window: 1024
///   sliding_window_pattern: 6
///   activation: GELU (for MLP)
public struct LTX2GemmaConfig: Codable, Sendable {
  public let vocabSize: Int
  public let hiddenSize: Int
  public let numHiddenLayers: Int
  public let numAttentionHeads: Int
  public let numKeyValueHeads: Int
  public let headDim: Int
  public let intermediateSize: Int
  public let rmsNormEps: Float
  public let ropeTheta: Float
  public let slidingWindow: Int
  public let slidingWindowPattern: Int
  public let quantization: LTX2GemmaQuantizationConfig?

  enum CodingKeys: String, CodingKey {
    case vocabSize = "vocab_size"
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case headDim = "head_dim"
    case intermediateSize = "intermediate_size"
    case rmsNormEps = "rms_norm_eps"
    case ropeTheta = "rope_theta"
    case slidingWindow = "sliding_window"
    case slidingWindowPattern = "sliding_window_pattern"
    case quantization
  }

  /// Number of GQA groups = numAttentionHeads / numKeyValueHeads
  public var numKeyValueGroups: Int { numAttentionHeads / numKeyValueHeads }

  /// Total number of hidden states produced (embedding + all layers)
  public var totalHiddenStates: Int { numHiddenLayers + 1 }

  /// Default configuration matching Gemma 3 12B
  public init(
    vocabSize: Int = 262144,
    hiddenSize: Int = 3840,
    numHiddenLayers: Int = 48,
    numAttentionHeads: Int = 16,
    numKeyValueHeads: Int = 8,
    headDim: Int = 256,
    intermediateSize: Int = 21504,
    rmsNormEps: Float = 1e-6,
    ropeTheta: Float = 1_000_000.0,
    slidingWindow: Int = 1024,
    slidingWindowPattern: Int = 6,
    quantization: LTX2GemmaQuantizationConfig? = nil
  ) {
    self.vocabSize = vocabSize
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.headDim = headDim
    self.intermediateSize = intermediateSize
    self.rmsNormEps = rmsNormEps
    self.ropeTheta = ropeTheta
    self.slidingWindow = slidingWindow
    self.slidingWindowPattern = slidingWindowPattern
    self.quantization = quantization
  }
}

// MARK: - Connector Config

/// Configuration for the 1D Embeddings Connector.
///
/// LTX-2 (original): dim=3840, 30 heads, head_dim=128, 2 layers, max_pos=[1]
/// LTX-2.3 (prompt adaln): separate video/audio connectors with different dims,
///   32 heads, 8 layers, max_pos=[4096], gate_logits=true
public struct LTX2ConnectorConfig: Sendable {
  /// Hidden dimension of the connector
  public let dim: Int

  /// Number of attention heads
  public let numHeads: Int

  /// Dimension per attention head
  public let headDim: Int

  /// Number of transformer layers in the connector
  public let numLayers: Int

  /// Number of learnable register tokens prepended to the sequence
  public let numLearnableRegisters: Int

  /// RoPE theta for positional embedding
  public let positionalEmbeddingTheta: Float

  /// Maximum position values for RoPE frequency computation
  public let positionalEmbeddingMaxPos: [Int]

  /// Whether attention layers have per-head gate logits
  public let hasGateLogits: Bool

  public init(
    dim: Int = 3840,
    numHeads: Int = 30,
    headDim: Int = 128,
    numLayers: Int = 2,
    numLearnableRegisters: Int = 128,
    positionalEmbeddingTheta: Float = 10000.0,
    positionalEmbeddingMaxPos: [Int] = [1],
    hasGateLogits: Bool = false
  ) {
    self.dim = dim
    self.numHeads = numHeads
    self.headDim = headDim
    self.numLayers = numLayers
    self.numLearnableRegisters = numLearnableRegisters
    self.positionalEmbeddingTheta = positionalEmbeddingTheta
    self.positionalEmbeddingMaxPos = positionalEmbeddingMaxPos
    self.hasGateLogits = hasGateLogits
  }

  /// Inner dimension = numHeads * headDim
  public var innerDim: Int { numHeads * headDim }
}

// MARK: - Feature Extractor Config

/// Configuration for the feature extractor (V1 or V2).
public struct LTX2FeatureExtractorConfig: Sendable {
  /// Input dimension: hiddenSize * numLayers (e.g. 3840 * 49 = 188160)
  public let inputDim: Int

  /// Gemma hidden dimension (for rescale normalization in V2)
  public let embeddingDim: Int

  /// Output dimension for video features
  public let videoOutputDim: Int

  /// Output dimension for audio features
  public let audioOutputDim: Int

  /// Whether to use V2 feature extraction (per-token RMSNorm + rescale)
  public let useV2: Bool

  /// Whether linear projections have bias
  public let bias: Bool

  public init(
    inputDim: Int = 188160,  // 3840 * 49
    embeddingDim: Int = 3840,
    videoOutputDim: Int = 4096,
    audioOutputDim: Int = 2048,
    useV2: Bool = true,
    bias: Bool = true
  ) {
    self.inputDim = inputDim
    self.embeddingDim = embeddingDim
    self.videoOutputDim = videoOutputDim
    self.audioOutputDim = audioOutputDim
    self.useV2 = useV2
    self.bias = bias
  }
}

// MARK: - Text Projection Config

/// Configuration for the PixArtAlpha text projection.
public struct LTX2TextProjectionConfig: Sendable {
  /// Input feature dimension (caption_channels)
  public let captionChannels: Int

  /// Hidden/output dimension (transformer inner_dim)
  public let innerDim: Int

  public init(
    captionChannels: Int = 3840,
    innerDim: Int = 4096
  ) {
    self.captionChannels = captionChannels
    self.innerDim = innerDim
  }
}

// MARK: - Aggregate Text Encoder Config

/// Combined configuration for the full LTX-2 text encoder pipeline.
///
/// Orchestrates: Gemma 3 -> Feature Extractor -> 1D Connector -> Embeddings
public struct LTX2TextEncoderConfig: Sendable {
  /// Gemma 3 language model configuration
  public let gemma: LTX2GemmaConfig

  /// Feature extractor configuration
  public let featureExtractor: LTX2FeatureExtractorConfig

  /// Video connector configuration
  public let videoConnector: LTX2ConnectorConfig

  /// Audio connector configuration
  public let audioConnector: LTX2ConnectorConfig

  /// Text projection configuration (for AdaLN conditioning)
  public let textProjection: LTX2TextProjectionConfig

  /// Whether this is an LTX-2.3 model (prompt adaln variant)
  public let hasPromptAdaLN: Bool

  /// Gemma hidden dimension
  public let hiddenDim: Int

  /// Audio embedding dimension
  public let audioDim: Int

  /// Total number of hidden states from Gemma (48 layers + 1 embedding = 49)
  public let numLayers: Int

  public init(
    gemma: LTX2GemmaConfig = LTX2GemmaConfig(),
    hasPromptAdaLN: Bool = true,
    hiddenDim: Int = 3840,
    audioDim: Int = 2048,
    numLayers: Int = 49
  ) {
    self.gemma = gemma
    self.hasPromptAdaLN = hasPromptAdaLN
    self.hiddenDim = hiddenDim
    self.audioDim = audioDim
    self.numLayers = numLayers

    let featureInputDim = hiddenDim * numLayers  // 3840 * 49 = 188160

    if hasPromptAdaLN {
      // LTX-2.3: V2 feature extractor, deeper connectors with separate dims
      self.featureExtractor = LTX2FeatureExtractorConfig(
        inputDim: featureInputDim,
        embeddingDim: hiddenDim,
        videoOutputDim: 4096,
        audioOutputDim: 2048,
        useV2: true,
        bias: true
      )

      self.videoConnector = LTX2ConnectorConfig(
        dim: 4096,
        numHeads: 32,
        headDim: 128,
        numLayers: 8,
        numLearnableRegisters: 128,
        positionalEmbeddingMaxPos: [4096],
        hasGateLogits: true
      )

      self.audioConnector = LTX2ConnectorConfig(
        dim: 2048,
        numHeads: 32,
        headDim: 64,
        numLayers: 8,
        numLearnableRegisters: 128,
        positionalEmbeddingMaxPos: [4096],
        hasGateLogits: true
      )

      self.textProjection = LTX2TextProjectionConfig(
        captionChannels: hiddenDim,
        innerDim: 4096  // = videoConnector.dim = cross_attention_dim
      )
    } else {
      // LTX-2: shared 3840-dim feature extractor, lighter connectors
      self.featureExtractor = LTX2FeatureExtractorConfig(
        inputDim: featureInputDim,
        embeddingDim: hiddenDim,
        videoOutputDim: hiddenDim,
        audioOutputDim: hiddenDim,
        useV2: false,
        bias: false
      )

      self.videoConnector = LTX2ConnectorConfig(
        dim: hiddenDim,
        numHeads: 30,
        headDim: 128,
        numLayers: 2,
        numLearnableRegisters: 128,
        positionalEmbeddingMaxPos: [1]
      )

      self.audioConnector = LTX2ConnectorConfig(
        dim: hiddenDim,
        numHeads: 30,
        headDim: 128,
        numLayers: 2,
        numLearnableRegisters: 128,
        positionalEmbeddingMaxPos: [1]
      )

      self.textProjection = LTX2TextProjectionConfig(
        captionChannels: hiddenDim,
        innerDim: hiddenDim
      )
    }
  }
}
