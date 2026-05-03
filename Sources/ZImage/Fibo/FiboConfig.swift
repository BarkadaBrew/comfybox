// FiboConfig.swift — Configuration structs parsed from FIBO model config.json files
// Ported from briaai/FIBO config.json + mflux fibo model config

import Foundation

// MARK: - Transformer Config

/// Configuration for the FIBO transformer (Bria4Transformer2DModel).
///
/// Key differences from Flux 1/2:
/// - 48 input channels (vs 64/128) from Wan 2.2 VAE's z_dim
/// - 8 joint attention blocks + 38 single blocks (Flux-derived architecture)
/// - 3072 hidden dim (24 heads * 128 head_dim)
/// - 4096 joint_attention_dim with 2048 text_encoder_dim (SmolLM3-3B)
/// - 46 caption_projection layers (DimFusion conditioning from text encoder hidden states)
/// - No guidance embeddings (guidance_embeds = false)
public struct FiboTransformerConfig: Codable, Sendable {
  public let className: String
  public let patchSize: Int
  public let inChannels: Int
  public let numLayers: Int
  public let numSingleLayers: Int
  public let attentionHeadDim: Int
  public let numAttentionHeads: Int
  public let jointAttentionDim: Int
  public let textEncoderDim: Int
  public let axesDimsRope: [Int]
  public let ropeTheta: Int
  public let timeTheta: Int
  public let guidanceEmbeds: Bool

  enum CodingKeys: String, CodingKey {
    case className = "_class_name"
    case patchSize = "patch_size"
    case inChannels = "in_channels"
    case numLayers = "num_layers"
    case numSingleLayers = "num_single_layers"
    case attentionHeadDim = "attention_head_dim"
    case numAttentionHeads = "num_attention_heads"
    case jointAttentionDim = "joint_attention_dim"
    case textEncoderDim = "text_encoder_dim"
    case axesDimsRope = "axes_dims_rope"
    case ropeTheta = "rope_theta"
    case timeTheta = "time_theta"
    case guidanceEmbeds = "guidance_embeds"
  }

  /// Hidden dimension = numAttentionHeads * attentionHeadDim
  public var hiddenDim: Int { numAttentionHeads * attentionHeadDim }

  /// Number of caption projection (DimFusion) layers.
  /// FIBO uses per-layer text encoder hidden state projections.
  /// The count matches the text encoder's num_hidden_layers + some extra.
  /// From the actual weights: caption_projection.0..45 = 46 layers.
  public var numCaptionProjectionLayers: Int { 46 }

  /// Default configuration matching briaai/FIBO transformer/config.json
  public init(
    className: String = "Bria4Transformer2DModel",
    patchSize: Int = 1,
    inChannels: Int = 48,
    numLayers: Int = 8,
    numSingleLayers: Int = 38,
    attentionHeadDim: Int = 128,
    numAttentionHeads: Int = 24,
    jointAttentionDim: Int = 4096,
    textEncoderDim: Int = 2048,
    axesDimsRope: [Int] = [16, 56, 56],
    ropeTheta: Int = 10000,
    timeTheta: Int = 10000,
    guidanceEmbeds: Bool = false
  ) {
    self.className = className
    self.patchSize = patchSize
    self.inChannels = inChannels
    self.numLayers = numLayers
    self.numSingleLayers = numSingleLayers
    self.attentionHeadDim = attentionHeadDim
    self.numAttentionHeads = numAttentionHeads
    self.jointAttentionDim = jointAttentionDim
    self.textEncoderDim = textEncoderDim
    self.axesDimsRope = axesDimsRope
    self.ropeTheta = ropeTheta
    self.timeTheta = timeTheta
    self.guidanceEmbeds = guidanceEmbeds
  }
}

// MARK: - VAE Config

/// Configuration for the FIBO VAE (AutoencoderKLWan / Wan 2.2 VAE).
///
/// This is a 3D VAE from Wan 2.2 video generation, repurposed for image use.
/// Key differences from Flux 1/2 VAE:
/// - 3D convolutions (Conv3d) instead of Conv2d
/// - z_dim = 48 latent channels (vs 16 for Flux)
/// - Spatial scale factor of 16 (vs 8 for Flux)
/// - Uses RMS norm with gamma weights (not GroupNorm)
/// - Has quant_conv and post_quant_conv layers
/// - Latent normalization via per-channel mean/std
public struct FiboVAEConfig: Codable, Sendable {
  public let className: String
  public let baseDim: Int
  public let decoderBaseDim: Int
  public let dimMult: [Int]
  public let inChannels: Int
  public let outChannels: Int
  public let zDim: Int
  public let numResBlocks: Int
  public let patchSize: Int
  public let scaleFactorSpatial: Int
  public let scaleFactorTemporal: Int
  public let temperalDownsample: [Bool]
  public let latentsMean: [Float]
  public let latentsStd: [Float]

  enum CodingKeys: String, CodingKey {
    case className = "_class_name"
    case baseDim = "base_dim"
    case decoderBaseDim = "decoder_base_dim"
    case dimMult = "dim_mult"
    case inChannels = "in_channels"
    case outChannels = "out_channels"
    case zDim = "z_dim"
    case numResBlocks = "num_res_blocks"
    case patchSize = "patch_size"
    case scaleFactorSpatial = "scale_factor_spatial"
    case scaleFactorTemporal = "scale_factor_temporal"
    case temperalDownsample = "temperal_downsample"
    case latentsMean = "latents_mean"
    case latentsStd = "latents_std"
  }

  /// Number of downsampling blocks = dimMult.count - 1
  public var numDownBlocks: Int { max(0, dimMult.count - 1) }

  /// Number of upsampling blocks (same as down)
  public var numUpBlocks: Int { numDownBlocks }

  /// Default configuration matching briaai/FIBO vae/config.json
  public init(
    className: String = "AutoencoderKLWan",
    baseDim: Int = 160,
    decoderBaseDim: Int = 256,
    dimMult: [Int] = [1, 2, 4, 4],
    inChannels: Int = 12,
    outChannels: Int = 12,
    zDim: Int = 48,
    numResBlocks: Int = 2,
    patchSize: Int = 2,
    scaleFactorSpatial: Int = 16,
    scaleFactorTemporal: Int = 4,
    temperalDownsample: [Bool] = [false, true, true],
    latentsMean: [Float] = [],
    latentsStd: [Float] = []
  ) {
    self.className = className
    self.baseDim = baseDim
    self.decoderBaseDim = decoderBaseDim
    self.dimMult = dimMult
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.zDim = zDim
    self.numResBlocks = numResBlocks
    self.patchSize = patchSize
    self.scaleFactorSpatial = scaleFactorSpatial
    self.scaleFactorTemporal = scaleFactorTemporal
    self.temperalDownsample = temperalDownsample
    self.latentsMean = latentsMean
    self.latentsStd = latentsStd
  }
}

// MARK: - Text Encoder Config

/// Configuration for the FIBO text encoder (SmolLM3-3B / SmolLM3ForCausalLM).
///
/// Key differences from Flux 1/2 text encoders (CLIP/T5/Qwen3):
/// - SmolLM3-3B architecture (LLaMA-style with SiLU activation)
/// - 2048 hidden size, 36 layers, 16 attention heads, 4 KV heads (GQA)
/// - RoPE with theta=5000000 and selective per-layer rope disable (no_rope_layers)
/// - Vocab size 128256 (different from Qwen3's 151936)
/// - No bias on attention projections
/// - SiLU activation in MLP (gate_proj + up_proj pattern)
public struct FiboTextEncoderConfig: Codable, Sendable {
  public let architectures: [String]
  public let hiddenSize: Int
  public let numHiddenLayers: Int
  public let numAttentionHeads: Int
  public let numKeyValueHeads: Int
  public let intermediateSize: Int
  public let maxPositionEmbeddings: Int
  public let ropeTheta: Float
  public let vocabSize: Int
  public let rmsNormEps: Float
  public let hiddenAct: String
  public let noRopeLayers: [Int]?

  enum CodingKeys: String, CodingKey {
    case architectures
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case ropeTheta = "rope_theta"
    case vocabSize = "vocab_size"
    case rmsNormEps = "rms_norm_eps"
    case hiddenAct = "hidden_act"
    case noRopeLayers = "no_rope_layers"
  }

  /// Head dimension = hiddenSize / numAttentionHeads
  public var headDim: Int { hiddenSize / numAttentionHeads }

  /// KV dimension = numKeyValueHeads * headDim
  public var kvDim: Int { numKeyValueHeads * headDim }

  /// Default configuration matching briaai/FIBO text_encoder/config.json
  public init(
    architectures: [String] = ["SmolLM3ForCausalLM"],
    hiddenSize: Int = 2048,
    numHiddenLayers: Int = 36,
    numAttentionHeads: Int = 16,
    numKeyValueHeads: Int = 4,
    intermediateSize: Int = 11008,
    maxPositionEmbeddings: Int = 65536,
    ropeTheta: Float = 5_000_000.0,
    vocabSize: Int = 128256,
    rmsNormEps: Float = 1e-6,
    hiddenAct: String = "silu",
    noRopeLayers: [Int]? = nil
  ) {
    self.architectures = architectures
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.intermediateSize = intermediateSize
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.ropeTheta = ropeTheta
    self.vocabSize = vocabSize
    self.rmsNormEps = rmsNormEps
    self.hiddenAct = hiddenAct
    self.noRopeLayers = noRopeLayers
  }
}

// MARK: - Aggregate Config

/// Combined configuration for all FIBO model components.
public struct FiboModelConfig: Sendable {
  public let transformer: FiboTransformerConfig
  public let vae: FiboVAEConfig
  public let textEncoder: FiboTextEncoderConfig

  public init(
    transformer: FiboTransformerConfig = FiboTransformerConfig(),
    vae: FiboVAEConfig = FiboVAEConfig(),
    textEncoder: FiboTextEncoderConfig = FiboTextEncoderConfig()
  ) {
    self.transformer = transformer
    self.vae = vae
    self.textEncoder = textEncoder
  }

  /// Load configuration from a model snapshot directory.
  public static func load(from snapshot: URL) throws -> FiboModelConfig {
    let decoder = JSONDecoder()

    let transformerData = try Data(contentsOf: snapshot
      .appendingPathComponent("transformer")
      .appendingPathComponent("config.json"))
    let transformer = try decoder.decode(FiboTransformerConfig.self, from: transformerData)

    let vaeData = try Data(contentsOf: snapshot
      .appendingPathComponent("vae")
      .appendingPathComponent("config.json"))
    let vae = try decoder.decode(FiboVAEConfig.self, from: vaeData)

    let textEncoderData = try Data(contentsOf: snapshot
      .appendingPathComponent("text_encoder")
      .appendingPathComponent("config.json"))
    let textEncoder = try decoder.decode(FiboTextEncoderConfig.self, from: textEncoderData)

    return FiboModelConfig(
      transformer: transformer,
      vae: vae,
      textEncoder: textEncoder
    )
  }
}
