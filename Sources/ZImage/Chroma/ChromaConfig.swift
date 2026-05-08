import Foundation

/// Configuration for the Chroma transformer model.
///
/// Chroma is based on FLUX.1-schnell with the 3.3B AdaLN modulation layer
/// replaced by a 250M parameter Approximator (FFN). Uses T5-XXL text encoder
/// (no CLIP). 19 double + 38 single stream blocks. Apache 2.0 license.
public struct ChromaConfig: Sendable {
  // Transformer dimensions
  public let inChannels: Int
  public let outChannels: Int
  public let contextInDim: Int     // T5-XXL output dim
  public let hiddenSize: Int       // Model dimension
  public let mlpRatio: Float
  public let numHeads: Int
  public let depth: Int            // Double stream block count
  public let depthSingleBlocks: Int // Single stream block count
  public let axesDim: [Int]        // RoPE axes dimensions
  public let theta: Int            // RoPE theta
  public let patchSize: Int
  public let qkvBias: Bool

  // Approximator dimensions
  public let approxInDim: Int      // Input: timestep(16) + guidance(16) + modIndex(32) = 64
  public let approxOutDim: Int     // Output: hiddenSize (3072)
  public let approxHiddenDim: Int  // Hidden: 5120
  public let approxNLayers: Int    // 5 residual MLP layers

  /// Total modulation vector count produced by the Approximator.
  /// Layout: single(depth_single*3) + double_img(depth*6) + double_txt(depth*6) + final(2)
  public var modIndexLength: Int {
    depthSingleBlocks * 3 + depth * 6 + depth * 6 + 2
  }

  /// Standard Chroma configuration matching lodestones/Chroma.
  public static let standard = ChromaConfig(
    inChannels: 64,
    outChannels: 64,
    contextInDim: 4096,
    hiddenSize: 3072,
    mlpRatio: 4.0,
    numHeads: 24,
    depth: 19,
    depthSingleBlocks: 38,
    axesDim: [16, 56, 56],
    theta: 10_000,
    patchSize: 2,
    qkvBias: true,
    approxInDim: 64,
    approxOutDim: 3072,
    approxHiddenDim: 5120,
    approxNLayers: 5
  )
}

/// T5-XXL text encoder configuration.
public struct T5Config {
  public let vocabSize: Int
  public let dModel: Int
  public let dFF: Int
  public let numHeads: Int
  public let numLayers: Int
  public let dKV: Int
  public let relativeAttentionNumBuckets: Int
  public let relativeAttentionMaxDistance: Int
  public let layerNormEpsilon: Float
  public let feedForwardProj: String

  /// Standard T5-v1.1-XXL configuration.
  public static let xxl = T5Config(
    vocabSize: 32128,
    dModel: 4096,
    dFF: 10240,
    numHeads: 64,
    numLayers: 24,
    dKV: 64,
    relativeAttentionNumBuckets: 32,
    relativeAttentionMaxDistance: 128,
    layerNormEpsilon: 1e-6,
    feedForwardProj: "gated-gelu"
  )
}
