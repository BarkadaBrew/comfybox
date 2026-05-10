import Foundation

/// Configuration for the Wan 2.2 UMT5-XXL text encoder.
///
/// This encoder is a modified T5 architecture with the following key differences
/// from standard T5:
/// - All layers have their own relative position bias (not just layer 0)
/// - Uses gated GELU FFN (not ReLU or SiLU)
/// - Wan-specific weight naming convention (blocks.N.attn.q, not encoder.block.N)
/// - Embedding vocab padded to 256384 (multiple of 128) vs 256300 in tokenizer
public struct WanUMT5Config {

  /// Number of transformer encoder layers.
  public let numLayers: Int

  /// Hidden dimension of the model.
  public let hiddenSize: Int

  /// Hidden dimension of the gated FFN.
  public let ffnHiddenSize: Int

  /// Number of attention heads.
  public let numHeads: Int

  /// Dimension per attention head.
  public let headDim: Int

  /// Vocabulary size of the embedding (padded to multiple of 128).
  public let vocabSize: Int

  /// Number of buckets for relative position bias.
  public let numBuckets: Int

  /// Maximum distance for relative position bucketing.
  public let maxDistance: Int

  /// Epsilon for RMS normalization.
  public let rmsNormEps: Float

  /// Maximum sequence length.
  public let maxSequenceLength: Int

  /// Default configuration matching the Wan 2.2 I2V UMT5-XXL checkpoint.
  public static let wan22I2V = WanUMT5Config(
    numLayers: 24,
    hiddenSize: 4096,
    ffnHiddenSize: 10240,
    numHeads: 64,
    headDim: 64,
    vocabSize: 256384,
    numBuckets: 32,
    maxDistance: 128,
    rmsNormEps: 1e-6,
    maxSequenceLength: 512
  )
}
