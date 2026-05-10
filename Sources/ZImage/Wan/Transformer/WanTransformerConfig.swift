import Foundation

/// Configuration for the Wan 2.2 I2V-A14B transformer.
///
/// Verified values from actual model weights and config.json.
/// The FDD contains wrong values -- use this config.
public struct WanTransformerConfig: Sendable {

  /// Hidden dimension of the transformer.
  public let dim: Int

  /// Intermediate dimension in feed-forward network.
  public let ffnDim: Int

  /// Dimension for sinusoidal time embeddings.
  public let freqDim: Int

  /// Input channels (latent channels + conditioning channels for I2V).
  public let inDim: Int

  /// Output channels (latent channels).
  public let outDim: Int

  /// Number of attention heads.
  public let numHeads: Int

  /// Number of transformer blocks.
  public let numLayers: Int

  /// Fixed length for text embeddings.
  public let textLen: Int

  /// Input dimension for text embeddings (T5 output dim).
  public let textDim: Int

  /// 3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
  public let patchSize: (Int, Int, Int)

  /// Epsilon value for normalization layers.
  public let eps: Float

  /// Enable query/key normalization.
  public let qkNorm: Bool

  /// Enable cross-attention normalization (norm3 has weight+bias).
  public let crossAttnNorm: Bool

  /// Window size for local attention (-1 indicates global attention).
  public let windowSize: (Int, Int)

  /// Model type: "t2v" or "i2v".
  public let modelType: String

  /// Per-head dimension (derived).
  public var headDim: Int { dim / numHeads }

  /// Default configuration matching the Wan 2.2 I2V-A14B checkpoint.
  ///
  /// Verified from actual model weights:
  /// - 40 layers, 40 heads, head_dim=128
  /// - in_dim=36 (16 noise + 20 conditioning), out_dim=16
  /// - FFN uses GELU(approximate='tanh'), 2 projections
  /// - All attention projections have bias
  /// - QK-norm via WanRMSNorm
  public static let i2vA14B = WanTransformerConfig(
    dim: 5120,
    ffnDim: 13824,
    freqDim: 256,
    inDim: 36,
    outDim: 16,
    numHeads: 40,
    numLayers: 40,
    textLen: 512,
    textDim: 4096,
    patchSize: (1, 2, 2),
    eps: 1e-6,
    qkNorm: true,
    crossAttnNorm: true,
    windowSize: (-1, -1),
    modelType: "i2v"
  )
}
