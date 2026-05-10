import Foundation
import MLX
import MLXNN

/// Pre-norm T5 encoder layer for the Wan 2.2 UMT5 encoder.
///
/// Each layer has:
/// - Self-attention with its own relative position bias
/// - Gated GELU FFN
/// - RMS normalization before both attention and FFN (pre-norm)
/// - Residual connections around both sub-layers
///
/// ## Forward Pass
///
/// ```
/// // Self-attention with pre-norm and residual
/// normed = layerNorm1(x)
/// positionBias = posEmbedding.computeBias(seqLen, seqLen)
/// attnOut = selfAttn(normed, positionBias, mask)
/// x = x + attnOut
///
/// // FFN with pre-norm and residual
/// normed = layerNorm2(x)
/// ffnOut = ffn(normed)
/// x = x + ffnOut
/// ```
///
/// ## Weight Mapping
///
/// | Wan Key | Module |
/// |---------|--------|
/// | `blocks.{i}.norm1.weight` | layerNorm1 |
/// | `blocks.{i}.attn.{q,k,v,o}.weight` | selfAttn |
/// | `blocks.{i}.pos_embedding.embedding.weight` | posEmbedding |
/// | `blocks.{i}.norm2.weight` | layerNorm2 |
/// | `blocks.{i}.ffn.gate.0.weight` | ffn.gate |
/// | `blocks.{i}.ffn.fc1.weight` | ffn.fc1 |
/// | `blocks.{i}.ffn.fc2.weight` | ffn.fc2 |
public final class WanT5EncoderLayer: Module {

  /// Self-attention.
  @ModuleInfo(key: "attn") var selfAttn: WanT5Attention

  /// Gated GELU FFN.
  @ModuleInfo(key: "ffn") var ffn: WanT5FFN

  /// Pre-attention RMS normalization.
  @ModuleInfo(key: "norm1") var layerNorm1: RMSNorm

  /// Pre-FFN RMS normalization.
  @ModuleInfo(key: "norm2") var layerNorm2: RMSNorm

  /// Relative position bias (each layer has its own in Wan).
  @ModuleInfo(key: "pos_embedding") var posEmbedding: WanT5RelativePositionBias

  /// Creates a T5 encoder layer.
  ///
  /// - Parameter config: Model configuration.
  public init(config: WanUMT5Config) {
    self._selfAttn.wrappedValue = WanT5Attention(
      hiddenSize: config.hiddenSize,
      numHeads: config.numHeads,
      headDim: config.headDim
    )
    self._ffn.wrappedValue = WanT5FFN(
      hiddenSize: config.hiddenSize,
      ffnHiddenSize: config.ffnHiddenSize
    )
    self._layerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    self._layerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    self._posEmbedding.wrappedValue = WanT5RelativePositionBias(
      numBuckets: config.numBuckets,
      numHeads: config.numHeads,
      maxDistance: config.maxDistance
    )

    super.init()
  }

  /// Applies the encoder layer.
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `[B, seqLen, hiddenSize]`.
  ///   - mask: Optional attention mask of shape `[B, 1, 1, seqLen]`.
  /// - Returns: Output tensor of shape `[B, seqLen, hiddenSize]`.
  public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let seqLen = x.dim(1)

    // Self-attention with pre-norm and residual
    let normed1 = layerNorm1(x)
    let positionBias = posEmbedding.computeBias(queryLength: seqLen, keyLength: seqLen)
    let attnOut = selfAttn(normed1, positionBias: positionBias, mask: mask)
    var hidden = x + attnOut

    // FFN with pre-norm and residual
    let normed2 = layerNorm2(hidden)
    let ffnOut = ffn(normed2)
    hidden = hidden + ffnOut

    return hidden
  }
}
