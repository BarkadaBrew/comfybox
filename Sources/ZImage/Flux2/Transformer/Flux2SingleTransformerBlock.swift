import Foundation
import MLX
import MLXNN

/// Single-stream transformer block for Flux 2.
///
/// After the double-stream blocks merge image and text into a single sequence,
/// these blocks process the unified stream with parallel self-attention + MLP.
/// Modulation (shift/scale/gate) from the timestep embedding is applied.
final class Flux2SingleTransformerBlock: Module {
  @ModuleInfo(key: "norm") var norm: LayerNorm
  @ModuleInfo(key: "attn") var attn: Flux2ParallelSelfAttention

  init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, mlpRatio: Float = 3.0) {
    self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._attn.wrappedValue = Flux2ParallelSelfAttention(
      dim: dim,
      heads: numAttentionHeads,
      dimHead: attentionHeadDim,
      mlpRatio: mlpRatio
    )
    super.init()
  }

  /// - Parameters:
  ///   - hiddenStates: Unified hidden states `[batch, seq, dim]`.
  ///   - tembModParams: Single modulation set `(shift, scale, gate)`.
  ///   - imageRotaryEmb: `(cos, sin)` for RoPE.
  func callAsFunction(
    hiddenStates: MLXArray,
    tembModParams: (MLXArray, MLXArray, MLXArray),
    imageRotaryEmb: (MLXArray, MLXArray)
  ) -> MLXArray {
    let (modShift, modScale, modGate) = tembModParams

    var normHidden = norm(hiddenStates)
    normHidden = (1 + modScale) * normHidden + modShift
    let attnOutput = attn(normHidden, imageRotaryEmb: imageRotaryEmb)
    return hiddenStates + modGate * attnOutput
  }
}
