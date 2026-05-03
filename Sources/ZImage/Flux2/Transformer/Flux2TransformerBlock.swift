import Foundation
import MLX
import MLXNN

/// Double-stream (joint) transformer block for Flux 2.
///
/// Processes image and text hidden states through parallel norm → attention → FFN paths.
/// Both streams share the same attention computation (cross-attention via Flux2Attention)
/// but have separate norms, modulation, and feed-forward networks.
final class Flux2TransformerBlock: Module {
  @ModuleInfo(key: "norm1") var norm1: LayerNorm
  @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
  @ModuleInfo(key: "attn") var attn: Flux2Attention
  @ModuleInfo(key: "norm2") var norm2: LayerNorm
  @ModuleInfo(key: "ff") var ff: Flux2FeedForward
  @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
  @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward

  init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, mlpRatio: Float = 3.0) {
    self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._norm1Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._attn.wrappedValue = Flux2Attention(
      dim: dim,
      heads: numAttentionHeads,
      dimHead: attentionHeadDim,
      addedKVProjDim: dim
    )
    self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._ff.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
    self._norm2Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._ffContext.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
    super.init()
  }

  /// - Parameters:
  ///   - hiddenStates: Image hidden states `[batch, imgSeq, dim]`.
  ///   - encoderHiddenStates: Text hidden states `[batch, txtSeq, dim]`.
  ///   - tembModParamsImg: Modulation params for image stream: `[[shift, scale, gate], [shift, scale, gate]]`.
  ///   - tembModParamsTxt: Modulation params for text stream: `[[shift, scale, gate], [shift, scale, gate]]`.
  ///   - imageRotaryEmb: `(cos, sin)` for RoPE.
  /// - Returns: `(encoderHiddenStates, hiddenStates)`.
  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    tembModParamsImg: [[MLXArray]],
    tembModParamsTxt: [[MLXArray]],
    imageRotaryEmb: (MLXArray, MLXArray)
  ) -> (MLXArray, MLXArray) {
    // Image modulation params
    let (shiftMSA, scaleMSA, gateMSA) = (tembModParamsImg[0][0], tembModParamsImg[0][1], tembModParamsImg[0][2])
    let (shiftMLP, scaleMLP, gateMLP) = (tembModParamsImg[1][0], tembModParamsImg[1][1], tembModParamsImg[1][2])
    // Text modulation params
    let (cShiftMSA, cScaleMSA, cGateMSA) = (tembModParamsTxt[0][0], tembModParamsTxt[0][1], tembModParamsTxt[0][2])
    let (cShiftMLP, cScaleMLP, cGateMLP) = (tembModParamsTxt[1][0], tembModParamsTxt[1][1], tembModParamsTxt[1][2])

    // Normalize and modulate image
    var normHidden = norm1(hiddenStates)
    normHidden = (1 + scaleMSA) * normHidden + shiftMSA

    // Normalize and modulate text
    var normEncoder = norm1Context(encoderHiddenStates)
    normEncoder = (1 + cScaleMSA) * normEncoder + cShiftMSA

    // Joint attention
    let (attnOutput, encoderAttnOutput) = attn(
      hiddenStates: normHidden,
      encoderHiddenStates: normEncoder,
      imageRotaryEmb: imageRotaryEmb
    )

    // Residual + gating for attention
    var hidden = hiddenStates + gateMSA * attnOutput
    var encoder = encoderHiddenStates + cGateMSA * encoderAttnOutput

    // FFN for image
    var normHidden2 = norm2(hidden)
    normHidden2 = (1 + scaleMLP) * normHidden2 + shiftMLP
    hidden = hidden + gateMLP * ff(normHidden2)

    // FFN for text
    var normEncoder2 = norm2Context(encoder)
    normEncoder2 = (1 + cScaleMLP) * normEncoder2 + cShiftMLP
    encoder = encoder + cGateMLP * ffContext(normEncoder2)

    return (encoder, hidden)
  }
}
