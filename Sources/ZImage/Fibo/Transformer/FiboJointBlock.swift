// FiboJointBlock.swift — Joint (double-stream) transformer block for FIBO
// Ported from mflux: joint_transformer_block.py (FiboJointTransformerBlock)
//
// Each joint block processes both the image and encoder (text) streams
// through parallel norm -> attention -> FFN paths. Uses FiboAdaLayerNormZero
// for 6-way modulation and FiboJointAttention for cross-attention.
//
// FIBO has 8 joint blocks (vs Flux's 19) and uses GELU-approximate FFN
// with bias (vs Flux's bias-free SwiGLU).

import Foundation
import MLX
import MLXNN

// MARK: - GELU Feed-Forward

/// GELU activation wrapper with linear projection.
///
/// Weight key path: `ff.net.0.proj.{weight,bias}` (GELU gate projection)
final class FiboGELU: Module {
  @ModuleInfo(key: "proj") var proj: Linear

  init(dimIn: Int, dimOut: Int) {
    self._proj.wrappedValue = Linear(dimIn, dimOut)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    geluApproximate(proj(x))
  }
}

/// GELU-approximate feed-forward network for FIBO joint blocks.
///
/// Architecture: Linear(dim, innerDim) -> GELU -> Linear(innerDim, dim)
/// innerDim = dim * 4 = 12288 for hidden_size=3072.
///
/// Weight key paths:
/// - `transformer_blocks.{i}.ff.net.0.proj.{weight,bias}`
/// - `transformer_blocks.{i}.ff.net.2.{weight,bias}`
///
/// The Python reference uses a sequential list 
/// with numeric indices as keys (0, 1, 2). MLXNN's ModuleParameters treats
/// numeric string keys as array indices, which does not match our module
/// structure. To work around this, we use a custom weight loading approach:
/// the weight keys are remapped from  -> 
/// and  ->  in FiboWeightMapping.
final class FiboFeedForward: Module {
  @ModuleInfo(key: "gelu") var gelu: FiboGELU
  @ModuleInfo(key: "linear_out") var linearOut: Linear

  init(dim: Int, dimOut: Int? = nil, mult: Int = 4) {
    let innerDim = dim * mult
    let outDim = dimOut ?? dim
    self._gelu.wrappedValue = FiboGELU(dimIn: dim, dimOut: innerDim)
    self._linearOut.wrappedValue = Linear(innerDim, outDim)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    linearOut(gelu(x))
  }
}

// MARK: - FiboJointTransformerBlock

/// Joint (double-stream) transformer block for FIBO.
///
/// Architecture per block:
/// 1. FiboAdaLayerNormZero for both image and text streams (6-way modulation each)
/// 2. FiboJointAttention: Q/K RMSNorm, RoPE, joint cross-attention with masks
/// 3. Residual + gating for attention outputs
/// 4. LayerNorm + shift/scale modulation for FFN inputs
/// 5. GELU FFN for both streams
/// 6. Residual + gating for FFN outputs
///
/// Weight key path: `transformer_blocks.{i}.*`
final class FiboJointTransformerBlock: Module {
  @ModuleInfo(key: "norm1") var norm1: FiboAdaLayerNormZero
  @ModuleInfo(key: "norm1_context") var norm1Context: FiboAdaLayerNormZero
  @ModuleInfo(key: "attn") var attn: FiboJointAttention
  @ModuleInfo(key: "norm2") var norm2: LayerNorm
  @ModuleInfo(key: "ff") var ff: FiboFeedForward
  @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
  @ModuleInfo(key: "ff_context") var ffContext: FiboFeedForward

  init(
    dim: Int = 3072,
    numAttentionHeads: Int = 24,
    attentionHeadDim: Int = 128,
    eps: Float = 1e-6
  ) {
    self._norm1.wrappedValue = FiboAdaLayerNormZero(embeddingDim: dim)
    self._norm1Context.wrappedValue = FiboAdaLayerNormZero(embeddingDim: dim)
    self._attn.wrappedValue = FiboJointAttention(
      dim: dim,
      numAttentionHeads: numAttentionHeads,
      attentionHeadDim: attentionHeadDim
    )
    self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
    self._ff.wrappedValue = FiboFeedForward(dim: dim, dimOut: dim, mult: 4)
    self._norm2Context.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
    self._ffContext.wrappedValue = FiboFeedForward(dim: dim, dimOut: dim, mult: 4)

    super.init()
  }

  /// - Parameters:
  ///   - temb: DimFusion time+text embedding `[batch, dim]`.
  ///   - hiddenStates: Image hidden states `[batch, imgSeq, dim]`.
  ///   - encoderHiddenStates: Text hidden states `[batch, txtSeq, dim]`.
  ///   - imageRotaryEmb: `(cos, sin)` for RoPE.
  ///   - attentionMask: Optional mask for attention.
  /// - Returns: `(encoderHiddenStates, hiddenStates)`.
  func callAsFunction(
    temb: MLXArray,
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    imageRotaryEmb: (MLXArray, MLXArray),
    attentionMask: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    // 1. AdaLayerNormZero for both streams
    let (normHiddenStates, gateMSA, shiftMLP, scaleMLP, gateMLP) = norm1(
      hiddenStates: hiddenStates,
      textEmbeddings: temb
    )
    let (normEncoderHiddenStates, cGateMSA, cShiftMLP, cScaleMLP, cGateMLP) = norm1Context(
      hiddenStates: encoderHiddenStates,
      textEmbeddings: temb
    )

    // 2. Joint attention
    let (attnOutput, contextAttnOutput) = attn(
      hiddenStates: normHiddenStates,
      encoderHiddenStates: normEncoderHiddenStates,
      imageRotaryEmb: imageRotaryEmb,
      attentionMask: attentionMask
    )

    // 3a. Image stream: residual + gated attention -> norm -> modulated FFN
    var hidden = hiddenStates + gateMSA.expandedDimensions(axis: 1) * attnOutput
    var normHidden = norm2(hidden)
    normHidden = normHidden * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)
    let ffOutput = ff(normHidden)
    hidden = hidden + gateMLP.expandedDimensions(axis: 1) * ffOutput

    // 3b. Encoder stream: residual + gated attention -> norm -> modulated FFN
    var encoder = encoderHiddenStates + cGateMSA.expandedDimensions(axis: 1) * contextAttnOutput
    var normEncoder = norm2Context(encoder)
    normEncoder = normEncoder * (1 + cScaleMLP.expandedDimensions(axis: 1)) + cShiftMLP.expandedDimensions(axis: 1)
    let contextFFOutput = ffContext(normEncoder)
    encoder = encoder + cGateMLP.expandedDimensions(axis: 1) * contextFFOutput

    return (encoder, hidden)
  }
}
