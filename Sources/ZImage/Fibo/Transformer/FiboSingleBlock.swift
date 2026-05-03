// FiboSingleBlock.swift — Single-stream transformer block for FIBO
// Ported from mflux: single_transformer_block.py (FiboSingleTransformerBlock)
//
// After the joint blocks process both streams separately, single blocks
// operate on a concatenated [encoder, image] sequence. Each block applies:
// 1. AdaLayerNormZeroSingle (3-way: shift, scale, gate)
// 2. FiboSingleAttention (Q/K RMSNorm + RoPE + optional mask)
// 3. Parallel MLP (GELU-approximate)
// 4. Concatenation + output projection
// 5. Gated residual connection
//
// FIBO has 38 single blocks (same as Flux). The architecture is similar
// to Flux's single blocks but uses separate attention (not fused QKV+MLP)
// and explicit attention masks.

import Foundation
import MLX
import MLXNN

/// Single-stream transformer block for FIBO.
///
/// Weight key path: `single_transformer_blocks.{i}.*`
final class FiboSingleTransformerBlock: Module {
  let mlpHiddenDim: Int

  @ModuleInfo(key: "norm") var norm: FiboAdaLayerNormZeroSingle
  @ModuleInfo(key: "proj_mlp") var projMLP: Linear
  @ModuleInfo(key: "proj_out") var projOut: Linear
  @ModuleInfo(key: "attn") var attn: FiboSingleAttention

  init(
    dim: Int = 3072,
    numAttentionHeads: Int = 24,
    attentionHeadDim: Int = 128,
    mlpRatio: Float = 4.0
  ) {
    self.mlpHiddenDim = Int(Float(dim) * mlpRatio)

    self._norm.wrappedValue = FiboAdaLayerNormZeroSingle(dim: dim)
    self._projMLP.wrappedValue = Linear(dim, mlpHiddenDim)
    self._projOut.wrappedValue = Linear(dim + mlpHiddenDim, dim)
    self._attn.wrappedValue = FiboSingleAttention(
      dim: dim,
      numAttentionHeads: numAttentionHeads,
      attentionHeadDim: attentionHeadDim
    )

    super.init()
  }

  /// - Parameters:
  ///   - temb: DimFusion time+text embedding `[batch, dim]`.
  ///   - hiddenStates: Combined [encoder, image] hidden states `[batch, seq, dim]`.
  ///   - imageRotaryEmb: `(cos, sin)` for RoPE.
  ///   - attentionMask: Optional mask for attention.
  /// - Returns: Updated hidden states `[batch, seq, dim]`.
  func callAsFunction(
    temb: MLXArray,
    hiddenStates: MLXArray,
    imageRotaryEmb: (MLXArray, MLXArray),
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    // 0. Residual connection
    let residual = hiddenStates

    // 1. AdaLayerNormZeroSingle: produces (normed+modulated, gate)
    let (normHiddenStates, gate) = norm(
      hiddenStates: hiddenStates,
      textEmbeddings: temb
    )

    // 2. Attention
    let attnOutput = attn(
      hiddenStates: normHiddenStates,
      imageRotaryEmb: imageRotaryEmb,
      attentionMask: attentionMask
    )

    // 3. Parallel MLP + projection
    let mlpHiddenStates = gelu(projMLP(normHiddenStates))
    let combined = MLX.concatenated([attnOutput, mlpHiddenStates], axis: 2)
    let gateExpanded = gate.expandedDimensions(axis: 1)
    let projected = gateExpanded * projOut(combined)

    return residual + projected
  }
}
