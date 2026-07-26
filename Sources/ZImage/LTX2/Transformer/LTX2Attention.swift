// LTX2Attention.swift — Multi-head attention with QK-RMSNorm and optional gating
// Phase 3 of the LTX-2 Swift/MLX port
//
// Supports both self-attention and cross-attention. Key features:
// - Q/K normalization via RMSNorm before RoPE application
// - Optional per-head gate logits (LTX-2.3) for output modulation
// - RoPE applied after QK-norm, before attention computation
// - Cross-attention: Q from hidden states, K/V from context (no RoPE)
// - Skip-attention mode for STG perturbation (bypasses Q*K*V)
//
// Reference: attention.py class Attention

import Foundation
import MLX
import MLXFast
import MLXNN

/// Multi-head attention with QK-RMSNorm, RoPE, and optional per-head gating.
///
/// Weight key mapping:
/// - `transformer_blocks.N.attn1.to_q.weight`, `.bias` (self-attention Q)
/// - `transformer_blocks.N.attn1.to_k.weight`, `.bias` (self-attention K)
/// - `transformer_blocks.N.attn1.to_v.weight`, `.bias` (self-attention V)
/// - `transformer_blocks.N.attn1.to_out.weight`, `.bias` (self-attention output)
/// - `transformer_blocks.N.attn1.q_norm.weight` (Q RMSNorm)
/// - `transformer_blocks.N.attn1.k_norm.weight` (K RMSNorm)
/// - `transformer_blocks.N.attn1.to_gate_logits.weight`, `.bias` (optional)
/// - Same pattern for `attn2` (cross-attention)
public final class LTX2Attention: Module {
  let numHeads: Int
  let headDim: Int
  let ropeMode: LTX2RoPEMode

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: Linear
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
  @ModuleInfo(key: "to_gate_logits") var toGateLogits: Linear?

  let hasGateLogits: Bool

  /// Initialize the attention module.
  ///
  /// - Parameters:
  ///   - queryDim: Dimension of query input.
  ///   - contextDim: Dimension of context input for cross-attention. Nil for self-attention.
  ///   - heads: Number of attention heads. Default 32.
  ///   - dimHead: Dimension per head. Default 128.
  ///   - normEps: Epsilon for RMSNorm. Default 1e-6.
  ///   - ropeMode: RoPE mode. Default `.split`.
  ///   - hasGateLogits: Whether to use per-head gate logits. Default false.
  public init(
    queryDim: Int,
    contextDim: Int? = nil,
    heads: Int = 32,
    dimHead: Int = 128,
    normEps: Float = 1e-6,
    ropeMode: LTX2RoPEMode = .split,
    hasGateLogits: Bool = false
  ) {
    self.numHeads = heads
    self.headDim = dimHead
    self.ropeMode = ropeMode
    self.hasGateLogits = hasGateLogits

    let innerDim = dimHead * heads
    let ctxDim = contextDim ?? queryDim

    self._toQ.wrappedValue = Linear(queryDim, innerDim, bias: true)
    self._toK.wrappedValue = Linear(ctxDim, innerDim, bias: true)
    self._toV.wrappedValue = Linear(ctxDim, innerDim, bias: true)
    self._toOut.wrappedValue = Linear(innerDim, queryDim, bias: true)

    self._qNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: normEps)
    self._kNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: normEps)

    if hasGateLogits {
      self._toGateLogits.wrappedValue = Linear(queryDim, heads, bias: true)
    }
  }

  /// Forward pass.
  ///
  /// - Parameters:
  ///   - x: Query input `(B, seqLen, queryDim)`.
  ///   - context: Context for cross-attention. If nil, uses x (self-attention).
  ///   - mask: Optional attention mask.
  ///   - pe: Position embeddings `(cos, sin)` for Q (and K if kPE is nil).
  ///   - kPE: Optional separate position embeddings for K.
  ///   - skipAttention: If true, bypass Q*K*V and use V only (STG perturbation).
  /// - Returns: Attention output `(B, seqLen, queryDim)`.
  public func callAsFunction(
    _ x: MLXArray,
    context: MLXArray? = nil,
    mask: MLXArray? = nil,
    pe: (cos: MLXArray, sin: MLXArray)? = nil,
    kPE: (cos: MLXArray, sin: MLXArray)? = nil,
    skipAttention: Bool = false
  ) -> MLXArray {
    let b = x.dim(0)
    let qSeqLen = x.dim(1)

    // Compute per-head gate early (from original input, before context mixing)
    var gate: MLXArray? = nil
    if hasGateLogits, let gateProj = toGateLogits {
      gate = 2.0 * sigmoid(gateProj(x))  // (B, seqLen, heads)
    }

    let ctx = context ?? x
    let v = toV(ctx)

    let out: MLXArray
    if skipAttention {
      // STG: bypass Q*K*V, use V only
      out = v
    } else {
      // Standard attention
      var q = qNorm(toQ(x))
      var k = kNorm(toK(ctx))

      // Reshape to per-head: (B, T, H*D) -> (B, T, H, D) -> (B, H, T, D)
      let kvSeqLen = ctx.dim(1)
      var qH = q.reshaped(b, qSeqLen, numHeads, headDim).transposed(0, 2, 1, 3)
      var kH = k.reshaped(b, kvSeqLen, numHeads, headDim).transposed(0, 2, 1, 3)
      let vH = v.reshaped(b, kvSeqLen, numHeads, headDim).transposed(0, 2, 1, 3)

      // Apply RoPE to Q and K if provided
      if let pe = pe {
        qH = ltx2ApplyRoPE(qH, freqsCIS: pe, mode: ropeMode)
        let kFreqs = kPE ?? pe
        kH = ltx2ApplyRoPE(kH, freqsCIS: kFreqs, mode: ropeMode)
      }

      // Scaled dot-product attention
      let scale = 1.0 / Float(headDim).squareRoot()

      var attnMask: MLXArray? = mask
      if let m = attnMask {
        // Ensure mask has batch and head dimensions
        var expanded = m
        if expanded.ndim == 2 {
          expanded = expanded.expandedDimensions(axis: 0)
        }
        if expanded.ndim == 3 {
          expanded = expanded.expandedDimensions(axis: 1)
        }
        attnMask = expanded
      }

      // Query-tiled attention (#37): at very long sequences (289f refine ≈
      // 53,872 tokens) the SDPA kernel's token×token index overflows int32
      // (same class as the conv/decode bug) and the process aborts. Splitting
      // the QUERIES into blocks — each attending to the FULL K/V set — keeps
      // every kernel call's qBlock×kv product under 2^31. This is exact (no
      // online softmax needed: each block's softmax is over the complete key
      // axis), so it's numerically identical to a single call, zero seams.
      // Threshold well below the 2^31/kv boundary; disabled at normal lengths.
      let attnProduct = qSeqLen * kvSeqLen
      let tileThreshold = 1_500_000_000  // < 2^31, margin for head batching
      let attnOut: MLXArray
      if attnProduct > tileThreshold {
        let maxQ = max(1, tileThreshold / max(kvSeqLen, 1))
        var blocks: [MLXArray] = []
        var qs = 0
        while qs < qSeqLen {
          let qe = min(qs + maxQ, qSeqLen)
          let qBlock = qH[0..., 0..., qs..<qe, 0...]
          let mBlock: MLXArray? = attnMask  // key-axis mask applies to every query block
          let ob = MLXFast.scaledDotProductAttention(
            queries: qBlock, keys: kH, values: vH, scale: scale, mask: mBlock)
          eval(ob)
          blocks.append(ob)
          qs = qe
        }
        attnOut = blocks.count == 1 ? blocks[0] : MLX.concatenated(blocks, axis: 2)
      } else {
        attnOut = MLXFast.scaledDotProductAttention(
          queries: qH, keys: kH, values: vH,
          scale: scale,
          mask: attnMask
        )
      }

      // Reshape back: (B, H, T, D) -> (B, T, H*D)
      out = attnOut.transposed(0, 2, 1, 3).reshaped(b, qSeqLen, numHeads * headDim)
    }

    // Apply per-head gating (LTX-2.3)
    var gatedOut = out
    if let gate = gate {
      gatedOut = gatedOut.reshaped(b, qSeqLen, numHeads, headDim)
      gatedOut = gatedOut * gate.expandedDimensions(axis: -1)
      gatedOut = gatedOut.reshaped(b, qSeqLen, -1)
    }

    return toOut(gatedOut)
  }
}
