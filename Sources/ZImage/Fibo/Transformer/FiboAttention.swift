// FiboAttention.swift — Attention modules for FIBO transformer
// Ported from mflux: fibo_joint_attention.py (FiboJointAttention) +
//                    fibo_single_attention.py (FiboSingleAttention)
//
// Key differences from Flux2 attention:
// - All projection layers have bias (Flux2 is bias-free)
// - Explicit attention masks are computed and passed through
// - Q/K RMSNorm applied in float32 before RoPE
// - Single attention is separate (not fused QKV+MLP like Flux2)
// - RoPE uses interleaved (real, imag) pair rotation

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - RoPE Utilities

/// RoPE utilities for FIBO's cos/sin format.
///
/// FIBO uses the same interleaved (real, imag) RoPE rotation as Flux2,
/// but in B,S,H,D layout for applying before transposing to B,H,S,D.
enum FiboRoPEUtils {

  /// Apply rotary position embeddings in B,S,H,D layout.
  ///
  /// - Parameters:
  ///   - x: Tensor `[batch, seq, heads, headDim]`.
  ///   - freqsCos: `[seq, halfDim]` cosine frequencies.
  ///   - freqsSin: `[seq, halfDim]` sine frequencies.
  /// - Returns: Rotated tensor in same shape.
  static func applyRotaryEmb(
    _ x: MLXArray,
    freqsCos: MLXArray,
    freqsSin: MLXArray
  ) -> MLXArray {
    let bsz = x.dim(0)
    let seqLen = x.dim(1)
    let numHeads = x.dim(2)
    let headDim = x.dim(3)
    // cos/sin: [seq, halfDim] -> [1, seq, 1, halfDim]
    let cos = freqsCos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let sin = freqsSin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    // x -> [bsz, seqLen, numHeads, halfDim, 2]
    let x2 = x.reshaped(bsz, seqLen, numHeads, -1, 2)
    let xReal = x2[0..., 0..., 0..., 0..., 0]
    let xImag = x2[0..., 0..., 0..., 0..., 1]
    let rotReal = -xImag
    let rotImag = xReal
    let xRotated = MLX.stacked([rotReal, rotImag], axis: -1).reshaped(bsz, seqLen, numHeads, headDim)
    let out = (x.asType(.float32) * cos + xRotated.asType(.float32) * sin).asType(x.dtype)
    return out
  }
}

// MARK: - FiboSingleAttention

/// Self-attention for FIBO single-stream transformer blocks.
///
/// Computes Q/K/V projections with bias, applies Q/K RMSNorm in float32,
/// applies RoPE, then scaled dot-product attention with optional mask.
///
/// Weight key path: `single_transformer_blocks.{i}.attn.{to_q,to_k,to_v,norm_q,norm_k}.*`
final class FiboSingleAttention: Module {
  let headDim: Int
  let numHeads: Int
  let innerDim: Int

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm

  init(dim: Int = 3072, numAttentionHeads: Int = 24, attentionHeadDim: Int = 128, eps: Float = 1e-6) {
    self.headDim = attentionHeadDim
    self.numHeads = numAttentionHeads
    self.innerDim = dim

    self._toQ.wrappedValue = Linear(dim, dim)
    self._toK.wrappedValue = Linear(dim, dim)
    self._toV.wrappedValue = Linear(dim, dim)
    self._normQ.wrappedValue = RMSNorm(dimensions: attentionHeadDim, eps: eps)
    self._normK.wrappedValue = RMSNorm(dimensions: attentionHeadDim, eps: eps)

    super.init()
  }

  /// - Parameters:
  ///   - hiddenStates: `[batch, seq, dim]`.
  ///   - imageRotaryEmb: `(cos, sin)` for RoPE.
  ///   - attentionMask: Optional mask `[batch, 1, seq, seq]`.
  /// - Returns: Attention output `[batch, seq, dim]`.
  func callAsFunction(
    hiddenStates: MLXArray,
    imageRotaryEmb: (MLXArray, MLXArray),
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    let batch = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)
    let (cos, sin) = imageRotaryEmb

    // Project Q/K/V: [B, S, dim]
    var query = toQ(hiddenStates)
    var key = toK(hiddenStates)
    let value = toV(hiddenStates)

    // Reshape to [B, S, H, D]
    query = query.reshaped(batch, seqLen, numHeads, headDim)
    key = key.reshaped(batch, seqLen, numHeads, headDim)
    let valueReshaped = value.reshaped(batch, seqLen, numHeads, headDim)

    // RMSNorm in float32
    let qDtype = query.dtype
    query = normQ(query.asType(.float32)).asType(qDtype)
    key = normK(key.asType(.float32)).asType(qDtype)

    // Apply RoPE in B,S,H,D layout
    query = FiboRoPEUtils.applyRotaryEmb(query, freqsCos: cos, freqsSin: sin)
    key = FiboRoPEUtils.applyRotaryEmb(key, freqsCos: cos, freqsSin: sin)

    // Transpose to [B, H, S, D] for SDPA
    let queryBHSD = query.transposed(0, 2, 1, 3)
    let keyBHSD = key.transposed(0, 2, 1, 3)
    let valueBHSD = valueReshaped.transposed(0, 2, 1, 3)

    let scale = 1.0 / sqrt(Float(headDim))
    let attnOutput = MLXFast.scaledDotProductAttention(
      queries: queryBHSD,
      keys: keyBHSD,
      values: valueBHSD,
      scale: scale,
      mask: attentionMask
    )

    // [B, H, S, D] -> [B, S, H*D]
    return attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, innerDim)
  }
}

// MARK: - FiboJointAttention

/// Joint cross-attention for FIBO double-stream transformer blocks.
///
/// Computes Q/K/V for both image and encoder (text) streams, concatenates
/// them along the sequence dimension, applies joint RoPE and attention,
/// then splits results back into separate image and encoder outputs.
///
/// Weight key path: `transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out,norm_q,norm_k,
///                  add_q_proj,add_k_proj,add_v_proj,to_add_out,norm_added_q,norm_added_k}.*`
final class FiboJointAttention: Module {
  let headDim: Int
  let numHeads: Int
  let innerDim: Int

  // Image stream projections
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm

  // Encoder stream projections
  @ModuleInfo(key: "add_q_proj") var addQProj: Linear
  @ModuleInfo(key: "add_k_proj") var addKProj: Linear
  @ModuleInfo(key: "add_v_proj") var addVProj: Linear
  @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
  @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

  // Output projections
  @ModuleInfo(key: "to_out") var toOut: Linear
  @ModuleInfo(key: "to_add_out") var toAddOut: Linear

  init(dim: Int = 3072, numAttentionHeads: Int = 24, attentionHeadDim: Int = 128, eps: Float = 1e-6) {
    self.headDim = attentionHeadDim
    self.numHeads = numAttentionHeads
    self.innerDim = dim

    self._toQ.wrappedValue = Linear(dim, dim)
    self._toK.wrappedValue = Linear(dim, dim)
    self._toV.wrappedValue = Linear(dim, dim)
    self._normQ.wrappedValue = RMSNorm(dimensions: attentionHeadDim, eps: eps)
    self._normK.wrappedValue = RMSNorm(dimensions: attentionHeadDim, eps: eps)

    self._addQProj.wrappedValue = Linear(dim, dim)
    self._addKProj.wrappedValue = Linear(dim, dim)
    self._addVProj.wrappedValue = Linear(dim, dim)
    self._normAddedQ.wrappedValue = RMSNorm(dimensions: attentionHeadDim, eps: eps)
    self._normAddedK.wrappedValue = RMSNorm(dimensions: attentionHeadDim, eps: eps)

    self._toOut.wrappedValue = Linear(dim, dim)
    self._toAddOut.wrappedValue = Linear(dim, dim)

    super.init()
  }

  /// - Parameters:
  ///   - hiddenStates: Image hidden states `[batch, imgSeq, dim]`.
  ///   - encoderHiddenStates: Text hidden states `[batch, txtSeq, dim]`.
  ///   - imageRotaryEmb: `(cos, sin)` for RoPE (covers full txt+img sequence).
  ///   - attentionMask: Optional mask `[batch, 1, seqTotal, seqTotal]`.
  /// - Returns: `(imageOutput, encoderOutput)`.
  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    imageRotaryEmb: (MLXArray, MLXArray),
    attentionMask: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    let batch = hiddenStates.dim(0)
    let seqImg = hiddenStates.dim(1)
    let seqCtx = encoderHiddenStates.dim(1)
    let (cos, sin) = imageRotaryEmb

    // Image stream QKV: [B, S_img, dim]
    var query = toQ(hiddenStates)
    var key = toK(hiddenStates)
    let value = toV(hiddenStates)

    // Encoder stream QKV: [B, S_ctx, dim]
    var encQuery = addQProj(encoderHiddenStates)
    var encKey = addKProj(encoderHiddenStates)
    let encValue = addVProj(encoderHiddenStates)

    // Reshape to [B, S, H, D]
    query = query.reshaped(batch, seqImg, numHeads, headDim)
    key = key.reshaped(batch, seqImg, numHeads, headDim)
    let valueReshaped = value.reshaped(batch, seqImg, numHeads, headDim)

    encQuery = encQuery.reshaped(batch, seqCtx, numHeads, headDim)
    encKey = encKey.reshaped(batch, seqCtx, numHeads, headDim)
    let encValueReshaped = encValue.reshaped(batch, seqCtx, numHeads, headDim)

    // RMSNorm in float32
    let qDtype = query.dtype
    query = normQ(query.asType(.float32)).asType(qDtype)
    key = normK(key.asType(.float32)).asType(qDtype)
    encQuery = normAddedQ(encQuery.asType(.float32)).asType(qDtype)
    encKey = normAddedK(encKey.asType(.float32)).asType(qDtype)

    // Concatenate encoder + image along sequence dim: [B, S_ctx+S_img, H, D]
    query = MLX.concatenated([encQuery, query], axis: 1)
    key = MLX.concatenated([encKey, key], axis: 1)
    let valueCat = MLX.concatenated([encValueReshaped, valueReshaped], axis: 1)

    // Apply RoPE in B,S,H,D layout
    query = FiboRoPEUtils.applyRotaryEmb(query, freqsCos: cos, freqsSin: sin)
    key = FiboRoPEUtils.applyRotaryEmb(key, freqsCos: cos, freqsSin: sin)

    // Transpose to [B, H, S, D] for SDPA
    let queryBHSD = query.transposed(0, 2, 1, 3)
    let keyBHSD = key.transposed(0, 2, 1, 3)
    let valueBHSD = valueCat.transposed(0, 2, 1, 3)

    // Prepare attention mask
    var attnMask: MLXArray? = nil
    if let mask = attentionMask {
      let seqTotal = seqCtx + seqImg
      attnMask = MLX.broadcast(mask, to: [batch, numHeads, seqTotal, seqTotal])
      attnMask = attnMask!.asType(queryBHSD.dtype)
    }

    let scale = 1.0 / sqrt(Float(headDim))
    let attnOutput = MLXFast.scaledDotProductAttention(
      queries: queryBHSD,
      keys: keyBHSD,
      values: valueBHSD,
      scale: scale,
      mask: attnMask
    )

    // [B, H, S, D] -> [B, S, H*D]
    let combined = attnOutput.transposed(0, 2, 1, 3).reshaped(batch, -1, innerDim)

    // Split back into encoder and image streams
    let contextAttnOutput = toAddOut(combined[0..., ..<seqCtx, 0...])
    let hiddenAttnOutput = toOut(combined[0..., seqCtx..., 0...])

    return (hiddenAttnOutput, contextAttnOutput)
  }
}
