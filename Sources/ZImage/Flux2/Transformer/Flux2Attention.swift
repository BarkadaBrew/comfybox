import Foundation
import MLX
import MLXFast
import MLXNN

/// Cross-attention for Flux 2 double-stream transformer blocks.
///
/// Computes joint attention between image hidden states and encoder (text) hidden states.
/// Both streams produce Q/K/V projections that are concatenated, RoPE-rotated, and
/// attended jointly, then split back into separate image and encoder outputs.
final class Flux2Attention: Module {
  let heads: Int
  let dimHead: Int
  let innerDim: Int
  let scale: Float

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm
  @ModuleInfo(key: "to_out") var toOut: Linear

  // Added KV projections for encoder hidden states
  @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
  @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
  @ModuleInfo(key: "add_q_proj") var addQProj: Linear
  @ModuleInfo(key: "add_k_proj") var addKProj: Linear
  @ModuleInfo(key: "add_v_proj") var addVProj: Linear
  @ModuleInfo(key: "to_add_out") var toAddOut: Linear

  init(dim: Int, heads: Int, dimHead: Int, addedKVProjDim: Int) {
    self.heads = heads
    self.dimHead = dimHead
    self.innerDim = heads * dimHead
    self.scale = 1.0 / sqrt(Float(dimHead))

    self._toQ.wrappedValue = Linear(dim, innerDim, bias: false)
    self._toK.wrappedValue = Linear(dim, innerDim, bias: false)
    self._toV.wrappedValue = Linear(dim, innerDim, bias: false)
    self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
    self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
    self._toOut.wrappedValue = Linear(innerDim, dim, bias: false)

    self._normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
    self._normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
    self._addQProj.wrappedValue = Linear(addedKVProjDim, innerDim, bias: false)
    self._addKProj.wrappedValue = Linear(addedKVProjDim, innerDim, bias: false)
    self._addVProj.wrappedValue = Linear(addedKVProjDim, innerDim, bias: false)
    self._toAddOut.wrappedValue = Linear(innerDim, dim, bias: false)

    super.init()
  }

  /// - Parameters:
  ///   - hiddenStates: Image hidden states `[batch, imgSeq, dim]`.
  ///   - encoderHiddenStates: Text hidden states `[batch, txtSeq, dim]`.
  ///   - imageRotaryEmb: Tuple of `(cos, sin)` for RoPE.
  /// - Returns: Tuple of `(imageOutput, encoderOutput)`.
  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    imageRotaryEmb: (MLXArray, MLXArray)
  ) -> (MLXArray, MLXArray) {
    let batch = hiddenStates.dim(0)
    let encSeqLen = encoderHiddenStates.dim(1)

    // Image Q/K/V
    var query = processQKV(hiddenStates, proj: toQ, norm: normQ)
    var key = processQKV(hiddenStates, proj: toK, norm: normK)
    var value = reshapeBSHD(toV(hiddenStates), batch: batch)

    // Encoder Q/K/V
    let encQuery = processQKV(encoderHiddenStates, proj: addQProj, norm: normAddedQ)
    let encKey = processQKV(encoderHiddenStates, proj: addKProj, norm: normAddedK)
    let encValue = reshapeBSHD(addVProj(encoderHiddenStates), batch: batch)

    // Concatenate: [enc, img] along sequence dimension
    query = MLX.concatenated([encQuery, query], axis: 2)
    key = MLX.concatenated([encKey, key], axis: 2)
    value = MLX.concatenated([encValue, value], axis: 2)

    // Apply RoPE
    let (cos, sin) = imageRotaryEmb
    (query, key) = Flux2AttentionUtils.applyRopeBSHD(query: query, key: key, cos: cos, sin: sin)

    // Scaled dot-product attention
    let attnOut = MLXFast.scaledDotProductAttention(
      queries: query,
      keys: key,
      values: value,
      scale: scale,
      mask: nil
    )
    // [B, H, S, D] -> [B, S, H*D]
    let combined = attnOut.transposed(0, 2, 1, 3).reshaped(batch, -1, innerDim)

    // Split back into encoder and image
    let encoderOut = toAddOut(combined[0..., ..<encSeqLen, 0...])
    let imageOut = toOut(combined[0..., encSeqLen..., 0...])

    return (imageOut, encoderOut)
  }

  /// Project, reshape to [B, H, S, D], normalize in float32.
  private func processQKV(_ x: MLXArray, proj: Linear, norm: RMSNorm) -> MLXArray {
    let batch = x.dim(0)
    let seqLen = x.dim(1)
    let projected = proj(x).reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)
    let origDtype = projected.dtype
    return norm(projected.asType(.float32)).asType(origDtype)
  }

  /// Reshape [B, S, H*D] -> [B, H, S, D].
  private func reshapeBSHD(_ x: MLXArray, batch: Int) -> MLXArray {
    let seqLen = x.dim(1)
    return x.reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)
  }
}

// MARK: - Flux2 RoPE Utilities

/// Attention utilities specific to Flux 2's cos/sin RoPE format.
///
/// Flux 2 uses explicit (cos, sin) pairs rather than Flux 1's complex-valued table.
/// The rotation formula is applied in the interleaved (real, imag) layout.
enum Flux2AttentionUtils {

  /// Apply rotary position embeddings in B,S,H,D layout using cos/sin pairs.
  ///
  /// - Parameters:
  ///   - query: `[batch, heads, seq, headDim]`
  ///   - key: `[batch, heads, seq, headDim]`
  ///   - cos: `[seq, halfDim]`
  ///   - sin: `[seq, halfDim]`
  /// - Returns: Rotated `(query, key)`.
  static func applyRopeBSHD(
    query: MLXArray,
    key: MLXArray,
    cos: MLXArray,
    sin: MLXArray
  ) -> (MLXArray, MLXArray) {
    let outDtype = query.dtype
    let qf = query.asType(.float32)
    let kf = key.asType(.float32)
    // Broadcast cos/sin: [seq, halfDim] -> [1, 1, seq, halfDim]
    let cosB = cos.reshaped(1, 1, cos.dim(0), cos.dim(1))
    let sinB = sin.reshaped(1, 1, sin.dim(0), sin.dim(1))

    return (mix(qf, cosB, sinB).asType(outDtype), mix(kf, cosB, sinB).asType(outDtype))
  }

  /// Rotate x using interleaved (real, imag) pairs.
  private static func mix(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
    // x shape: [B, H, S, D] -> [B, H, S, D/2, 2]
    var shape = x.shape
    shape[shape.count - 1] = shape.last! / 2
    shape.append(2)
    let x2 = x.reshaped(shape)
    let real = x2[0..., 0..., 0..., 0..., 0]
    let imag = x2[0..., 0..., 0..., 0..., 1]
    let out0 = real * cos + (-imag) * sin
    let out1 = imag * cos + real * sin
    return MLX.stacked([out0, out1], axis: -1).reshaped(x.shape)
  }
}
