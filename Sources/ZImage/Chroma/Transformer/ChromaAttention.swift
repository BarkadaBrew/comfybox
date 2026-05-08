import Foundation
import MLX
import MLXFast
import MLXNN

/// QK normalization for Chroma attention.
///
/// Applies separate RMSNorm to query and key projections for stable attention.
final class ChromaQKNorm: Module {
  @ModuleInfo(key: "query_norm") var queryNorm: RMSNorm
  @ModuleInfo(key: "key_norm") var keyNorm: RMSNorm

  init(dim: Int) {
    self._queryNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
    self._keyNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
    super.init()
  }

  func callAsFunction(q: MLXArray, k: MLXArray) -> (MLXArray, MLXArray) {
    let dtype = q.dtype
    return (queryNorm(q.asType(.float32)).asType(dtype),
            keyNorm(k.asType(.float32)).asType(dtype))
  }
}

/// Self-attention with fused QKV projection.
///
/// Used in both Chroma DoubleStreamBlock (one per stream) and SingleStreamBlock.
/// Key difference from Flux 2: fused QKV linear with bias, separate norm module.
///
/// Weight keys: `{img,txt}_attn.qkv.{weight,bias}`, `{img,txt}_attn.norm.*`, `{img,txt}_attn.proj.{weight,bias}`
final class ChromaSelfAttention: Module {
  let numHeads: Int
  let headDim: Int

  @ModuleInfo(key: "qkv") var qkv: Linear
  @ModuleInfo(key: "norm") var norm: ChromaQKNorm
  @ModuleInfo(key: "proj") var proj: Linear

  init(dim: Int, numHeads: Int, qkvBias: Bool = true) {
    self.numHeads = numHeads
    self.headDim = dim / numHeads
    self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
    self._norm.wrappedValue = ChromaQKNorm(dim: headDim)
    self._proj.wrappedValue = Linear(dim, dim, bias: true)
    super.init()
  }

  /// Project input to Q/K/V, reshape for multi-head, apply QK norm.
  ///
  /// - Returns: (q, k, v) each `[B, H, S, D]`
  func projectQKV(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let b = x.dim(0)
    let s = x.dim(1)
    let qkvOut = qkv(x)
    let chunkSize = qkvOut.dim(-1) / 3
    let q = qkvOut[0..., 0..., 0..<chunkSize]
    let k = qkvOut[0..., 0..., chunkSize..<(2 * chunkSize)]
    let v = qkvOut[0..., 0..., (2 * chunkSize)...]

    let qR = q.reshaped(b, s, numHeads, headDim).transposed(0, 2, 1, 3)
    let kR = k.reshaped(b, s, numHeads, headDim).transposed(0, 2, 1, 3)
    let vR = v.reshaped(b, s, numHeads, headDim).transposed(0, 2, 1, 3)

    let (qN, kN) = norm(q: qR, k: kR)
    return (qN, kN, vR)
  }

  /// Apply output projection.
  func projectOut(_ x: MLXArray) -> MLXArray {
    proj(x)
  }
}
