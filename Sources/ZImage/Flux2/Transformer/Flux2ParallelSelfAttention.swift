import Foundation
import MLX
import MLXFast
import MLXNN

/// Parallel self-attention with fused QKV + MLP projection for Flux 2 single-stream blocks.
///
/// A single linear projection produces Q, K, V for attention AND the MLP hidden state
/// simultaneously. After attention and SwiGLU, the outputs are concatenated and projected
/// back to the model dimension. This fused approach is more efficient than separate
/// attention + MLP passes.
final class Flux2ParallelSelfAttention: Module {
  let heads: Int
  let dimHead: Int
  let innerDim: Int
  let mlpHiddenDim: Int
  let scale: Float

  @ModuleInfo(key: "to_qkv_mlp_proj") var toQkvMlpProj: Linear
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm
  @ModuleInfo(key: "mlp_act") var mlpAct: Flux2SwiGLU
  @ModuleInfo(key: "to_out") var toOut: Linear

  init(dim: Int, heads: Int, dimHead: Int, mlpRatio: Float = 3.0) {
    self.heads = heads
    self.dimHead = dimHead
    self.innerDim = heads * dimHead
    self.mlpHiddenDim = Int(Float(dim) * mlpRatio)
    self.scale = 1.0 / sqrt(Float(dimHead))

    // Single projection for Q + K + V + MLP gate + MLP value
    self._toQkvMlpProj.wrappedValue = Linear(dim, innerDim * 3 + mlpHiddenDim * 2, bias: false)
    self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
    self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
    self._mlpAct.wrappedValue = Flux2SwiGLU()
    self._toOut.wrappedValue = Linear(innerDim + mlpHiddenDim, dim, bias: false)
    super.init()
  }

  func callAsFunction(_ hiddenStates: MLXArray, imageRotaryEmb: (MLXArray, MLXArray)) -> MLXArray {
    let proj = toQkvMlpProj(hiddenStates)

    // Split: [QKV | MLP_hidden]
    let splits = MLX.split(proj, indices: [innerDim * 3], axis: -1)
    let qkvPart = splits[0]
    let mlpHidden = splits[1]

    // Split QKV into Q, K, V
    let qkvSplits = MLX.split(qkvPart, parts: 3, axis: -1)
    let queryRaw = qkvSplits[0]
    let keyRaw = qkvSplits[1]
    let valueRaw = qkvSplits[2]

    let batch = queryRaw.dim(0)
    let seqLen = queryRaw.dim(1)

    // Reshape to [B, H, S, D]
    var query = queryRaw.reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)
    var key = keyRaw.reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)
    let value = valueRaw.reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)

    // Normalize Q/K in float32
    let qDtype = query.dtype
    query = normQ(query.asType(.float32)).asType(qDtype)
    key = normK(key.asType(.float32)).asType(qDtype)

    // Apply RoPE
    let (cos, sin) = imageRotaryEmb
    (query, key) = Flux2AttentionUtils.applyRopeBSHD(query: query, key: key, cos: cos, sin: sin)

    // Attention
    let attnOut = MLXFast.scaledDotProductAttention(
      queries: query,
      keys: key,
      values: value,
      scale: scale,
      mask: nil
    )
    // [B, H, S, D] -> [B, S, H*D]
    var hiddenOut = attnOut.transposed(0, 2, 1, 3).reshaped(batch, seqLen, innerDim)

    // Parallel MLP: SwiGLU on the MLP hidden part
    let mlpOut = mlpAct(mlpHidden)

    // Concatenate attention output and MLP output, project
    hiddenOut = MLX.concatenated([hiddenOut, mlpOut], axis: -1)
    return toOut(hiddenOut)
  }
}
