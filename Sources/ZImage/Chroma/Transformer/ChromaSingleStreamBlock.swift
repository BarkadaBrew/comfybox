import Foundation
import MLX
import MLXFast
import MLXNN

/// Single-stream block for Chroma.
///
/// After double-stream blocks merge image and text into a single sequence,
/// these blocks process the unified stream with parallel attention + MLP.
/// Uses fused linear1 (QKV + MLP input) and linear2 (attn + MLP output).
///
/// Weight key prefix: `single_blocks.N.`
/// Sub-keys: `linear1`, `linear2`, `norm.{query_norm,key_norm}`, `pre_norm`
public final class ChromaSingleStreamBlock: Module {
  let hiddenSize: Int
  let numHeads: Int
  let headDim: Int
  let mlpHiddenDim: Int

  @ModuleInfo(key: "linear1") var linear1: Linear
  @ModuleInfo(key: "linear2") var linear2: Linear
  @ModuleInfo(key: "norm") var norm: ChromaQKNorm
  @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm

  public init(hiddenSize: Int, numHeads: Int, mlpRatio: Float = 4.0) {
    self.hiddenSize = hiddenSize
    self.numHeads = numHeads
    self.headDim = hiddenSize / numHeads
    self.mlpHiddenDim = Int(Float(hiddenSize) * mlpRatio)

    // Fused projection: QKV (3 * hidden) + MLP input (mlpHidden)
    self._linear1.wrappedValue = Linear(hiddenSize, hiddenSize * 3 + mlpHiddenDim, bias: true)
    // Fused output: attention (hidden) + MLP output (mlpHidden) -> hidden
    self._linear2.wrappedValue = Linear(hiddenSize + mlpHiddenDim, hiddenSize, bias: true)
    self._norm.wrappedValue = ChromaQKNorm(dim: headDim)
    self._preNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
    super.init()
  }

  /// Run single-stream block.
  ///
  /// - Parameters:
  ///   - x: Unified hidden states `[B, seq, dim]`
  ///   - mod: Modulation (shift, scale, gate) for this block
  ///   - pe: Positional encoding (cos, sin) for RoPE
  /// - Returns: Updated hidden states
  public func callAsFunction(
    _ x: MLXArray,
    mod: ChromaModulation,
    pe: (MLXArray, MLXArray)
  ) -> MLXArray {
    let b = x.dim(0)
    let seqLen = x.dim(1)

    // Modulate
    let xMod = (1 + mod.scale) * preNorm(x) + mod.shift

    // Fused QKV + MLP projection
    let projected = linear1(xMod)
    let qEnd = hiddenSize
    let kEnd = 2 * hiddenSize
    let vEnd = 3 * hiddenSize

    let q = projected[0..., 0..., 0..<qEnd]
    let k = projected[0..., 0..., qEnd..<kEnd]
    let v = projected[0..., 0..., kEnd..<vEnd]
    let mlpIn = projected[0..., 0..., vEnd...]

    // Reshape for multi-head attention
    var qR = q.reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
    var kR = k.reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
    let vR = v.reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

    // QK norm
    (qR, kR) = norm(q: qR, k: kR)

    // Apply RoPE
    let (cos, sin) = pe
    (qR, kR) = Flux2AttentionUtils.applyRopeBSHD(query: qR, key: kR, cos: cos, sin: sin)

    // Attention
    let scale = Float(1.0 / sqrt(Float(headDim)))
    let attnOut = MLXFast.scaledDotProductAttention(
      queries: qR, keys: kR, values: vR,
      scale: scale, mask: nil
    )
    let y = attnOut.transposed(0, 2, 1, 3).reshaped(b, seqLen, hiddenSize)

    // Fused output: concat attention + activated MLP, then project
    let mlpAct = geluApproximate(mlpIn)
    let combined = MLX.concatenated([y, mlpAct], axis: 2)
    let out = linear2(combined)

    return x + mod.gate * out
  }
}
