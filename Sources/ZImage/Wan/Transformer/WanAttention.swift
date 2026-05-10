import Foundation
import MLX
import MLXFast
import MLXNN

/// Self-attention for the Wan 2.2 transformer.
///
/// All projections (q, k, v, o) have bias, unlike T5.
/// QK-norm via WanRMSNorm on q and k before reshaping.
/// RoPE applied to q and k after norm, before attention.
///
/// Weight keys:
/// ```
/// self_attn.q.weight  [dim, dim]
/// self_attn.q.bias    [dim]
/// self_attn.k.weight  [dim, dim]
/// self_attn.k.bias    [dim]
/// self_attn.v.weight  [dim, dim]
/// self_attn.v.bias    [dim]
/// self_attn.o.weight  [dim, dim]
/// self_attn.o.bias    [dim]
/// self_attn.norm_q.weight  [dim]
/// self_attn.norm_k.weight  [dim]
/// ```
public final class WanSelfAttention: Module {

  @ModuleInfo(key: "q") var qProj: Linear
  @ModuleInfo(key: "k") var kProj: Linear
  @ModuleInfo(key: "v") var vProj: Linear
  @ModuleInfo(key: "o") var oProj: Linear
  @ModuleInfo(key: "norm_q") var normQ: WanRMSNorm?
  @ModuleInfo(key: "norm_k") var normK: WanRMSNorm?

  public let numHeads: Int
  public let headDim: Int
  public let dim: Int

  /// Creates a self-attention module.
  ///
  /// - Parameters:
  ///   - dim: Model dimension.
  ///   - numHeads: Number of attention heads.
  ///   - qkNorm: Whether to apply RMS normalization to q and k. Default true.
  ///   - eps: Epsilon for RMS normalization.
  public init(dim: Int, numHeads: Int, qkNorm: Bool = true, eps: Float = 1e-6) {
    self.dim = dim
    self.numHeads = numHeads
    self.headDim = dim / numHeads

    self._qProj.wrappedValue = Linear(dim, dim)
    self._kProj.wrappedValue = Linear(dim, dim)
    self._vProj.wrappedValue = Linear(dim, dim)
    self._oProj.wrappedValue = Linear(dim, dim)

    if qkNorm {
      self._normQ.wrappedValue = WanRMSNorm(dim: dim, eps: eps)
      self._normK.wrappedValue = WanRMSNorm(dim: dim, eps: eps)
    }

    super.init()
  }

  /// Self-attention forward pass.
  ///
  /// - Parameters:
  ///   - x: Input tensor `[B, seqLen, dim]`.
  ///   - seqLens: Actual sequence lengths per sample (for masking).
  ///   - gridSizes: 3D grid dimensions per sample `[[F, H, W], ...]`.
  ///   - freqs: Precomputed RoPE frequencies.
  /// - Returns: Output tensor `[B, seqLen, dim]`.
  public func callAsFunction(
    _ x: MLXArray,
    seqLens: [Int],
    gridSizes: [[Int]],
    freqs: MLXArray
  ) -> MLXArray {
    let b = x.dim(0)
    let s = x.dim(1)

    // Project
    var q = qProj(x)
    var k = kProj(x)
    let vFlat = vProj(x)

    // QK-norm
    if let nq = normQ { q = nq(q) }
    if let nk = normK { k = nk(k) }

    // Reshape to heads: [B, S, numHeads, headDim]
    q = q.reshaped(b, s, numHeads, headDim)
    k = k.reshaped(b, s, numHeads, headDim)
    let v = vFlat.reshaped(b, s, numHeads, headDim)

    // Apply RoPE
    q = WanRoPE.ropeApply(q, gridSizes: gridSizes, freqs: freqs)
    k = WanRoPE.ropeApply(k, gridSizes: gridSizes, freqs: freqs)

    // Transpose to [B, numHeads, S, headDim] for attention
    let qT = q.transposed(0, 2, 1, 3)
    let kT = k.transposed(0, 2, 1, 3)
    let vT = v.transposed(0, 2, 1, 3)

    // Build causal-compatible mask for variable-length sequences
    let mask = buildSeqMask(seqLens: seqLens, maxLen: s)

    // Scaled dot-product attention
    let scale = 1.0 / Float(headDim).squareRoot()
    let attnOut: MLXArray
    if let mask = mask {
      attnOut = MLXFast.scaledDotProductAttention(
        queries: qT, keys: kT, values: vT, scale: scale, mask: mask
      )
    } else {
      attnOut = MLXFast.scaledDotProductAttention(
        queries: qT, keys: kT, values: vT, scale: scale, mask: nil
      )
    }

    // Transpose back and flatten: [B, S, dim]
    let output = attnOut.transposed(0, 2, 1, 3).reshaped(b, s, dim)
    return oProj(output)
  }

  /// Builds an attention mask for variable-length sequences.
  ///
  /// - Parameters:
  ///   - seqLens: Actual sequence lengths per sample.
  ///   - maxLen: Maximum (padded) sequence length.
  /// - Returns: Mask tensor of shape `[B, 1, 1, maxLen]` or nil if no masking needed.
  func buildSeqMask(seqLens: [Int], maxLen: Int) -> MLXArray? {
    // Check if all sequences are full length (no padding)
    let allFull = seqLens.allSatisfy { $0 >= maxLen }
    if allFull { return nil }

    var masks: [MLXArray] = []
    for len in seqLens {
      let ones = MLXArray.ones([len], type: Float.self)
      let zeros = MLXArray.zeros([maxLen - len], type: Float.self)
      let row = MLX.concatenated([ones, zeros], axis: 0)
      masks.append(row)
    }
    let mask = MLX.stacked(masks, axis: 0)  // [B, maxLen]

    // Convert to attention mask: 0 -> -inf, 1 -> 0
    let attnMask = (1.0 - mask) * MLXArray(Float(-1e9))
    return attnMask.reshaped(seqLens.count, 1, 1, maxLen)
  }
}

/// Cross-attention for the Wan 2.2 transformer.
///
/// Same projections as self-attention but:
/// - q from x, k/v from context
/// - NO RoPE applied
/// - Uses context_lens for masking
///
/// Weight keys: cross_attn.q/k/v/o.weight/bias, cross_attn.norm_q/k.weight
public final class WanCrossAttention: Module {

  @ModuleInfo(key: "q") var qProj: Linear
  @ModuleInfo(key: "k") var kProj: Linear
  @ModuleInfo(key: "v") var vProj: Linear
  @ModuleInfo(key: "o") var oProj: Linear
  @ModuleInfo(key: "norm_q") var normQ: WanRMSNorm?
  @ModuleInfo(key: "norm_k") var normK: WanRMSNorm?

  public let numHeads: Int
  public let headDim: Int
  public let dim: Int

  /// Creates a cross-attention module.
  ///
  /// - Parameters:
  ///   - dim: Model dimension.
  ///   - numHeads: Number of attention heads.
  ///   - qkNorm: Whether to apply RMS normalization to q and k. Default true.
  ///   - eps: Epsilon for RMS normalization.
  public init(dim: Int, numHeads: Int, qkNorm: Bool = true, eps: Float = 1e-6) {
    self.dim = dim
    self.numHeads = numHeads
    self.headDim = dim / numHeads

    self._qProj.wrappedValue = Linear(dim, dim)
    self._kProj.wrappedValue = Linear(dim, dim)
    self._vProj.wrappedValue = Linear(dim, dim)
    self._oProj.wrappedValue = Linear(dim, dim)

    if qkNorm {
      self._normQ.wrappedValue = WanRMSNorm(dim: dim, eps: eps)
      self._normK.wrappedValue = WanRMSNorm(dim: dim, eps: eps)
    }

    super.init()
  }

  /// Cross-attention forward pass.
  ///
  /// - Parameters:
  ///   - x: Query tensor `[B, L1, dim]`.
  ///   - context: Key/value tensor `[B, L2, dim]`.
  ///   - contextLens: Actual context lengths per sample (for masking), or nil.
  /// - Returns: Output tensor `[B, L1, dim]`.
  public func callAsFunction(
    _ x: MLXArray,
    context: MLXArray,
    contextLens: [Int]?
  ) -> MLXArray {
    let b = x.dim(0)

    // q from x, k/v from context
    var q = qProj(x)
    var k = kProj(context)
    let vFlat = vProj(context)

    // QK-norm
    if let nq = normQ { q = nq(q) }
    if let nk = normK { k = nk(k) }

    // Reshape to heads
    let sQ = x.dim(1)
    let sK = context.dim(1)
    q = q.reshaped(b, sQ, numHeads, headDim)
    k = k.reshaped(b, sK, numHeads, headDim)
    let v = vFlat.reshaped(b, sK, numHeads, headDim)

    // Transpose to [B, numHeads, S, headDim]
    let qT = q.transposed(0, 2, 1, 3)
    let kT = k.transposed(0, 2, 1, 3)
    let vT = v.transposed(0, 2, 1, 3)

    // Attention (no RoPE for cross-attention)
    let scale = 1.0 / Float(headDim).squareRoot()
    let attnOut = MLXFast.scaledDotProductAttention(
      queries: qT, keys: kT, values: vT, scale: scale, mask: nil
    )

    // Transpose back and flatten
    let output = attnOut.transposed(0, 2, 1, 3).reshaped(b, sQ, dim)
    return oProj(output)
  }
}
