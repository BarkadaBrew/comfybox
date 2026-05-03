// SmolLM3Attention.swift — Grouped Query Attention for SmolLM3-3B
// Ported from mflux: smol_lm3_3b_attention.py

import MLX
import MLXNN
import MLXFast

/// Grouped Query Attention (GQA) for SmolLM3-3B text encoder.
///
/// Architecture:
/// - 16 query heads, 4 key-value heads (4x GQA ratio)
/// - Head dimension 128 (hiddenSize 2048 / 16 heads)
/// - No bias on projections (attention_bias = false)
/// - RoPE applied externally via cos/sin tables
/// - Uses MLXFast.scaledDotProductAttention for efficiency
///
/// Weight key mapping (safetensors -> model):
/// - `layers.N.self_attn.q_proj.weight`
/// - `layers.N.self_attn.k_proj.weight`
/// - `layers.N.self_attn.v_proj.weight`
/// - `layers.N.self_attn.o_proj.weight`
public final class SmolLM3Attention: Module {
  let hiddenSize: Int
  let numAttentionHeads: Int
  let numKeyValueHeads: Int
  let headDim: Int
  let numKeyValueGroups: Int
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  public init(config: FiboTextEncoderConfig) {
    self.hiddenSize = config.hiddenSize
    self.numAttentionHeads = config.numAttentionHeads
    self.numKeyValueHeads = config.numKeyValueHeads
    self.headDim = config.headDim
    self.numKeyValueGroups = config.numAttentionHeads / config.numKeyValueHeads
    self.scale = 1.0 / Float(config.headDim).squareRoot()

    // No bias — matching SmolLM3 config (attention_bias = false)
    self._qProj.wrappedValue = Linear(hiddenSize, numAttentionHeads * headDim, bias: false)
    self._kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
    self._vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
    self._oProj.wrappedValue = Linear(numAttentionHeads * headDim, hiddenSize, bias: false)
  }

  /// Forward pass through self-attention.
  ///
  /// - Parameters:
  ///   - hiddenStates: Input tensor `[B, S, hiddenSize]`.
  ///   - attentionMask: 4D additive mask `[B, 1, S, S]` (0 = attend, -inf = mask).
  ///   - cosSin: `(cos, sin)` from `SmolLM3RoPE`, each `[1, 1, S, headDim]`.
  ///     Pass `nil` for layers where RoPE is disabled (`no_rope_layers`).
  /// - Returns: Output tensor `[B, S, hiddenSize]`.
  public func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    cosSin: (cos: MLXArray, sin: MLXArray)?
  ) -> MLXArray {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)

    // Project Q, K, V
    var q = qProj(hiddenStates)
    var k = kProj(hiddenStates)
    var v = vProj(hiddenStates)

    // Cast to input dtype if needed (Linear may output float32)
    let inputDtype = hiddenStates.dtype
    if q.dtype != inputDtype {
      q = q.asType(inputDtype)
      k = k.asType(inputDtype)
      v = v.asType(inputDtype)
    }

    // Reshape to [B, heads, S, headDim]
    q = q.reshaped(batchSize, seqLen, numAttentionHeads, headDim).transposed(0, 2, 1, 3)
    k = k.reshaped(batchSize, seqLen, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
    v = v.reshaped(batchSize, seqLen, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

    // Apply RoPE (if cos/sin provided — some layers skip RoPE per no_rope_layers)
    if let (cos, sin) = cosSin {
      (q, k) = SmolLM3Attention.applyRoPE(q: q, k: k, cos: cos, sin: sin)
    }

    // GQA: repeat KV heads to match Q heads
    if numKeyValueHeads != numAttentionHeads {
      k = SmolLM3Attention.repeatKV(k, nRep: numKeyValueGroups)
      v = SmolLM3Attention.repeatKV(v, nRep: numKeyValueGroups)
    }

    // Scaled dot-product attention
    var attnOutput = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: attentionMask.map { .array($0) } ?? .none
    )

    // Reshape back to [B, S, hiddenSize]
    attnOutput = attnOutput.transposed(0, 2, 1, 3).reshaped(batchSize, seqLen, hiddenSize)
    return oProj(attnOutput)
  }

  // MARK: - RoPE Application

  /// Apply rotary position embeddings to query and key tensors.
  ///
  /// Uses the rotate-half method: `x_embed = x * cos + rotate_half(x) * sin`
  /// Computation is done in float32 for numerical stability.
  static func applyRoPE(
    q: MLXArray, k: MLXArray,
    cos: MLXArray, sin: MLXArray
  ) -> (MLXArray, MLXArray) {
    let qDtype = q.dtype
    let kDtype = k.dtype
    let qF = q.asType(.float32)
    let kF = k.asType(.float32)
    let cosF = cos.asType(.float32)
    let sinF = sin.asType(.float32)

    let qEmbed = (qF * cosF) + (rotateHalf(qF) * sinF)
    let kEmbed = (kF * cosF) + (rotateHalf(kF) * sinF)

    return (qEmbed.asType(qDtype), kEmbed.asType(kDtype))
  }

  /// Rotate the second half of the last dimension: `[-x2, x1]`
  static func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<halfDim]
    let x2 = x[.ellipsis, halfDim...]
    return MLX.concatenated([-x2, x1], axis: -1)
  }

  /// Expand KV heads by repeating along the head dimension for GQA.
  static func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
    guard nRep > 1 else { return x }
    let shape = x.shape  // [B, numKVHeads, S, headDim]
    var expanded = x.expandedDimensions(axis: 2)  // [B, numKVHeads, 1, S, headDim]
    expanded = MLX.broadcast(expanded, to: [shape[0], shape[1], nRep, shape[2], shape[3]])
    return expanded.reshaped(shape[0], shape[1] * nRep, shape[2], shape[3])
  }
}
