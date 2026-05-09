// LTX2GemmaModel.swift — Gemma 3 12B language model for LTX-2 text encoding
// Phase 2 of the LTX-2 Swift/MLX port
//
// Implements the Gemma 3 12B transformer model that serves as LTX-2's text
// encoder backbone. The critical difference from standard LLM inference is that
// ALL 49 hidden states (1 embedding + 48 transformer layers) must be extracted,
// not just the final output.
//
// Architecture (Gemma 3 12B from unsloth/gemma-3-12b-it):
//   - Embedding: vocab 262208 -> 3840, with sqrt(3840) scaling built into embed
//   - 48 transformer layers with sliding window attention pattern
//   - Dual RoPE: global (base=10000, linear scale 8x) and local (base=10000)
//   - Every 6th layer (5, 11, 17, ...) uses full attention with global RoPE
//   - All other layers use sliding window attention with local RoPE
//   - GQA: 16 query heads, 8 KV heads, head_dim 256
//   - QK normalization: per-head RMSNorm on Q and K after projection
//   - MLP: gelu_pytorch_tanh-gated, intermediate 15360
//   - 4-norm layers: input/post_attention/pre_feedforward/post_feedforward
//   - RMSNorm uses (1 + weight) convention
//
// Q4 quantization support keeps memory at ~6 GB instead of ~24 GB.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Gemma 3 RMS Norm

/// RMSNorm for Gemma 3, using the (1 + weight) convention.
///
/// Gemma uses `output = rms_norm(x) * (1 + weight)` rather than just `* weight`.
/// This matches the HuggingFace Gemma3RMSNorm implementation.
public final class Gemma3RMSNorm: Module {
  @ModuleInfo(key: "weight") var weight: MLXArray
  let eps: Float

  public init(hiddenSize: Int, eps: Float = 1e-6) {
    self.eps = eps
    self._weight.wrappedValue = MLX.ones([hiddenSize])
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    return MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
  }
}

// MARK: - Gemma 3 RoPE

/// Rotary position embeddings for Gemma 3 12B.
///
/// Supports two modes used by Gemma 3:
/// - **Local** (sliding window layers): base=10000, no scaling
/// - **Global** (full attention layers): base=10000, linear scaling factor 8
///   (positions are divided by 8, equivalent to dividing inv_freq by 8)
///
/// Output: (cos, sin) each `[1, 1, seqLen, headDim]`.
public final class Gemma3RoPE: Module {
  let dim: Int
  let base: Float
  let scaleFactor: Float
  let invFreq: MLXArray

  public init(dim: Int, base: Float = 10_000.0, scaleFactor: Float = 1.0) {
    self.dim = dim
    self.base = base
    self.scaleFactor = scaleFactor

    // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
    var freq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))

    // Linear scaling: divide frequencies by scale factor
    // This makes positions evolve slower (wider context window)
    if scaleFactor > 1.0 {
      freq = freq / MLXArray(scaleFactor)
    }

    self.invFreq = freq
  }

  /// Compute cos/sin tables for the given sequence length.
  public func callAsFunction(_ seqLen: Int) -> (cos: MLXArray, sin: MLXArray) {
    let positions = MLXArray(0..<seqLen).asType(.float32)
    let freqs = MLX.outer(positions, invFreq)
    let emb = MLX.concatenated([freqs, freqs], axis: -1)
    let cos = MLX.cos(emb).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    let sin = MLX.sin(emb).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    return (cos, sin)
  }
}

// MARK: - Gemma 3 Attention

/// Multi-head attention with GQA for Gemma 3 12B.
///
/// Features:
/// - 16 query heads, 8 KV heads (2x GQA ratio), head_dim 256
/// - Per-head QK normalization (RMSNorm on dim=head_dim)
/// - RoPE applied via rotate-half method
/// - No bias on projections
public final class Gemma3Attention: Module {
  let numHeads: Int
  let numKVHeads: Int
  let headDim: Int
  let numKVGroups: Int
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear
  @ModuleInfo(key: "q_norm") var qNorm: Gemma3RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: Gemma3RMSNorm

  public init(config: LTX2GemmaConfig) {
    self.numHeads = config.numAttentionHeads
    self.numKVHeads = config.numKeyValueHeads
    self.headDim = config.headDim
    self.numKVGroups = config.numAttentionHeads / config.numKeyValueHeads
    self.scale = 1.0 / Float(config.headDim).squareRoot()

    let hiddenSize = config.hiddenSize
    self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
    self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
    self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
    self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
    self._qNorm.wrappedValue = Gemma3RMSNorm(hiddenSize: headDim, eps: config.rmsNormEps)
    self._kNorm.wrappedValue = Gemma3RMSNorm(hiddenSize: headDim, eps: config.rmsNormEps)
  }

  public func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    cosSin: (cos: MLXArray, sin: MLXArray)
  ) -> MLXArray {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)

    var q = qProj(hiddenStates)
    var k = kProj(hiddenStates)
    var v = vProj(hiddenStates)

    // Reshape to [B, S, heads, headDim] for per-head QK norm
    q = q.reshaped(batchSize, seqLen, numHeads, headDim)
    k = k.reshaped(batchSize, seqLen, numKVHeads, headDim)
    v = v.reshaped(batchSize, seqLen, numKVHeads, headDim)

    // Per-head QK normalization
    q = qNorm(q)
    k = kNorm(k)

    // Transpose to [B, heads, S, headDim]
    q = q.transposed(0, 2, 1, 3)
    k = k.transposed(0, 2, 1, 3)
    v = v.transposed(0, 2, 1, 3)

    // Apply RoPE
    (q, k) = Gemma3Attention.applyRoPE(q: q, k: k, cos: cosSin.cos, sin: cosSin.sin)

    // GQA: repeat KV heads
    if numKVHeads != numHeads {
      k = Gemma3Attention.repeatKV(k, nRep: numKVGroups)
      v = Gemma3Attention.repeatKV(v, nRep: numKVGroups)
    }

    var attnOutput = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v,
      scale: scale,
      mask: attentionMask.map { .array($0) } ?? .none
    )

    attnOutput = attnOutput.transposed(0, 2, 1, 3)
      .reshaped(batchSize, seqLen, numHeads * headDim)
    return oProj(attnOutput)
  }

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

  static func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<halfDim]
    let x2 = x[.ellipsis, halfDim...]
    return MLX.concatenated([-x2, x1], axis: -1)
  }

  static func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
    guard nRep > 1 else { return x }
    let shape = x.shape
    var expanded = x.expandedDimensions(axis: 2)
    expanded = MLX.broadcast(expanded, to: [shape[0], shape[1], nRep, shape[2], shape[3]])
    return expanded.reshaped(shape[0], shape[1] * nRep, shape[2], shape[3])
  }
}

// MARK: - Gemma 3 MLP

/// Gated MLP for Gemma 3 12B using approximate GELU (gelu_pytorch_tanh).
public final class Gemma3MLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear

  public init(hiddenSize: Int, intermediateSize: Int) {
    self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(geluApproximate(gateProj(x)) * upProj(x))
  }
}

// MARK: - Gemma 3 Transformer Layer

/// Single transformer layer for Gemma 3 12B with 4-norm architecture.
///
/// Architecture:
///   residual -> input_layernorm -> attention -> post_attention_layernorm -> + residual
///   residual -> pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm -> + residual
public final class Gemma3Layer: Module {
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma3RMSNorm
  @ModuleInfo(key: "self_attn") var selfAttn: Gemma3Attention
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma3RMSNorm
  @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma3RMSNorm
  @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma3RMSNorm
  let mlp: Gemma3MLP

  public init(config: LTX2GemmaConfig) {
    self._inputLayerNorm.wrappedValue = Gemma3RMSNorm(
      hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
    self._selfAttn.wrappedValue = Gemma3Attention(config: config)
    self._postAttentionLayerNorm.wrappedValue = Gemma3RMSNorm(
      hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
    self._preFeedforwardLayerNorm.wrappedValue = Gemma3RMSNorm(
      hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
    self._postFeedforwardLayerNorm.wrappedValue = Gemma3RMSNorm(
      hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
    self.mlp = Gemma3MLP(
      hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
  }

  public func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    cosSin: (cos: MLXArray, sin: MLXArray)
  ) -> MLXArray {
    // Attention block: pre-norm -> attn -> post-norm -> residual
    let residual1 = hiddenStates
    var h = inputLayerNorm(hiddenStates)
    h = selfAttn(hiddenStates: h, attentionMask: attentionMask, cosSin: cosSin)
    h = postAttentionLayerNorm(h)
    h = residual1 + h

    // Feed-forward block: pre-norm -> MLP -> post-norm -> residual
    let residual2 = h
    h = preFeedforwardLayerNorm(h)
    h = mlp(h)
    h = postFeedforwardLayerNorm(h)
    h = residual2 + h

    return h
  }
}

// MARK: - Gemma 3 Model

/// Full Gemma 3 12B model for LTX-2 text encoding.
///
/// This wraps the complete Gemma 3 architecture and provides the critical
/// `outputHiddenStates: true` mode that extracts ALL 49 hidden states
/// (1 embedding + 48 layer outputs) needed by LTX-2's feature extractor.
///
/// Uses dual RoPE: global (for full attention layers) and local (for sliding
/// window layers), matching the HuggingFace Gemma 3 implementation.
public final class LTX2GemmaModel: Module {
  public let config: LTX2GemmaConfig

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [Gemma3Layer]
  @ModuleInfo(key: "norm") var norm: Gemma3RMSNorm

  /// Global RoPE: used by full attention layers (every 6th layer)
  /// base=10000, linear scaling factor=8 (theta=1M equiv)
  let rotaryEmbGlobal: Gemma3RoPE

  /// Local RoPE: used by sliding window attention layers
  /// base=10000, no scaling
  let rotaryEmbLocal: Gemma3RoPE

  public init(config: LTX2GemmaConfig = LTX2GemmaConfig()) {
    self.config = config

    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: config.vocabSize,
      dimensions: config.hiddenSize
    )

    self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
      Gemma3Layer(config: config)
    }

    self._norm.wrappedValue = Gemma3RMSNorm(
      hiddenSize: config.hiddenSize,
      eps: config.rmsNormEps
    )

    // Dual RoPE matching HuggingFace Gemma 3:
    // Local: standard base=10000 (rope_local_base_freq)
    // Global: base=10000 with linear scaling factor 8 (rope_scaling.factor)
    self.rotaryEmbLocal = Gemma3RoPE(
      dim: config.headDim,
      base: 10_000.0,
      scaleFactor: 1.0
    )
    self.rotaryEmbGlobal = Gemma3RoPE(
      dim: config.headDim,
      base: 10_000.0,
      scaleFactor: 8.0  // rope_scaling.factor from config
    )
  }

  /// Forward pass through the full Gemma 3 model.
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs `[B, S]`.
  ///   - attentionMask: Padding mask `[B, S]` (1 = attend, 0 = pad).
  ///   - outputHiddenStates: When true, returns all 49 hidden states.
  /// - Returns: Array of hidden states `[embedding, layer0, ..., layer47]` where
  ///   the final entry has the final RMSNorm applied. Each tensor is `[B, S, 3840]`.
  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    outputHiddenStates: Bool = true
  ) -> [MLXArray] {
    let seqLen = inputIds.dim(1)

    // Token embedding with Gemma scaling (sqrt(hidden_size))
    var h = embedTokens(inputIds)
    let scaleFactor = MLXArray(Float(config.hiddenSize).squareRoot()).asType(h.dtype)
    h = h * scaleFactor
    eval(h)

    // Build causal + padding attention mask
    let fullCausalMask = LTX2GemmaModel.buildCausalMask(
      seqLen: seqLen,
      attentionMask: attentionMask,
      dtype: h.dtype
    )

    // Pre-compute both RoPE variants
    let globalCosSin = rotaryEmbGlobal(seqLen)
    let localCosSin = rotaryEmbLocal(seqLen)

    // Collect hidden states
    var hiddenStatesList: [MLXArray] = outputHiddenStates ? [h] : []

    let numLayers = layers.count
    for (i, layer) in layers.enumerated() {
      // Gemma 3 sliding window pattern:
      // Every 6th layer (indices 5, 11, 17, ...) uses full attention + global RoPE
      // All other layers use sliding window attention + local RoPE
      let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
      let cosSin = isGlobal ? globalCosSin : localCosSin

      h = layer(
        hiddenStates: h,
        attentionMask: fullCausalMask,
        cosSin: cosSin
      )
      eval(h)

      if outputHiddenStates && i < numLayers - 1 {
        hiddenStatesList.append(h)
      }
    }

    // Apply final RMSNorm
    h = norm(h)
    eval(h)

    if outputHiddenStates {
      hiddenStatesList.append(h)
      return hiddenStatesList
    }

    return [h]
  }

  // MARK: - Attention Mask

  /// Build a 4D causal + padding attention mask.
  public static func buildCausalMask(
    seqLen: Int,
    attentionMask: MLXArray,
    dtype: DType
  ) -> MLXArray {
    let batchSize = attentionMask.dim(0)
    let minVal: Float = (dtype == .float16 || dtype == .bfloat16) ? -65504.0 : -1e9
    let maskDtype: DType = .float32

    // Padding mask
    let keepMask = attentionMask .== MLXArray(1).asType(attentionMask.dtype)
    let padOnes = MLX.zeros(attentionMask.shape, dtype: maskDtype)
    let padNegInf = MLX.full(attentionMask.shape, values: MLXArray(minVal), dtype: maskDtype)
    var paddingMask = MLX.where(keepMask, padOnes, padNegInf)
    paddingMask = paddingMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)

    // Causal mask
    let idx = MLXArray(0..<seqLen).asType(.int32)
    let j = idx.expandedDimensions(axis: 0)
    let i = idx.expandedDimensions(axis: 1)
    let triBool = j .> i
    let zeros2D = MLX.zeros([seqLen, seqLen], dtype: maskDtype)
    let negInf2D = MLX.full([seqLen, seqLen], values: MLXArray(minVal), dtype: maskDtype)
    var causalMask = MLX.where(triBool, negInf2D, zeros2D)
    causalMask = causalMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    causalMask = MLX.broadcast(causalMask, to: [batchSize, 1, seqLen, seqLen])

    return (causalMask + paddingMask).asType(dtype)
  }

  // MARK: - Weight Loading

  /// Sanitize weight keys from safetensors format.
  public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]
    for (key, value) in weights {
      var newKey: String? = nil
      if key.hasPrefix("language_model.model.") {
        newKey = String(key.dropFirst("language_model.".count))
      } else if key.hasPrefix("model.language_model.") {
        newKey = String(key.dropFirst("model.language_model.".count))
      } else if key.hasPrefix("language_model.") {
        newKey = String(key.dropFirst("language_model.".count))
      }

      guard let finalKey = newKey else { continue }
      guard !finalKey.hasPrefix("vision_tower.") else { continue }

      if value.dtype == .float32 {
        sanitized[finalKey] = value.asType(.bfloat16)
      } else {
        sanitized[finalKey] = value
      }
    }
    return sanitized
  }
}
