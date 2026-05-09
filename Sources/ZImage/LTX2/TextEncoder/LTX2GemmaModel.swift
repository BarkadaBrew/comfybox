// LTX2GemmaModel.swift — Gemma 3 12B language model for LTX-2 text encoding
// Phase 2 of the LTX-2 Swift/MLX port
//
// Implements the Gemma 3 12B transformer model that serves as LTX-2's text
// encoder backbone. The critical difference from standard LLM inference is that
// ALL 49 hidden states (1 embedding + 48 transformer layers) must be extracted,
// not just the final output.
//
// Architecture (Gemma 3 12B from unsloth/gemma-3-12b-it):
//   - Embedding: vocab 262208 -> 3840
//   - 48 transformer layers with sliding window attention
//   - Sliding window pattern of 6 (every 6th layer is global attention)
//   - GQA: 16 query heads, 8 KV heads, head_dim 256
//   - QK normalization: per-head RMSNorm on Q and K after projection
//   - MLP: gelu_pytorch_tanh-gated (gate_proj + up_proj -> down_proj), intermediate 15360
//   - Pre-norm: input_layernorm, post_attention_layernorm
//   - Extra norms: pre_feedforward_layernorm, post_feedforward_layernorm
//   - RMSNorm with eps=1e-6
//   - RoPE with theta=1,000,000
//   - Embedding scaling: h *= sqrt(hidden_size)
//
// Q4 quantization support keeps memory at ~6 GB instead of ~24 GB.
//
// Reference: text_encoder.py class LanguageModel, wrapping mlx_vlm Gemma3Model

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Gemma 3 RMS Norm

/// RMSNorm for Gemma 3, matching the Gemma normalization convention.
///
/// Casts to float32 for variance computation, applies learned weight (+1),
/// then casts back to input dtype for numerical stability.
/// Gemma uses weight + 1 convention (like LLaMA).
public final class Gemma3RMSNorm: Module {
  @ModuleInfo(key: "weight") var weight: MLXArray
  let eps: Float

  public init(hiddenSize: Int, eps: Float = 1e-6) {
    self.eps = eps
    self._weight.wrappedValue = MLX.ones([hiddenSize])
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Use MLXFast.rmsNorm with weight+1 to match Gemma convention
    return MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
  }
}

// MARK: - Gemma 3 RoPE

/// Rotary position embeddings for Gemma 3 12B.
///
/// Uses theta=1,000,000 for extended context support.
/// Output: (cos, sin) each `[1, 1, seqLen, headDim]`.
public final class Gemma3RoPE: Module {
  let dim: Int
  let base: Float
  let invFreq: MLXArray

  public init(dim: Int, base: Float = 1_000_000.0) {
    self.dim = dim
    self.base = base

    // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
    self.invFreq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))
  }

  /// Compute cos/sin tables for the given sequence length.
  ///
  /// - Parameter seqLen: Sequence length.
  /// - Returns: `(cos, sin)` each `[1, 1, seqLen, headDim]`.
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
/// Architecture:
/// - 16 query heads, 8 KV heads (2x GQA ratio)
/// - Head dimension 256
/// - No bias on projections
/// - QK normalization: per-head RMSNorm (dim=head_dim) on Q and K
/// - RoPE applied via rotate-half method
///
/// Weight key mapping:
/// - `model.layers.N.self_attn.q_proj.weight`
/// - `model.layers.N.self_attn.k_proj.weight`
/// - `model.layers.N.self_attn.v_proj.weight`
/// - `model.layers.N.self_attn.o_proj.weight`
/// - `model.layers.N.self_attn.q_norm.weight`
/// - `model.layers.N.self_attn.k_norm.weight`
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
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

  public init(config: LTX2GemmaConfig) {
    self.numHeads = config.numAttentionHeads
    self.numKVHeads = config.numKeyValueHeads
    self.headDim = config.headDim
    self.numKVGroups = config.numAttentionHeads / config.numKeyValueHeads
    // Gemma 3 uses query_pre_attn_scalar = head_dim for scaling
    self.scale = 1.0 / Float(config.headDim).squareRoot()

    let hiddenSize = config.hiddenSize
    self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
    self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
    self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
    self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
    // Per-head QK normalization (dim = head_dim = 256)
    self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
  }

  /// Forward pass through self-attention.
  ///
  /// - Parameters:
  ///   - hiddenStates: Input `[B, S, hiddenSize]`.
  ///   - attentionMask: 4D additive mask `[B, 1, S, S]`.
  ///   - cosSin: `(cos, sin)` from `Gemma3RoPE`, each `[1, 1, S, headDim]`.
  /// - Returns: Output `[B, S, hiddenSize]`.
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

    // Apply per-head QK normalization
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

    // Scaled dot-product attention
    var attnOutput = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: attentionMask.map { .array($0) } ?? .none
    )

    attnOutput = attnOutput.transposed(0, 2, 1, 3)
      .reshaped(batchSize, seqLen, numHeads * headDim)
    return oProj(attnOutput)
  }

  // MARK: - RoPE Application

  /// Apply rotary position embeddings using rotate-half method.
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

  /// Rotate the second half: [-x2, x1]
  static func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<halfDim]
    let x2 = x[.ellipsis, halfDim...]
    return MLX.concatenated([-x2, x1], axis: -1)
  }

  /// Expand KV heads for GQA.
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
///
/// Implements: output = down_proj(GELU_approx(gate_proj(x)) * up_proj(x))
///
/// Weight key mapping:
/// - `model.layers.N.mlp.gate_proj.weight`
/// - `model.layers.N.mlp.up_proj.weight`
/// - `model.layers.N.mlp.down_proj.weight`
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
    // Gemma 3 uses gelu_pytorch_tanh (approximate GELU)
    downProj(geluApproximate(gateProj(x)) * upProj(x))
  }
}

// MARK: - Gemma 3 Transformer Layer

/// Single pre-norm transformer layer for Gemma 3 12B.
///
/// Architecture (matches unsloth/gemma-3-12b-it weights):
///   x -> input_layernorm -> GQA Attention (+ QK norm + RoPE) -> residual add
///   x -> post_attention_layernorm (not used in forward, kept for weight compat)
///   x -> pre_feedforward_layernorm -> Gated MLP -> post_feedforward_layernorm -> residual add
///
/// Note: Gemma 3 has a 4-norm architecture per layer. The naming from HuggingFace:
///   - input_layernorm: pre-attention norm
///   - post_attention_layernorm: post-attention pre-residual norm
///   - pre_feedforward_layernorm: pre-MLP norm
///   - post_feedforward_layernorm: post-MLP pre-residual norm
///
/// Weight key mapping:
/// - `model.layers.N.input_layernorm.weight`
/// - `model.layers.N.self_attn.*`
/// - `model.layers.N.post_attention_layernorm.weight`
/// - `model.layers.N.pre_feedforward_layernorm.weight`
/// - `model.layers.N.post_feedforward_layernorm.weight`
/// - `model.layers.N.mlp.*`
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

  /// Forward pass through the transformer layer.
  ///
  /// Gemma 3 uses a 4-norm architecture:
  ///   residual -> input_layernorm -> attention -> post_attention_layernorm -> + residual
  ///   residual -> pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm -> + residual
  ///
  /// - Parameters:
  ///   - hiddenStates: Input `[B, S, hiddenSize]`.
  ///   - attentionMask: 4D additive mask `[B, 1, S, S]`.
  ///   - cosSin: `(cos, sin)` from `Gemma3RoPE`.
  /// - Returns: Output `[B, S, hiddenSize]`.
  public func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    cosSin: (cos: MLXArray, sin: MLXArray)
  ) -> MLXArray {
    // Self-attention block: pre-norm -> attn -> post-norm -> residual
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
/// Gemma 3 uses a sliding window attention pattern where every 6th layer
/// (i.e., layers 5, 11, 17, ...) uses full global attention, and all other
/// layers use sliding window attention of 1024 tokens.
///
/// Weight key mapping (after `language_model.` prefix strip):
/// - `model.embed_tokens.weight`
/// - `model.layers.N.*`
/// - `model.norm.weight`
public final class LTX2GemmaModel: Module {
  public let config: LTX2GemmaConfig

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [Gemma3Layer]
  @ModuleInfo(key: "norm") var norm: Gemma3RMSNorm

  let rotaryEmb: Gemma3RoPE

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

    self.rotaryEmb = Gemma3RoPE(
      dim: config.headDim,
      base: config.ropeTheta
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
  ///
  ///   When `outputHiddenStates` is false, returns `[finalNormedOutput]`.
  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    outputHiddenStates: Bool = true
  ) -> [MLXArray] {
    let seqLen = inputIds.dim(1)

    // Token embedding with Gemma scaling
    var h = embedTokens(inputIds)
    let scaleFactor = MLXArray(Float(config.hiddenSize).squareRoot()).asType(h.dtype)
    h = h * scaleFactor
    eval(h)

    // Build causal + padding attention masks
    let fullCausalMask = LTX2GemmaModel.buildCausalMask(
      seqLen: seqLen,
      attentionMask: attentionMask,
      dtype: h.dtype
    )

    // Pre-compute RoPE
    let (cos, sin) = rotaryEmb(seqLen)

    // Collect hidden states
    var hiddenStatesList: [MLXArray] = outputHiddenStates ? [h] : []

    let numLayers = layers.count
    for (i, layer) in layers.enumerated() {
      // Gemma 3 sliding window: every Nth layer is global (N = slidingWindowPattern)
      // For layer index i, global when (i % pattern == pattern - 1)
      // Other layers use sliding window mask
      // NOTE: Using full causal mask for all layers (correct but slightly less efficient).
      // This matches the Python reference which also uses the full mask for both.
      let layerMask = fullCausalMask

      h = layer(
        hiddenStates: h,
        attentionMask: layerMask,
        cosSin: (cos, sin)
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
  ///
  /// Matches the Python `_create_causal_mask_with_padding` function:
  /// - Lower triangular causal mask
  /// - Combined with padding mask from attention_mask
  /// - 0 where attend, large negative where masked
  ///
  /// - Parameters:
  ///   - seqLen: Sequence length.
  ///   - attentionMask: `[B, S]` with 1 = attend, 0 = pad.
  ///   - dtype: Output dtype.
  /// - Returns: `[B, 1, S, S]` additive mask.
  public static func buildCausalMask(
    seqLen: Int,
    attentionMask: MLXArray,
    dtype: DType
  ) -> MLXArray {
    let batchSize = attentionMask.dim(0)
    let minVal: Float = (dtype == .float16 || dtype == .bfloat16) ? -65504.0 : -1e9
    let maskDtype: DType = .float32

    // Padding mask: 0 where attend, minVal where pad -> [B, 1, 1, S]
    let keepMask = attentionMask .== MLXArray(1).asType(attentionMask.dtype)
    let padOnes = MLX.zeros(attentionMask.shape, dtype: maskDtype)
    let padNegInf = MLX.full(attentionMask.shape, values: MLXArray(minVal), dtype: maskDtype)
    var paddingMask = MLX.where(keepMask, padOnes, padNegInf)
    // [B, S] -> [B, 1, 1, S]
    paddingMask = paddingMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)

    // Causal mask: upper triangle = minVal -> [1, 1, S, S]
    let idx = MLXArray(0..<seqLen).asType(.int32)
    let j = idx.expandedDimensions(axis: 0)  // (1, S)
    let i = idx.expandedDimensions(axis: 1)  // (S, 1)
    let triBool = j .> i  // upper triangular = true
    let zeros2D = MLX.zeros([seqLen, seqLen], dtype: maskDtype)
    let negInf2D = MLX.full([seqLen, seqLen], values: MLXArray(minVal), dtype: maskDtype)
    var causalMask = MLX.where(triBool, negInf2D, zeros2D)
    // [S, S] -> [1, 1, S, S] -> [B, 1, S, S]
    causalMask = causalMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    causalMask = MLX.broadcast(causalMask, to: [batchSize, 1, seqLen, seqLen])

    // Combined: causal + padding (additive masks, broadcasts [B, 1, 1, S] + [B, 1, S, S])
    return (causalMask + paddingMask).asType(dtype)
  }

  // MARK: - Weight Loading

  /// Sanitize weight keys from safetensors format.
  ///
  /// Handles two common prefix patterns:
  /// - `language_model.model.` -> `model.` (MLX community 4bit)
  /// - `language_model.` -> `` (kept as model.* after strip)
  /// - `model.language_model.` -> `model.` (full VLM from transformers)
  ///
  /// Strips the language_model prefix and keeps the model.* structure that
  /// matches our Module hierarchy (model.embed_tokens, model.layers, model.norm).
  public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]
    for (key, value) in weights {
      var newKey: String? = nil
      if key.hasPrefix("language_model.model.") {
        // MLX community format: language_model.model.layers.0.* -> model.layers.0.*
        newKey = String(key.dropFirst("language_model.".count))
      } else if key.hasPrefix("model.language_model.") {
        // Full VLM format: model.language_model.layers.0.* -> model.layers.0.*
        newKey = String(key.dropFirst("model.language_model.".count))
      } else if key.hasPrefix("language_model.") {
        // Generic: language_model.* -> *
        newKey = String(key.dropFirst("language_model.".count))
      }

      guard let finalKey = newKey else { continue }

      // Skip vision tower weights
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
