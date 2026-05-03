// SmolLM3TextEncoder.swift — SmolLM3-3B text encoder for FIBO
// Ported from mflux: smol_lm3_3b_text_encoder.py
//
// SmolLM3-3B is a 3-billion parameter causal language model used as FIBO's
// text encoder. Unlike CLIP or T5, it's a full LLM — the key difference
// from normal LLM use is that FIBO extracts ALL 36 hidden state layers
// (not just the last) and feeds them per-layer into the transformer via
// DimFusion conditioning.
//
// Architecture:
//   vocab_size: 128256
//   hidden_size: 2048
//   num_hidden_layers: 36
//   num_attention_heads: 16
//   num_key_value_heads: 4 (GQA)
//   intermediate_size: 11008
//   head_dim: 128
//   rope_theta: 5,000,000
//   rms_norm_eps: 1e-6
//   activation: SiLU (for MLP gate)
//   max_position_embeddings: 65536

import MLX
import MLXNN

// MARK: - RMS Norm

/// RMSNorm for SmolLM3-3B, matching the mflux SmolLM3_3B_RMSNorm.
///
/// Casts to float32 for variance computation, applies learned weight
/// parameter, then casts back to input dtype for numerical stability.
public final class SmolLM3RMSNorm: Module {
  @ModuleInfo(key: "weight") var weight: MLXArray
  let eps: Float

  public init(hiddenSize: Int, eps: Float = 1e-6) {
    self.eps = eps
    self._weight.wrappedValue = MLX.ones([hiddenSize])
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let inputDtype = x.dtype
    let h = x.asType(.float32)
    let variance = MLX.mean(h * h, axis: -1, keepDims: true)
    let normed = h * MLX.rsqrt(variance + MLXArray(eps))
    return (weight.asType(.float32) * normed).asType(inputDtype)
  }
}

// MARK: - SmolLM3 Text Encoder

/// Full SmolLM3-3B text encoder model for FIBO.
///
/// Embeds input tokens, applies 36 transformer layers with grouped query
/// attention and rotary position embeddings, and collects hidden states
/// from every layer for DimFusion conditioning in the diffusion transformer.
///
/// The `no_rope_layers` config array controls which layers skip RoPE:
/// a value of `1` at index `i` means layer `i` does NOT apply rotary
/// embeddings. This is a SmolLM3-specific feature (every 4th layer uses
/// plain attention without positional encoding).
///
/// Weight key mapping (safetensors -> model):
/// The `model.` prefix is stripped by `FiboWeightMapping.mapTextEncoderKey`:
/// - `model.embed_tokens.weight` -> `embed_tokens.weight`
/// - `model.layers.N.*` -> `layers.N.*`
/// - `model.norm.weight` -> `norm.weight`
public final class SmolLM3TextEncoder: Module {
  public let config: FiboTextEncoderConfig

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [SmolLM3Layer]
  @ModuleInfo(key: "norm") var norm: SmolLM3RMSNorm

  let rotaryEmb: SmolLM3RoPE

  /// Per-layer RoPE disable flags. `true` = skip RoPE for that layer.
  let noRopeFlags: [Bool]

  public init(config: FiboTextEncoderConfig = FiboTextEncoderConfig()) {
    self.config = config

    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: config.vocabSize,
      dimensions: config.hiddenSize
    )

    self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
      SmolLM3Layer(config: config)
    }

    self._norm.wrappedValue = SmolLM3RMSNorm(
      hiddenSize: config.hiddenSize,
      eps: config.rmsNormEps
    )

    self.rotaryEmb = SmolLM3RoPE(
      dim: config.headDim,
      maxPositionEmbeddings: config.maxPositionEmbeddings,
      base: config.ropeTheta
    )

    // Parse no_rope_layers: array of 0/1 where 1 = skip RoPE
    if let noRopeLayers = config.noRopeLayers, noRopeLayers.count == config.numHiddenLayers {
      self.noRopeFlags = noRopeLayers.map { $0 == 1 }
    } else {
      // Default: all layers use RoPE (matching mflux behavior)
      self.noRopeFlags = Array(repeating: false, count: config.numHiddenLayers)
    }
  }

  /// Forward pass through the full encoder.
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs `[B, S]`.
  ///   - attentionMask: Padding mask `[B, S]` (1 = attend, 0 = pad).
  ///   - outputHiddenStates: When true, collects hidden states from all layers.
  /// - Returns: Array of hidden states: `[embedding, layer0, ..., layer35]` where
  ///   the final layer output has the final RMSNorm applied. Each tensor has
  ///   shape `[B, S, 2048]`. Total of 37 tensors (1 embedding + 36 layers).
  ///
  ///   When `outputHiddenStates` is false, returns a single-element array
  ///   containing the final normed hidden state.
  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    outputHiddenStates: Bool = true
  ) -> [MLXArray] {
    let seqLen = inputIds.dim(1)

    // Token embedding
    var hiddenStates = embedTokens(inputIds)

    // Build 4D causal + padding attention mask (cast to hiddenStates dtype for SDPA)
    let mask4D = SmolLM3TextEncoder.buildAttentionMask(attentionMask: attentionMask)
      .asType(hiddenStates.dtype)

    // Pre-compute RoPE cos/sin for the full sequence
    let (cos, sin) = rotaryEmb(seqLen)

    // Collect hidden states: index 0 = embedding output
    var hiddenStatesList: [MLXArray] = outputHiddenStates ? [hiddenStates] : []

    for (layerIdx, layer) in layers.enumerated() {
      // Determine if this layer should apply RoPE
      let cosSin: (cos: MLXArray, sin: MLXArray)? = noRopeFlags[layerIdx] ? nil : (cos, sin)

      hiddenStates = layer(
        hiddenStates: hiddenStates,
        attentionMask: mask4D,
        cosSin: cosSin
      )

      if outputHiddenStates {
        hiddenStatesList.append(hiddenStates)
      }
    }

    // Apply final layer norm
    hiddenStates = norm(hiddenStates)

    if outputHiddenStates {
      // Replace the last layer output with the normed version
      hiddenStatesList[hiddenStatesList.count - 1] = hiddenStates
      return hiddenStatesList
    }

    return [hiddenStates]
  }

  // MARK: - Attention Mask Construction

  /// Build a combined causal + padding 4D attention mask.
  ///
  /// Matches the mflux Python implementation exactly:
  /// - Padding mask: 0 where attend, -inf where pad, shape `[B, 1, 1, S]`
  /// - Causal mask: upper triangular -inf, shape `[B, 1, S, S]`
  /// - Combined: causal + padding (broadcasting handles the dimension mismatch)
  ///
  /// - Parameter attentionMask: `[B, S]` with 1 = attend, 0 = pad.
  /// - Returns: `[B, 1, S, S]` additive mask.
  static func buildAttentionMask(attentionMask: MLXArray) -> MLXArray {
    let batchSize = attentionMask.dim(0)
    let seqLen = attentionMask.dim(1)
    let maskDtype: DType = .float32
    let minVal = -Float.infinity

    // Padding mask: 0 where attend, -inf where pad
    let ones = MLX.zeros(attentionMask.shape, dtype: maskDtype)
    let negInf = MLX.full(attentionMask.shape, values: MLXArray(minVal), dtype: maskDtype)
    let keepMask = attentionMask .== MLXArray(1).asType(attentionMask.dtype)
    var paddingMask = MLX.where(keepMask, ones, negInf)
    // [B, S] -> [B, 1, 1, S]
    paddingMask = paddingMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)

    // Causal triangular mask: upper triangle = -inf
    let idx = MLXArray(0..<seqLen).asType(.int32)
    let j = idx.expandedDimensions(axis: 0)  // [1, S]
    let i = idx.expandedDimensions(axis: 1)  // [S, 1]
    let triBool = j .> i
    let zeros2D = MLX.zeros([seqLen, seqLen], dtype: maskDtype)
    let negInf2D = MLX.full([seqLen, seqLen], values: MLXArray(minVal), dtype: maskDtype)
    var causalMask = MLX.where(triBool, negInf2D, zeros2D)
    // [S, S] -> [1, 1, S, S] -> [B, 1, S, S]
    causalMask = causalMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    causalMask = MLX.broadcast(causalMask, to: [batchSize, 1, seqLen, seqLen])

    // Combined mask
    return causalMask + paddingMask
  }
}
