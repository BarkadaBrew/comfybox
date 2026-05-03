// Qwen3TextEncoder.swift — Qwen3 text encoder for Flux 2 Klein
// Ported from mflux: qwen3_text_encoder.py, qwen3_vl_decoder_layer.py,
//                     qwen3_vl_attention.py, qwen3_vl_mlp.py, qwen3_vl_rms_norm.py

import MLX
import MLXNN
import MLXFast

// MARK: - Configuration

public struct Qwen3TextEncoderConfiguration {
  public var vocabSize: Int
  public var hiddenSize: Int
  public var numHiddenLayers: Int
  public var numAttentionHeads: Int
  public var numKeyValueHeads: Int
  public var intermediateSize: Int
  public var maxPositionEmbeddings: Int
  public var ropeTheta: Float
  public var rmsNormEps: Float
  public var headDim: Int
  public var attentionBias: Bool
  public var attentionScaling: Float

  public init(
    vocabSize: Int = 151_936,
    hiddenSize: Int = 2560,
    numHiddenLayers: Int = 36,
    numAttentionHeads: Int = 32,
    numKeyValueHeads: Int = 8,
    intermediateSize: Int = 9_728,
    maxPositionEmbeddings: Int = 40_960,
    ropeTheta: Float = 1_000_000.0,
    rmsNormEps: Float = 1e-6,
    headDim: Int = 128,
    attentionBias: Bool = false,
    attentionScaling: Float = 1.0
  ) {
    self.vocabSize = vocabSize
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.intermediateSize = intermediateSize
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.ropeTheta = ropeTheta
    self.rmsNormEps = rmsNormEps
    self.headDim = headDim
    self.attentionBias = attentionBias
    self.attentionScaling = attentionScaling
  }
}

// MARK: - RMS Norm (Qwen3VLRMSNorm port)

/// RMSNorm matching the mflux Qwen3VLRMSNorm: casts to float32 for variance
/// computation, applies learned weight, then casts back to input dtype.
public final class Qwen3RMSNorm: Module {
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

// MARK: - MLP (Qwen3VLMLP port)

/// SwiGLU MLP: gate_proj -> silu, up_proj, multiply, down_proj.
public final class Qwen3MLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear

  public init(hiddenSize: Int, intermediateSize: Int) {
    self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(silu(gateProj(x)) * upProj(x))
  }
}

// MARK: - Attention (Qwen3VLAttention port)

/// Multi-head attention with grouped query attention and rotary position embeddings.
/// Position embeddings are passed as `(cos, sin)` from the external `Qwen3RoPE`.
public final class Qwen3Attention: Module {
  let hiddenSize: Int
  let numAttentionHeads: Int
  let numKeyValueHeads: Int
  let headDim: Int
  let numKeyValueGroups: Int
  let scaling: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear
  @ModuleInfo(key: "q_norm") var qNorm: Qwen3RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: Qwen3RMSNorm

  public init(configuration: Qwen3TextEncoderConfiguration) {
    self.hiddenSize = configuration.hiddenSize
    self.numAttentionHeads = configuration.numAttentionHeads
    self.numKeyValueHeads = configuration.numKeyValueHeads
    self.headDim = configuration.headDim
    self.numKeyValueGroups = configuration.numAttentionHeads / configuration.numKeyValueHeads
    self.scaling = 1.0 / Float(configuration.headDim).squareRoot()

    let bias = configuration.attentionBias
    self._qProj.wrappedValue = Linear(hiddenSize, numAttentionHeads * headDim, bias: bias)
    self._kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: bias)
    self._vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: bias)
    self._oProj.wrappedValue = Linear(numAttentionHeads * headDim, hiddenSize, bias: bias)
    self._qNorm.wrappedValue = Qwen3RMSNorm(hiddenSize: headDim, eps: configuration.rmsNormEps)
    self._kNorm.wrappedValue = Qwen3RMSNorm(hiddenSize: headDim, eps: configuration.rmsNormEps)
  }

  /// - Parameters:
  ///   - hiddenStates: Input tensor `[B, S, hiddenSize]`.
  ///   - attentionMask: 4D additive mask `[B, 1, S, S]` (0 = attend, -inf = mask).
  ///   - positionEmbeddings: `(cos, sin)` from `Qwen3RoPE`.
  /// - Returns: `(output, (cachedKeys, cachedValues))`.
  public func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    let bsz = hiddenStates.dim(0)
    let qLen = hiddenStates.dim(1)

    // Project and reshape to [B, heads, S, headDim]
    var queryStates = qProj(hiddenStates)
      .reshaped(bsz, qLen, numAttentionHeads, headDim)
    var keyStates = kProj(hiddenStates)
      .reshaped(bsz, qLen, numKeyValueHeads, headDim)
    var valueStates = vProj(hiddenStates)
      .reshaped(bsz, qLen, numKeyValueHeads, headDim)

    // QK norm
    queryStates = qNorm(queryStates)
    keyStates = kNorm(keyStates)

    // Transpose to [B, heads, S, headDim]
    queryStates = queryStates.transposed(0, 2, 1, 3)
    keyStates = keyStates.transposed(0, 2, 1, 3)
    valueStates = valueStates.transposed(0, 2, 1, 3)

    // Apply rotary embeddings
    if let (cos, sin) = positionEmbeddings {
      let cosExpanded = cos.expandedDimensions(axis: 1) // [B, 1, S, dim]
      let sinExpanded = sin.expandedDimensions(axis: 1)
      queryStates = (queryStates * cosExpanded) + (Qwen3Attention.rotateHalf(queryStates) * sinExpanded)
      keyStates = (keyStates * cosExpanded) + (Qwen3Attention.rotateHalf(keyStates) * sinExpanded)
    }

    // GQA: expand KV heads to match Q heads
    if numKeyValueHeads != numAttentionHeads {
      keyStates = Qwen3Attention.repeatKV(keyStates, nRep: numKeyValueGroups)
      valueStates = Qwen3Attention.repeatKV(valueStates, nRep: numKeyValueGroups)
    }

    // Attention in float32
    let qF32 = queryStates.asType(.float32)
    let kF32 = keyStates.asType(.float32)
    let vF32 = valueStates.asType(.float32)

    var attnOutput = MLXFast.scaledDotProductAttention(
      queries: qF32,
      keys: kF32,
      values: vF32,
      scale: scaling,
      mask: attentionMask.map { .array($0) } ?? .none
    )

    attnOutput = attnOutput.asType(queryStates.dtype)

    // Reshape back to [B, S, hiddenSize]
    attnOutput = attnOutput.transposed(0, 2, 1, 3).reshaped(bsz, qLen, numAttentionHeads * headDim)
    let output = oProj(attnOutput)

    return (output, (keyStates, valueStates))
  }

  // MARK: - Rotary helpers

  /// Rotate the second half of the last dimension: [-x2, x1]
  static func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<halfDim]
    let x2 = x[.ellipsis, halfDim...]
    return MLX.concatenated([-x2, x1], axis: -1)
  }

  /// Expand KV heads by repeating along the head dimension.
  static func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
    guard nRep > 1 else { return x }
    let shape = x.shape // [B, numKVHeads, S, headDim]
    var expanded = x.expandedDimensions(axis: 2) // [B, numKVHeads, 1, S, headDim]
    expanded = MLX.broadcast(expanded, to: [shape[0], shape[1], nRep, shape[2], shape[3]])
    return expanded.reshaped(shape[0], shape[1] * nRep, shape[2], shape[3])
  }
}

// MARK: - Decoder Layer (Qwen3VLDecoderLayer port)

/// Single transformer decoder layer: layernorm -> attention -> residual -> layernorm -> MLP -> residual.
public final class Qwen3DecoderLayer: Module {
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Qwen3RMSNorm
  @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Qwen3RMSNorm
  let mlp: Qwen3MLP

  public init(configuration: Qwen3TextEncoderConfiguration) {
    self._inputLayerNorm.wrappedValue = Qwen3RMSNorm(
      hiddenSize: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self._selfAttn.wrappedValue = Qwen3Attention(configuration: configuration)
    self._postAttentionLayerNorm.wrappedValue = Qwen3RMSNorm(
      hiddenSize: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self.mlp = Qwen3MLP(
      hiddenSize: configuration.hiddenSize, intermediateSize: configuration.intermediateSize)
  }

  /// - Returns: `(hiddenStates, (cachedKeys, cachedValues))`
  public func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    let residual = hiddenStates
    let normed = inputLayerNorm(hiddenStates)
    let (attnOutput, pastKV) = selfAttn(
      hiddenStates: normed,
      attentionMask: attentionMask,
      positionEmbeddings: positionEmbeddings
    )
    var h = residual + attnOutput

    let residual2 = h
    h = postAttentionLayerNorm(h)
    h = mlp(h)
    h = residual2 + h

    return (h, pastKV)
  }
}

// MARK: - Qwen3 Text Encoder (top-level model)

/// Full Qwen3 text encoder model for Flux 2 Klein.
///
/// Embeds input tokens, applies 36 transformer decoder layers with rotary
/// position embeddings, and produces either the final hidden state or
/// multi-layer prompt embeddings (layers 9, 18, 27) for the diffusion model.
public final class Qwen3TextEncoder: Module {
  public let configuration: Qwen3TextEncoderConfiguration

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [Qwen3DecoderLayer]
  @ModuleInfo(key: "norm") var norm: Qwen3RMSNorm

  let rotaryEmb: Qwen3RoPE

  public init(configuration: Qwen3TextEncoderConfiguration = .init()) {
    self.configuration = configuration

    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: configuration.vocabSize,
      dimensions: configuration.hiddenSize
    )

    self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
      Qwen3DecoderLayer(configuration: configuration)
    }

    self._norm.wrappedValue = Qwen3RMSNorm(
      hiddenSize: configuration.hiddenSize,
      eps: configuration.rmsNormEps
    )

    self.rotaryEmb = Qwen3RoPE(
      dim: configuration.headDim,
      maxPositionEmbeddings: configuration.maxPositionEmbeddings,
      base: configuration.ropeTheta,
      scalingFactor: configuration.attentionScaling
    )
  }

  /// Forward pass through the full encoder.
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs `[B, S]`.
  ///   - attentionMask: Optional `[B, S]` mask (1 = attend, 0 = pad).
  ///   - outputHiddenStates: When true, collects hidden states after each layer.
  /// - Returns: `(lastHiddenState, hiddenStatesList)` where the list includes
  ///   the embedding output at index 0 and each layer output at indices 1..N.
  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray? = nil,
    outputHiddenStates: Bool = false
  ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
    let batchSize = inputIds.dim(0)
    let seqLen = inputIds.dim(1)

    var hiddenStates = embedTokens(inputIds)

    // Build 4D causal + padding attention mask
    let mask: MLXArray?
    if let attentionMask {
      mask = buildAttentionMask4D(
        attentionMask: attentionMask,
        batchSize: batchSize,
        seqLen: seqLen,
        dtype: hiddenStates.dtype
      )
    } else {
      mask = buildCausalMask(batchSize: batchSize, seqLen: seqLen, dtype: hiddenStates.dtype)
    }

    // Position IDs: [B, S]
    let posIds = MLX.broadcast(
      MLXArray(0..<seqLen).expandedDimensions(axis: 0),
      to: [batchSize, seqLen]
    )
    let positionEmbeddings = rotaryEmb(hiddenStates, positionIds: posIds)

    // Collect hidden states (embedding output is index 0)
    var hiddenStatesList: [MLXArray]? = outputHiddenStates ? [hiddenStates] : nil

    for layer in layers {
      let (output, _) = layer(
        hiddenStates,
        attentionMask: mask,
        positionEmbeddings: positionEmbeddings
      )
      hiddenStates = output
      if outputHiddenStates {
        hiddenStatesList?.append(hiddenStates)
      }
    }

    hiddenStates = norm(hiddenStates)

    return (hiddenStates, hiddenStatesList)
  }

  /// Extract multi-layer prompt embeddings for the diffusion model.
  ///
  /// Runs the full encoder with hidden state collection, then stacks outputs
  /// from the specified layers and interleaves them into a single embedding.
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs `[B, S]`.
  ///   - attentionMask: Optional `[B, S]` mask.
  ///   - hiddenStateLayers: Layer indices to extract (default: 9, 18, 27).
  /// - Returns: Prompt embeddings `[B, S, numLayers * hiddenDim]`.
  public func getPromptEmbeds(
    inputIds: MLXArray,
    attentionMask: MLXArray? = nil,
    hiddenStateLayers: [Int] = [9, 18, 27]
  ) -> MLXArray {
    let (_, hiddenStatesList) = self(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: true
    )

    guard let allStates = hiddenStatesList else {
      fatalError("Hidden states not available for prompt embedding")
    }

    // Stack selected layers: [B, numLayers, S, hiddenDim]
    let selected = hiddenStateLayers.map { allStates[$0] }
    let stacked = MLX.stacked(selected, axis: 1)

    let batchSize = stacked.dim(0)
    let numLayers = stacked.dim(1)
    let seqLen = stacked.dim(2)
    let hiddenDim = stacked.dim(3)

    // Transpose to [B, S, numLayers, hiddenDim] then reshape to [B, S, numLayers * hiddenDim]
    let transposed = stacked.transposed(0, 2, 1, 3)
    return transposed.reshaped(batchSize, seqLen, numLayers * hiddenDim)
  }

  // MARK: - Mask construction

  /// Build a combined causal + padding 4D attention mask.
  ///
  /// Matches the mflux Python implementation exactly:
  /// - Padding mask: 0 where attend, -inf where pad, expanded to `[B, 1, 1, S]`
  /// - Causal mask: upper triangular -inf, expanded to `[B, 1, S, S]`
  /// - Combined: causal + padding
  private func buildAttentionMask4D(
    attentionMask: MLXArray,
    batchSize: Int,
    seqLen: Int,
    dtype: DType
  ) -> MLXArray {
    let ones = MLX.zeros(attentionMask.shape, dtype: dtype)
    let negInf = MLX.full(attentionMask.shape, values: MLXArray(-Float.infinity), dtype: dtype)
    let keepMask = attentionMask .== MLXArray(1).asType(attentionMask.dtype)
    var paddingMask = MLX.where(keepMask, ones, negInf)
    // [B, S] -> [B, 1, 1, S]
    paddingMask = paddingMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)

    if seqLen == 1 {
      return MLX.zeros([batchSize, 1, 1, 1], dtype: dtype)
    }

    // Causal triangular mask
    let idx = MLXArray(0..<seqLen)
    let j = idx.expandedDimensions(axis: 0) // [1, S]
    let i = idx.expandedDimensions(axis: 1) // [S, 1]
    let triBool = j .> i
    let zeros2D = MLX.zeros([seqLen, seqLen], dtype: dtype)
    let negInf2D = MLX.full([seqLen, seqLen], values: MLXArray(-Float.infinity), dtype: dtype)
    var causalMask = MLX.where(triBool, negInf2D, zeros2D)
    // [S, S] -> [1, 1, S, S] -> [B, 1, S, S]
    causalMask = causalMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    causalMask = MLX.broadcast(causalMask, to: [batchSize, 1, seqLen, seqLen])

    return causalMask + paddingMask
  }

  /// Build a causal-only mask (no padding) for when attentionMask is nil.
  private func buildCausalMask(
    batchSize: Int,
    seqLen: Int,
    dtype: DType
  ) -> MLXArray? {
    guard seqLen > 1 else { return nil }

    let idx = MLXArray(0..<seqLen)
    let j = idx.expandedDimensions(axis: 0)
    let i = idx.expandedDimensions(axis: 1)
    let triBool = j .> i
    let zeros2D = MLX.zeros([seqLen, seqLen], dtype: dtype)
    let negInf2D = MLX.full([seqLen, seqLen], values: MLXArray(-Float.infinity), dtype: dtype)
    var causalMask = MLX.where(triBool, negInf2D, zeros2D)
    causalMask = causalMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    causalMask = MLX.broadcast(causalMask, to: [batchSize, 1, seqLen, seqLen])
    return causalMask
  }
}
