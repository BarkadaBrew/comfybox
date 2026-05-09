// LTX2Connector1D.swift — 1D transformer connector for dual video/audio embeddings
// Phase 2 of the LTX-2 Swift/MLX port
//
// The 1D connector takes text features from the feature extractor and produces
// embeddings suitable for cross-attention in the main diffusion transformer.
// It uses learnable register tokens to replace padding, RoPE for position
// encoding, and a stack of transformer blocks.
//
// LTX-2 (original): 2-layer connector, 30 heads, shared 3840-dim for video/audio
// LTX-2.3 (prompt adaln): 8-layer connector, 32 heads, separate video (4096) and
//   audio (2048) connectors with gate logits
//
// The 1D connector uses SPLIT RoPE (not interleaved) to match the Python reference.
//
// Reference: text_encoder.py classes Embeddings1DConnector, ConnectorTransformerBlock,
//            ConnectorAttention, ConnectorFeedForward

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Connector Attention

/// Self-attention for the 1D connector with QK-RMSNorm and optional gate logits.
///
/// Key differences from standard attention:
/// - RMSNorm applied to full Q and K vectors BEFORE reshape to heads
/// - SPLIT RoPE (not rotate-half): splits along last dim, applies cos/sin
/// - Optional per-head gate logits for output modulation (LTX-2.3)
///
/// Weight key mapping:
/// - `transformer_1d_blocks.N.attn1.to_q.weight`, `.bias`
/// - `transformer_1d_blocks.N.attn1.to_k.weight`, `.bias`
/// - `transformer_1d_blocks.N.attn1.to_v.weight`, `.bias`
/// - `transformer_1d_blocks.N.attn1.to_out.weight`, `.bias`
/// - `transformer_1d_blocks.N.attn1.q_norm.weight`
/// - `transformer_1d_blocks.N.attn1.k_norm.weight`
/// - `transformer_1d_blocks.N.attn1.to_gate_logits.weight`, `.bias` (optional)
public final class LTX2ConnectorAttention: Module {
  let numHeads: Int
  let headDim: Int
  let scale: Float
  let hasGateLogits: Bool

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: Linear
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

  // Optional gate logits (LTX-2.3 only)
  @ModuleInfo(key: "to_gate_logits") var toGateLogits: Linear?

  public init(config: LTX2ConnectorConfig) {
    self.numHeads = config.numHeads
    self.headDim = config.headDim
    self.scale = 1.0 / Float(config.headDim).squareRoot()
    self.hasGateLogits = config.hasGateLogits

    let innerDim = config.innerDim

    self._toQ.wrappedValue = Linear(config.dim, innerDim, bias: true)
    self._toK.wrappedValue = Linear(config.dim, innerDim, bias: true)
    self._toV.wrappedValue = Linear(config.dim, innerDim, bias: true)
    self._toOut.wrappedValue = Linear(innerDim, config.dim, bias: true)

    self._qNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: 1e-6)
    self._kNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: 1e-6)

    if config.hasGateLogits {
      self._toGateLogits.wrappedValue = Linear(config.dim, config.numHeads, bias: true)
    }
  }

  /// Forward pass through connector attention.
  ///
  /// - Parameters:
  ///   - x: Input `[B, S, dim]`.
  ///   - pe: Optional RoPE frequencies `(cos, sin)`, each `[1, H, S, D//2]`.
  /// - Returns: Output `[B, S, dim]`.
  public func callAsFunction(
    _ x: MLXArray,
    pe: (cos: MLXArray, sin: MLXArray)? = nil
  ) -> MLXArray {
    let batchSize = x.dim(0)
    let seqLen = x.dim(1)

    // Compute per-head gate early (from original input)
    var gate: MLXArray? = nil
    if hasGateLogits, let gateProj = toGateLogits {
      gate = 2.0 * sigmoid(gateProj(x))  // (B, S, H)
    }

    // Project Q, K, V
    var q = toQ(x)
    var k = toK(x)
    var v = toV(x)

    // QK normalization BEFORE reshape (on full inner_dim)
    q = qNorm(q)
    k = kNorm(k)

    // Reshape to (B, H, S, D) for SPLIT RoPE
    q = q.reshaped(batchSize, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
    k = k.reshaped(batchSize, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
    v = v.reshaped(batchSize, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

    // Apply SPLIT RoPE if provided
    if let pe = pe {
      q = LTX2ConnectorAttention.applySplitRoPE(q, cos: pe.cos, sin: pe.sin)
      k = LTX2ConnectorAttention.applySplitRoPE(k, cos: pe.cos, sin: pe.sin)
    }

    // Scaled dot-product attention (no mask needed after register replacement)
    var out = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v,
      scale: scale,
      mask: .none
    )

    // Reshape back: (B, H, S, D) -> (B, S, H*D)
    out = out.transposed(0, 2, 1, 3).reshaped(batchSize, seqLen, -1)

    // Apply per-head gating (LTX-2.3)
    if let gate = gate {
      out = out.reshaped(batchSize, seqLen, numHeads, headDim)
      out = out * gate.expandedDimensions(axis: -1)
      out = out.reshaped(batchSize, seqLen, -1)
    }

    return toOut(out)
  }

  // MARK: - SPLIT RoPE

  /// Apply SPLIT RoPE to input tensor.
  ///
  /// Unlike rotate-half RoPE, SPLIT RoPE splits the tensor in two halves along
  /// the last dimension and applies rotation to each half:
  ///   out1 = x1 * cos - x2 * sin
  ///   out2 = x2 * cos + x1 * sin
  ///
  /// - Parameters:
  ///   - x: Input `(B, H, S, D)`.
  ///   - cos: Cosine frequencies `(1, H, S, D//2)`.
  ///   - sin: Sine frequencies `(1, H, S, D//2)`.
  /// - Returns: Tensor with SPLIT rotary embeddings applied.
  static func applySplitRoPE(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray
  ) -> MLXArray {
    let inputDtype = x.dtype

    let xF = x.asType(.float32)
    let cosF = cos.asType(.float32)
    let sinF = sin.asType(.float32)

    let halfDim = x.dim(-1) / 2
    let x1 = xF[.ellipsis, ..<halfDim]
    let x2 = xF[.ellipsis, halfDim...]

    let out1 = x1 * cosF - x2 * sinF
    let out2 = x2 * cosF + x1 * sinF

    return MLX.concatenated([out1, out2], axis: -1).asType(inputDtype)
  }
}

// MARK: - Connector Feed Forward

/// Feed-forward network for the 1D connector.
///
/// Uses GELU approximate activation (matching Python `gelu_approx`).
///
/// Weight key mapping:
/// - `transformer_1d_blocks.N.ff.proj_in.weight`, `.bias`
/// - `transformer_1d_blocks.N.ff.proj_out.weight`, `.bias`
public final class LTX2ConnectorFeedForward: Module {
  @ModuleInfo(key: "proj_in") var projIn: Linear
  @ModuleInfo(key: "proj_out") var projOut: Linear

  public init(dim: Int, mult: Int = 4) {
    let innerDim = dim * mult
    self._projIn.wrappedValue = Linear(dim, innerDim, bias: true)
    self._projOut.wrappedValue = Linear(innerDim, dim, bias: true)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = projIn(x)
    h = geluApproximate(h)
    h = projOut(h)
    return h
  }
}

// MARK: - Connector Transformer Block

/// Single transformer block for the 1D connector.
///
/// Architecture:
///   x -> RMSNorm(weight-free) -> Attention(+SPLIT RoPE, +QK-Norm) -> residual
///   x -> RMSNorm(weight-free) -> FFN(GELU) -> residual
///
/// Uses weight-free RMSNorm (unit weight vector) for pre-norm, matching the
/// Python reference which uses the `rms_norm` utility function.
public final class LTX2ConnectorTransformerBlock: Module {
  @ModuleInfo(key: "attn1") var attn1: LTX2ConnectorAttention
  @ModuleInfo(key: "ff") var ff: LTX2ConnectorFeedForward

  public init(config: LTX2ConnectorConfig) {
    self._attn1.wrappedValue = LTX2ConnectorAttention(config: config)
    self._ff.wrappedValue = LTX2ConnectorFeedForward(dim: config.dim)
  }

  /// Forward pass through the block.
  ///
  /// - Parameters:
  ///   - x: Input `[B, S, dim]`.
  ///   - pe: Optional RoPE frequencies.
  /// - Returns: Output `[B, S, dim]`.
  public func callAsFunction(
    _ x: MLXArray,
    pe: (cos: MLXArray, sin: MLXArray)? = nil
  ) -> MLXArray {
    // Pre-norm + attention + residual
    var normX = weightFreeRMSNorm(x)
    if normX.ndim == 4 {
      normX = normX.squeezed(axis: 1)
    }
    let attnOut = attn1(normX, pe: pe)
    var h = x + attnOut
    if h.ndim == 4 {
      h = h.squeezed(axis: 1)
    }

    // Pre-norm + FFN + residual
    normX = weightFreeRMSNorm(h)
    let ffOut = ff(normX)
    h = h + ffOut
    if h.ndim == 4 {
      h = h.squeezed(axis: 1)
    }

    return h
  }

  /// Weight-free RMSNorm using unit weight vector.
  private func weightFreeRMSNorm(_ x: MLXArray) -> MLXArray {
    let dim = x.dim(-1)
    let ones = MLXArray.ones([dim])
    return MLXFast.rmsNorm(x, weight: ones, eps: 1e-6)
  }
}

// MARK: - 1D Embeddings Connector

/// 1D Embeddings Connector that produces text embeddings for the diffusion transformer.
///
/// Pipeline:
/// 1. Replace padded tokens with learnable register tokens
/// 2. Compute 1D SPLIT RoPE frequencies
/// 3. Process through N transformer blocks
/// 4. Apply final weight-free RMSNorm
///
/// The connector does NOT project to different dimensions — it maintains the input
/// dimension throughout. The dual video/audio output comes from having separate
/// connector instances (one for video features, one for audio features) when using
/// the V2 feature extractor (LTX-2.3).
///
/// For LTX-2 original, a single connector is used with shared features, and separate
/// `AudioEmbeddingsConnector` projections are applied afterwards.
///
/// Weight key mapping:
/// - `video_embeddings_connector.transformer_1d_blocks.N.*`
/// - `video_embeddings_connector.learnable_registers`
/// - `audio_embeddings_connector.transformer_1d_blocks.N.*`
/// - `audio_embeddings_connector.learnable_registers`
public final class LTX2Connector1D: Module {
  let config: LTX2ConnectorConfig

  @ModuleInfo(key: "transformer_1d_blocks") var blocks: [LTX2ConnectorTransformerBlock]

  /// Learnable register tokens. Plain array, not a Module parameter — must be
  /// loaded manually from weights (not auto-discovered by MLXNN).
  public var learnableRegisters: MLXArray

  public init(config: LTX2ConnectorConfig) {
    self.config = config

    self._blocks.wrappedValue = (0..<config.numLayers).map { _ in
      LTX2ConnectorTransformerBlock(config: config)
    }

    if config.numLearnableRegisters > 0 {
      self.learnableRegisters = MLX.zeros([config.numLearnableRegisters, config.dim])
    } else {
      self.learnableRegisters = MLX.zeros([0, config.dim])
    }
  }

  /// Forward pass through the 1D connector.
  ///
  /// - Parameters:
  ///   - hiddenStates: Text features `[B, S, dim]`.
  ///   - attentionMask: Optional additive mask `[B, 1, 1, S]` (0 = attend, -1e9 = pad).
  /// - Returns: `(processedHiddenStates, processedMask)` with same shapes.
  public func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> (MLXArray, MLXArray?) {
    var h = hiddenStates
    var mask = attentionMask

    // Replace padded tokens with learnable registers
    if config.numLearnableRegisters > 0, let attMask = mask {
      (h, mask) = replacePaddedWithRegisters(hiddenStates: h, attentionMask: attMask)
    }

    // Compute SPLIT RoPE frequencies
    let seqLen = h.dim(1)
    let pe = precomputeFreqsCIS(seqLen: seqLen, dtype: h.dtype)

    // Process through transformer blocks
    for block in blocks {
      h = block(h, pe: pe)
    }

    // Final weight-free RMSNorm
    let dim = h.dim(-1)
    let ones = MLXArray.ones([dim])
    h = MLXFast.rmsNorm(h, weight: ones, eps: 1e-6)

    return (h, mask)
  }

  // MARK: - RoPE Frequency Computation

  /// Compute SPLIT RoPE frequencies for the connector.
  ///
  /// Returns `(cos, sin)` each shaped `[1, numHeads, seqLen, headDim//2]`.
  ///
  /// The frequency computation follows the Python `_precompute_freqs_cis`:
  /// 1. Generate log-spaced indices from 1.0 to theta
  /// 2. Scale by pi/2
  /// 3. Compute fractional positions scaled to [-1, 1]
  /// 4. Outer product: scaled_positions * indices
  /// 5. Pad if needed (when n_elem < dim/2)
  /// 6. cos/sin, reshape to per-head layout
  private func precomputeFreqsCIS(
    seqLen: Int,
    dtype: DType
  ) -> (cos: MLXArray, sin: MLXArray) {
    let dim = config.innerDim  // numHeads * headDim
    let theta = config.positionalEmbeddingTheta
    let maxPos = config.positionalEmbeddingMaxPos
    let nElem = 2 * maxPos.count  // typically 2

    let numIndices = dim / nElem

    // Generate log-spaced indices: theta^linspace(0, 1, numIndices) * pi/2
    var indices = [Float](repeating: 0, count: numIndices)
    for i in 0..<numIndices {
      let t = numIndices > 1 ? Float(i) / Float(numIndices - 1) : 0.0
      indices[i] = powf(theta, t) * (Float.pi / 2.0)
    }

    // Generate positions and scale to [-1, 1]
    var positions = [Float](repeating: 0, count: seqLen)
    for i in 0..<seqLen {
      let fractional = Float(i) / Float(maxPos[0])
      positions[i] = fractional * 2.0 - 1.0
    }

    // Outer product: scaled_positions * indices -> (seqLen, numIndices)
    var freqs = [Float](repeating: 0, count: seqLen * numIndices)
    for i in 0..<seqLen {
      for j in 0..<numIndices {
        freqs[i * numIndices + j] = positions[i] * indices[j]
      }
    }

    // cos/sin
    let expectedFreqs = dim / 2
    let padSize = expectedFreqs - numIndices

    var cosFreqs = [Float](repeating: 0, count: seqLen * expectedFreqs)
    var sinFreqs = [Float](repeating: 0, count: seqLen * expectedFreqs)

    for i in 0..<seqLen {
      // Padding at the front (cos = 1, sin = 0)
      for j in 0..<padSize {
        cosFreqs[i * expectedFreqs + j] = 1.0
        sinFreqs[i * expectedFreqs + j] = 0.0
      }
      // Computed frequencies
      for j in 0..<numIndices {
        let f = freqs[i * numIndices + j]
        cosFreqs[i * expectedFreqs + padSize + j] = cosf(f)
        sinFreqs[i * expectedFreqs + padSize + j] = sinf(f)
      }
    }

    // Reshape: (seqLen, expectedFreqs) -> (seqLen, numHeads, headDim/2) -> (1, numHeads, seqLen, headDim/2)
    let halfHeadDim = config.headDim / 2
    let cosArr = MLXArray(cosFreqs, [seqLen, config.numHeads, halfHeadDim])
    let sinArr = MLXArray(sinFreqs, [seqLen, config.numHeads, halfHeadDim])

    // Transpose: (S, H, D/2) -> (H, S, D/2) -> (1, H, S, D/2)
    let cosFinal = cosArr.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(dtype)
    let sinFinal = sinArr.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(dtype)

    return (cosFinal, sinFinal)
  }

  // MARK: - Register Replacement

  /// Replace padded positions with learnable register tokens.
  ///
  /// For left-padded input, valid tokens are at the end. This function:
  /// 1. Moves valid tokens to the front
  /// 2. Fills remaining positions with tiled register tokens
  /// 3. Resets the attention mask to all-zeros (no masking needed)
  ///
  /// Matches Python `_replace_padded_with_registers`.
  private func replacePaddedWithRegisters(
    hiddenStates: MLXArray,
    attentionMask: MLXArray
  ) -> (MLXArray, MLXArray?) {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)
    let dim = hiddenStates.dim(2)
    let dtype = hiddenStates.dtype

    // Binary mask: 1 for valid, 0 for padded
    // attentionMask is additive: 0 for valid, large negative for padded
    let maskSquashed = attentionMask.squeezed(axis: 1).squeezed(axis: 1)  // (B, S)
    let maskBinary = (maskSquashed .>= MLXArray(Float(-9000.0))).asType(.int32)

    // Tile registers to fill sequence length
    let numTiles = seqLen / config.numLearnableRegisters
    let registers = MLX.tiled(learnableRegisters, repetitions: [numTiles, 1])
      .asType(dtype)  // (seqLen, dim)

    // Process each batch item
    var resultList: [MLXArray] = []
    for b in 0..<batchSize {
      let maskB = maskBinary[b]  // (S,)
      let hsB = hiddenStates[b]  // (S, dim)

      let numValid = Int(MLX.sum(maskB).item(Int.self))
      let padLength = seqLen - numValid

      // Valid tokens are at the end (left-padded)
      let validTokens = hsB[(seqLen - numValid)...]  // (numValid, dim)

      // Build: valid tokens at front, registers at back
      let adjusted: MLXArray
      if padLength > 0 {
        let padding = MLX.zeros([padLength, dim], dtype: dtype)
        adjusted = MLX.concatenated([validTokens, padding], axis: 0)
      } else {
        adjusted = validTokens
      }

      // Flipped mask: 1s at front (valid), 0s at back (register positions)
      let flippedMask: MLXArray
      if padLength > 0 {
        flippedMask = MLX.concatenated([
          MLX.ones([numValid], dtype: .int32),
          MLX.zeros([padLength], dtype: .int32),
        ], axis: 0)
      } else {
        flippedMask = MLX.ones([seqLen], dtype: .int32)
      }

      // Combine: valid tokens where mask=1, registers where mask=0
      let flippedMaskExpanded = flippedMask.expandedDimensions(axis: 1).asType(dtype)
      let combined = flippedMaskExpanded * adjusted + (1 - flippedMaskExpanded) * registers
      resultList.append(combined)
    }

    let result = MLX.stacked(resultList, axis: 0)

    // Reset attention mask to all zeros (no masking after register replacement)
    let newMask = MLX.zeros(like: attentionMask)

    return (result, newMask)
  }
}
