// LTX2TransformerBlock.swift — BasicAVTransformerBlock for the DiT transformer
// Phase 3 of the LTX-2 Swift/MLX port
//
// Each of the 48 blocks contains:
// 1. RMSNorm -> AdaLN modulate (scale1, shift1, gate1) -> Self-Attention (+RoPE) -> gate * output -> residual
// 2. RMSNorm -> Cross-Attention(text) -> residual
//    (LTX-2.3: Q modulated by timestep, context modulated by prompt_adaln)
// 3. [Stub] Cross-Modal Attention (A2V/V2A) — deferred to Phase 5
// 4. RMSNorm -> AdaLN modulate (scale2, shift2, gate2) -> GEGLU FFN -> gate * output -> residual
//
// AdaLN produces 6 params (LTX-2) or 9 params (LTX-2.3) from the scale_shift_table
// + timestep embedding. Applied as: x = gate * op(norm(x) * (1 + scale) + shift)
//
// Reference: transformer.py class BasicAVTransformerBlock

import Foundation
import MLX
import MLXFast
import MLXNN

/// A single transformer block in the LTX-2 DiT architecture.
///
/// Processes video-only in Phase 3 (audio support deferred to Phase 5).
/// Each block has self-attention with RoPE, cross-attention with text,
/// and a feed-forward network, all with adaptive layer normalization.
///
/// Weight key mapping:
/// - `transformer_blocks.N.attn1.*` (self-attention)
/// - `transformer_blocks.N.attn2.*` (cross-attention)
/// - `transformer_blocks.N.ff.*` (feed-forward)
/// - `transformer_blocks.N.scale_shift_table` (AdaLN parameters)
/// - `transformer_blocks.N.prompt_scale_shift_table` (LTX-2.3 only)
public final class LTX2TransformerBlock: Module {
  let normEps: Float
  let hasPromptAdaLN: Bool
  let numAdaParams: Int
  private let rmsNormOnes: MLXArray

  // Self-attention
  @ModuleInfo(key: "attn1") var attn1: LTX2Attention

  // Cross-attention with text
  @ModuleInfo(key: "attn2") var attn2: LTX2Attention

  // Feed-forward network
  @ModuleInfo(key: "ff") var ff: LTX2FeedForward

  // AdaLN scale-shift table: (numAdaParams, dim)
  // LTX-2: 6 params (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
  // LTX-2.3: 9 params (adds shift_q, scale_q, gate_q for cross-attention)
  @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

  // LTX-2.3: prompt-conditioned scale-shift for cross-attention context
  @ParameterInfo(key: "prompt_scale_shift_table") var promptScaleShiftTable: MLXArray?

  /// Initialize a transformer block.
  ///
  /// - Parameters:
  ///   - dim: Hidden dimension (inner_dim, e.g. 4096).
  ///   - contextDim: Text embedding dimension for cross-attention (e.g. 4096).
  ///   - heads: Number of attention heads. Default 32.
  ///   - dimHead: Dimension per head. Default 128.
  ///   - normEps: Epsilon for RMSNorm. Default 1e-6.
  ///   - ropeMode: RoPE mode. Default `.split`.
  ///   - hasPromptAdaLN: Whether to use 9-param AdaLN (LTX-2.3). Default false.
  public init(
    dim: Int,
    contextDim: Int,
    heads: Int = 32,
    dimHead: Int = 128,
    normEps: Float = 1e-6,
    ropeMode: LTX2RoPEMode = .split,
    hasPromptAdaLN: Bool = false
  ) {
    self.normEps = normEps
    self.hasPromptAdaLN = hasPromptAdaLN
    self.numAdaParams = hasPromptAdaLN ? 9 : 6
    self.rmsNormOnes = MLXArray.ones([dim]).asType(.bfloat16)

    // Self-attention (no context_dim = self-attention)
    self._attn1.wrappedValue = LTX2Attention(
      queryDim: dim,
      contextDim: nil,
      heads: heads,
      dimHead: dimHead,
      normEps: normEps,
      ropeMode: ropeMode,
      hasGateLogits: hasPromptAdaLN
    )

    // Cross-attention with text
    self._attn2.wrappedValue = LTX2Attention(
      queryDim: dim,
      contextDim: contextDim,
      heads: heads,
      dimHead: dimHead,
      normEps: normEps,
      ropeMode: ropeMode,
      hasGateLogits: hasPromptAdaLN
    )

    // Feed-forward
    self._ff.wrappedValue = LTX2FeedForward(dim: dim, dimOut: dim)

    // AdaLN parameters
    self._scaleShiftTable.wrappedValue = MLXArray.zeros([numAdaParams, dim])

    if hasPromptAdaLN {
      self._promptScaleShiftTable.wrappedValue = MLXArray.zeros([2, dim])
    }
  }

  /// Extract adaptive normalization values from the scale-shift table.
  ///
  /// Combines the learned table parameters with timestep embeddings.
  ///
  /// - Parameters:
  ///   - table: Scale-shift table `(numParams, dim)`.
  ///   - batchSize: Batch size.
  ///   - timestep: Timestep embeddings `(B, 1, numParams * dim)`.
  ///   - range: Range of parameters to extract.
  /// - Returns: Tuple of extracted parameters, each `(B, 1, dim)`.
  private func getAdaValues(
    table: MLXArray,
    batchSize: Int,
    timestep: MLXArray,
    range: Range<Int>
  ) -> [MLXArray] {
    let numTotalParams = table.dim(0)
    let dim = table.dim(1)

    // table[range]: (numSelected, dim) -> (1, 1, numSelected, dim)
    let tableSlice = table[range.lowerBound..<range.upperBound]
    let tableExpanded = tableSlice.expandedDimensions(axes: [0, 1])

    // Reshape timestep: (B, 1, numParams * dim) -> (B, 1, numParams, dim)
    let tsReshaped = timestep.reshaped(batchSize, timestep.dim(1), numTotalParams, dim)

    // Extract relevant indices: (B, 1, numSelected, dim)
    let tsSlice = tsReshaped[0..., 0..., range.lowerBound..<range.upperBound]

    // Add table + timestep: (B, 1, numSelected, dim)
    let adaValues = tableExpanded + tsSlice

    // Unbind along parameter dimension
    var result: [MLXArray] = []
    for i in 0..<(range.upperBound - range.lowerBound) {
      result.append(adaValues[0..., 0..., i])
    }
    return result
  }

  /// Forward pass through the transformer block.
  ///
  /// - Parameters:
  ///   - x: Hidden states `(B, T, dim)`.
  ///   - context: Text embeddings for cross-attention `(B, S, contextDim)`.
  ///   - contextMask: Optional attention mask for cross-attention.
  ///   - timestep: Timestep embeddings `(B, 1, numAdaParams * dim)`.
  ///   - pe: Position embeddings `(cos, sin)` for self-attention RoPE.
  ///   - skipSelfAttn: Skip self-attention (STG perturbation). Default false.
  ///   - promptTimestep: Prompt-conditioned timestep for LTX-2.3 cross-attention.
  /// - Returns: Updated hidden states `(B, T, dim)`.
  public func callAsFunction(
    _ x: MLXArray,
    context: MLXArray,
    contextMask: MLXArray? = nil,
    timestep: MLXArray,
    pe: (cos: MLXArray, sin: MLXArray)? = nil,
    skipSelfAttn: Bool = false,
    promptTimestep: MLXArray? = nil
  ) -> MLXArray {
    let batchSize = x.dim(0)
    var h = x

    // ---- 1. Self-Attention with AdaLN ----

    let msaParams = getAdaValues(
      table: scaleShiftTable,
      batchSize: batchSize,
      timestep: timestep,
      range: 0..<3
    )
    let shiftMSA = msaParams[0]
    let scaleMSA = msaParams[1]
    let gateMSA = msaParams[2]

    // Pre-norm + modulate
    var normH = weightFreeRMSNorm(h)
    normH = normH * (1 + scaleMSA) + shiftMSA

    // Self-attention with RoPE
    let attnOut = attn1(normH, pe: pe, skipAttention: skipSelfAttn)
    h = h + attnOut * gateMSA

    // ---- 2. Cross-Attention with Text ----

    if hasPromptAdaLN {
      // LTX-2.3: Q modulated by timestep (indices 6-8), context modulated by prompt_adaln
      let crossParams = getAdaValues(
        table: scaleShiftTable,
        batchSize: batchSize,
        timestep: timestep,
        range: 6..<9
      )
      let shiftQ = crossParams[0]
      let scaleQ = crossParams[1]
      let gateQ = crossParams[2]

      let attnInput = weightFreeRMSNorm(h) * (1 + scaleQ) + shiftQ

      // Modulate context with prompt timestep
      var encoderStates = context
      if let promptTS = promptTimestep, let promptTable = promptScaleShiftTable {
        let promptParams = getAdaValues(
          table: promptTable,
          batchSize: batchSize,
          timestep: promptTS,
          range: 0..<2
        )
        let promptShift = promptParams[0]
        let promptScale = promptParams[1]
        encoderStates = context * (1 + promptScale) + promptShift
      }

      let crossOut = attn2(attnInput, context: encoderStates, mask: contextMask)
      h = h + crossOut * gateQ
    } else {
      // LTX-2: simple cross-attention with RMSNorm
      let crossOut = attn2(weightFreeRMSNorm(h), context: context, mask: contextMask)
      h = h + crossOut
    }

    // ---- 3. Feed-Forward with AdaLN ----

    let mlpParams = getAdaValues(
      table: scaleShiftTable,
      batchSize: batchSize,
      timestep: timestep,
      range: 3..<6
    )
    let shiftMLP = mlpParams[0]
    let scaleMLP = mlpParams[1]
    let gateMLP = mlpParams[2]

    var normFF = weightFreeRMSNorm(h)
    normFF = normFF * (1 + scaleMLP) + shiftMLP
    let ffOut = ff(normFF)
    h = h + ffOut * gateMLP

    return h
  }

  /// Weight-free RMSNorm using unit weight vector.
  ///
  /// Matches Python's `rms_norm(x, eps)` utility which uses `mx.ones` as weight.
  private func weightFreeRMSNorm(_ x: MLXArray) -> MLXArray {
    return MLXFast.rmsNorm(x, weight: rmsNormOnes, eps: normEps)
  }
}
