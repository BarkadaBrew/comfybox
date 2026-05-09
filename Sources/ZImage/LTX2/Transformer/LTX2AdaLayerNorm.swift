// LTX2AdaLayerNorm.swift — Adaptive Layer Normalization with timestep conditioning
// Phase 3 of the LTX-2 Swift/MLX port
//
// AdaLayerNormSingle takes a timestep, computes a sinusoidal+MLP embedding, then
// projects it through a linear layer to produce per-block modulation parameters.
//
// For LTX-2: 6 params per block (shift1, scale1, gate1, shift2, scale2, gate2)
// For LTX-2.3: 9 params (adds cross-attn shift, scale, gate)
//
// The output is chunked into groups of inner_dim and used by each transformer block
// to modulate the hidden states via: x = gate * attn(norm(x) * (1 + scale) + shift)
//
// Reference: adaln.py class AdaLayerNormSingle

import Foundation
import MLX
import MLXNN

/// Adaptive Layer Normalization conditioned on timestep.
///
/// Pipeline:
///   timestep -> CombinedTimestepEmbeddings -> SiLU -> Linear(dim, coeff * dim)
///   Returns (scale_shift_params, embedded_timestep)
///
/// The scale_shift_params are later chunked by transformer blocks to extract
/// per-operation modulation values (shift, scale, gate).
///
/// Weight key mapping:
/// - `adaln_single.emb.*` (CombinedTimestepEmbeddings)
/// - `adaln_single.linear.weight`, `.bias`
public final class LTX2AdaLayerNormSingle: Module {
  @ModuleInfo(key: "emb") var emb: LTX2CombinedTimestepEmbeddings
  @ModuleInfo(key: "linear") var linear: Linear

  /// Number of modulation parameters per block.
  /// LTX-2: 6 (shift1, scale1, gate1, shift2, scale2, gate2)
  /// LTX-2.3: 9 (adds cross-attn shift, scale, gate)
  let embeddingCoefficient: Int

  public init(
    embeddingDim: Int,
    embeddingCoefficient: Int = 6
  ) {
    self.embeddingCoefficient = embeddingCoefficient
    self._emb.wrappedValue = LTX2CombinedTimestepEmbeddings(embeddingDim: embeddingDim)
    self._linear.wrappedValue = Linear(embeddingDim, embeddingCoefficient * embeddingDim, bias: true)
  }

  /// Compute adaptive normalization parameters from timestep.
  ///
  /// - Parameters:
  ///   - timestep: Flattened timestep values `(B,)`.
  ///   - hiddenDtype: Target dtype for computation.
  /// - Returns: `(scaleShiftParams, embeddedTimestep)` where
  ///   scaleShiftParams is `(B, coeff * dim)` and embeddedTimestep is `(B, dim)`.
  public func callAsFunction(
    _ timestep: MLXArray,
    hiddenDtype: DType? = nil
  ) -> (MLXArray, MLXArray) {
    let embeddedTimestep = emb(timestep, hiddenDtype: hiddenDtype)
    let scaleShiftParams = linear(silu(embeddedTimestep))
    return (scaleShiftParams, embeddedTimestep)
  }
}
