// FiboAdaLayerNorm.swift — Adaptive layer normalization variants for FIBO
// Ported from mflux: fibo_ada_layer_norm_zero.py (FiboAdaLayerNormZero) +
//                    ada_layer_norm_zero_single.py (AdaLayerNormZeroSingle) +
//                    ada_layer_norm_continuous.py (AdaLayerNormContinuous)
//
// FIBO uses three AdaLN variants:
// 1. FiboAdaLayerNormZero — joint blocks: 6-way split (shift/scale/gate for both streams)
// 2. AdaLayerNormZeroSingle — single blocks: 3-way split (shift/scale/gate)
// 3. AdaLayerNormContinuous — output norm: 2-way split (scale/shift)

import Foundation
import MLX
import MLXNN

// MARK: - FiboAdaLayerNormZero (Joint Blocks)

/// Adaptive layer norm with gating for FIBO joint transformer blocks.
///
/// Produces 6 modulation parameters from the DimFusion text embedding:
/// `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`
///
/// The norm is applied as manual float32 layer norm (mean/var normalization)
/// matching the reference implementation's `_layer_norm` static method.
///
/// Weight key path: `transformer_blocks.{i}.norm1.linear.{weight,bias}` (image stream)
///                  `transformer_blocks.{i}.norm1_context.linear.{weight,bias}` (text stream)
final class FiboAdaLayerNormZero: Module {
  let embeddingDim: Int
  let eps: Float

  @ModuleInfo(key: "linear") var linear: Linear

  init(embeddingDim: Int = 3072, eps: Float = 1e-6) {
    self.embeddingDim = embeddingDim
    self.eps = eps
    self._linear.wrappedValue = Linear(embeddingDim, 6 * embeddingDim)
    super.init()
  }

  /// - Parameters:
  ///   - hiddenStates: Input tensor `[batch, seq, dim]`.
  ///   - textEmbeddings: DimFusion conditioning `[batch, dim]`.
  /// - Returns: `(normedHidden, gateMSA, shiftMLP, scaleMLP, gateMLP)`.
  func callAsFunction(
    hiddenStates: MLXArray,
    textEmbeddings: MLXArray
  ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
    let emb = linear(silu(textEmbeddings))
    let chunk = embeddingDim
    let shiftMSA = emb[0..., (0 * chunk)..<(1 * chunk)]
    let scaleMSA = emb[0..., (1 * chunk)..<(2 * chunk)]
    let gateMSA = emb[0..., (2 * chunk)..<(3 * chunk)]
    let shiftMLP = emb[0..., (3 * chunk)..<(4 * chunk)]
    let scaleMLP = emb[0..., (4 * chunk)..<(5 * chunk)]
    let gateMLP = emb[0..., (5 * chunk)..<(6 * chunk)]

    // Manual layer norm in float32
    let normed = Self.layerNorm(hiddenStates, eps: eps)
    let modulated = normed * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
    return (modulated, gateMSA, shiftMLP, scaleMLP, gateMLP)
  }

  /// Float32 layer normalization matching the reference implementation.
  static func layerNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let xf = x.asType(.float32)
    let mean = MLX.mean(xf, axis: -1, keepDims: true)
    let variance = MLX.mean(MLX.pow(xf - mean, 2), axis: -1, keepDims: true)
    let normed = (xf - mean) / MLX.sqrt(variance + MLXArray(eps))
    return normed.asType(x.dtype)
  }
}

// MARK: - FiboAdaLayerNormZeroSingle (Single Blocks)

/// Adaptive layer norm for FIBO single transformer blocks.
///
/// Produces 3 modulation parameters: `(shift, scale, gate)`.
/// Uses standard `LayerNorm` (affine=false) and a linear projection.
///
/// This is the same `AdaLayerNormZeroSingle` from Flux, reused by FIBO.
///
/// Weight key path: `single_transformer_blocks.{i}.norm.linear.{weight,bias}`
final class FiboAdaLayerNormZeroSingle: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(dim: Int = 3072) {
    self._linear.wrappedValue = Linear(dim, 3 * dim)
    self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    super.init()
  }

  /// - Parameters:
  ///   - hiddenStates: Input tensor `[batch, seq, dim]`.
  ///   - textEmbeddings: DimFusion conditioning `[batch, dim]`.
  /// - Returns: `(normedModulated, gate)`.
  func callAsFunction(
    hiddenStates: MLXArray,
    textEmbeddings: MLXArray
  ) -> (MLXArray, MLXArray) {
    let emb = linear(silu(textEmbeddings))
    let chunkSize = emb.dim(-1) / 3
    let shiftMSA = emb[0..., (0 * chunkSize)..<(1 * chunkSize)]
    let scaleMSA = emb[0..., (1 * chunkSize)..<(2 * chunkSize)]
    let gateMSA = emb[0..., (2 * chunkSize)..<(3 * chunkSize)]
    let normed = norm(hiddenStates) * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
    return (normed, gateMSA)
  }
}

// MARK: - FiboAdaLayerNormContinuous (Output Norm)

/// Adaptive layer norm for the FIBO transformer output projection.
///
/// Produces 2 modulation parameters: `(scale, shift)`.
/// Applied as: `norm(x) * (1 + scale) + shift`
///
/// Weight key path: `norm_out.linear.{weight,bias}`
final class FiboAdaLayerNormContinuous: Module {
  let embeddingDim: Int

  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(embeddingDim: Int = 3072, conditioningEmbeddingDim: Int = 3072) {
    self.embeddingDim = embeddingDim
    self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2, bias: false)
    self._norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
    let mod = linear(silu(conditioning))
    let chunkSize = embeddingDim
    let scale = mod[0..., (0 * chunkSize)..<(1 * chunkSize)]
    let shift = mod[0..., (1 * chunkSize)..<(2 * chunkSize)]
    return norm(x) * (1 + scale).expandedDimensions(axis: 1) + shift.expandedDimensions(axis: 1)
  }
}
