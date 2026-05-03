// Qwen3RoPE.swift — Rotary position embeddings for Qwen3 text encoder
// Ported from mflux: qwen3_text_rotary_embedding.py

import MLX
import MLXNN

/// Rotary position embeddings for the Qwen3 text encoder.
///
/// Unlike the existing Flux 1 `RoPE` (which uses MLXFast), this implementation
/// matches the mflux Qwen3TextRotaryEmbedding exactly: pre-computed inverse
/// frequencies scaled by `scalingFactor`, returning `(cos, sin)` tuples that
/// the attention layer applies via rotate-half.
public final class Qwen3RoPE: Module {
  let dim: Int
  let maxPositionEmbeddings: Int
  let base: Float
  let scalingFactor: Float
  let invFreq: MLXArray

  public init(
    dim: Int,
    maxPositionEmbeddings: Int = 40_960,
    base: Float = 1_000_000.0,
    scalingFactor: Float = 1.0
  ) {
    self.dim = dim
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.base = base
    self.scalingFactor = scalingFactor

    // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
    self.invFreq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))
  }

  /// Compute rotary embeddings for the given hidden states and position IDs.
  ///
  /// - Parameters:
  ///   - x: Hidden states, used only for dtype casting. Shape: `[B, S, H]`.
  ///   - positionIds: Integer position IDs. Shape: `[B, S]` or `[S]`.
  /// - Returns: `(cos, sin)` each of shape `[B, S, dim]`, cast to `x.dtype`.
  public func callAsFunction(
    _ x: MLXArray,
    positionIds: MLXArray
  ) -> (cos: MLXArray, sin: MLXArray) {
    var ids = positionIds
    if ids.ndim == 1 {
      ids = ids.expandedDimensions(axis: 0)
    }

    // inv_freq: [dim/2] -> [1, 1, dim/2]
    let invFreqExpanded = invFreq
      .expandedDimensions(axis: 0)
      .expandedDimensions(axis: 0)

    // pos: [B, S] -> [B, S, 1]
    let pos = ids.asType(.float32).expandedDimensions(axis: -1)

    // freqs: [B, S, dim/2]
    let freqs = pos * invFreqExpanded

    // emb: [B, S, dim] (concatenate freqs with itself)
    let emb = MLX.concatenated([freqs, freqs], axis: -1)

    let cosValues = MLX.cos(emb) * scalingFactor
    let sinValues = MLX.sin(emb) * scalingFactor

    return (cosValues.asType(x.dtype), sinValues.asType(x.dtype))
  }
}
