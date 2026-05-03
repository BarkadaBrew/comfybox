// SmolLM3RoPE.swift — Rotary position embeddings for SmolLM3-3B text encoder
// Ported from mflux: smol_lm3_3b_rope.py

import MLX
import MLXNN

/// Rotary position embeddings for SmolLM3-3B.
///
/// Uses a very high `theta` of 5,000,000 (compared to standard 10,000)
/// for extended context support. Pre-computes inverse frequencies and
/// builds cos/sin tables on demand for a given sequence length.
///
/// Output shape: `(cos, sin)` each `[1, 1, seqLen, headDim]` — ready for
/// broadcasting across batch and head dimensions in attention.
public final class SmolLM3RoPE: Module {
  let dim: Int
  let maxPositionEmbeddings: Int
  let base: Float
  let invFreq: MLXArray

  public init(
    dim: Int,
    maxPositionEmbeddings: Int = 65_536,
    base: Float = 5_000_000.0
  ) {
    self.dim = dim
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.base = base

    // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
    self.invFreq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))
  }

  /// Compute cos/sin tables for rotary embeddings.
  ///
  /// - Parameter seqLen: Sequence length to compute embeddings for.
  /// - Returns: `(cos, sin)` each of shape `[1, 1, seqLen, headDim]`.
  public func callAsFunction(_ seqLen: Int) -> (cos: MLXArray, sin: MLXArray) {
    let (cos, sin) = SmolLM3RoPE.buildCosSin(invFreq: invFreq, seqLen: seqLen)
    // Expand to [1, 1, seqLen, headDim] for broadcasting
    let cosExpanded = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    let sinExpanded = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    return (cosExpanded, sinExpanded)
  }

  /// Build raw cos/sin tables from inverse frequencies.
  ///
  /// - Parameters:
  ///   - invFreq: Precomputed inverse frequency vector `[dim/2]`.
  ///   - seqLen: Sequence length.
  /// - Returns: `(cos, sin)` each of shape `[seqLen, headDim]`.
  static func buildCosSin(invFreq: MLXArray, seqLen: Int) -> (MLXArray, MLXArray) {
    let positions = MLXArray(0..<seqLen).asType(.float32)
    // freqs: [seqLen, dim/2]
    let freqs = MLX.outer(positions, invFreq)
    // emb: [seqLen, dim] (concatenate freqs with itself for full head_dim)
    let emb = MLX.concatenated([freqs, freqs], axis: -1)
    return (MLX.cos(emb), MLX.sin(emb))
  }
}
