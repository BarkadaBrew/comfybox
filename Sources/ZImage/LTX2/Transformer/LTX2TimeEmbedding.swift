// LTX2TimeEmbedding.swift — Sinusoidal timestep embedding for AdaLayerNorm
// Phase 3 of the LTX-2 Swift/MLX port
//
// Standard sinusoidal timestep embedding: [cos(t*f0), sin(t*f0), cos(t*f1), ...]
// Used by AdaLayerNormSingle to condition the transformer blocks on the
// denoising timestep.
//
// Pipeline:
//   timestep (scalar) -> sinusoidal(256) -> Linear(256, inner_dim) -> SiLU -> Linear(inner_dim, inner_dim)
//
// Reference: adaln.py classes Timesteps, TimestepEmbedding, PixArtAlphaCombinedTimestepSizeEmbeddings

import Foundation
import MLX
import MLXNN

// MARK: - Timesteps (Sinusoidal Embedding)

/// Sinusoidal timestep embedding.
///
/// Computes: freq_i = exp(-ln(max_period) * i / (half_dim - downscale_freq_shift))
/// Then: [sin(t * freq), cos(t * freq)] or [cos, sin] if flipped.
///
/// Weight key mapping: None (no learnable parameters).
public final class LTX2Timesteps: Module {
  let numChannels: Int
  let flipSinToCos: Bool
  let downscaleFreqShift: Float

  public init(
    numChannels: Int = 256,
    flipSinToCos: Bool = true,
    downscaleFreqShift: Float = 0
  ) {
    self.numChannels = numChannels
    self.flipSinToCos = flipSinToCos
    self.downscaleFreqShift = downscaleFreqShift
  }

  /// Compute sinusoidal embedding from timestep values.
  ///
  /// - Parameter timesteps: 1D array of timestep values `(B,)`.
  /// - Returns: Sinusoidal embeddings `(B, numChannels)`.
  public func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
    let halfDim = numChannels / 2
    let maxPeriod: Float = 10000.0

    // freqs = exp(-log(max_period) * arange(half_dim) / (half_dim - downscale_freq_shift))
    let indices = MLXArray(Int32(0) ..< Int32(halfDim)).asType(.float32)
    let exponent = indices * (-Foundation.log(maxPeriod) / (Float(halfDim) - downscaleFreqShift))
    let freqs = exponent.exp()

    // args = timesteps[:, None] * freqs[None, :]
    let args = timesteps.expandedDimensions(axis: 1).asType(.float32) * freqs

    // Concatenate sin and cos
    let emb: MLXArray
    if flipSinToCos {
      emb = MLX.concatenated([args.cos(), args.sin()], axis: -1)
    } else {
      emb = MLX.concatenated([args.sin(), args.cos()], axis: -1)
    }

    return emb
  }
}

// MARK: - TimestepEmbedding (MLP)

/// Two-layer MLP for projecting sinusoidal embeddings to the model dimension.
///
/// Architecture: Linear -> SiLU -> Linear
///
/// Weight key mapping:
/// - `adaln_single.emb.timestep_embedder.linear1.weight`, `.bias`
/// - `adaln_single.emb.timestep_embedder.linear2.weight`, `.bias`
public final class LTX2TimestepEmbedding: Module {
  @ModuleInfo(key: "linear1") var linear1: Linear
  @ModuleInfo(key: "linear2") var linear2: Linear

  public init(
    inChannels: Int,
    timeEmbedDim: Int,
    outDim: Int? = nil
  ) {
    let outputDim = outDim ?? timeEmbedDim
    self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
    self._linear2.wrappedValue = Linear(timeEmbedDim, outputDim)
  }

  public func callAsFunction(_ sample: MLXArray) -> MLXArray {
    var h = linear1(sample)
    h = silu(h)
    h = linear2(h)
    return h
  }
}

// MARK: - Combined Timestep+Size Embeddings

/// Combined timestep embedding used by AdaLayerNormSingle.
///
/// Chains: Timesteps (sinusoidal) -> TimestepEmbedding (MLP)
///
/// Weight key mapping:
/// - `adaln_single.emb.time_proj.*` (Timesteps has no weights)
/// - `adaln_single.emb.timestep_embedder.linear1.*`
/// - `adaln_single.emb.timestep_embedder.linear2.*`
public final class LTX2CombinedTimestepEmbeddings: Module {
  @ModuleInfo(key: "time_proj") var timeProj: LTX2Timesteps
  @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: LTX2TimestepEmbedding

  public init(
    embeddingDim: Int,
    timestepProjDim: Int = 256
  ) {
    self._timeProj.wrappedValue = LTX2Timesteps(
      numChannels: timestepProjDim,
      flipSinToCos: true,
      downscaleFreqShift: 0
    )
    self._timestepEmbedder.wrappedValue = LTX2TimestepEmbedding(
      inChannels: timestepProjDim,
      timeEmbedDim: embeddingDim,
      outDim: embeddingDim
    )
  }

  /// Compute timestep embedding.
  ///
  /// - Parameters:
  ///   - timestep: 1D timestep values `(B,)`.
  ///   - hiddenDtype: Target dtype for the projected embedding.
  /// - Returns: Timestep embedding `(B, embeddingDim)`.
  public func callAsFunction(
    _ timestep: MLXArray,
    hiddenDtype: DType? = nil
  ) -> MLXArray {
    var proj = timeProj(timestep)
    if let dtype = hiddenDtype {
      proj = proj.asType(dtype)
    }
    return timestepEmbedder(proj)
  }
}
