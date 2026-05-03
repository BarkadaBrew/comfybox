// FiboTimestepEmbedding.swift — Timestep embeddings for FIBO transformer
// Ported from mflux: time_embed.py (BriaFiboTimestepProjEmbeddings) +
//                    bria_fibo_timesteps.py (BriaFiboTimesteps) +
//                    timestep_embedder.py (TimestepEmbedder)
//
// FIBO uses a two-stage timestep embedding:
// 1. BriaFiboTimesteps: sinusoidal frequency embedding (256 channels, flip_sin_to_cos)
// 2. TimestepEmbedder: 2-layer MLP (256 → 3072 → 3072) with SiLU activation
//
// Unlike Flux 2, FIBO has NO guidance embeddings (guidance_embeds=false in config).

import Foundation
import MLX
import MLXNN

/// Sinusoidal timestep frequency embedding.
///
/// Computes `[sin, cos]` frequency features from scalar timestep values.
/// Uses `flip_sin_to_cos=true` convention (cos before sin in output).
///
/// Weight key path: none (no learnable parameters)
struct FiboTimesteps {
  let numChannels: Int
  let flipSinToCos: Bool
  let downscaleFreqShift: Float
  let scale: Float
  let timeTheta: Int

  init(
    numChannels: Int = 256,
    flipSinToCos: Bool = true,
    downscaleFreqShift: Float = 0,
    scale: Float = 1.0,
    timeTheta: Int = 10000
  ) {
    self.numChannels = numChannels
    self.flipSinToCos = flipSinToCos
    self.downscaleFreqShift = downscaleFreqShift
    self.scale = scale
    self.timeTheta = timeTheta
  }

  func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
    let halfDim = numChannels / 2
    let logTheta = -Foundation.log(Float(timeTheta))
    var exponent = MLXArray(stride(from: 0, to: halfDim, by: 1).map { Float($0) })
    exponent = MLXArray(logTheta) * exponent
    exponent = exponent / MLXArray(Float(halfDim) - downscaleFreqShift)
    var emb = MLX.exp(exponent)
    // [batch, 1] * [1, halfDim] -> [batch, halfDim]
    emb = timesteps.asType(.float32).expandedDimensions(axis: -1) * emb.expandedDimensions(axis: 0)
    emb = MLXArray(scale) * emb
    let sinEmb = MLX.sin(emb)
    let cosEmb = MLX.cos(emb)
    var result = MLX.concatenated([sinEmb, cosEmb], axis: -1)
    if flipSinToCos {
      // Swap: put cos before sin
      let cosHalf = result[0..., halfDim...]
      let sinHalf = result[0..., ..<halfDim]
      result = MLX.concatenated([cosHalf, sinHalf], axis: -1)
    }
    if numChannels % 2 == 1 {
      let pad = MLX.zeros([result.dim(0), 1], dtype: result.dtype)
      result = MLX.concatenated([result, pad], axis: -1)
    }
    return result
  }
}

/// Timestep MLP embedder (256 → 3072 → 3072 with SiLU).
///
/// Weight key path: `time_embed.timestep_embedder.linear_{1,2}.{weight,bias}`
final class FiboTimestepEmbedder: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  init(inDim: Int = 256, outDim: Int = 3072) {
    self._linear1.wrappedValue = Linear(inDim, outDim)
    self._linear2.wrappedValue = Linear(outDim, outDim)
    super.init()
  }

  func callAsFunction(_ sample: MLXArray) -> MLXArray {
    linear2(silu(linear1(sample)))
  }
}

/// Combined timestep projection + MLP embedding for FIBO.
///
/// Two-stage process:
/// 1. Sinusoidal frequency embedding (256 channels)
/// 2. MLP projection (256 → 3072 → 3072)
///
/// Weight key path: `time_embed.timestep_embedder.*`
final class FiboTimestepProjEmbedding: Module {
  let timeProj: FiboTimesteps

  @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: FiboTimestepEmbedder

  init(embeddingDim: Int = 3072, timeTheta: Int = 10000) {
    self.timeProj = FiboTimesteps(
      numChannels: 256,
      flipSinToCos: true,
      downscaleFreqShift: 0,
      timeTheta: timeTheta
    )
    self._timestepEmbedder.wrappedValue = FiboTimestepEmbedder(inDim: 256, outDim: embeddingDim)
    super.init()
  }

  func callAsFunction(timestep: MLXArray, dtype: DType) -> MLXArray {
    let timestepsProj = timeProj(timestep)
    return timestepEmbedder(timestepsProj.asType(dtype))
  }
}
