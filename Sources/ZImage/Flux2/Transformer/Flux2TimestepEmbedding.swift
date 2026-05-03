import Foundation
import MLX
import MLXNN

/// Timestep and optional guidance embeddings for Flux 2.
///
/// Computes sinusoidal frequency embeddings from scalar timestep values,
/// projects through a 2-layer MLP (with SiLU activation), and optionally
/// adds a guidance embedding computed through a parallel MLP.
final class Flux2TimestepEmbedding: Module {
  let inChannels: Int
  let embeddingDim: Int
  let guidanceEmbeds: Bool

  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear
  @ModuleInfo(key: "guidance_linear_1") var guidanceLinear1: Linear?
  @ModuleInfo(key: "guidance_linear_2") var guidanceLinear2: Linear?

  init(inChannels: Int = 256, embeddingDim: Int = 6144, guidanceEmbeds: Bool = true) {
    self.inChannels = inChannels
    self.embeddingDim = embeddingDim
    self.guidanceEmbeds = guidanceEmbeds

    self._linear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
    self._linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)

    if guidanceEmbeds {
      self._guidanceLinear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
      self._guidanceLinear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
    }

    super.init()
  }

  func callAsFunction(timestep: MLXArray, guidance: MLXArray?) -> MLXArray {
    let timestepEmb = Self.timestepEmbedding(timestep.asType(.float32), dim: inChannels)
    let tEmb = linear2(silu(linear1(timestepEmb)))

    if let guidance, let gLinear1 = guidanceLinear1, let gLinear2 = guidanceLinear2 {
      let guidanceEmb = Self.timestepEmbedding(guidance.asType(.float32), dim: inChannels)
      let gEmb = gLinear2(silu(gLinear1(guidanceEmb)))
      return tEmb + gEmb
    }

    return tEmb
  }

  /// Compute sinusoidal timestep embeddings.
  ///
  /// Matches the diffusers convention: `flip_sin_to_cos=True` puts cos before sin.
  static func timestepEmbedding(
    _ timesteps: MLXArray,
    dim: Int,
    flipSinToCos: Bool = true
  ) -> MLXArray {
    let half = dim / 2
    let freqs = MLX.exp(
      MLXArray(-Foundation.log(10000.0))
        * MLXArray(stride(from: 0, to: half, by: 1).map { Float($0) })
        / MLXArray(Float(half))
    )
    let args = timesteps.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
    var emb = MLX.concatenated([MLX.sin(args), MLX.cos(args)], axis: -1)

    if flipSinToCos {
      // Swap sin and cos halves
      let cosHalf = emb[0..., half...]
      let sinHalf = emb[0..., ..<half]
      emb = MLX.concatenated([cosHalf, sinHalf], axis: -1)
    }

    if dim % 2 == 1 {
      let pad = MLX.zeros([emb.dim(0), 1], dtype: emb.dtype)
      emb = MLX.concatenated([emb, pad], axis: -1)
    }

    return emb
  }
}
