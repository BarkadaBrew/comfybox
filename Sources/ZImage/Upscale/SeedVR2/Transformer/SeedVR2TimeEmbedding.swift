import Foundation
import MLX
import MLXNN

/// Sinusoidal timestep embedding followed by a three-layer MLP for the SeedVR2 transformer.
///
/// ## Architecture
///
/// ```
/// timestep (scalar or [B])
///   → sinusoidal embedding (B, sinusoidal_dim=256)
///   → Linear(256, 2560) → SiLU
///   → Linear(2560, 2560) → SiLU
///   → Linear(2560, 15360)
///   → output (B, 15360)
/// ```
///
/// The output is later reshaped to `(B, vid_dim, 2, 3)` in the top-level transformer,
/// where the `2` axis separates attention vs MLP modulation parameters, and the `3` axis
/// holds (shift, scale, gate) for each.
///
/// ## Weight Key Paths
///
/// - `proj_in.weight`, `proj_in.bias`
/// - `proj_hid.weight`, `proj_hid.bias`
/// - `proj_out.weight`, `proj_out.bias`
public final class SeedVR2TimeEmbedding: Module {

  /// Dimension of the sinusoidal positional embedding.
  public let sinusoidalDim: Int

  /// First linear: sinusoidal_dim -> hidden_dim.
  @ModuleInfo(key: "proj_in") var projIn: Linear

  /// Second linear: hidden_dim -> hidden_dim.
  @ModuleInfo(key: "proj_hid") var projHid: Linear

  /// Third linear: hidden_dim -> output_dim.
  @ModuleInfo(key: "proj_out") var projOut: Linear

  /// Creates a timestep embedding module.
  ///
  /// - Parameters:
  ///   - sinusoidalDim: Dimension of the sinusoidal embedding. Default `256`.
  ///   - hiddenDim: Hidden dimension of the MLP. Default `2560`.
  ///   - outputDim: Output dimension. Default `15360` (= 6 * 2560).
  public init(sinusoidalDim: Int = 256, hiddenDim: Int = 2560, outputDim: Int = 15360) {
    self.sinusoidalDim = sinusoidalDim
    self._projIn.wrappedValue = Linear(sinusoidalDim, hiddenDim)
    self._projHid.wrappedValue = Linear(hiddenDim, hiddenDim)
    self._projOut.wrappedValue = Linear(hiddenDim, outputDim)
    super.init()
  }

  /// Computes the timestep embedding.
  ///
  /// - Parameter timestep: Scalar or 1D array of timestep values.
  /// - Returns: Embedding array of shape `(B, outputDim)`.
  public func callAsFunction(_ timestep: MLXArray) -> MLXArray {
    // Ensure at least 1D
    var t = timestep
    if t.ndim == 0 {
      t = t.expandedDimensions(axis: 0)
    }

    // Sinusoidal embedding
    var emb = SeedVR2TimeEmbedding.sinusoidalEmbedding(timesteps: t, dim: sinusoidalDim)

    // MLP: Linear -> SiLU -> Linear -> SiLU -> Linear
    emb = projIn(emb)
    emb = MLXNN.silu(emb)
    emb = projHid(emb)
    emb = MLXNN.silu(emb)
    emb = projOut(emb)

    return emb
  }

  /// Creates sinusoidal positional embeddings from timestep values.
  ///
  /// Matches the Python `_get_timestep_embedding` implementation:
  ///
  ///     half_dim = dim // 2
  ///     freqs = exp(arange(half_dim) * (-log(10000) / half_dim))
  ///     args = timesteps[:, None] * freqs
  ///     return concat([sin(args), cos(args)], axis=-1)
  ///
  /// - Parameters:
  ///   - timesteps: 1D array of timestep values, shape `(B,)`.
  ///   - dim: Embedding dimension.
  /// - Returns: Sinusoidal embeddings of shape `(B, dim)`.
  public static func sinusoidalEmbedding(timesteps: MLXArray, dim: Int) -> MLXArray {
    let halfDim = dim / 2
    // freqs = exp(arange(halfDim, float32) * (-log(10000) / halfDim))
    let indices = MLXArray(Int32(0) ..< Int32(halfDim)).asType(.float32)
    let freqs = (indices * Float(-Foundation.log(10000.0) / Float(halfDim))).exp()

    // args = timesteps[:, None] * freqs[None, :]
    let args = timesteps.expandedDimensions(axis: 1).asType(.float32) * freqs

    // concat([sin, cos], axis=-1)
    return MLX.concatenated([args.sin(), args.cos()], axis: -1)
  }
}
