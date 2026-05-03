import Foundation
import MLX
import MLXNN

/// Adaptive modulation for Flux 2 transformer blocks.
///
/// Produces shift/scale/gate parameter sets from the timestep embedding.
/// Double-stream blocks use `modParamSets=2` (one set per norm),
/// single-stream blocks use `modParamSets=1`.
final class Flux2Modulation: Module {
  let modParamSets: Int

  @ModuleInfo(key: "linear") var linear: Linear

  init(dim: Int, modParamSets: Int = 2) {
    self.modParamSets = modParamSets
    self._linear.wrappedValue = Linear(dim, dim * 3 * modParamSets, bias: false)
    super.init()
  }

  /// Compute modulation parameters from timestep embedding.
  ///
  /// - Parameter temb: Timestep embedding of shape `[batch, dim]`.
  /// - Returns: Array of `modParamSets` tuples, each containing `(shift, scale, gate)`.
  func callAsFunction(_ temb: MLXArray) -> [[(MLXArray)]] {
    var mod = linear(silu(temb))
    if mod.ndim == 2 {
      mod = mod.expandedDimensions(axis: 1)
    }

    // Split into 3 * modParamSets chunks along last axis
    let totalChunks = 3 * modParamSets
    let chunkSize = mod.dim(-1) / totalChunks
    var allParams: [MLXArray] = []
    for c in 0..<totalChunks {
      let start = c * chunkSize
      let end = (c + 1) * chunkSize
      allParams.append(mod[0..., 0..., start..<end])
    }

    // Group into sets of 3: (shift, scale, gate)
    var result: [[(MLXArray)]] = []
    for i in 0..<modParamSets {
      let base = 3 * i
      result.append([allParams[base], allParams[base + 1], allParams[base + 2]])
    }
    return result
  }
}
