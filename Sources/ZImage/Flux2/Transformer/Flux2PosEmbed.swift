import Foundation
import MLX

/// Rotary position embeddings for Flux 2 transformer.
///
/// Unlike Flux 1's complex-valued RoPE table, Flux 2 uses explicit cos/sin pairs
/// computed per-axis from position IDs, then concatenated across all axes.
///
/// Supports DyPE (Dynamic Positional Encoding) for high-resolution generation
/// beyond the 1024px training resolution. When scales are provided, axes 1 and 2
/// (height, width) use NTK-aware interpolation while axes 0 and 3 (time, layer)
/// remain vanilla.
final class Flux2PosEmbed {
  let theta: Int
  let axesDim: [Int]

  /// DyPE configuration. Set before calling with scales to enable NTK scaling.
  var dyPE: DyPEConfig = .disabled

  init(theta: Int = 2000, axesDim: [Int] = [32, 32, 32, 32]) {
    self.theta = theta
    self.axesDim = axesDim
  }

  /// Compute rotary embeddings for the given position IDs.
  ///
  /// - Parameters:
  ///   - ids: Position IDs of shape `[seqLen, numAxes]`.
  ///   - scales: Optional per-axis scale factors. When provided and DyPE is enabled,
  ///     axes with scale > 1.0 use NTK-scaled theta. Expected order: [t, h, w, layer].
  /// - Returns: Tuple of `(cos, sin)` each of shape `[seqLen, totalHalfDim]`.
  func callAsFunction(_ ids: MLXArray, scales: [Float]? = nil) -> (MLXArray, MLXArray) {
    let pos = ids.asType(.float32)
    var cosOut: [MLXArray] = []
    var sinOut: [MLXArray] = []

    for (i, dim) in axesDim.enumerated() {
      let scale = (dyPE.enabled && scales != nil) ? scales![i] : 1.0
      let (cos, sin) = get1dRope(dim: dim, pos: pos[0..., i], scale: scale)
      cosOut.append(cos)
      sinOut.append(sin)
    }

    let freqsCos = MLX.concatenated(cosOut, axis: -1)
    let freqsSin = MLX.concatenated(sinOut, axis: -1)
    return (freqsCos, freqsSin)
  }

  /// Compute 1D RoPE frequencies for a single axis.
  ///
  /// When scale > 1.0, applies NTK-aware interpolation:
  /// `theta' = theta * scale^(dim / (dim - 2))`
  private func get1dRope(dim: Int, pos: MLXArray, scale: Float = 1.0) -> (MLXArray, MLXArray) {
    // NTK-aware theta scaling for high-resolution generation
    let effectiveTheta: Float
    if scale > 1.0 {
      let d = Float(dim)
      effectiveTheta = Float(theta) * pow(scale, d / (d - 2.0))
    } else {
      effectiveTheta = Float(theta)
    }

    // scale = arange(0, dim, 2) / dim
    let freqScale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
    // omega = 1.0 / (theta ^ scale)
    let omega = 1.0 / MLX.pow(MLXArray(effectiveTheta), freqScale)

    // [seqLen, 1] * [1, halfDim] -> [seqLen, halfDim]
    let posExpanded = pos.expandedDimensions(axis: -1)
    let omegaExpanded = omega.expandedDimensions(axis: 0)
    let out = posExpanded * omegaExpanded

    return (MLX.cos(out), MLX.sin(out))
  }
}
