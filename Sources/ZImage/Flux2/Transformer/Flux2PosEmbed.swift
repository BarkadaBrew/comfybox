import Foundation
import MLX

/// Rotary position embeddings for Flux 2 transformer.
///
/// Unlike Flux 1's complex-valued RoPE table, Flux 2 uses explicit cos/sin pairs
/// computed per-axis from position IDs, then concatenated across all axes.
final class Flux2PosEmbed {
  let theta: Int
  let axesDim: [Int]

  init(theta: Int = 2000, axesDim: [Int] = [32, 32, 32, 32]) {
    self.theta = theta
    self.axesDim = axesDim
  }

  /// Compute rotary embeddings for the given position IDs.
  ///
  /// - Parameter ids: Position IDs of shape `[seqLen, numAxes]`.
  /// - Returns: Tuple of `(cos, sin)` each of shape `[seqLen, totalHalfDim]`.
  func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
    let pos = ids.asType(.float32)
    var cosOut: [MLXArray] = []
    var sinOut: [MLXArray] = []

    for (i, dim) in axesDim.enumerated() {
      let (cos, sin) = get1dRope(dim: dim, pos: pos[0..., i])
      cosOut.append(cos)
      sinOut.append(sin)
    }

    let freqsCos = MLX.concatenated(cosOut, axis: -1)
    let freqsSin = MLX.concatenated(sinOut, axis: -1)
    return (freqsCos, freqsSin)
  }

  /// Compute 1D RoPE frequencies for a single axis.
  private func get1dRope(dim: Int, pos: MLXArray) -> (MLXArray, MLXArray) {
    // scale = arange(0, dim, 2) / dim
    let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
    // omega = 1.0 / (theta ^ scale)
    let omega = 1.0 / MLX.pow(MLXArray(Float(theta)), scale)

    // [seqLen, 1] * [1, halfDim] -> [seqLen, halfDim]
    let posExpanded = pos.expandedDimensions(axis: -1)
    let omegaExpanded = omega.expandedDimensions(axis: 0)
    let out = posExpanded * omegaExpanded

    return (MLX.cos(out), MLX.sin(out))
  }
}
