import Foundation
import MLX
import MLXNN

/// SwiGLU activation for Flux 2.
///
/// Splits input in half along the last axis, applies SiLU to the first half,
/// and element-wise multiplies with the second half.
final class Flux2SwiGLU: Module {
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let chunks = MLX.split(x, parts: 2, axis: -1)
    return silu(chunks[0]) * chunks[1]
  }
}

/// Feed-forward network (SwiGLU) for Flux 2 double-stream transformer blocks.
///
/// Projects `dim` -> `innerDim * 2` (for SwiGLU gate+value), applies SwiGLU,
/// then projects `innerDim` -> `dim`.
final class Flux2FeedForward: Module {
  @ModuleInfo(key: "linear_in") var linearIn: Linear
  @ModuleInfo(key: "act") var act: Flux2SwiGLU
  @ModuleInfo(key: "linear_out") var linearOut: Linear

  init(dim: Int, mult: Float = 3.0) {
    let innerDim = Int(Float(dim) * mult)
    self._linearIn.wrappedValue = Linear(dim, innerDim * 2, bias: false)
    self._act.wrappedValue = Flux2SwiGLU()
    self._linearOut.wrappedValue = Linear(innerDim, dim, bias: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    linearOut(act(linearIn(x)))
  }
}
