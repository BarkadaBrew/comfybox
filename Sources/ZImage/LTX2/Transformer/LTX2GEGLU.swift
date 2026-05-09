// LTX2GEGLU.swift — GELU-gated feed-forward network
// Phase 3 of the LTX-2 Swift/MLX port
//
// Feed-forward network with GELU (tanh approximation) activation.
// Note: Despite the "GEGLU" name in the task spec, the Python reference uses
// a simple GELU activation (not gated-GELU split). The proj_in produces a
// single 4x-expanded tensor, applies GELU, then proj_out projects back.
//
// Architecture:
//   Linear(dim, dim * 4) -> GELU(approx=tanh) -> Linear(dim * 4, dim)
//
// Reference: feed_forward.py class FeedForward

import MLX
import MLXNN

/// Feed-forward network with GELU (tanh approximation) activation.
///
/// Weight key mapping:
/// - `transformer_blocks.N.ff.proj_in.weight`, `.bias`
/// - `transformer_blocks.N.ff.proj_out.weight`, `.bias`
public final class LTX2FeedForward: Module {
  @ModuleInfo(key: "proj_in") var projIn: Linear
  @ModuleInfo(key: "proj_out") var projOut: Linear

  /// Initialize the feed-forward network.
  ///
  /// - Parameters:
  ///   - dim: Input and output dimension.
  ///   - dimOut: Output dimension (defaults to `dim` if nil).
  ///   - mult: Expansion multiplier for hidden dimension. Default 4.
  ///   - bias: Whether to use bias in linear layers. Default true.
  public init(
    dim: Int,
    dimOut: Int? = nil,
    mult: Int = 4,
    bias: Bool = true
  ) {
    let outputDim = dimOut ?? dim
    let innerDim = dim * mult
    self._projIn.wrappedValue = Linear(dim, innerDim, bias: bias)
    self._projOut.wrappedValue = Linear(innerDim, outputDim, bias: bias)
  }

  /// Forward pass: proj_in -> GELU(tanh) -> proj_out
  ///
  /// - Parameter x: Input tensor `(B, T, dim)`.
  /// - Returns: Output tensor `(B, T, dim)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = projIn(x)
    h = geluApproximate(h)
    h = projOut(h)
    return h
  }
}
