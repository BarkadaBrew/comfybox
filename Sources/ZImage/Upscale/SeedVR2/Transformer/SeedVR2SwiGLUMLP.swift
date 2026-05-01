import Foundation
import MLX
import MLXNN

/// Unified feed-forward network for the SeedVR2 transformer.
///
/// Supports both SwiGLU (3B) and standard GELU (7B) modes:
///
/// SwiGLU mode (3B):
///     gate  = SiLU(proj_in_gate(x))
///     value = proj_in(x)
///     out   = proj_out(gate * value)
///     Hidden dim = multiple_of * ceil(2 * dim * expand_ratio / 3 / multiple_of)
///     No bias on any projection.
///
/// GELU mode (7B):
///     out = proj_out(GELU(proj_in(x)))
///     Hidden dim = dim * expand_ratio
///     Bias on both projections.
///
/// All projections are registered via ModuleInfo so weight keys map correctly.
public final class SeedVR2SwiGLUMLP: Module {

  @ModuleInfo(key: "proj_in") var projIn: Linear
  @ModuleInfo(key: "proj_in_gate") var projInGate: Linear?
  @ModuleInfo(key: "proj_out") var projOut: Linear

  public let dim: Int
  public let hiddenDim: Int
  public let mlpType: SeedVR2MLPType

  /// Creates a feed-forward layer.
  ///
  /// - Parameters:
  ///   - dim: Input and output feature dimension.
  ///   - expandRatio: Expansion factor for the hidden dimension. Default 4.
  ///   - multipleOf: Alignment for SwiGLU hidden dim. Default 256.
  ///   - mlpType: .swiglu for 3B, .gelu for 7B. Default .swiglu.
  public init(dim: Int, expandRatio: Int = 4, multipleOf: Int = 256, mlpType: SeedVR2MLPType = .swiglu) {
    self.dim = dim
    self.mlpType = mlpType

    switch mlpType {
    case .swiglu:
      let raw = 2 * dim * expandRatio / 3
      let aligned = multipleOf * ((raw + multipleOf - 1) / multipleOf)
      self.hiddenDim = aligned
      self._projIn.wrappedValue = Linear(dim, aligned, bias: false)
      self._projInGate.wrappedValue = Linear(dim, aligned, bias: false)
      self._projOut.wrappedValue = Linear(aligned, dim, bias: false)

    case .gelu:
      self.hiddenDim = dim * expandRatio
      self._projIn.wrappedValue = Linear(dim, dim * expandRatio, bias: true)
      self._projInGate.wrappedValue = nil
      self._projOut.wrappedValue = Linear(dim * expandRatio, dim, bias: true)
    }

    super.init()
  }

  /// Applies the feed-forward to the input.
  ///
  /// - Parameter x: Input array of shape (..., dim).
  /// - Returns: Output array of shape (..., dim).
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    switch mlpType {
    case .swiglu:
      let gate = MLXNN.silu(projInGate!(x))
      let value = projIn(x)
      return projOut(gate * value)
    case .gelu:
      var h = projIn(x)
      h = MLXNN.gelu(h)
      return projOut(h)
    }
  }
}
