import Foundation
import MLX
import MLXNN

/// Spatial (and optionally temporal) downsampling for the SeedVR2 3D VAE encoder.
///
/// Halves the spatial dimensions using a strided ``CausalConv3d``. When temporal
/// downsampling is enabled, the temporal dimension is also halved.
///
/// ## Padding
///
/// The Python reference applies asymmetric padding of `(0, 1)` on both H and W
/// before the convolution, rather than using the convolutions own padding parameter.
/// This ensures correct output dimensions when the spatial size is odd.
///
/// ## Modes
///
/// - **Spatial only** (`spatialOnly = true`): kernel `(1, 3, 3)`, stride `(1, 2, 2)`.
///   Temporal dimension is unchanged.
/// - **Spatial + temporal** (`spatialOnly = false`): kernel `(3, 3, 3)`, stride `(2, 2, 2)`.
///   All three dimensions are halved.
public final class SeedVR2Downsample3D: Module {

  /// Strided causal 3D convolution that performs the downsampling.
  @ModuleInfo(key: "conv") var conv: CausalConv3d

  /// Creates a downsampling module.
  ///
  /// - Parameters:
  ///   - channels: Number of input/output channels (preserved).
  ///   - spatialOnly: If true, only spatial dimensions are halved. If false,
  ///     temporal dimension is also halved. Default `false`.
  public init(channels: Int, spatialOnly: Bool = false) {
    let kt = spatialOnly ? 1 : 3
    let st = spatialOnly ? 1 : 2
    let pt = spatialOnly ? 0 : 1

    self._conv.wrappedValue = CausalConv3d(
      inChannels: channels, outChannels: channels,
      kernelSize: (kt, 3, 3),
      stride: (st, 2, 2),
      padding: (pt, 0, 0)
    )

    super.init()
  }

  /// Applies the downsampling.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Downsampled tensor. Spatial dimensions are halved (ceiling division).
  ///   Temporal dimension is halved only when `spatialOnly` was false at init.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Asymmetric pad: add 1 to the right of H and W dimensions.
    // Input layout: (B, C, T, H, W) — H is axis 3, W is axis 4.
    let padded = MLX.padded(
      x,
      widths: [
        IntOrPair((0, 0)),  // B
        IntOrPair((0, 0)),  // C
        IntOrPair((0, 0)),  // T
        IntOrPair((0, 1)),  // H: pad right by 1
        IntOrPair((0, 1)),  // W: pad right by 1
      ]
    )
    return conv(padded)
  }
}
