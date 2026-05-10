import Foundation
import MLX
import MLXNN

/// 3D convolution with causal temporal padding for the Wan 2.2 VAE.
///
/// Applies asymmetric causal padding on the temporal dimension: `2 * padding`
/// frames prepended at the start, zero at the end. Spatial padding is symmetric.
///
/// ## Weight Layout
///
/// PyTorch stores weights as `(out_ch, in_ch, kT, kH, kW)`. After loading,
/// these are transposed to MLX channels-last format `(out_ch, kT, kH, kW, in_ch)`.
///
/// ## Padding Scheme
///
/// ```
/// Temporal: [2 * pad_t, 0]  (causal — only depends on past frames)
/// Height:   [pad_h, pad_h]  (symmetric)
/// Width:    [pad_w, pad_w]  (symmetric)
/// ```
public final class WanCausalConv3d: Module {

  /// 3D convolution layer (MLX channels-last).
  @ModuleInfo(key: "conv3d") var conv3d: Conv3d

  /// Padding amount (same for all spatial dims in the Wan VAE).
  public let padding: Int

  /// Convolution stride.
  public let stride: Int

  /// Convolution kernel size.
  public let kernelSize: Int

  /// Creates a Wan causal 3D convolution.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - kernelSize: Kernel size (isotropic). Default `3`.
  ///   - stride: Convolution stride. Default `1`.
  ///   - padding: Padding amount. Default `1`.
  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int = 3,
    stride: Int = 1,
    padding: Int = 1
  ) {
    self.padding = padding
    self.stride = stride
    self.kernelSize = kernelSize

    self._conv3d.wrappedValue = Conv3d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: IntOrTriple(kernelSize),
      stride: IntOrTriple(stride),
      padding: IntOrTriple(0)
    )

    super.init()
  }

  /// Applies the causal 3D convolution.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C_out, T_out, H_out, W_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x

    // Apply asymmetric causal padding
    let padT = padding
    let padH = padding
    let padW = padding

    if padT > 0 || padH > 0 || padW > 0 {
      // Temporal: 2*padT at start, 0 at end (causal)
      // Spatial: symmetric
      input = MLX.padded(
        input,
        widths: [
          IntOrPair((0, 0)),  // batch
          IntOrPair((0, 0)),  // channels
          IntOrPair((2 * padT, 0)),  // temporal (causal)
          IntOrPair((padH, padH)),  // height
          IntOrPair((padW, padW)),  // width
        ]
      )
    }

    // Transpose BCTHW -> BTHWC for MLX Conv3d
    input = input.transposed(0, 2, 3, 4, 1)

    // Apply convolution
    var output = conv3d(input)

    // Transpose BTHWC -> BCTHW
    output = output.transposed(0, 4, 1, 2, 3)

    return output
  }
}
