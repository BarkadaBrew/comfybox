import Foundation
import MLX
import MLXNN

/// Stride-2 convolution downsampling for the Flux2 VAE encoder.
///
/// Uses a 3x3 convolution with stride 2 to halve spatial dimensions.
/// All operations use NHWC layout.
///
/// ## Architecture
///
/// ```
/// Input (B, H, W, C)
///   -> pad (right and bottom by 1)
///   -> Conv2d(C -> C, k=3, s=2, p=0)
///   -> Output (B, H/2, W/2, C)
/// ```
///
/// Note: The Python source uses `padding=0` for the downsample conv, which
/// means asymmetric padding (pad-then-conv) is needed to match behavior.
public final class Flux2Downsample: Module {

  @ModuleInfo(key: "conv") var conv: Conv2d

  /// Creates a Flux2 stride-2 downsampling layer.
  ///
  /// - Parameter channels: Number of input/output channels.
  public init(channels: Int) {
    // Python uses padding=0 on the conv, with asymmetric pad before
    self._conv.wrappedValue = Conv2d(
      inputChannels: channels, outputChannels: channels,
      kernelSize: 3, stride: 2)
    super.init()
  }

  /// Applies stride-2 downsampling.
  ///
  /// - Parameter x: Input tensor in NHWC layout `(B, H, W, C)`.
  /// - Returns: Output tensor in NHWC layout `(B, H/2, W/2, C)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Pad right and bottom by 1 (NHWC: pad H and W dims)
    let padded = padded(x, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
    return conv(padded)
  }
}
