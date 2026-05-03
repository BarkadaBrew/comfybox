import Foundation
import MLX
import MLXNN

/// 2x nearest-neighbor upsampling followed by a 3x3 convolution for the Flux2 VAE decoder.
///
/// All operations use NHWC layout.
///
/// ## Architecture
///
/// ```
/// Input (B, H, W, C)
///   -> nearest-neighbor 2x upsample -> (B, 2H, 2W, C)
///   -> Conv2d(C -> C_out, k=3, p=1)
///   -> Output (B, 2H, 2W, C_out)
/// ```
public final class Flux2Upsample: Module {

  @ModuleInfo(key: "conv") var conv: Conv2d

  /// Creates a Flux2 2x upsampling layer.
  ///
  /// - Parameters:
  ///   - channels: Number of input channels.
  ///   - outChannels: Number of output channels. If `nil`, uses `channels`.
  public init(channels: Int, outChannels: Int? = nil) {
    let out = outChannels ?? channels
    self._conv.wrappedValue = Conv2d(
      inputChannels: channels, outputChannels: out,
      kernelSize: 3, stride: 1, padding: 1)
    super.init()
  }

  /// Applies 2x nearest-neighbor upsampling and convolution.
  ///
  /// - Parameter x: Input tensor in NHWC layout `(B, H, W, C)`.
  /// - Returns: Output tensor in NHWC layout `(B, 2H, 2W, C_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Nearest-neighbor 2x upsample using MLXNN.Upsample
    let upsampled = Upsample(scaleFactor: .array([2.0, 2.0]), mode: .nearest)(x)
    return conv(upsampled)
  }
}
