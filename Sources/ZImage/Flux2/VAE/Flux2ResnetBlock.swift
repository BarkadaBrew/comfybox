import Foundation
import MLX
import MLXNN

/// ResNet block for the Flux2 VAE encoder and decoder.
///
/// Two group-norm + SiLU + conv layers with a residual connection.
/// When input and output channel counts differ, a 1x1 shortcut convolution
/// adapts the residual path.
///
/// All operations use NHWC layout (the native layout for MLX Conv2d / GroupNorm).
///
/// ## Architecture
///
/// ```
/// Input (B, H, W, C_in)
///   |-- GroupNorm(32, C_in) -> SiLU -> Conv2d(C_in -> C_out, k=3, p=1)
///   |-- GroupNorm(32, C_out) -> SiLU -> Conv2d(C_out -> C_out, k=3, p=1)
///   |-- + residual (with optional 1x1 shortcut if C_in != C_out)
///   --> Output (B, H, W, C_out)
/// ```
public final class Flux2ResnetBlock: Module {

  @ModuleInfo(key: "norm1") var norm1: GroupNorm
  @ModuleInfo(key: "norm2") var norm2: GroupNorm
  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

  /// Whether a shortcut convolution is needed.
  public let hasShortcut: Bool

  /// Creates a Flux2 ResNet block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - groups: Number of groups for GroupNorm. Default `32`.
  ///   - eps: Epsilon for GroupNorm. Default `1e-6`.
  public init(inChannels: Int, outChannels: Int, groups: Int = 32, eps: Float = 1e-6) {
    self.hasShortcut = inChannels != outChannels

    self._norm1.wrappedValue = GroupNorm(
      groupCount: groups, dimensions: inChannels, eps: eps, pytorchCompatible: true)
    self._norm2.wrappedValue = GroupNorm(
      groupCount: groups, dimensions: outChannels, eps: eps, pytorchCompatible: true)
    self._conv1.wrappedValue = Conv2d(
      inputChannels: inChannels, outputChannels: outChannels,
      kernelSize: 3, stride: 1, padding: 1)
    self._conv2.wrappedValue = Conv2d(
      inputChannels: outChannels, outputChannels: outChannels,
      kernelSize: 3, stride: 1, padding: 1)

    if hasShortcut {
      self._convShortcut.wrappedValue = Conv2d(
        inputChannels: inChannels, outputChannels: outChannels,
        kernelSize: 1, stride: 1)
    }

    super.init()
  }

  /// Applies the residual block.
  ///
  /// - Parameter x: Input tensor in NHWC layout `(B, H, W, C_in)`.
  /// - Returns: Output tensor in NHWC layout `(B, H, W, C_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let residual = hasShortcut ? convShortcut!(x) : x

    var hidden = norm1(x.asType(.float32)).asType(x.dtype)
    hidden = silu(hidden)
    hidden = conv1(hidden)
    hidden = norm2(hidden.asType(.float32)).asType(x.dtype)
    hidden = silu(hidden)
    hidden = conv2(hidden)

    return hidden + residual
  }
}
