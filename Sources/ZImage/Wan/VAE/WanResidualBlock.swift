import Foundation
import MLX
import MLXNN

/// Residual block for the Wan 2.2 VAE.
///
/// Uses RMSNorm (video mode) and SiLU activation with two CausalConv3d layers.
/// Includes a 1x1 shortcut convolution when input and output channel counts differ.
///
/// ## Architecture
///
/// ```
/// Input (B, C_in, T, H, W)
///   ├─ Shortcut: identity or CausalConv3d(1x1, C_in → C_out)
///   ├─ Main: RMSNorm → SiLU → CausalConv3d(3x3, C_in → C_out)
///   │        → RMSNorm → SiLU → CausalConv3d(3x3, C_out → C_out)
///   ├─ + shortcut
///   └─ Output (B, C_out, T, H, W)
/// ```
public final class WanResidualBlock: Module {

  /// First normalization layer.
  @ModuleInfo(key: "norm1") var norm1: WanRMSNorm

  /// First convolution (C_in → C_out).
  @ModuleInfo(key: "conv1") var conv1: WanCausalConv3d

  /// Second normalization layer.
  @ModuleInfo(key: "norm2") var norm2: WanRMSNorm

  /// Second convolution (C_out → C_out).
  @ModuleInfo(key: "conv2") var conv2: WanCausalConv3d

  /// Optional 1x1 shortcut convolution when channels differ.
  @ModuleInfo(key: "conv_shortcut") var convShortcut: WanCausalConv3d?

  /// Whether the shortcut uses a convolution.
  public let hasShortcut: Bool

  /// Creates a Wan residual block.
  ///
  /// - Parameters:
  ///   - inDim: Number of input channels.
  ///   - outDim: Number of output channels.
  public init(inDim: Int, outDim: Int) {
    self.hasShortcut = (inDim != outDim)

    self._norm1.wrappedValue = WanRMSNorm(dim: inDim, images: false)
    self._conv1.wrappedValue = WanCausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: 3, padding: 1)
    self._norm2.wrappedValue = WanRMSNorm(dim: outDim, images: false)
    self._conv2.wrappedValue = WanCausalConv3d(inChannels: outDim, outChannels: outDim, kernelSize: 3, padding: 1)

    if inDim != outDim {
      self._convShortcut.wrappedValue = WanCausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: 1, padding: 0)
    }

    super.init()
  }

  /// Applies the residual block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C_in, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C_out, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Shortcut path
    let h = hasShortcut ? convShortcut!(x) : x

    // Main path
    var out = norm1(x)
    out = silu(out)
    out = conv1(out)
    out = norm2(out)
    out = silu(out)
    out = conv2(out)

    return out + h
  }
}
