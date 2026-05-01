import Foundation
import MLX
import MLXNN

/// A 3D residual block for the SeedVR2 video VAE.
///
/// Applies two sequential CausalConv3d layers with GroupNorm and SiLU activation,
/// connected by a residual shortcut. When the input and output channel counts differ,
/// a 1x1x1 convolution adapts the residual path.
///
/// ## Architecture
///
/// ```
/// Input (B, C_in, T, H, W)
///   ├─ GroupNorm(32, C_in) → SiLU → CausalConv3d(C_in → C_out, k=3)
///   ├─ GroupNorm(32, C_out) → SiLU → CausalConv3d(C_out → C_out, k=3)
///   ├─ + residual (with optional 1x1x1 shortcut if C_in ≠ C_out)
///   └─ Output (B, C_out, T, H, W)
/// ```
///
/// GroupNorm operates on channels-last layout (BTHWC), so the forward pass
/// transposes around each normalization call, matching the Python reference.
///
/// Used by both the encoder and decoder paths of the SeedVR2 3D VAE.
public final class SeedVR2ResnetBlock3D: Module {

  /// First group normalization over input channels.
  @ModuleInfo(key: "norm1") var norm1: GroupNorm

  /// Second group normalization over output channels.
  @ModuleInfo(key: "norm2") var norm2: GroupNorm

  /// First causal 3D convolution (C_in → C_out).
  @ModuleInfo(key: "conv1") var conv1: CausalConv3d

  /// Second causal 3D convolution (C_out → C_out).
  @ModuleInfo(key: "conv2") var conv2: CausalConv3d

  /// Optional 1x1x1 shortcut convolution when channel counts differ.
  @ModuleInfo(key: "conv_shortcut") var convShortcut: CausalConv3d?

  /// Whether a shortcut convolution is used (input channels ≠ output channels).
  public let hasShortcut: Bool

  /// Creates a 3D residual block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  public init(inChannels: Int, outChannels: Int) {
    self.hasShortcut = inChannels != outChannels

    self._norm1.wrappedValue = GroupNorm(
      groupCount: 32, dimensions: inChannels, eps: 1e-6, pytorchCompatible: true
    )
    self._norm2.wrappedValue = GroupNorm(
      groupCount: 32, dimensions: outChannels, eps: 1e-6, pytorchCompatible: true
    )
    self._conv1.wrappedValue = CausalConv3d(
      inChannels: inChannels, outChannels: outChannels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )
    self._conv2.wrappedValue = CausalConv3d(
      inChannels: outChannels, outChannels: outChannels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    if hasShortcut {
      self._convShortcut.wrappedValue = CausalConv3d(
        inChannels: inChannels, outChannels: outChannels,
        kernelSize: (1, 1, 1), stride: (1, 1, 1), padding: (0, 0, 0)
      )
    }

    super.init()
  }

  /// Applies the residual block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C_in, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C_out, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let residual = hasShortcut ? convShortcut!(x) : x

    // First norm+activation+conv: transpose BCTHW → BTHWC for GroupNorm, then back.
    var h = x.transposed(0, 2, 3, 4, 1)
    h = norm1(h.asType(.float32)).asType(x.dtype)
    h = h.transposed(0, 4, 1, 2, 3)
    h = silu(h)
    h = conv1(h)

    // Second norm+activation+conv.
    h = h.transposed(0, 2, 3, 4, 1)
    h = norm2(h.asType(.float32)).asType(x.dtype)
    h = h.transposed(0, 4, 1, 2, 3)
    h = silu(h)
    h = conv2(h)

    return h + residual
  }
}
