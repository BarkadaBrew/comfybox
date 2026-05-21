import Foundation
import MLX
import MLXNN

/// A 3D residual block for the LTX-2 Video VAE.
///
/// Uses PixelNorm (not GroupNorm) as the default normalization, matching the
/// LTX-2 architecture. Applies two sequential CausalConv3d layers with PixelNorm
/// and SiLU activation, connected by a residual shortcut.
///
/// ## Architecture
///
/// ```
/// Input (B, C_in, T, H, W)
///   ├─ PixelNorm → SiLU → CausalConv3d(C_in → C_out, k=3)
///   ├─ PixelNorm → SiLU → CausalConv3d(C_out → C_out, k=3)
///   ├─ + residual (with optional 1x1x1 shortcut if C_in ≠ C_out)
///   └─ Output (B, C_out, T, H, W)
/// ```
///
/// Unlike SeedVR2's ResnetBlock3D which uses GroupNorm, LTX-2 uses PixelNorm
/// (L2 normalization over channels). The convolutions use zero spatial padding
/// (encoder) or reflect padding (decoder), controlled by the padding mode.
public final class LTX2ResnetBlock3D: Module {

  /// First causal 3D convolution (C_in → C_out).
  @ModuleInfo(key: "conv1") var conv1: CausalConv3d

  /// Second causal 3D convolution (C_out → C_out).
  @ModuleInfo(key: "conv2") var conv2: CausalConv3d

  /// Optional 1x1x1 shortcut convolution when channel counts differ.
  @ModuleInfo(key: "shortcut") var shortcut: CausalConv3d?

  /// Whether a shortcut convolution is used (input channels != output channels).
  public let hasShortcut: Bool

  /// Epsilon for PixelNorm.
  public let eps: Float

  /// Creates a 3D residual block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels. If nil, equals inChannels.
  ///   - eps: Epsilon for normalization. Default `1e-6`.
  public init(
    inChannels: Int,
    outChannels: Int? = nil,
    eps: Float = 1e-6
  ) {
    let outCh = outChannels ?? inChannels
    self.hasShortcut = inChannels != outCh
    self.eps = eps

    self._conv1.wrappedValue = CausalConv3d(
      inChannels: inChannels, outChannels: outCh,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )
    self._conv2.wrappedValue = CausalConv3d(
      inChannels: outCh, outChannels: outCh,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    if hasShortcut {
      self._shortcut.wrappedValue = CausalConv3d(
        inChannels: inChannels, outChannels: outCh,
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
    let residual = hasShortcut ? shortcut!(x) : x

    // First: PixelNorm → SiLU → Conv3d
    var h = pixelNorm(x)
    h = silu(h)
    h = conv1(h)

    // Second: PixelNorm → SiLU → Conv3d
    h = pixelNorm(h)
    h = silu(h)
    h = conv2(h)

    return h + residual
  }

  /// PixelNorm: L2 normalize over the channel dimension.
  ///
  /// `x / sqrt(mean(x^2, axis=channel) + eps)`
  ///
  /// - Parameter x: Input tensor of shape `(B, C, ...)`.
  /// - Returns: Normalized tensor.
  private func pixelNorm(_ x: MLXArray) -> MLXArray {
    x / MLX.sqrt(MLX.mean(x * x, axis: 1, keepDims: true) + eps)
  }
}

/// A group of residual blocks, forming the "res_x" block type in the config.
///
/// Equivalent to UNetMidBlock3D in the Python reference: a sequence of
/// ResnetBlock3D without any attention layers.
///
/// Weight path: `res_blocks.{i}.conv1`, `res_blocks.{i}.conv2`, etc.
public final class LTX2UNetMidBlock3D: Module {

  /// Residual blocks in this group.
  @ModuleInfo(key: "res_blocks") var resBlocks: [LTX2ResnetBlock3D]

  /// Creates a group of residual blocks.
  ///
  /// - Parameters:
  ///   - inChannels: Number of channels (preserved through the block).
  ///   - numLayers: Number of residual blocks.
  public init(inChannels: Int, numLayers: Int) {
    self._resBlocks.wrappedValue = (0..<numLayers).map { _ in
      LTX2ResnetBlock3D(inChannels: inChannels)
    }
    super.init()
  }

  /// Applies all residual blocks sequentially.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resBlock in resBlocks {
      hidden = resBlock(hidden)
    }
    return hidden
  }
}
