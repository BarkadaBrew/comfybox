// CausalConv3d.swift — 3D causal convolution for the Wan 2.2 VAE
// Ported from mflux: wan_2_2_causal_conv_3d.py
//
// Wan 2.2's CausalConv3d is simpler than SeedVR2's: it always uses causal
// temporal padding (pad past, not future) and does not support frame replication.
// Instead it uses explicit mx.pad with (2*pad_t, 0) on the temporal axis.
//
// For image generation (T=1), the causal padding adds 2 zero-frames before the
// single frame, which the 3D convolution consumes to produce a single output frame.
//
// Weight layout: MLX Conv3d expects (C_out, K_t, K_h, K_w, C_in) after the
// transposition done by FiboWeightMapping during loading.

import MLX
import MLXNN

/// A 3D convolution with causal temporal padding for the Wan 2.2 VAE.
///
/// Temporal causality is enforced by padding `(2 * padding, 0)` on the temporal
/// axis — all padding goes before the first frame, none after the last. Spatial
/// padding is symmetric `(pad_h, pad_h)` and `(pad_w, pad_w)`.
///
/// ## Differences from SeedVR2's CausalConv3d
///
/// - No frame replication — uses zero-padding via `mx.pad`
/// - Scalar kernel/stride/padding (not tuple) — Wan 2.2 VAE uses uniform 3x3x3
///   or 1x1x1 kernels exclusively
/// - Causal pad formula: `2 * pad_t` prepended on temporal axis
///
/// ## Input/Output Layout
///
/// External interface: `(B, C, T, H, W)` (channels-first, matching mflux convention).
/// The convolution internally transposes to `(B, T, H, W, C)` for MLX's channels-last
/// `convGeneral`, then transposes back.
public final class FiboCausalConv3d: Module {

  /// Convolution kernel weights: shape `(C_out, K_t, K_h, K_w, C_in)`.
  public var weight: MLXArray

  /// Bias vector: shape `(C_out,)`.
  public var bias: MLXArray

  /// Spatial and temporal padding value. Applied as `(2*padding, 0)` temporally
  /// and `(padding, padding)` spatially.
  public let padding: Int

  /// Convolution stride (uniform across all 3 dimensions).
  public let stride: Int

  /// Convolution kernel size (uniform across all 3 dimensions).
  public let kernelSize: Int

  /// Creates a Wan 2.2 causal 3D convolution layer.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - kernelSize: Uniform kernel size for all 3 dimensions. Default `3`.
  ///   - stride: Uniform stride for all 3 dimensions. Default `1`.
  ///   - padding: Padding value. Applied causally on temporal axis. Default `1`.
  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int = 3,
    stride: Int = 1,
    padding: Int = 1
  ) {
    self.kernelSize = kernelSize
    self.stride = stride
    self.padding = padding

    // Initialize with zeros — will be overwritten by weight loading
    self.weight = MLXArray.zeros([outChannels, kernelSize, kernelSize, kernelSize, inChannels])
    self.bias = MLXArray.zeros([outChannels])

    super.init()
  }

  /// Applies the causal 3D convolution.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C_out, T_out, H_out, W_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x

    // Apply causal padding: (2*pad, 0) on temporal, (pad, pad) on spatial
    if padding > 0 {
      input = MLX.padded(
        input,
        widths: [
          IntOrPair((0, 0)),      // batch
          IntOrPair((0, 0)),      // channels
          IntOrPair((2 * padding, 0)),  // temporal: causal (pad before, none after)
          IntOrPair((padding, padding)),  // height: symmetric
          IntOrPair((padding, padding)),  // width: symmetric
        ]
      )
    }

    // Transpose BCTHW -> BTHWC for convGeneral
    input = input.transposed(0, 2, 3, 4, 1)

    // 3D convolution with no additional padding (already applied above)
    var out = convGeneral(
      input, weight,
      strides: IntOrArray([stride, stride, stride]),
      padding: IntOrArray([0, 0, 0])
    )

    // Add bias
    out = out + bias

    // Transpose BTHWC -> BCTHW
    out = out.transposed(0, 4, 1, 2, 3)

    return out
  }
}
