import Foundation
import MLX
import MLXNN

/// A 3D convolution with causal temporal padding for the SeedVR2 video VAE.
///
/// In video autoencoders, temporal causality ensures that the encoding of frame `t`
/// depends only on frames `<= t`. This is achieved by replicating the first frame
/// to fill the temporal receptive field, rather than using symmetric zero-padding.
///
/// ## Weight Layout
///
/// Weights are stored as `(C_out, K_t, K_h, K_w, C_in)` — the MLX channels-last
/// layout for 3D convolution kernels.
///
/// ## Input Layout
///
/// The Python reference uses BCTHW (batch, channels, time, height, width) as the
/// external interface. Internally, the convolution operates on BTHWC (channels-last)
/// because MLX `convGeneral` expects `(N, ..., C_in)` inputs. This module handles
/// the necessary transposes.
///
/// ## Padding Modes
///
/// - `causalTemporal = true` (default): Replicates the first temporal frame
///   `(kernel_t - 1)` times and prepends them, with zero temporal padding in the
///   convolution itself. Spatial padding is applied normally.
/// - `causalTemporal = true, usePaddingCausal = true`: Uses `2 * padding_t` replicated
///   frames instead of `kernel_t - 1`.
/// - `causalTemporal = false`: Standard symmetric temporal padding (no replication).
public final class CausalConv3d: Module {

  /// Convolution kernel weights: shape `(C_out, K_t, K_h, K_w, C_in)`.
  public var weight: MLXArray

  /// Bias vector: shape `(C_out,)`.
  public var bias: MLXArray

  /// Kernel size as `(temporal, height, width)`.
  public let kernelSize: (Int, Int, Int)

  /// Stride as `(temporal, height, width)`.
  public let stride: (Int, Int, Int)

  /// Nominal padding as `(temporal, height, width)`.
  /// When ``causalTemporal`` is true, the temporal component drives the causal
  /// pad size but is not passed to the convolution.
  public let padding: (Int, Int, Int)

  /// Whether to apply causal temporal padding via frame replication.
  public let causalTemporal: Bool

  /// When true and ``causalTemporal`` is true, uses `2 * padding.temporal`
  /// replicated frames instead of `kernel_t - 1`.
  public let usePaddingCausal: Bool

  /// Creates a causal 3D convolution layer.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - kernelSize: Convolution kernel size as `(t, h, w)`.
  ///   - stride: Convolution stride as `(t, h, w)`. Default `(1, 1, 1)`.
  ///   - padding: Nominal padding as `(t, h, w)`. Default `(1, 1, 1)`.
  ///   - causalTemporal: Whether to replicate the first frame for causal padding.
  ///     Default `true`.
  ///   - usePaddingCausal: If true, the causal pad count is `2 * padding.t` instead
  ///     of `kernelSize.t - 1`. Default `false`.
  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: (Int, Int, Int) = (3, 3, 3),
    stride: (Int, Int, Int) = (1, 1, 1),
    padding: (Int, Int, Int) = (1, 1, 1),
    causalTemporal: Bool = true,
    usePaddingCausal: Bool = false
  ) {
    self.kernelSize = kernelSize
    self.stride = stride
    self.padding = padding
    self.causalTemporal = causalTemporal
    self.usePaddingCausal = usePaddingCausal

    let (kt, kh, kw) = kernelSize
    self.weight = MLXArray.zeros([outChannels, kt, kh, kw, inChannels])
    self.bias = MLXArray.zeros([outChannels])

    super.init()
  }

  /// Applies the causal 3D convolution.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)` (channels-first,
  ///   matching the Python SeedVR2 VAE convention).
  /// - Returns: Output tensor of shape `(B, C_out, T_out, H_out, W_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x
    let (kt, _, _) = kernelSize
    let (pt, ph, pw) = padding

    // --- Temporal padding ---
    let temporalPadding: Int
    if causalTemporal && kt > 1 {
      let causalPad = usePaddingCausal ? (2 * pt) : (kt - 1)
      if causalPad > 0 {
        // Replicate the first frame along the temporal axis.
        // Input shape: (B, C, T, H, W) — temporal is axis 2.
        let firstFrame = input[0..., 0..., ..<1, 0..., 0...]
        let padFrames = MLX.repeated(firstFrame, count: causalPad, axis: 2)
        input = MLX.concatenated([padFrames, input], axis: 2)
      }
      temporalPadding = 0
    } else {
      temporalPadding = pt
    }

    // --- Transpose BCTHW -> BTHWC for convGeneral ---
    input = input.transposed(0, 2, 3, 4, 1)

    // --- 3D convolution ---
    var out = convGeneral(
      input, weight,
      strides: IntOrArray([stride.0, stride.1, stride.2]),
      padding: IntOrArray([temporalPadding, ph, pw])
    )

    // --- Add bias ---
    out = out + bias

    // --- Transpose BTHWC -> BCTHW ---
    out = out.transposed(0, 4, 1, 2, 3)

    return out
  }
}
