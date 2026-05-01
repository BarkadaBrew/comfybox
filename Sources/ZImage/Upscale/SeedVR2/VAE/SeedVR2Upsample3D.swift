import Foundation
import MLX
import MLXNN

/// Spatial (and optionally temporal) upsampling for the SeedVR2 3D VAE decoder.
///
/// Uses sub-pixel convolution (depth-to-space) to increase resolution: a causal
/// convolution expands the channel dimension by `spatial_factor^2 * temporal_factor`,
/// then the extra channels are rearranged into spatial/temporal dimensions via
/// reshape and transpose.
///
/// ## Architecture
///
/// ```
/// Input (B, C, T, H, W)
///   ├─ upscale_conv: CausalConv3d(C → C * factor, k=1) — expand channels
///   ├─ reshape to (B, sf, sf, tf, C, T, H, W)
///   ├─ transpose to (B, C, T*tf, H*sf, W*sf)         — depth-to-space
///   ├─ conv: CausalConv3d(C → C, k=3)                — smooth
///   └─ Output (B, C, T*tf, H*sf, W*sf)
/// ```
///
/// When T=1 and temporal upsampling is enabled, the duplicated temporal frames
/// are trimmed back to 1 to maintain single-frame semantics.
public final class SeedVR2Upsample3D: Module {

  /// 1x1x1 convolution to expand channels for sub-pixel rearrangement.
  @ModuleInfo(key: "upscale_conv") var upscaleConv: CausalConv3d

  /// 3x3x3 causal convolution to smooth after rearrangement.
  @ModuleInfo(key: "conv") var conv: CausalConv3d

  /// Spatial upsampling factor (always 2).
  public let spatialFactor: Int

  /// Temporal upsampling factor (2 if temporal, 1 otherwise).
  public let temporalFactor: Int

  /// Creates an upsampling module.
  ///
  /// - Parameters:
  ///   - channels: Number of input/output channels (preserved).
  ///   - temporalUp: If true, also doubles the temporal dimension. Default `false`.
  public init(channels: Int, temporalUp: Bool = false) {
    self.spatialFactor = 2
    self.temporalFactor = temporalUp ? 2 : 1

    let totalFactor = spatialFactor * spatialFactor * temporalFactor

    self._upscaleConv.wrappedValue = CausalConv3d(
      inChannels: channels, outChannels: channels * totalFactor,
      kernelSize: (1, 1, 1), stride: (1, 1, 1), padding: (0, 0, 0)
    )
    self._conv.wrappedValue = CausalConv3d(
      inChannels: channels, outChannels: channels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1),
      usePaddingCausal: true
    )

    super.init()
  }

  /// Applies the upsampling via sub-pixel convolution.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Upsampled tensor of shape `(B, C, T*tf, H*sf, W*sf)`, where
  ///   `sf` is the spatial factor (2) and `tf` is the temporal factor.
  ///   When T=1 and `temporalUp` is true, T remains 1.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    let sf = spatialFactor
    let tf = temporalFactor

    // Expand channels: (B, C * sf*sf*tf, T, H, W)
    var out = upscaleConv(x)

    // Sub-pixel rearrangement:
    // (B, sf, sf, tf, C, T, H, W) → transpose → (B, C, T*tf, H*sf, W*sf)
    out = out.reshaped(b, sf, sf, tf, c, t, h, w)
    out = out.transposed(0, 4, 5, 3, 6, 1, 7, 2)
    out = out.reshaped(b, c, t * tf, h * sf, w * sf)

    // When input is a single frame, trim duplicated temporal frames.
    if t == 1 && tf > 1 {
      out = out[0..., 0..., ..<1, 0..., 0...]
    }

    // Smoothing convolution
    out = conv(out)

    return out
  }
}
