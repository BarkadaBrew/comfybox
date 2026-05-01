import Foundation
import MLX
import MLXNN

/// Encoder downsampling block for the SeedVR2 3D VAE.
///
/// Chains a configurable number of ``SeedVR2ResnetBlock3D`` layers followed by
/// an optional ``SeedVR2Downsample3D``. The first resnet handles the channel
/// transition from `inChannels` to `outChannels`; subsequent resnets operate
/// at `outChannels`.
///
/// ## Architecture
///
/// ```
/// Input (B, C_in, T, H, W)
///   ├─ ResnetBlock3D[0] (C_in → C_out)
///   ├─ ResnetBlock3D[1] (C_out → C_out)
///   ├─ ...
///   ├─ [optional] Downsample3D (halves spatial, optionally temporal)
///   └─ Output (B, C_out, T/?, H/2, W/2) or (B, C_out, T, H, W) if no downsample
/// ```
public final class SeedVR2DownBlock3D: Module {

  /// Residual blocks in this stage.
  @ModuleInfo(key: "resnets") var resnets: [SeedVR2ResnetBlock3D]

  /// Optional downsampler (single-element array to match checkpoint key paths).
  @ModuleInfo(key: "downsamplers") var downsamplers: [SeedVR2Downsample3D]

  /// Creates an encoder down-block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - numLayers: Number of residual blocks. Default `2`.
  ///   - addDownsample: Whether to append a downsampling layer. Default `true`.
  ///   - temporalDown: Whether downsampling also halves the temporal dimension.
  ///     Ignored when `addDownsample` is false. Default `false`.
  public init(
    inChannels: Int,
    outChannels: Int,
    numLayers: Int = 2,
    addDownsample: Bool = true,
    temporalDown: Bool = false
  ) {
    self._resnets.wrappedValue = (0..<numLayers).map { i in
      let inputCh = i == 0 ? inChannels : outChannels
      return SeedVR2ResnetBlock3D(inChannels: inputCh, outChannels: outChannels)
    }

    if addDownsample {
      self._downsamplers.wrappedValue = [
        SeedVR2Downsample3D(channels: outChannels, spatialOnly: !temporalDown)
      ]
    } else {
      self._downsamplers.wrappedValue = []
    }

    super.init()
  }

  /// Applies the down-block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C_in, T, H, W)`.
  /// - Returns: Output tensor with channels set to `outChannels` and spatial (and
  ///   optionally temporal) dimensions halved if downsampling is present.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    for downsampler in downsamplers {
      hidden = downsampler(hidden)
    }
    return hidden
  }
}
