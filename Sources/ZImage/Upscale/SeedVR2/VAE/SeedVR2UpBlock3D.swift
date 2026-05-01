import Foundation
import MLX
import MLXNN

/// Decoder upsampling block for the SeedVR2 3D VAE.
///
/// Chains a configurable number of ``SeedVR2ResnetBlock3D`` layers followed by
/// an optional ``SeedVR2Upsample3D``. The first resnet handles the channel
/// transition from `inChannels` to `outChannels`; subsequent resnets operate
/// at `outChannels`.
///
/// ## Architecture
///
/// ```
/// Input (B, C_in, T, H, W)
///   ├─ ResnetBlock3D[0] (C_in → C_out)
///   ├─ ResnetBlock3D[1] (C_out → C_out)
///   ├─ ResnetBlock3D[2] (C_out → C_out)
///   ├─ [optional] Upsample3D (doubles spatial, optionally temporal)
///   └─ Output (B, C_out, T*?, H*2, W*2) or (B, C_out, T, H, W) if no upsample
/// ```
public final class SeedVR2UpBlock3D: Module {

  /// Residual blocks in this stage.
  @ModuleInfo(key: "resnets") var resnets: [SeedVR2ResnetBlock3D]

  /// Optional upsampler (single-element array to match checkpoint key paths).
  @ModuleInfo(key: "upsamplers") var upsamplers: [SeedVR2Upsample3D]

  /// Creates a decoder up-block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - numLayers: Number of residual blocks. Default `3`.
  ///   - addUpsample: Whether to append an upsampling layer. Default `true`.
  ///   - temporalUp: Whether upsampling also doubles the temporal dimension.
  ///     Ignored when `addUpsample` is false. Default `false`.
  public init(
    inChannels: Int,
    outChannels: Int,
    numLayers: Int = 3,
    addUpsample: Bool = true,
    temporalUp: Bool = false
  ) {
    self._resnets.wrappedValue = (0..<numLayers).map { i in
      let inputCh = i == 0 ? inChannels : outChannels
      return SeedVR2ResnetBlock3D(inChannels: inputCh, outChannels: outChannels)
    }

    if addUpsample {
      self._upsamplers.wrappedValue = [
        SeedVR2Upsample3D(channels: outChannels, temporalUp: temporalUp)
      ]
    } else {
      self._upsamplers.wrappedValue = []
    }

    super.init()
  }

  /// Applies the up-block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C_in, T, H, W)`.
  /// - Returns: Output tensor with channels set to `outChannels` and spatial (and
  ///   optionally temporal) dimensions doubled if upsampling is present.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    for upsampler in upsamplers {
      hidden = upsampler(hidden)
    }
    return hidden
  }
}
