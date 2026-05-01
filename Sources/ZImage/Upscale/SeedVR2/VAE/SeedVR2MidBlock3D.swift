import Foundation
import MLX
import MLXNN

/// Mid-block for the SeedVR2 3D VAE encoder and decoder.
///
/// Sits between the down/up-sampling stages, applying two residual blocks with
/// a spatial self-attention layer sandwiched between them. This allows the model
/// to capture long-range spatial dependencies at the bottleneck resolution.
///
/// ## Architecture
///
/// ```
/// Input (B, C, T, H, W)
///   ├─ ResnetBlock3D (C → C)
///   ├─ Attention3D (spatial self-attention per frame)
///   ├─ ResnetBlock3D (C → C)
///   └─ Output (B, C, T, H, W)
/// ```
///
/// Both the encoder and decoder use identical mid-block structures. The Python
/// reference has separate `MidBlock3D` classes in encoder/ and decoder/ directories,
/// but they share the same architecture and can be unified into a single type.
public final class SeedVR2MidBlock3D: Module {

  /// Two residual blocks: applied before and after the attention layer.
  @ModuleInfo(key: "resnets") var resnets: [SeedVR2ResnetBlock3D]

  /// Spatial self-attention (single element array matching checkpoint structure).
  @ModuleInfo(key: "attentions") var attentions: [SeedVR2Attention3D]

  /// Creates a mid-block.
  ///
  /// - Parameter channels: Number of channels (preserved through the block).
  public init(channels: Int) {
    self._resnets.wrappedValue = [
      SeedVR2ResnetBlock3D(inChannels: channels, outChannels: channels),
      SeedVR2ResnetBlock3D(inChannels: channels, outChannels: channels),
    ]
    self._attentions.wrappedValue = [
      SeedVR2Attention3D(channels: channels)
    ]

    super.init()
  }

  /// Applies the mid-block: resnet → attention → resnet.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = resnets[0](x)
    hidden = attentions[0](hidden)
    hidden = resnets[1](hidden)
    return hidden
  }
}
