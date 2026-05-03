import Foundation
import MLX
import MLXNN

/// UNet-style mid block for the Flux2 VAE encoder and decoder.
///
/// Contains two ResNet blocks with an optional attention block between them.
/// All operations use NHWC layout.
///
/// ## Architecture
///
/// ```
/// Input (B, H, W, C)
///   -> ResNet(C -> C)
///   -> [Attention(C)]   (optional)
///   -> ResNet(C -> C)
///   -> Output (B, H, W, C)
/// ```
public final class Flux2MidBlock: Module {

  @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock]
  @ModuleInfo(key: "attentions") var attentions: [Flux2VAEAttention]

  /// Creates a Flux2 VAE mid block.
  ///
  /// - Parameters:
  ///   - channels: Number of channels (constant through the block).
  ///   - groups: Number of groups for GroupNorm. Default `32`.
  ///   - eps: Epsilon for GroupNorm. Default `1e-6`.
  ///   - addAttention: Whether to include an attention block. Default `true`.
  public init(channels: Int, groups: Int = 32, eps: Float = 1e-6, addAttention: Bool = true) {
    self._resnets.wrappedValue = [
      Flux2ResnetBlock(inChannels: channels, outChannels: channels, groups: groups, eps: eps),
      Flux2ResnetBlock(inChannels: channels, outChannels: channels, groups: groups, eps: eps),
    ]
    self._attentions.wrappedValue = addAttention
      ? [Flux2VAEAttention(channels: channels, groups: groups, eps: eps)]
      : []
    super.init()
  }

  /// Applies the mid block.
  ///
  /// - Parameter x: Input tensor in NHWC layout `(B, H, W, C)`.
  /// - Returns: Output tensor in NHWC layout `(B, H, W, C)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = resnets[0](x)
    if !attentions.isEmpty {
      hidden = attentions[0](hidden)
    }
    hidden = resnets[1](hidden)
    return hidden
  }
}
