import Foundation
import MLX
import MLXNN

/// Down-encoder block for the Flux2 VAE.
///
/// Contains a sequence of ResNet blocks followed by an optional downsampler.
/// All operations use NHWC layout.
private final class Flux2DownEncoderBlock: Module {

  @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock]
  @ModuleInfo(key: "downsamplers") var downsamplers: [Flux2Downsample]

  /// Creates a down-encoder block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - numLayers: Number of ResNet blocks. Default `2`.
  ///   - groups: Number of groups for GroupNorm. Default `32`.
  ///   - eps: Epsilon for GroupNorm. Default `1e-6`.
  ///   - addDownsample: Whether to add a downsampler. Default `true`.
  init(
    inChannels: Int,
    outChannels: Int,
    numLayers: Int = 2,
    groups: Int = 32,
    eps: Float = 1e-6,
    addDownsample: Bool = true
  ) {
    self._resnets.wrappedValue = (0..<numLayers).map { i in
      Flux2ResnetBlock(
        inChannels: i == 0 ? inChannels : outChannels,
        outChannels: outChannels,
        groups: groups,
        eps: eps)
    }
    self._downsamplers.wrappedValue = addDownsample
      ? [Flux2Downsample(channels: outChannels)]
      : []
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
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

/// Encoder for the Flux2 VAE.
///
/// Takes an image in NHWC layout and produces latent features. The architecture
/// follows a standard VAE encoder pattern: conv_in, a series of down-encoder
/// blocks, a mid block, normalization, and conv_out.
///
/// ## Architecture
///
/// ```
/// Input (B, H, W, 3)
///   -> Conv2d(3 -> 128, k=3, p=1)           // conv_in
///   -> DownBlock(128 -> 128, downsample)
///   -> DownBlock(128 -> 256, downsample)
///   -> DownBlock(256 -> 512, downsample)
///   -> DownBlock(512 -> 512, no downsample)  // final block
///   -> MidBlock(512, attention)
///   -> GroupNorm(32, 512)
///   -> SiLU
///   -> Conv2d(512 -> 64, k=3, p=1)          // conv_out (2 * latent_channels)
///   -> Output (B, H/8, W/8, 64)
/// ```
public final class Flux2Encoder: Module {

  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "down_blocks") fileprivate var downBlocks: [Flux2DownEncoderBlock]
  @ModuleInfo(key: "mid_block") var midBlock: Flux2MidBlock
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  /// Creates a Flux2 VAE encoder.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels (RGB = 3). Default `3`.
  ///   - outChannels: Number of latent channels. Default `32`.
  ///   - blockOutChannels: Channel counts per stage. Default `[128, 256, 512, 512]`.
  ///   - layersPerBlock: Number of ResNet blocks per stage. Default `2`.
  ///   - normNumGroups: Number of groups for GroupNorm. Default `32`.
  ///   - eps: Epsilon for GroupNorm. Default `1e-6`.
  ///   - midBlockAddAttention: Whether the mid block includes attention. Default `true`.
  public init(
    inChannels: Int = 3,
    outChannels: Int = 32,
    blockOutChannels: [Int] = [128, 256, 512, 512],
    layersPerBlock: Int = 2,
    normNumGroups: Int = 32,
    eps: Float = 1e-6,
    midBlockAddAttention: Bool = true
  ) {
    self._convIn.wrappedValue = Conv2d(
      inputChannels: inChannels, outputChannels: blockOutChannels[0],
      kernelSize: 3, stride: 1, padding: 1)

    var downs: [Flux2DownEncoderBlock] = []
    for (i, outCh) in blockOutChannels.enumerated() {
      let inCh = i > 0 ? blockOutChannels[i - 1] : blockOutChannels[0]
      let isFinal = i == blockOutChannels.count - 1
      downs.append(Flux2DownEncoderBlock(
        inChannels: inCh,
        outChannels: outCh,
        numLayers: layersPerBlock,
        groups: normNumGroups,
        eps: eps,
        addDownsample: !isFinal))
    }
    self._downBlocks.wrappedValue = downs

    self._midBlock.wrappedValue = Flux2MidBlock(
      channels: blockOutChannels.last!,
      groups: normNumGroups,
      eps: eps,
      addAttention: midBlockAddAttention)

    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: normNumGroups, dimensions: blockOutChannels.last!,
      eps: eps, pytorchCompatible: true)

    self._convOut.wrappedValue = Conv2d(
      inputChannels: blockOutChannels.last!, outputChannels: 2 * outChannels,
      kernelSize: 3, stride: 1, padding: 1)

    super.init()
  }

  /// Encodes an image into latent features.
  ///
  /// - Parameter x: Input tensor in NHWC layout `(B, H, W, C_in)`.
  /// - Returns: Encoded features in NHWC layout `(B, H/8, W/8, 2*latent_channels)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = convIn(x)
    for block in downBlocks {
      hidden = block(hidden)
    }
    hidden = midBlock(hidden)
    hidden = convNormOut(hidden.asType(.float32)).asType(x.dtype)
    hidden = silu(hidden)
    hidden = convOut(hidden)
    return hidden
  }
}
