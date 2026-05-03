import Foundation
import MLX
import MLXNN

/// Up-decoder block for the Flux2 VAE.
///
/// Contains a sequence of ResNet blocks followed by an optional upsampler.
/// All operations use NHWC layout.
private final class Flux2UpDecoderBlock: Module {

  @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock]
  @ModuleInfo(key: "upsamplers") var upsamplers: [Flux2Upsample]

  /// Creates an up-decoder block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - numLayers: Number of ResNet blocks. Default `3`.
  ///   - groups: Number of groups for GroupNorm. Default `32`.
  ///   - eps: Epsilon for GroupNorm. Default `1e-6`.
  ///   - addUpsample: Whether to add an upsampler. Default `true`.
  init(
    inChannels: Int,
    outChannels: Int,
    numLayers: Int = 3,
    groups: Int = 32,
    eps: Float = 1e-6,
    addUpsample: Bool = true
  ) {
    self._resnets.wrappedValue = (0..<numLayers).map { i in
      Flux2ResnetBlock(
        inChannels: i == 0 ? inChannels : outChannels,
        outChannels: outChannels,
        groups: groups,
        eps: eps)
    }
    self._upsamplers.wrappedValue = addUpsample
      ? [Flux2Upsample(channels: outChannels, outChannels: outChannels)]
      : []
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
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

/// Decoder for the Flux2 VAE.
///
/// Takes latent features in NHWC layout and reconstructs an image. The architecture
/// mirrors the encoder: conv_in, mid block, a series of up-decoder blocks,
/// normalization, and conv_out.
///
/// ## Architecture
///
/// ```
/// Input (B, H/8, W/8, 32)
///   -> Conv2d(32 -> 512, k=3, p=1)            // conv_in
///   -> MidBlock(512, attention)
///   -> UpBlock(512 -> 512, upsample)
///   -> UpBlock(512 -> 512, upsample)
///   -> UpBlock(512 -> 256, upsample)
///   -> UpBlock(256 -> 128, no upsample)        // final block
///   -> GroupNorm(32, 128)
///   -> SiLU
///   -> Conv2d(128 -> 3, k=3, p=1)             // conv_out
///   -> Output (B, H, W, 3)
/// ```
public final class Flux2Decoder: Module {

  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") var midBlock: Flux2MidBlock
  @ModuleInfo(key: "up_blocks") fileprivate var upBlocks: [Flux2UpDecoderBlock]
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  /// Creates a Flux2 VAE decoder.
  ///
  /// - Parameters:
  ///   - inChannels: Number of latent channels. Default `32`.
  ///   - outChannels: Number of output channels (RGB = 3). Default `3`.
  ///   - blockOutChannels: Channel counts per stage (in encoder order). Default `[128, 256, 512, 512]`.
  ///   - layersPerBlock: Number of ResNet blocks per stage (decoder adds 1). Default `2`.
  ///   - normNumGroups: Number of groups for GroupNorm. Default `32`.
  ///   - eps: Epsilon for GroupNorm. Default `1e-6`.
  ///   - midBlockAddAttention: Whether the mid block includes attention. Default `true`.
  public init(
    inChannels: Int = 32,
    outChannels: Int = 3,
    blockOutChannels: [Int] = [128, 256, 512, 512],
    layersPerBlock: Int = 2,
    normNumGroups: Int = 32,
    eps: Float = 1e-6,
    midBlockAddAttention: Bool = true
  ) {
    // Decoder processes channels in reverse order
    let reversedChannels = Array(blockOutChannels.reversed())

    self._convIn.wrappedValue = Conv2d(
      inputChannels: inChannels, outputChannels: reversedChannels[0],
      kernelSize: 3, stride: 1, padding: 1)

    self._midBlock.wrappedValue = Flux2MidBlock(
      channels: reversedChannels[0],
      groups: normNumGroups,
      eps: eps,
      addAttention: midBlockAddAttention)

    var ups: [Flux2UpDecoderBlock] = []
    for (i, outCh) in reversedChannels.enumerated() {
      let prevOutCh = i == 0 ? outCh : reversedChannels[i - 1]
      let isFinal = i == reversedChannels.count - 1
      ups.append(Flux2UpDecoderBlock(
        inChannels: prevOutCh,
        outChannels: outCh,
        numLayers: layersPerBlock + 1,
        groups: normNumGroups,
        eps: eps,
        addUpsample: !isFinal))
    }
    self._upBlocks.wrappedValue = ups

    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: normNumGroups, dimensions: blockOutChannels[0],
      eps: eps, pytorchCompatible: true)

    self._convOut.wrappedValue = Conv2d(
      inputChannels: blockOutChannels[0], outputChannels: outChannels,
      kernelSize: 3, stride: 1, padding: 1)

    super.init()
  }

  /// Decodes latent features into an image.
  ///
  /// - Parameter x: Latent tensor in NHWC layout `(B, H/8, W/8, latent_channels)`.
  /// - Returns: Decoded image in NHWC layout `(B, H, W, C_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = convIn(x)
    hidden = midBlock(hidden)
    for block in upBlocks {
      hidden = block(hidden)
    }
    hidden = convNormOut(hidden.asType(.float32)).asType(x.dtype)
    hidden = silu(hidden)
    hidden = convOut(hidden)
    return hidden
  }
}
