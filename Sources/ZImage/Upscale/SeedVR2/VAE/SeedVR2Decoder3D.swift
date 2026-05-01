import Foundation
import MLX
import MLXNN

/// 3D decoder for the SeedVR2 video VAE.
///
/// Decodes a latent tensor back into an RGB video by progressively increasing
/// spatial (and optionally temporal) resolution through a series of up-blocks,
/// with a mid-block bottleneck featuring spatial self-attention.
///
/// ## Architecture
///
/// ```
/// Input (B, 16, T_lat, H/8, W/8)
///   ├─ conv_in:  CausalConv3d(16 → 512, k=3)
///   ├─ mid_block:      ResNet → Attention → ResNet (512 channels)
///   ├─ up_blocks[0]: UpBlock3D(512 → 512, 3 resnets, upsample spatial+temporal)
///   ├─ up_blocks[1]: UpBlock3D(512 → 256, 3 resnets, upsample spatial+temporal)
///   ├─ up_blocks[2]: UpBlock3D(256 → 128, 3 resnets, upsample spatial only)
///   ├─ up_blocks[3]: UpBlock3D(128 → 128, 3 resnets, no upsample)
///   ├─ GroupNorm(32, 128) → SiLU
///   ├─ conv_out: CausalConv3d(128 → 3, k=3)
///   └─ Output (B, 3, T, H, W)
/// ```
///
/// ## Temporal Upsampling
///
/// The `temporalUpBlocks` parameter controls how many of the first up-blocks
/// also double the temporal dimension. With the default of 2, up_blocks[0] and
/// up_blocks[1] apply temporal upsampling, yielding a 4x temporal expansion
/// (matching the encoders 4x temporal reduction).
public final class SeedVR2Decoder3D: Module {

  /// Initial 3D convolution from latent channels to the bottleneck width.
  @ModuleInfo(key: "conv_in") var convIn: CausalConv3d

  /// Bottleneck mid-block with spatial self-attention.
  @ModuleInfo(key: "mid_block") var midBlock: SeedVR2MidBlock3D

  /// Progressive upsampling blocks.
  @ModuleInfo(key: "up_blocks") var upBlocks: [SeedVR2UpBlock3D]

  /// Group normalization before the output convolution.
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm

  /// Final convolution producing RGB output.
  @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

  /// Creates a 3D decoder.
  ///
  /// - Parameters:
  ///   - inChannels: Number of latent channels. Default `16`.
  ///   - outChannels: Number of output channels (RGB = 3). Default `3`.
  ///   - blockOutChannels: Channel counts for each stage (in encoder order).
  ///     Default `[128, 256, 512, 512]`. These are reversed internally so
  ///     the decoder mirrors the encoder.
  ///   - layersPerBlock: Number of residual blocks per stage. Default `3`.
  ///   - temporalUpBlocks: Number of initial up-blocks that also upsample
  ///     temporally. Default `2`.
  public init(
    inChannels: Int = 16,
    outChannels: Int = 3,
    blockOutChannels: [Int] = [128, 256, 512, 512],
    layersPerBlock: Int = 3,
    temporalUpBlocks: Int = 2
  ) {
    let reversedChannels = Array(blockOutChannels.reversed())
    let numBlocks = reversedChannels.count

    // Input convolution
    self._convIn.wrappedValue = CausalConv3d(
      inChannels: inChannels, outChannels: reversedChannels[0],
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    // Mid-block
    self._midBlock.wrappedValue = SeedVR2MidBlock3D(
      channels: reversedChannels[0]
    )

    // Up-blocks with progressive channel reduction
    var outputChannel = reversedChannels[0]
    var ups: [SeedVR2UpBlock3D] = []

    for i in 0..<numBlocks {
      let inputChannel = outputChannel
      outputChannel = reversedChannels[i]
      let isFinalBlock = i == numBlocks - 1

      // Temporal upsampling applies to the first N blocks.
      // Python: temporal_up = i < temporal_up_blocks
      let temporalUp = i < temporalUpBlocks

      ups.append(SeedVR2UpBlock3D(
        inChannels: inputChannel,
        outChannels: outputChannel,
        numLayers: layersPerBlock,
        addUpsample: !isFinalBlock,
        temporalUp: temporalUp
      ))
    }
    self._upBlocks.wrappedValue = ups

    // Output normalization + convolution
    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: 32,
      dimensions: reversedChannels[numBlocks - 1],
      eps: 1e-6,
      pytorchCompatible: true
    )
    self._convOut.wrappedValue = CausalConv3d(
      inChannels: reversedChannels[numBlocks - 1],
      outChannels: outChannels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    super.init()
  }

  /// Decodes a latent tensor into an RGB video.
  ///
  /// - Parameter z: Latent tensor of shape `(B, latentChannels, T_lat, H_lat, W_lat)`.
  /// - Returns: Decoded tensor of shape `(B, 3, T, H, W)`.
  public func callAsFunction(_ z: MLXArray) -> MLXArray {


    var hidden = convIn(z)

    hidden = midBlock(hidden)

    for (i, upBlock) in upBlocks.enumerated() {
      hidden = upBlock(hidden)
    }

    // GroupNorm in channels-last: BCTHW → BTHWC → norm → BCTHW
    hidden = hidden.transposed(0, 2, 3, 4, 1)
    hidden = convNormOut(hidden.asType(.float32)).asType(z.dtype)
    hidden = hidden.transposed(0, 4, 1, 2, 3)

    hidden = silu(hidden)

    hidden = convOut(hidden)

    return hidden
  }
}
