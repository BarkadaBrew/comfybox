import Foundation
import MLX
import MLXNN

/// 3D encoder for the SeedVR2 video VAE.
///
/// Encodes an RGB video tensor into a latent representation by progressively
/// reducing spatial (and optionally temporal) resolution through a series of
/// down-blocks, with a mid-block bottleneck featuring spatial self-attention.
///
/// ## Architecture
///
/// ```
/// Input (B, 3, T, H, W)
///   ├─ conv_in:  CausalConv3d(3 → 128, k=3)
///   ├─ down_blocks[0]: DownBlock3D(128 → 128, downsample spatial)
///   ├─ down_blocks[1]: DownBlock3D(128 → 256, downsample spatial+temporal)
///   ├─ down_blocks[2]: DownBlock3D(256 → 512, downsample spatial+temporal)
///   ├─ down_blocks[3]: DownBlock3D(512 → 512, no downsample)
///   ├─ mid_block:      ResNet → Attention → ResNet (512 channels)
///   ├─ GroupNorm(32, 512) → SiLU
///   ├─ conv_out: CausalConv3d(512 → 32, k=3)     // 32 = 2 * latent_channels
///   └─ Output (B, 32, T/4, H/8, W/8)
/// ```
///
/// The output has `2 * latentChannels` channels because the VAE produces both
/// mean and log-variance (only the mean is used at inference time).
///
/// ## Temporal Downsampling
///
/// The `temporalDownBlocks` parameter controls how many of the non-final
/// down-blocks also halve the temporal dimension. With the default of 2,
/// blocks at indices 1 and 2 (counting from the right before the final block)
/// apply temporal downsampling, yielding a 4x temporal reduction.
public final class SeedVR2Encoder3D: Module {

  /// Initial 3D convolution from RGB to first blocks channel count.
  @ModuleInfo(key: "conv_in") var convIn: CausalConv3d

  /// Progressive downsampling blocks.
  @ModuleInfo(key: "down_blocks") var downBlocks: [SeedVR2DownBlock3D]

  /// Bottleneck mid-block with spatial self-attention.
  @ModuleInfo(key: "mid_block") var midBlock: SeedVR2MidBlock3D

  /// Group normalization before the output convolution.
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm

  /// Final convolution producing 2x latent channels (mean + logvar).
  @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

  /// Creates a 3D encoder.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels (RGB = 3). Default `3`.
  ///   - outChannels: Number of latent channels. Default `16`.
  ///   - blockOutChannels: Channel counts for each down-block stage.
  ///     Default `[128, 256, 512, 512]`.
  ///   - layersPerBlock: Number of residual blocks per stage. Default `2`.
  ///   - temporalDownBlocks: Number of non-final blocks that also downsample
  ///     temporally. Default `2`.
  public init(
    inChannels: Int = 3,
    outChannels: Int = 16,
    blockOutChannels: [Int] = [128, 256, 512, 512],
    layersPerBlock: Int = 2,
    temporalDownBlocks: Int = 2
  ) {
    // Input convolution
    self._convIn.wrappedValue = CausalConv3d(
      inChannels: inChannels, outChannels: blockOutChannels[0],
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    // Down-blocks with progressive channel expansion
    let numBlocks = blockOutChannels.count
    var outputChannel = blockOutChannels[0]
    var downs: [SeedVR2DownBlock3D] = []

    for i in 0..<numBlocks {
      let inputChannel = outputChannel
      outputChannel = blockOutChannels[i]
      let isFinalBlock = i == numBlocks - 1

      // Temporal downsampling applies to blocks near the end (but not the final block).
      // Python: temporal_down = (i >= num_blocks - temporal_down_blocks - 1) and not is_final_block
      let temporalDown = (i >= numBlocks - temporalDownBlocks - 1) && !isFinalBlock

      downs.append(SeedVR2DownBlock3D(
        inChannels: inputChannel,
        outChannels: outputChannel,
        numLayers: layersPerBlock,
        addDownsample: !isFinalBlock,
        temporalDown: temporalDown
      ))
    }
    self._downBlocks.wrappedValue = downs

    // Mid-block
    self._midBlock.wrappedValue = SeedVR2MidBlock3D(
      channels: blockOutChannels[numBlocks - 1]
    )

    // Output normalization + convolution
    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: 32,
      dimensions: blockOutChannels[numBlocks - 1],
      eps: 1e-6,
      pytorchCompatible: true
    )
    self._convOut.wrappedValue = CausalConv3d(
      inChannels: blockOutChannels[numBlocks - 1],
      outChannels: 2 * outChannels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    super.init()
  }

  /// Encodes an input video tensor into the latent space.
  ///
  /// - Parameter x: Input tensor of shape `(B, C_in, T, H, W)`.
  /// - Returns: Latent tensor of shape `(B, 2 * latentChannels, T_out, H/8, W/8)`,
  ///   where the first half of channels is the mean and the second half is the
  ///   log-variance.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = convIn(x)

    for downBlock in downBlocks {
      hidden = downBlock(hidden)
    }

    hidden = midBlock(hidden)

    // GroupNorm in channels-last: BCTHW → BTHWC → norm → BCTHW
    hidden = hidden.transposed(0, 2, 3, 4, 1)
    hidden = convNormOut(hidden.asType(.float32)).asType(x.dtype)
    hidden = hidden.transposed(0, 4, 1, 2, 3)

    hidden = silu(hidden)
    hidden = convOut(hidden)

    return hidden
  }
}
