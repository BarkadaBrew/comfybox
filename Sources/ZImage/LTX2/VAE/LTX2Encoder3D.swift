import Foundation
import MLX
import MLXNN

/// 3D encoder for the LTX-2 Video VAE.
///
/// Compresses an RGB video to a 128-channel latent representation using
/// configurable blocks of residual layers and SpaceToDepth downsampling.
/// The default architecture provides 32x spatial compression (patchify 4x +
/// three 2x spatial downsamples) and 8x temporal compression.
///
/// ## Architecture (Default)
///
/// ```
/// Input (B, 3, F, H, W)
///   ├─ patchify(patch_size=4): (B, 48, F, H/4, W/4)
///   ├─ conv_in: CausalConv3d(48 → 128, k=3)
///   ├─ down_blocks[0]: res_x (4 layers)                           128 ch
///   ├─ down_blocks[1]: compress_space_res(2x) → 256 ch
///   ├─ down_blocks[2]: res_x (6 layers)                           256 ch
///   ├─ down_blocks[3]: compress_time_res(2x) → 512 ch
///   ├─ down_blocks[4]: res_x (6 layers)                           512 ch
///   ├─ down_blocks[5]: compress_all_res(2x) → 1024 ch
///   ├─ down_blocks[6]: res_x (2 layers)                           1024 ch
///   ├─ down_blocks[7]: compress_all_res(2x) → 2048 ch
///   ├─ down_blocks[8]: res_x (2 layers)                           2048 ch
///   ├─ PixelNorm → SiLU
///   ├─ conv_out: CausalConv3d(2048 → 129, k=3)                   128 + 1 (uniform logvar)
///   └─ Output: normalized means (B, 128, F', H', W')
/// ```
///
/// Frame count F must satisfy `(F - 1) % 8 == 0` (i.e., 1, 9, 17, 25, ...).
public final class LTX2Encoder3D: Module {

  /// Initial 3D convolution after patchify.
  @ModuleInfo(key: "conv_in") var convIn: CausalConv3d

  /// Encoder blocks: residual groups and downsamplers.
  /// Uses a dictionary (not array) because MLX-Swift only tracks dict-keyed modules.
  @ModuleInfo(key: "down_blocks") var downBlocks: [Int: Module]

  /// Output convolution producing latent channels (+1 for uniform logvar).
  @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

  /// Per-channel statistics for normalizing encoder output.
  @ModuleInfo(key: "per_channel_statistics") var perChannelStatistics: LTX2PerChannelStatistics

  /// Spatial patch size.
  public let patchSize: Int

  /// Number of latent channels.
  public let latentChannels: Int

  /// Creates an LTX-2 3D encoder.
  ///
  /// - Parameter config: Video VAE configuration. Defaults to `.default`.
  public init(config: LTX2VideoVAEConfig = .default) {
    self.patchSize = config.patchSize
    self.latentChannels = config.latentChannels

    let inChannels = config.inChannels * config.patchSize * config.patchSize
    var featureChannels = config.latentChannels

    // Initial convolution
    self._convIn.wrappedValue = CausalConv3d(
      inChannels: inChannels, outChannels: featureChannels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    // Build encoder blocks
    var blocks: [Int: Module] = [:]
    for (idx, blockDef) in config.encoderBlocks.enumerated() {
      switch blockDef {
      case .resX(let numLayers):
        blocks[idx] = LTX2UNetMidBlock3D(inChannels: featureChannels, numLayers: numLayers)

      case .compressSpaceRes(let multiplier):
        let outChannels = featureChannels * multiplier
        blocks[idx] = LTX2SpaceToDepthDownsample(
          inChannels: featureChannels, outChannels: outChannels,
          stride: (1, 2, 2)
        )
        featureChannels = outChannels

      case .compressTimeRes(let multiplier):
        let outChannels = featureChannels * multiplier
        blocks[idx] = LTX2SpaceToDepthDownsample(
          inChannels: featureChannels, outChannels: outChannels,
          stride: (2, 1, 1)
        )
        featureChannels = outChannels

      case .compressAllRes(let multiplier):
        let outChannels = featureChannels * multiplier
        blocks[idx] = LTX2SpaceToDepthDownsample(
          inChannels: featureChannels, outChannels: outChannels,
          stride: (2, 2, 2)
        )
        featureChannels = outChannels
      }
    }
    self._downBlocks.wrappedValue = blocks

    // Output convolution: latentChannels + 1 for uniform logvar
    let convOutChannels = config.latentChannels + 1
    self._convOut.wrappedValue = CausalConv3d(
      inChannels: featureChannels, outChannels: convOutChannels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1)
    )

    self._perChannelStatistics.wrappedValue = LTX2PerChannelStatistics(
      latentChannels: config.latentChannels
    )

    super.init()
  }

  /// Encodes a video to normalized latent means.
  ///
  /// - Parameter x: Input video tensor of shape `(B, 3, F, H, W)`.
  ///   F must be `1 + 8*k` (e.g., 1, 9, 17, 25, ...).
  /// - Returns: Normalized latent tensor of shape `(B, 128, F', H', W')`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Patchify: (B, 3, F, H, W) -> (B, 48, F, H/4, W/4)
    var hidden = LTX2Patchify.patchify(x, patchSizeHW: patchSize, patchSizeT: 1)

    // Initial convolution
    hidden = convIn(hidden)

    // Process through encoder blocks
    let sortedKeys = downBlocks.keys.sorted()
    for key in sortedKeys {
      let block = downBlocks[key]!
      if let resBlock = block as? LTX2UNetMidBlock3D {
        hidden = resBlock(hidden)
      } else if let downBlock = block as? LTX2SpaceToDepthDownsample {
        hidden = downBlock(hidden)
      }
    }

    // Output: PixelNorm → SiLU → Conv
    hidden = pixelNorm(hidden)
    hidden = silu(hidden)
    hidden = convOut(hidden)

    // Handle uniform logvar: split off last channel, expand, concatenate
    // means = sample[:, :-1, ...], logvar = sample[:, -1:, ...]
    let means = hidden[0..., ..<latentChannels, 0..., 0..., 0...]

    // Normalize means using per-channel statistics
    return perChannelStatistics.normalize(means)
  }

  /// PixelNorm: L2 normalize over channel dimension.
  private func pixelNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    x / MLX.sqrt(MLX.mean(x * x, axis: 1, keepDims: true) + eps)
  }
}
