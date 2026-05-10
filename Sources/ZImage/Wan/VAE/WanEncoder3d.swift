import Foundation
import MLX
import MLXNN

/// 3D encoder for the Wan 2.2 VAE.
///
/// Compresses patchified RGB video to a latent representation. Uses CausalConv3d
/// layers throughout with RMSNorm normalization.
///
/// ## Architecture
///
/// ```
/// Input (B, 12, T, H/2, W/2)  (after patchify)
///   ├─ conv_in: CausalConv3d(12 → 160, k=3)
///   ├─ down_blocks[0]: DownBlock(160 → 320, spatial downsample)
///   ├─ down_blocks[1]: DownBlock(320 → 640, spatial downsample + temporal downsample)
///   ├─ down_blocks[2]: DownBlock(640 → 640, spatial downsample + temporal downsample)
///   ├─ down_blocks[3]: DownBlock(640 → 640, no downsample — last)
///   ├─ mid_block: MidBlock(640)
///   ├─ norm_out: RMSNorm(640, video)
///   ├─ SiLU
///   ├─ conv_out: CausalConv3d(640 → 96, k=3)
///   └─ Output (B, 96, T', H', W')
/// ```
public final class WanEncoder3d: Module {

  /// Initial convolution.
  @ModuleInfo(key: "conv_in") var convIn: WanCausalConv3d

  /// Encoder down blocks.
  @ModuleInfo(key: "down_blocks") var downBlocks: [WanDownBlock]

  /// Mid block.
  @ModuleInfo(key: "mid_block") var midBlock: WanMidBlock

  /// Output normalization.
  @ModuleInfo(key: "norm_out") var normOut: WanRMSNorm

  /// Output convolution.
  @ModuleInfo(key: "conv_out") var convOut: WanCausalConv3d

  /// Creates a Wan 3D encoder with default Wan 2.2 architecture.
  ///
  /// - Parameters:
  ///   - inChannels: Input channels (after patchify). Default `12`.
  ///   - dim: Base channel dimension. Default `160`.
  ///   - zDim: Latent dimension. Default `96` (48 * 2).
  ///   - dimMult: Channel multipliers per block. Default `[1, 2, 4, 4]`.
  ///   - numResBlocks: Residual blocks per down block. Default `2`.
  ///   - temporalDownsample: Per-block temporal downsample flags. Default `[false, true, true]`.
  public init(
    inChannels: Int = 12,
    dim: Int = 160,
    zDim: Int = 96,
    dimMult: [Int] = [1, 2, 4, 4],
    numResBlocks: Int = 2,
    temporalDownsample: [Bool] = [false, true, true]
  ) {
    // Compute channel dimensions
    let dims = ([1] + dimMult).map { dim * $0 }

    self._convIn.wrappedValue = WanCausalConv3d(
      inChannels: inChannels, outChannels: dims[0], kernelSize: 3, padding: 1
    )

    // Build down blocks
    var blocks: [WanDownBlock] = []
    for i in 0..<dimMult.count {
      let inDim = dims[i]
      let outDim = dims[i + 1]
      let isLast = (i == dimMult.count - 1)
      let tempDown = (i < temporalDownsample.count) ? temporalDownsample[i] : false

      blocks.append(WanDownBlock(
        inDim: inDim, outDim: outDim,
        numResBlocks: numResBlocks,
        temporalDownsample: tempDown,
        isLast: isLast
      ))
    }
    self._downBlocks.wrappedValue = blocks

    let lastDim = dims.last!
    self._midBlock.wrappedValue = WanMidBlock(dim: lastDim)
    self._normOut.wrappedValue = WanRMSNorm(dim: lastDim, images: false)
    self._convOut.wrappedValue = WanCausalConv3d(
      inChannels: lastDim, outChannels: zDim, kernelSize: 3, padding: 1
    )

    super.init()
  }

  /// Encodes the input to latent space.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Latent tensor of shape `(B, zDim, T', H', W')`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = convIn(x)

    for block in downBlocks {
      h = block(h)
    }

    h = midBlock(h)
    h = normOut(h)
    h = silu(h)
    h = convOut(h)

    return h
  }
}
