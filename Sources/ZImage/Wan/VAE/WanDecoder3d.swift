import Foundation
import MLX
import MLXNN

/// 3D decoder for the Wan 2.2 VAE.
///
/// Decompresses latents back to patchified video. Uses CausalConv3d layers
/// throughout with RMSNorm normalization.
///
/// ## Architecture
///
/// ```
/// Input (B, 48, T', H', W')
///   ├─ conv_in: CausalConv3d(48 → 1024, k=3)
///   ├─ mid_block: MidBlock(1024)
///   ├─ up_blocks[0]: ResidualUpBlock(1024 → 1024, no upsample)
///   ├─ up_blocks[1]: ResidualUpBlock(1024 → 640, spatial upsample)
///   ├─ up_blocks[2]: ResidualUpBlock(640 → 320, spatial upsample)
///   ├─ up_blocks[3]: ResidualUpBlock(320 → 256, spatial upsample)
///   ├─ norm_out: RMSNorm(256, video)
///   ├─ SiLU
///   ├─ conv_out: CausalConv3d(256 → 12, k=3)
///   └─ Output (B, 12, T, H/2, W/2)  (before unpatchify)
/// ```
public final class WanDecoder3d: Module {

  /// Initial convolution.
  @ModuleInfo(key: "conv_in") var convIn: WanCausalConv3d

  /// Mid block.
  @ModuleInfo(key: "mid_block") var midBlock: WanMidBlock

  /// Decoder up blocks.
  @ModuleInfo(key: "up_blocks") var upBlocks: [WanResidualUpBlock]

  /// Output normalization.
  @ModuleInfo(key: "norm_out") var normOut: WanRMSNorm

  /// Output convolution.
  @ModuleInfo(key: "conv_out") var convOut: WanCausalConv3d

  /// Creates a Wan 3D decoder with default Wan 2.2 architecture.
  ///
  /// - Parameters:
  ///   - dim: Base channel dimension. Default `256`.
  ///   - zDim: Latent dimension. Default `48`.
  ///   - dimMult: Channel multipliers per block. Default `[1, 2, 4, 4]`.
  ///   - numResBlocks: Residual blocks per up block. Default `2`.
  ///   - outChannels: Output channels. Default `12`.
  public init(
    dim: Int = 256,
    zDim: Int = 48,
    dimMult: [Int] = [1, 2, 4, 4],
    numResBlocks: Int = 2,
    outChannels: Int = 12
  ) {
    // Compute channel dimensions (reversed for decoder)
    let dims = ([dimMult.last!] + dimMult.reversed()).map { dim * $0 }

    self._convIn.wrappedValue = WanCausalConv3d(
      inChannels: zDim, outChannels: dims[0], kernelSize: 3, padding: 1
    )

    self._midBlock.wrappedValue = WanMidBlock(dim: dims[0])

    // Build up blocks
    var blocks: [WanResidualUpBlock] = []
    for i in 0..<dimMult.count {
      let inDim = dims[i]
      let outDim = dims[i + 1]
      let hasUpsample = (i != dimMult.count - 1)

      blocks.append(WanResidualUpBlock(
        inDim: inDim, outDim: outDim,
        numResBlocks: numResBlocks,
        temporalUpsample: false,
        upFlag: hasUpsample
      ))
    }
    self._upBlocks.wrappedValue = blocks

    let lastDim = dims.last!
    self._normOut.wrappedValue = WanRMSNorm(dim: lastDim, images: false)
    self._convOut.wrappedValue = WanCausalConv3d(
      inChannels: lastDim, outChannels: outChannels, kernelSize: 3, padding: 1
    )

    super.init()
  }

  /// Decodes latents to patchified video.
  ///
  /// - Parameter x: Latent tensor of shape `(B, zDim, T', H', W')`.
  /// - Returns: Decoded tensor of shape `(B, outChannels, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = convIn(x)
    h = midBlock(h)

    for (i, block) in upBlocks.enumerated() {
      h = block(h, firstChunk: true)
      _ = i  // suppress unused warning
    }

    h = normOut(h)
    h = silu(h)
    h = convOut(h)

    return h
  }
}
