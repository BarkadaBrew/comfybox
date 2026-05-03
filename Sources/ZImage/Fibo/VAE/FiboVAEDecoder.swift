// FiboVAEDecoder.swift — Wan 2.2 VAE decoder (upsampling path)
// Ported from mflux:
//   - wan_2_2_decoder_3d.py
//   - wan_2_2_residual_up_block.py
//   - wan_2_2_up_block.py
//
// The decoder maps latent (B, z_dim, T_lat, H_lat, W_lat) back to pixel space
// (B, out_channels, T, H/2, W/2) through progressive upsampling.
//
// Architecture:
//   conv_in → mid_block → up_blocks[0..3] → norm_out → SiLU → conv_out
//
// Each residual up block has:
//   - Residual blocks (num_res_blocks + 1)
//   - A main-path upsampler (Resample) for all but the last block
//   - A shortcut-path upsampler (DupUp3D) for all but the last block
//   - The block output is: main_path(x) + shortcut(x_copy)
//
// For FIBO image generation (temporal_upsample=[]):
//   - No temporal upsampling — all blocks use upsample2d mode
//   - DupUp3D uses factor_t=1

import MLX
import MLXNN

// MARK: - Residual Up Block

/// Decoder up block with residual shortcut for the Wan 2.2 VAE.
///
/// Uses DupUp3D for the shortcut path (inverse of AvgDown3D in the encoder).
/// The main path has residual blocks followed by spatial upsampling.
public final class FiboVAEResidualUpBlock: Module {

  let inDim: Int
  let outDim: Int

  @ModuleInfo(key: "resnets") var resnets: [FiboVAEResidualBlock]
  @ModuleInfo(key: "upsampler") var upsampler: FiboVAEResample?
  @ModuleInfo(key: "avg_shortcut") var avgShortcut: FiboVAEDupUp3D?

  /// Creates a residual up block.
  ///
  /// - Parameters:
  ///   - inDim: Input channel dimension.
  ///   - outDim: Output channel dimension.
  ///   - numResBlocks: Number of residual blocks (total = numResBlocks + 1).
  ///   - temporalUpsample: Whether to also upsample temporally.
  ///   - upFlag: If true, adds upsampler and shortcut. False for the last block.
  public init(
    inDim: Int,
    outDim: Int,
    numResBlocks: Int,
    temporalUpsample: Bool = false,
    upFlag: Bool = false
  ) {
    self.inDim = inDim
    self.outDim = outDim

    // Shortcut path
    if upFlag {
      self._avgShortcut.wrappedValue = FiboVAEDupUp3D(
        inChannels: inDim,
        outChannels: outDim,
        factorT: temporalUpsample ? 2 : 1,
        factorS: 2
      )
    } else {
      self._avgShortcut.wrappedValue = nil
    }

    // Residual blocks: num_res_blocks + 1 (matching mflux)
    var resnetList: [FiboVAEResidualBlock] = []
    var currentDim = inDim
    for _ in 0..<(numResBlocks + 1) {
      resnetList.append(FiboVAEResidualBlock(inDim: currentDim, outDim: outDim))
      currentDim = outDim
    }
    self._resnets.wrappedValue = resnetList

    // Main path upsampler
    if upFlag {
      let mode: FiboVAEResample.Mode = temporalUpsample ? .upsample3d : .upsample2d
      self._upsampler.wrappedValue = FiboVAEResample(
        dim: outDim, mode: mode, upsampleOutDim: outDim
      )
    } else {
      self._upsampler.wrappedValue = nil
    }

    super.init()
  }

  /// Applies the up block.
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `(B, C, T, H, W)`.
  ///   - firstChunk: Passed to DupUp3D for temporal trim. Default `false`.
  /// - Returns: Upsampled tensor.
  public func callAsFunction(_ x: MLXArray, firstChunk: Bool = false) -> MLXArray {
    let xCopy = x

    var h = x
    for resnet in resnets {
      h = resnet(h)
    }

    if let us = upsampler {
      h = us(h)
    }

    if let shortcut = avgShortcut {
      h = h + shortcut(xCopy, firstChunk: firstChunk)
    }

    return h
  }
}

// MARK: - Decoder 3D

/// Full decoder for the Wan 2.2 VAE.
///
/// Progressive upsampling from latent space back to pixel space:
///
/// ```
/// Input (B, 48, T_lat, H_lat, W_lat)    — after post_quant_conv
///   ├─ conv_in:       CausalConv3d(48 → 1024, k=3)
///   ├─ mid_block:     MidBlock(1024)
///   ├─ up_blocks[0]:  ResidualUpBlock(1024 → 640, upsample)
///   ├─ up_blocks[1]:  ResidualUpBlock(640 → 512, upsample)
///   ├─ up_blocks[2]:  ResidualUpBlock(512 → 256, upsample)
///   ├─ up_blocks[3]:  ResidualUpBlock(256 → 256, no upsample, is_last)
///   ├─ norm_out:      RMSNorm(256) → SiLU
///   ├─ conv_out:      CausalConv3d(256 → 12, k=3)
///   └─ Output (B, 12, T, H/2, W/2)
/// ```
///
/// The output has `out_channels=12` which is then unpatchified to `(B, 3, T, H, W)`.
public final class FiboVAEDecoder: Module {

  @ModuleInfo(key: "conv_in") var convIn: FiboCausalConv3d
  @ModuleInfo(key: "mid_block") var midBlock: FiboVAEMidBlock
  @ModuleInfo(key: "up_blocks") var upBlocks: [FiboVAEResidualUpBlock]
  @ModuleInfo(key: "norm_out") var normOut: FiboVAERMSNorm
  @ModuleInfo(key: "conv_out") var convOut: FiboCausalConv3d

  /// Creates a Wan 2.2 VAE decoder.
  ///
  /// - Parameters:
  ///   - dim: Base channel dimension. Default `256`.
  ///   - zDim: Latent channel dimension. Default `48`.
  ///   - dimMult: Channel multipliers. Default `[1, 2, 4, 4]`.
  ///   - numResBlocks: Residual blocks per stage. Default `2`.
  ///   - temporalUpsample: Per-block temporal upsample flags. Default `[]` (none).
  ///   - outChannels: Output channels (before unpatchify). Default `12`.
  public init(
    dim: Int = 256,
    zDim: Int = 48,
    dimMult: [Int] = [1, 2, 4, 4],
    numResBlocks: Int = 2,
    temporalUpsample: [Bool] = [],
    outChannels: Int = 12
  ) {
    // Decoder dims are reversed from encoder:
    // dims = [dim*dimMult[-1], dim*dimMult[-1], dim*dimMult[-2], ..., dim*dimMult[0]]
    let reversedMult = [dimMult.last!] + dimMult.reversed()
    let dims = reversedMult.map { dim * $0 }

    self._convIn.wrappedValue = FiboCausalConv3d(
      inChannels: zDim, outChannels: dims[0],
      kernelSize: 3, padding: 1
    )

    self._midBlock.wrappedValue = FiboVAEMidBlock(dim: dims[0], numLayers: 1)

    // Build up blocks
    var blocks: [FiboVAEResidualUpBlock] = []
    for i in 0..<dimMult.count {
      let inDim = dims[i]
      let outDim = dims[i + 1]
      let upFlag = i != dimMult.count - 1

      let tempUp: Bool
      if upFlag && i < temporalUpsample.count {
        tempUp = temporalUpsample[i]
      } else {
        tempUp = false
      }

      blocks.append(FiboVAEResidualUpBlock(
        inDim: inDim,
        outDim: outDim,
        numResBlocks: numResBlocks,
        temporalUpsample: tempUp,
        upFlag: upFlag
      ))
    }
    self._upBlocks.wrappedValue = blocks

    let lastDim = dims.last!
    self._normOut.wrappedValue = FiboVAERMSNorm(dim: lastDim, images: false)
    self._convOut.wrappedValue = FiboCausalConv3d(
      inChannels: lastDim, outChannels: outChannels,
      kernelSize: 3, padding: 1
    )

    super.init()
  }

  /// Decodes a latent tensor back to pixel space.
  ///
  /// - Parameter x: Latent tensor of shape `(B, z_dim, T_lat, H_lat, W_lat)`.
  /// - Returns: Decoded tensor of shape `(B, out_channels, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = convIn(x)
    h = midBlock(h)

    for upBlock in upBlocks {
      h = upBlock(h, firstChunk: true)
    }

    h = normOut(h)
    h = silu(h)
    h = convOut(h)

    return h
  }
}
