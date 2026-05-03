// FiboVAEEncoder.swift — Wan 2.2 VAE encoder (downsampling path)
// Ported from mflux:
//   - wan_2_2_encoder_3d.py
//   - wan_2_2_down_block.py
//
// The encoder maps patchified input (B, C_in*4, T, H/2, W/2) to the latent
// space (B, z_dim*2, T_lat, H_lat, W_lat) through progressive downsampling.
//
// Architecture:
//   conv_in → down_blocks[0..3] → mid_block → norm_out → SiLU → conv_out
//
// Each down block has:
//   - Residual blocks (with optional attention, unused for FIBO)
//   - A main-path downsampler (Resample) for all but the last block
//   - A shortcut-path downsampler (AvgDown3D)
//   - The block output is: main_path(x) + shortcut(x)

import MLX
import MLXNN

// MARK: - Down Block

/// Encoder down block for the Wan 2.2 VAE.
///
/// Each block applies residual blocks, optionally downsamples the main path,
/// and adds an average-pool shortcut for the residual connection.
///
/// For FIBO (attn_scales=[]):
///   - No attention blocks are inserted
///   - Each block has `numResBlocks` residual blocks
///   - All but the last block have a spatial downsampler
public final class FiboVAEDownBlock: Module {

  @ModuleInfo(key: "resnets") var resnets: [FiboVAEResidualBlock]
  @ModuleInfo(key: "downsampler") var downsampler: FiboVAEResample?
  @ModuleInfo(key: "avg_shortcut") var avgShortcut: FiboVAEAvgDown3D

  /// Creates an encoder down block.
  ///
  /// - Parameters:
  ///   - inDim: Input channel dimension.
  ///   - outDim: Output channel dimension.
  ///   - numResBlocks: Number of residual blocks.
  ///   - temporalDownsample: Whether this block also downsamples temporally.
  ///   - isLast: If true, no spatial downsampler is added.
  public init(
    inDim: Int,
    outDim: Int,
    numResBlocks: Int,
    temporalDownsample: Bool = false,
    isLast: Bool = false
  ) {
    // Build residual blocks
    var resnetList: [FiboVAEResidualBlock] = []
    var currentDim = inDim
    for _ in 0..<numResBlocks {
      resnetList.append(FiboVAEResidualBlock(inDim: currentDim, outDim: outDim))
      currentDim = outDim
    }
    self._resnets.wrappedValue = resnetList

    // Shortcut path with downsample
    self._avgShortcut.wrappedValue = FiboVAEAvgDown3D(
      inChannels: inDim,
      outChannels: outDim,
      factorT: temporalDownsample ? 2 : 1,
      factorS: isLast ? 1 : 2
    )

    // Main path downsampler (all but last block)
    if !isLast {
      let mode: FiboVAEResample.Mode = temporalDownsample ? .downsample3d : .downsample2d
      self._downsampler.wrappedValue = FiboVAEResample(dim: outDim, mode: mode)
    } else {
      self._downsampler.wrappedValue = nil
    }

    super.init()
  }

  /// Applies the down block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Downsampled tensor.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let shortcut = avgShortcut(x)

    var h = x
    for resnet in resnets {
      h = resnet(h)
    }
    if let ds = downsampler {
      h = ds(h)
    }

    return h + shortcut
  }
}

// MARK: - Encoder 3D

/// Full encoder for the Wan 2.2 VAE.
///
/// Progressive downsampling from input to latent space:
///
/// ```
/// Input (B, 48, T, H/2, W/2)    — after patchify
///   ├─ conv_in:       CausalConv3d(48 → 160, k=3)
///   ├─ down_blocks[0]: DownBlock(160 → 320, no temporal, spatial 2x)
///   ├─ down_blocks[1]: DownBlock(320 → 640, temporal 2x, spatial 2x)
///   ├─ down_blocks[2]: DownBlock(640 → 640, temporal 2x, spatial 2x)
///   ├─ down_blocks[3]: DownBlock(640 → 640, no downsample, is_last)
///   ├─ mid_block:      MidBlock(640)
///   ├─ norm_out:       RMSNorm(640) → SiLU
///   ├─ conv_out:       CausalConv3d(640 → 96, k=3)
///   └─ Output (B, 96, T_lat, H_lat, W_lat)
/// ```
///
/// The output has `z_dim * 2 = 96` channels (mean + logvar). Only the first
/// 48 channels (mean) are used by the VAE.
public final class FiboVAEEncoder: Module {

  @ModuleInfo(key: "conv_in") var convIn: FiboCausalConv3d
  @ModuleInfo(key: "down_blocks") var downBlocks: [FiboVAEDownBlock]
  @ModuleInfo(key: "mid_block") var midBlock: FiboVAEMidBlock
  @ModuleInfo(key: "norm_out") var normOut: FiboVAERMSNorm
  @ModuleInfo(key: "conv_out") var convOut: FiboCausalConv3d

  /// Creates a Wan 2.2 VAE encoder.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels (after patchify). Default `12`
  ///     (3 RGB * patch_size^2 = 12, then patchified to 48).
  ///   - dim: Base channel dimension. Default `160`.
  ///   - zDim: Latent dimension (doubled for mean+logvar). Default `96`.
  ///   - dimMult: Channel multipliers for each stage. Default `[1, 2, 4, 4]`.
  ///   - numResBlocks: Residual blocks per stage. Default `2`.
  ///   - temporalDownsample: Per-block temporal downsample flags.
  ///     Default `[false, true, true]`.
  public init(
    inChannels: Int = 12,
    dim: Int = 160,
    zDim: Int = 96,
    dimMult: [Int] = [1, 2, 4, 4],
    numResBlocks: Int = 2,
    temporalDownsample: [Bool] = [false, true, true]
  ) {
    // Channel dimensions: [dim, dim*1, dim*2, dim*4, dim*4]
    let dims = [dim] + dimMult.map { dim * $0 }

    self._convIn.wrappedValue = FiboCausalConv3d(
      inChannels: inChannels, outChannels: dims[0],
      kernelSize: 3, padding: 1
    )

    // Build down blocks
    var blocks: [FiboVAEDownBlock] = []
    for i in 0..<dimMult.count {
      let inDim = dims[i]
      let outDim = dims[i + 1]
      let tempDown = i < temporalDownsample.count ? temporalDownsample[i] : false
      let isLast = i == dimMult.count - 1

      blocks.append(FiboVAEDownBlock(
        inDim: inDim,
        outDim: outDim,
        numResBlocks: numResBlocks,
        temporalDownsample: tempDown,
        isLast: isLast
      ))
    }
    self._downBlocks.wrappedValue = blocks

    let lastDim = dims.last!
    self._midBlock.wrappedValue = FiboVAEMidBlock(dim: lastDim, numLayers: 1)
    self._normOut.wrappedValue = FiboVAERMSNorm(dim: lastDim, images: false)
    self._convOut.wrappedValue = FiboCausalConv3d(
      inChannels: lastDim, outChannels: zDim,
      kernelSize: 3, padding: 1
    )

    super.init()
  }

  /// Encodes an input tensor into the latent space.
  ///
  /// - Parameter x: Patchified input of shape `(B, C_in, T, H, W)`.
  /// - Returns: Latent tensor of shape `(B, z_dim*2, T_lat, H_lat, W_lat)`.
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
