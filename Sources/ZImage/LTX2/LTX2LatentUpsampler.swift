// LTX2LatentUpsampler.swift -- Spatial latent upsampler for two-stage pipeline
// Phase 4 of the LTX-2 Swift/MLX port
//
// Provides 2x spatial upsampling in latent space for the distilled two-stage
// pipeline. Architecture: Conv3d -> ResBlocks -> SpatialUpsampler -> ResBlocks -> Conv3d.
//
// The upsampler operates in channels-last format (B, F, H, W, C) internally
// but accepts and returns channels-first (B, C, F, H, W) for pipeline compat.
//
// Reference: upsampler.py classes LatentUpsampler, ResBlock3D, SpatialUpsampler2x

import Foundation
import MLX
import MLXRandom
import MLXNN

// MARK: - Conv3d Wrapper

/// Simple Conv3d wrapper using MLX's conv3d.
///
/// Weight shape: `(C_out, KD, KH, KW, C_in)` (MLX channels-last format).
/// Input/output: `(N, D, H, W, C)`.
final class LTX2UpsamplerConv3d: Module {
  @ParameterInfo(key: "weight") var weight: MLXArray
  @ParameterInfo(key: "bias") var bias: MLXArray?

  let inChannels: Int
  let outChannels: Int
  let kernelSize: Int
  let stride: Int
  let padding: Int

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int = 3,
    stride: Int = 1,
    padding: Int = 0
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize
    self.stride = stride
    self.padding = padding

    let scale = 1.0 / Float(inChannels * kernelSize * kernelSize * kernelSize).squareRoot()
    self._weight.wrappedValue = MLXRandom.uniform(
      low: MLXArray(-scale), high: MLXArray(scale),
      [outChannels, kernelSize, kernelSize, kernelSize, inChannels]
    )
    self._bias.wrappedValue = MLXArray.zeros([outChannels])

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = conv3d(x, weight,
                   stride: IntOrTriple(integerLiteral: stride),
                   padding: IntOrTriple(integerLiteral: padding))
    if let b = bias {
      y = y + b
    }
    return y
  }
}

// MARK: - GroupNorm3d

/// Group normalization for 3D tensors `(N, D, H, W, C)`.
final class LTX2GroupNorm3d: Module {
  @ParameterInfo(key: "weight") var weight: MLXArray
  @ParameterInfo(key: "bias") var bias: MLXArray

  let numGroups: Int
  let numChannels: Int
  let eps: Float

  init(numGroups: Int, numChannels: Int, eps: Float = 1e-5) {
    self.numGroups = numGroups
    self.numChannels = numChannels
    self.eps = eps
    self._weight.wrappedValue = MLXArray.ones([numChannels])
    self._bias.wrappedValue = MLXArray.zeros([numChannels])
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let n = x.dim(0)
    let d = x.dim(1)
    let h = x.dim(2)
    let w = x.dim(3)
    let c = x.dim(4)
    let inputDtype = x.dtype

    var xF = x.asType(.float32)
    xF = xF.reshaped(n, d * h * w, numGroups, c / numGroups)
    let mean = MLX.mean(xF, axes: [1, 3], keepDims: true)
    let v = xF.variance(axes: [1, 3], keepDims: true)
    xF = (xF - mean) / MLX.sqrt(v + MLXArray(eps))
    xF = xF.reshaped(n, d, h, w, c)
    xF = xF * weight.asType(.float32) + bias.asType(.float32)
    return xF.asType(inputDtype)
  }
}

// MARK: - PixelShuffle2D

/// Pixel shuffle for 2D spatial upsampling (per-frame).
///
/// Rearranges channels `(N, H, W, C * r^2)` -> `(N, H*r, W*r, C)`.
final class LTX2PixelShuffle2D: Module {
  let upscaleH: Int
  let upscaleW: Int

  init(upscaleH: Int = 2, upscaleW: Int = 2) {
    self.upscaleH = upscaleH
    self.upscaleW = upscaleW
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let n = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)
    let outC = c / (upscaleH * upscaleW)

    // (N, H, W, outC, rH, rW)
    var y = x.reshaped(n, h, w, outC, upscaleH, upscaleW)
    // (N, H, rH, W, rW, outC)
    y = y.transposed(0, 1, 4, 2, 5, 3)
    // (N, H*rH, W*rW, outC)
    y = y.reshaped(n, h * upscaleH, w * upscaleW, outC)
    return y
  }
}

// MARK: - Spatial Upsampler 2x

/// 2x spatial upsampler: Conv2d -> PixelShuffle(2).
final class LTX2SpatialUpsampler2x: Module {
  // Checkpoint stores the conv as Sequential index 0 (`upsampler.0.weight`); the
  // loader remaps `upsampler.0.*` -> `upsampler.conv.*` so this dict key matches.
  @ModuleInfo(key: "conv") var conv: Conv2d
  let pixelShuffle: LTX2PixelShuffle2D

  init(midChannels: Int = 1024) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: midChannels,
      outputChannels: 4 * midChannels,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self.pixelShuffle = LTX2PixelShuffle2D(upscaleH: 2, upscaleW: 2)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (N, D, H, W, C)
    let n = x.dim(0)
    let d = x.dim(1)
    let h = x.dim(2)
    let w = x.dim(3)
    let c = x.dim(4)

    // Process frame-by-frame: (N*D, H, W, C)
    var y = x.reshaped(n * d, h, w, c)
    y = conv(y)
    y = pixelShuffle(y)
    y = y.reshaped(n, d, h * 2, w * 2, c)
    return y
  }
}

// MARK: - ResBlock3D

/// Residual block with two Conv3d + GroupNorm + SiLU.
final class LTX2UpsamplerResBlock: Module {
  @ModuleInfo(key: "conv1") var conv1: LTX2UpsamplerConv3d
  @ModuleInfo(key: "norm1") var norm1: LTX2GroupNorm3d
  @ModuleInfo(key: "conv2") var conv2: LTX2UpsamplerConv3d
  @ModuleInfo(key: "norm2") var norm2: LTX2GroupNorm3d

  init(channels: Int) {
    self._conv1.wrappedValue = LTX2UpsamplerConv3d(
      inChannels: channels, outChannels: channels, kernelSize: 3, padding: 1)
    self._norm1.wrappedValue = LTX2GroupNorm3d(numGroups: 32, numChannels: channels)
    self._conv2.wrappedValue = LTX2UpsamplerConv3d(
      inChannels: channels, outChannels: channels, kernelSize: 3, padding: 1)
    self._norm2.wrappedValue = LTX2GroupNorm3d(numGroups: 32, numChannels: channels)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let residual = x
    var y = conv1(x)
    y = norm1(y)
    y = silu(y)
    y = conv2(y)
    y = norm2(y)
    y = silu(y + residual)
    return y
  }
}

// MARK: - Latent Upsampler

/// Spatial latent upsampler for the two-stage distilled pipeline.
///
/// Architecture: Conv3d -> GroupNorm -> SiLU -> ResBlocks -> Upsample2x -> ResBlocks -> Conv3d
///
/// Operates in channels-last format internally but accepts/returns channels-first
/// `(B, C, F, H, W)` for pipeline compatibility.
public final class LTX2LatentUpsampler: Module {
  @ModuleInfo(key: "initial_conv") var initialConv: LTX2UpsamplerConv3d
  @ModuleInfo(key: "initial_norm") var initialNorm: LTX2GroupNorm3d
  @ModuleInfo(key: "res_blocks") var resBlocks: [LTX2UpsamplerResBlock]
  @ModuleInfo(key: "upsampler") var upsampler: LTX2SpatialUpsampler2x
  @ModuleInfo(key: "post_upsample_res_blocks") var postResBlocks: [LTX2UpsamplerResBlock]
  @ModuleInfo(key: "final_conv") var finalConv: LTX2UpsamplerConv3d

  public let inChannels: Int
  public let midChannels: Int

  public init(
    inChannels: Int = 128,
    midChannels: Int = 1024,
    numBlocksPerStage: Int = 4
  ) {
    self.inChannels = inChannels
    self.midChannels = midChannels

    self._initialConv.wrappedValue = LTX2UpsamplerConv3d(
      inChannels: inChannels, outChannels: midChannels, kernelSize: 3, padding: 1)
    self._initialNorm.wrappedValue = LTX2GroupNorm3d(numGroups: 32, numChannels: midChannels)

    self._resBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
      LTX2UpsamplerResBlock(channels: midChannels)
    }

    self._upsampler.wrappedValue = LTX2SpatialUpsampler2x(midChannels: midChannels)

    self._postResBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
      LTX2UpsamplerResBlock(channels: midChannels)
    }

    self._finalConv.wrappedValue = LTX2UpsamplerConv3d(
      inChannels: midChannels, outChannels: inChannels, kernelSize: 3, padding: 1)

    super.init()
  }

  /// Upsample latents 2x spatially.
  ///
  /// - Parameter latent: Input `(B, C, F, H, W)` channels-first.
  /// - Returns: Upsampled `(B, C, F, H*2, W*2)` channels-first.
  public func callAsFunction(_ latent: MLXArray) -> MLXArray {
    // Convert to channels-last: (B, C, F, H, W) -> (B, F, H, W, C)
    var x = latent.transposed(0, 2, 3, 4, 1)

    x = initialConv(x)
    x = initialNorm(x)
    x = silu(x)

    for block in resBlocks {
      x = block(x)
    }

    x = upsampler(x)

    for block in postResBlocks {
      x = block(x)
    }

    x = finalConv(x)

    // Convert back to channels-first: (B, F, H, W, C) -> (B, C, F, H, W)
    return x.transposed(0, 4, 1, 2, 3)
  }
}

// MARK: - Upsample Utility

/// Upsample latents with denormalization/renormalization.
///
/// 1. Un-normalize: latent * std + mean
/// 2. Upsample via the upsampler network
/// 3. Re-normalize: (latent - mean) / std
///
/// - Parameters:
///   - latent: Input latent `(B, C, F, H, W)`.
///   - upsampler: The latent upsampler network.
///   - latentMean: Per-channel mean `(C,)`.
///   - latentStd: Per-channel std `(C,)`.
/// - Returns: Upsampled latent `(B, C, F, H*2, W*2)`.
public func ltx2UpsampleLatents(
  _ latent: MLXArray,
  upsampler: LTX2LatentUpsampler,
  latentMean: MLXArray,
  latentStd: MLXArray
) -> MLXArray {
  let mean = latentMean.reshaped(1, -1, 1, 1, 1)
  let std = latentStd.reshaped(1, -1, 1, 1, 1)

  // Un-normalize
  var x = latent * std + mean

  // Upsample
  x = upsampler(x)

  // Re-normalize
  x = (x - mean) / std

  return x
}
