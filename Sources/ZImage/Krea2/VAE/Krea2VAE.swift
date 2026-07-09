// Krea2VAE.swift — Qwen-Image VAE (AutoencoderKLQwenImage) decoder for Krea-2.
//
// The checkpoint is a 3D causal (Wan-style) video VAE, but for single images
// (T=1) every causal Conv3d — front-padded by 2 along time — sees the real frame
// only in its LAST temporal kernel slice. So the decoder reduces EXACTLY to 2D
// convs using `weight[:, :, -1, :, :]`; the temporal `time_conv` paths are never
// executed for images (mflux's QwenVAE does the same). We therefore implement
// the decoder 2D + NHWC-native and slice the temporal kernels at load time.
//
// @ModuleInfo keys mirror the checkpoint names (decoder.*, post_quant_conv) so
// Krea2WeightLoader only transforms shapes, never renames modules.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Norm

/// Qwen-Image VAE norm: x normalized by the L2 norm over channels (NOT rms),
/// scaled by sqrt(C) * gamma. eps 1e-12. Operates on NHWC (channels last).
public final class Krea2VAENorm: Module {
  @ModuleInfo(key: "gamma") var gamma: MLXArray
  let scale: Float
  let eps: Float

  public init(_ channels: Int, eps: Float = 1e-12) {
    self._gamma.wrappedValue = MLX.ones([channels])
    self.scale = Float(channels).squareRoot()
    self.eps = eps
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let l2 = MLX.sqrt(MLX.sum(x * x, axis: -1, keepDims: true))
    let denom = MLX.maximum(l2, MLXArray(eps).asType(l2.dtype))
    return (x / denom) * scale * gamma
  }
}

// MARK: - Blocks

public final class Krea2VAEResBlock: Module {
  @ModuleInfo(key: "norm1") var norm1: Krea2VAENorm
  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "norm2") var norm2: Krea2VAENorm
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

  public init(_ inChannels: Int, _ outChannels: Int) {
    self._norm1.wrappedValue = Krea2VAENorm(inChannels)
    self._conv1.wrappedValue = Conv2d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
    self._norm2.wrappedValue = Krea2VAENorm(outChannels)
    self._conv2.wrappedValue = Conv2d(
      inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
    self._convShortcut.wrappedValue = inChannels != outChannels
      ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
      : nil
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = conv1(silu(norm1(x)))
    h = conv2(silu(norm2(h)))
    let residual = convShortcut.map { $0(x) } ?? x
    return h + residual
  }
}

/// Single-head self-attention over the H*W tokens with full channel width.
public final class Krea2VAEAttnBlock: Module {
  let dim: Int
  @ModuleInfo(key: "norm") var norm: Krea2VAENorm
  @ModuleInfo(key: "to_qkv") var toQkv: Conv2d
  @ModuleInfo(key: "proj") var proj: Conv2d

  public init(_ dim: Int) {
    self.dim = dim
    self._norm.wrappedValue = Krea2VAENorm(dim)
    self._toQkv.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
    self._proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    let qkv = toQkv(norm(x)).reshaped(b, h * w, 3 * c)  // NHWC 1x1 conv
    let q = qkv[.ellipsis, 0..<c].expandedDimensions(axis: 1)         // (B,1,HW,C)
    let k = qkv[.ellipsis, c..<(2 * c)].expandedDimensions(axis: 1)
    let v = qkv[.ellipsis, (2 * c)...].expandedDimensions(axis: 1)
    var out = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v, scale: 1.0 / Float(c).squareRoot(), mask: .none)
    out = out[0..., 0].reshaped(b, h, w, c)
    return proj(out) + x
  }
}

public final class Krea2VAEMidBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [Krea2VAEResBlock]
  @ModuleInfo(key: "attentions") var attentions: [Krea2VAEAttnBlock]

  public init(_ dim: Int) {
    self._resnets.wrappedValue = [Krea2VAEResBlock(dim, dim), Krea2VAEResBlock(dim, dim)]
    self._attentions.wrappedValue = [Krea2VAEAttnBlock(dim)]
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = resnets[0](x)
    h = attentions[0](h)
    h = resnets[1](h)
    return h
  }
}

/// Checkpoint stores the spatial conv at `upsamplers.0.resample.1`; the loader
/// remaps that numeric segment to `resample.conv` (numeric keys unflatten as
/// array indices in MLX-Swift).
public final class Krea2VAEResampleSeq: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d
  public init(_ inChannels: Int, _ outChannels: Int) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
  }
}

public final class Krea2VAEUpsampler: Module {
  @ModuleInfo(key: "resample") var resample: Krea2VAEResampleSeq

  public init(_ inChannels: Int, _ outChannels: Int) {
    self._resample.wrappedValue = Krea2VAEResampleSeq(inChannels, outChannels)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Nearest-neighbor 2x upsample in NHWC, then 3x3 conv (halves channels).
    var h = MLX.repeated(x, count: 2, axis: 1)
    h = MLX.repeated(h, count: 2, axis: 2)
    return resample.conv(h)
  }
}

public final class Krea2VAEUpBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [Krea2VAEResBlock]
  @ModuleInfo(key: "upsamplers") var upsamplers: [Krea2VAEUpsampler]?

  public init(_ inChannels: Int, _ outChannels: Int, upsampleTo: Int?) {
    var blocks: [Krea2VAEResBlock] = []
    var dim = inChannels
    for _ in 0..<3 {
      blocks.append(Krea2VAEResBlock(dim, outChannels))
      dim = outChannels
    }
    self._resnets.wrappedValue = blocks
    self._upsamplers.wrappedValue = upsampleTo.map { [Krea2VAEUpsampler(outChannels, $0)] }
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = x
    for block in resnets { h = block(h) }
    if let ups = upsamplers { h = ups[0](h) }
    return h
  }
}

// MARK: - Decoder

public final class Krea2VAEDecoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") var midBlock: Krea2VAEMidBlock
  @ModuleInfo(key: "up_blocks") var upBlocks: [Krea2VAEUpBlock]
  @ModuleInfo(key: "norm_out") var normOut: Krea2VAENorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  public override init() {
    self._convIn.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 384, kernelSize: 3, padding: 1)
    self._midBlock.wrappedValue = Krea2VAEMidBlock(384)
    self._upBlocks.wrappedValue = [
      Krea2VAEUpBlock(384, 384, upsampleTo: 192),  // upsample3d (temporal path unused for images)
      Krea2VAEUpBlock(192, 384, upsampleTo: 192),  // upsample3d
      Krea2VAEUpBlock(192, 192, upsampleTo: 96),   // upsample2d
      Krea2VAEUpBlock(96, 96, upsampleTo: nil),
    ]
    self._normOut.wrappedValue = Krea2VAENorm(96)
    self._convOut.wrappedValue = Conv2d(inputChannels: 96, outputChannels: 3, kernelSize: 3, padding: 1)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = convIn(x)
    h = midBlock(h)
    for block in upBlocks { h = block(h) }
    h = convOut(silu(normOut(h)))
    return h
  }
}

// MARK: - VAE (decoder-only)

public final class Krea2VAE: Module {
  public static let spatialScale = 8
  public static let latentChannels = 16

  static let latentsMean: [Float] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
  ]
  static let latentsStd: [Float] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
  ]

  @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
  @ModuleInfo(key: "decoder") var decoder: Krea2VAEDecoder

  public override init() {
    self._postQuantConv.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: 1)
    self._decoder.wrappedValue = Krea2VAEDecoder()
  }

  /// latents: (B, latH, latW, 16) NHWC, normalized (as sampled).
  /// Returns RGB in [0,1]: (B, H, W, 3), H = latH*8.
  public func decode(_ latents: MLXArray) -> MLXArray {
    let mean = MLXArray(Krea2VAE.latentsMean).reshaped(1, 1, 1, 16).asType(latents.dtype)
    let std = MLXArray(Krea2VAE.latentsStd).reshaped(1, 1, 1, 16).asType(latents.dtype)
    var h = latents * std + mean
    h = postQuantConv(h)
    h = decoder(h)
    return MLX.clip(h, min: -1, max: 1) * 0.5 + 0.5
  }
}
