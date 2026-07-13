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

// MARK: - Encoder blocks

/// Checkpoint stores the spatial downsample conv at `resample.1`; the loader
/// remaps that numeric segment to `resample.conv`, mirroring the decoder's
/// upsampler convention. Ported from mflux's `QwenImageResample3D`
/// downsample2d/downsample3d mode (`qwen_image_resample_3d.py`): the two
/// modes are architecturally identical in the reference forward pass — both
/// only ever exercise the spatial `resample_conv`; a `time_conv` submodule is
/// constructed for the "3d" variant but is dead code (never invoked), so it
/// is not ported here, same as the decoder skips it.
public final class Krea2VAEDownsampleSeq: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d
  public init(_ channels: Int) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0)
  }
}

public final class Krea2VAEDownsampler: Module {
  @ModuleInfo(key: "resample") var resample: Krea2VAEDownsampleSeq

  public init(_ channels: Int) {
    self._resample.wrappedValue = Krea2VAEDownsampleSeq(channels)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Asymmetric zero-pad (bottom/right by 1 on H, W) then a stride-2 3x3
    // conv, halving spatial dims — matches QwenImageResample3D exactly.
    let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))])
    return resample.conv(padded)
  }
}

/// One flat entry in the encoder's `down_blocks` list — either a residual
/// block or a spatial downsampler, mirroring the checkpoint's own flat
/// indexing exactly (mflux's `QwenImageEncoder3D` groups these into 4
/// logical stages for construction, but the raw weights are one flat list —
/// modeling it flat here means the loader needs no index remapping).
public final class Krea2VAEEncDownEntry: Module {
  @ModuleInfo(key: "norm1") var norm1: Krea2VAENorm?
  @ModuleInfo(key: "conv1") var conv1: Conv2d?
  @ModuleInfo(key: "norm2") var norm2: Krea2VAENorm?
  @ModuleInfo(key: "conv2") var conv2: Conv2d?
  @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?
  @ModuleInfo(key: "resample") var resample: Krea2VAEDownsampleSeq?

  private let isDownsampler: Bool

  public static func resBlock(_ inChannels: Int, _ outChannels: Int) -> Krea2VAEEncDownEntry {
    Krea2VAEEncDownEntry(resBlockIn: inChannels, out: outChannels)
  }
  public static func downsampler(_ channels: Int) -> Krea2VAEEncDownEntry {
    Krea2VAEEncDownEntry(downsampleChannels: channels)
  }

  private init(resBlockIn inChannels: Int, out outChannels: Int) {
    self.isDownsampler = false
    self._norm1.wrappedValue = Krea2VAENorm(inChannels)
    self._conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
    self._norm2.wrappedValue = Krea2VAENorm(outChannels)
    self._conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
    self._convShortcut.wrappedValue = inChannels != outChannels
      ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
      : nil
  }

  private init(downsampleChannels channels: Int) {
    self.isDownsampler = true
    self._resample.wrappedValue = Krea2VAEDownsampleSeq(channels)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    if isDownsampler {
      let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))])
      return resample!.conv(padded)
    }
    var h = conv1!(silu(norm1!(x)))
    h = conv2!(silu(norm2!(h)))
    let residual = convShortcut.map { $0(x) } ?? x
    return h + residual
  }
}

/// Qwen-Image VAE encoder, ported from mflux's `QwenImageEncoder3D`. 4 stages
/// (dims 96→96→192→384→384, downsampling after the first 3), a shared
/// `Krea2VAEMidBlock`, then `norm_out`/`conv_out` to 32 channels (mean +
/// logvar, kept concatenated to match the checkpoint — see `Krea2VAE.encode`).
public final class Krea2VAEEncoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "down_blocks") var downBlocks: [Krea2VAEEncDownEntry]
  @ModuleInfo(key: "mid_block") var midBlock: Krea2VAEMidBlock
  @ModuleInfo(key: "norm_out") var normOut: Krea2VAENorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  public override init() {
    self._convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 96, kernelSize: 3, padding: 1)
    self._downBlocks.wrappedValue = [
      .resBlock(96, 96), .resBlock(96, 96), .downsampler(96),
      .resBlock(96, 192), .resBlock(192, 192), .downsampler(192),
      .resBlock(192, 384), .resBlock(384, 384), .downsampler(384),
      .resBlock(384, 384), .resBlock(384, 384),
    ]
    self._midBlock.wrappedValue = Krea2VAEMidBlock(384)
    self._normOut.wrappedValue = Krea2VAENorm(384)
    self._convOut.wrappedValue = Conv2d(inputChannels: 384, outputChannels: 32, kernelSize: 3, padding: 1)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = convIn(x)
    for entry in downBlocks { h = entry(h) }
    h = midBlock(h)
    h = convOut(silu(normOut(h)))
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

// MARK: - VAE

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
  @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
  @ModuleInfo(key: "encoder") var encoder: Krea2VAEEncoder

  public override init() {
    self._postQuantConv.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: 1)
    self._decoder.wrappedValue = Krea2VAEDecoder()
    self._quantConv.wrappedValue = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 1)
    self._encoder.wrappedValue = Krea2VAEEncoder()
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

  /// pixels: (B, H, W, 3) NHWC, RGB in [-1,1] (see `QwenImageIO.normalizeForEncoder`).
  /// Returns normalized latents (B, H/8, W/8, 16) NHWC, in the same
  /// (mean-0-ish) space `decode` expects and the Euler loop's noise uses.
  ///
  /// The encoder's conv_out produces 32 channels (mean + logvar of the
  /// diagonal Gaussian posterior); matching mflux's `QwenVAE.encode` and this
  /// codebase's `AutoencoderKL`/`InpaintUtilities` convention, only the
  /// deterministic mean half is kept — no posterior sampling.
  public func encode(_ pixels: MLXArray) -> MLXArray {
    var h = encoder(pixels)
    h = quantConv(h)
    let mean = h[0..., 0..., 0..., 0..<Krea2VAE.latentChannels]
    let datasetMean = MLXArray(Krea2VAE.latentsMean).reshaped(1, 1, 1, 16).asType(mean.dtype)
    let datasetStd = MLXArray(Krea2VAE.latentsStd).reshaped(1, 1, 1, 16).asType(mean.dtype)
    return (mean - datasetMean) / datasetStd
  }
}
