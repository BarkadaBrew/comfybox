// FiboVAEBlocks.swift — Shared building blocks for the Wan 2.2 VAE
// Ported from mflux:
//   - wan_2_2_rms_norm.py
//   - wan_2_2_residual_block.py
//   - wan_2_2_attention_block.py
//   - wan_2_2_mid_block.py
//   - wan_2_2_resample.py
//   - wan_2_2_avg_down_3d.py
//   - wan_2_2_dup_up_3d.py
//
// These differ from both the Flux VAE blocks (which use GroupNorm + Conv2d)
// and SeedVR2 blocks (which use GroupNorm + CausalConv3d with frame replication):
// - RMSNorm with L2-norm-based normalization (not variance-based)
// - Weight parameter named .gamma in safetensors (loaded via FiboWeightMapping)
// - All convolutions are FiboCausalConv3d (3D with causal temporal padding)
// - Attention collapses temporal dim, operates on 2D spatial with Conv2d

import MLX
import MLXFast
import MLXNN

// MARK: - RMS Norm (Wan 2.2 Style)

/// RMS normalization for the Wan 2.2 VAE.
///
/// Unlike standard RMSNorm (which uses variance along the last dim), this
/// uses L2 norm along the channel axis (axis=1) and scales by sqrt(dim).
/// The weight parameter is stored as `.gamma` in safetensors.
///
/// Two modes:
/// - `images=true`: weight shape `(dim, 1, 1)` — for 2D spatial attention
/// - `images=false`: weight shape `(dim, 1, 1, 1)` — for 3D volume data
///
/// Both modes normalize along axis 1 (channels) and broadcast the weight.
public final class FiboVAERMSNorm: Module {

  /// Learned scale parameter. Named `.gamma` in safetensors,
  /// loaded as `.weight` by the weight mapping.
  @ModuleInfo(key: "gamma") var gamma: MLXArray

  let eps: Float
  let scale: Float
  let images: Bool

  /// Creates a Wan 2.2 RMSNorm layer.
  ///
  /// - Parameters:
  ///   - dim: Number of channels to normalize over.
  ///   - eps: Numerical stability constant. Default `1e-12`.
  ///   - images: If true, weight is `(dim, 1, 1)` for 2D data.
  ///     If false, weight is `(dim, 1, 1, 1)` for 3D data. Default `false`.
  public init(dim: Int, eps: Float = 1e-12, images: Bool = false) {
    self.eps = eps
    self.scale = Float(dim).squareRoot()
    self.images = images

    if images {
      self._gamma.wrappedValue = MLX.ones([dim, 1, 1])
    } else {
      self._gamma.wrappedValue = MLX.ones([dim, 1, 1, 1])
    }

    super.init()
  }

  /// Applies L2-based RMS normalization.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, ...)`.
  /// - Returns: Normalized tensor of the same shape.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // L2 norm along channel axis
    let sumSq = MLX.sum(x * x, axis: 1, keepDims: true)
    let l2Norm = MLX.sqrt(sumSq)
    let denom = MLX.maximum(l2Norm, MLXArray(eps, dtype: l2Norm.dtype))
    let normalized = x / denom

    // Reshape weight for broadcasting
    let w: MLXArray
    if x.ndim == 5 && !images {
      w = gamma.reshaped(1, -1, 1, 1, 1)
    } else if x.ndim == 4 && images {
      w = gamma.reshaped(1, -1, 1, 1)
    } else if x.ndim == 5 {
      w = gamma.reshaped(1, -1, 1, 1, 1)
    } else if x.ndim == 4 {
      w = gamma.reshaped(1, -1, 1, 1)
    } else {
      w = gamma
    }

    return normalized * scale * w
  }
}

// MARK: - Residual Block

/// Residual block for the Wan 2.2 VAE.
///
/// Uses RMSNorm (not GroupNorm) and CausalConv3d (not Conv2d). The residual
/// connection uses a 1x1x1 convolution when input and output dims differ.
///
/// ```
/// x ──→ norm1 → SiLU → conv1 → norm2 → SiLU → conv2 → (+) → out
///  └──────────────── conv_shortcut (if dims differ) ──────┘
/// ```
public final class FiboVAEResidualBlock: Module {

  @ModuleInfo(key: "norm1") var norm1: FiboVAERMSNorm
  @ModuleInfo(key: "conv1") var conv1: FiboCausalConv3d
  @ModuleInfo(key: "norm2") var norm2: FiboVAERMSNorm
  @ModuleInfo(key: "conv2") var conv2: FiboCausalConv3d
  @ModuleInfo(key: "conv_shortcut") var convShortcut: FiboCausalConv3d?

  /// Creates a residual block.
  ///
  /// - Parameters:
  ///   - inDim: Input channel dimension.
  ///   - outDim: Output channel dimension.
  public init(inDim: Int, outDim: Int) {
    self._norm1.wrappedValue = FiboVAERMSNorm(dim: inDim, images: false)
    self._conv1.wrappedValue = FiboCausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: 3, padding: 1)
    self._norm2.wrappedValue = FiboVAERMSNorm(dim: outDim, images: false)
    self._conv2.wrappedValue = FiboCausalConv3d(inChannels: outDim, outChannels: outDim, kernelSize: 3, padding: 1)

    if inDim != outDim {
      self._convShortcut.wrappedValue = FiboCausalConv3d(
        inChannels: inDim, outChannels: outDim, kernelSize: 1, padding: 0
      )
    } else {
      self._convShortcut.wrappedValue = nil
    }

    super.init()
  }

  /// Applies the residual block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C_out, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let h = (convShortcut != nil) ? convShortcut!(x) : x
    var out = norm1(x)
    out = silu(out)
    out = conv1(out)
    out = norm2(out)
    out = silu(out)
    out = conv2(out)
    return out + h
  }
}

// MARK: - Attention Block

/// Spatial self-attention block for the Wan 2.2 VAE.
///
/// Operates on 2D spatial dimensions by collapsing the temporal axis into batch.
/// Uses Conv2d for Q/K/V projections and output projection. RMSNorm is applied
/// in `images=true` mode (2D spatial data).
///
/// ```
/// x (B, C, T, H, W) → collapse T into B → (B*T, C, H, W)
///   → RMSNorm → Conv2d(C → 3C) → split Q/K/V
///   → scaled_dot_product_attention
///   → Conv2d(C → C) → restore T → (B, C, T, H, W) → + identity
/// ```
public final class FiboVAEAttentionBlock: Module {

  let dim: Int

  @ModuleInfo(key: "norm") var norm: FiboVAERMSNorm
  @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
  @ModuleInfo(key: "proj") var proj: Conv2d

  /// Creates an attention block.
  ///
  /// - Parameter dim: Channel dimension for Q/K/V projections.
  public init(dim: Int) {
    self.dim = dim
    self._norm.wrappedValue = FiboVAERMSNorm(dim: dim, images: true)
    self._toQKV.wrappedValue = Conv2d(
      inputChannels: dim, outputChannels: dim * 3,
      kernelSize: 1
    )
    self._proj.wrappedValue = Conv2d(
      inputChannels: dim, outputChannels: dim,
      kernelSize: 1
    )

    super.init()
  }

  /// Applies spatial self-attention.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C, T, H, W)` with attention residual.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let identity = x
    let (batchSize, channels, time, height, width) = (
      x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4)
    )

    // Collapse temporal into batch: (B, C, T, H, W) -> (B*T, C, H, W)
    var h = x.transposed(0, 2, 1, 3, 4)
    h = h.reshaped(batchSize * time, channels, height, width)

    // RMSNorm on spatial data
    h = norm(h)

    // Transpose to NHWC for Conv2d
    h = h.transposed(0, 2, 3, 1)

    // QKV projection
    var qkv = toQKV(h)

    // Transpose back to NCHW
    qkv = qkv.transposed(0, 3, 1, 2)

    // Reshape for attention: (B*T, 1, C*3, H*W) -> (B*T, 1, H*W, C*3)
    qkv = qkv.reshaped(batchSize * time, 1, channels * 3, height * width)
    qkv = qkv.transposed(0, 1, 3, 2)

    // Split into Q, K, V
    let parts = split(qkv, parts: 3, axis: 3)
    let q = parts[0]
    let k = parts[1]
    let v = parts[2]

    // Scaled dot-product attention
    let scale = 1.0 / Float(channels).squareRoot()
    var attnOut = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v,
      scale: scale, mask: nil
    )

    // Reshape back: (B*T, 1, H*W, C) -> (B*T, C, H, W)
    attnOut = attnOut.reshaped(batchSize * time, height * width, channels)
    attnOut = attnOut.transposed(0, 2, 1)
    attnOut = attnOut.reshaped(batchSize * time, channels, height, width)

    // Output projection: NCHW -> NHWC -> Conv2d -> NCHW
    attnOut = attnOut.transposed(0, 2, 3, 1)
    attnOut = proj(attnOut)
    attnOut = attnOut.transposed(0, 3, 1, 2)

    // Restore temporal: (B*T, C, H, W) -> (B, T, C, H, W) -> (B, C, T, H, W)
    attnOut = attnOut.reshaped(batchSize, time, channels, height, width)
    attnOut = attnOut.transposed(0, 2, 1, 3, 4)

    return attnOut + identity
  }
}

// MARK: - Mid Block

/// Mid-block (bottleneck) for the Wan 2.2 VAE encoder and decoder.
///
/// Consists of an initial residual block, then alternating attention and
/// residual blocks. For the Wan 2.2 VAE, `numLayers=1` gives:
///   resnet[0] → attention[0] → resnet[1]
public final class FiboVAEMidBlock: Module {

  @ModuleInfo(key: "resnets") var resnets: [FiboVAEResidualBlock]
  @ModuleInfo(key: "attentions") var attentions: [FiboVAEAttentionBlock]

  /// Creates a mid-block.
  ///
  /// - Parameters:
  ///   - dim: Channel dimension for all blocks.
  ///   - numLayers: Number of attention + residual pairs after the initial
  ///     residual block. Default `1`.
  public init(dim: Int, numLayers: Int = 1) {
    var resnetList: [FiboVAEResidualBlock] = [FiboVAEResidualBlock(inDim: dim, outDim: dim)]
    var attnList: [FiboVAEAttentionBlock] = []

    for _ in 0..<numLayers {
      attnList.append(FiboVAEAttentionBlock(dim: dim))
      resnetList.append(FiboVAEResidualBlock(inDim: dim, outDim: dim))
    }

    self._resnets.wrappedValue = resnetList
    self._attentions.wrappedValue = attnList

    super.init()
  }

  /// Applies the mid-block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = resnets[0](x)
    for (attn, resnet) in zip(attentions, resnets.dropFirst()) {
      h = attn(h)
      h = resnet(h)
    }
    return h
  }
}

// MARK: - Resample (Up/Down)

/// Spatial resampling module for the Wan 2.2 VAE.
///
/// Supports four modes:
/// - `downsample2d`: 2x spatial downsampling via strided Conv2d
/// - `downsample3d`: Same as downsample2d (temporal handled by AvgDown3D)
/// - `upsample2d`: 2x spatial upsampling via nearest-neighbor + Conv2d
/// - `upsample3d`: Not used for FIBO image generation (temporal_upsample is empty)
///
/// The module collapses the temporal dimension into batch, operates in 2D,
/// then restores the temporal dimension.
public final class FiboVAEResample: Module {

  /// Resampling mode.
  public enum Mode: String {
    case upsample2d
    case upsample3d
    case downsample2d
    case downsample3d
  }

  let mode: Mode

  @ModuleInfo(key: "resample_conv") var resampleConv: Conv2d

  /// Creates a resampling module.
  ///
  /// - Parameters:
  ///   - dim: Input channel dimension.
  ///   - mode: Resampling mode.
  ///   - upsampleOutDim: Output channels for upsample modes. Default `nil`
  ///     (uses `dim / 2` for upsample, `dim` for downsample).
  public init(dim: Int, mode: Mode, upsampleOutDim: Int? = nil) {
    self.mode = mode
    let outDim = upsampleOutDim ?? (mode == .upsample2d || mode == .upsample3d ? dim / 2 : dim)

    switch mode {
    case .upsample2d, .upsample3d:
      self._resampleConv.wrappedValue = Conv2d(
        inputChannels: dim, outputChannels: outDim,
        kernelSize: 3, stride: 1, padding: 1
      )
    case .downsample2d, .downsample3d:
      self._resampleConv.wrappedValue = Conv2d(
        inputChannels: dim, outputChannels: dim,
        kernelSize: 3, stride: 2, padding: 0
      )
    }

    super.init()
  }

  /// Applies the resampling operation.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor with modified spatial dimensions.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))

    switch mode {
    case .upsample2d, .upsample3d:
      return upsample(x, b: b, c: c, t: t, h: h, w: w)
    case .downsample2d, .downsample3d:
      return downsample(x, b: b, c: c, t: t, h: h, w: w)
    }
  }

  private func upsample(_ x: MLXArray, b: Int, c: Int, t: Int, h: Int, w: Int) -> MLXArray {
    // Collapse temporal: (B, C, T, H, W) -> (B*T, C, H, W) -> (B*T, H, W, C)
    var out = x.transposed(0, 2, 1, 3, 4)
    out = out.reshaped(b * t, c, h, w)
    out = out.transposed(0, 2, 3, 1)

    // Nearest-neighbor 2x upsample: repeat each pixel 2x in H and W
    out = MLX.repeated(out, count: 2, axis: 1)
    out = MLX.repeated(out, count: 2, axis: 2)

    // Conv2d (already in NHWC)
    out = resampleConv(out)

    // Restore: NHWC -> NCHW -> (B, T, C', H', W') -> (B, C', T, H', W')
    out = out.transposed(0, 3, 1, 2)
    let newC = out.dim(1)
    let newH = out.dim(2)
    let newW = out.dim(3)
    out = out.reshaped(b, t, newC, newH, newW)
    out = out.transposed(0, 2, 1, 3, 4)
    return out
  }

  private func downsample(_ x: MLXArray, b: Int, c: Int, t: Int, h: Int, w: Int) -> MLXArray {
    // Collapse temporal: (B, C, T, H, W) -> (B*T, C, H, W) -> (B*T, H, W, C)
    var out = x.transposed(0, 2, 1, 3, 4)
    out = out.reshaped(b * t, c, h, w)
    out = out.transposed(0, 2, 3, 1)

    // Pad (0, 1) on H and W for strided conv (matching mflux)
    out = MLX.padded(out, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))])

    // Strided Conv2d (already in NHWC)
    out = resampleConv(out)

    // Restore: NHWC -> NCHW -> (B, T, C', H', W') -> (B, C', T, H', W')
    out = out.transposed(0, 3, 1, 2)
    let newC = out.dim(1)
    let newH = out.dim(2)
    let newW = out.dim(3)
    out = out.reshaped(b, t, newC, newH, newW)
    out = out.transposed(0, 2, 1, 3, 4)
    return out
  }
}

// MARK: - AvgDown3D (Encoder Shortcut)

/// Average-pool downsampling for the Wan 2.2 VAE encoder shortcut path.
///
/// Reshapes the input to expose spatial (and optionally temporal) sub-pixels,
/// then averages groups of channels to produce the target output dimension.
/// This is the residual shortcut in each encoder down block.
///
/// For image generation (T=1, factor_t=1 in most blocks):
///   - Spatial-only: reshapes (B, C, 1, H, W) to expose 2x2 patches,
///     channel-interleaves them, and averages groups to produce out_channels.
public final class FiboVAEAvgDown3D: Module {

  let inChannels: Int
  let outChannels: Int
  let factorT: Int
  let factorS: Int
  let factor: Int
  let groupSize: Int

  /// Creates an AvgDown3D layer.
  ///
  /// - Parameters:
  ///   - inChannels: Input channel count.
  ///   - outChannels: Output channel count.
  ///   - factorT: Temporal downsampling factor. Default `1`.
  ///   - factorS: Spatial downsampling factor. Default `1`.
  public init(inChannels: Int, outChannels: Int, factorT: Int, factorS: Int = 1) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.factorT = factorT
    self.factorS = factorS
    self.factor = factorT * factorS * factorS
    self.groupSize = inChannels * factor / outChannels

    super.init()
  }

  /// Applies average-pool downsampling.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Downsampled tensor.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x

    // Pad temporal axis if needed
    let padT = (factorT - input.dim(2) % factorT) % factorT
    if padT > 0 {
      input = MLX.padded(input, widths: [
        IntOrPair((0, 0)),    // batch
        IntOrPair((0, 0)),    // channels
        IntOrPair((padT, 0)), // time (pad before)
        IntOrPair((0, 0)),    // height
        IntOrPair((0, 0)),    // width
      ])
    }

    let (b, c, t, h, w) = (input.dim(0), input.dim(1), input.dim(2), input.dim(3), input.dim(4))

    // Reshape to expose sub-pixels
    input = input.reshaped(
      b, c,
      t / factorT, factorT,
      h / factorS, factorS,
      w / factorS, factorS
    )

    // Reorder to interleave spatial/temporal factors into channels
    input = input.transposed(0, 1, 3, 5, 7, 2, 4, 6)
    input = input.reshaped(
      b, c * factor,
      t / factorT,
      h / factorS,
      w / factorS
    )

    // Group and average
    input = input.reshaped(
      b, outChannels, groupSize,
      t / factorT,
      h / factorS,
      w / factorS
    )
    input = MLX.mean(input, axis: 2)

    return input
  }
}

// MARK: - DupUp3D (Decoder Shortcut)

/// Duplication-based upsampling for the Wan 2.2 VAE decoder shortcut path.
///
/// The inverse of AvgDown3D: repeats channels to fill the spatial (and optionally
/// temporal) expansion, then reshapes to the upsampled dimensions. This is the
/// residual shortcut in each decoder up block.
public final class FiboVAEDupUp3D: Module {

  let inChannels: Int
  let outChannels: Int
  let factorT: Int
  let factorS: Int
  let factor: Int
  let repeats: Int

  /// Creates a DupUp3D layer.
  ///
  /// - Parameters:
  ///   - inChannels: Input channel count.
  ///   - outChannels: Output channel count.
  ///   - factorT: Temporal upsampling factor. Default `1`.
  ///   - factorS: Spatial upsampling factor. Default `1`.
  public init(inChannels: Int, outChannels: Int, factorT: Int, factorS: Int = 1) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.factorT = factorT
    self.factorS = factorS
    self.factor = factorT * factorS * factorS
    self.repeats = outChannels * factor / inChannels

    super.init()
  }

  /// Applies duplication-based upsampling.
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `(B, C, T, H, W)`.
  ///   - firstChunk: If true and factorT > 1, trims the leading temporal
  ///     frames that result from causal padding. Default `false`.
  /// - Returns: Upsampled tensor.
  public func callAsFunction(_ x: MLXArray, firstChunk: Bool = false) -> MLXArray {
    let (b, _, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))

    // Repeat channels
    var out = MLX.repeated(x, count: repeats, axis: 1)

    // Reshape to separate upsampling factors
    out = out.reshaped(
      b, outChannels, factorT, factorS, factorS, t, h, w
    )

    // Interleave factors back into spatial/temporal dimensions
    out = out.transposed(0, 1, 5, 2, 6, 3, 7, 4)
    out = out.reshaped(
      b, outChannels, t * factorT, h * factorS, w * factorS
    )

    // Trim leading temporal frame for causal alignment
    if firstChunk && factorT > 1 {
      out = out[0..., 0..., (factorT - 1)..., 0..., 0...]
    }

    return out
  }
}
