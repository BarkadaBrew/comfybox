import Foundation
import MLX
import MLXNN

/// Space-to-depth downsampling with 3x3 conv and skip connection.
///
/// Used as the encoder downsample block in LTX-2. Differs from SeedVR2's
/// strided convolution approach: instead, applies a 3x3 conv then rearranges
/// spatial/temporal elements into channels (space-to-depth), with a group-mean
/// skip connection from the input.
///
/// ## Architecture
///
/// ```
/// Input (B, C, D, H, W)
///   ├─ Skip: space_to_depth(input) → group_mean → (B, out_channels, D', H', W')
///   ├─ Conv: CausalConv3d(C → C_mid, k=3) → space_to_depth → (B, out_channels, D', H', W')
///   ├─ + skip connection
///   └─ Output (B, out_channels, D', H', W')
/// ```
///
/// Temporal stride of 2 includes causal padding (first frame duplicated).
public final class LTX2SpaceToDepthDownsample: Module {

  /// 3x3 causal convolution before space-to-depth rearrangement.
  /// Wrapped in LTX2ConvWrapper to match checkpoint key path (conv.conv.weight).
  @ModuleInfo(key: "conv") var conv: LTX2ConvWrapper

  /// Stride as `(temporal, height, width)`.
  public let stride: (Int, Int, Int)

  /// Number of output channels.
  public let outChannels: Int

  /// Group size for the skip connection's group mean.
  public let groupSize: Int

  /// Creates a SpaceToDepth downsample block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - stride: Downsampling stride as `(temporal, height, width)`.
  public init(
    inChannels: Int,
    outChannels: Int,
    stride: (Int, Int, Int)
  ) {
    self.stride = stride
    self.outChannels = outChannels

    let multiplier = stride.0 * stride.1 * stride.2
    self.groupSize = inChannels * multiplier / outChannels
    let convOutChannels = outChannels / multiplier

    self._conv.wrappedValue = LTX2ConvWrapper(
      inChannels: inChannels, outChannels: convOutChannels
    )

    super.init()
  }

  /// Rearrange spatial/temporal elements into channel dimension.
  ///
  /// `b c (d st) (h sh) (w sw) -> b (c st sh sw) d h w`
  private func spaceToDepth(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let d = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    let (st, sh, sw) = stride

    // Reshape: (B, C, D/st, st, H/sh, sh, W/sw, sw)
    var out = x.reshaped(b, c, d / st, st, h / sh, sh, w / sw, sw)

    // Permute: -> (B, C, st, sh, sw, D', H', W')
    out = out.transposed(0, 1, 3, 5, 7, 2, 4, 6)

    // Reshape: -> (B, C*st*sh*sw, D', H', W')
    let newC = c * st * sh * sw
    out = out.reshaped(b, newC, d / st, h / sh, w / sw)

    return out
  }

  /// Applies the downsampling block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, D, H, W)`.
  /// - Returns: Downsampled tensor of shape `(B, outChannels, D', H', W')`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x
    let (st, sh, sw) = stride

    // Temporal causal padding: duplicate first frame
    if st == 2 {
      let firstFrame = input[0..., 0..., ..<1, 0..., 0...]
      input = MLX.concatenated([firstFrame, input], axis: 2)
    }

    // Pad spatial dimensions if not divisible by stride
    let d = input.dim(2)
    let h = input.dim(3)
    let w = input.dim(4)
    let padD = (st - d % st) % st
    let padH = (sh - h % sh) % sh
    let padW = (sw - w % sw) % sw

    if padD > 0 || padH > 0 || padW > 0 {
      input = MLX.padded(
        input,
        widths: [
          IntOrPair((0, 0)),
          IntOrPair((0, 0)),
          IntOrPair((0, padD)),
          IntOrPair((0, padH)),
          IntOrPair((0, padW)),
        ]
      )
    }

    // Skip connection: space-to-depth on input, then group mean
    var xIn = spaceToDepth(input)
    let b2 = xIn.dim(0)
    let d2 = xIn.dim(2)
    let h2 = xIn.dim(3)
    let w2 = xIn.dim(4)
    xIn = xIn.reshaped(b2, outChannels, groupSize, d2, h2, w2)
    xIn = MLX.mean(xIn, axis: 2)

    // Conv branch: conv then space-to-depth
    let xConv = spaceToDepth(conv(input))

    return xConv + xIn
  }
}

/// Depth-to-space upsampling with optional residual connection.
///
/// Used as the decoder upsample block in LTX-2. Applies a 3x3 conv to expand
/// channels, then rearranges channels into spatial/temporal dimensions
/// (depth-to-space). Optionally adds a residual connection using the same
/// depth-to-space rearrangement on the input.
///
/// ## Architecture
///
/// ```
/// Input (B, C, D, H, W)
///   ├─ Conv: CausalConv3d(C → C_out * prod(stride), k=3)
///   ├─ depth_to_space → (B, C_out, D*st, H*sh, W*sw)
///   ├─ [optional] + residual (depth_to_space(input) tiled)
///   ├─ trim first temporal frame if temporal stride > 1
///   └─ Output (B, C_out, D_out, H_out, W_out)
/// ```
public final class LTX2DepthToSpaceUpsample: Module {

  /// 3x3 causal convolution to expand channels for depth-to-space.
  /// Wrapped in LTX2ConvWrapper to match checkpoint key path (conv.conv.weight).
  @ModuleInfo(key: "conv") var conv: LTX2ConvWrapper

  /// Stride as `(temporal, height, width)`.
  public let stride: (Int, Int, Int)

  /// Whether to add a residual connection.
  public let residual: Bool

  /// Output channel reduction factor.
  public let outChannelsReductionFactor: Int

  /// Number of output channels.
  public let outChannels: Int

  /// Creates a DepthToSpace upsample block.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - stride: Upsampling stride as `(temporal, height, width)`.
  ///   - residual: Whether to use residual connection. Default `false`.
  ///   - outChannelsReductionFactor: Channel reduction factor. Default `1`.
  public init(
    inChannels: Int,
    stride: (Int, Int, Int),
    residual: Bool = false,
    outChannelsReductionFactor: Int = 1,
    causalTemporal: Bool = true
  ) {
    self.stride = stride
    self.residual = residual
    self.outChannelsReductionFactor = outChannelsReductionFactor

    let multiplier = stride.0 * stride.1 * stride.2
    self.outChannels = inChannels / outChannelsReductionFactor

    self._conv.wrappedValue = LTX2ConvWrapper(
      inChannels: inChannels, outChannels: outChannels * multiplier,
      causalTemporal: causalTemporal
    )

    super.init()
  }

  // MARK: Streaming decode state (#36)
  /// During a streamed decode the causal first-frame trim after temporal x2
  /// upsampling must apply only to the FIRST chunk of the stream (ComfyUI's
  /// drop_first_conv). The inner conv streams via its own CausalConv3d state.
  public var streamActive = false
  /// Whether the first-chunk trim has already been applied in this stream.
  public var streamDroppedFirst = false

  public func resetStream(active: Bool) {
    streamActive = active
    streamDroppedFirst = false
  }

  /// Rearrange channels into spatial/temporal dimensions.
  ///
  /// `b (c st sh sw) d h w -> b c (d st) (h sh) (w sw)`
  private func depthToSpace(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0)
    let cPacked = x.dim(1)
    let d = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    let (st, sh, sw) = stride
    let c = cPacked / (st * sh * sw)

    // Reshape: (B, C, st, sh, sw, D, H, W)
    var out = x.reshaped(b, c, st, sh, sw, d, h, w)

    // Permute: -> (B, C, D, st, H, sh, W, sw)
    out = out.transposed(0, 1, 5, 2, 6, 3, 7, 4)

    // Reshape: -> (B, C, D*st, H*sh, W*sw)
    out = out.reshaped(b, c, d * st, h * sh, w * sw)

    return out
  }

  /// Applies the upsampling block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, D, H, W)`.
  /// - Returns: Upsampled tensor.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (st, _, _) = stride
    // Streaming + residual would need ComfyUI's deferred-residual exchange
    // cache (conv output lags the raw input under streaming). The v2.3
    // decoder has no residual upsamples, so this combination is unsupported.
    precondition(!(residual && streamActive),
                 "streamed decode does not support residual DepthToSpaceUpsample")

    // Compute residual path if enabled
    var xResidual: MLXArray? = nil
    if residual {
      xResidual = depthToSpace(x)

      // Tile channels to match output
      let numRepeat = (stride.0 * stride.1 * stride.2) / outChannelsReductionFactor
      xResidual = MLX.tiled(xResidual!, repetitions: [1, numRepeat, 1, 1, 1])

      // Remove first temporal frame if temporal upsampling
      if st > 1 {
        xResidual = xResidual![0..., 0..., 1..., 0..., 0...]
      }
    }

    // Conv then depth-to-space
    var out = conv(x)
    out = depthToSpace(out)

    // Remove first frame for causal temporal upsampling. Under streaming this
    // trim belongs to the FIRST emitted chunk only (the offset is a per-clip
    // causal alignment, not per-chunk).
    if st > 1 && out.dim(2) > 0 && (!streamActive || !streamDroppedFirst) {
      out = out[0..., 0..., 1..., 0..., 0...]
      streamDroppedFirst = true
    }

    // Add residual
    if let res = xResidual {
      out = out + res
    }

    return out
  }
}
