import Foundation
import MLX
import MLXNN

/// Average-pooling downsample for the Wan 2.2 VAE shortcut path.
///
/// Rearranges spatial/temporal elements into the channel dimension via
/// reshape + transpose, then averages groups to reduce to the target
/// output channels. Used in DownBlock's skip connection.
public final class WanAvgDown3D: Module {

  /// Input channels.
  public let inChannels: Int

  /// Output channels.
  public let outChannels: Int

  /// Temporal downsample factor.
  public let factorT: Int

  /// Spatial downsample factor.
  public let factorS: Int

  /// Total factor (factorT * factorS * factorS).
  public let factor: Int

  /// Group size for averaging.
  public let groupSize: Int

  /// Creates an average 3D downsampler.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - factorT: Temporal downsample factor.
  ///   - factorS: Spatial downsample factor. Default `1`.
  public init(inChannels: Int, outChannels: Int, factorT: Int, factorS: Int = 1) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.factorT = factorT
    self.factorS = factorS
    self.factor = factorT * factorS * factorS
    self.groupSize = inChannels * self.factor / outChannels
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x

    // Pad temporal dimension if needed
    let padT = (factorT - input.dim(2) % factorT) % factorT
    if padT > 0 {
      input = MLX.padded(input, widths: [
        IntOrPair((0, 0)),
        IntOrPair((0, 0)),
        IntOrPair((padT, 0)),
        IntOrPair((0, 0)),
        IntOrPair((0, 0)),
      ])
    }

    let b = input.dim(0)
    let c = input.dim(1)
    let t = input.dim(2)
    let h = input.dim(3)
    let w = input.dim(4)

    // Reshape to separate factors
    var out = input.reshaped(
      b, c,
      t / factorT, factorT,
      h / factorS, factorS,
      w / factorS, factorS
    )

    // Transpose to group factors together
    out = out.transposed(0, 1, 3, 5, 7, 2, 4, 6)

    // Reshape to combine factors into channels
    out = out.reshaped(b, c * factor, t / factorT, h / factorS, w / factorS)

    // Group and average
    out = out.reshaped(b, outChannels, groupSize, t / factorT, h / factorS, w / factorS)
    out = MLX.mean(out, axis: 2)

    return out
  }
}

/// Duplicate-and-upsample for the Wan 2.2 VAE shortcut path.
///
/// Expands channels via repeat, then rearranges them into spatial/temporal
/// dimensions. Used in ResidualUpBlock's skip connection.
public final class WanDupUp3D: Module {

  /// Input channels.
  public let inChannels: Int

  /// Output channels.
  public let outChannels: Int

  /// Temporal upsample factor.
  public let factorT: Int

  /// Spatial upsample factor.
  public let factorS: Int

  /// Total factor (factorT * factorS * factorS).
  public let factor: Int

  /// Number of repeats per channel.
  public let repeats: Int

  /// Creates a duplicate-and-upsample 3D module.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - factorT: Temporal upsample factor.
  ///   - factorS: Spatial upsample factor. Default `1`.
  public init(inChannels: Int, outChannels: Int, factorT: Int, factorS: Int = 1) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.factorT = factorT
    self.factorS = factorS
    self.factor = factorT * factorS * factorS
    self.repeats = outChannels * self.factor / inChannels
    super.init()
  }

  /// Applies the upsample operation.
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `(B, C, T, H, W)`.
  ///   - firstChunk: If true and factorT > 1, trims the leading temporal frames.
  /// - Returns: Upsampled tensor.
  public func callAsFunction(_ x: MLXArray, firstChunk: Bool = false) -> MLXArray {
    let b = x.dim(0)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    // Repeat channels
    var out = MLX.repeated(x, count: repeats, axis: 1)

    // Reshape to separate factors
    out = out.reshaped(b, outChannels, factorT, factorS, factorS, t, h, w)

    // Transpose to interleave
    out = out.transposed(0, 1, 5, 2, 6, 3, 7, 4)

    // Reshape to final spatial dimensions
    out = out.reshaped(b, outChannels, t * factorT, h * factorS, w * factorS)

    // Trim leading temporal frames for first chunk
    if firstChunk && factorT > 1 {
      out = out[0..., 0..., (factorT - 1)..., 0..., 0...]
    }

    return out
  }
}

/// Resampling layer for the Wan 2.2 VAE.
///
/// Supports four modes:
/// - `downsample2d`: spatial 2x downsample via strided Conv2d
/// - `downsample3d`: spatial 2x downsample via strided Conv2d (same as 2d in the reference)
/// - `upsample2d`: spatial 2x upsample via nearest-neighbor + Conv2d
/// - `upsample3d`: temporal 2x upsample via CausalConv3d + spatial 2x upsample
public final class WanResample: Module {

  /// Resample mode.
  public let mode: WanResampleMode

  /// Channel dimension.
  public let dim: Int

  /// Spatial resampling convolution.
  @ModuleInfo(key: "resample_conv") var resampleConv: Conv2d

  /// Temporal upsample convolution (only for upsample3d mode).
  @ModuleInfo(key: "time_conv") var timeConv: WanCausalConv3d?

  /// Creates a resampling layer.
  ///
  /// - Parameters:
  ///   - dim: Channel dimension.
  ///   - mode: Resampling mode.
  ///   - upsampleOutDim: Output dimension for upsample modes. If nil, uses dim.
  public init(dim: Int, mode: WanResampleMode, upsampleOutDim: Int? = nil) {
    self.dim = dim
    self.mode = mode

    let outDim = upsampleOutDim ?? dim

    switch mode {
    case .upsample3d:
      // Temporal conv doubles channels, then we split for interleaving
      self._timeConv.wrappedValue = WanCausalConv3d(
        inChannels: dim, outChannels: dim * 2,
        kernelSize: 3, stride: 1, padding: 1
      )
      self._resampleConv.wrappedValue = Conv2d(
        inputChannels: dim, outputChannels: outDim,
        kernelSize: 3, stride: 1, padding: 1
      )

    case .upsample2d:
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

  /// Applies resampling.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Resampled tensor.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var b = x.dim(0)
    var c = x.dim(1)
    var t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    switch mode {
    case .upsample2d, .upsample3d:
      var input = x

      // Temporal upsample if 3d mode
      if mode == .upsample3d, let tConv = timeConv {
        input = tConv(input)
        // Split doubled channels and interleave temporally
        input = input.reshaped(b, 2, c, t, h, w)
        input = input.transposed(0, 2, 3, 1, 4, 5)
        input = input.reshaped(b, c, t * 2, h, w)
        t = t * 2
      }

      // Collapse time into batch: (B, C, T, H, W) → (B*T, C, H, W)
      input = input.transposed(0, 2, 1, 3, 4)
      input = input.reshaped(b * t, c, h, w)

      // Channels-last for Conv2d: (B*T, H, W, C)
      input = input.transposed(0, 2, 3, 1)

      // Spatial 2x upsample via nearest-neighbor repeat
      input = MLX.repeated(input, count: 2, axis: 1)  // H*2
      input = MLX.repeated(input, count: 2, axis: 2)  // W*2

      // Conv2d
      input = resampleConv(input)

      // Back to channels-first: (B*T, H', W', C') → (B*T, C', H', W')
      input = input.transposed(0, 3, 1, 2)

      let newC = input.dim(1)
      let newH = input.dim(2)
      let newW = input.dim(3)

      // Restore time dimension: (B*T, C', H', W') → (B, T, C', H', W') → (B, C', T, H', W')
      input = input.reshaped(b, t, newC, newH, newW)
      input = input.transposed(0, 2, 1, 3, 4)

      return input

    case .downsample2d, .downsample3d:
      // Collapse time into batch
      var input = x.transposed(0, 2, 1, 3, 4)
      input = input.reshaped(b * t, c, h, w)

      // Channels-last for Conv2d
      input = input.transposed(0, 2, 3, 1)

      // Pad asymmetric for stride-2 conv
      input = MLX.padded(input, widths: [
        IntOrPair((0, 0)),
        IntOrPair((0, 1)),
        IntOrPair((0, 1)),
        IntOrPair((0, 0)),
      ])

      input = resampleConv(input)

      // Back to channels-first
      input = input.transposed(0, 3, 1, 2)

      let newC = input.dim(1)
      let newH = input.dim(2)
      let newW = input.dim(3)

      input = input.reshaped(b, t, newC, newH, newW)
      input = input.transposed(0, 2, 1, 3, 4)

      return input
    }
  }
}

/// Resampling modes for the Wan VAE.
public enum WanResampleMode {
  case upsample2d
  case upsample3d
  case downsample2d
  case downsample3d
}
