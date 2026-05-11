import Foundation
import MLX
import MLXNN

/// Resampling modes for the Wan 2.1 VAE.
public enum WanResampleMode: String {
  case upsample2d
  case upsample3d
  case downsample2d
  case downsample3d
}

/// Resampling layer for the Wan 2.1 VAE.
///
/// Weight keys: resample.1.weight/bias, optionally time_conv.weight/bias.
public final class WanResample: Module {

  public let mode: WanResampleMode
  public let dim: Int

  @ModuleInfo(key: "resample") var resample: WanResampleSeq
  @ModuleInfo(key: "time_conv") var timeConv: WanCausalConv3d?

  public init(dim: Int, mode: WanResampleMode) {
    self.dim = dim
    self.mode = mode

    switch mode {
    case .downsample2d:
      self._resample.wrappedValue = WanResampleSeq(
        inputChannels: dim, outputChannels: dim,
        kernelSize: 3, stride: 2, padding: 0
      )

    case .downsample3d:
      self._resample.wrappedValue = WanResampleSeq(
        inputChannels: dim, outputChannels: dim,
        kernelSize: 3, stride: 2, padding: 0
      )
      self._timeConv.wrappedValue = WanCausalConv3d(
        inChannels: dim, outChannels: dim,
        kernelSize: (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0)
      )

    case .upsample2d:
      self._resample.wrappedValue = WanResampleSeq(
        inputChannels: dim, outputChannels: dim / 2,
        kernelSize: 3, stride: 1, padding: 1
      )

    case .upsample3d:
      self._resample.wrappedValue = WanResampleSeq(
        inputChannels: dim, outputChannels: dim / 2,
        kernelSize: 3, stride: 1, padding: 1
      )
      self._timeConv.wrappedValue = WanCausalConv3d(
        inChannels: dim, outChannels: dim * 2,
        kernelSize: (3, 1, 1), stride: (1, 1, 1), padding: (1, 0, 0)
      )
    }

    super.init()
  }

  /// Standard forward (no cache) -- used by decoder.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    var t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    var input = x

    switch mode {
    case .upsample2d, .upsample3d:
      if mode == .upsample3d, let tConv = timeConv {
        input = tConv(input)
        input = input.reshaped(b, 2, c, t, h, w)
        let first = input[0..., 0...0, 0..., 0..., 0..., 0...].squeezed(axis: 1)
        let second = input[0..., 1...1, 0..., 0..., 0..., 0...].squeezed(axis: 1)
        let stacked = MLX.stacked([first, second], axis: 3)
        input = stacked.reshaped(b, c, t * 2, h, w)
        t = t * 2
      }

      input = input.transposed(0, 2, 1, 3, 4)
      input = input.reshaped(b * t, c, h, w)
      input = input.transposed(0, 2, 3, 1)

      input = MLX.repeated(input, count: 2, axis: 1)
      input = MLX.repeated(input, count: 2, axis: 2)

      input = resample(input)
      input = input.transposed(0, 3, 1, 2)

      let newC = input.dim(1)
      let newH = input.dim(2)
      let newW = input.dim(3)

      input = input.reshaped(b, t, newC, newH, newW)
      input = input.transposed(0, 2, 1, 3, 4)
      return input

    case .downsample2d, .downsample3d:
      input = input.transposed(0, 2, 1, 3, 4)
      input = input.reshaped(b * t, c, h, w)
      input = input.transposed(0, 2, 3, 1)

      input = MLX.padded(input, widths: [
        IntOrPair((0, 0)),
        IntOrPair((0, 1)),
        IntOrPair((0, 1)),
        IntOrPair((0, 0)),
      ])

      input = resample(input)
      input = input.transposed(0, 3, 1, 2)

      let newC2 = input.dim(1)
      let newH2 = input.dim(2)
      let newW2 = input.dim(3)

      input = input.reshaped(b, t, newC2, newH2, newW2)
      input = input.transposed(0, 2, 1, 3, 4)

      // In non-cached mode, skip time_conv for downsample3d.
      // The Python code only applies time_conv in the cached path.
      // For the decoder, downsample3d is not used anyway.
      return input
    }
  }

  /// Cached forward for chunk-by-chunk encoding.
  ///
  /// Matches Python's Resample.forward() for downsample3d with feat_cache:
  /// - First chunk: cache the spatially-downsampled result, do NOT apply time_conv
  /// - Later chunks: concatenate cached last frame, apply time_conv for temporal downsample
  public func forward(_ x: MLXArray, cache: WanEncoderCache) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    var input = x

    switch mode {
    case .downsample2d:
      // Spatial-only downsample, no cache needed (just pass through)
      return callAsFunction(x)

    case .downsample3d:
      // Step 1: Spatial downsample (same as non-cached)
      input = input.transposed(0, 2, 1, 3, 4)
      input = input.reshaped(b * t, c, h, w)
      input = input.transposed(0, 2, 3, 1)

      input = MLX.padded(input, widths: [
        IntOrPair((0, 0)),
        IntOrPair((0, 1)),
        IntOrPair((0, 1)),
        IntOrPair((0, 0)),
      ])

      input = resample(input)
      input = input.transposed(0, 3, 1, 2)

      let newC = input.dim(1)
      let newH = input.dim(2)
      let newW = input.dim(3)

      input = input.reshaped(b, t, newC, newH, newW)
      input = input.transposed(0, 2, 1, 3, 4)

      // Step 2: Temporal downsample with cache
      let slotIdx = cache.advance()
      guard let tConv = timeConv else { return input }

      if cache.slots[slotIdx] == nil {
        // First chunk: cache the spatially-downsampled result, skip time_conv
        cache.slots[slotIdx] = input
        return input
      } else {
        // Later chunks: concat cached last frame + current, apply time_conv
        let cached = cache.slots[slotIdx]!
        let cachedLast = cached[0..., 0..., (cached.dim(2) - 1)...(cached.dim(2) - 1), 0..., 0...]
        let newCacheX = input[0..., 0..., (input.dim(2) - 1)...(input.dim(2) - 1), 0..., 0...]

        // Concatenate: [cached_last_frame, current_frames]
        let combined = MLX.concatenated([cachedLast, input], axis: 2)

        // Apply time_conv (Conv3d kernel=3, stride=2, padding=0)
        // This is a raw conv, NOT the cached CausalConv3d path.
        // The Python time_conv has CausalConv3d with padding=(0,0,0),
        // which means _padding = 2*0 = 0, so no causal padding at all.
        input = tConv(combined)

        // Update cache with last frame of the spatial output
        cache.slots[slotIdx] = newCacheX
        return input
      }

    case .upsample2d, .upsample3d:
      // Upsample modes don't need encoding cache
      return callAsFunction(x)
    }
  }
}

/// Sequential wrapper for resample Conv2d.
/// In Python: nn.Sequential(ZeroPad2d/Upsample, Conv2d)
/// Conv2d is at index 1. Uses [Module] array for index-based weight keys.
public final class WanResampleSeq: Module {

  public let layers: [Module]

  public init(
    inputChannels: Int,
    outputChannels: Int,
    kernelSize: IntOrPair,
    stride: IntOrPair,
    padding: IntOrPair
  ) {
    self.layers = [
      SiLUPlaceholder(),  // index 0: ZeroPad2d or Upsample (no params)
      Conv2d(
        inputChannels: inputChannels,
        outputChannels: outputChannels,
        kernelSize: kernelSize,
        stride: stride,
        padding: padding
      ),  // index 1: Conv2d
    ]
    super.init()
  }

  /// Forward -- just delegates to Conv2d at index 1.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    return (layers[1] as! Conv2d)(x)
  }
}
