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

      if mode == .downsample3d, let tConv = timeConv {
        input = tConv(input)
      }
      return input
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
