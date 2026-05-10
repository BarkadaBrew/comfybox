import Foundation
import MLX
import MLXNN

/// 3D encoder for the Wan 2.1 VAE.
///
/// Input (B, 3, T, H, W) -> Output (B, 32, T/4, H/8, W/8)
public final class WanEncoder3d: Module {

  @ModuleInfo(key: "conv1") var conv1: WanCausalConv3d
  @ModuleInfo(key: "downsamples") var downsamples: WanSequentialLayers
  @ModuleInfo(key: "middle") var middle: WanMiddleLayers
  @ModuleInfo(key: "head") var head: WanEncoderHead

  public init(
    dim: Int = 96,
    zDim: Int = 16,
    dimMult: [Int] = [1, 2, 4, 4],
    numResBlocks: Int = 2,
    temporalDownsample: [Bool] = [false, true, true]
  ) {
    let dims = ([1] + dimMult).map { dim * $0 }

    self._conv1.wrappedValue = WanCausalConv3d(
      inChannels: 3, outChannels: dims[0], kernelSize: 3, padding: 1
    )

    var layers: [Module] = []
    for i in 0..<dimMult.count {
      let inDim = dims[i]
      let outDim = dims[i + 1]
      var currentIn = inDim
      for _ in 0..<numResBlocks {
        layers.append(WanResidualBlock(inDim: currentIn, outDim: outDim))
        currentIn = outDim
      }
      if i != dimMult.count - 1 {
        let mode: WanResampleMode = temporalDownsample[i] ? .downsample3d : .downsample2d
        layers.append(WanResample(dim: outDim, mode: mode))
      }
    }

    self._downsamples.wrappedValue = WanSequentialLayers(layers: layers)
    let lastDim = dims.last!
    self._middle.wrappedValue = WanMiddleLayers(dim: lastDim)
    self._head.wrappedValue = WanEncoderHead(dim: lastDim, zDim: zDim * 2)

    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = conv1(x)
    h = downsamples(h)
    h = middle(h)
    h = head(h)
    return h
  }
}
