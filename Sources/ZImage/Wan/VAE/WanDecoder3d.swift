import Foundation
import MLX
import MLXNN

/// 3D decoder for the Wan 2.1 VAE.
///
/// Input (B, 16, T/4, H/8, W/8) -> Output (B, 3, T, H, W)
public final class WanDecoder3d: Module {

  @ModuleInfo(key: "conv1") var conv1: WanCausalConv3d
  @ModuleInfo(key: "middle") var middle: WanMiddleLayers
  @ModuleInfo(key: "upsamples") var upsamples: WanSequentialLayers
  @ModuleInfo(key: "head") var head: WanDecoderHead

  public init(
    dim: Int = 96,
    zDim: Int = 16,
    dimMult: [Int] = [1, 2, 4, 4],
    numResBlocks: Int = 2,
    temporalUpsample: [Bool] = [true, true, false]
  ) {
    let reversedMult = [dimMult.last!] + dimMult.reversed()
    let dims = reversedMult.map { dim * $0 }

    self._conv1.wrappedValue = WanCausalConv3d(
      inChannels: zDim, outChannels: dims[0], kernelSize: 3, padding: 1
    )
    self._middle.wrappedValue = WanMiddleLayers(dim: dims[0])

    var layers: [Module] = []
    for i in 0..<dimMult.count {
      var inDim = dims[i]
      let outDim = dims[i + 1]
      if i == 1 || i == 2 || i == 3 {
        inDim = inDim / 2
      }
      var currentIn = inDim
      for _ in 0..<(numResBlocks + 1) {
        layers.append(WanResidualBlock(inDim: currentIn, outDim: outDim))
        currentIn = outDim
      }
      if i != dimMult.count - 1 {
        let mode: WanResampleMode = temporalUpsample[i] ? .upsample3d : .upsample2d
        layers.append(WanResample(dim: outDim, mode: mode))
      }
    }

    self._upsamples.wrappedValue = WanSequentialLayers(layers: layers)
    let lastDim = dims.last!
    self._head.wrappedValue = WanDecoderHead(dim: lastDim, outChannels: 3)

    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = conv1(x)
    h = middle(h)
    h = upsamples(h)
    h = head(h)
    return h
  }
}
