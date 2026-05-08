import Foundation
import MLX
import MLXNN

/// ESRGAN residual dense block with five densely-connected 3x3 convolutions.
public final class ResidualDenseBlock: Module {
  @ModuleInfo(key: "conv1") public var conv1: Conv2d
  @ModuleInfo(key: "conv2") public var conv2: Conv2d
  @ModuleInfo(key: "conv3") public var conv3: Conv2d
  @ModuleInfo(key: "conv4") public var conv4: Conv2d
  @ModuleInfo(key: "conv5") public var conv5: Conv2d

  public let numFeat: Int
  public let numGrowCh: Int
  public let residualScale: Float

  public init(numFeat: Int = 64, numGrowCh: Int = 32, residualScale: Float = 0.2) {
    self.numFeat = numFeat
    self.numGrowCh = numGrowCh
    self.residualScale = residualScale

    self._conv1.wrappedValue = Conv2d(
      inputChannels: numFeat,
      outputChannels: numGrowCh,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._conv2.wrappedValue = Conv2d(
      inputChannels: numFeat + numGrowCh,
      outputChannels: numGrowCh,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._conv3.wrappedValue = Conv2d(
      inputChannels: numFeat + 2 * numGrowCh,
      outputChannels: numGrowCh,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._conv4.wrappedValue = Conv2d(
      inputChannels: numFeat + 3 * numGrowCh,
      outputChannels: numGrowCh,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._conv5.wrappedValue = Conv2d(
      inputChannels: numFeat + 4 * numGrowCh,
      outputChannels: numFeat,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let x1 = MLXNN.leakyRelu(conv1(x), negativeSlope: 0.2)
    let x2Input = MLX.concatenated([x, x1], axis: -1)
    let x2 = MLXNN.leakyRelu(conv2(x2Input), negativeSlope: 0.2)
    let x3Input = MLX.concatenated([x, x1, x2], axis: -1)
    let x3 = MLXNN.leakyRelu(conv3(x3Input), negativeSlope: 0.2)
    let x4Input = MLX.concatenated([x, x1, x2, x3], axis: -1)
    let x4 = MLXNN.leakyRelu(conv4(x4Input), negativeSlope: 0.2)
    let x5Input = MLX.concatenated([x, x1, x2, x3, x4], axis: -1)
    let x5 = conv5(x5Input)
    return x5 * residualScale + x
  }
}

/// Residual-in-residual dense block: three RDBs with an outer residual scale.
public final class RRDB: Module {
  @ModuleInfo(key: "rdb1") public var rdb1: ResidualDenseBlock
  @ModuleInfo(key: "rdb2") public var rdb2: ResidualDenseBlock
  @ModuleInfo(key: "rdb3") public var rdb3: ResidualDenseBlock

  public let residualScale: Float

  public init(numFeat: Int = 64, numGrowCh: Int = 32, residualScale: Float = 0.2) {
    self.residualScale = residualScale
    self._rdb1.wrappedValue = ResidualDenseBlock(numFeat: numFeat, numGrowCh: numGrowCh)
    self._rdb2.wrappedValue = ResidualDenseBlock(numFeat: numFeat, numGrowCh: numGrowCh)
    self._rdb3.wrappedValue = ResidualDenseBlock(numFeat: numFeat, numGrowCh: numGrowCh)
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let out = rdb3(rdb2(rdb1(x)))
    return out * residualScale + x
  }
}

/// BasicSR-style ESRGAN RRDBNet in MLX native NHWC layout.
public final class RRDBNet: Module {
  @ModuleInfo(key: "conv_first") public var convFirst: Conv2d
  @ModuleInfo(key: "body") public var body: [RRDB]
  @ModuleInfo(key: "conv_body") public var convBody: Conv2d
  @ModuleInfo(key: "conv_up1") public var convUp1: Conv2d
  @ModuleInfo(key: "conv_up2") public var convUp2: Conv2d
  @ModuleInfo(key: "conv_hr") public var convHR: Conv2d
  @ModuleInfo(key: "conv_last") public var convLast: Conv2d

  public let config: ESRGANConfig

  public init(config: ESRGANConfig = .ultraSharp4x) {
    precondition(config.scale == 4, "RRDBNet currently implements the 4x ESRGAN upsampling head")
    self.config = config

    self._convFirst.wrappedValue = Conv2d(
      inputChannels: config.numInCh,
      outputChannels: config.numFeat,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    self._body.wrappedValue = (0..<config.numBlock).map { _ in
      RRDB(numFeat: config.numFeat, numGrowCh: config.numGrowCh)
    }

    self._convBody.wrappedValue = Conv2d(
      inputChannels: config.numFeat,
      outputChannels: config.numFeat,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._convUp1.wrappedValue = Conv2d(
      inputChannels: config.numFeat,
      outputChannels: config.numFeat,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._convUp2.wrappedValue = Conv2d(
      inputChannels: config.numFeat,
      outputChannels: config.numFeat,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._convHR.wrappedValue = Conv2d(
      inputChannels: config.numFeat,
      outputChannels: config.numFeat,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._convLast.wrappedValue = Conv2d(
      inputChannels: config.numFeat,
      outputChannels: config.numOutCh,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var feat = convFirst(x)
    var bodyFeat = feat
    for block in body {
      bodyFeat = block(bodyFeat)
    }
    bodyFeat = convBody(bodyFeat)
    feat = feat + bodyFeat
    feat = MLXNN.leakyRelu(convUp1(Self.nearestUpsample2x(feat)), negativeSlope: 0.2)
    feat = MLXNN.leakyRelu(convUp2(Self.nearestUpsample2x(feat)), negativeSlope: 0.2)
    let hr = MLXNN.leakyRelu(convHR(feat), negativeSlope: 0.2)
    return convLast(hr)
  }

  public static func nearestUpsample2x(_ x: MLXArray) -> MLXArray {
    precondition(x.ndim == 4, "Expected NHWC tensor")
    var out = MLX.repeated(x, count: 2, axis: 1)
    out = MLX.repeated(out, count: 2, axis: 2)
    return out
  }
}
