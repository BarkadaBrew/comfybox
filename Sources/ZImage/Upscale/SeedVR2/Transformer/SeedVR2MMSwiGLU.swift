import Foundation
import MLX
import MLXNN

/// Multi-modal feed-forward wrapper for the SeedVR2 transformer.
///
/// Supports both SwiGLU (3B) and GELU (7B) MLP variants.
/// The SeedVR2SwiGLUMLP class handles both modes internally.
public final class SeedVR2MMSwiGLU: Module {

  public let sharedWeights: Bool
  public let isLastLayer: Bool

  @ModuleInfo(key: "all") var mlpAll: SeedVR2SwiGLUMLP?
  @ModuleInfo(key: "vid") var mlpVid: SeedVR2SwiGLUMLP?
  @ModuleInfo(key: "txt") var mlpTxt: SeedVR2SwiGLUMLP?

  public init(
    vidDim: Int = 2560,
    txtDim: Int = 2560,
    expandRatio: Int = 4,
    sharedWeights: Bool = false,
    isLastLayer: Bool = false,
    mlpType: SeedVR2MLPType = .swiglu
  ) {
    self.sharedWeights = sharedWeights
    self.isLastLayer = isLastLayer

    if sharedWeights {
      self._mlpAll.wrappedValue = SeedVR2SwiGLUMLP(dim: vidDim, expandRatio: expandRatio, mlpType: mlpType)
    } else {
      self._mlpVid.wrappedValue = SeedVR2SwiGLUMLP(dim: vidDim, expandRatio: expandRatio, mlpType: mlpType)
      if !isLastLayer {
        self._mlpTxt.wrappedValue = SeedVR2SwiGLUMLP(dim: txtDim, expandRatio: expandRatio, mlpType: mlpType)
      }
    }

    super.init()
  }

  public func callAsFunction(_ vid: MLXArray, _ txt: MLXArray) -> (MLXArray, MLXArray) {
    if sharedWeights {
      let vidOut = mlpAll!(vid)
      let txtOut = isLastLayer ? txt : mlpAll!(txt)
      return (vidOut, txtOut)
    } else {
      let vidOut = mlpVid!(vid)
      let txtOut = isLastLayer ? txt : mlpTxt!(txt)
      return (vidOut, txtOut)
    }
  }
}
