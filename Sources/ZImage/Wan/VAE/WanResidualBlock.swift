import Foundation
import MLX
import MLXNN

/// Residual block for the Wan 2.1 VAE.
///
/// Weight keys: residual.0.gamma, residual.2.weight/bias,
/// residual.3.gamma, residual.6.weight/bias,
/// optionally shortcut.weight/bias.
public final class WanResidualBlock: Module {

  @ModuleInfo(key: "residual") var residual: WanResidualPath
  @ModuleInfo(key: "shortcut") var shortcut: WanCausalConv3d?

  public let hasShortcut: Bool

  public init(inDim: Int, outDim: Int) {
    self.hasShortcut = (inDim != outDim)
    self._residual.wrappedValue = WanResidualPath(inDim: inDim, outDim: outDim)
    if inDim != outDim {
      self._shortcut.wrappedValue = WanCausalConv3d(
        inChannels: inDim, outChannels: outDim, kernelSize: 1, padding: 0
      )
    }
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let h: MLXArray
    if let sc = shortcut {
      h = sc(x)
    } else {
      h = x
    }
    let out = residual(x)
    return out + h
  }
}

/// The residual path matching PyTorch Sequential index-based weight keys.
/// Uses [Module] array so indices 0-6 map to weight keys.
/// Index 0: norm1, 1: SiLU, 2: conv1, 3: norm2, 4: SiLU, 5: Dropout, 6: conv2
public final class WanResidualPath: Module {

  public let layers: [Module]

  public init(inDim: Int, outDim: Int) {
    self.layers = [
      WanVAENorm(dim: inDim, images: false),     // 0
      SiLUPlaceholder(),                          // 1
      WanCausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: 3, padding: 1),  // 2
      WanVAENorm(dim: outDim, images: false),     // 3
      SiLUPlaceholder(),                          // 4
      SiLUPlaceholder(),                          // 5 (dropout)
      WanCausalConv3d(inChannels: outDim, outChannels: outDim, kernelSize: 3, padding: 1), // 6
    ]
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = (layers[0] as! WanVAENorm)(x)
    h = silu(h)
    h = (layers[2] as! WanCausalConv3d)(h)
    h = (layers[3] as! WanVAENorm)(h)
    h = silu(h)
    h = (layers[6] as! WanCausalConv3d)(h)
    return h
  }
}
