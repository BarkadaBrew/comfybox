import Foundation
import MLX
import MLXNN

/// Single MLP block: Linear → SiLU → Linear.
///
/// Used inside the Approximator's residual stack.
/// Weight keys: `layers.N.in_layer.{weight,bias}`, `layers.N.out_layer.{weight,bias}`
final class ChromaMLPEmbedder: Module {
  @ModuleInfo(key: "in_layer") var inLayer: Linear
  @ModuleInfo(key: "out_layer") var outLayer: Linear

  init(inDim: Int, hiddenDim: Int) {
    self._inLayer.wrappedValue = Linear(inDim, hiddenDim, bias: true)
    self._outLayer.wrappedValue = Linear(hiddenDim, hiddenDim, bias: true)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let origDtype = x.dtype
    return outLayer(silu(inLayer(x))).asType(origDtype)
  }
}

/// Chroma Approximator — replaces the 3.3B parameter AdaLN modulation from FLUX.
///
/// Architecture: in_proj → 5 × (RMSNorm + MLPEmbedder + residual) → out_proj
///
/// Input: [B, 344, 64] (timestep + guidance + modulation index embeddings)
/// Output: [B, 344, 3072] (modulation vectors for all transformer blocks)
///
/// Weight keys: `distilled_guidance_layer.{in_proj,out_proj,layers.N,norms.N}`
public final class ChromaApproximator: Module {
  @ModuleInfo(key: "in_proj") var inProj: Linear
  @ModuleInfo(key: "out_proj") var outProj: Linear
  @ModuleInfo(key: "layers") var layers: [ChromaMLPEmbedder]
  @ModuleInfo(key: "norms") var norms: [RMSNorm]

  public init(inDim: Int, outDim: Int, hiddenDim: Int, nLayers: Int = 5) {
    self._inProj.wrappedValue = Linear(inDim, hiddenDim, bias: true)
    self._outProj.wrappedValue = Linear(hiddenDim, outDim, bias: true)

    var mlpLayers: [ChromaMLPEmbedder] = []
    var normLayers: [RMSNorm] = []
    for _ in 0..<nLayers {
      mlpLayers.append(ChromaMLPEmbedder(inDim: hiddenDim, hiddenDim: hiddenDim))
      normLayers.append(RMSNorm(dimensions: hiddenDim, eps: 1e-6))
    }
    self._layers.wrappedValue = mlpLayers
    self._norms.wrappedValue = normLayers
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let origDtype = x.dtype
    var h = inProj(x)

    for (layer, norm) in zip(layers, norms) {
      h = h + layer(norm(h))
    }

    return outProj(h).asType(origDtype)
  }
}
