import Foundation
import MLX
import MLXNN

/// RMS normalization for the Wan 2.1 VAE (RMS_norm from source).
///
/// Uses L2 normalization (F.normalize), not rsqrt(mean(x^2)).
/// Multiplies by sqrt(dim) as scale factor.
/// Learnable parameter is called ``gamma`` (not weight).
public final class WanVAENorm: Module {

  public var gamma: MLXArray
  public let scale: Float
  public let images: Bool
  public let dim: Int

  public init(dim: Int, images: Bool = true) {
    self.dim = dim
    self.scale = Float(dim).squareRoot()
    self.images = images

    if images {
      self.gamma = MLXArray.ones([dim, 1, 1])
    } else {
      self.gamma = MLXArray.ones([dim, 1, 1, 1])
    }

    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // F.normalize(x, dim=1): L2 normalize along channel dimension
    let normSq = MLX.sum(x * x, axis: 1, keepDims: true)
    let norm = MLX.sqrt(normSq + 1e-12)
    let normalized = x / norm

    let g: MLXArray
    if x.ndim == 5 {
      g = gamma.reshaped(1, dim, 1, 1, 1)
    } else {
      g = gamma.reshaped(1, dim, 1, 1)
    }

    return normalized * scale * g
  }
}
