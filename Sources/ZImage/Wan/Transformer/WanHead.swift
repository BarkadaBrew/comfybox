import Foundation
import MLX
import MLXRandom
import MLXNN

/// Output projection head for the Wan 2.2 transformer.
///
/// Applies modulated normalization and projects to output patch features.
///
/// ## Architecture
/// ```
/// norm = WanLayerNorm(dim, elementwise_affine=false)
/// head = Linear(dim, out_dim * prod(patch_size))
/// modulation = Parameter([1, 2, dim])  -- shift and scale
/// ```
///
/// Weight keys:
/// ```
/// head.head.weight  [out_dim * prod(patch_size), dim]
/// head.head.bias    [out_dim * prod(patch_size)]
/// head.modulation   [1, 2, dim]
/// ```
public final class WanHead: Module {

  @ModuleInfo(key: "norm") var norm: WanLayerNorm
  @ModuleInfo(key: "head") var head: Linear

  /// Modulation parameter `[1, 2, dim]` for shift and scale.
  public var modulation: MLXArray

  public let dim: Int
  public let outFeatures: Int

  /// Creates the output head.
  ///
  /// - Parameters:
  ///   - dim: Model dimension.
  ///   - outDim: Output channels (latent channels).
  ///   - patchSize: 3D patch dimensions `(t, h, w)`.
  ///   - eps: Normalization epsilon.
  public init(dim: Int, outDim: Int, patchSize: (Int, Int, Int), eps: Float = 1e-6) {
    self.dim = dim
    self.outFeatures = outDim * patchSize.0 * patchSize.1 * patchSize.2

    self._norm.wrappedValue = WanLayerNorm(dim: dim, eps: eps, elementwiseAffine: false)
    self._head.wrappedValue = Linear(dim, outFeatures)
    self.modulation = MLXRandom.normal([1, 2, dim]) / MLXArray(Float(dim).squareRoot())

    super.init()
  }

  /// Forward pass.
  ///
  /// - Parameters:
  ///   - x: Input tensor `[B, seqLen, dim]`.
  ///   - e: Time embedding `[B, seqLen, dim]` (float32).
  /// - Returns: Output tensor `[B, seqLen, out_dim * prod(patch_size)]`.
  public func callAsFunction(_ x: MLXArray, e: MLXArray) -> MLXArray {
    // modulation: [1, 2, dim] -> [1, 1, 2, dim] + e: [B, seqLen, 1, dim] -> [B, seqLen, 2, dim]
    let eExp = e.expandedDimensions(axis: 2)  // [B, seqLen, 1, dim]
    let modBroadcast = modulation.expandedDimensions(axis: 0)  // [1, 1, 2, dim]
    let eMod = (modBroadcast + eExp).asType(.float32)

    // Split into shift and scale
    let shift = eMod[0..., 0..., 0...0, 0...].squeezed(axis: 2)  // [B, seqLen, dim]
    let scale = eMod[0..., 0..., 1...1, 0...].squeezed(axis: 2)  // [B, seqLen, dim]

    // Apply modulated norm + projection (all in float32)
    let normed = norm(x).asType(.float32)
    let modulated = normed * (1 + scale) + shift
    let output = head(modulated)
    return output
  }
}
