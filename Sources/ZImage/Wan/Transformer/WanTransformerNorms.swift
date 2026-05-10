import Foundation
import MLX
import MLXNN

/// RMS normalization for the Wan 2.2 transformer (different from VAE version).
///
/// Uses rsqrt(mean(x^2) + eps) normalization, matching the Python WanRMSNorm.
/// Applied to q and k in self-attention and cross-attention.
///
/// Weight key: ``weight`` (not ``gamma``).
///
/// ## Computation
/// ```
/// _norm(x) = x * rsqrt(mean(x^2, dim=-1) + eps)
/// forward(x) = _norm(x.float()) * weight
/// ```
public final class WanRMSNorm: Module {

  /// Learnable scale parameter.
  public var weight: MLXArray

  /// Normalization epsilon.
  public let eps: Float

  /// Feature dimension.
  public let dim: Int

  /// Creates an RMS normalization layer.
  ///
  /// - Parameters:
  ///   - dim: Feature dimension.
  ///   - eps: Epsilon for numerical stability. Default 1e-5.
  public init(dim: Int, eps: Float = 1e-5) {
    self.dim = dim
    self.eps = eps
    self.weight = MLXArray.ones([dim])
    super.init()
  }

  /// Applies RMS normalization.
  ///
  /// - Parameter x: Input tensor of shape `[..., dim]`.
  /// - Returns: Normalized tensor of same shape.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let normalized = xf * MLX.rsqrt(MLX.mean(xf * xf, axis: -1, keepDims: true) + MLXArray(eps))
    return normalized.asType(x.dtype) * weight
  }
}

/// Layer normalization for the Wan 2.2 transformer.
///
/// Wraps MLX LayerNorm. When `elementwiseAffine` is false, has no learnable
/// parameters (norm1, norm2 in blocks). When true, has weight and bias (norm3).
///
/// Always casts to float32 for normalization, then casts back.
public final class WanLayerNorm: Module {

  /// Feature dimension.
  public let dim: Int

  /// Epsilon for normalization.
  public let eps: Float

  /// Whether this norm has learnable parameters.
  public let elementwiseAffine: Bool

  /// Learnable scale (only when elementwiseAffine=true).
  public var weight: MLXArray?

  /// Learnable bias (only when elementwiseAffine=true).
  public var bias: MLXArray?

  /// Creates a layer normalization module.
  ///
  /// - Parameters:
  ///   - dim: Feature dimension.
  ///   - eps: Epsilon for numerical stability.
  ///   - elementwiseAffine: Whether to include learnable weight and bias.
  public init(dim: Int, eps: Float = 1e-6, elementwiseAffine: Bool = false) {
    self.dim = dim
    self.eps = eps
    self.elementwiseAffine = elementwiseAffine

    if elementwiseAffine {
      self.weight = MLXArray.ones([dim])
      self.bias = MLXArray.zeros([dim])
    }
    super.init()
  }

  /// Applies layer normalization.
  ///
  /// - Parameter x: Input tensor of shape `[..., dim]`.
  /// - Returns: Normalized tensor of same shape.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let mean = MLX.mean(xf, axis: -1, keepDims: true)
    let variance = MLX.mean((xf - mean) * (xf - mean), axis: -1, keepDims: true)
    var normalized = (xf - mean) * MLX.rsqrt(variance + MLXArray(eps))

    if let w = weight, let b = bias {
      normalized = normalized * w + b
    }

    return normalized.asType(x.dtype)
  }
}

/// Sinusoidal positional embedding for diffusion timesteps.
///
/// Produces [cos(theta), sin(theta)] embeddings matching the Python implementation:
/// ```python
/// sinusoid = outer(position, pow(10000, -arange(half)/half))
/// return cat([cos(sinusoid), sin(sinusoid)], dim=1)
/// ```
///
/// - Parameters:
///   - dim: Embedding dimension (must be even). First half = cos, second half = sin.
///   - position: Position values, shape `[N]`.
/// - Returns: Embeddings of shape `[N, dim]`.
public func sinusoidalEmbedding1D(dim: Int, position: MLXArray) -> MLXArray {
  precondition(dim % 2 == 0, "sinusoidal embedding dim must be even")
  let half = dim / 2

  // Compute frequencies: 10000^(-k/half) for k in 0..<half
  let posFloat = position.asType(.float32)
  let exponents = MLXArray(Array(0..<half).map { Float($0) / Float(half) })
  let freqs = MLX.pow(MLXArray(Float(10000.0)), -exponents)

  // Outer product: [N, half]
  let flat = posFloat.reshaped(-1)
  let sinusoid = MLX.matmul(flat.reshaped(-1, 1), freqs.reshaped(1, -1))

  // [cos, sin] concatenation
  let result = MLX.concatenated([MLX.cos(sinusoid), MLX.sin(sinusoid)], axis: 1)
  return result
}
