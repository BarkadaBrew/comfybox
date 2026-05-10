import Foundation
import MLX
import MLXNN

/// RMS normalization for the Wan 2.2 VAE.
///
/// Normalizes input by its L2 norm over the channel dimension, then scales by
/// `sqrt(dim) * weight`. This is the normalization used throughout the Wan VAE
/// in place of standard LayerNorm or GroupNorm.
///
/// ## Formula
///
/// ```
/// x_norm = x / max(||x||_2, eps)
/// output = x_norm * sqrt(dim) * weight
/// ```
///
/// The weight shape depends on the `images` parameter:
/// - `images=true`: weight shape `(dim, 1, 1)` for 4D image tensors `(B, C, H, W)`
/// - `images=false`: weight shape `(dim, 1, 1, 1)` for 5D video tensors `(B, C, T, H, W)`
public final class WanRMSNorm: Module {

  /// Learnable scale weight.
  public var weight: MLXArray

  /// Epsilon for numerical stability.
  public let eps: Float

  /// Scale factor (sqrt(dim)).
  public let scale: Float

  /// Whether this norm operates on images (4D) or video (5D).
  public let images: Bool

  /// Creates a Wan RMS normalization layer.
  ///
  /// - Parameters:
  ///   - dim: Channel dimension size.
  ///   - eps: Epsilon for numerical stability. Default `1e-12`.
  ///   - images: If true, uses 4D weight shape for images; if false, 5D for video. Default `true`.
  public init(dim: Int, eps: Float = 1e-12, images: Bool = true) {
    self.eps = eps
    self.scale = Float(dim).squareRoot()
    self.images = images

    if images {
      self.weight = MLXArray.ones([dim, 1, 1])
    } else {
      self.weight = MLXArray.ones([dim, 1, 1, 1])
    }

    super.init()
  }

  /// Applies RMS normalization.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, H, W)` or `(B, C, T, H, W)`.
  /// - Returns: Normalized tensor with same shape.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Compute L2 norm over channel dimension (axis=1)
    let sumSq = MLX.sum(x * x, axis: 1, keepDims: true)
    let l2Norm = MLX.sqrt(sumSq)
    let denom = MLX.maximum(l2Norm, MLXArray(eps))
    let xNormalized = x / denom

    // Reshape weight for broadcasting
    let w: MLXArray
    if x.ndim == 5 && !images {
      w = weight.reshaped(1, -1, 1, 1, 1)
    } else if x.ndim == 4 && images {
      w = weight.reshaped(1, -1, 1, 1)
    } else if x.ndim == 5 {
      w = weight.reshaped(1, -1, 1, 1, 1)
    } else {
      w = weight.reshaped(1, -1, 1, 1)
    }

    return xNormalized * scale * w
  }
}

/// GroupNorm with 32 groups for the Wan 2.2 VAE.
///
/// Wraps MLX's built-in GroupNorm with groups fixed to 32, matching the
/// standard configuration used in the Wan VAE attention blocks.
public final class WanGroupNorm32: Module {

  /// The underlying GroupNorm layer.
  @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm

  /// Number of channels.
  public let channels: Int

  /// Creates a GroupNorm with 32 groups.
  ///
  /// - Parameter channels: Number of channels. Must be divisible by 32.
  public init(channels: Int) {
    precondition(channels % 32 == 0, "channels (\(channels)) must be divisible by 32")
    self.channels = channels
    self._groupNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels)
    super.init()
  }

  /// Applies group normalization.
  ///
  /// MLX GroupNorm expects channels-last format, so this method handles the
  /// transpose from BCTHW to channels-last and back.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Normalized tensor with same shape.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // GroupNorm in MLX expects channels-last: (B, ..., C)
    // Input is (B, C, T, H, W), transpose to (B, T, H, W, C)
    let transposed = x.transposed(0, 2, 3, 4, 1)

    // Reshape to (B*T, H*W, C) for GroupNorm
    let b = transposed.dim(0)
    let t = transposed.dim(1)
    let h = transposed.dim(2)
    let w = transposed.dim(3)
    let c = transposed.dim(4)

    let flat = transposed.reshaped(b * t, h * w, c)
    let normed = groupNorm(flat)

    // Reshape back and transpose to BCTHW
    let unflat = normed.reshaped(b, t, h, w, c)
    return unflat.transposed(0, 4, 1, 2, 3)
  }
}
