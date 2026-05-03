import Foundation
import MLX
import MLXFast
import MLXNN

/// Self-attention block used in the Flux2 VAE mid-block.
///
/// Applies group normalization, then single-head scaled dot-product attention
/// over the spatial dimensions, followed by a linear projection. Connected
/// by a residual shortcut.
///
/// All operations use NHWC layout internally.
///
/// ## Architecture
///
/// ```
/// Input (B, H, W, C)
///   |-- GroupNorm(32, C)
///   |-- to_q, to_k, to_v: Linear(C -> C)
///   |-- reshape to (B, 1, H*W, C) -> scaled_dot_product_attention
///   |-- to_out: Linear(C -> C)
///   |-- + residual
///   --> Output (B, H, W, C)
/// ```
public final class Flux2VAEAttention: Module {

  @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: Linear

  /// Creates a Flux2 VAE attention block.
  ///
  /// - Parameters:
  ///   - channels: Number of input/output channels.
  ///   - groups: Number of groups for GroupNorm. Default `32`.
  ///   - eps: Epsilon for GroupNorm. Default `1e-6`.
  public init(channels: Int, groups: Int = 32, eps: Float = 1e-6) {
    self._groupNorm.wrappedValue = GroupNorm(
      groupCount: groups, dimensions: channels, eps: eps, pytorchCompatible: true)
    self._toQ.wrappedValue = Linear(channels, channels)
    self._toK.wrappedValue = Linear(channels, channels)
    self._toV.wrappedValue = Linear(channels, channels)
    self._toOut.wrappedValue = Linear(channels, channels)
    super.init()
  }

  /// Applies self-attention over spatial dimensions.
  ///
  /// - Parameter x: Input tensor in NHWC layout `(B, H, W, C)`.
  /// - Returns: Output tensor in NHWC layout `(B, H, W, C)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (batch, height, width, channels) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])

    let normed = groupNorm(x.asType(.float32)).asType(x.dtype)

    // Project to Q, K, V and reshape for single-head attention
    // Shape: (B, H*W, C) -> (B, H*W, 1, C) -> (B, 1, H*W, C)
    var q = toQ(normed).reshaped(batch, height * width, 1, channels)
    var k = toK(normed).reshaped(batch, height * width, 1, channels)
    var v = toV(normed).reshaped(batch, height * width, 1, channels)

    q = q.transposed(0, 2, 1, 3)
    k = k.transposed(0, 2, 1, 3)
    v = v.transposed(0, 2, 1, 3)

    let scale = 1.0 / sqrt(Float(channels))

    let attended = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v, scale: scale, mask: nil)

    // Reshape back to spatial: (B, 1, H*W, C) -> (B, H, W, C)
    let reshaped = attended.transposed(0, 2, 1, 3).reshaped(batch, height, width, channels)
    let projected = toOut(reshaped)

    return x + projected
  }
}
