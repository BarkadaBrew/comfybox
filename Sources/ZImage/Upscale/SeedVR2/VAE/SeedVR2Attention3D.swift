import Foundation
import MLX
import MLXFast
import MLXNN

/// Spatial self-attention for the SeedVR2 3D video VAE.
///
/// Operates per-frame: the temporal dimension is folded into the batch so that
/// attention is computed independently on each `(H, W)` spatial grid. This avoids
/// the quadratic cost of attending across time while still allowing spatial
/// feature mixing within each frame.
///
/// ## Architecture
///
/// ```
/// Input (B, C, T, H, W)
///   ├─ reshape to (B*T, H*W, C)
///   ├─ GroupNorm(32, C)
///   ├─ Q/K/V linear projections
///   ├─ Scaled dot-product attention (per frame)
///   ├─ Output linear projection
///   ├─ reshape back to (B, C, T, H, W)
///   └─ + residual
/// ```
///
/// The Python reference uses single-head attention (head_dim = C), which we
/// replicate here by inserting a singleton head dimension for the MLXFast SDPA call.
public final class SeedVR2Attention3D: Module {

  /// Group normalization applied in channels-last layout.
  @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm

  /// Query projection.
  @ModuleInfo(key: "to_q") var toQ: Linear

  /// Key projection.
  @ModuleInfo(key: "to_k") var toK: Linear

  /// Value projection.
  @ModuleInfo(key: "to_v") var toV: Linear

  /// Output projection (wrapped in an array to match checkpoint key path `to_out.0`).
  @ModuleInfo(key: "to_out") var toOut: [Linear]

  /// The channel dimension, used to compute the attention scale.
  public let channels: Int

  /// Creates a spatial self-attention module.
  ///
  /// - Parameter channels: Number of input/output channels.
  public init(channels: Int) {
    self.channels = channels

    self._groupNorm.wrappedValue = GroupNorm(
      groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true
    )
    self._toQ.wrappedValue = Linear(channels, channels)
    self._toK.wrappedValue = Linear(channels, channels)
    self._toV.wrappedValue = Linear(channels, channels)
    self._toOut.wrappedValue = [Linear(channels, channels)]

    super.init()
  }

  /// Applies per-frame spatial self-attention.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C, T, H, W)` with spatial attention applied.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    let residual = x

    // (B, C, T, H, W) → (B, T, C, H, W) → (B*T, C, H*W) → (B*T, H*W, C)
    var hidden = x.transposed(0, 2, 1, 3, 4)
    hidden = hidden.reshaped(b * t, c, h * w)
    hidden = hidden.transposed(0, 2, 1)

    // GroupNorm in channels-last: (B*T, H*W, C)
    hidden = groupNorm(hidden.asType(.float32)).asType(x.dtype)

    // Q, K, V projections — each (B*T, H*W, C)
    // Add singleton head dimension → (B*T, 1, H*W, C) for SDPA
    let q = toQ(hidden).expandedDimensions(axis: 1)
    let k = toK(hidden).expandedDimensions(axis: 1)
    let v = toV(hidden).expandedDimensions(axis: 1)

    let scale = pow(Float(c), -0.5)

    // Scaled dot-product attention (single head)
    var attn = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v,
      scale: scale, mask: nil
    )

    // Remove head dimension → (B*T, H*W, C)
    attn = attn.squeezed(axis: 1)

    // Output projection
    hidden = toOut[0](attn)

    // (B*T, H*W, C) → (B*T, C, H*W) → (B, T, C, H, W) → (B, C, T, H, W)
    hidden = hidden.transposed(0, 2, 1)
    hidden = hidden.reshaped(b, t, c, h, w)
    hidden = hidden.transposed(0, 2, 1, 3, 4)

    return hidden + residual
  }
}
