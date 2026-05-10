import Foundation
import MLX
import MLXFast
import MLXNN

/// Self-attention block for the Wan 2.2 VAE.
///
/// Applies single-head self-attention over spatial dimensions, independently
/// per temporal frame. Uses RMSNorm (image mode) before attention and Conv2d
/// for QKV projection and output projection.
///
/// ## Architecture
///
/// ```
/// Input (B, C, T, H, W)
///   ├─ Reshape → (B*T, C, H, W)
///   ├─ RMSNorm(images=true)
///   ├─ Conv2d(C → 3C, k=1) → Q, K, V
///   ├─ Reshape → (B*T, 1, H*W, C)  (single-head)
///   ├─ ScaledDotProductAttention
///   ├─ Reshape → (B*T, C, H, W)
///   ├─ Conv2d(C → C, k=1) (output projection)
///   ├─ Reshape → (B, C, T, H, W)
///   ├─ + identity
///   └─ Output (B, C, T, H, W)
/// ```
public final class WanAttentionBlock: Module {

  /// Channel dimension.
  public let dim: Int

  /// RMS normalization (image mode).
  @ModuleInfo(key: "norm") var norm: WanRMSNorm

  /// QKV projection via 1x1 Conv2d.
  @ModuleInfo(key: "to_qkv") var toQKV: Conv2d

  /// Output projection via 1x1 Conv2d.
  @ModuleInfo(key: "proj") var proj: Conv2d

  /// Creates a Wan attention block.
  ///
  /// - Parameter dim: Channel dimension.
  public init(dim: Int) {
    self.dim = dim

    self._norm.wrappedValue = WanRMSNorm(dim: dim, images: true)
    self._toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
    self._proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)

    super.init()
  }

  /// Applies self-attention.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let identity = x
    let batchSize = x.dim(0)
    let channels = x.dim(1)
    let time = x.dim(2)
    let height = x.dim(3)
    let width = x.dim(4)

    // Collapse batch and time: (B, C, T, H, W) → (B*T, C, H, W)
    var h = x.transposed(0, 2, 1, 3, 4)
    h = h.reshaped(batchSize * time, channels, height, width)

    // RMSNorm (image mode on 4D tensor)
    h = norm(h)

    // Conv2d expects channels-last: (B*T, H, W, C)
    h = h.transposed(0, 2, 3, 1)

    // QKV projection
    var qkv = toQKV(h)

    // Transpose back: (B*T, H, W, 3C) → (B*T, 3C, H, W)
    qkv = qkv.transposed(0, 3, 1, 2)

    // Reshape for attention: (B*T, 1, H*W, 3C)
    qkv = qkv.reshaped(batchSize * time, 1, channels * 3, height * width)
    qkv = qkv.transposed(0, 1, 3, 2)

    // Split Q, K, V: each (B*T, 1, H*W, C)
    let parts = MLX.split(qkv, parts: 3, axis: 3)
    let q = parts[0]
    let k = parts[1]
    let v = parts[2]

    // Scaled dot product attention
    let scale = 1.0 / Float(channels).squareRoot()
    h = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v, scale: scale, mask: nil
    )

    // Reshape back: (B*T, 1, H*W, C) → (B*T, C, H, W)
    h = h.reshaped(batchSize * time, height * width, channels)
    h = h.transposed(0, 2, 1)
    h = h.reshaped(batchSize * time, channels, height, width)

    // Output projection: transpose to channels-last for Conv2d
    h = h.transposed(0, 2, 3, 1)
    h = proj(h)
    h = h.transposed(0, 3, 1, 2)

    // Reshape back to 5D: (B*T, C, H, W) → (B, T, C, H, W) → (B, C, T, H, W)
    h = h.reshaped(batchSize, time, channels, height, width)
    h = h.transposed(0, 2, 1, 3, 4)

    return h + identity
  }
}
