import Foundation
import MLX
import MLXFast
import MLXNN

/// Spatial self-attention block for the Wan 2.1 VAE.
public final class WanAttentionBlock: Module {

  public let dim: Int

  @ModuleInfo(key: "norm") var norm: WanVAENorm
  @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
  @ModuleInfo(key: "proj") var proj: Conv2d

  public init(dim: Int) {
    self.dim = dim
    self._norm.wrappedValue = WanVAENorm(dim: dim, images: true)
    self._toQKV.wrappedValue = Conv2d(
      inputChannels: dim, outputChannels: dim * 3, kernelSize: 1
    )
    self._proj.wrappedValue = Conv2d(
      inputChannels: dim, outputChannels: dim, kernelSize: 1
    )
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let identity = x
    let b = x.dim(0)
    let c = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    var out = x.transposed(0, 2, 1, 3, 4)
    out = out.reshaped(b * t, c, h, w)
    out = norm(out)
    out = out.transposed(0, 2, 3, 1)
    var qkv = toQKV(out)
    qkv = qkv.transposed(0, 3, 1, 2)
    qkv = qkv.reshaped(b * t, 1, c * 3, h * w)
    qkv = qkv.transposed(0, 1, 3, 2)
    let parts = MLX.split(qkv, parts: 3, axis: 3)
    let q = parts[0]
    let k = parts[1]
    let v = parts[2]
    let scale = 1.0 / Float(c).squareRoot()
    out = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v, scale: scale, mask: nil
    )
    out = out.squeezed(axis: 1)
    out = out.transposed(0, 2, 1)
    out = out.reshaped(b * t, c, h, w)
    out = out.transposed(0, 2, 3, 1)
    out = proj(out)
    out = out.transposed(0, 3, 1, 2)
    out = out.reshaped(b, t, c, h, w)
    out = out.transposed(0, 2, 1, 3, 4)
    return out + identity
  }
}
