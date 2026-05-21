import Foundation
import MLX
import MLXNN

/// Patchify and unpatchify operations for the LTX-2 Video VAE.
///
/// These operations convert between spatial pixel layout and a channel-packed
/// layout. Patchify moves spatial pixels (H, W) into the channel dimension,
/// while unpatchify reverses the operation.
///
/// ## Patchify
///
/// ```
/// Input:  (B, C, F, H, W)
/// Output: (B, C * patchSize^2, F, H/patchSize, W/patchSize)
/// ```
///
/// ## Unpatchify
///
/// ```
/// Input:  (B, C * patchSize^2, F, H', W')
/// Output: (B, C, F, H' * patchSize, W' * patchSize)
/// ```
///
/// The default patch size is 4, giving a 16x channel expansion. Combined with
/// the encoder's strided convolutions, the total spatial compression is 32x.
public enum LTX2Patchify {

  /// Patchify: move spatial pixels into channel dimension.
  ///
  /// Matches the Python einops pattern:
  /// `"b c (f p) (h q) (w r) -> b (c p r q) f h w"`
  /// where p = patchSizeT, r = patchSizeHW (width), q = patchSizeHW (height).
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `(B, C, F, H, W)`.
  ///   - patchSizeHW: Spatial patch size. Default `4`.
  ///   - patchSizeT: Temporal patch size. Default `1`.
  /// - Returns: Patched tensor of shape `(B, C * patchSizeHW^2 * patchSizeT, F/patchSizeT, H/patchSizeHW, W/patchSizeHW)`.
  public static func patchify(
    _ x: MLXArray,
    patchSizeHW: Int = 4,
    patchSizeT: Int = 1
  ) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let f = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    let newF = f / patchSizeT
    let newH = h / patchSizeHW
    let newW = w / patchSizeHW
    let newC = c * patchSizeHW * patchSizeHW * patchSizeT

    // Reshape: (B, C, F, H, W) -> (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
    var out = x.reshaped(b, c, newF, patchSizeT, newH, patchSizeHW, newW, patchSizeHW)

    // Permute: (B, C, F', pt, H', ph, W', pw) -> (B, C, pt, pw, ph, F', H', W')
    out = out.transposed(0, 1, 3, 7, 5, 2, 4, 6)

    // Reshape: (B, C, pt, pw, ph, F', H', W') -> (B, C*pt*pw*ph, F', H', W')
    out = out.reshaped(b, newC, newF, newH, newW)

    return out
  }

  /// Unpatchify: move channel-packed pixels back to spatial dimensions.
  ///
  /// Inverse of patchify. Matches the Python einops pattern:
  /// `"b (c p r q) f h w -> b c (f p) (h q) (w r)"`
  /// where p = patchSizeT, r = patchSizeHW (width), q = patchSizeHW (height).
  ///
  /// - Parameters:
  ///   - x: Patched tensor of shape `(B, C_packed, F, H, W)`.
  ///   - patchSizeHW: Spatial patch size. Default `4`.
  ///   - patchSizeT: Temporal patch size. Default `1`.
  /// - Returns: Unpatched tensor of shape `(B, C, F * patchSizeT, H * patchSizeHW, W * patchSizeHW)`.
  public static func unpatchify(
    _ x: MLXArray,
    patchSizeHW: Int = 4,
    patchSizeT: Int = 1
  ) -> MLXArray {
    let b = x.dim(0)
    let cPacked = x.dim(1)
    let f = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    let c = cPacked / (patchSizeHW * patchSizeHW * patchSizeT)

    // Reshape: (B, C*pt*pr*pq, F, H, W) -> (B, C, pt, pr, pq, F, H, W)
    // Channel layout: (c, p=temporal, r=width, q=height)
    var out = x.reshaped(b, c, patchSizeT, patchSizeHW, patchSizeHW, f, h, w)

    // Permute: (B, C, pt, pr, pq, F, H, W) -> (B, C, F, pt, H, pq, W, pr)
    out = out.transposed(0, 1, 5, 2, 6, 4, 7, 3)

    // Reshape: (B, C, F, pt, H, pq, W, pr) -> (B, C, F*pt, H*pq, W*pr)
    out = out.reshaped(b, c, f * patchSizeT, h * patchSizeHW, w * patchSizeHW)

    return out
  }
}

/// Per-channel normalization statistics for VAE latent space.
///
/// Stores learned mean and standard deviation per channel, used to normalize
/// encoder output and denormalize decoder input. These are loaded from the
/// checkpoint (`per_channel_statistics.mean` and `per_channel_statistics.std`).
public final class LTX2PerChannelStatistics: Module {

  /// Per-channel mean, shape `(latentChannels,)`.
  public var mean: MLXArray

  /// Per-channel standard deviation, shape `(latentChannels,)`.
  public var std: MLXArray

  /// Number of latent channels.
  public let latentChannels: Int

  /// Creates per-channel statistics.
  ///
  /// - Parameter latentChannels: Number of latent channels. Default `128`.
  public init(latentChannels: Int = 128) {
    self.latentChannels = latentChannels
    self.mean = MLXArray.zeros([latentChannels])
    self.std = MLXArray.ones([latentChannels])
    super.init()
  }

  /// Normalize latents: `(x - mean) / std`.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, ...)`.
  /// - Returns: Normalized tensor.
  public func normalize(_ x: MLXArray) -> MLXArray {
    let dtype = x.dtype
    let m = mean.asType(.float32).reshaped(1, -1, 1, 1, 1)
    let s = std.asType(.float32).reshaped(1, -1, 1, 1, 1)
    return ((x - m) / s).asType(dtype)
  }

  /// Denormalize latents: `x * std + mean`.
  ///
  /// - Parameter x: Normalized tensor of shape `(B, C, ...)`.
  /// - Returns: Denormalized tensor.
  public func unNormalize(_ x: MLXArray) -> MLXArray {
    let dtype = x.dtype
    let m = mean.asType(.float32).reshaped(1, -1, 1, 1, 1)
    let s = std.asType(.float32).reshaped(1, -1, 1, 1, 1)
    return (x * s + m).asType(dtype)
  }
}
