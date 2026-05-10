import Foundation
import MLX
import MLXNN

/// Wan 2.2 Video VAE — the top-level variational autoencoder.
///
/// Encodes RGB video to a 48-channel latent space with spatial and temporal
/// compression, and decodes latents back to video. Uses patchify/unpatchify
/// with patch size 2 for initial spatial folding.
///
/// ## Encode
///
/// ```
/// Input (B, 3, T, H, W)
///   → patchify(p=2): (B, 12, T, H/2, W/2)
///   → Encoder3d → down blocks + mid block
///   → quant_conv: CausalConv3d(96 → 96, k=1)
///   → take mean (first 48 channels)
///   → normalize: (mean - latentsMean) / latentsStd
///   → Output (B, 48, T', H', W')
/// ```
///
/// ## Decode
///
/// ```
/// Input (B, 48, T', H', W')
///   → denormalize: z * latentsStd + latentsMean
///   → post_quant_conv: CausalConv3d(48 → 48, k=1)
///   → Decoder3d → mid block + up blocks
///   → unpatchify(p=2): (B, 3, T, H, W)
///   → Output (B, 3, T, H, W)
/// ```
public final class WanVAE: Module {

  // MARK: - Constants

  /// Number of latent channels.
  public static let zDim = 48

  /// Encoder base channel dimension.
  public static let encoderBaseDim = 160

  /// Decoder base channel dimension.
  public static let decoderBaseDim = 256

  /// Channel multipliers.
  public static let dimMult = [1, 2, 4, 4]

  /// Residual blocks per encoder down block.
  public static let numResBlocks = 2

  /// Output channels (patchified RGB: 3 * 2 * 2 = 12).
  public static let outChannels = 12

  /// Spatial compression factor.
  public static let spatialScale = 16

  /// Number of latent channels.
  public static let latentChannels = 48

  /// Patch size for patchify/unpatchify.
  public static let patchSize = 2

  /// Per-channel mean for latent normalization.
  public static let latentsMean: [Float] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
    -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
     0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
  ]

  /// Per-channel std for latent normalization.
  public static let latentsStd: [Float] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
  ]

  // MARK: - Modules

  /// The 3D encoder.
  @ModuleInfo(key: "encoder") var encoder: WanEncoder3d

  /// Quantization convolution (encoder output).
  @ModuleInfo(key: "quant_conv") var quantConv: WanCausalConv3d

  /// The 3D decoder.
  @ModuleInfo(key: "decoder") var decoder: WanDecoder3d

  /// Post-quantization convolution (decoder input).
  @ModuleInfo(key: "post_quant_conv") var postQuantConv: WanCausalConv3d

  // MARK: - Init

  /// Creates a Wan 2.2 VAE with default architecture.
  public override init() {
    self._encoder.wrappedValue = WanEncoder3d(
      inChannels: Self.outChannels,
      dim: Self.encoderBaseDim,
      zDim: Self.zDim * 2,
      dimMult: Self.dimMult,
      numResBlocks: Self.numResBlocks,
      temporalDownsample: [false, true, true]
    )

    self._quantConv.wrappedValue = WanCausalConv3d(
      inChannels: Self.zDim * 2,
      outChannels: Self.zDim * 2,
      kernelSize: 1,
      padding: 0
    )

    self._decoder.wrappedValue = WanDecoder3d(
      dim: Self.decoderBaseDim,
      zDim: Self.zDim,
      dimMult: Self.dimMult,
      numResBlocks: Self.numResBlocks,
      outChannels: Self.outChannels
    )

    self._postQuantConv.wrappedValue = WanCausalConv3d(
      inChannels: Self.zDim,
      outChannels: Self.zDim,
      kernelSize: 1,
      padding: 0
    )

    super.init()
  }

  // MARK: - Encode / Decode

  /// Encodes RGB video to normalized latent space.
  ///
  /// - Parameter images: Input tensor of shape `(B, 3, T, H, W)` or `(B, 3, H, W)`.
  /// - Returns: Normalized latent tensor of shape `(B, 48, T', H', W')`.
  public func encode(_ images: MLXArray) -> MLXArray {
    var x = images

    // Add temporal dimension if single image
    if x.ndim == 4 {
      x = x.reshaped(x.dim(0), x.dim(1), 1, x.dim(2), x.dim(3))
    }

    // Patchify
    x = Self.patchify(x, patchSize: Self.patchSize)

    // Encode
    var h = encoder(x)
    h = quantConv(h)

    // Take mean (first 48 channels)
    let mean = h[0..., ..<Self.zDim, 0..., 0..., 0...]

    // Normalize
    let latMean = MLXArray(Self.latentsMean).reshaped(1, Self.zDim, 1, 1, 1)
    let latStd = MLXArray(Self.latentsStd).reshaped(1, Self.zDim, 1, 1, 1)
    let encoded = (mean - latMean) / latStd

    return encoded
  }

  /// Decodes normalized latents back to RGB video.
  ///
  /// - Parameter latents: Normalized latent tensor of shape `(B, 48, T', H', W')` or 4D.
  /// - Returns: Decoded video tensor of shape `(B, 3, T, H, W)`.
  public func decode(_ latents: MLXArray) -> MLXArray {
    var z = latents

    // Add temporal dimension if needed
    if z.ndim == 4 {
      z = z.reshaped(z.dim(0), z.dim(1), 1, z.dim(2), z.dim(3))
    }

    // Denormalize
    let latMean = MLXArray(Self.latentsMean).reshaped(1, Self.zDim, 1, 1, 1)
    let latStd = MLXArray(Self.latentsStd).reshaped(1, Self.zDim, 1, 1, 1)
    z = z * latStd + latMean

    // Post-quant conv
    z = postQuantConv(z)

    // Decode
    var decoded = decoder(z)

    // Unpatchify
    decoded = Self.unpatchify(decoded, patchSize: Self.patchSize)

    return decoded
  }

  // MARK: - Patchify / Unpatchify

  /// Patchify: fold spatial pixels into channels.
  ///
  /// `(B, C, T, H, W) → (B, C*p*p, T, H/p, W/p)`
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `(B, C, T, H, W)`.
  ///   - patchSize: Spatial patch size.
  /// - Returns: Patchified tensor.
  public static func patchify(_ x: MLXArray, patchSize: Int) -> MLXArray {
    if patchSize == 1 { return x }

    let b = x.dim(0)
    let c = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    // (B, C, T, H/p, p, W/p, p)
    var out = x.reshaped(b, c, t, h / patchSize, patchSize, w / patchSize, patchSize)

    // (B, C, p_w, p_h, T, H/p, W/p)
    out = out.transposed(0, 1, 6, 4, 2, 3, 5)

    // (B, C*p*p, T, H/p, W/p)
    out = out.reshaped(b, c * patchSize * patchSize, t, h / patchSize, w / patchSize)

    return out
  }

  /// Unpatchify: unfold channel-packed pixels back to spatial dimensions.
  ///
  /// `(B, C*p*p, T, H', W') → (B, C, T, H'*p, W'*p)`
  ///
  /// - Parameters:
  ///   - x: Patchified tensor.
  ///   - patchSize: Spatial patch size.
  /// - Returns: Unpatchified tensor.
  public static func unpatchify(_ x: MLXArray, patchSize: Int) -> MLXArray {
    if patchSize == 1 { return x }

    let b = x.dim(0)
    let cPatches = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)
    let c = cPatches / (patchSize * patchSize)

    // (B, C, p, p, T, H', W')
    var out = x.reshaped(b, c, patchSize, patchSize, t, h, w)

    // (B, C, T, H', p, W', p)
    out = out.transposed(0, 1, 4, 5, 3, 6, 2)

    // (B, C, T, H'*p, W'*p)
    out = out.reshaped(b, c, t, h * patchSize, w * patchSize)

    return out
  }
}
