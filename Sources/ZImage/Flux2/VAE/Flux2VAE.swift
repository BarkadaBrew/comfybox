import Foundation
import MLX
import MLXNN

/// Top-level VAE for the Flux2 image generation model.
///
/// The Flux2 VAE is a completely different architecture from the Flux1 AutoencoderKL.
/// It uses batch normalization statistics for packed latent decoding, different block
/// structures, and 32 latent channels (vs 16 for Flux1).
///
/// ## Key Differences from Flux1
///
/// | Feature | Flux1 (AutoencoderKL) | Flux2 (Flux2VAE) |
/// |---------|----------------------|------------------|
/// | Latent channels | 16 | 32 |
/// | Scaling factor | 0.3611 | 1.0 |
/// | Shift factor | 0.1159 | 0.0 |
/// | Packed latents | No | Yes (decode_packed_latents) |
/// | Batch norm stats | No | Yes (for unpacking) |
/// | quant_conv / post_quant_conv | No | Yes (1x1 convolutions) |
///
/// ## Decode Pipeline
///
/// The primary decode path for generation is `decodePackedLatents`:
///
/// ```
/// Packed latents (B, 4*C, H/2, W/2) in NCHW
///   -> un-normalize using batch norm stats (mean + std)
///   -> unpatchify: (B, 4*C, H/2, W/2) -> (B, C, H, W)
///   -> scale/shift: (x / scaling_factor) + shift_factor
///   -> NCHW -> NHWC transpose
///   -> post_quant_conv (1x1)
///   -> NHWC -> NCHW transpose (for decoder entry in Python; we stay NHWC)
///   -> decoder
///   -> NCHW -> NHWC transpose (Python output; we're already NHWC)
///   -> Output image (B, H, W, 3) in NHWC
/// ```
public final class Flux2VAE: Module, VAEImageDecoding {

  /// Scaling factor for latents. Flux2 uses 1.0 (no scaling).
  public let scalingFactor: Float = 1.0

  /// Shift factor for latents. Flux2 uses 0.0 (no shift).
  public let shiftFactor: Float = 0.0

  /// Number of latent channels.
  public let latentChannels: Int = 32

  @ModuleInfo(key: "encoder") var encoder: Flux2Encoder
  @ModuleInfo(key: "decoder") var decoder: Flux2Decoder
  @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
  @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
  @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats

  /// Creates a Flux2 VAE.
  public override init() {
    self._encoder.wrappedValue = Flux2Encoder(
      inChannels: 3, outChannels: 32,
      blockOutChannels: [128, 256, 512, 512])
    self._decoder.wrappedValue = Flux2Decoder(
      inChannels: 32, outChannels: 3,
      blockOutChannels: [128, 256, 512, 512])
    self._quantConv.wrappedValue = Conv2d(
      inputChannels: 2 * 32, outputChannels: 2 * 32,
      kernelSize: 1, stride: 1)
    self._postQuantConv.wrappedValue = Conv2d(
      inputChannels: 32, outputChannels: 32,
      kernelSize: 1, stride: 1)
    self._bn.wrappedValue = Flux2BatchNormStats(numFeatures: 4 * 32, eps: 1e-4)
    super.init()
  }

  /// Encodes an image into latent space.
  ///
  /// - Parameter image: Image tensor in NCHW layout `(B, 3, H, W)` or `(B, 3, 1, H, W)`.
  /// - Returns: Latent tensor in NCHW layout `(B, latent_channels, H/8, W/8)`.
  public func encode(_ image: MLXArray) -> MLXArray {
    var x = image
    // Remove temporal dimension if present
    if x.ndim == 5 {
      x = x.squeezed(axis: 2)
    }

    // NCHW -> NHWC for encoder
    x = x.transposed(0, 2, 3, 1)
    x = encoder(x)

    // NHWC -> NCHW for quant_conv, then back to NHWC
    x = x.transposed(0, 3, 1, 2)
    x = x.transposed(0, 2, 3, 1)
    x = quantConv(x)
    x = x.transposed(0, 3, 1, 2)

    // Split mean/logvar, keep mean
    let parts = split(x, parts: 2, axis: 1)
    let mean = parts[0]

    // Apply scaling
    return (mean - shiftFactor) * scalingFactor
  }

  /// Decodes latent features into an image.
  ///
  /// This implements the standard VAE decode path (not packed latents).
  ///
  /// - Parameter latents: Latent tensor in NCHW layout `(B, latent_channels, H/8, W/8)`.
  /// - Returns: Decoded image in NCHW layout `(B, 3, H, W)`.
  public func decodeLatents(_ latents: MLXArray) -> MLXArray {
    var x = latents
    // Remove temporal dimension if present
    if x.ndim == 5 {
      x = x.squeezed(axis: 2)
    }

    // Un-scale
    x = (x / scalingFactor) + shiftFactor

    // NCHW -> NHWC for post_quant_conv
    x = x.transposed(0, 2, 3, 1)
    x = postQuantConv(x)

    // Decode (stays in NHWC)
    x = decoder(x)

    // NHWC -> NCHW for output
    return x.transposed(0, 3, 1, 2)
  }

  /// Decodes packed latents into an image.
  ///
  /// This is the primary decode path used during Flux2 image generation.
  /// Packed latents have their channels patchified (4x channels, half spatial size)
  /// and are normalized by batch norm statistics.
  ///
  /// - Parameter packedLatents: Packed latent tensor in NCHW layout
  ///   `(B, 4*latent_channels, H/16, W/16)` or with temporal dim `(B, 4*C, 1, H/16, W/16)`.
  /// - Returns: Decoded image in NCHW layout `(B, 3, H, W)`.
  public func decodePackedLatents(_ packedLatents: MLXArray) -> MLXArray {
    var x = packedLatents
    // Remove temporal dimension if present
    if x.ndim == 5 {
      x = x.squeezed(axis: 2)
    }

    // Un-normalize using batch norm running statistics
    // bn_mean: (1, num_features, 1, 1), bn_std: (1, num_features, 1, 1)
    let bnMean = bn.runningMean.reshaped(1, -1, 1, 1)
    let bnStd = sqrt(bn.runningVar.reshaped(1, -1, 1, 1) + bn.eps)
    x = x * bnStd + bnMean

    // Unpatchify: (B, 4*C, H/2, W/2) -> (B, C, H, W)
    x = Flux2VAE.unpatchifyLatents(x)

    // Standard decode
    return decodeLatents(x)
  }

  /// VAEImageDecoding conformance -- delegates to standard latent decode.
  ///
  /// - Parameters:
  ///   - latents: Latent tensor in NCHW layout.
  ///   - return_dict: Ignored (kept for protocol conformance).
  /// - Returns: Tuple of decoded image and empty dict.
  public func decode(_ latents: MLXArray, return_dict: Bool = false) -> (MLXArray, Any) {
    let decoded = decodeLatents(latents)
    return (decoded, [:] as [String: Int])
  }

  /// Unpatchifies latents from packed format to standard spatial layout.
  ///
  /// Converts `(B, 4*C, H, W)` -> `(B, C, 2*H, 2*W)` by rearranging the
  /// channel dimension into 2x2 spatial patches.
  ///
  /// - Parameter latents: Packed latent tensor `(B, 4*C, H, W)` in NCHW.
  /// - Returns: Unpatchified tensor `(B, C, 2*H, 2*W)` in NCHW.
  static func unpatchifyLatents(_ latents: MLXArray) -> MLXArray {
    let (batch, numChannels, height, width) = (latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
    // (B, 4C, H, W) -> (B, C, 2, 2, H, W)
    var x = latents.reshaped(batch, numChannels / 4, 2, 2, height, width)
    // (B, C, 2, 2, H, W) -> (B, C, H, 2, W, 2)
    x = x.transposed(0, 1, 4, 2, 5, 3)
    // (B, C, H, 2, W, 2) -> (B, C, 2H, 2W)
    x = x.reshaped(batch, numChannels / 4, height * 2, width * 2)
    return x
  }
}
