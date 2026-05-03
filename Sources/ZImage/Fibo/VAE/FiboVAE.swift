// FiboVAE.swift — Top-level Wan 2.2 VAE for the FIBO model
// Ported from mflux: wan_2_2_vae.py
//
// The Wan 2.2 VAE is a 3D video autoencoder repurposed for image generation.
// It differs from Flux's AutoencoderKL in several fundamental ways:
//
// - 48 latent channels (vs 16 for Flux 1, 64 for Flux 2)
// - 3D convolutions with causal temporal padding (vs 2D)
// - RMSNorm with L2-based normalization (vs GroupNorm)
// - Spatial patchification (patch_size=2, folding 2x2 patches into channels)
// - Per-channel latent normalization (48 mean + 48 std values)
// - quant_conv and post_quant_conv layers in the latent space
//
// For image generation (not video), the temporal dimension is always 1.
// The VAE adds this dimension automatically for 4D inputs and preserves it
// throughout the pipeline. The CausalConv3d layers handle T=1 correctly
// via their causal padding scheme.
//
// Encode pipeline: image → patchify → encoder → quant_conv → normalize
// Decode pipeline: denormalize → post_quant_conv → decoder → unpatchify

import MLX
import MLXNN

/// Top-level Wan 2.2 VAE (AutoencoderKLWan) for the FIBO model.
///
/// Provides encode and decode operations for mapping between RGB image tensors
/// and the 48-channel latent space used by the FIBO diffusion transformer.
///
/// ## Encode
///
/// ```
/// Input (B, 3, H, W) or (B, 3, T, H, W)
///   → add temporal dim if 4D: (B, 3, 1, H, W)
///   → patchify: (B, 12, 1, H/2, W/2)
///   → encoder: (B, 96, 1, H/16, W/16)
///   → quant_conv: (B, 96, 1, H/16, W/16)
///   → take first 48 channels (mean): (B, 48, 1, H/16, W/16)
///   → normalize per-channel: (B, 48, 1, H/16, W/16)
/// ```
///
/// ## Decode
///
/// ```
/// Input (B, 48, H/16, W/16) or (B, 48, 1, H/16, W/16)
///   → add temporal dim if 4D: (B, 48, 1, H/16, W/16)
///   → denormalize per-channel
///   → post_quant_conv: (B, 48, 1, H/16, W/16)
///   → decoder: (B, 12, 1, H/2, W/2)
///   → unpatchify: (B, 3, 1, H, W)
/// ```
///
/// ## Spatial Scale
///
/// The VAE provides a 16x spatial reduction (vs 8x for Flux/SeedVR2).
/// An H x W input produces H/16 x W/16 latents.
public final class FiboVAE: Module {

  /// Number of latent channels.
  public static let zDim = 48

  /// Spatial downsampling factor.
  public static let spatialScale = 16

  /// Spatial patch size for patchify/unpatchify.
  public static let patchSize = 2

  /// Per-channel normalization means for the 48 latent channels.
  public static let latentsMean: [Float] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
    -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
     0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
  ]

  /// Per-channel normalization standard deviations for the 48 latent channels.
  public static let latentsStd: [Float] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
  ]

  /// The 3D encoder.
  @ModuleInfo(key: "encoder") var encoder: FiboVAEEncoder

  /// The 3D decoder.
  @ModuleInfo(key: "decoder") var decoder: FiboVAEDecoder

  /// Quantization convolution (encoder output → latent space).
  @ModuleInfo(key: "quant_conv") var quantConv: FiboCausalConv3d

  /// Post-quantization convolution (latent space → decoder input).
  @ModuleInfo(key: "post_quant_conv") var postQuantConv: FiboCausalConv3d

  /// Creates the Wan 2.2 VAE with default FIBO configuration.
  public override init() {
    self._encoder.wrappedValue = FiboVAEEncoder(
      inChannels: 12,
      dim: 160,
      zDim: Self.zDim * 2,  // 96 (mean + logvar)
      dimMult: [1, 2, 4, 4],
      numResBlocks: 2,
      temporalDownsample: [false, true, true]
    )

    self._quantConv.wrappedValue = FiboCausalConv3d(
      inChannels: Self.zDim * 2,
      outChannels: Self.zDim * 2,
      kernelSize: 1,
      padding: 0
    )

    self._decoder.wrappedValue = FiboVAEDecoder(
      dim: 256,
      zDim: Self.zDim,
      dimMult: [1, 2, 4, 4],
      numResBlocks: 2,
      temporalUpsample: [],  // No temporal upsampling for image gen
      outChannels: 12
    )

    self._postQuantConv.wrappedValue = FiboCausalConv3d(
      inChannels: Self.zDim,
      outChannels: Self.zDim,
      kernelSize: 1,
      padding: 0
    )

    super.init()
  }

  // MARK: - Encode

  /// Encodes an image into the normalized latent space.
  ///
  /// - Parameter images: Image tensor of shape `(B, 3, H, W)` or `(B, 3, T, H, W)`.
  /// - Returns: Normalized latent tensor of shape `(B, 48, T_lat, H/16, W/16)`.
  public func encode(_ images: MLXArray) -> MLXArray {
    // Ensure 5D: (B, C, T, H, W)
    var x: MLXArray
    if images.ndim == 4 {
      x = images.reshaped(images.dim(0), images.dim(1), 1, images.dim(2), images.dim(3))
    } else if images.ndim == 5 {
      x = images
    } else {
      fatalError("FiboVAE.encode: expected 4D or 5D input, got \(images.ndim)D")
    }

    // Patchify: (B, 3, T, H, W) → (B, 12, T, H/2, W/2)
    x = FiboVAEPatchify.patchify(x, patchSize: Self.patchSize)

    // Encode
    let h = encoder(x)

    // Quant conv
    let quantized = quantConv(h)

    // Take mean (first z_dim channels), discard logvar
    let mean = quantized[0..., ..<Self.zDim, 0..., 0..., 0...]

    // Normalize per-channel
    return normalizeLatents(mean)
  }

  // MARK: - Decode

  /// Decodes a latent tensor back to an image.
  ///
  /// - Parameter latents: Latent tensor of shape `(B, 48, H/16, W/16)` or
  ///   `(B, 48, T_lat, H/16, W/16)`.
  /// - Returns: Decoded image tensor of shape `(B, 3, T, H, W)`.
  public func decode(_ latents: MLXArray) -> MLXArray {
    // Ensure 5D
    var z: MLXArray
    if latents.ndim == 4 {
      z = latents.reshaped(latents.dim(0), latents.dim(1), 1, latents.dim(2), latents.dim(3))
    } else {
      z = latents
    }

    // Denormalize per-channel
    z = denormalizeLatents(z)

    // Post-quant conv
    z = postQuantConv(z)

    // Decode
    let decoded = decoder(z)

    // Unpatchify: (B, 12, T, H/2, W/2) → (B, 3, T, H, W)
    return FiboVAEPatchify.unpatchify(decoded, patchSize: Self.patchSize)
  }

  // MARK: - Latent Normalization

  /// Normalizes latents using per-channel mean and std.
  ///
  /// Applied after encoding: `(latents - mean) / std`
  ///
  /// - Parameter latents: Raw latent tensor of shape `(B, 48, ...)`.
  /// - Returns: Normalized latent tensor.
  public func normalizeLatents(_ latents: MLXArray) -> MLXArray {
    let mean = MLXArray(Self.latentsMean).reshaped(1, Self.zDim, 1, 1, 1)
    let std = MLXArray(Self.latentsStd).reshaped(1, Self.zDim, 1, 1, 1)
    return (latents - mean) / std
  }

  /// Denormalizes latents using per-channel mean and std.
  ///
  /// Applied before decoding: `latents * std + mean`
  ///
  /// - Parameter latents: Normalized latent tensor of shape `(B, 48, ...)`.
  /// - Returns: Denormalized latent tensor.
  public func denormalizeLatents(_ latents: MLXArray) -> MLXArray {
    let mean = MLXArray(Self.latentsMean).reshaped(1, Self.zDim, 1, 1, 1)
    let std = MLXArray(Self.latentsStd).reshaped(1, Self.zDim, 1, 1, 1)
    return latents * std + mean
  }
}
