import Foundation
import MLX
import MLXRandom

/// Creates noise latents and conditioning tensors for SeedVR2 upscale inference.
///
/// The latent creator produces two components needed by the diffusion loop:
///
/// 1. **Noise latents**: Random normal noise in the latent space, shaped for the
///    SeedVR2 transformer. These serve as the starting point for denoising.
///
/// 2. **Condition tensor**: The VAE-encoded input image concatenated with a
///    spatial mask of ones. This conditions the transformer on the low-resolution
///    input during each denoising step.
///
/// ## Tensor Shapes
///
/// All tensors use the SeedVR2 5D video format: `(B, C, T, H, W)`.
///
/// - Noise latents:  `(B, 16, 1, H_lat, W_lat)`
/// - Encoded image:  `(B, 16, 1, H_lat, W_lat)`
/// - Condition mask:  `(1, 1, 1, H_lat, W_lat)` — broadcast across batch and channels
/// - Full condition: `(B, 17, 1, H_lat, W_lat)` — 16 latent channels + 1 mask channel
public enum SeedVR2LatentCreator {

  /// Creates random normal noise latents for the diffusion process.
  ///
  /// - Parameters:
  ///   - seed: Random seed for reproducible generation.
  ///   - height: Latent height (input height / 8).
  ///   - width: Latent width (input width / 8).
  ///   - batchSize: Number of samples in the batch. Default `1`.
  ///   - latentChannels: Number of latent channels. Default `16`.
  /// - Returns: An MLXArray of shape `(batchSize, latentChannels, 1, height, width)`.
  public static func createNoiseLatents(
    seed: Int,
    height: Int,
    width: Int,
    batchSize: Int = 1,
    latentChannels: Int = 16
  ) -> MLXArray {
    let key = MLXRandom.key(UInt64(seed))
    return MLXRandom.normal(
      [batchSize, latentChannels, 1, height, width],
      key: key
    )
  }

  /// Creates the conditioning tensor from a VAE-encoded latent.
  ///
  /// The condition concatenates the encoded image latent with a spatial mask
  /// of ones along the channel dimension. The mask indicates valid conditioning
  /// regions (the entire image, for upscaling).
  ///
  /// - Parameter encodedLatent: The VAE-encoded input image. Accepts either
  ///   4D `(B, C, H, W)` or 5D `(B, C, T, H, W)` input. If 4D, a temporal
  ///   dimension of 1 is inserted at axis 2.
  /// - Returns: An MLXArray of shape `(B, C+1, T, H, W)`.
  public static func createCondition(encodedLatent: MLXArray) -> MLXArray {
    // Ensure 5D: (B, C, T, H, W)
    var latent = encodedLatent
    if latent.ndim == 4 {
      latent = latent.expandedDimensions(axis: 2)
    }

    let height = latent.dim(3)
    let width = latent.dim(4)

    // Create a mask of ones: (1, 1, 1, H, W) — broadcasts across batch
    let mask = MLXArray.ones([1, 1, 1, height, width])

    // Concatenate along channel axis: (B, C, T, H, W) + (1, 1, 1, H, W) -> (B, C+1, T, H, W)
    return MLX.concatenated([latent, mask], axis: 1)
  }
}
