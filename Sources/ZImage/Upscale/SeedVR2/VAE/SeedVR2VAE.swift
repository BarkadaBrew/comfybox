import Foundation
import MLX
import MLXNN

/// Top-level 3D Video VAE for the SeedVR2 upscaler.
///
/// Provides encode and decode operations for mapping between RGB image/video
/// tensors and the latent space used by the SeedVR2 diffusion transformer.
///
/// ## Encode
///
/// Accepts an image `(B, 3, H, W)` or video `(B, 3, T, H, W)`. For single images,
/// a temporal dimension of 1 is inserted automatically. The encoder produces both
/// mean and log-variance channels; only the mean is used, scaled by ``scalingFactor``.
///
/// ```
/// Input (B, 3, H, W)
///   → (B, 3, 1, H, W)           // add temporal dim
///   → Encoder3D → (B, 32, 1, H/8, W/8)
///   → split first 16 channels   // take mean, discard logvar
///   → × scalingFactor
///   → Output (B, 16, 1, H/8, W/8)
/// ```
///
/// ## Decode
///
/// Accepts latent `(B, 16, T, H/8, W/8)`. Divides by ``scalingFactor`` then
/// passes through the decoder to reconstruct RGB.
///
/// ```
/// Input (B, 16, 1, H/8, W/8)
///   → ÷ scalingFactor
///   → Decoder3D
///   → Output (B, 3, 1, H, W)
/// ```
///
/// ## Spatial Scale
///
/// The VAE provides an 8x spatial reduction: an H x W input produces H/8 x W/8
/// latents. This is exposed via ``spatialScale`` for use by the diffusion pipeline.
public final class SeedVR2VAE: Module {

  /// Scaling factor applied to latents after encoding.
  /// Normalizes the latent distribution for the downstream diffusion model.
  public let scalingFactor: Float = 0.9152

  /// Spatial downsampling factor (8x for the default architecture).
  public let spatialScale: Int = 8

  /// Number of latent channels (mean only, not including log-variance).
  public let latentChannels: Int

  /// The 3D encoder.
  @ModuleInfo(key: "encoder") var encoder: SeedVR2Encoder3D

  /// The 3D decoder.
  @ModuleInfo(key: "decoder") var decoder: SeedVR2Decoder3D

  /// Creates a SeedVR2 3D Video VAE.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels (RGB = 3). Default `3`.
  ///   - outChannels: Number of output channels (RGB = 3). Default `3`.
  ///   - latentChannels: Number of latent channels. Default `16`.
  ///   - blockOutChannels: Channel counts for each encoder/decoder stage.
  ///     Default `[128, 256, 512, 512]`.
  public init(
    inChannels: Int = 3,
    outChannels: Int = 3,
    latentChannels: Int = 16,
    blockOutChannels: [Int] = [128, 256, 512, 512]
  ) {
    self.latentChannels = latentChannels

    self._encoder.wrappedValue = SeedVR2Encoder3D(
      inChannels: inChannels,
      outChannels: latentChannels,
      blockOutChannels: blockOutChannels,
      layersPerBlock: 2,
      temporalDownBlocks: 2
    )

    self._decoder.wrappedValue = SeedVR2Decoder3D(
      inChannels: latentChannels,
      outChannels: outChannels,
      blockOutChannels: blockOutChannels,
      layersPerBlock: 3,
      temporalUpBlocks: 2
    )

    super.init()
  }

  /// Encodes an image or video into the scaled latent space.
  ///
  /// - Parameter x: Input tensor of shape `(B, 3, H, W)` or `(B, 3, T, H, W)`.
  /// - Returns: Scaled latent tensor of shape `(B, latentChannels, T_out, H/8, W/8)`.
  public func encode(_ x: MLXArray) -> MLXArray {
    // Add temporal dimension if input is a single image (4D → 5D).
    var input = x
    if input.ndim == 4 {
      input = input.expandedDimensions(axis: 2)
    }

    // Encode → (B, 2*latentChannels, T_out, H/8, W/8)
    let encoded = encoder(input)

    // Split into mean and logvar, keep only the mean.
    let parts = split(encoded, parts: 2, axis: 1)
    let mean = parts[0]

    // Scale the latents.
    return mean * scalingFactor
  }

  /// Decodes a latent tensor back to an image or video.
  ///
  /// - Parameter z: Latent tensor of shape `(B, latentChannels, T, H/8, W/8)`
  ///   or `(B, latentChannels, H/8, W/8)`.
  /// - Returns: Decoded tensor of shape `(B, 3, T, H, W)`.
  public func decode(_ z: MLXArray) -> MLXArray {
    // Add temporal dimension if input is 4D.
    var latent = z
    if latent.ndim == 4 {
      latent = latent.expandedDimensions(axis: 2)
    }

    // Unscale the latents.
    latent = latent / scalingFactor

    // Decode → (B, 3, T, H, W)
    let decoded = decoder(latent)
    return decoded
  }
}
