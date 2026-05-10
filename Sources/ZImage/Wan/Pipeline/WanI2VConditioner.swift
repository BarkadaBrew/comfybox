import Foundation
import MLX
import MLXNN
import MLXRandom

/// Builds the 36-channel conditioning input for Wan 2.2 I2V.
///
/// The I2V transformer expects a 36-channel input assembled from:
/// - 16ch noisy latent (the starting point for denoising)
/// - 4ch temporal mask (first-frame indicator across latent time)
/// - 16ch VAE-encoded init image (padded to video length)
///
/// ## Conditioning Flow (from image2video.py)
///
/// ```
/// noise[16ch] + mask[4ch] + vae_encoded[16ch] = 36 channels total
///
/// 1. Normalize image to [-1, 1]
/// 2. Compute latent resolution from max_area + aspect ratio
/// 3. Generate noise: [16, latent_t, lat_h, lat_w]
/// 4. Build mask: first-frame temporal masking → [4, latent_t, lat_h, lat_w]
/// 5. VAE-encode the init image (padded to video length)
/// 6. Concatenate conditioning: mask(4ch) + vae_encoded(16ch) = 20ch
/// 7. In forward: noise(16ch) is concatenated with conditioning(20ch) → 36ch
/// ```
public enum WanI2VConditioner {

  // MARK: - Constants

  /// VAE spatial/temporal strides: (temporal=4, height=8, width=8).
  public static let vaeStride: (Int, Int, Int) = (4, 8, 8)

  /// Transformer patch size: (1, 2, 2).
  public static let patchSize: (Int, Int, Int) = (1, 2, 2)

  /// VAE latent channels.
  public static let latentChannels = 16

  // MARK: - Resolution

  /// Computes latent resolution and pixel resolution from image aspect ratio and max pixel area.
  ///
  /// The latent dimensions must be divisible by the patch size.
  ///
  /// - Parameters:
  ///   - imageHeight: Source image pixel height.
  ///   - imageWidth: Source image pixel width.
  ///   - maxArea: Maximum pixel area (default 720*1280 = 921600).
  /// - Returns: Tuple of (latH, latW, pixelH, pixelW).
  public static func computeResolution(
    imageHeight: Int,
    imageWidth: Int,
    maxArea: Int = 720 * 1280
  ) -> (latH: Int, latW: Int, pixelH: Int, pixelW: Int) {
    let aspectRatio = Float(imageHeight) / Float(imageWidth)

    // lat_h = round(sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    var latH = Int(round(
      sqrt(Float(maxArea) * aspectRatio)
    )) / vaeStride.1 / patchSize.1 * patchSize.1

    var latW = Int(round(
      sqrt(Float(maxArea) / aspectRatio)
    )) / vaeStride.2 / patchSize.2 * patchSize.2

    // Ensure minimums
    latH = max(latH, patchSize.1)
    latW = max(latW, patchSize.2)

    let pixelH = latH * vaeStride.1
    let pixelW = latW * vaeStride.2

    return (latH: latH, latW: latW, pixelH: pixelH, pixelW: pixelW)
  }

  /// Computes resolution with explicit overrides for width/height.
  public static func computeResolution(
    explicitHeight: Int?,
    explicitWidth: Int?,
    imageHeight: Int,
    imageWidth: Int,
    maxArea: Int = 720 * 1280
  ) -> (latH: Int, latW: Int, pixelH: Int, pixelW: Int) {
    if let h = explicitHeight, let w = explicitWidth {
      // Round down to nearest valid latent dimension
      let latH = (h / vaeStride.1) / patchSize.1 * patchSize.1
      let latW = (w / vaeStride.2) / patchSize.2 * patchSize.2
      return (latH: max(latH, patchSize.1),
              latW: max(latW, patchSize.2),
              pixelH: max(latH, patchSize.1) * vaeStride.1,
              pixelW: max(latW, patchSize.2) * vaeStride.2)
    }
    return computeResolution(imageHeight: imageHeight, imageWidth: imageWidth, maxArea: maxArea)
  }

  // MARK: - Mask Construction

  /// Builds the 4-channel temporal mask for I2V conditioning.
  ///
  /// The mask marks the first frame as 1 and all subsequent frames as 0,
  /// then reshapes to account for the VAE temporal stride of 4.
  ///
  /// From Python:
  /// ```python
  /// msk = ones(1, F, lat_h, lat_w)
  /// msk[:, 1:] = 0
  /// msk = concat([repeat_interleave(msk[:,0:1], repeats=4, dim=1), msk[:,1:]], dim=1)
  /// msk = msk.view(1, msk.shape[1]//4, 4, lat_h, lat_w)
  /// msk = msk.transpose(1, 2)[0]  # [4, latent_t, lat_h, lat_w]
  /// ```
  ///
  /// - Parameters:
  ///   - frameNum: Total number of video frames (must be 4n+1, e.g., 81).
  ///   - latH: Latent height.
  ///   - latW: Latent width.
  /// - Returns: Mask tensor of shape `[4, latent_t, latH, latW]`.
  public static func buildMask(
    frameNum: Int,
    latH: Int,
    latW: Int
  ) -> MLXArray {
    precondition((frameNum - 1) % 4 == 0, "frameNum must be 4n+1, got \(frameNum)")

    let F = frameNum

    // Start with ones [1, F, lat_h, lat_w], set frames 1.. to 0
    var msk = MLXArray.ones([1, F, latH, latW])

    // msk[:, 1:] = 0 — zero out all frames except the first
    if F > 1 {
      let zeros = MLXArray.zeros([1, F - 1, latH, latW])
      let first = msk[0..., 0..<1, 0..., 0...]
      msk = MLX.concatenated([first, zeros], axis: 1)
    }

    // Repeat the first frame 4 times, then concatenate with the rest
    // repeat_interleave(msk[:,0:1], repeats=4, dim=1) → [1, 4, latH, latW]
    let firstFrame = msk[0..., 0..<1, 0..., 0...]  // [1, 1, latH, latW]
    let firstRepeated = MLX.repeated(firstFrame, count: 4, axis: 1)  // [1, 4, latH, latW]
    let rest = msk[0..., 1..., 0..., 0...]  // [1, F-1, latH, latW]
    msk = MLX.concatenated([firstRepeated, rest], axis: 1)  // [1, F+3, latH, latW]

    // Reshape: [1, (F+3)//4, 4, latH, latW]
    // Note: F+3 = F-1+4 = 4n+4 = 4*(n+1), so (F+3)//4 = n+1 = latent_t
    let latentT = msk.dim(1) / 4
    msk = msk.reshaped(1, latentT, 4, latH, latW)

    // Transpose dims 1 and 2: [1, 4, latent_t, latH, latW]
    msk = msk.transposed(0, 2, 1, 3, 4)

    // Remove batch dim: [4, latent_t, latH, latW]
    msk = msk.squeezed(axis: 0)

    return msk
  }

  // MARK: - Image Normalization

  /// Normalizes an image from [0, 1] range to [-1, 1] range.
  ///
  /// Equivalent to Python: `img.sub_(0.5).div_(0.5)`
  ///
  /// - Parameter image: Image tensor, expected in [0, 1] range with shape [C, H, W] or [B, C, H, W].
  /// - Returns: Normalized image in [-1, 1] range.
  public static func normalizeImage(_ image: MLXArray) -> MLXArray {
    return (image - 0.5) / 0.5
  }

  // MARK: - Resize

  /// Resizes an image tensor using bilinear interpolation.
  ///
  /// For simplicity, uses nearest-neighbor for the initial implementation
  /// (bicubic in PyTorch, but the difference is negligible for latent space).
  ///
  /// - Parameters:
  ///   - image: Image tensor [C, H, W].
  ///   - height: Target height.
  ///   - width: Target width.
  /// - Returns: Resized image [C, height, width].
  public static func resizeImage(_ image: MLXArray, height: Int, width: Int) -> MLXArray {
    precondition(image.ndim == 3, "Expected [C, H, W], got ndim=\(image.ndim)")
    let c = image.dim(0)
    let srcH = image.dim(1)
    let srcW = image.dim(2)

    if srcH == height && srcW == width {
      return image
    }

    // Bilinear interpolation: work per channel
    // Create coordinate grids for the target resolution
    var channels: [MLXArray] = []
    for ch in 0..<c {
      let src = image[ch]  // [srcH, srcW]

      // Generate target coordinates mapping to source
      let yCoords = MLXArray(Array(0..<height).map { Float($0) * Float(srcH) / Float(height) })
      let xCoords = MLXArray(Array(0..<width).map { Float($0) * Float(srcW) / Float(width) })

      // Floor coordinates for nearest-neighbor (sufficient for latent VAE input)
      let yIdx = MLX.minimum(MLX.floor(yCoords).asType(.int32), MLXArray(Int32(srcH - 1)))
      let xIdx = MLX.minimum(MLX.floor(xCoords).asType(.int32), MLXArray(Int32(srcW - 1)))

      // Index: src[yIdx][:, xIdx]
      let rowSelected = src[yIdx]  // [height, srcW]
      let result = rowSelected[0..., xIdx]  // [height, width]

      channels.append(result.expandedDimensions(axis: 0))
    }

    return MLX.concatenated(channels, axis: 0)  // [C, height, width]
  }

  // MARK: - VAE Encode Init Image

  /// Encodes the init image through the VAE, padded to video length.
  ///
  /// From Python:
  /// ```python
  /// video_input = concat([
  ///     interpolate(img, size=(h,w), mode='bicubic').transpose(0,1),  # [3, 1, h, w]
  ///     zeros(3, F-1, h, w)                                           # [3, F-1, h, w]
  /// ], dim=1)  # [3, F, h, w]
  /// y_vae = vae.encode([video_input])  # → [16, latent_t, lat_h, lat_w]
  /// ```
  ///
  /// - Parameters:
  ///   - image: Normalized image tensor [C, H, W] in [-1, 1] range.
  ///   - vae: The Wan VAE encoder.
  ///   - frameNum: Total video frames (must be 4n+1).
  ///   - pixelH: Target pixel height.
  ///   - pixelW: Target pixel width.
  /// - Returns: VAE-encoded latent [16, latent_t, latH, latW].
  public static func vaeEncodeInitImage(
    _ image: MLXArray,
    vae: WanVAE,
    frameNum: Int,
    pixelH: Int,
    pixelW: Int
  ) -> MLXArray {
    // Resize image to target resolution
    let resized = resizeImage(image, height: pixelH, width: pixelW)  // [3, pixelH, pixelW]

    // Create video input: init frame + zeros for remaining frames
    // [3, 1, h, w] + [3, F-1, h, w] → [3, F, h, w]
    let firstFrame = resized.expandedDimensions(axis: 1)  // [3, 1, pixelH, pixelW]
    let remainingFrames = MLXArray.zeros([3, frameNum - 1, pixelH, pixelW])
    let videoInput = MLX.concatenated([firstFrame, remainingFrames], axis: 1)  // [3, F, h, w]

    // Add batch dim for VAE: [1, 3, F, h, w]
    let batched = videoInput.expandedDimensions(axis: 0)

    // Encode through VAE
    let encoded = vae.encode(batched)  // [1, 16, latent_t, latH, latW]

    // Remove batch dim
    return encoded.squeezed(axis: 0)  // [16, latent_t, latH, latW]
  }

  // MARK: - Full Conditioning Assembly

  /// Assembles the complete 20-channel I2V conditioning tensor.
  ///
  /// This combines the mask (4ch) and VAE-encoded init image (16ch) into
  /// the conditioning tensor that gets concatenated with noise in the forward pass.
  ///
  /// - Parameters:
  ///   - mask: Temporal mask [4, latent_t, latH, latW].
  ///   - vaeEncoded: VAE-encoded init image [16, latent_t, latH, latW].
  /// - Returns: Conditioning tensor [20, latent_t, latH, latW].
  public static func assembleConditioning(
    mask: MLXArray,
    vaeEncoded: MLXArray
  ) -> MLXArray {
    precondition(mask.dim(0) == 4, "Mask must have 4 channels, got \(mask.dim(0))")
    precondition(vaeEncoded.dim(0) == latentChannels, "VAE encoded must have \(latentChannels) channels")
    return MLX.concatenated([mask, vaeEncoded], axis: 0)  // [20, latent_t, latH, latW]
  }

  // MARK: - Noise Generation

  /// Generates the initial noise tensor for denoising.
  ///
  /// - Parameters:
  ///   - frameNum: Total video frames (must be 4n+1).
  ///   - latH: Latent height.
  ///   - latW: Latent width.
  ///   - seed: Random seed (nil for random).
  /// - Returns: Noise tensor [16, latent_t, latH, latW].
  public static func generateNoise(
    frameNum: Int,
    latH: Int,
    latW: Int,
    seed: UInt64?
  ) -> MLXArray {
    let latentT = (frameNum - 1) / vaeStride.0 + 1

    if let seed = seed {
      MLXRandom.seed(seed)
    }

    return MLXRandom.normal([latentChannels, latentT, latH, latW])
  }

  // MARK: - Sequence Length

  /// Computes the maximum sequence length for the transformer.
  ///
  /// `max_seq_len = latent_t * lat_h * lat_w / (patch_h * patch_w)`
  ///
  /// - Parameters:
  ///   - frameNum: Total video frames.
  ///   - latH: Latent height.
  ///   - latW: Latent width.
  /// - Returns: Maximum sequence length.
  public static func computeSeqLen(
    frameNum: Int,
    latH: Int,
    latW: Int
  ) -> Int {
    let latentT = (frameNum - 1) / vaeStride.0 + 1
    return latentT * latH * latW / (patchSize.1 * patchSize.2)
  }
}
