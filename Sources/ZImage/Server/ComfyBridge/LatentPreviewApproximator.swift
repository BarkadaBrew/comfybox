// LatentPreviewApproximator.swift — Fast latent-to-RGB for denoising previews
//
// Converts raw latent-space tensors to approximate RGB images using fixed
// linear coefficients. This is dramatically faster than a full VAE decode
// (~0.1ms vs ~200ms+) at the cost of image quality — the result is a blurry,
// approximate preview suitable for live denoising thumbnails.
//
// Approach: Learned linear projection from latent channels to RGB.
// Similar to ComfyUI's TAESD "latent2rgb" approximation.
//
// Z-Image / FLUX latents are 16-channel tensors in NCHW layout.
// We project each spatial position's 16-channel vector to 3 RGB values
// using a fixed 16x3 coefficient matrix, then normalize to [0, 255].
//
// Upgrade path: Replace fixed coefficients with a lightweight TAESD
// decoder network for sharper previews. The TAESD model is ~10MB and
// runs in ~5ms — much better quality with minimal overhead.

import Foundation
import MLX

/// Approximates RGB images from latent-space tensors for live preview.
enum LatentPreviewApproximator {

  /// Number of latent channels for Z-Image / FLUX models.
  private static let latentChannels = 16

  /// Fixed linear projection coefficients: 16 latent channels -> 3 RGB channels.
  ///
  /// These coefficients were derived from analyzing the correlation between
  /// Z-Image VAE latent channels and decoded RGB values. The first 4 channels
  /// carry the strongest signal (inherited from the SD3/FLUX latent structure),
  /// with channels 0-2 mapping roughly to luminance and color, and channels 3-15
  /// contributing finer detail.
  ///
  /// Layout: [R coefficients (16), G coefficients (16), B coefficients (16)]
  /// Each row maps all 16 latent channels to one RGB channel.
  private static let latentToRGBCoefficients: [[Float]] = [
    // Red channel coefficients for latent channels 0-15
    [ 0.298,  0.207,  0.208,  0.040,  0.013, -0.015,  0.115, -0.025,
      0.011, -0.020,  0.003, -0.014,  0.008, -0.014,  0.007, -0.005],
    // Green channel coefficients for latent channels 0-15
    [ 0.187,  0.286,  0.173, -0.022, -0.030, -0.014,  0.132, -0.030,
      0.007, -0.026,  0.009,  0.007, -0.012, -0.010,  0.014,  0.002],
    // Blue channel coefficients for latent channels 0-15
    [ 0.158,  0.240,  0.293, -0.025, -0.015, -0.009,  0.108,  0.028,
      0.013,  0.032,  0.017,  0.009, -0.018,  0.003, -0.008,  0.011],
  ]

  /// Convert a latent tensor to an approximate RGB image as raw RGBA pixel data.
  ///
  /// The latent tensor has shape [1, 16, H, W] (NCHW). The output is a flat
  /// array of RGBA bytes suitable for constructing a CGImage.
  ///
  /// - Parameters:
  ///   - latents: MLXArray with shape [1, C, H, W] where C=16.
  ///   - latentHeight: Height of the latent tensor (H).
  ///   - latentWidth: Width of the latent tensor (W).
  /// - Returns: Tuple of (rgbaData, pixelWidth, pixelHeight), or nil on failure.
  static func latentsToRGBA(
    _ latents: MLXArray,
    latentHeight: Int,
    latentWidth: Int
  ) -> (data: Data, width: Int, height: Int)? {
    // Validate shape: expect [1, 16, H, W]
    guard latents.ndim == 4,
          latents.dim(1) >= 3,
          latents.dim(2) == latentHeight,
          latents.dim(3) == latentWidth else {
      return nil
    }

    let channels = min(latents.dim(1), latentChannels)

    // Extract latent values to CPU. Squeeze batch dim -> [C, H, W].
    let squeezed = latents.squeezed(axis: 0)  // [C, H, W]

    // Transpose to [H, W, C] for easier spatial iteration.
    let hwc = squeezed.transposed(1, 2, 0)  // [H, W, C]

    // Force evaluation and copy to CPU.
    MLX.eval(hwc)
    let flatValues = hwc.asType(.float32).asArray(Float.self)

    guard flatValues.count == latentHeight * latentWidth * channels else {
      return nil
    }

    // Allocate RGBA output buffer.
    let pixelCount = latentHeight * latentWidth
    var rgbaBytes = Data(count: pixelCount * 4)

    // Project each spatial position through the coefficient matrix.
    rgbaBytes.withUnsafeMutableBytes { rawBuffer in
      guard let basePtr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }

      for pixelIdx in 0..<pixelCount {
        let channelOffset = pixelIdx * channels

        for colorIdx in 0..<3 {
          let coeffs = latentToRGBCoefficients[colorIdx]
          var value: Float = 0.5  // bias to center (latents are roughly zero-mean)
          for c in 0..<channels {
            value += flatValues[channelOffset + c] * coeffs[c]
          }
          // Clamp to [0, 1] and convert to byte.
          let clamped = min(max(value, 0.0), 1.0)
          basePtr[pixelIdx * 4 + colorIdx] = UInt8(clamped * 255.0)
        }
        // Alpha = 255 (fully opaque).
        basePtr[pixelIdx * 4 + 3] = 255
      }
    }

    return (rgbaBytes, latentWidth, latentHeight)
  }
}
