// InpaintUtilities.swift — Latent-space inpainting helpers
//
// Provides VAE encoding of input images, pixel-to-latent mask conversion,
// and CGImage loading from raw Data. Used by ZImagePipeline for inpainting.

import Foundation
import Logging
import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import CoreImage
#endif

/// Utilities for latent-space inpainting in the base ZImagePipeline.
enum InpaintUtilities {

  enum InpaintError: Error, CustomStringConvertible {
    case imageLoadFailed(String)
    case maskLoadFailed(String)

    var description: String {
      switch self {
      case .imageLoadFailed(let msg): return "Inpaint image load failed: \(msg)"
      case .maskLoadFailed(let msg): return "Mask load failed: \(msg)"
      }
    }
  }

  // MARK: - CGImage from Data

  /// Load a CGImage from raw PNG/JPEG data.
  static func loadCGImage(from data: Data) throws -> CGImage {
    #if canImport(CoreGraphics)
    guard let provider = CGDataProvider(data: data as CFData),
          let cgImage = CGImage(
            pngDataProviderSource: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
          ) else {
      // Try JPEG fallback
      guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw InpaintError.imageLoadFailed("Failed to decode image data (\(data.count) bytes)")
      }
      return image
    }
    return cgImage
    #else
    throw InpaintError.imageLoadFailed("CoreGraphics not available on this platform")
    #endif
  }

  // MARK: - VAE Encode

  /// Encode a CGImage to latent space using the VAE encoder.
  static func encodeImageToLatents(
    cgImage: CGImage,
    vae: AutoencoderKL,
    vaeConfig: ZImageVAEConfig,
    pixelH: Int,
    pixelW: Int,
    logger: Logger
  ) throws -> MLXArray {
    logger.info("InpaintUtilities: VAE encoding \(cgImage.width)x\(cgImage.height) -> \(pixelW)x\(pixelH)")

    #if canImport(CoreGraphics)
    guard let rgbaImage = convertToRGBA(cgImage) else {
      throw InpaintError.imageLoadFailed("Failed to convert to RGBA")
    }

    let imageArray = try QwenImageIO.resizedPixelArray(
      from: rgbaImage,
      width: pixelW,
      height: pixelH,
      addBatchDimension: true,
      dtype: .float32
    )
    guard imageArray.ndim == 4 else {
      throw InpaintError.imageLoadFailed("Image array has wrong dimensions: \(imageArray.ndim), expected 4")
    }

    let normalized = QwenImageIO.normalizeForEncoder(imageArray)
    MLX.eval(normalized)

    // Check for NaN values
    let hasNaN = MLX.any(MLX.isNaN(normalized)).item(Bool.self)
    if hasNaN {
      throw InpaintError.imageLoadFailed("Image contains NaN values after normalization")
    }

    GPU.clearCache()

    let encodedLatents = vae.encode(normalized)
    MLX.eval(encodedLatents)

    let latentChannels = vaeConfig.latentChannels
    let latents = encodedLatents[0..., 0..<latentChannels, 0..., 0...]
    let normalizedLatents = (latents - vaeConfig.shiftFactor) * vaeConfig.scalingFactor

    logger.info("InpaintUtilities: encoded to latent shape \(normalizedLatents.shape)")
    return normalizedLatents
    #else
    throw InpaintError.imageLoadFailed("CoreGraphics not available")
    #endif
  }

  // MARK: - Mask to Latent

  /// Convert a pixel-space mask image to latent-space mask.
  ///
  /// Input: CGImage mask where white (255) = region to regenerate, black (0) = preserve.
  /// Output: MLXArray of shape [1, 1, latentH, latentW] with 1.0 = regenerate, 0.0 = preserve.
  ///
  /// The mask is downsampled to latent dimensions using nearest-neighbor interpolation
  /// (preserves sharp mask boundaries).
  static func pixelMaskToLatent(_ maskImage: CGImage, latentH: Int, latentW: Int) throws -> MLXArray {
    #if canImport(CoreGraphics)
    guard let rgbaMask = convertToRGBA(maskImage) else {
      throw InpaintError.maskLoadFailed("Failed to convert mask to RGBA")
    }

    // Resize mask to latent dimensions
    let maskArray = try QwenImageIO.resizedPixelArray(
      from: rgbaMask,
      width: latentW,
      height: latentH,
      addBatchDimension: true,
      dtype: .float32,
      interpolation: .none  // Nearest-neighbor for sharp edges
    )

    // Convert to grayscale and binarize (threshold at 0.5)
    let grayscale = MLX.mean(maskArray, axis: 1, keepDims: true)  // [1, 1, H, W]
    let binarized = MLX.where(grayscale .>= 0.5, MLXArray(Float(1.0)), MLXArray(Float(0.0)))

    MLX.eval(binarized)
    return binarized
    #else
    throw InpaintError.maskLoadFailed("CoreGraphics not available")
    #endif
  }

  // MARK: - Image Compositing

  /// Composite generated output onto original image using mask.
  ///
  /// For latent-space inpainting, the blending happens in latent space during denoising.
  /// This method provides an additional pixel-space composite for clean edges.
  ///
  /// - Parameters:
  ///   - generated: The fully decoded generated image (MLXArray [1, C, H, W]).
  ///   - original: The original image data (PNG).
  ///   - mask: The mask data (PNG, white = generated region).
  ///   - width: Target pixel width.
  ///   - height: Target pixel height.
  /// - Returns: Composited MLXArray.
  static func compositeWithMask(
    generated: MLXArray,
    originalData: Data,
    maskData: Data,
    width: Int,
    height: Int
  ) throws -> MLXArray {
    #if canImport(CoreGraphics)
    let origCG = try loadCGImage(from: originalData)
    let maskCG = try loadCGImage(from: maskData)

    guard let origRGBA = convertToRGBA(origCG),
          let maskRGBA = convertToRGBA(maskCG) else {
      throw InpaintError.imageLoadFailed("Failed to convert images for compositing")
    }

    let origArray = try QwenImageIO.resizedPixelArray(
      from: origRGBA, width: width, height: height,
      addBatchDimension: true, dtype: .float32
    )
    let maskArray = try QwenImageIO.resizedPixelArray(
      from: maskRGBA, width: width, height: height,
      addBatchDimension: true, dtype: .float32
    )

    // Grayscale mask [1, 1, H, W]
    let maskGray = MLX.mean(maskArray, axis: 1, keepDims: true)
    // Broadcast to match image channels
    let maskBroadcast = MLX.broadcast(maskGray, to: generated.shape)

    // Blend: mask=1 → generated, mask=0 → original
    let composited = maskBroadcast * generated + (1.0 - maskBroadcast) * origArray
    return composited
    #else
    throw InpaintError.imageLoadFailed("CoreGraphics not available")
    #endif
  }

  // MARK: - Helpers

  #if canImport(CoreGraphics)
  /// Convert a CGImage to RGBA format (required for pixel array conversion).
  private static func convertToRGBA(_ image: CGImage) -> CGImage? {
    let width = image.width
    let height = image.height
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    guard let context = CGContext(
      data: nil, width: width, height: height,
      bitsPerComponent: 8, bytesPerRow: width * 4,
      space: colorSpace, bitmapInfo: bitmapInfo.rawValue
    ) else {
      return nil
    }

    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
  }
  #endif
}
