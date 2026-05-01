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

  /// Convert a pixel-space mask image to latent-space mask with optional grow and feather.
  ///
  /// Input: CGImage mask where white (255) = region to regenerate, black (0) = preserve.
  /// Output: MLXArray of shape [1, 1, latentH, latentW] with 1.0 = regenerate, 0.0 = preserve.
  ///
  /// Processing pipeline:
  /// 1. Convert to grayscale float array at working resolution
  /// 2. Grow (dilate) the mask using conv2d with ones kernel + threshold
  /// 3. Feather (blur) the mask edges with separable Gaussian convolution
  /// 4. Downsample to latent dimensions
  static func pixelMaskToLatent(
    _ maskImage: CGImage,
    latentH: Int,
    latentW: Int,
    grow: Int = 0,
    feather: Int = 0,
    cropX: Int = 0,
    cropY: Int = 0,
    cropWidth: Int = 0,
    cropHeight: Int = 0,
    logger: Logger? = nil
  ) throws -> MLXArray {
    #if canImport(CoreGraphics)
    // If crop coordinates are provided, crop the full-canvas mask to selection bounds.
    // The Krita AI Diffusion plugin sends masks at full canvas resolution, but the
    // source image is pre-cropped to the selection bounding box. The ImageCrop node
    // in the workflow specifies the crop region.
    var effectiveMask = maskImage
    // Crop when mask dimensions differ from the target generation dimensions.
    // Krita "Selection bounds" sends mask at full canvas resolution but source image
    // is pre-cropped to the selection bounding box.
    let needsCrop = cropWidth > 0 && cropHeight > 0 && (maskImage.width != cropWidth || maskImage.height != cropHeight)
    if needsCrop {
      let cropRect = CGRect(x: cropX, y: cropY, width: cropWidth, height: cropHeight)
      if let cropped = maskImage.cropping(to: cropRect) {
        logger?.info("InpaintUtilities: cropped mask from \(maskImage.width)x\(maskImage.height) to \(cropped.width)x\(cropped.height) at (\(cropX),\(cropY))")
        effectiveMask = cropped
      } else {
        logger?.warning("InpaintUtilities: mask crop failed for rect \(cropRect), using full mask")
      }
    }

    guard let rgbaMask = convertToRGBA(effectiveMask) else {
      throw InpaintError.maskLoadFailed("Failed to convert mask to RGBA")
    }

    // Work at pixel resolution for grow/feather, then downsample.
    // Cap at 8x latent dims to avoid excessive memory usage.
    let workH = min(effectiveMask.height, latentH * 8)
    let workW = min(effectiveMask.width, latentW * 8)

    // Load mask at working resolution — resizedPixelArray returns NCHW [1, 3, H, W]
    let maskArrayNCHW = try QwenImageIO.resizedPixelArray(
      from: rgbaMask,
      width: workW,
      height: workH,
      addBatchDimension: true,
      dtype: .float32
    )
    // Transpose to NHWC [1, H, W, 3] for conv2d operations downstream
    let maskArray = maskArrayNCHW.transposed(0, 2, 3, 1)

    // Convert to single-channel grayscale [1, H, W, 1]
    var mask = MLX.mean(maskArray, axis: -1, keepDims: true)
    // Binarize at 0.5 threshold
    // Pre-binarization stats
    let preBinMin = MLX.min(mask).item(Float.self)
    let preBinMax = MLX.max(mask).item(Float.self)
    let preBinMean = MLX.mean(mask).item(Float.self)
    let preBinNonzero = MLX.sum(MLX.where(mask .> 0.01, MLXArray(Float(1.0)), MLXArray(Float(0.0)))).item(Float.self)
    logger?.info("InpaintUtilities: pre-binarize grayscale — min=\(preBinMin), max=\(preBinMax), mean=\(String(format: "%.4f", preBinMean)), nonzero(>0.01)=\(Int(preBinNonzero))/\(workH*workW)")

    mask = MLX.where(mask .>= 0.5, MLXArray(Float(1.0)), MLXArray(Float(0.0)))
    MLX.eval(mask)
    
    let postBinWhite = MLX.sum(mask).item(Float.self)
    logger?.info("InpaintUtilities: post-binarize — white=\(Int(postBinWhite))/\(workH*workW) (\(String(format: "%.1f", (postBinWhite / Float(workH*workW)) * 100))%)")

    // Scale grow/feather from original pixel space to working resolution
    let scale = Float(workW) / Float(effectiveMask.width)

    // --- Grow (dilate) using conv2d with ones kernel + threshold ---
    let scaledGrow = max(0, Int(Float(grow) * scale))
    if scaledGrow > 0 {
      let kernelSize = scaledGrow * 2 + 1
      // Conv2d weight: [Cout, kH, kW, Cin] = [1, k, k, 1], all ones
      let onesKernel = MLXArray.ones([1, kernelSize, kernelSize, 1])
      // Pad input to maintain spatial dimensions
      let padded = MLX.padded(mask, widths: [IntOrPair((0, 0)), IntOrPair((scaledGrow, scaledGrow)), IntOrPair((scaledGrow, scaledGrow)), IntOrPair((0, 0))])
      // Convolve — any overlap with a white pixel gives a positive value
      let dilated = conv2d(padded, onesKernel)
      // Threshold: > 0 means at least one white pixel was in the kernel
      mask = MLX.where(dilated .> 0, MLXArray(Float(1.0)), MLXArray(Float(0.0)))
      MLX.eval(mask)
      logger?.info("InpaintUtilities: mask grown by \(scaledGrow)px (kernel \(kernelSize))")
    }

    // --- Feather (Gaussian blur) using separable convolution ---
    let scaledFeather = max(0, Int(Float(feather) * scale))
    if scaledFeather > 0 {
      let sigma = Float(scaledFeather) / 3.0  // 3-sigma rule
      let kSize = scaledFeather * 2 + 1

      // Build 1D Gaussian kernel
      var kernel1D: [Float] = []
      var sum: Float = 0
      for i in 0..<kSize {
        let x = Float(i - scaledFeather)
        let g = exp(-(x * x) / (2.0 * sigma * sigma))
        kernel1D.append(g)
        sum += g
      }
      kernel1D = kernel1D.map { $0 / sum }

      // Horizontal pass: weight [1, 1, kSize, 1]
      let hKernel = MLXArray(kernel1D).reshaped([1, 1, kSize, 1])
      let hPadded = MLX.padded(mask, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((scaledFeather, scaledFeather)), IntOrPair((0, 0))], mode: .edge)
      let blurH = conv2d(hPadded, hKernel)

      // Vertical pass: weight [1, kSize, 1, 1]
      let vKernel = MLXArray(kernel1D).reshaped([1, kSize, 1, 1])
      let vPadded = MLX.padded(blurH, widths: [IntOrPair((0, 0)), IntOrPair((scaledFeather, scaledFeather)), IntOrPair((0, 0)), IntOrPair((0, 0))], mode: .edge)
      mask = conv2d(vPadded, vKernel)
      MLX.eval(mask)
      logger?.info("InpaintUtilities: mask feathered with radius \(scaledFeather)px")
    }

    // --- Downsample to latent dimensions ---
    // Use QwenImageIO for reliable resize, then extract single channel
    // First convert back to CGImage for resize
    let clampedMask = MLX.clip(mask, min: 0.0, max: 1.0)

    // Simple nearest-neighbor downsample via stride
    // mask is [1, workH, workW, 1], need [1, 1, latentH, latentW]
    let strideH = max(1, workH / latentH)
    let strideW = max(1, workW / latentW)
    let downsampled = clampedMask[0..., .stride(by: strideH), .stride(by: strideW), 0...]
    // Trim or pad to exact latent dimensions
    let trimH = min(downsampled.dim(1), latentH)
    let trimW = min(downsampled.dim(2), latentW)
    var result = downsampled[0..<1, 0..<trimH, 0..<trimW]

    // If trimmed dimensions don't match, pad with zeros
    if trimH < latentH || trimW < latentW {
      result = MLX.padded(result, widths: [
        IntOrPair((0, 0)),
        IntOrPair((0, latentH - trimH)),
        IntOrPair((0, latentW - trimW))
      ])
    }

    // Reshape to [1, 1, latentH, latentW] for NCHW format expected by the denoising loop
    let final = result.reshaped([1, 1, latentH, latentW])
    MLX.eval(final)

    let totalPixels = Float(latentH * latentW)
    let regeneratePixels = final.sum().item(Float.self)
    let pct = (regeneratePixels / totalPixels) * 100.0
    logger?.info("InpaintUtilities: final mask shape \(final.shape), range [\(final.min().item(Float.self)), \(final.max().item(Float.self))], regenerate=\(Int(regeneratePixels))/\(Int(totalPixels)) (\(String(format: "%.1f", pct))%)")
    return final
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
