import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO

/// Image preprocessing and post-processing utilities for the SeedVR2 upscaler.
///
/// ## Preprocessing Pipeline
///
/// Prepares an input image for VAE encoding:
///
/// ```
/// Load CGImage
///   -> Compute scale: targetResolution / min(width, height)
///   -> Resize to (trueW, trueH), both even
///   -> Optional softness: downscale then upscale via bicubic
///   -> Pad to multiples of 16
///   -> Normalize: [0,255] -> [0,1] -> [-1,1]
///   -> CHW layout, add batch dim
///   -> Output: (1, 3, H_pad, W_pad)
/// ```
///
/// ## Color Correction
///
/// After the diffusion model produces an upscaled image, the colors may drift from
/// the input. Color correction addresses this through two stages:
///
/// 1. **Wavelet reconstruction**: Extracts high-frequency detail from the upscaled
///    result and low-frequency color structure from the original, then combines them.
///
/// 2. **LAB color transfer**: Converts both images to CIE LAB space, histogram-matches
///    the chrominance (a, b) channels, and blends luminance. This preserves the
///    fine detail of the upscaled result while matching the color palette of the input.
public enum SeedVR2Util {

  /// Result of image preprocessing.
  public struct PreprocessResult {
    /// The preprocessed image tensor: `(1, 3, H_padded, W_padded)`, values in [-1, 1].
    public let tensor: MLXArray
    /// True (unpadded) height after scaling.
    public let trueHeight: Int
    /// True (unpadded) width after scaling.
    public let trueWidth: Int
  }

  /// Preprocesses an image for VAE encoding.
  ///
  /// - Parameters:
  ///   - image: The input CGImage to upscale.
  ///   - targetResolution: Target resolution for the shortest side. Default `2048`.
  ///   - softness: Preprocessing softness in range `[0.0, 1.0]`. Maps to a blur factor
  ///     of `1.0` (no softening) to `8.0` (maximum softening). Default `0.0`.
  /// - Returns: A `PreprocessResult` containing the tensor and true dimensions.
  /// - Throws: `QwenImageIOError` if image processing fails.
  public static func preprocessImage(
    _ image: CGImage,
    targetResolution: Int = 2048,
    softness: Float = 0.0
  ) throws -> PreprocessResult {
    let w = image.width
    let h = image.height

    // Compute scale to bring the shortest side to targetResolution.
    let scale = Float(targetResolution) / Float(min(w, h))
    var trueW = Int(Float(w) * scale)
    var trueH = Int(Float(h) * scale)

    // Ensure both dimensions are even.
    trueW = (trueW / 2) * 2
    trueH = (trueH / 2) * 2

    // Map normalized softness [0, 1] to blur factor [1, 8].
    let clampedSoftness = max(0.0, min(1.0, softness))
    let factor = 1.0 + clampedSoftness * 7.0

    // Resize with optional softness (downscale then upscale).
    let resizedImage: CGImage
    if factor > 1.0 {
      let downW = max(2, Int(Float(trueW) / factor))
      let downH = max(2, Int(Float(trueH) / factor))
      let downscaled = try QwenImageIO.resizedCGImage(
        from: image, width: downW, height: downH,
        interpolation: .high
      )
      resizedImage = try QwenImageIO.resizedCGImage(
        from: downscaled, width: trueW, height: trueH,
        interpolation: .high
      )
    } else {
      resizedImage = try QwenImageIO.resizedCGImage(
        from: image, width: trueW, height: trueH,
        interpolation: .high
      )
    }

    // Pad to multiples of 16.
    let padW = (16 - (trueW % 16)) % 16
    let padH = (16 - (trueH % 16)) % 16
    let paddedW = trueW + padW
    let paddedH = trueH + padH

    let finalImage: CGImage
    if padW > 0 || padH > 0 {
      finalImage = try padImage(resizedImage, toWidth: paddedW, height: paddedH)
    } else {
      finalImage = resizedImage
    }

    // Convert to MLXArray: CHW layout, [0, 1] range, batch dim.
    var tensor = try QwenImageIO.array(
      from: finalImage, addBatchDimension: true, dtype: .float32
    )
    // tensor is now (1, 3, H, W) in [0, 1] from QwenImageIO.array

    // Normalize to [-1, 1].
    tensor = MLX.clip(tensor, min: 0, max: 1)
    tensor = tensor * 2.0 - 1.0

    return PreprocessResult(tensor: tensor, trueHeight: trueH, trueWidth: trueW)
  }

  /// Loads a CGImage from a file path.
  ///
  /// - Parameter path: Absolute path to an image file (PNG, JPEG, etc.).
  /// - Returns: The loaded CGImage.
  /// - Throws: If the file cannot be read or decoded.
  public static func loadImage(from path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
      throw SeedVR2UtilError.imageLoadFailed(path)
    }
    return cgImage
  }

  // MARK: - Color Correction

  /// Applies wavelet reconstruction and LAB color transfer to correct color drift.
  ///
  /// - Parameters:
  ///   - content: The upscaled (diffusion output) tensor, shape `(B, 3, H, W)`, values in [-1, 1].
  ///   - style: The preprocessed input (style reference) tensor, same shape and range.
  ///   - luminanceWeight: How much of the content luminance to preserve (0.0-1.0).
  ///     Default `0.8` (80% content luminance, 20% style-matched luminance).
  /// - Returns: Color-corrected tensor, same shape and range.
  public static func applyColorCorrection(
    content: MLXArray,
    style: MLXArray,
    luminanceWeight: Float = 0.8
  ) -> MLXArray {
    // Convert MLXArray to Float arrays for CPU-side processing.
    let contentF32 = content.asType(.float32)
    let styleF32 = style.asType(.float32)
    MLX.eval(contentF32)
    MLX.eval(styleF32)

    let shape = contentF32.shape  // [B, 3, H, W]
    let B = shape[0]
    let C = shape[1]
    let H = shape[2]
    let W = shape[3]

    guard C == 3 else { return content }

    var contentArr = contentF32.asArray(Float.self)
    let styleArr = styleF32.asArray(Float.self)

    // Wavelet reconstruction: high-freq from content + low-freq from style.
    waveletReconstruction(
      content: &contentArr, style: styleArr,
      B: B, C: C, H: H, W: W
    )

    // Denormalize from [-1,1] to [0,1] for LAB conversion.
    let pixelCount = B * H * W
    var contentRGB = [Float](repeating: 0, count: pixelCount * 3)
    var styleRGB = [Float](repeating: 0, count: pixelCount * 3)

    bchwToBhwc(contentArr, out: &contentRGB, B: B, C: C, H: H, W: W, denormalize: true)
    bchwToBhwc(styleArr, out: &styleRGB, B: B, C: C, H: H, W: W, denormalize: true)

    // Convert to LAB.
    var contentLAB = rgbToLAB(contentRGB, count: pixelCount)
    let styleLAB = rgbToLAB(styleRGB, count: pixelCount)

    // Histogram-match a and b channels per batch.
    for b in 0..<B {
      let offset = b * H * W
      let count = H * W

      // Match 'a' channel (index 1).
      histogramMatch(
        source: &contentLAB, sourceOffset: offset, sourceStride: 3, channel: 1,
        reference: styleLAB, refOffset: offset, refStride: 3, refChannel: 1,
        count: count
      )

      // Match 'b' channel (index 2).
      histogramMatch(
        source: &contentLAB, sourceOffset: offset, sourceStride: 3, channel: 2,
        reference: styleLAB, refOffset: offset, refStride: 3, refChannel: 2,
        count: count
      )

      // Blend luminance.
      if luminanceWeight < 1.0 {
        var srcL = [Float](repeating: 0, count: count)
        var refL = [Float](repeating: 0, count: count)
        var matchedL = [Float](repeating: 0, count: count)
        for i in 0..<count {
          srcL[i] = contentLAB[(offset + i) * 3]
          refL[i] = styleLAB[(offset + i) * 3]
        }
        histogramMatchChannel(source: &srcL, reference: refL, output: &matchedL, count: count)
        for i in 0..<count {
          let idx = (offset + i) * 3
          contentLAB[idx] = luminanceWeight * contentLAB[idx]
            + (1.0 - luminanceWeight) * matchedL[i]
        }
      }
    }

    // Convert back to RGB.
    let resultRGB = labToRGB(contentLAB, count: pixelCount)

    // Re-normalize to [-1, 1] and convert back to BCHW.
    var resultArr = [Float](repeating: 0, count: B * C * H * W)
    bhwcToBchw(resultRGB, out: &resultArr, B: B, C: C, H: H, W: W, renormalize: true)

    return MLXArray(resultArr, shape).asType(content.dtype)
  }

  // MARK: - Layout Conversion Helpers

  /// BCHW -> BHWC with optional [-1,1] -> [0,1] denormalization.
  private static func bchwToBhwc(
    _ src: [Float], out: inout [Float],
    B: Int, C: Int, H: Int, W: Int,
    denormalize: Bool
  ) {
    for b in 0..<B {
      for h in 0..<H {
        for w in 0..<W {
          for c in 0..<C {
            let bchwIdx = b * (C * H * W) + c * (H * W) + h * W + w
            let bhwcIdx = b * (H * W * C) + h * (W * C) + w * C + c
            var val = src[bchwIdx]
            if denormalize {
              val = max(0, min(1, (val + 1.0) * 0.5))
            }
            out[bhwcIdx] = val
          }
        }
      }
    }
  }

  /// BHWC -> BCHW with optional [0,1] -> [-1,1] renormalization.
  private static func bhwcToBchw(
    _ src: [Float], out: inout [Float],
    B: Int, C: Int, H: Int, W: Int,
    renormalize: Bool
  ) {
    for b in 0..<B {
      for h in 0..<H {
        for w in 0..<W {
          for c in 0..<C {
            let bhwcIdx = b * (H * W * C) + h * (W * C) + w * C + c
            let bchwIdx = b * (C * H * W) + c * (H * W) + h * W + w
            var val = src[bhwcIdx]
            if renormalize {
              val = max(0, min(1, val)) * 2.0 - 1.0
            }
            out[bchwIdx] = val
          }
        }
      }
    }
  }

  // MARK: - Wavelet Decomposition

  /// Applies a blur at the given radius using a 3x3 binomial kernel with edge-replicated padding.
  private static func waveletBlurChannel(
    _ channel: [Float], H: Int, W: Int, radius: Int
  ) -> [Float] {
    let r = max(1, min(radius, max(1, min(H, W) / 8)))
    var output = [Float](repeating: 0, count: H * W)

    let offsets: [(Int, Int, Float)] = [
      (-1, -1, 0.0625), ( 0, -1, 0.125), ( 1, -1, 0.0625),
      (-1,  0, 0.125),  ( 0,  0, 0.25),  ( 1,  0, 0.125),
      (-1,  1, 0.0625), ( 0,  1, 0.125), ( 1,  1, 0.0625)
    ]

    for y in 0..<H {
      for x in 0..<W {
        var sum: Float = 0
        for (dx, dy, wt) in offsets {
          let sy = max(0, min(H - 1, y + dy * r))
          let sx = max(0, min(W - 1, x + dx * r))
          sum += wt * channel[sy * W + sx]
        }
        output[y * W + x] = sum
      }
    }

    return output
  }

  /// Performs 5-level wavelet decomposition and reconstruction.
  private static func waveletReconstruction(
    content: inout [Float], style: [Float],
    B: Int, C: Int, H: Int, W: Int,
    levels: Int = 5
  ) {
    guard content.count == style.count else { return }
    let planeSize = H * W

    for b in 0..<B {
      for c in 0..<C {
        let baseOffset = b * (C * H * W) + c * planeSize

        // Extract content channel.
        var contentHigh = [Float](repeating: 0, count: planeSize)
        var contentCur = [Float](repeating: 0, count: planeSize)
        for i in 0..<planeSize {
          contentCur[i] = content[baseOffset + i]
        }

        for level in 0..<levels {
          let radius = 1 << level
          let blurred = waveletBlurChannel(contentCur, H: H, W: W, radius: radius)
          for i in 0..<planeSize {
            contentHigh[i] += contentCur[i] - blurred[i]
            contentCur[i] = blurred[i]
          }
        }

        // Extract style channel and decompose for low frequencies.
        var styleCur = [Float](repeating: 0, count: planeSize)
        for i in 0..<planeSize {
          styleCur[i] = style[baseOffset + i]
        }
        for level in 0..<levels {
          let radius = 1 << level
          styleCur = waveletBlurChannel(styleCur, H: H, W: W, radius: radius)
        }

        // Reconstruct: high-freq content + low-freq style.
        for i in 0..<planeSize {
          content[baseOffset + i] = max(-1, min(1, contentHigh[i] + styleCur[i]))
        }
      }
    }
  }

  // MARK: - Color Space Conversion

  @inline(__always)
  private static func srgbToLinear(_ x: Float) -> Float {
    x > 0.04045 ? powf((x + 0.055) / 1.055, 2.4) : x / 12.92
  }

  @inline(__always)
  private static func linearToSRGB(_ x: Float) -> Float {
    x > 0.0031308 ? 1.055 * powf(max(x, 0), 1.0 / 2.4) - 0.055 : 12.92 * x
  }

  /// Converts RGB pixels (interleaved, [0,1]) to LAB (interleaved).
  private static func rgbToLAB(_ rgb: [Float], count: Int) -> [Float] {
    var lab = [Float](repeating: 0, count: count * 3)

    let m00: Float = 0.4124564, m01: Float = 0.3575761, m02: Float = 0.1804375
    let m10: Float = 0.2126729, m11: Float = 0.7151522, m12: Float = 0.0721750
    let m20: Float = 0.0193339, m21: Float = 0.1191920, m22: Float = 0.9503041

    let eps: Float = 6.0 / 29.0
    let eps3: Float = eps * eps * eps
    let kappa: Float = (29.0 / 3.0) * (29.0 / 3.0) * (29.0 / 3.0)

    for i in 0..<count {
      let idx = i * 3
      let r = srgbToLinear(rgb[idx])
      let g = srgbToLinear(rgb[idx + 1])
      let b = srgbToLinear(rgb[idx + 2])

      var x = m00 * r + m01 * g + m02 * b
      var y = m10 * r + m11 * g + m12 * b
      var z = m20 * r + m21 * g + m22 * b

      x /= 0.95047
      z /= 1.08883

      let fx = x > eps3 ? cbrtf(x) : (kappa * x + 16.0) / 116.0
      let fy = y > eps3 ? cbrtf(y) : (kappa * y + 16.0) / 116.0
      let fz = z > eps3 ? cbrtf(z) : (kappa * z + 16.0) / 116.0

      lab[idx]     = 116.0 * fy - 16.0
      lab[idx + 1] = 500.0 * (fx - fy)
      lab[idx + 2] = 200.0 * (fy - fz)
    }

    return lab
  }

  /// Converts LAB pixels (interleaved) to RGB ([0,1], interleaved).
  private static func labToRGB(_ lab: [Float], count: Int) -> [Float] {
    var rgb = [Float](repeating: 0, count: count * 3)

    let eps: Float = 6.0 / 29.0
    let kappa: Float = (29.0 / 3.0) * (29.0 / 3.0) * (29.0 / 3.0)

    let m00: Float =  3.2404542, m01: Float = -1.5371385, m02: Float = -0.4985314
    let m10: Float = -0.9692660, m11: Float =  1.8760108, m12: Float =  0.0415560
    let m20: Float =  0.0556434, m21: Float = -0.2040259, m22: Float =  1.0572252

    for i in 0..<count {
      let idx = i * 3
      let L = lab[idx]
      let a = lab[idx + 1]
      let b = lab[idx + 2]

      let fy = (L + 16.0) / 116.0
      let fx = a / 500.0 + fy
      let fz = fy - b / 200.0

      var x = fx > eps ? fx * fx * fx : (116.0 * fx - 16.0) / kappa
      var y = fy > eps ? fy * fy * fy : (116.0 * fy - 16.0) / kappa
      var z = fz > eps ? fz * fz * fz : (116.0 * fz - 16.0) / kappa

      x *= 0.95047
      z *= 1.08883

      let rLin = m00 * x + m01 * y + m02 * z
      let gLin = m10 * x + m11 * y + m12 * z
      let bLin = m20 * x + m21 * y + m22 * z

      rgb[idx]     = linearToSRGB(rLin)
      rgb[idx + 1] = linearToSRGB(gLin)
      rgb[idx + 2] = linearToSRGB(bLin)
    }

    return rgb
  }

  // MARK: - Histogram Matching

  /// In-place histogram matching of a single channel within interleaved LAB data.
  private static func histogramMatch(
    source: inout [Float], sourceOffset: Int, sourceStride: Int, channel: Int,
    reference: [Float], refOffset: Int, refStride: Int, refChannel: Int,
    count: Int
  ) {
    var srcValues = [Float](repeating: 0, count: count)
    var refValues = [Float](repeating: 0, count: count)
    for i in 0..<count {
      srcValues[i] = source[(sourceOffset + i) * sourceStride + channel]
      refValues[i] = reference[(refOffset + i) * refStride + refChannel]
    }

    var matched = [Float](repeating: 0, count: count)
    histogramMatchChannel(source: &srcValues, reference: refValues, output: &matched, count: count)

    for i in 0..<count {
      source[(sourceOffset + i) * sourceStride + channel] = matched[i]
    }
  }

  /// Histogram matching via sort-rank transfer for a single channel.
  private static func histogramMatchChannel(
    source: inout [Float], reference: [Float], output: inout [Float], count: Int
  ) {
    var srcIndices = Array(0..<count)
    srcIndices.sort { source[$0] < source[$1] }

    var refSorted = reference
    refSorted.sort()

    var invPerm = [Int](repeating: 0, count: count)
    for i in 0..<count {
      invPerm[srcIndices[i]] = i
    }

    for i in 0..<count {
      output[i] = refSorted[invPerm[i]]
    }
  }

  // MARK: - Image Padding

  private static func padImage(_ image: CGImage, toWidth width: Int, height: Int) throws -> CGImage {
    guard let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) else {
      throw SeedVR2UtilError.paddingFailed
    }
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw SeedVR2UtilError.paddingFailed
    }

    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    let drawY = height - image.height
    context.draw(image, in: CGRect(x: 0, y: drawY, width: image.width, height: image.height))

    guard let padded = context.makeImage() else {
      throw SeedVR2UtilError.paddingFailed
    }
    return padded
  }

  /// Converts an MLXArray in [-1,1] to a CGImage.
  ///
  /// - Parameters:
  ///   - tensor: The image tensor, shape `(B, 3, H, W)`, `(B, 3, T, H, W)`, or `(3, H, W)`.
  ///   - cropHeight: If provided, crop to this height before converting.
  ///   - cropWidth: If provided, crop to this width before converting.
  /// - Returns: The resulting CGImage.
  /// - Throws: If the tensor has an unexpected shape.
  public static func tensorToImage(
    _ tensor: MLXArray,
    cropHeight: Int? = nil,
    cropWidth: Int? = nil
  ) throws -> CGImage {
    var t = tensor.asType(.float32)

    // Handle 5D: (B, C, T, H, W) -> take first batch, squeeze T
    if t.ndim == 5 {
      t = t[0]  // (C, T, H, W)
      if t.dim(1) == 1 {
        t = t[0..., 0, 0..., 0...]  // (C, H, W)
      }
    }

    // Handle 4D: (B, C, H, W) -> take first batch
    if t.ndim == 4 {
      t = t[0]
    }

    guard t.ndim == 3 && t.dim(0) == 3 else {
      throw SeedVR2UtilError.invalidTensorShape(Array(t.shape))
    }

    // Crop if requested.
    if let ch = cropHeight, let cw = cropWidth {
      t = t[0..., ..<ch, ..<cw]
    }

    // Denormalize [-1, 1] -> [0, 1].
    t = (t + 1.0) / 2.0

    return try QwenImageIO.image(from: t)
  }
}

/// Errors specific to SeedVR2 utility operations.
public enum SeedVR2UtilError: Error, CustomStringConvertible {
  case imageLoadFailed(String)
  case paddingFailed
  case invalidTensorShape([Int])

  public var description: String {
    switch self {
    case .imageLoadFailed(let path):
      return "Failed to load image from \(path)"
    case .paddingFailed:
      return "Failed to pad image to target dimensions"
    case .invalidTensorShape(let shape):
      return "Invalid tensor shape for image conversion: \(shape)"
    }
  }
}
#endif
