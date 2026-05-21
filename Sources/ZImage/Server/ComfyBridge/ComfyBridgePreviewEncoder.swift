// ComfyBridgePreviewEncoder.swift — Binary WebSocket preview image encoder
//
// Encodes preview images with the ComfyUI binary header format for Krita
// AI Diffusion plugin live denoising previews.
//
// Binary frame format (ComfyUI protocol — struct.pack(">II", type, format)):
//   Bytes 0-3: event type as UInt32 big-endian (1 = PREVIEW_IMAGE)
//   Bytes 4-7: image format as UInt32 big-endian (1 = JPEG, 2 = PNG)
//   Bytes 8+:  encoded image data

import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Encodes preview images for ComfyUI binary WebSocket frames.
enum ComfyBridgePreviewEncoder {

  /// ComfyUI binary frame event types.
  enum EventType: UInt32 {
    case previewImage = 1
  }

  /// ComfyUI binary frame image formats.
  enum ImageFormat: UInt32 {
    case jpeg = 1
    case png = 2
  }

  /// Maximum preview dimension (longest edge).
  static let defaultPreviewSize = 256

  /// Default JPEG compression quality for previews (0.0-1.0).
  static let defaultJPEGQuality: CGFloat = 0.6

  /// Build a ComfyUI binary preview frame from PNG image data.
  ///
  /// Takes a full-resolution PNG, decodes it, downscales to preview size,
  /// re-encodes as JPEG, and prepends the 8-byte ComfyUI binary header.
  ///
  /// - Parameters:
  ///   - pngData: Full-resolution PNG image data.
  ///   - maxDimension: Maximum preview dimension (longest edge). Default 256.
  ///   - jpegQuality: JPEG compression quality (0.0-1.0). Default 0.6.
  ///   - imageFormat: Output format for the preview. Default `.jpeg`.
  /// - Returns: Binary frame data ready to send via WebSocket, or nil on failure.
  static func encodePreviewFrame(
    fromPNG pngData: Data,
    maxDimension: Int = defaultPreviewSize,
    jpegQuality: CGFloat = defaultJPEGQuality,
    imageFormat: ImageFormat = .jpeg
  ) -> Data? {
    // Decode the PNG to a CGImage.
    guard let imageSource = CGImageSourceCreateWithData(pngData as CFData, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
      return nil
    }

    return encodePreviewFrame(
      fromCGImage: cgImage,
      maxDimension: maxDimension,
      jpegQuality: jpegQuality,
      imageFormat: imageFormat
    )
  }

  /// Build a ComfyUI binary preview frame from a CGImage.
  ///
  /// Downscales the image to preview size, encodes as JPEG, and prepends
  /// the 8-byte ComfyUI binary header.
  ///
  /// - Parameters:
  ///   - cgImage: Source image.
  ///   - maxDimension: Maximum preview dimension (longest edge). Default 256.
  ///   - jpegQuality: JPEG compression quality (0.0-1.0). Default 0.6.
  ///   - imageFormat: Output format for the preview. Default `.jpeg`.
  /// - Returns: Binary frame data ready to send via WebSocket, or nil on failure.
  static func encodePreviewFrame(
    fromCGImage cgImage: CGImage,
    maxDimension: Int = defaultPreviewSize,
    jpegQuality: CGFloat = defaultJPEGQuality,
    imageFormat: ImageFormat = .jpeg
  ) -> Data? {
    // Calculate preview dimensions preserving aspect ratio.
    let srcWidth = cgImage.width
    let srcHeight = cgImage.height
    let (previewWidth, previewHeight) = previewDimensions(
      srcWidth: srcWidth, srcHeight: srcHeight, maxDimension: maxDimension
    )

    // Downscale the image.
    guard let previewImage = downscale(cgImage, width: previewWidth, height: previewHeight) else {
      return nil
    }

    // Encode as JPEG.
    guard let jpegData = encodeJPEG(previewImage, quality: jpegQuality) else {
      return nil
    }

    // Build the binary frame: 8-byte header + JPEG data.
    return buildBinaryFrame(
      imageFormat: imageFormat,
      imageData: jpegData
    )
  }

  /// Build a ComfyUI binary preview frame from raw RGBA pixel data.
  ///
  /// Used for latent-to-RGB approximations where we already have pixel data
  /// and just need to encode + frame it. Skips CGImage decode step.
  ///
  /// - Parameters:
  ///   - rgbaData: Raw RGBA pixel bytes (width * height * 4 bytes).
  ///   - width: Image width in pixels.
  ///   - height: Image height in pixels.
  ///   - maxDimension: Maximum preview dimension (longest edge). Default 256.
  ///   - jpegQuality: JPEG compression quality (0.0-1.0). Default 0.6.
  /// - Returns: Binary frame data ready to send via WebSocket, or nil on failure.
  static func encodePreviewFrame(
    fromRGBA rgbaData: Data,
    width: Int,
    height: Int,
    maxDimension: Int = defaultPreviewSize,
    jpegQuality: CGFloat = defaultJPEGQuality
  ) -> Data? {
    guard width > 0, height > 0, rgbaData.count >= width * height * 4 else { return nil }

    // Create CGImage from raw RGBA.
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    guard let provider = CGDataProvider(data: rgbaData as CFData),
          let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
          ) else {
      return nil
    }

    return encodePreviewFrame(
      fromCGImage: cgImage,
      maxDimension: maxDimension,
      jpegQuality: jpegQuality
    )
  }

  // MARK: - Internal Helpers

  /// Calculate preview dimensions preserving aspect ratio.
  static func previewDimensions(
    srcWidth: Int, srcHeight: Int, maxDimension: Int
  ) -> (width: Int, height: Int) {
    let longestEdge = max(srcWidth, srcHeight)
    guard longestEdge > maxDimension else {
      return (srcWidth, srcHeight)
    }
    let scale = Double(maxDimension) / Double(longestEdge)
    let width = max(1, Int(Double(srcWidth) * scale))
    let height = max(1, Int(Double(srcHeight) * scale))
    return (width, height)
  }

  /// Downscale a CGImage using bilinear interpolation.
  private static func downscale(_ image: CGImage, width: Int, height: Int) -> CGImage? {
    guard width > 0, height > 0 else { return nil }
    guard let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) else {
      return nil
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
      return nil
    }
    // Use low quality for speed — this is a preview thumbnail.
    context.interpolationQuality = .low
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
  }

  /// Encode a CGImage as JPEG data.
  private static func encodeJPEG(_ image: CGImage, quality: CGFloat) -> Data? {
    let mutableData = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(
      mutableData as CFMutableData,
      "public.jpeg" as CFString,
      1,
      nil
    ) else {
      return nil
    }

    let options: [CFString: Any] = [
      kCGImageDestinationLossyCompressionQuality: quality
    ]
    CGImageDestinationAddImage(destination, image, options as CFDictionary)

    guard CGImageDestinationFinalize(destination) else {
      return nil
    }
    return mutableData as Data
  }

  /// Build the ComfyUI binary frame with 8-byte header + image data.
  ///
  /// Matches ComfyUI's Python: `struct.pack(">II", event_type, format)`
  ///
  /// Header layout (big-endian):
  ///   [0..3] UInt32 event type (1 = PREVIEW_IMAGE)
  ///   [4..7] UInt32 image format (1 = JPEG, 2 = PNG)
  static func buildBinaryFrame(
    eventType: EventType = .previewImage,
    imageFormat: ImageFormat,
    imageData: Data
  ) -> Data {
    var frame = Data(capacity: 8 + imageData.count)

    // Event type — UInt32 big-endian.
    var eventTypeValue = eventType.rawValue.bigEndian
    frame.append(Data(bytes: &eventTypeValue, count: 4))

    // Image format — UInt32 big-endian.
    var formatValue = imageFormat.rawValue.bigEndian
    frame.append(Data(bytes: &formatValue, count: 4))

    // Image data.
    frame.append(imageData)

    return frame
  }
}
#endif
