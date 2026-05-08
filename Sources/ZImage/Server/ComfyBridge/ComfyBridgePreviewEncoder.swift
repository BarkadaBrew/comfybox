// ComfyBridgePreviewEncoder.swift — Binary WebSocket preview image encoder
//
// Encodes preview images with the ComfyUI binary header format for Krita
// AI Diffusion plugin live denoising previews.
//
// Binary frame format (ComfyUI protocol):
//   Bytes 0-3: event type as UInt32 big-endian (1 = preview, 2 = final)
//   Bytes 4-5: output_id as UInt16 big-endian (which output, usually 0)
//   Bytes 6-7: image format as UInt16 big-endian (1 = JPEG, 2 = PNG)
//   Bytes 8+:  JPEG or PNG encoded image data

import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Encodes preview images for ComfyUI binary WebSocket frames.
enum ComfyBridgePreviewEncoder {

  /// ComfyUI binary frame event types.
  enum EventType: UInt32 {
    case preview = 1
    case finalImage = 2
  }

  /// ComfyUI binary frame image formats.
  enum ImageFormat: UInt16 {
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
  ///   - eventType: Binary frame event type. Default `.preview`.
  ///   - outputId: Output node ID. Default 0.
  /// - Returns: Binary frame data ready to send via WebSocket, or nil on failure.
  static func encodePreviewFrame(
    fromPNG pngData: Data,
    maxDimension: Int = defaultPreviewSize,
    jpegQuality: CGFloat = defaultJPEGQuality,
    eventType: EventType = .preview,
    outputId: UInt16 = 0
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
      eventType: eventType,
      outputId: outputId
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
  ///   - eventType: Binary frame event type. Default `.preview`.
  ///   - outputId: Output node ID. Default 0.
  /// - Returns: Binary frame data ready to send via WebSocket, or nil on failure.
  static func encodePreviewFrame(
    fromCGImage cgImage: CGImage,
    maxDimension: Int = defaultPreviewSize,
    jpegQuality: CGFloat = defaultJPEGQuality,
    eventType: EventType = .preview,
    outputId: UInt16 = 0
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
      eventType: eventType,
      outputId: outputId,
      imageFormat: .jpeg,
      imageData: jpegData
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
  /// Header layout (big-endian):
  ///   [0..3] UInt32 event type (1=preview, 2=final)
  ///   [4..5] UInt16 output_id
  ///   [6..7] UInt16 image format (1=JPEG, 2=PNG)
  static func buildBinaryFrame(
    eventType: EventType,
    outputId: UInt16,
    imageFormat: ImageFormat,
    imageData: Data
  ) -> Data {
    var frame = Data(capacity: 8 + imageData.count)

    // Event type — UInt32 big-endian.
    var eventTypeValue = eventType.rawValue.bigEndian
    frame.append(Data(bytes: &eventTypeValue, count: 4))

    // Output ID — UInt16 big-endian.
    var outputIdValue = outputId.bigEndian
    frame.append(Data(bytes: &outputIdValue, count: 2))

    // Image format — UInt16 big-endian.
    var formatValue = imageFormat.rawValue.bigEndian
    frame.append(Data(bytes: &formatValue, count: 2))

    // Image data.
    frame.append(imageData)

    return frame
  }
}
#endif
