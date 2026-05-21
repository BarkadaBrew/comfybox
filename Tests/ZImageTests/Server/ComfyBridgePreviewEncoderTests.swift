import XCTest
@testable import ZImage

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO

final class ComfyBridgePreviewEncoderTests: XCTestCase {

  // MARK: - Binary Frame Header Format

  /// Verify the binary header matches ComfyUI's `struct.pack(">II", type, format)`.
  /// Header: [0..3] UInt32 event_type, [4..7] UInt32 image_format — both big-endian.
  func testBinaryFrameHeaderFormat() {
    let imageData = Data(repeating: 0xFF, count: 100)
    let frame = ComfyBridgePreviewEncoder.buildBinaryFrame(
      eventType: .previewImage,
      imageFormat: .jpeg,
      imageData: imageData
    )

    // Total size = 8-byte header + image data.
    XCTAssertEqual(frame.count, 8 + imageData.count)

    // Verify event type = 1 (PREVIEW_IMAGE) as big-endian UInt32.
    let eventType = frame.withUnsafeBytes { ptr -> UInt32 in
      ptr.load(fromByteOffset: 0, as: UInt32.self).bigEndian
    }
    XCTAssertEqual(eventType, 1, "Event type should be 1 (PREVIEW_IMAGE)")

    // Verify image format = 1 (JPEG) as big-endian UInt32.
    let imageFormat = frame.withUnsafeBytes { ptr -> UInt32 in
      ptr.load(fromByteOffset: 4, as: UInt32.self).bigEndian
    }
    XCTAssertEqual(imageFormat, 1, "Image format should be 1 (JPEG)")

    // Verify image data follows the header.
    let payload = frame.suffix(from: 8)
    XCTAssertEqual(payload, imageData)
  }

  func testBinaryFrameHeaderPNG() {
    let imageData = Data(repeating: 0xAB, count: 50)
    let frame = ComfyBridgePreviewEncoder.buildBinaryFrame(
      imageFormat: .png,
      imageData: imageData
    )

    XCTAssertEqual(frame.count, 8 + 50)

    let imageFormat = frame.withUnsafeBytes { ptr -> UInt32 in
      ptr.load(fromByteOffset: 4, as: UInt32.self).bigEndian
    }
    XCTAssertEqual(imageFormat, 2, "Image format should be 2 (PNG)")
  }

  // MARK: - Preview Dimensions

  func testPreviewDimensionsDownscale() {
    // Landscape image larger than max dimension.
    let (w, h) = ComfyBridgePreviewEncoder.previewDimensions(
      srcWidth: 1024, srcHeight: 768, maxDimension: 256
    )
    XCTAssertEqual(w, 256, "Longest edge should be clamped to maxDimension")
    XCTAssertEqual(h, 192, "Height should scale proportionally")
  }

  func testPreviewDimensionsPortrait() {
    let (w, h) = ComfyBridgePreviewEncoder.previewDimensions(
      srcWidth: 768, srcHeight: 1024, maxDimension: 256
    )
    XCTAssertEqual(w, 192)
    XCTAssertEqual(h, 256)
  }

  func testPreviewDimensionsAlreadySmall() {
    // Image smaller than max dimension — no scaling.
    let (w, h) = ComfyBridgePreviewEncoder.previewDimensions(
      srcWidth: 128, srcHeight: 96, maxDimension: 256
    )
    XCTAssertEqual(w, 128)
    XCTAssertEqual(h, 96)
  }

  func testPreviewDimensionsSquare() {
    let (w, h) = ComfyBridgePreviewEncoder.previewDimensions(
      srcWidth: 512, srcHeight: 512, maxDimension: 256
    )
    XCTAssertEqual(w, 256)
    XCTAssertEqual(h, 256)
  }

  // MARK: - CGImage Encoding

  func testEncodePreviewFrameFromCGImage() {
    // Create a small test CGImage (4x4 red pixels).
    guard let cgImage = makeTestCGImage(width: 4, height: 4) else {
      XCTFail("Failed to create test CGImage")
      return
    }

    let frame = ComfyBridgePreviewEncoder.encodePreviewFrame(
      fromCGImage: cgImage,
      maxDimension: 256,
      jpegQuality: 0.5
    )

    XCTAssertNotNil(frame, "Should produce a valid binary frame")
    if let frame = frame {
      XCTAssertGreaterThan(frame.count, 8, "Frame should have header + JPEG data")

      // Verify header event type.
      let eventType = frame.withUnsafeBytes { ptr -> UInt32 in
        ptr.load(fromByteOffset: 0, as: UInt32.self).bigEndian
      }
      XCTAssertEqual(eventType, 1)
    }
  }

  // MARK: - RGBA Encoding

  func testEncodePreviewFrameFromRGBA() {
    let width = 8
    let height = 8
    // Create RGBA pixel data (solid blue).
    var rgbaData = Data(count: width * height * 4)
    rgbaData.withUnsafeMutableBytes { ptr in
      guard let base = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for i in 0..<(width * height) {
        base[i * 4 + 0] = 0     // R
        base[i * 4 + 1] = 0     // G
        base[i * 4 + 2] = 255   // B
        base[i * 4 + 3] = 255   // A
      }
    }

    let frame = ComfyBridgePreviewEncoder.encodePreviewFrame(
      fromRGBA: rgbaData,
      width: width,
      height: height,
      maxDimension: 256,
      jpegQuality: 0.6
    )

    XCTAssertNotNil(frame, "Should produce a valid binary frame from RGBA data")
    if let frame = frame {
      XCTAssertGreaterThan(frame.count, 8)
    }
  }

  func testEncodePreviewFrameFromRGBAInvalidSize() {
    // Too little data for the claimed dimensions.
    let frame = ComfyBridgePreviewEncoder.encodePreviewFrame(
      fromRGBA: Data(count: 10),
      width: 100,
      height: 100,
      maxDimension: 256
    )
    XCTAssertNil(frame, "Should return nil for insufficient RGBA data")
  }

  // MARK: - Helpers

  private func makeTestCGImage(width: Int, height: Int) -> CGImage? {
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else { return nil }
    // Fill with red.
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
  }
}
#endif
