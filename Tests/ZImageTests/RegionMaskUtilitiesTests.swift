import XCTest
import CoreGraphics
import ImageIO
@testable import ZImage

final class RegionMaskUtilitiesTests: XCTestCase {

  /// Decode a PNG and sample the gray value (0-255) at a visual pixel
  /// coordinate, (0,0) = top-left.
  private func grayValue(in pngData: Data, x: Int, y: Int) throws -> UInt8 {
    let source = try XCTUnwrap(CGImageSourceCreateWithData(pngData as CFData, nil))
    let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
    let width = image.width
    let height = image.height
    var buffer = [UInt8](repeating: 0, count: width * height)
    let context = try XCTUnwrap(CGContext(
      data: &buffer,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width,
      space: CGColorSpaceCreateDeviceGray(),
      bitmapInfo: CGImageAlphaInfo.none.rawValue
    ))
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return buffer[y * width + x]
  }

  func testUpperRegionMaskWhiteOnTopBlackOnBottom() throws {
    let png = try RegionMaskUtilities.makeRegionMaskPNG(
      region: "upper", sourceImage: nil, width: 64, height: 64)
    XCTAssertEqual(try grayValue(in: png, x: 32, y: 8), 255, "visual top should be white (regenerate)")
    XCTAssertEqual(try grayValue(in: png, x: 32, y: 56), 0, "visual bottom should be black (keep)")
  }

  func testLowerRegionMaskBlackOnTopWhiteOnBottom() throws {
    let png = try RegionMaskUtilities.makeRegionMaskPNG(
      region: "lower", sourceImage: nil, width: 64, height: 64)
    XCTAssertEqual(try grayValue(in: png, x: 32, y: 8), 0)
    XCTAssertEqual(try grayValue(in: png, x: 32, y: 56), 255)
  }

  func testInvertFlagFlipsRegions() throws {
    let png = try RegionMaskUtilities.makeRegionMaskPNG(
      region: "upper", sourceImage: nil, width: 64, height: 64, invert: true)
    XCTAssertEqual(try grayValue(in: png, x: 32, y: 8), 0)
    XCTAssertEqual(try grayValue(in: png, x: 32, y: 56), 255)
  }

  func testMaskDimensionsMatchRequested() throws {
    let png = try RegionMaskUtilities.makeRegionMaskPNG(
      region: "lower", sourceImage: nil, width: 128, height: 96)
    let source = try XCTUnwrap(CGImageSourceCreateWithData(png as CFData, nil))
    let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
    XCTAssertEqual(image.width, 128)
    XCTAssertEqual(image.height, 96)
  }

  func testUnsupportedRegionThrows() {
    XCTAssertThrowsError(try RegionMaskUtilities.makeRegionMaskPNG(
      region: "torso", sourceImage: nil, width: 64, height: 64)) { error in
      guard case RegionMaskError.unsupportedRegion = error else {
        return XCTFail("expected unsupportedRegion, got \(error)")
      }
    }
  }

  func testFaceRegionWithoutSourceImageThrows() {
    XCTAssertThrowsError(try RegionMaskUtilities.makeRegionMaskPNG(
      region: "face", sourceImage: nil, width: 64, height: 64)) { error in
      guard case RegionMaskError.sourceImageRequired = error else {
        return XCTFail("expected sourceImageRequired, got \(error)")
      }
    }
  }

  func testInvertMaskPNGFlipsValues() throws {
    let png = try RegionMaskUtilities.makeRegionMaskPNG(
      region: "upper", sourceImage: nil, width: 64, height: 64)
    let inverted = try RegionMaskUtilities.invertMaskPNG(png)
    XCTAssertEqual(try grayValue(in: inverted, x: 32, y: 8), 0)
    XCTAssertEqual(try grayValue(in: inverted, x: 32, y: 56), 255)
  }
}
