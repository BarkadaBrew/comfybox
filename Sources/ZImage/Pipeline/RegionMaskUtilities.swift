import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif
#if canImport(Vision)
import Vision
#endif

/// Errors from auto-generated region masks (img2img `mask_region`).
public enum RegionMaskError: Error, LocalizedError, CustomStringConvertible {
  case unsupportedRegion(String)
  case sourceImageRequired(String)
  case noFaceDetected
  case renderFailed(String)

  public var description: String {
    switch self {
    case .unsupportedRegion(let r):
      return "Unsupported mask_region '\(r)' — expected face | upper | lower"
    case .sourceImageRequired(let r):
      return "mask_region '\(r)' requires a decodable source image"
    case .noFaceDetected:
      return "mask_region 'face': no face detected in the source image"
    case .renderFailed(let msg):
      return "Region mask render failed: \(msg)"
    }
  }

  public var errorDescription: String? { description }
}

/// Synthesizes inpainting masks from a region keyword so callers (humans or an
/// operator LLM) describe intent — "face", "upper", "lower" — and the engine
/// does the pixel work. White = regenerate, black = keep, matching the
/// `mask_path` convention. `invert` flips the roles (e.g. region "face" +
/// invert = lock the face, regenerate everything else).
public enum RegionMaskUtilities {

  public enum Region: String, CaseIterable {
    case face
    case upper
    case lower
  }

  /// Fractional padding applied around each Vision face rect. Vision boxes are
  /// tight (no hair/chin); without headroom an inpaint seam cuts through them.
  static let faceRectPadding: CGFloat = 0.3

  #if canImport(CoreGraphics)

  /// Build a mask PNG at the (round-to-16) generation dimensions.
  /// `sourceImage` is only consulted for `.face` (Vision face detection).
  public static func makeRegionMaskPNG(
    region rawRegion: String,
    sourceImage: CGImage?,
    width: Int,
    height: Int,
    invert: Bool = false
  ) throws -> Data {
    guard let region = Region(rawValue: rawRegion.lowercased()) else {
      throw RegionMaskError.unsupportedRegion(rawRegion)
    }

    let whiteRects: [CGRect]
    switch region {
    case .upper:
      // CG coordinates: origin bottom-left, so the visual top half is the
      // upper half of the y range.
      whiteRects = [CGRect(x: 0, y: CGFloat(height) / 2, width: CGFloat(width), height: CGFloat(height) / 2)]
    case .lower:
      whiteRects = [CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height) / 2)]
    case .face:
      guard let sourceImage else {
        throw RegionMaskError.sourceImageRequired(rawRegion)
      }
      let normalized = try detectFaceRects(in: sourceImage)
      guard !normalized.isEmpty else { throw RegionMaskError.noFaceDetected }
      whiteRects = normalized.map { rect in
        // Vision rects are normalized with bottom-left origin — same convention
        // as CG, so scale directly, then pad and clamp.
        let scaled = CGRect(
          x: rect.minX * CGFloat(width),
          y: rect.minY * CGFloat(height),
          width: rect.width * CGFloat(width),
          height: rect.height * CGFloat(height)
        )
        let padded = scaled.insetBy(
          dx: -scaled.width * faceRectPadding,
          dy: -scaled.height * faceRectPadding
        )
        return padded.intersection(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
      }
    }

    return try renderMaskPNG(whiteRects: whiteRects, width: width, height: height, invert: invert)
  }

  /// Invert an existing mask PNG (white ⇄ black). Used when the caller supplies
  /// `mask_path` + `mask_invert`.
  public static func invertMaskPNG(_ data: Data) throws -> Data {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
      throw RegionMaskError.renderFailed("Could not decode mask PNG for inversion")
    }
    let width = image.width
    let height = image.height
    guard let context = grayContext(width: width, height: height) else {
      throw RegionMaskError.renderFailed("Could not create CGContext for inversion")
    }
    let bounds = CGRect(x: 0, y: 0, width: width, height: height)
    context.draw(image, in: bounds)
    // difference-against-white inverts the drawn grayscale values.
    context.setBlendMode(.difference)
    context.setFillColor(gray: 1.0, alpha: 1.0)
    context.fill(bounds)
    return try encodePNG(from: context)
  }

  /// Detected face rects, normalized [0,1] with bottom-left origin.
  static func detectFaceRects(in image: CGImage) throws -> [CGRect] {
    #if canImport(Vision)
    let request = VNDetectFaceRectanglesRequest()
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
      try handler.perform([request])
    } catch {
      throw RegionMaskError.renderFailed("Face detection failed: \(error.localizedDescription)")
    }
    return (request.results ?? []).map { $0.boundingBox }
    #else
    throw RegionMaskError.renderFailed("Vision framework unavailable on this platform")
    #endif
  }

  static func renderMaskPNG(
    whiteRects: [CGRect],
    width: Int,
    height: Int,
    invert: Bool
  ) throws -> Data {
    guard let context = grayContext(width: width, height: height) else {
      throw RegionMaskError.renderFailed("Could not create CGContext (\(width)x\(height))")
    }
    let bounds = CGRect(x: 0, y: 0, width: width, height: height)
    context.setFillColor(gray: invert ? 1.0 : 0.0, alpha: 1.0)
    context.fill(bounds)
    context.setFillColor(gray: invert ? 0.0 : 1.0, alpha: 1.0)
    for rect in whiteRects where !rect.isNull && !rect.isEmpty {
      context.fill(rect)
    }
    return try encodePNG(from: context)
  }

  private static func grayContext(width: Int, height: Int) -> CGContext? {
    CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceGray(),
      bitmapInfo: CGImageAlphaInfo.none.rawValue
    )
  }

  private static func encodePNG(from context: CGContext) throws -> Data {
    guard let image = context.makeImage() else {
      throw RegionMaskError.renderFailed("Could not create CGImage from mask context")
    }
    let output = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(output, "public.png" as CFString, 1, nil) else {
      throw RegionMaskError.renderFailed("Could not create PNG destination")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw RegionMaskError.renderFailed("Could not finalize mask PNG")
    }
    return output as Data
  }

  #endif
}
