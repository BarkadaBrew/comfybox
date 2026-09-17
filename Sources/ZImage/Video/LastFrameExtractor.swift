import Foundation

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import CoreImage
import ImageIO

/// Extracts a clip's last frame as a PNG — the anchor for the next shot in a
/// storyboard chain (comfybox#237). Chaining each shot's i2v from the previous
/// shot's final frame locks face/angle/character across the scene.
public enum LastFrameExtractor {

  public enum ExtractError: Error, LocalizedError, CustomStringConvertible {
    case unreadable(String)
    case writeFailed(String)

    public var description: String {
      switch self {
      case .unreadable(let p): return "Could not read a last frame from \(p)"
      case .writeFailed(let p): return "Could not write extracted frame to \(p)"
      }
    }

    public var errorDescription: String? { description }
  }

  /// Extract the last frame of `videoPath` and write it as a PNG at
  /// `outputPath`. Returns the output path for chaining convenience.
  @discardableResult
  public static func extractLastFrame(from videoPath: String, to outputPath: String) throws -> String {
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let duration = CMTimeGetSeconds(asset.duration)
    guard duration > 0 else { throw ExtractError.unreadable(videoPath) }

    // EXACT last frame (2026-09-17). The previous AVAssetImageGenerator call
    // used `requestedTimeToleranceBefore = .positiveInfinity`, which lets
    // AVFoundation return any earlier frame that is cheaper to decode — in
    // practice the nearest sync frame. A 193-frame Director chunk handed back
    // frame 178; a single-GOP clip hands back frame 0. Director carry-over,
    // storyboard chaining and /v1/video/extend all restarted from that frame.
    // (Zero tolerance at the final sample is refused by the generator:
    // "Cannot Open".) So decode the track with AVAssetReader — its
    // decompressed output is in presentation order — and keep the last frame.
    let image = try decodeLastFrame(of: asset, videoPath: videoPath)

    let outURL = URL(fileURLWithPath: outputPath)
    try FileManager.default.createDirectory(
      at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, "public.png" as CFString, 1, nil) else {
      throw ExtractError.writeFailed(outputPath)
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
      throw ExtractError.writeFailed(outputPath)
    }
    return outputPath
  }

  /// Decode every frame of the first video track (BGRA) and return the last
  /// one in presentation order as a CGImage, with the track's preferred
  /// transform applied (what `appliesPreferredTrackTransform` did before).
  static func decodeLastFrame(of asset: AVAsset, videoPath: String) throws -> CGImage {
    guard let track = asset.tracks(withMediaType: .video).first else {
      throw ExtractError.unreadable("\(videoPath): no video track")
    }
    let reader: AVAssetReader
    do {
      reader = try AVAssetReader(asset: asset)
    } catch {
      throw ExtractError.unreadable("\(videoPath): \(error.localizedDescription)")
    }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    ])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw ExtractError.unreadable("\(videoPath): reader refused the track output") }
    reader.add(output)
    guard reader.startReading() else {
      throw ExtractError.unreadable("\(videoPath): \(reader.error?.localizedDescription ?? "startReading failed")")
    }
    var last: CVPixelBuffer?
    var lastPTS = CMTime.negativeInfinity
    while let sample = output.copyNextSampleBuffer() {
      guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
      let pts = CMSampleBufferGetPresentationTimeStamp(sample)
      // Presentation order is the decoder's output order; the comparison
      // guards against a reader that ever emits out of order.
      if last == nil || !pts.isValid || CMTimeCompare(pts, lastPTS) >= 0 {
        last = pixels
        if pts.isValid { lastPTS = pts }
      }
    }
    if reader.status == .failed {
      throw ExtractError.unreadable("\(videoPath): \(reader.error?.localizedDescription ?? "decode failed")")
    }
    guard let frame = last else { throw ExtractError.unreadable("\(videoPath): no decodable frames") }
    var ci = CIImage(cvPixelBuffer: frame)
    let transform = track.preferredTransform
    if !transform.isIdentity {
      ci = ci.transformed(by: transform)
      ci = ci.transformed(by: CGAffineTransform(translationX: -ci.extent.origin.x, y: -ci.extent.origin.y))
    }
    let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    guard let cg = context.createCGImage(ci, from: ci.extent) else {
      throw ExtractError.unreadable("\(videoPath): could not render the last frame")
    }
    return cg
  }
}

#endif
