import Foundation

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
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

    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .positiveInfinity
    generator.requestedTimeToleranceAfter = .zero

    // Ask for a hair before the nominal end — the true last sample's PTS is
    // one frame earlier than the duration, and zero-after tolerance walks
    // back to it.
    let target = CMTime(seconds: max(0, duration - 0.001), preferredTimescale: 600)
    let image: CGImage
    do {
      image = try generator.copyCGImage(at: target, actualTime: nil)
    } catch {
      throw ExtractError.unreadable("\(videoPath): \(error.localizedDescription)")
    }

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
}

#endif
