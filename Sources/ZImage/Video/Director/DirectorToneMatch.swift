// DirectorToneMatch.swift — per-chunk exposure match for Director renders.
//
// Measured defect (2-chunk, 385f render, seam at timeline frame 192): mean
// luma flat within each chunk (~92.8 chunk 0, ~107.2 chunk 1) with a +15%
// step at the seam. `LTX2PostProcess.stabilizeColor` renormalises every
// frame of a render to that render's DECODED frame 0, and decoded frame 0
// comes out brighter than its conditioning image. Every continuation chunk is
// conditioned on the previous chunk's already-lifted last frame, so the lift
// compounds chunk over chunk. The stitcher and the H.264 round-trip were both
// ruled out.
//
// The fix is Director-scoped (stabilizeColor and the i2v path are untouched):
// chunk k > 0 gets a per-channel gain/offset that maps its first frame's
// statistics onto chunk k-1's corrected last frame (the frame it was
// conditioned on). The transform is applied to chunk k's carry-over PNG
// before it conditions chunk k+1 (stopping the compounding) and to chunk k's
// frames at stitch time (removing the seam pop).
//
// Pure maths on 8-bit interleaved buffers, plus small ImageIO/AVFoundation
// readers/writers for the carry-over PNG and a chunk's first frame.

import Foundation

/// Byte order of an 8-bit, 4-channel interleaved buffer. Alpha is skipped.
public enum ToneMatchPixelFormat: Sendable {
  case rgba
  case bgra

  /// Byte offsets of R, G, B within a pixel.
  var channelOffsets: (Int, Int, Int) {
    switch self {
    case .rgba: return (0, 1, 2)
    case .bgra: return (2, 1, 0)
    }
  }
}

/// Per-channel mean and standard deviation, always ordered R, G, B, on the
/// 0...255 scale.
public struct ChannelStats: Sendable, Equatable, Codable {
  public var mean: [Double]
  public var std: [Double]

  public init(mean: [Double], std: [Double]) {
    precondition(mean.count == 3 && std.count == 3, "ChannelStats needs 3 channels")
    self.mean = mean
    self.std = std
  }

  /// Statistics of a 4-channel interleaved buffer. `stride` subsamples both
  /// axes (every `stride`-th row and column), so a 576x896 frame at stride 4
  /// reads ~32k pixels.
  public init(
    buffer: UnsafePointer<UInt8>, width: Int, height: Int, bytesPerRow: Int,
    format: ToneMatchPixelFormat, stride: Int = 1
  ) {
    let step = max(1, stride)
    let (ro, go, bo) = format.channelOffsets
    var sum = [Double](repeating: 0, count: 3)
    var sq = [Double](repeating: 0, count: 3)
    var n = 0.0
    var y = 0
    while y < height {
      let row = buffer + y * bytesPerRow
      var x = 0
      while x < width {
        let p = row + x * 4
        let r = Double(p[ro]), g = Double(p[go]), b = Double(p[bo])
        sum[0] += r; sum[1] += g; sum[2] += b
        sq[0] += r * r; sq[1] += g * g; sq[2] += b * b
        n += 1
        x += step
      }
      y += step
    }
    guard n > 0 else {
      self.init(mean: [0, 0, 0], std: [0, 0, 0])
      return
    }
    var mean = [Double](repeating: 0, count: 3)
    var std = [Double](repeating: 0, count: 3)
    for c in 0..<3 {
      mean[c] = sum[c] / n
      std[c] = (max(0, sq[c] / n - mean[c] * mean[c])).squareRoot()
    }
    self.init(mean: mean, std: std)
  }
}

/// `out = gain * in + offset` per channel (R, G, B), 0...255 scale.
public struct ToneTransform: Sendable, Equatable, Codable {
  public var gain: [Double]
  public var offset: [Double]

  public init(gain: [Double], offset: [Double]) {
    precondition(gain.count == 3 && offset.count == 3, "ToneTransform needs 3 channels")
    self.gain = gain
    self.offset = offset
  }

  public static let identity = ToneTransform(gain: [1, 1, 1], offset: [0, 0, 0])

  public var isIdentity: Bool { self == .identity }

  /// Apply to one RGB pixel (unclamped).
  public func apply(to pixel: [Double]) -> [Double] {
    (0..<3).map { gain[$0] * pixel[$0] + offset[$0] }
  }

  /// `self` applied after `other`: `self(other(x))`.
  public func composed(after other: ToneTransform) -> ToneTransform {
    ToneTransform(
      gain: (0..<3).map { gain[$0] * other.gain[$0] },
      offset: (0..<3).map { gain[$0] * other.offset[$0] + offset[$0] })
  }

  /// Apply in place to a 4-channel interleaved buffer, rounding and clamping
  /// to 0...255. Alpha is untouched. Identity is a no-op.
  public func apply(
    buffer: UnsafeMutablePointer<UInt8>, width: Int, height: Int, bytesPerRow: Int,
    format: ToneMatchPixelFormat
  ) {
    guard !isIdentity else { return }
    let luts: [[UInt8]] = (0..<3).map { c in
      (0..<256).map { v in
        UInt8(min(255, max(0, (gain[c] * Double(v) + offset[c]).rounded())))
      }
    }
    let (ro, go, bo) = format.channelOffsets
    luts[0].withUnsafeBufferPointer { lr in
      luts[1].withUnsafeBufferPointer { lg in
        luts[2].withUnsafeBufferPointer { lb in
          for y in 0..<height {
            let row = buffer + y * bytesPerRow
            for x in 0..<width {
              let p = row + x * 4
              p[ro] = lr[Int(p[ro])]
              p[go] = lg[Int(p[go])]
              p[bo] = lb[Int(p[bo])]
            }
          }
        }
      }
    }
  }
}

public enum DirectorToneMatch {

  /// Stride used on full-size frames (576x896 → ~32k samples).
  public static let defaultStride = 4
  /// Offset clamp, 0...255 scale.
  public static let maxOffset: Double = 64
  /// A channel whose source std is below this is treated as flat: gain 1.
  public static let minStd: Double = 1

  /// The transform that maps `source` statistics onto `target`, per channel:
  /// gain = targetStd / sourceStd clamped to [1-d, 1+d] (1 for a flat
  /// source), offset = targetMean - gain * sourceMean clamped to ±64.
  public static func match(
    source: ChannelStats, target: ChannelStats, maxGainDeviation: Double = 0.35
  ) -> ToneTransform {
    var gain = [Double](repeating: 1, count: 3)
    var offset = [Double](repeating: 0, count: 3)
    for c in 0..<3 {
      if source.std[c] >= minStd {
        gain[c] = min(1 + maxGainDeviation, max(1 - maxGainDeviation, target.std[c] / source.std[c]))
      }
      offset[c] = min(maxOffset, max(-maxOffset, target.mean[c] - gain[c] * source.mean[c]))
    }
    return ToneTransform(gain: gain, offset: offset)
  }

  /// One-line rendering for logs: `gain=[0.912,0.915,0.920] offset=[-1.2,…]`.
  public static func describe(_ t: ToneTransform) -> String {
    func fmt(_ v: [Double], _ p: Int) -> String {
      "[" + v.map { String(format: "%.\(p)f", $0) }.joined(separator: ",") + "]"
    }
    return "gain=\(fmt(t.gain, 3)) offset=\(fmt(t.offset, 2))"
  }
}

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import ImageIO

extension DirectorToneMatch {

  public enum ToneMatchError: Error, LocalizedError, CustomStringConvertible {
    case unreadable(String)
    case writeFailed(String)

    public var description: String {
      switch self {
      case .unreadable(let p): return "tone match could not read \(p)"
      case .writeFailed(let p): return "tone match could not write \(p)"
      }
    }

    public var errorDescription: String? { description }
  }

  /// Draw `image` into an RGBA8 buffer in the image's own RGB colour space (no
  /// colour conversion), hand it to `body`, and return the context.
  private static func withRGBA<T>(
    _ image: CGImage, _ body: (UnsafeMutablePointer<UInt8>, Int, Int, Int) -> T
  ) -> (CGContext, T)? {
    let w = image.width, h = image.height
    let space: CGColorSpace
    if let cs = image.colorSpace, cs.model == .rgb {
      space = cs
    } else {
      space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    }
    guard w > 0, h > 0, let ctx = CGContext(
      data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
      let data = ctx.data
    else { return nil }
    ctx.interpolationQuality = .none
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    let result = body(data.assumingMemoryBound(to: UInt8.self), w, h, ctx.bytesPerRow)
    return (ctx, result)
  }

  private static func stats(of image: CGImage, stride: Int) -> ChannelStats? {
    withRGBA(image) { buf, w, h, bpr in
      ChannelStats(buffer: buf, width: w, height: h, bytesPerRow: bpr, format: .rgba, stride: stride)
    }?.1
  }

  private static func loadPNG(_ path: String) throws -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { throw ToneMatchError.unreadable(path) }
    return image
  }

  /// Statistics of an image file (the carry-over PNG).
  public static func pngStats(path: String, stride: Int = defaultStride) throws -> ChannelStats {
    guard let s = stats(of: try loadPNG(path), stride: stride) else { throw ToneMatchError.unreadable(path) }
    return s
  }

  /// Statistics of a clip's first frame, decoded through the same
  /// AVAssetImageGenerator path `LastFrameExtractor` uses for the last frame.
  public static func firstFrameStats(videoPath: String, stride: Int = defaultStride) throws -> ChannelStats {
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .positiveInfinity
    let image: CGImage
    do {
      image = try generator.copyCGImage(at: .zero, actualTime: nil)
    } catch {
      throw ToneMatchError.unreadable("\(videoPath): \(error.localizedDescription)")
    }
    guard let s = stats(of: image, stride: stride) else { throw ToneMatchError.unreadable(videoPath) }
    return s
  }

  /// Apply `transform` to the PNG at `path` and rewrite it in place (same
  /// colour space; written to a sibling temp file then moved over). Returns
  /// the statistics of the rewritten pixels. Identity skips the rewrite.
  @discardableResult
  public static func rewritePNG(
    at path: String, applying transform: ToneTransform, stride: Int = defaultStride
  ) throws -> ChannelStats {
    let image = try loadPNG(path)
    guard let drawn = withRGBA(image, { buf, w, h, bpr -> ChannelStats in
      transform.apply(buffer: buf, width: w, height: h, bytesPerRow: bpr, format: .rgba)
      return ChannelStats(buffer: buf, width: w, height: h, bytesPerRow: bpr, format: .rgba, stride: stride)
    }) else { throw ToneMatchError.unreadable(path) }
    let (ctx, s) = drawn
    if transform.isIdentity { return s }
    guard let out = ctx.makeImage() else { throw ToneMatchError.writeFailed(path) }
    let url = URL(fileURLWithPath: path)
    let tmp = url.deletingLastPathComponent()
      .appendingPathComponent(".\(url.lastPathComponent).tone-\(UUID().uuidString).png")
    guard let dest = CGImageDestinationCreateWithURL(tmp as CFURL, "public.png" as CFString, 1, nil) else {
      throw ToneMatchError.writeFailed(path)
    }
    CGImageDestinationAddImage(dest, out, nil)
    guard CGImageDestinationFinalize(dest) else {
      try? FileManager.default.removeItem(at: tmp)
      throw ToneMatchError.writeFailed(path)
    }
    do {
      _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
    } catch {
      try? FileManager.default.removeItem(at: tmp)
      throw ToneMatchError.writeFailed("\(path): \(error.localizedDescription)")
    }
    return s
  }
}
#endif
