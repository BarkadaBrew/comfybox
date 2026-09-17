import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import XCTest

@testable import ZImage

/// Director chunk tone match: continuation chunks inherit a per-render
/// exposure lift (stabilizeColor renormalises to a decoded frame 0 that is
/// brighter than its conditioning keyframe), so each chunk k > 0 is matched
/// back to chunk k-1's corrected last frame. Pure buffer maths + the
/// carry-over PNG rewrite. No models.
final class DirectorToneMatchTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("director-tone-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  // MARK: - Fixtures

  /// RGBA interleaved w x h: r ramps 40...160 along x, g ramps 60...200 along
  /// y, b is a checker of 80/120. Alpha 255.
  private func synthetic(width w: Int, height h: Int) -> [UInt8] {
    var buf = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h {
      for x in 0..<w {
        let i = (y * w + x) * 4
        buf[i] = UInt8(40 + (120 * x) / max(w - 1, 1))
        buf[i + 1] = UInt8(60 + (140 * y) / max(h - 1, 1))
        buf[i + 2] = ((x / 4 + y / 4) % 2 == 0) ? 80 : 120
      }
    }
    return buf
  }

  private func stats(_ buf: [UInt8], _ w: Int, _ h: Int, _ format: ToneMatchPixelFormat = .rgba, stride: Int = 1) -> ChannelStats {
    buf.withUnsafeBufferPointer {
      ChannelStats(buffer: $0.baseAddress!, width: w, height: h, bytesPerRow: w * 4, format: format, stride: stride)
    }
  }

  private func apply(_ t: ToneTransform, _ buf: inout [UInt8], _ w: Int, _ h: Int, _ format: ToneMatchPixelFormat = .rgba) {
    buf.withUnsafeMutableBufferPointer {
      t.apply(buffer: $0.baseAddress!, width: w, height: h, bytesPerRow: w * 4, format: format)
    }
  }

  private func assertStats(_ a: ChannelStats, _ b: ChannelStats, accuracy: Double, _ label: String,
                           file: StaticString = #filePath, line: UInt = #line) {
    for c in 0..<3 {
      XCTAssertEqual(a.mean[c], b.mean[c], accuracy: accuracy, "\(label) mean[\(c)]", file: file, line: line)
      XCTAssertEqual(a.std[c], b.std[c], accuracy: accuracy, "\(label) std[\(c)]", file: file, line: line)
    }
  }

  // MARK: - Stats

  func testStatsOnSyntheticBuffer() {
    // 2x1: (10, 20, 30) and (30, 60, 90).
    let buf: [UInt8] = [10, 20, 30, 255, 30, 60, 90, 255]
    let s = stats(buf, 2, 1)
    XCTAssertEqual(s.mean, [20, 40, 60])
    XCTAssertEqual(s.std, [10, 20, 30])
  }

  func testStrideSubsampleApproximatesFullStats() {
    let w = 576, h = 896
    let buf = synthetic(width: w, height: h)
    let full = stats(buf, w, h)
    let sub = stats(buf, w, h, stride: 4)
    assertStats(sub, full, accuracy: 1.5, "stride 4 vs full")
  }

  func testBGRAAndRGBAChannelOrder() {
    let rgba: [UInt8] = [10, 20, 30, 255, 30, 60, 90, 255]
    let bgra: [UInt8] = [30, 20, 10, 255, 90, 60, 30, 255]
    let a = stats(rgba, 2, 1, .rgba)
    let b = stats(bgra, 2, 1, .bgra)
    XCTAssertEqual(a, b, "stats are always reported R, G, B")
    XCTAssertEqual(b.mean, [20, 40, 60])

    // A red-only lift on BGRA touches byte 2, not byte 0; alpha untouched.
    var buf = bgra
    let t = ToneTransform(gain: [1, 1, 1], offset: [10, 0, 0])
    apply(t, &buf, 2, 1, .bgra)
    XCTAssertEqual(buf, [30, 20, 20, 255, 90, 60, 40, 255])
  }

  // MARK: - Match

  func testMatchInvertsABrightenedCopyAndRestoresStats() {
    let w = 64, h = 64
    let original = synthetic(width: w, height: h)
    var brightened = original
    apply(ToneTransform(gain: [1.15, 1.15, 1.15], offset: [12, 12, 12]), &brightened, w, h)

    let src = stats(brightened, w, h)
    let dst = stats(original, w, h)
    let t = DirectorToneMatch.match(source: src, target: dst)
    for c in 0..<3 {
      XCTAssertEqual(t.gain[c], 1 / 1.15, accuracy: 0.02, "gain[\(c)]")
      XCTAssertEqual(t.offset[c], -12 / 1.15, accuracy: 2.0, "offset[\(c)]")
    }

    var restored = brightened
    apply(t, &restored, w, h)
    assertStats(stats(restored, w, h), dst, accuracy: 1.0, "restored vs original")
  }

  func testMatchClampsGainAndOffsetForWildTarget() {
    let source = ChannelStats(mean: [100, 100, 100], std: [20, 20, 20])
    let target = ChannelStats(mean: [250, 5, 100], std: [80, 2, 20])
    let t = DirectorToneMatch.match(source: source, target: target)
    XCTAssertEqual(t.gain[0], 1.35, accuracy: 1e-9, "gain clamped high")
    XCTAssertEqual(t.gain[1], 0.65, accuracy: 1e-9, "gain clamped low")
    XCTAssertEqual(t.gain[2], 1.0, accuracy: 1e-9)
    XCTAssertEqual(t.offset[0], 64, accuracy: 1e-9, "offset clamped +64 (250 - 135 = 115)")
    XCTAssertEqual(t.offset[1], -60, accuracy: 1e-9, "5 - 65 = -60, inside the clamp")
    XCTAssertEqual(t.offset[2], 0, accuracy: 1e-9)

    let t2 = DirectorToneMatch.match(
      source: ChannelStats(mean: [200, 200, 200], std: [20, 20, 20]),
      target: ChannelStats(mean: [10, 10, 10], std: [20, 20, 20]))
    XCTAssertEqual(t2.offset, [-64, -64, -64], "offset clamped -64")

    let custom = DirectorToneMatch.match(source: source, target: target, maxGainDeviation: 0.1)
    XCTAssertEqual(custom.gain[0], 1.1, accuracy: 1e-9)
    XCTAssertEqual(custom.gain[1], 0.9, accuracy: 1e-9)
  }

  func testMatchOfEqualStatsIsIdentity() {
    let s = ChannelStats(mean: [92.8, 90, 88], std: [40, 41, 39])
    let t = DirectorToneMatch.match(source: s, target: s)
    XCTAssertEqual(t, .identity)
    var buf = synthetic(width: 8, height: 8)
    let before = buf
    apply(t, &buf, 8, 8)
    XCTAssertEqual(buf, before)
  }

  func testDegenerateFlatImageGivesUnitGain() {
    let flat = [UInt8](repeating: 100, count: 16 * 4)
    let s = stats(flat, 4, 4)
    XCTAssertEqual(s.std, [0, 0, 0])
    let t = DirectorToneMatch.match(source: s, target: ChannelStats(mean: [110, 110, 110], std: [30, 30, 30]))
    XCTAssertEqual(t.gain, [1, 1, 1])
    XCTAssertEqual(t.offset, [10, 10, 10])
  }

  func testApplyClampsAndComposes() {
    var buf: [UInt8] = [250, 5, 128, 255]
    apply(ToneTransform(gain: [1.2, 1, 1], offset: [0, -20, 0]), &buf, 1, 1)
    XCTAssertEqual(buf, [255, 0, 128, 255])

    let a = ToneTransform(gain: [2, 1, 1], offset: [10, 0, 0])
    let b = ToneTransform(gain: [0.5, 1, 1], offset: [3, 0, 0])
    let ba = b.composed(after: a)  // b(a(x))
    XCTAssertEqual(ba.apply(to: [20, 20, 20])[0], b.apply(to: a.apply(to: [20, 20, 20]))[0], accuracy: 1e-9)
    XCTAssertEqual(ToneTransform.identity.apply(to: [1, 2, 3]), [1, 2, 3])
  }

  // MARK: - Carry-over (the runDirector step)

  private func writePNG(_ rgba: [UInt8], _ w: Int, _ h: Int, to path: String) throws {
    let data = CFDataCreate(nil, rgba, rgba.count)!
    let provider = CGDataProvider(data: data)!
    let image = CGImage(
      width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
      space: CGColorSpace(name: CGColorSpace.sRGB)!,
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    XCTAssertTrue(CGImageDestinationFinalize(dest))
  }

  func testCarryOverPNGIsRewrittenWithTheChunkTransform() throws {
    let w = 64, h = 64
    let original = synthetic(width: w, height: h)
    let path = tempDir.appendingPathComponent("lastframe.png").path
    try writePNG(original, w, h, to: path)
    let before = try DirectorToneMatch.pngStats(path: path, stride: 1)
    assertStats(before, stats(original, w, h), accuracy: 1.0, "png round-trip")

    let t = ToneTransform(gain: [0.9, 0.9, 0.9], offset: [-8, -8, -8])
    let after = try DirectorToneMatch.rewritePNG(at: path, applying: t, stride: 1)
    let reread = try DirectorToneMatch.pngStats(path: path, stride: 1)
    assertStats(after, reread, accuracy: 0.5, "returned stats describe the rewritten file")
    var expected = original
    apply(t, &expected, w, h)
    assertStats(reread, stats(expected, w, h), accuracy: 1.0, "file carries the transform")

    // Identity leaves the file's stats unchanged.
    let same = try DirectorToneMatch.rewritePNG(at: path, applying: .identity, stride: 1)
    assertStats(same, reread, accuracy: 0.5, "identity rewrite")
  }

  func testChunkToneMatchStepPullsABrightChunkBackToTheCarryTarget() throws {
    // A chunk whose every frame is the carry frame brightened ~15% (the
    // measured 92.8 -> 107.2 step), matched against the carry frame's stats.
    let w = 64, h = 64
    let carry = synthetic(width: w, height: h)
    var bright = carry
    apply(ToneTransform(gain: [1.155, 1.155, 1.155], offset: [0, 0, 0]), &bright, w, h)
    let chunkPath = tempDir.appendingPathComponent("chunk1.mp4").path
    let frame = try makeImage(bright, w, h)
    try LTX2PostProcess.writeMP4(frames: Array(repeating: frame, count: 9), outputPath: chunkPath, fps: 24, width: w, height: h)

    let target = stats(carry, w, h)
    let firstFrame = try DirectorToneMatch.firstFrameStats(videoPath: chunkPath, stride: 1)
    XCTAssertGreaterThan(firstFrame.mean[0], target.mean[0] + 8, "fixture really is brighter")
    let t = DirectorToneMatch.match(source: firstFrame, target: target)
    let brightStats = stats(bright, w, h)
    for c in 0..<3 {
      // The decoded first frame is the brightened frame (H.264 drift only).
      XCTAssertEqual(firstFrame.mean[c], brightStats.mean[c], accuracy: 6.0, "first-frame mean[\(c)] decoded")
      // Unclamped, the match maps the decoded frame's mean exactly onto the
      // carry target and darkens (the measured lift is removed, not added).
      XCTAssertLessThan(t.gain[c] * firstFrame.mean[c] + t.offset[c], firstFrame.mean[c] - 8, "mean[\(c)] darkened")
      XCTAssertEqual(t.apply(to: firstFrame.mean)[c], target.mean[c], accuracy: 0.5, "mean[\(c)] mapped to target")
    }
  }

  private func makeImage(_ rgba: [UInt8], _ w: Int, _ h: Int) throws -> CGImage {
    let ctx = CGContext(
      data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    rgba.withUnsafeBytes { src in
      ctx.data!.copyMemory(from: src.baseAddress!, byteCount: rgba.count)
    }
    return try XCTUnwrap(ctx.makeImage())
  }
}
