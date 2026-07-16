import XCTest
import AVFoundation
import CoreGraphics
import ImageIO
@testable import ZImage

/// End-to-end composer tests on tiny synthetic assets — no model weights.
final class MontageComposerTests: XCTestCase {

  private var workDir: URL!

  override func setUpWithError() throws {
    workDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("montage-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: workDir)
  }

  /// Write a solid-color PNG.
  private func makeImage(name: String, r: CGFloat, g: CGFloat, b: CGFloat,
                         width: Int = 64, height: Int = 96) throws -> String {
    let ctx = try XCTUnwrap(CGContext(
      data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue))
    ctx.setFillColor(CGColor(red: r, green: g, blue: b, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let image = try XCTUnwrap(ctx.makeImage())
    let url = workDir.appendingPathComponent(name)
    let dest = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(dest, image, nil)
    XCTAssertTrue(CGImageDestinationFinalize(dest))
    return url.path
  }

  /// Average RGB of a frame at time `t` in an mp4.
  private func averageColor(of videoPath: String, at t: Double) throws -> (r: Double, g: Double, b: Double) {
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let generator = AVAssetImageGenerator(asset: asset)
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    let cg = try generator.copyCGImage(
      at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil)
    var buffer = [UInt8](repeating: 0, count: cg.width * cg.height * 4)
    let ctx = try XCTUnwrap(CGContext(
      data: &buffer, width: cg.width, height: cg.height, bitsPerComponent: 8,
      bytesPerRow: cg.width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue))
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
    var r = 0.0, g = 0.0, b = 0.0
    let count = cg.width * cg.height
    for i in 0..<count {
      // noneSkipFirst little-endian default order: skip, R, G, B per CG.
      r += Double(buffer[i * 4 + 1])
      g += Double(buffer[i * 4 + 2])
      b += Double(buffer[i * 4 + 3])
    }
    return (r / Double(count) / 255, g / Double(count) / 255, b / Double(count) / 255)
  }

  func testComposeDurationAndTransitions() throws {
    let red = try makeImage(name: "red.png", r: 1, g: 0, b: 0)
    let green = try makeImage(name: "green.png", r: 0, g: 1, b: 0)
    let blue = try makeImage(name: "blue.png", r: 0, g: 0, b: 1)
    let out = workDir.appendingPathComponent("montage.mp4").path

    let result = try MontageComposer.compose(
      segments: [
        MontageSegment(kind: .image, path: red, durationS: 2),
        MontageSegment(kind: .image, path: green, durationS: 2),
        MontageSegment(kind: .image, path: blue, durationS: 2),
      ],
      transitions: [
        MontageTransition(kind: .dissolve, durationS: 1.0),
        MontageTransition(kind: .cut, durationS: 0),
      ],
      width: 128, height: 128, fps: 10,
      outputPath: out)

    // Duration math: 6 − 1 = 5s → 50 frames.
    XCTAssertEqual(result.durationS, 5.0, accuracy: 1e-9)
    XCTAssertEqual(result.frameCount, 50)
    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    XCTAssertEqual(CMTimeGetSeconds(asset.duration), 5.0, accuracy: 0.15)

    // Solid segment 0 is red.
    let early = try averageColor(of: out, at: 0.5)
    XCTAssertGreaterThan(early.r, 0.6)
    XCTAssertLessThan(early.g, 0.35)
    // Dissolve midpoint (overlap zone [1,2] → t=1.5): red and green both visible.
    let mid = try averageColor(of: out, at: 1.5)
    XCTAssertGreaterThan(mid.r, 0.2, "outgoing red visible at dissolve midpoint")
    XCTAssertGreaterThan(mid.g, 0.2, "incoming green visible at dissolve midpoint")
    // After dissolve: green.
    let second = try averageColor(of: out, at: 2.6)
    XCTAssertGreaterThan(second.g, 0.6)
    // After the cut boundary (segment 2 starts at 3.0): blue immediately.
    let third = try averageColor(of: out, at: 3.3)
    XCTAssertGreaterThan(third.b, 0.6)
    XCTAssertLessThan(third.g, 0.35)
  }

  func testFadeReachesBlackAtBoundary() throws {
    let red = try makeImage(name: "red.png", r: 1, g: 0, b: 0)
    let green = try makeImage(name: "green.png", r: 0, g: 1, b: 0)
    let out = workDir.appendingPathComponent("fade.mp4").path

    _ = try MontageComposer.compose(
      segments: [
        MontageSegment(kind: .image, path: red, durationS: 2),
        MontageSegment(kind: .image, path: green, durationS: 2),
      ],
      transitions: [MontageTransition(kind: .fade, durationS: 1.0)],
      width: 128, height: 128, fps: 10,
      outputPath: out)

    // Fade midpoint (overlap [1,2] → t=1.5) is black-ish.
    let mid = try averageColor(of: out, at: 1.5)
    XCTAssertLessThan(mid.r + mid.g + mid.b, 0.35, "fade midpoint should be near black, got \(mid)")
  }

  func testKenBurnsProducesMotion() throws {
    // A half-red/half-green image: zooming shifts the visible mix over time.
    let ctx = try XCTUnwrap(CGContext(
      data: nil, width: 128, height: 128, bitsPerComponent: 8, bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue))
    ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 128))
    ctx.setFillColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
    ctx.fill(CGRect(x: 64, y: 0, width: 64, height: 128))
    let image = try XCTUnwrap(ctx.makeImage())
    let url = workDir.appendingPathComponent("split.png")
    let dest = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(dest, image, nil)
    XCTAssertTrue(CGImageDestinationFinalize(dest))

    let out = workDir.appendingPathComponent("kenburns.mp4").path
    _ = try MontageComposer.compose(
      segments: [
        MontageSegment(
          kind: .image, path: url.path, durationS: 2,
          // Zoomed 2x so the pan reveals hidden image area (real ken-burns),
          // not background: +x shifts the image right → more of its red left
          // half enters the crop window.
          kenBurns: .init(zoomStart: 2.0, zoomEnd: 2.0, panStart: (0, 0), panEnd: (0.25, 0))),
      ],
      transitions: [],
      width: 128, height: 128, fps: 10,
      outputPath: out)

    // Panning +x moves the image right → more red (left half) becomes visible.
    let first = try averageColor(of: out, at: 0.05)
    let last = try averageColor(of: out, at: 1.9)
    XCTAssertGreaterThan(last.r - first.r, 0.15,
      "ken-burns pan should shift the color mix (first \(first), last \(last))")
  }

  func testMissingAssetFailsWithoutPartialOutput() {
    let out = workDir.appendingPathComponent("missing.mp4").path
    XCTAssertThrowsError(try MontageComposer.compose(
      segments: [MontageSegment(kind: .image, path: "/nonexistent.png", durationS: 2)],
      transitions: [],
      outputPath: out)) { error in
      guard case MontageError.assetNotFound = error else {
        return XCTFail("expected assetNotFound, got \(error)")
      }
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: out), "no partial output file")
  }
}
