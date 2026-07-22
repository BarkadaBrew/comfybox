import XCTest
import CoreGraphics
import ImageIO
@testable import ZImage

final class StoryboardTests: XCTestCase {

  private func shot(
    prompt: String = "she smiles",
    durationS: Double? = 4,
    anchor: String? = nil
  ) -> StoryboardSpec.Shot {
    StoryboardSpec.Shot(prompt: prompt, durationS: durationS, anchorImage: anchor)
  }

  func testValidationRequiresShotsAndFirstAnchor() {
    XCTAssertThrowsError(try StoryboardSpec(shots: []).validate()) { error in
      guard case StoryboardError.noShots = error else { return XCTFail("\(error)") }
    }
    // First shot without anchor: the chain has nothing to start from.
    XCTAssertThrowsError(try StoryboardSpec(shots: [shot()]).validate()) { error in
      guard case StoryboardError.firstShotNeedsAnchor = error else { return XCTFail("\(error)") }
    }
    // First shot anchored, later shots chain: valid.
    XCTAssertNoThrow(try StoryboardSpec(
      shots: [shot(anchor: "/tmp/a.png"), shot()]).validate())
  }

  func testValidationDelegatesTransitionMathToMontage() {
    // 2 shots need exactly 1 transition; a bad count must throw.
    let spec = StoryboardSpec(
      shots: [shot(anchor: "/tmp/a.png"), shot()],
      transitions: [
        MontageTransition(kind: .dissolve, durationS: 0.5),
        MontageTransition(kind: .cut),
      ])
    XCTAssertThrowsError(try spec.validate()) { error in
      guard case MontageError.badTransitionCount = error else { return XCTFail("\(error)") }
    }
    // A transition longer than a shot's nominal duration must throw too.
    let tooLong = StoryboardSpec(
      shots: [shot(anchor: "/tmp/a.png", ), shot(durationS: 2)],
      transitions: [MontageTransition(kind: .dissolve, durationS: 3.0)])
    XCTAssertThrowsError(try tooLong.validate())
  }

  func testLastFrameExtraction() throws {
    // Compose a 2-color montage (red → cut → green) and extract its LAST
    // frame — it must be green, proving we get the end, not the start.
    let workDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("storyboard-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: workDir) }

    func solid(_ name: String, r: CGFloat, g: CGFloat) throws -> String {
      let ctx = try XCTUnwrap(CGContext(
        data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue))
      ctx.setFillColor(CGColor(red: r, green: g, blue: 0, alpha: 1))
      ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
      let url = workDir.appendingPathComponent(name)
      let dest = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
      CGImageDestinationAddImage(dest, try XCTUnwrap(ctx.makeImage()), nil)
      XCTAssertTrue(CGImageDestinationFinalize(dest))
      return url.path
    }

    let clip = workDir.appendingPathComponent("clip.mp4").path
    _ = try MontageComposer.compose(
      segments: [
        MontageSegment(kind: .image, path: try solid("red.png", r: 1, g: 0), durationS: 1),
        MontageSegment(kind: .image, path: try solid("green.png", r: 0, g: 1), durationS: 1),
      ],
      transitions: [],
      width: 64, height: 64, fps: 10,
      outputPath: clip)

    let framePath = workDir.appendingPathComponent("last.png").path
    let extracted = try LastFrameExtractor.extractLastFrame(from: clip, to: framePath)
    XCTAssertEqual(extracted, framePath)

    let src = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: framePath) as CFURL, nil))
    let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(src, 0, nil))
    var buffer = [UInt8](repeating: 0, count: image.width * image.height * 4)
    let ctx = try XCTUnwrap(CGContext(
      data: &buffer, width: image.width, height: image.height, bitsPerComponent: 8,
      bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue))
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    var r = 0.0, g = 0.0
    for i in 0..<(image.width * image.height) {
      r += Double(buffer[i * 4 + 1])
      g += Double(buffer[i * 4 + 2])
    }
    let count = Double(image.width * image.height) * 255
    XCTAssertGreaterThan(g / count, 0.6, "last frame should be green")
    XCTAssertLessThan(r / count, 0.35, "last frame should not be red")
  }

  func testExtractFromMissingFileThrows() {
    XCTAssertThrowsError(try LastFrameExtractor.extractLastFrame(
      from: "/nonexistent.mp4",
      to: FileManager.default.temporaryDirectory.appendingPathComponent("x.png").path))
  }
}
