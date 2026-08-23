import XCTest
@testable import ZImage

/// `/health.progress_percent` stayed at 0 for the whole of a Krea 2 render:
/// the Z-Image path publishes from its `GenerationProgress` callback, but the
/// Krea 2 arm's `(step, total)` callback only wrote a log line, so the desktop
/// progress bar sat still for ~2 minutes. The Krea 2 loop DOES report every
/// step; the percent it maps to is the only thing worth a test of its own.
final class RenderProgressPercentTests: XCTestCase {

  func testPercentTracksTheStepsWalked() {
    XCTAssertEqual(RenderProgressPercent.of(step: 0, total: 30), 0)
    XCTAssertEqual(RenderProgressPercent.of(step: 15, total: 30), 50)
    XCTAssertEqual(RenderProgressPercent.of(step: 30, total: 30), 100)
    XCTAssertEqual(RenderProgressPercent.of(step: 1, total: 9), 11)
  }

  /// Truncation, not rounding — the same `Int(fraction * 100)` the Z-Image
  /// path publishes, so the two families report on one scale.
  func testTruncatesLikeTheZImagePath() {
    XCTAssertEqual(RenderProgressPercent.of(step: 8, total: 9), 88)
    XCTAssertEqual(RenderProgressPercent.of(step: 2, total: 3), 66)
  }

  /// Never out of range, never a divide-by-zero: a degenerate total reports
  /// 0 rather than crashing a render for a progress bar.
  func testDegenerateInputsAreClamped() {
    XCTAssertEqual(RenderProgressPercent.of(step: 5, total: 0), 0)
    XCTAssertEqual(RenderProgressPercent.of(step: -1, total: 30), 0)
    XCTAssertEqual(RenderProgressPercent.of(step: 99, total: 30), 100)
  }
}
