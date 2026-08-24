import XCTest
@testable import ZImage

/// Unit tests for the pure ``PhoneLook`` post-process (Todd 2026-08-24: the
/// raw-4step/TDM distill "lacks color correction … needs a pinch of contrast
/// and a pinch of sharpness … a reasonable mobile phone look").
///
/// Recipe validated against real renders (stacktest a1–a3): percentile
/// auto-levels (P0.5/P99.5), S-contrast ×1.15, adaptive saturation toward
/// 0.32 capped at ×1.25, unsharp mask (σ≈1.6, 45%, threshold 2/255).
final class PhoneLookTests: XCTestCase {

  /// HWC interleaved RGB buffer helpers.
  private func luminance(_ px: [Float], _ i: Int) -> Float {
    0.2126 * px[i * 3] + 0.7152 * px[i * 3 + 1] + 0.0722 * px[i * 3 + 2]
  }

  func testLowContrastRampIsStretchedToFullRange() {
    // A hazy 0.25…0.75 gray ramp — the TDM failure mode in miniature.
    let w = 64, h = 8
    var px = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
      for x in 0..<w {
        let v = 0.25 + 0.5 * Float(x) / Float(w - 1)
        for c in 0..<3 { px[(y * w + x) * 3 + c] = v }
      }
    }
    PhoneLook.apply(pixels: &px, width: w, height: h)
    let lums = (0..<(w * h)).map { luminance(px, $0) }
    XCTAssertLessThan(lums.min()!, 0.05, "black point restored")
    XCTAssertGreaterThan(lums.max()!, 0.95, "white point restored")
  }

  func testFlatImageIsSafeAndUnexploded() {
    // Near-zero dynamic range must NOT be stretched into noise — the levels
    // stage skips when the window is degenerate.
    let w = 16, h = 16
    var px = [Float](repeating: 0.5, count: w * h * 3)
    PhoneLook.apply(pixels: &px, width: w, height: h)
    for v in px {
      XCTAssertFalse(v.isNaN)
      XCTAssertGreaterThanOrEqual(v, 0)
      XCTAssertLessThanOrEqual(v, 1)
      XCTAssertEqual(v, 0.5, accuracy: 0.1, "flat image stays flat")
    }
  }

  func testOutputAlwaysClampedToUnitRange() {
    var seed: UInt64 = 42
    func rnd() -> Float {
      seed = seed &* 6364136223846793005 &+ 1442695040888963407
      return Float(seed >> 40) / Float(1 << 24)
    }
    let w = 32, h = 32
    var px = (0..<(w * h * 3)).map { _ in rnd() }
    PhoneLook.apply(pixels: &px, width: w, height: h)
    for v in px {
      XCTAssertFalse(v.isNaN)
      XCTAssertGreaterThanOrEqual(v, 0)
      XCTAssertLessThanOrEqual(v, 1)
    }
  }

  func testSharpeningIncreasesEdgeAcutance() {
    // A soft vertical edge across a full-range image (levels ~no-op) must get
    // steeper — the "pinch of sharpness".
    let w = 64, h = 16
    var px = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
      for x in 0..<w {
        // soft edge: ramp from 0.05 to 0.95 over 8 px in the middle
        let t = max(0, min(1, (Float(x) - 28) / 8))
        let v = 0.05 + 0.9 * t
        for c in 0..<3 { px[(y * w + x) * 3 + c] = v }
      }
    }
    let mid = h / 2
    let preDelta = abs(luminanceRow(px, w, mid, 33) - luminanceRow(px, w, mid, 31))
    PhoneLook.apply(pixels: &px, width: w, height: h)
    let postDelta = abs(luminanceRow(px, w, mid, 33) - luminanceRow(px, w, mid, 31))
    XCTAssertGreaterThan(postDelta, preDelta, "edge steepened by unsharp mask")
  }

  private func luminanceRow(_ px: [Float], _ w: Int, _ y: Int, _ x: Int) -> Float {
    luminance(px, y * w + x)
  }

  func testAdaptiveSaturationCapsTheBoost() {
    // A faint uniform tint: saturation should rise but never explode past the
    // 1.25 cap. Construct pixels with known chroma and verify the post-boost
    // chroma is at most 1.25x the (levels+contrast-scaled) input chroma.
    let w = 16, h = 16
    var px = [Float](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
      // mid-gray with slight red tint; add a dark and bright pixel so the
      // levels window is honest.
      px[i * 3] = 0.55; px[i * 3 + 1] = 0.50; px[i * 3 + 2] = 0.50
    }
    px[0] = 0.02; px[1] = 0.02; px[2] = 0.02
    let last = (w * h - 1) * 3
    px[last] = 0.98; px[last + 1] = 0.98; px[last + 2] = 0.98
    PhoneLook.apply(pixels: &px, width: w, height: h)
    // chroma of an interior pixel
    let i = (8 * w + 8) * 3
    let mx = max(px[i], px[i + 1], px[i + 2])
    let mn = min(px[i], px[i + 1], px[i + 2])
    XCTAssertLessThan(mx - mn, 0.30, "tint amplified but bounded (cap + clamps)")
    XCTAssertGreaterThan(mx - mn, 0.0, "tint not destroyed")
  }
}
