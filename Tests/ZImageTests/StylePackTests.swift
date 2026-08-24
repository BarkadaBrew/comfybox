import XCTest
@testable import ZImage

/// Unit tests for the ``StylePack`` registry (Todd 2026-08-24 "I prefer style
/// pack"): named, deterministic, engine-applied post-process looks. v1 pack:
/// `phone` (the PhoneLook recipe) and `trix-bw` (guaranteed monochrome
/// pushed-film look — prompts alone rendered Tri-X in color).
final class StylePackTests: XCTestCase {

  private func lum(_ px: [Float], _ i: Int) -> Float {
    0.2126 * px[i * 3] + 0.7152 * px[i * 3 + 1] + 0.0722 * px[i * 3 + 2]
  }

  /// A small synthetic color scene: gradient + colored regions + true
  /// black/white pixels so levels have an honest window.
  private func scene(w: Int, h: Int) -> [Float] {
    var px = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
      for x in 0..<w {
        let i = (y * w + x) * 3
        let t = Float(x) / Float(w - 1)
        px[i] = 0.2 + 0.6 * t          // red ramp
        px[i + 1] = 0.5                 // flat green
        px[i + 2] = 0.8 - 0.6 * t       // blue counter-ramp
      }
    }
    px[0] = 0.02; px[1] = 0.02; px[2] = 0.02
    let last = (w * h - 1) * 3
    px[last] = 0.97; px[last + 1] = 0.97; px[last + 2] = 0.97
    return px
  }

  func testRegistryResolvesKnownNamesAndRejectsUnknown() {
    XCTAssertNotNil(StylePack.named("phone"))
    XCTAssertNotNil(StylePack.named("trix-bw"))
    XCTAssertNil(StylePack.named("sepia-dreams"))
  }

  func testPhoneStyleMatchesPhoneLookRecipe() {
    let w = 32, h = 32
    var viaStyle = scene(w: w, h: h)
    var viaPhoneLook = scene(w: w, h: h)
    StylePack.named("phone")!.apply(pixels: &viaStyle, width: w, height: h)
    PhoneLook.apply(pixels: &viaPhoneLook, width: w, height: h)
    XCTAssertEqual(viaStyle, viaPhoneLook, "style \"phone\" IS the PhoneLook recipe")
  }

  func testTrixBWIsTrulyMonochrome() {
    let w = 32, h = 32
    var px = scene(w: w, h: h)
    StylePack.named("trix-bw")!.apply(pixels: &px, width: w, height: h)
    for i in 0..<(w * h) {
      XCTAssertEqual(px[i * 3], px[i * 3 + 1], accuracy: 1e-5)
      XCTAssertEqual(px[i * 3 + 1], px[i * 3 + 2], accuracy: 1e-5)
    }
  }

  func testTrixBWHasPushedFilmTonality() {
    let w = 64, h = 64
    var px = scene(w: w, h: h)
    StylePack.named("trix-bw")!.apply(pixels: &px, width: w, height: h)
    let lums = (0..<(w * h)).map { lum(px, $0) }
    XCTAssertLessThan(lums.min()!, 0.02, "deep blacks")
    XCTAssertGreaterThan(lums.max()!, 0.9, "highlights reach up")
    for v in px {
      XCTAssertFalse(v.isNaN)
      XCTAssertGreaterThanOrEqual(v, 0)
      XCTAssertLessThanOrEqual(v, 1)
    }
  }

  func testTrixBWFlatImageSafe() {
    let w = 16, h = 16
    var px = [Float](repeating: 0.5, count: w * h * 3)
    StylePack.named("trix-bw")!.apply(pixels: &px, width: w, height: h)
    for v in px {
      XCTAssertFalse(v.isNaN)
      XCTAssertGreaterThanOrEqual(v, 0)
      XCTAssertLessThanOrEqual(v, 1)
    }
  }
}
