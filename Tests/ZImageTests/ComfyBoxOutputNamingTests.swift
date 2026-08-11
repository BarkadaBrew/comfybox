import XCTest

@testable import ZImage

/// Gallery filenames (Todd 2026-08-11): comfybox-<model>-<tier>-<stamp>-<salt>
/// instead of zimage-krea2-<uuid> — names that say what rendered them.
final class ComfyBoxOutputNamingTests: XCTestCase {

  private let fixedDate = Date(timeIntervalSince1970: 1_776_000_000)

  func testDirectorySpecUsesLastComponentAndTier() {
    let name = ComfyBoxOutputNaming.defaultFilename(
      modelSpec: "/Users/toddwalderman/LocalModels/kroma-v0.2",
      contentMode: "avocado", date: fixedDate)
    XCTAssertTrue(name.hasPrefix("comfybox-kroma-v0.2-avocado-"), name)
    XCTAssertTrue(name.hasSuffix(".png"))
    // shape: prefix + yyyyMMdd-HHmmss + 4-hex salt
    let pattern = #"^comfybox-kroma-v0\.2-avocado-\d{8}-\d{6}-[0-9a-f]{4}\.png$"#
    XCTAssertNotNil(name.range(of: pattern, options: .regularExpression), name)
  }

  func testAliasSpecAndMissingModeFallBack() {
    let name = ComfyBoxOutputNaming.defaultFilename(
      modelSpec: "krea2", contentMode: nil, date: fixedDate)
    XCTAssertTrue(name.hasPrefix("comfybox-krea2-manual-"), name)

    let none = ComfyBoxOutputNaming.defaultFilename(
      modelSpec: nil, contentMode: "banana", date: fixedDate)
    XCTAssertTrue(none.hasPrefix("comfybox-model-banana-"), none)
  }

  func testSanitizationCollapsesJunk() {
    XCTAssertEqual(ComfyBoxOutputNaming.sanitize("Kroma V0.2 (Turbo)!"), "kroma-v0.2-turbo")
    XCTAssertEqual(ComfyBoxOutputNaming.shortModelName("~/Models/My Model@8bit"), "my-model-8bit")
  }

  func testSaltVariesAcrossCalls() {
    let names = Set((0..<8).map {
      _ in ComfyBoxOutputNaming.defaultFilename(
        modelSpec: "krea2", contentMode: "apple", date: fixedDate)
    })
    XCTAssertGreaterThan(names.count, 1, "salt should differentiate same-second renders")
  }
}
