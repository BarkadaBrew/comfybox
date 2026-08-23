import XCTest
@testable import ZImage

/// WP-E10 "E9b" (FDD §7.1 row "Krea 2 Raw DiT", Addendum A.2): the SHA-256 of
/// `~/LocalModels/krea2-raw/raw.safetensors` is PINNED in
/// `Tests/ZImageTests/Fixtures/krea2-raw.sha256` (computed 2026-08-22 with
/// `shasum -a 256`, Comfy-Org `krea2_raw_bf16.safetensors`, 26,283,332,608 B).
/// This unit test keeps the fixture well-formed in the default gate; the
/// 26 GB hash itself runs in `ZImageIntegrationTests.Krea2RawChecksumTests`,
/// skipped unless the file is present.
final class Krea2RawChecksumFixtureTests: XCTestCase {

  static let fixture = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()   // Krea2/
    .deletingLastPathComponent()   // ZImageTests/
    .appendingPathComponent("Fixtures/krea2-raw.sha256")

  func testFixtureIsAWellFormedShasumLine() throws {
    let pin = try Krea2RawChecksum.parseShasumLine(String(contentsOf: Self.fixture, encoding: .utf8))
    XCTAssertEqual(pin.filename, "raw.safetensors")
    XCTAssertEqual(pin.sha256.count, 64)
    XCTAssertTrue(pin.sha256.allSatisfy { $0.isHexDigit && ($0.isNumber || $0.isLowercase) }, pin.sha256)
    XCTAssertEqual(pin.sha256, "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7")
  }

  func testShasumLineParserRejectsGarbage() {
    XCTAssertThrowsError(try Krea2RawChecksum.parseShasumLine("not a shasum line"))
    XCTAssertThrowsError(try Krea2RawChecksum.parseShasumLine("abc  raw.safetensors"))
    XCTAssertThrowsError(try Krea2RawChecksum.parseShasumLine(String(repeating: "g", count: 64) + "  raw.safetensors"))
    XCTAssertThrowsError(try Krea2RawChecksum.parseShasumLine(""))
    let ok = try? Krea2RawChecksum.parseShasumLine(String(repeating: "A", count: 64) + "  raw.safetensors\n")
    XCTAssertEqual(ok?.filename, "raw.safetensors")
    XCTAssertEqual(ok?.sha256, String(repeating: "a", count: 64), "normalised to lowercase hex")
  }

  /// The streaming hasher agrees with `shasum -a 256` on a small known input.
  func testSHA256OfASmallFileMatchesAKnownDigest() throws {
    let file = FileManager.default.temporaryDirectory.appending(path: "sha-\(UUID()).bin")
    try Data("abc".utf8).write(to: file)
    defer { try? FileManager.default.removeItem(at: file) }
    XCTAssertEqual(
      try Krea2RawChecksum.sha256Hex(of: file),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  }
}
