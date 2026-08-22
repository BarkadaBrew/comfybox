// Krea2RawChecksumTests.swift — WP-E10 "E9b" (FDD §7.1, Addendum A.2).
//
// The Raw DiT on disk is the Comfy-Org mirror's `krea2_raw_bf16.safetensors`,
// not the gated official repo's file; its SHA-256 is pinned in
// `Tests/ZImageTests/Fixtures/krea2-raw.sha256` so a silently replaced or
// truncated download is caught before a "parity" render is read as evidence.
// Hashing 26 GB takes ~2 min — integration batch only (FDD §5.3, Raw batch),
// skipped with a named message unless the file is present.

import XCTest
@testable import ZImage

final class Krea2RawChecksumTests: XCTestCase {

  private static let fixture = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()   // ZImageIntegrationTests/
    .deletingLastPathComponent()   // Tests/
    .appendingPathComponent("ZImageTests/Fixtures/krea2-raw.sha256")

  func testRawSafetensorsMatchesThePinnedSHA256() throws {
    let raw = URL(fileURLWithPath: ("~/LocalModels/krea2-raw/raw.safetensors" as NSString).expandingTildeInPath)
    try XCTSkipUnless(
      FileManager.default.fileExists(atPath: raw.path),
      "~/LocalModels/krea2-raw/raw.safetensors not on disk — SHA pin not verifiable here")
    let pin = try Krea2RawChecksum.parseShasumLine(String(contentsOf: Self.fixture, encoding: .utf8))
    XCTAssertEqual(pin.filename, raw.lastPathComponent)
    let actual = try Krea2RawChecksum.sha256Hex(of: raw)
    XCTAssertEqual(
      actual, pin.sha256,
      "raw.safetensors on disk does not match the pinned SHA-256 — the file was replaced or is incomplete; "
        + "re-pin ONLY after verifying the source (FDD §7.1)")
  }
}
