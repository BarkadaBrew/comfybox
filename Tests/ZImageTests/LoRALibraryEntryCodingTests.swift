import Foundation
import XCTest

@testable import ZImage

/// #313 (review round 1, Important finding 2): decode-tolerance coverage for
/// the additive `compatibility_source` field on `LoRALibraryEntry`. Existing
/// `library.json` files on disk predate this field entirely, and a future
/// build could write a value this build doesn't recognize — neither may ever
/// fail to load the library (the same tolerance `krea2_relative` already gets,
/// per the comment right above this field's decode in
/// `LoRALibraryEntry.swift`).
final class LoRALibraryEntryCodingTests: XCTestCase {

  /// A complete, valid `LoRALibraryEntry` JSON blob with every REQUIRED field
  /// present. `extra` is spliced in just before the closing brace so callers
  /// can add/omit `compatibility_source` without hand-maintaining the whole
  /// object twice.
  private func entryJSON(extra: String) -> String {
    """
    {
      "id": "manual-tag-test",
      "filename": "manual-tag-test.safetensors",
      "relative_path": "manual-tag-test.safetensors",
      "size_bytes": 1024,
      "model_compatibility": ["ltx"],
      "format": "lora",
      "rank": 64,
      "key_count": 10,
      "layer_targets": ["attention"],
      "triggerwords": [],
      "recommended_scale": 1.0,
      "scale_range": [0.0, 2.0],
      "tags": [],
      "category": "uncategorized",
      "notes": "",
      "date_added": "2026-09-04",
      "quarantined": false
      \(extra)
    }
    """
  }

  private func decode(_ json: String) throws -> LoRALibraryEntry {
    try JSONDecoder().decode(LoRALibraryEntry.self, from: Data(json.utf8))
  }

  func testMissingCompatibilitySourceDecodesAsAuto() throws {
    let entry = try decode(entryJSON(extra: ""))
    XCTAssertEqual(entry.compatibilitySource, .auto)
  }

  func testUnrecognizedCompatibilitySourceDecodesAsAuto() throws {
    let entry = try decode(entryJSON(extra: #", "compatibility_source": "some-future-value""#))
    XCTAssertEqual(entry.compatibilitySource, .auto)
  }

  func testManualCompatibilitySourceRoundTrips() throws {
    let entry = try decode(entryJSON(extra: #", "compatibility_source": "manual""#))
    XCTAssertEqual(entry.compatibilitySource, .manual)

    // Round-trip through encode -> decode: .manual must not decay to .auto.
    let encoder = JSONEncoder()
    let reencoded = try encoder.encode(entry)
    let redecoded = try JSONDecoder().decode(LoRALibraryEntry.self, from: reencoded)
    XCTAssertEqual(redecoded.compatibilitySource, .manual)
  }

  func testAutoCompatibilitySourceRoundTrips() throws {
    let entry = try decode(entryJSON(extra: #", "compatibility_source": "auto""#))
    XCTAssertEqual(entry.compatibilitySource, .auto)

    let encoder = JSONEncoder()
    let reencoded = try encoder.encode(entry)
    let redecoded = try JSONDecoder().decode(LoRALibraryEntry.self, from: reencoded)
    XCTAssertEqual(redecoded.compatibilitySource, .auto)
  }

  // MARK: - #273 fix round 1 (C1): additive `anchored` field

  func testMissingAnchoredDecodesAsFalse() throws {
    let entry = try decode(entryJSON(extra: ""))
    XCTAssertEqual(entry.anchored, false)
  }

  func testAnchoredTrueRoundTrips() throws {
    let entry = try decode(entryJSON(extra: #", "anchored": true"#))
    XCTAssertEqual(entry.anchored, true)

    let reencoded = try JSONEncoder().encode(entry)
    let redecoded = try JSONDecoder().decode(LoRALibraryEntry.self, from: reencoded)
    XCTAssertEqual(redecoded.anchored, true)
  }
}
