import XCTest
@testable import ZImage

/// The embedded `parametersJSON` must be BYTE-STABLE for equal inputs, in this
/// process and the next one.
///
/// `JSONSerialization` walks a Swift `Dictionary` in hash order, and Swift
/// seeds its hasher per process — so the same render wrote a different byte
/// sequence into EXIF `UserComment` every time the server restarted, and the
/// whole-file SHA of a PNG (AC-5's byte-identity oracle) could never be
/// compared across runs. Sorting the keys is the whole fix; these tests pin
/// it, since a same-process repeat would pass either way.
final class ImageMetadataDeterminismTests: XCTestCase {

  /// The keys of one JSON object, in the order they physically appear in the
  /// text (not the order a parser hands them back).
  private func textualKeyOrder(_ json: String, inObjectStartingAt start: String.Index) -> [String] {
    var keys: [String] = []
    var depth = 0
    var index = start
    var inString = false
    var escaped = false
    var current = ""
    var expectingKey = false
    while index < json.endIndex {
      let ch = json[index]
      if inString {
        if escaped { escaped = false; current.append(ch) }
        else if ch == "\\" { escaped = true }
        else if ch == "\"" {
          inString = false
          if depth == 1, expectingKey {
            // A key is a string immediately followed by ':'.
            var peek = json.index(after: index)
            while peek < json.endIndex, json[peek] == " " { peek = json.index(after: peek) }
            if peek < json.endIndex, json[peek] == ":" { keys.append(current) }
          }
        } else { current.append(ch) }
      } else {
        switch ch {
        case "{", "[":
          depth += 1
          expectingKey = (ch == "{")
        case "}", "]":
          depth -= 1
          if depth == 0 { return keys }
        case "\"":
          inString = true; current = ""
        case ",":
          expectingKey = (depth == 1)
        default: break
        }
      }
      index = json.index(after: index)
    }
    return keys
  }

  private func topLevelKeys(_ json: String) throws -> [String] {
    let start = try XCTUnwrap(json.firstIndex(of: "{"))
    return textualKeyOrder(json, inObjectStartingAt: start)
  }

  private func metadata(applied: RenderRecipe?) -> QwenImageIO.ImageMetadata {
    QwenImageIO.ImageMetadata.generation(
      prompt: "a wooden table by a window", negativePrompt: "blurry", seed: 44821,
      steps: 30, guidance: 2.0, width: 1024, height: 1024, model: "krea2-raw",
      generatedBy: "bree", contentMode: "apple",
      loras: [.local("/vault/kroma.safetensors", scale: 0.3)], applied: applied)
  }

  func testTopLevelKeysAreSorted() throws {
    let json = try XCTUnwrap(metadata(applied: nil).parametersJSON)
    let keys = try topLevelKeys(json)
    XCTAssertFalse(keys.isEmpty)
    XCTAssertEqual(keys, keys.sorted(), "unsorted keys make the PNG bytes process-dependent: \(keys)")
  }

  /// The nested record is part of the same payload and must sort too — the
  /// `applied` block is by far the biggest object in the comment.
  func testTheAppliedRecordIsSortedToo() throws {
    let json = try XCTUnwrap(metadata(applied: RenderRecipeFixture.recipe(guidance: 2.0, negativePrompt: "blurry")).parametersJSON)
    let appliedStart = try XCTUnwrap(json.range(of: "\"applied\":"))
    let objectStart = try XCTUnwrap(json[appliedStart.upperBound...].firstIndex(of: "{"))
    let keys = textualKeyOrder(json, inObjectStartingAt: objectStart)
    XCTAssertTrue(keys.contains("base_variant"), "\(keys)")
    XCTAssertEqual(keys, keys.sorted(), "\(keys)")
  }

  func testTwoEncodesOfEqualMetadataAreByteIdentical() throws {
    let recipe = RenderRecipeFixture.recipe(guidance: 2.0, negativePrompt: "blurry")
    let a = try XCTUnwrap(metadata(applied: recipe).parametersJSON)
    let b = try XCTUnwrap(metadata(applied: recipe).parametersJSON)
    XCTAssertEqual(Data(a.utf8), Data(b.utf8))
  }

  /// The values survive the sort — this is an ordering change, not a
  /// re-shaping of the sidecar.
  func testSortingDoesNotChangeTheContent() throws {
    let json = try XCTUnwrap(metadata(applied: RenderRecipeFixture.recipe()).parametersJSON)
    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    XCTAssertEqual(object["prompt"] as? String, "a wooden table by a window")
    XCTAssertEqual(object["negative_prompt"] as? String, "blurry")
    XCTAssertEqual(object["seed"] as? UInt64, 44821)
    XCTAssertEqual(object["model"] as? String, "krea2-raw")
    XCTAssertEqual((object["loras"] as? [[String: Any]])?.first?["scale"] as? Double, 0.3)
    XCTAssertEqual((object["applied"] as? [String: Any])?["base_variant"] as? String, "raw")
  }
}
