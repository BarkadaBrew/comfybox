import XCTest
@testable import ZImage

/// WP2c: the aggregate director sidecar's additive fields
/// (`audio_source`, `chunk_count`, `stitch_path`).
final class DirectorRecordTests: XCTestCase {

  func testAggregateRecordFieldsRoundTrip() throws {
    let record = VideoGenerationRecord(
      prompt: "global", seed: 42, model: "ltx2-director", width: 576, height: 896,
      frames: 577, fps: 24, resolvedWidth: 576, resolvedHeight: 896,
      twoPass: false, refine: false, audio: true, kind: "director",
      audioSource: "imported", chunkCount: 2, stitchPath: "reencode")
    let data = try record.encodeJSON()
    let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    XCTAssertEqual(obj["audio_source"] as? String, "imported")
    XCTAssertEqual(obj["chunk_count"] as? Int, 2)
    XCTAssertEqual(obj["stitch_path"] as? String, "reencode")
    XCTAssertEqual(obj["kind"] as? String, "director")
    XCTAssertEqual(try VideoGenerationRecord.decodeJSON(data), record)

    let plain = VideoGenerationRecord(
      prompt: "p", model: "m", width: 1, height: 1, frames: 1, fps: 24,
      resolvedWidth: 1, resolvedHeight: 1, twoPass: false, refine: false, audio: false, kind: "t2v")
    let plainObj = try XCTUnwrap(JSONSerialization.jsonObject(with: plain.encodeJSON()) as? [String: Any])
    XCTAssertNil(plainObj["audio_source"])
    XCTAssertNil(plainObj["chunk_count"])
    XCTAssertNil(plainObj["stitch_path"])
    // A sidecar written before WP2c still decodes.
    XCTAssertNil(try VideoGenerationRecord.decodeJSON(plain.encodeJSON()).chunkCount)
  }
}
