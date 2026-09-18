import Foundation
import XCTest

@testable import ZImage

/// Sequence presets: reusable Director shapes (FDD §4.9.3, WP14).
final class SequencePresetTests: XCTestCase {

  private var dir: URL!
  private var store: SequencePresetStore!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory.appendingPathComponent("seqp-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    store = SequencePresetStore(path: dir.appendingPathComponent("sequence-presets.json"))
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  private func preset(
    _ id: String = "p1", seconds: Double = 10, fps: Int = 24,
    keyframes: String = "first_only", cadence: Double? = nil, audio: String = "generated"
  ) -> SequencePreset {
    SequencePreset(
      id: id, name: "Ten seconds", lengthSeconds: seconds, fps: fps,
      keyframePolicy: keyframes, segmentCadenceSeconds: cadence, audio: audio)
  }

  // MARK: shape

  func testLengthSnapsToTheEnginesFrameGrid() {
    // 10 s at 24 fps = 240 frames -> 241 (1 + 8k).
    XCTAssertEqual(preset(seconds: 10).snappedFrames, 241)
    XCTAssertEqual(preset(seconds: 10).snappedSeconds, 241.0 / 24.0, accuracy: 1e-9)
    // Under the production floor, a preset still promises a renderable length.
    XCTAssertEqual(preset(seconds: 1).snappedFrames, 97)
    // And never more than Director's 16 chunks.
    XCTAssertEqual(preset(seconds: 600).snappedFrames, 4609)
  }

  func testChunkCountMatchesDirectorsCeiling() {
    XCTAssertEqual(preset(seconds: 10).chunkCount, 1)
    XCTAssertEqual(preset(seconds: 30).chunkCount, 3, "30 s at 24 fps is three chunks")
    XCTAssertEqual(preset(seconds: 600).chunkCount, 16)
  }

  func testKeyframePolicies() {
    XCTAssertEqual(preset(keyframes: "none").keyframeFrames(), [])
    XCTAssertEqual(preset(keyframes: "first_only").keyframeFrames(), [0])
    XCTAssertEqual(preset(keyframes: "fflf").keyframeFrames(), [0, 240])
    let every4 = preset(seconds: 12, keyframes: "every_n_seconds:4").keyframeFrames()
    XCTAssertEqual(every4.first, 0)
    XCTAssertTrue(every4.allSatisfy { $0 % 8 == 0 }, "keyframes land on the latent grid")
    XCTAssertGreaterThan(every4.count, 2)
  }

  func testSegmentSpansCoverTheClipExactly() {
    let spans = preset(seconds: 12, cadence: 3).segmentSpans()
    XCTAssertEqual(spans.first?.start, 0)
    XCTAssertEqual(spans.map(\.length).reduce(0, +), preset(seconds: 12, cadence: 3).snappedFrames)
    // No cadence: one segment for the whole clip.
    XCTAssertEqual(preset(seconds: 12).segmentSpans().count, 1)
  }

  func testAFinalSliverIsFoldedIntoThePreviousSegment() {
    // 10 s at 24 fps = 241 frames with a 4 s (96-frame) cadence: 96, 96, 49.
    // 49 is more than half a step, so it stays its own segment…
    XCTAssertEqual(preset(seconds: 10, cadence: 4).segmentSpans().count, 3)
    // …but a 7-frame tail would not.
    let spans = preset(seconds: 10, cadence: 4.875).segmentSpans()
    XCTAssertTrue(spans.allSatisfy { $0.length > 20 }, "no one-frame prompt segments")
  }

  func testSeedPolicy() {
    XCTAssertNil(preset().seed)
    var fixed = preset()
    fixed.seedPolicy = "fixed:4242"
    XCTAssertEqual(fixed.seed, 4242)
  }

  // MARK: validation

  func testValidationRefusesNonsense() {
    func expectInvalid(_ mutate: (inout SequencePreset) -> Void) {
      var p = preset()
      mutate(&p)
      XCTAssertThrowsError(try store.upsert(p))
    }
    expectInvalid { $0.id = "  " }
    expectInvalid { $0.name = "" }
    expectInvalid { $0.lengthSeconds = 0 }
    expectInvalid { $0.lengthSeconds = 601 }
    expectInvalid { $0.fps = 0 }
    expectInvalid { $0.width = 0 }
    expectInvalid { $0.audio = "sometimes" }
    expectInvalid { $0.keyframePolicy = "whenever" }
    expectInvalid { $0.seedPolicy = "fixed:abc" }
    expectInvalid { $0.segmentCadenceSeconds = -1 }
  }

  func testADrivingVoiceNeedsAFaceToDrive() {
    var p = preset(keyframes: "none", audio: "driven")
    XCTAssertThrowsError(try store.upsert(p)) { error in
      XCTAssertEqual(
        error as? SequencePresetError,
        .invalid("'audio: driven' needs at least a first keyframe — the voice drives a face"))
    }
    p.keyframePolicy = "first_only"
    XCTAssertNoThrow(try store.upsert(p))
  }

  // MARK: store

  func testUpsertPersistsAndKeepsCreatedAt() throws {
    let first = try store.upsert(preset())
    let second = try store.upsert(preset(seconds: 20))
    XCTAssertEqual(second.createdAt, first.createdAt, "an edit does not reset createdAt")
    XCTAssertEqual(second.lengthSeconds, 20)

    let reopened = SequencePresetStore(path: dir.appendingPathComponent("sequence-presets.json"))
    XCTAssertEqual(reopened.preset(id: "p1")?.lengthSeconds, 20)
    XCTAssertEqual(reopened.all().count, 1)
  }

  func testDelete() throws {
    try store.upsert(preset())
    XCTAssertTrue(try store.delete(id: "p1"))
    XCTAssertFalse(try store.delete(id: "p1"))
    XCTAssertTrue(store.all().isEmpty)
  }

  func testWireShapeIsSnakeCaseAndRoundTrips() throws {
    var p = preset(seconds: 30, cadence: 3, audio: "driven")
    p.videoPresetId = "kira-video-apple"
    p.authorNotes = "static camera"
    let data = try JSONEncoder().encode(p)
    let raw = try XCTUnwrap(String(data: data, encoding: .utf8))
    XCTAssertTrue(raw.contains("\"length_seconds\""))
    XCTAssertTrue(raw.contains("\"video_preset_id\""))
    XCTAssertTrue(raw.contains("\"keyframe_policy\""))
    var decoded = try JSONDecoder().decode(SequencePreset.self, from: data)
    // Timestamps are ISO-8601 to the second on the wire, so compare the rest.
    decoded.createdAt = p.createdAt
    decoded.updatedAt = p.updatedAt
    XCTAssertEqual(decoded, p)
  }

  // MARK: routes

  private func status(_ response: RoutedResponse) -> Int {
    switch response {
    case .json(let http), .error(let http), .shutdown(let http): return http.status
    case .websocketUpgrade: return -1
    }
  }

  private func json(_ response: RoutedResponse) throws -> [String: Any] {
    switch response {
    case .json(let http), .error(let http), .shutdown(let http):
      return (try JSONSerialization.jsonObject(with: http.body) as? [String: Any]) ?? [:]
    case .websocketUpgrade: return [:]
    }
  }

  func testUpsertRouteReturnsTheShapeAPresetImplies() throws {
    let body = try JSONSerialization.data(withJSONObject: [
      "id": "apple-30", "name": "Apple 30s", "length_seconds": 30,
      "keyframe_policy": "first_only", "segment_cadence_seconds": 10, "audio": "driven",
    ])
    let (response, saved) = WarmServer.upsertSequencePreset(store: store, body: body)
    XCTAssertEqual(status(response), 200)
    XCTAssertEqual(saved?.id, "apple-30")

    let payload = try json(response)
    XCTAssertEqual(payload["snapped_frames"] as? Int, 721)
    XCTAssertEqual(payload["chunks"] as? Int, 3)
    XCTAssertEqual(payload["estimated_gpu_minutes"] as? Int, 75, "3 chunks × 25 min")
    XCTAssertEqual((payload["keyframe_frames"] as? [Int]), [0])
    XCTAssertEqual((payload["segments"] as? [[String: Int]])?.count, 3)
  }

  func testUpsertRouteRefusesAnInvalidPresetAndStoresNothing() throws {
    let body = try JSONSerialization.data(withJSONObject: [
      "id": "bad", "name": "Bad", "audio": "sometimes",
    ])
    let (response, saved) = WarmServer.upsertSequencePreset(store: store, body: body)
    XCTAssertEqual(status(response), 400)
    XCTAssertNil(saved)
    XCTAssertTrue(store.all().isEmpty)
  }
}
