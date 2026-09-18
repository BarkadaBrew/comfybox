import Foundation
import XCTest

@testable import ZImage

/// The Sequence document: Director's saved, replayable product
/// (FDD-ltx-director-tab §4.9.2, WP13).
final class SequenceDocumentTests: XCTestCase {

  private var dir: URL!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory.appendingPathComponent("seq-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  private func timeline(keyframeAt path: String? = nil) -> DirectorTimeline {
    DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 289),
      globalPrompt: "a barista at a counter",
      keyframes: path.map { [.init(id: "k1", imagePath: $0, frame: 0)] } ?? [],
      promptSegments: [
        .init(id: "p1", startFrame: 0, lengthFrames: 144, prompt: "she stirs her coffee"),
        .init(id: "p2", startFrame: 144, lengthFrames: 145, prompt: "she waves"),
      ])
  }

  private func document(timeline: DirectorTimeline) -> SequenceDocument {
    SequenceDocument(
      id: "ABC123",
      name: "clip",
      source: "ladder",
      timeline: timeline,
      chunks: [SequenceChunkRecord(index: 0, startFrame: 0, frames: 289, seed: 42, recipeHash: "r1")],
      outputs: [SequenceOutput(kind: "mp4", path: "/out/clip.mp4", frames: 289, durationSeconds: 12.04)],
      engine: SequenceEngineRecord(buildSha: "abc1234", stitchPath: "reencode", toneMatch: true))
  }

  // MARK: round trip

  func testSidecarRoundTripsBesideTheMedia() throws {
    let media = dir.appendingPathComponent("clip.mp4").path
    let doc = document(timeline: timeline())
    XCTAssertTrue(SequenceSidecar.write(doc, forMediaAt: media))
    XCTAssertEqual(
      SequenceSidecar.path(forMediaAt: media), dir.appendingPathComponent("clip.sequence.json").path)

    let read = try XCTUnwrap(SequenceSidecar.read(forMediaAt: media))
    XCTAssertEqual(read.id, "ABC123")
    XCTAssertEqual(read.schema, "comfybox.sequence")
    XCTAssertEqual(read.timeline.globalPrompt, "a barista at a counter")
    XCTAssertEqual(read.timeline.promptSegments.count, 2, "the timeline comes back whole")
    XCTAssertEqual(read.chunks.first?.seed, 42)
    XCTAssertEqual(read.engine.buildSha, "abc1234")
    XCTAssertEqual(read.outputs.first?.frames, 289)
  }

  func testWireShapeIsSnakeCase() throws {
    let media = dir.appendingPathComponent("clip.mp4").path
    SequenceSidecar.write(document(timeline: timeline()), forMediaAt: media)
    let raw = try String(
      contentsOf: URL(fileURLWithPath: SequenceSidecar.path(forMediaAt: media)), encoding: .utf8)
    XCTAssertTrue(raw.contains("\"created_at\""))
    XCTAssertTrue(raw.contains("\"build_sha\""))
    XCTAssertTrue(raw.contains("\"start_frame\""))
    XCTAssertFalse(raw.contains("\"createdAt\""))
  }

  func testAFutureSchemaVersionIsRefusedRatherThanMisread() throws {
    let path = dir.appendingPathComponent("future.sequence.json")
    let json = """
      {"schema":"comfybox.sequence","version":99,"id":"x","name":"x",
       "timeline":{"version":1,"settings":{"width":576,"height":896,"length_frames":289},
       "global_prompt":"x","keyframes":[],"prompt_segments":[],"audio_clips":[]}}
      """
    try Data(json.utf8).write(to: path)
    XCTAssertNil(SequenceSidecar.read(at: path.path), "a newer document is not guessed at")
  }

  func testMissingSidecarReadsAsNil() {
    XCTAssertNil(SequenceSidecar.read(forMediaAt: dir.appendingPathComponent("nope.mp4").path))
  }

  // MARK: assets

  func testAssetsAreHashedFromTheTimeline() throws {
    let keyframe = dir.appendingPathComponent("frame.png")
    try Data("pretend png".utf8).write(to: keyframe)
    let assets = SequenceSidecar.assets(of: timeline(keyframeAt: keyframe.path))
    XCTAssertEqual(assets.count, 1)
    XCTAssertEqual(assets.first?.role, "keyframe")
    XCTAssertEqual(assets.first?.sha256?.count, 64)
  }

  func testAMissingAssetIsStillRecordedWithoutAHash() {
    let assets = SequenceSidecar.assets(of: timeline(keyframeAt: "/no/such/frame.png"))
    XCTAssertEqual(assets.first?.path, "/no/such/frame.png")
    XCTAssertNil(assets.first?.sha256)
  }

  // MARK: replay check

  func testReplayCheckIsCleanWhenNothingMoved() throws {
    let keyframe = dir.appendingPathComponent("frame.png")
    try Data("pretend png".utf8).write(to: keyframe)
    var doc = document(timeline: timeline(keyframeAt: keyframe.path))
    doc.assets = SequenceSidecar.assets(of: doc.timeline)
    let check = SequenceSidecar.check(doc, currentRecipeHash: "r1", currentEngineBuild: "abc1234")
    XCTAssertTrue(check.ok)
  }

  func testReplayCheckNamesMissingAndChangedAssets() throws {
    let keyframe = dir.appendingPathComponent("frame.png")
    try Data("original".utf8).write(to: keyframe)
    var doc = document(timeline: timeline(keyframeAt: keyframe.path))
    doc.assets = SequenceSidecar.assets(of: doc.timeline)
    doc.assets.append(SequenceAsset(role: "audio", path: "/no/such/voice.wav", sha256: "deadbeef"))

    try Data("edited".utf8).write(to: keyframe)
    let check = SequenceSidecar.check(doc, currentRecipeHash: "r1", currentEngineBuild: "abc1234")
    XCTAssertEqual(check.changedAssets, [keyframe.path])
    XCTAssertEqual(check.missingAssets, ["/no/such/voice.wav"])
    XCTAssertFalse(check.ok, "a replay that would look different is not reported as fine")
  }

  func testReplayCheckFlagsRecipeAndEngineDrift() {
    let doc = document(timeline: timeline())
    let drift = SequenceSidecar.check(doc, currentRecipeHash: "r2", currentEngineBuild: "abc1234")
    XCTAssertTrue(drift.recipeDrift)
    XCTAssertFalse(drift.engineChanged)

    let newEngine = SequenceSidecar.check(doc, currentRecipeHash: "r1", currentEngineBuild: "zzz9999")
    XCTAssertTrue(newEngine.engineChanged)
    XCTAssertFalse(newEngine.recipeDrift)

    let unknown = SequenceSidecar.check(doc, currentRecipeHash: nil, currentEngineBuild: nil)
    XCTAssertFalse(unknown.recipeDrift, "unknown is not drift")
    XCTAssertFalse(unknown.engineChanged)
  }

  // MARK: routes

  private func status(_ response: RoutedResponse) -> Int {
    switch response {
    case .json(let http), .error(let http), .shutdown(let http): return http.status
    case .websocketUpgrade: return -1
    }
  }

  private func json(_ response: RoutedResponse) throws -> Any {
    switch response {
    case .json(let http), .error(let http), .shutdown(let http):
      return try JSONSerialization.jsonObject(with: http.body)
    case .websocketUpgrade: return [:]
    }
  }

  func testReadRouteFindsTheSidecarFromTheMediaPath() throws {
    let media = dir.appendingPathComponent("clip.mp4").path
    SequenceSidecar.write(document(timeline: timeline()), forMediaAt: media)
    let body = try JSONSerialization.data(withJSONObject: ["path": media])
    let response = WarmServer.sequenceRead(body: body, allowedOutputDirectory: dir.path)
    XCTAssertEqual(status(response), 200)
    let payload = try XCTUnwrap(try json(response) as? [String: Any])
    XCTAssertEqual(payload["id"] as? String, "ABC123")
  }

  func testReadRouteIs404ForAClipRenderedBeforeSequences() throws {
    let body = try JSONSerialization.data(
      withJSONObject: ["path": dir.appendingPathComponent("old.mp4").path])
    XCTAssertEqual(status(WarmServer.sequenceRead(body: body, allowedOutputDirectory: dir.path)), 404)
  }

  func testListReturnsNewestFirstAndRespectsTheLimit() throws {
    for (index, name) in ["a", "b", "c"].enumerated() {
      var doc = document(timeline: timeline())
      doc.id = name
      doc.createdAt = Date(timeIntervalSince1970: TimeInterval(1_000 + index * 60))
      SequenceSidecar.write(doc, forMediaAt: dir.appendingPathComponent("\(name).mp4").path)
    }
    let all = try XCTUnwrap(
      try json(WarmServer.sequenceList(directory: dir.path, query: [:])) as? [[String: Any]])
    XCTAssertEqual(all.compactMap { $0["id"] as? String }, ["c", "b", "a"])
    XCTAssertEqual(all.first?["chunks"] as? Int, 1)

    let limited = try XCTUnwrap(
      try json(WarmServer.sequenceList(directory: dir.path, query: ["limit": "2"])) as? [[String: Any]])
    XCTAssertEqual(limited.count, 2)
  }

  func testListIgnoresUnrelatedFiles() throws {
    try Data("{}".utf8).write(to: dir.appendingPathComponent("clip.json"))
    try Data("not json".utf8).write(to: dir.appendingPathComponent("broken.sequence.json"))
    let rows = try XCTUnwrap(
      try json(WarmServer.sequenceList(directory: dir.path, query: [:])) as? [[String: Any]])
    XCTAssertTrue(rows.isEmpty, "a generation sidecar is not a sequence, and a broken one is skipped")
  }
}
