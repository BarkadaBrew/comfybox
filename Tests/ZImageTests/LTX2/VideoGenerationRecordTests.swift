import XCTest
@testable import ZImage

/// comfybox#401: the video generation record is the mp4 twin of the PNG side's
/// `ImageMetadata.generation` (EXIF `UserComment` JSON). All of this is pure —
/// no model weights — per intent.md's "agents run unit tests only".
final class VideoGenerationRecordTests: XCTestCase {

  // MARK: - kind()

  func testKindClassifiesFromInitImageAndExtend() {
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: nil, extendToSeconds: 0), "t2v")
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: nil, extendToSeconds: 8), "t2v",
                   "extend_to_seconds is meaningless without an init image — still t2v")
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: "/tmp/src.png", extendToSeconds: 0), "i2v")
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: "/tmp/src.png", extendToSeconds: 8), "extend")
  }

  // MARK: - build() matches the request field-for-field (ruling 4: "a test that

  // the record for a t2v matches the request fields")

  func testBuildMatchesT2VRequestFields() {
    let request = LTX2VideoRequest(
      prompt: "a fox in a snowy forest",
      negativePrompt: "blurry",
      width: 704, height: 448,
      framesPerChunk: 97,
      steps: 8,
      seed: 12345,
      guidance: 3.5,
      loras: [LTX2LoRAReference(path: "/loras/motion_v2.safetensors", scale: 0.8)],
      outputPath: "/tmp/out.mp4",
      audio: false,
      source: "bree",
      contentMode: "apple")

    let record = VideoGenerationRecord.build(
      request: request,
      transformerFile: "/weights/transformer-distilled.safetensors",
      frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448,
      twoStageRequested: false,
      refineSkippedReason: nil,
      audioWritten: false)

    XCTAssertEqual(record.prompt, request.prompt)
    XCTAssertEqual(record.negativePrompt, request.negativePrompt)
    XCTAssertEqual(record.seed, request.seed)
    XCTAssertEqual(record.steps, request.steps)
    XCTAssertEqual(record.guidance, request.guidance)
    XCTAssertEqual(record.model, "transformer-distilled")
    XCTAssertEqual(record.width, request.width)
    XCTAssertEqual(record.height, request.height)
    XCTAssertEqual(record.frames, 97)
    XCTAssertEqual(record.fps, request.fps)
    XCTAssertEqual(record.resolvedWidth, 704)
    XCTAssertEqual(record.resolvedHeight, 448)
    XCTAssertNil(record.dimensionReason, "not populated until #405/#408 lands — see VideoGenerationRecord.build's doc comment")
    XCTAssertFalse(record.twoPass)
    XCTAssertFalse(record.refine)
    XCTAssertNil(record.refineSkippedReason)
    XCTAssertFalse(record.audio)
    XCTAssertEqual(record.kind, "t2v")
    XCTAssertEqual(record.source, "bree")
    XCTAssertEqual(record.contentMode, "apple")
    XCTAssertEqual(record.loras, [.init(name: "motion_v2", scale: 0.8)])
  }

  func testBuildClassifiesI2VAndExtend() {
    let i2v = LTX2VideoRequest(
      prompt: "p", initImagePath: "/tmp/src.png", width: 704, height: 448,
      framesPerChunk: 97, steps: 8, extendToSeconds: 0, outputPath: "/tmp/o.mp4")
    let i2vRecord = VideoGenerationRecord.build(
      request: i2v, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: false)
    XCTAssertEqual(i2vRecord.kind, "i2v")

    let extend = LTX2VideoRequest(
      prompt: "p", initImagePath: "/tmp/src.png", width: 704, height: 448,
      framesPerChunk: 97, steps: 8, extendToSeconds: 8, outputPath: "/tmp/o.mp4")
    let extendRecord = VideoGenerationRecord.build(
      request: extend, transformerFile: "t.safetensors", frameCount: 193,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: false)
    XCTAssertEqual(extendRecord.kind, "extend")
    XCTAssertEqual(extendRecord.frames, 193)
  }

  func testRefineIsFalseWhenRequestedButSkipped() {
    let request = LTX2VideoRequest(prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8, outputPath: "/tmp/o.mp4")
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: true,
      refineSkippedReason: "upsampler_unavailable", audioWritten: false)
    XCTAssertTrue(record.twoPass, "two_stage WAS requested")
    XCTAssertFalse(record.refine, "…but it did not run")
    XCTAssertEqual(record.refineSkippedReason, "upsampler_unavailable")
  }

  func testRefineIsTrueWhenRequestedAndRan() {
    let request = LTX2VideoRequest(prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8, outputPath: "/tmp/o.mp4")
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 1408, resolvedHeight: 896, twoStageRequested: true,
      refineSkippedReason: nil, audioWritten: false)
    XCTAssertTrue(record.twoPass)
    XCTAssertTrue(record.refine)
    XCTAssertEqual(record.resolvedWidth, 1408, "2x-refined size, not the request budget")
  }

  func testMultipleLoRAsAndDeprecatedSingleLoRAFieldBothMap() {
    let request = LTX2VideoRequest(
      prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8,
      loraPath: "/loras/old_single.safetensors", loraStrength: 1.0,
      loras: [LTX2LoRAReference(path: "/loras/new_a.safetensors", scale: 0.5)],
      outputPath: "/tmp/o.mp4")
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: false)
    XCTAssertEqual(record.loras, [
      .init(name: "old_single", scale: 1.0),
      .init(name: "new_a", scale: 0.5),
    ], "effectiveLoRAs prepends the deprecated single field, same order the pipeline applies them in")
  }

  // MARK: - JSON round trip (ruling 4: "encode/decode round-trip tests")

  func testJSONRoundTrip() throws {
    let record = VideoGenerationRecord(
      prompt: "a fox", negativePrompt: "blurry", seed: 42, steps: 8,
      model: "transformer-distilled", width: 704, height: 448, frames: 97, fps: 24,
      resolvedWidth: 704, resolvedHeight: 448, dimensionReason: "source_aspect",
      twoPass: true, refine: true, refineSkippedReason: nil, audio: true,
      kind: "i2v", loras: [.init(name: "motion_v2", scale: 0.8)])

    let data = try record.encodeJSON()
    let decoded = try VideoGenerationRecord.decodeJSON(data)
    XCTAssertEqual(decoded, record)
  }

  func testJSONRoundTripWithNilOptionalFields() throws {
    // The storyboard-assembly shape: no single seed/steps apply.
    let record = VideoGenerationRecord(
      prompt: "assembly", model: "ltx2-storyboard", width: 640, height: 640,
      frames: 240, fps: 24, resolvedWidth: 640, resolvedHeight: 640,
      twoPass: false, refine: false, audio: false, kind: "storyboard")
    let decoded = try VideoGenerationRecord.decodeJSON(try record.encodeJSON())
    XCTAssertEqual(decoded, record)
    XCTAssertNil(decoded.seed)
    XCTAssertNil(decoded.steps)
  }

  func testWireKeysAreSnakeCase() throws {
    let record = VideoGenerationRecord(
      prompt: "p", model: "m", width: 1, height: 1, frames: 1, fps: 1,
      resolvedWidth: 1, resolvedHeight: 1, twoPass: false, refine: false, audio: false, kind: "t2v")
    let json = try JSONSerialization.jsonObject(with: record.encodeJSON()) as? [String: Any]
    XCTAssertNotNil(json?["resolved_width"], "camelCase properties must encode snake_case, same convention as RenderRecipe/ImageMetadata")
    XCTAssertNotNil(json?["two_pass"])
    XCTAssertNil(json?["resolvedWidth"], "must not ALSO carry the camelCase spelling")
  }

  /// The PNG side's schema (ruling 1): prompt, seed, loras, steps, model must
  /// all be present keys on the wire for a fully-populated record.
  func testWireContainsThePNGSchemaKeys() throws {
    let record = VideoGenerationRecord(
      prompt: "p", seed: 7, steps: 8, model: "m", width: 1, height: 1, frames: 1, fps: 1,
      resolvedWidth: 1, resolvedHeight: 1, twoPass: false, refine: false, audio: false,
      kind: "t2v", loras: [.init(name: "l", scale: 1.0)])
    let json = try JSONSerialization.jsonObject(with: record.encodeJSON()) as? [String: Any]
    for key in ["prompt", "seed", "loras", "steps", "model", "frames", "fps", "audio"] {
      XCTAssertNotNil(json?[key], "missing PNG-parity key: \(key)")
    }
  }

  // MARK: - Sidecar (ruling 2: "next to the mp4, same convention as the editor's")

  func testSidecarPathMatchesTheEditorConvention() {
    // Mirrors EditSidecar.sidecarPath(forImageAt:) exactly: strip the
    // extension, append .json, same directory.
    XCTAssertEqual(VideoSidecar.path(forMediaAt: "/gallery/kira/clip.mp4"), "/gallery/kira/clip.json")
    XCTAssertEqual(VideoSidecar.path(forMediaAt: "/a/b/c.mov"), "/a/b/c.json")
  }

  func testSidecarWriteThenReadRoundTrips() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let mediaPath = dir.appendingPathComponent("clip.mp4").path

    let record = VideoGenerationRecord(
      prompt: "a fox", seed: 42, steps: 8, model: "transformer-distilled",
      width: 704, height: 448, frames: 97, fps: 24, resolvedWidth: 704, resolvedHeight: 448,
      twoPass: false, refine: false, audio: false, kind: "i2v",
      loras: [.init(name: "motion_v2", scale: 0.8)])

    XCTAssertTrue(VideoSidecar.write(record, forMediaAt: mediaPath))
    XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("clip.json").path))
    XCTAssertEqual(VideoSidecar.read(forMediaAt: mediaPath), record)
  }

  func testSidecarReadReturnsNilWhenMissing() {
    XCTAssertNil(VideoSidecar.read(forMediaAt: "/does/not/exist/clip.mp4"))
  }

  /// The DAM ingestor already reads a `.json` sidecar next to any media file
  /// with these exact keys (`AssetIngestor.readSidecar`/`embeddedLoras`,
  /// `Sources/ComfyBoxDesktop/DAM/AssetIngestor.swift`). Pin the shapes it
  /// depends on so this record stays compatible with that reader without
  /// either side needing to change.
  func testSidecarShapeIsCompatibleWithTheDAMIngestorReader() throws {
    let record = VideoGenerationRecord(
      prompt: "a fox", seed: 42, steps: 8, model: "transformer-distilled",
      width: 704, height: 448, frames: 97, fps: 24, resolvedWidth: 704, resolvedHeight: 448,
      twoPass: false, refine: false, audio: false, kind: "i2v",
      loras: [.init(name: "motion_v2", scale: 0.8)])
    let json = try JSONSerialization.jsonObject(with: record.encodeJSON()) as! [String: Any]

    XCTAssertEqual(json["prompt"] as? String, "a fox")
    XCTAssertEqual(json["seed"] as? Int, 42)
    XCTAssertEqual(json["steps"] as? Int, 8)
    XCTAssertEqual(json["model"] as? String, "transformer-distilled")
    let loras = json["loras"] as? [[String: Any]]
    XCTAssertEqual(loras?.first?["name"] as? String, "motion_v2")
    XCTAssertEqual(loras?.first?["scale"] as? Double ?? -1, 0.8, accuracy: 0.0001)
  }

  /// Review round 2, ruling 4: a seed above Int32.max must not silently drop
  /// when it goes through the same `as? Int` read `AssetIngestor.readSidecar`
  /// uses (`json["seed"] as? Int`, line ~647). `Int` on this platform is
  /// 64-bit, same range as the `UInt64` this record stores a seed as, so a
  /// seed of this magnitude survives — this pins that fact so a future
  /// change (e.g. a narrower `Int32`-backed field somewhere in the chain)
  /// fails loudly instead of quietly truncating a real render's seed.
  func testSeedAboveInt32MaxSurvivesTheJSONSerializationIntCastReadSidecarUses() throws {
    let bigSeed: UInt64 = 5_000_000_123  // > Int32.max (2_147_483_647)
    let record = VideoGenerationRecord(
      prompt: "p", seed: bigSeed, steps: 8, model: "m", width: 1, height: 1,
      frames: 1, fps: 1, resolvedWidth: 1, resolvedHeight: 1,
      twoPass: false, refine: false, audio: false, kind: "t2v")
    let json = try JSONSerialization.jsonObject(with: record.encodeJSON()) as! [String: Any]
    XCTAssertEqual(json["seed"] as? Int, Int(bigSeed), "readSidecar's exact cast must not return nil for a real production-magnitude seed")
  }

  // MARK: - Atom size cap (ruling 5)

  func testAtomJSONReturnsTheFullRecordWhenUnderTheCap() throws {
    let record = VideoGenerationRecord(
      prompt: "a fox", seed: 42, steps: 8, model: "m", width: 1, height: 1,
      frames: 1, fps: 1, resolvedWidth: 1, resolvedHeight: 1,
      twoPass: false, refine: false, audio: false, kind: "t2v")
    let atom = try XCTUnwrap(record.atomJSON())
    XCTAssertLessThan(atom.count, VideoGenerationRecord.atomSizeCap)
    let decoded = try VideoGenerationRecord.decodeJSON(atom)
    XCTAssertEqual(decoded, record)
    XCTAssertFalse(decoded.truncated)
  }

  func testAtomJSONTruncatesAnOversizedRecordButTheSidecarStaysFull() throws {
    // A single absurdly long field is enough to blow the 64KB cap.
    let hugePrompt = String(repeating: "a very long scene description. ", count: 4000)
    let record = VideoGenerationRecord(
      prompt: hugePrompt, seed: 42, steps: 8, model: "m", width: 1, height: 1,
      frames: 1, fps: 1, resolvedWidth: 1, resolvedHeight: 1,
      twoPass: false, refine: false, audio: false, kind: "t2v")
    XCTAssertGreaterThan(try record.encodeJSON().count, VideoGenerationRecord.atomSizeCap,
                          "the fixture must actually exceed the cap or this test proves nothing")

    let atom = try XCTUnwrap(record.atomJSON())
    XCTAssertLessThanOrEqual(atom.count, VideoGenerationRecord.atomSizeCap)
    let decodedAtom = try VideoGenerationRecord.decodeJSON(atom)
    XCTAssertTrue(decodedAtom.truncated)
    XCTAssertLessThanOrEqual(decodedAtom.prompt.count, 200)
    XCTAssertEqual(decodedAtom.seed, 42, "cheap identifying fields survive truncation")
    XCTAssertEqual(decodedAtom.model, "m")

    // The sidecar path is untouched by truncation — it always writes the
    // FULL record via `encodeJSON()`, never `atomJSON()`.
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let mediaPath = dir.appendingPathComponent("clip.mp4").path
    XCTAssertTrue(VideoSidecar.write(record, forMediaAt: mediaPath))
    let fromSidecar = try XCTUnwrap(VideoSidecar.read(forMediaAt: mediaPath))
    XCTAssertEqual(fromSidecar.prompt, hugePrompt, "sidecar keeps the untruncated prompt")
    XCTAssertFalse(fromSidecar.truncated)
  }

  func testAtomJSONStringMatchesAtomJSON() throws {
    let record = VideoGenerationRecord(
      prompt: "p", model: "m", width: 1, height: 1, frames: 1, fps: 1,
      resolvedWidth: 1, resolvedHeight: 1, twoPass: false, refine: false, audio: false, kind: "t2v")
    let expected = String(data: try XCTUnwrap(record.atomJSON()), encoding: .utf8)
    XCTAssertEqual(record.atomJSONString, expected)
  }
}
