import XCTest
@testable import ZImage

/// comfybox#401: the video generation record is the mp4 twin of the PNG side's
/// `ImageMetadata.generation` (EXIF `UserComment` JSON). All of this is pure —
/// no model weights — per intent.md's "agents run unit tests only".
final class VideoGenerationRecordTests: XCTestCase {

  func testResolvedRecipeHashIsDeterministicAndRenderSensitive() throws {
    var tuning = LTX2VideoTuning()
    tuning.colorAnchor = 0
    tuning.nagScale = 11
    let snapshot = LTX2ConfigResolver.resolveTyped(
      request: tuning, preset: nil,
      environment: ["LTX2_SAMPLER": "euler_ancestral_cfg_pp"], configFile: [:])
    var request = LTX2VideoRequest(
      prompt: "she steps closer, then turns", negativePrompt: "ghosting",
      initImagePath: "/tmp/source.png", width: 512, height: 320,
      framesPerChunk: 289, steps: 8, seed: 42, guidance: 1,
      loras: [.init(path: "/tmp/motion.safetensors", scale: 0.8)],
      outputPath: "/tmp/out.mp4",
      resolvedConfigSnapshot: snapshot,
      beatSchedule: [
        .init(text: "she steps closer", startFrac: 0, endFrac: 0.5),
        .init(text: "then turns", startFrac: 0.5, endFrac: 1),
      ])

    let accepted = try ResolvedVideoRecipe.build(
      request: request, transformerFile: "transformer-distilled.safetensors")
    XCTAssertEqual(try accepted.fingerprint().count, 64)
    XCTAssertEqual(
      try accepted.fingerprint(),
      try ResolvedVideoRecipe.build(
        request: request, transformerFile: "transformer-distilled.safetensors").fingerprint())

    request.prompt += ", smiling"
    let changed = try ResolvedVideoRecipe.build(
      request: request, transformerFile: "transformer-distilled.safetensors")
    XCTAssertNotEqual(try changed.fingerprint(), try accepted.fingerprint())
  }

  func testResolvedRecipeUsesTheAcceptedConfigSnapshot() throws {
    let acceptedConfig = LTX2ConfigResolver.resolveTyped(
      request: nil, preset: nil,
      environment: ["LTX2_COLOR_ANCHOR": "0", "LTX2_SAMPLER": "accepted"],
      configFile: [:])
    let laterConfig = LTX2ConfigResolver.resolveTyped(
      request: nil, preset: nil,
      environment: ["LTX2_COLOR_ANCHOR": "1", "LTX2_SAMPLER": "changed"],
      configFile: [:])
    let base = LTX2VideoRequest(
      prompt: "p", framesPerChunk: 97, outputPath: "/tmp/o.mp4",
      resolvedConfigSnapshot: acceptedConfig)
    let later = LTX2VideoRequest(
      prompt: "p", framesPerChunk: 97, outputPath: "/tmp/o.mp4",
      resolvedConfigSnapshot: laterConfig)

    let accepted = try ResolvedVideoRecipe.build(request: base, transformerFile: "t.safetensors")
    let changed = try ResolvedVideoRecipe.build(request: later, transformerFile: "t.safetensors")
    XCTAssertNotEqual(try accepted.fingerprint(), try changed.fingerprint(),
                      "config drift must produce a different recipe identity")
    XCTAssertEqual(accepted.parameters.first { $0.name == "color_anchor" }?.value, "0")
    XCTAssertEqual(accepted.parameters.first { $0.name == "sampler" }?.value, "accepted")
  }

  // MARK: - kind()

  func testKindClassifiesFromInitImageAndExtend() {
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: nil, extendToSeconds: 0), "t2v")
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: nil, extendToSeconds: 8), "t2v",
                   "extend_to_seconds is meaningless without an init image — still t2v")
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: "/tmp/src.png", extendToSeconds: 0), "i2v")
    XCTAssertEqual(VideoGenerationRecord.kind(initImagePath: "/tmp/src.png", extendToSeconds: 8), "extend")
    XCTAssertEqual(
      VideoGenerationRecord.kind(
        initImagePath: nil, extendToSeconds: 0, frameCount: 1, outputPath: "/tmp/image.png"),
      "t2i")
    XCTAssertEqual(
      VideoGenerationRecord.kind(
        initImagePath: nil, extendToSeconds: 0, frameCount: 1, outputPath: "/tmp/video.mp4"),
      "t2v")
  }

  // MARK: - build() matches the request field-for-field (ruling 4: "a test that

  // the record for a t2v matches the request fields")

  func testBuildMatchesT2VRequestFields() throws {
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
      audioWritten: false,
      // Deliberately different from request.guidance (3.5) — the request
      // override must win; see testBuildGuidanceRequestOverrideWinsOverConfig.
      configGuidance: 1.0)

    XCTAssertEqual(record.prompt, request.prompt)
    XCTAssertEqual(record.negativePrompt, request.negativePrompt)
    XCTAssertEqual(record.seed, request.seed)
    XCTAssertEqual(record.steps, request.steps)
    XCTAssertEqual(record.guidance, request.guidance)
    XCTAssertEqual(record.model, "transformer-distilled")
    XCTAssertEqual(record.engine, "ltx2")
    XCTAssertEqual(record.width, request.width)
    XCTAssertEqual(record.height, request.height)
    XCTAssertEqual(record.frames, 97)
    XCTAssertEqual(record.fps, request.fps)
    XCTAssertEqual(record.resolvedWidth, 704)
    XCTAssertEqual(record.resolvedHeight, 448)
    XCTAssertNil(
      record.dimensionReason,
      "a request that carries no reason still records none — the field is optional")
    XCTAssertFalse(record.twoPass)
    XCTAssertFalse(record.refine)
    XCTAssertNil(record.refineSkippedReason)
    XCTAssertFalse(record.audio)
    XCTAssertEqual(record.kind, "t2v")
    XCTAssertEqual(record.source, "bree")
    XCTAssertEqual(record.contentMode, "apple")
    XCTAssertEqual(record.loras, [.init(name: "motion_v2", scale: 0.8)])

    // Review round 3, minor 2: end-to-end through the real encoder too — an
    // ENGINE-produced record (via build(), not a hand-written JSON literal)
    // must actually carry guidance/source/content_mode on the wire, not
    // just as Swift struct fields the encoder happens to drop.
    let json = try JSONSerialization.jsonObject(with: record.encodeJSON()) as? [String: Any]
    XCTAssertEqual(json?["guidance"] as? Double ?? -1, 3.5, accuracy: 0.0001)
    XCTAssertEqual(json?["source"] as? String, "bree")
    XCTAssertEqual(json?["content_mode"] as? String, "apple")
    XCTAssertEqual(json?["engine"] as? String, "ltx2")
  }

  // MARK: - guidance is the value ACTUALLY USED, not just the request field
  // (review round 3, ruling 1)

  /// `pipeline.generateT2V`/`generateI2V` resolve CFG as
  /// `guidance ?? config.guidance` — `LTX2PipelineConfig.guidance` is never
  /// nil, so a request with no override must record THAT value, not `null`.
  func testBuildGuidanceFallsBackToConfigGuidanceWhenRequestHasNoOverride() {
    let request = LTX2VideoRequest(prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8, outputPath: "/tmp/o.mp4")
    XCTAssertNil(request.guidance, "fixture sanity: no override on the request")
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: false, configGuidance: 1.0)
    XCTAssertEqual(record.guidance, 1.0, "the distilled pipeline's actual CFG scale, not null")
  }

  func testBuildGuidanceRequestOverrideWinsOverConfig() {
    let request = LTX2VideoRequest(
      prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8, guidance: 4.5,
      outputPath: "/tmp/o.mp4")
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: false, configGuidance: 3.5)
    XCTAssertEqual(record.guidance, 4.5, "an explicit request override must win over the config default")
  }

  func testBuildRecordsTheRecipeThatActuallyExecutedSeparatelyFromTheRequest() throws {
    let request = LTX2VideoRequest(
      prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8,
      outputPath: "/tmp/o.mp4")
    let nag = LTX2NAGConfig(scale: 11, alpha: 0.25, tau: 2.5)
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: true, configGuidance: 1.0,
      actualSteps: 10, sampler: "euler_ancestral_cfg_pp",
      stage1Sigmas: [1, 0.5, 0], refineSigmas: nil,
      nagConfig: nag, nagApplied: true, audioRefine: false)

    XCTAssertEqual(record.requestedSteps, 8)
    XCTAssertEqual(record.steps, 10)
    XCTAssertEqual(record.sampler, "euler_ancestral_cfg_pp")
    XCTAssertEqual(record.stage1Sigmas, [1, 0.5, 0])
    XCTAssertNil(record.refineSigmas)
    XCTAssertEqual(record.nagScale, 11)
    XCTAssertEqual(record.nagAlpha, 0.25)
    XCTAssertEqual(record.nagTau, 2.5)
    XCTAssertEqual(record.nagApplied, true)
    XCTAssertEqual(record.audioRefine, false)

    let json = try JSONSerialization.jsonObject(with: record.encodeJSON()) as? [String: Any]
    XCTAssertEqual(json?["requested_steps"] as? Int, 8)
    XCTAssertEqual(json?["steps"] as? Int, 10)
    XCTAssertEqual(json?["sampler"] as? String, "euler_ancestral_cfg_pp")
    XCTAssertEqual(json?["nag_applied"] as? Bool, true)
    XCTAssertEqual(json?["audio_refine"] as? Bool, false)
  }

  // `WarmServer.buildLocalVideoRequest` is where "request override, else
  // preset guidance" actually happens — `build()` above only sees whatever
  // `request.guidance` already holds by the time it gets there. See
  // `LocalVideoRequestDecodeTests` (Tests/ZImageTests/LocalVideoRequestDecodeTests.swift)
  // for that half of ruling 1's fallback chain: a preset-carried guidance is
  // honored when the request has none, and an explicit request override
  // still wins over the preset.

  func testBuildClassifiesI2VAndExtend() {
    let i2v = LTX2VideoRequest(
      prompt: "p", initImagePath: "/tmp/src.png", width: 704, height: 448,
      framesPerChunk: 97, steps: 8, extendToSeconds: 0, outputPath: "/tmp/o.mp4")
    let i2vRecord = VideoGenerationRecord.build(
      request: i2v, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: false, configGuidance: 1.0)
    XCTAssertEqual(i2vRecord.kind, "i2v")

    let extend = LTX2VideoRequest(
      prompt: "p", initImagePath: "/tmp/src.png", width: 704, height: 448,
      framesPerChunk: 97, steps: 8, extendToSeconds: 8, outputPath: "/tmp/o.mp4")
    let extendRecord = VideoGenerationRecord.build(
      request: extend, transformerFile: "t.safetensors", frameCount: 193,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: false,
      refineSkippedReason: nil, audioWritten: false, configGuidance: 1.0)
    XCTAssertEqual(extendRecord.kind, "extend")
    XCTAssertEqual(extendRecord.frames, 193)
  }

  func testRefineIsFalseWhenRequestedButSkipped() {
    let request = LTX2VideoRequest(prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8, outputPath: "/tmp/o.mp4")
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 704, resolvedHeight: 448, twoStageRequested: true,
      refineSkippedReason: "upsampler_unavailable", audioWritten: false, configGuidance: 3.5)
    XCTAssertTrue(record.twoPass, "two_stage WAS requested")
    XCTAssertFalse(record.refine, "…but it did not run")
    XCTAssertEqual(record.refineSkippedReason, "upsampler_unavailable")
  }

  func testRefineIsTrueWhenRequestedAndRan() {
    let request = LTX2VideoRequest(prompt: "p", width: 704, height: 448, framesPerChunk: 97, steps: 8, outputPath: "/tmp/o.mp4")
    let record = VideoGenerationRecord.build(
      request: request, transformerFile: "t.safetensors", frameCount: 97,
      resolvedWidth: 1408, resolvedHeight: 896, twoStageRequested: true,
      refineSkippedReason: nil, audioWritten: false, configGuidance: 3.5)
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
      refineSkippedReason: nil, audioWritten: false, configGuidance: 1.0)
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

  func testSidecarWrittenBeforeExecutionRecipeFieldsStillDecodes() throws {
    let legacy = #"""
      {"prompt":"old clip","steps":8,"model":"ltx2","width":704,"height":448,
       "frames":97,"fps":24,"resolved_width":704,"resolved_height":448,
       "two_pass":false,"refine":false,"audio":false,"kind":"t2v",
       "truncated":false,"loras":[]}
      """#
    let decoded = try VideoGenerationRecord.decodeJSON(Data(legacy.utf8))
    XCTAssertEqual(decoded.steps, 8)
    XCTAssertNil(decoded.requestedSteps)
    XCTAssertNil(decoded.sampler)
    XCTAssertNil(decoded.nagApplied)
    XCTAssertNil(decoded.audioRefine)
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

// MARK: - comfybox#405: the dimension_reason wire-up
//
// comfybox#401 shipped `dimensionReason` on the record and on
// `LTX2VideoRequest`, always nil, with the wire-up explicitly left to #405/#408
// ("the field exists here, additive and currently always nil, so wiring it
// later is a one-line change"). #408 owns `VideoDimensionResolver`, so it owns
// the wire-up. These pin that the record stops writing null.

extension VideoGenerationRecordTests {

  private func recordForRequest(_ request: LTX2VideoRequest) -> VideoGenerationRecord {
    VideoGenerationRecord.build(
      request: request,
      transformerFile: "/weights/transformer-distilled.safetensors",
      frameCount: 49,
      resolvedWidth: request.width, resolvedHeight: request.height,
      twoStageRequested: false,
      refineSkippedReason: nil,
      audioWritten: false,
      configGuidance: 1.0)
  }

  /// The case the ticket is about: an i2v render whose shape came from the
  /// SOURCE IMAGE must say so in its sidecar, so a wrong-shaped clip is
  /// diagnosable from the file itself.
  func testI2VRecordCarriesSourceAspectAsTheDimensionReason() {
    let request = LTX2VideoRequest(
      prompt: "she turns toward the window",
      initImagePath: "/tmp/kira-portrait.png",
      width: 512, height: 896,
      outputPath: "/tmp/out.mp4",
      dimensionReason: VideoDimensionReason.sourceAspect.rawValue)
    let record = recordForRequest(request)
    XCTAssertEqual(record.kind, "i2v")
    XCTAssertEqual(record.dimensionReason, "source_aspect")
  }

  func testTheOtherTwoReasonsRoundTripToo() {
    let explicit = recordForRequest(LTX2VideoRequest(
      prompt: "p", width: 960, height: 576, outputPath: "/tmp/out.mp4",
      dimensionReason: VideoDimensionReason.explicit.rawValue))
    XCTAssertEqual(explicit.dimensionReason, "explicit")
    XCTAssertEqual(explicit.kind, "t2v")

    let fallback = recordForRequest(LTX2VideoRequest(
      prompt: "p", outputPath: "/tmp/out.mp4",
      dimensionReason: VideoDimensionReason.default.rawValue))
    XCTAssertEqual(fallback.dimensionReason, "default")
  }

  /// The wire-up is only useful if it survives to the sidecar JSON under the
  /// snake_case key the DAM ingest path reads.
  func testDimensionReasonSurvivesEncodingAsSnakeCase() throws {
    let record = recordForRequest(LTX2VideoRequest(
      prompt: "p",
      initImagePath: "/tmp/kira-portrait.png",
      width: 512, height: 896,
      outputPath: "/tmp/out.mp4",
      dimensionReason: VideoDimensionReason.sourceAspect.rawValue))
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let json = try JSONSerialization.jsonObject(
      with: try encoder.encode(record)) as? [String: Any]
    XCTAssertEqual(json?["dimension_reason"] as? String, "source_aspect")
  }

  /// The values the record can carry are exactly the resolver's, so the
  /// sidecar and the render trace cannot describe the same render differently.
  func testTheRecordsVocabularyIsTheResolversVocabulary() {
    XCTAssertEqual(VideoDimensionReason.sourceAspect.rawValue, "source_aspect")
    XCTAssertEqual(VideoDimensionReason.explicit.rawValue, "explicit")
    XCTAssertEqual(VideoDimensionReason.default.rawValue, "default")
  }
}
