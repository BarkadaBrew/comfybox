import XCTest
@testable import ZImage

/// WP2c (docs/FDD-ltx-director-tab.md): the pure seams behind
/// POST /v1/video/director and /v1/video/director/validate — envelope decode,
/// the validate response body, the submit-time dry-run gate, carry-over chunk
/// bodies, route parsing / 405 allow-list, and the additive status fields.
/// No engine, no weights, no coordinator.
final class DirectorRouteTests: XCTestCase {

  private static let repoRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()  // Server
    .deletingLastPathComponent()  // ZImageTests
    .deletingLastPathComponent()  // Tests
    .deletingLastPathComponent()  // root

  private let minimalTimelineJSON = #"""
    {"version":1,"settings":{"width":576,"height":896,"length_frames":289},
     "global_prompt":"a woman walks through a market",
     "keyframes":[{"id":"k1","image_path":"/img/k1.png","frame":0}]}
    """#

  private func routeDecoder() -> JSONDecoder {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return d
  }

  private func payload(_ timelineJSON: String, extra: String = "") throws -> WarmServer.DirectorPayload {
    let json = "{\"timeline\":\(timelineJSON)\(extra)}"
    return try routeDecoder().decode(WarmServer.DirectorPayload.self, from: Data(json.utf8))
  }

  // MARK: - Envelope

  func testDirectorPayloadDecodesEnvelope() throws {
    let p = try payload(minimalTimelineJSON, extra: #","output_path":"final.mp4","source":"desktop""#)
    XCTAssertEqual(p.outputPath, "final.mp4")
    XCTAssertEqual(p.source, "desktop")
    XCTAssertEqual(p.timeline.settings.fps, 24)
    XCTAssertEqual(p.timeline.settings.seed, 42)
    XCTAssertEqual(p.timeline.settings.lengthFrames, 289)
    XCTAssertEqual(p.timeline.keyframes.first?.strength, 1.0)
    XCTAssertEqual(p.timeline.audio.mode, .generated)
    XCTAssertFalse(p.timeline.retake.enabled)

    let bare = try payload(minimalTimelineJSON)
    XCTAssertNil(bare.outputPath)
    XCTAssertNil(bare.source)
  }

  func testFutureVersionEnvelopeIsA400() throws {
    let json = #"{"timeline":{"version":2,"settings":{"width":576,"height":896,"length_frames":289},"global_prompt":"x"}}"#
    XCTAssertThrowsError(try routeDecoder().decode(WarmServer.DirectorPayload.self, from: Data(json.utf8))) { error in
      XCTAssertEqual(error as? DirectorError, .unsupportedVersion(2))
      XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
    }
  }

  // MARK: - /validate body

  func testValidateResponseShape() throws {
    let ok = try WarmServer.directorValidateBody(
      payload(minimalTimelineJSON), fileExists: { _ in true }, audioProbe: { _ in nil })
    let okObj = try XCTUnwrap(JSONSerialization.jsonObject(with: ok) as? [String: Any])
    XCTAssertEqual(okObj["ok"] as? Bool, true)
    XCTAssertEqual(okObj["snapped_length_frames"] as? Int, 289)
    let plan = try XCTUnwrap(okObj["plan"] as? [String: Any])
    XCTAssertEqual(plan["length_frames"] as? Int, 289)
    XCTAssertNotNil(plan["boundary_frames"])
    XCTAssertNotNil(plan["keyframe_ticks"])
    XCTAssertEqual((plan["chunks"] as? [Any])?.count, 1)
    XCTAssertNotNil(okObj["issues"] as? [Any])
    let text = String(decoding: ok, as: UTF8.self)
    XCTAssertFalse(text.contains("snappedLengthFrames"))
    XCTAssertFalse(text.contains("lengthFrames"))

    // Invalid: off-grid keyframe + 96f -> snapped 97 still fine; empty prompt errors.
    let bad = #"{"settings":{"width":576,"height":896,"length_frames":290},"global_prompt":"  ","keyframes":[{"id":"k1","image_path":"/img/k1.png","frame":100}]}"#
    let body = try WarmServer.directorValidateBody(
      payload(bad), fileExists: { _ in true }, audioProbe: { _ in nil })
    let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
    XCTAssertEqual(obj["ok"] as? Bool, false)
    XCTAssertEqual(obj["snapped_length_frames"] as? Int, 297)
    XCTAssertTrue(obj.keys.contains("plan"), "plan key must be present as null when invalid")
    XCTAssertTrue(obj["plan"] is NSNull)
    let issues = try XCTUnwrap(obj["issues"] as? [[String: Any]])
    let codes = Set(issues.compactMap { $0["code"] as? String })
    XCTAssertTrue(codes.contains("missing_global_prompt"))
    XCTAssertTrue(codes.contains("keyframe_off_grid"))
    XCTAssertTrue(issues.allSatisfy { $0["severity"] is String && $0["ids"] is [Any] })
  }

  func testInvalidBodyEnvelopeCarriesIssues() throws {
    let data = try WarmServer.directorInvalidBody([.error("keyframe_off_grid", "off grid", ids: ["k1"])])
    let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    XCTAssertEqual(obj["error"] as? String, "timeline invalid")
    let issues = try XCTUnwrap(obj["issues"] as? [[String: Any]])
    XCTAssertEqual(issues.first?["code"] as? String, "keyframe_off_grid")
    XCTAssertEqual(issues.first?["ids"] as? [String], ["k1"])
  }

  // MARK: - Dry run

  private func request(
    frames: Int = 289, extend: Float = 0, fps: Int = 24, temporalUpscale: Int? = nil
  ) -> LTX2VideoRequest {
    var tuning: LTX2VideoTuning? = nil
    if let temporalUpscale {
      var t = LTX2VideoTuning()
      t.temporalUpscale = temporalUpscale
      tuning = t
    }
    let snapshot = LTX2ConfigResolver.resolveTyped(request: tuning, preset: nil, environment: [:])
    return LTX2VideoRequest(
      prompt: "x", width: 576, height: 896, framesPerChunk: frames,
      extendToSeconds: extend, fps: fps, outputPath: "/tmp/director-test.mp4",
      resolvedConfigSnapshot: snapshot)
  }

  private func prep(_ request: LTX2VideoRequest, beatIgnored: String? = nil) -> WarmServer.PreparedLocalVideo {
    WarmServer.PreparedLocalVideo(
      generator: LTX2VideoGenerator(config: .init(weightsDir: "/nonexistent", gemmaPath: "/nonexistent")),
      request: request, mode: .i2v, source: "test", optimizationAttemptId: nil,
      enhancementSkippedReason: nil, beatScheduleIgnoredReason: beatIgnored, recipeHash: "h",
      resolvedDimensions: ResolvedVideoDimensions(
        width: 576, height: 896, reason: .explicit, budgetWidth: 576, budgetHeight: 896,
        sourceWidth: nil, sourceHeight: nil, stage1Width: nil, stage1Height: nil, ceilingPreClamp: nil))
  }

  func testDryRunRejectsTemporalUpscaleAndMultiChunk() throws {
    XCTAssertNil(WarmServer.directorDryRunError(prep: prep(request())))

    let upscale = try XCTUnwrap(WarmServer.directorDryRunError(prep: prep(request(temporalUpscale: 2))))
    XCTAssertTrue(upscale.hasPrefix("temporal_upscale_unsupported"), upscale)

    let multi = try XCTUnwrap(WarmServer.directorDryRunError(prep: prep(request(frames: 289, extend: 30))))
    XCTAssertTrue(multi.hasPrefix("chunk_not_single_pass"), multi)

    let beats = try XCTUnwrap(WarmServer.directorDryRunError(
      prep: prep(request(), beatIgnored: "multi_chunk_unsupported")))
    XCTAssertTrue(beats.hasPrefix("chunk_not_single_pass"), beats)

    // A compiled chunk body pins extend_to_seconds: 0, so a `duration` a preset
    // or caller might add can never turn the chunk into a continuation render
    // (prepareLocalVideo reads `extend_to_seconds ?? duration`).
    let t = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 577),
      globalPrompt: "x", keyframes: [.init(id: "k1", imagePath: "/img/k1.png", frame: 0)])
    let v = DirectorValidator.validate(t, fileExists: { _ in true }, audioProbe: { _ in nil })
    let c = try DirectorCompiler.compile(v, session: "s1", source: "test")
    for chunk in c.chunks {
      var body = chunk.body
      body["duration"] = 30.0
      let data = try JSONSerialization.data(withJSONObject: body)
      let req = try routeDecoder().decode(WarmServer.LocalVideoRequest.self, from: data)
      let frames = chunk.span.frames
      let extend = req.extendToSeconds
        ?? WarmServer.extendSecondsFromDuration(req.duration, framesPerChunk: frames, fps: req.fps ?? 24)
      XCTAssertEqual(extend, 0)
      let built = WarmServer.buildLocalVideoRequest(
        req: req, videoPreset: nil, effectivePrompt: req.prompt, effectiveInitImage: nil,
        renderWidth: 576, renderHeight: 896, foldedFramesPerChunk: frames, foldedExtendSeconds: extend,
        resolvedLoRAs: [], effectiveBeatSchedule: nil,
        resolvedConfigSnapshot: LTX2ConfigResolver.resolveTyped(request: nil, preset: nil, environment: [:]),
        resolvedOutput: "/tmp/\(chunk.outputName)")
      XCTAssertNil(WarmServer.directorDryRunError(request: built, beatScheduleIgnoredReason: nil))
    }
  }

  // MARK: - Carry-over body

  func testChunkBodyWithCarryOverPrepares() throws {
    let t = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 577),
      globalPrompt: "x",
      keyframes: [
        .init(id: "k1", imagePath: "/img/k1.png", frame: 0),
        .init(id: "k2", imagePath: "/img/k2.png", frame: 0, isEndFrame: true),
      ])
    let v = DirectorValidator.validate(t, fileExists: { _ in true }, audioProbe: { _ in nil })
    let c = try DirectorCompiler.compile(v, session: "s1", source: "test")
    XCTAssertEqual(c.chunks.count, 2)
    let chunk1 = c.chunks[1]
    XCTAssertEqual(chunk1.carryOverFromChunk, 0)
    let body = chunk1.bodyWithCarryOver(imagePath: "/out/director-s1-chunk0-lastframe.png")
    let data = try JSONSerialization.data(withJSONObject: body)
    let req = try routeDecoder().decode(WarmServer.LocalVideoRequest.self, from: data)
    let resolved = try WarmServer.resolveKeyframes(req)
    XCTAssertEqual(resolved.initImagePath, "/out/director-s1-chunk0-lastframe.png")
    XCTAssertEqual(resolved.strength, 1.0)
    XCTAssertEqual(resolved.extras.map(\.imagePath), ["/img/k2.png"])
    XCTAssertEqual(resolved.extras.map(\.frame), [288])
  }

  // MARK: - Assets

  func testMaterializeAssetsWritesFilesAndRewritesPaths() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("director-assets-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let bytes = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3])
    var t = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 289),
      globalPrompt: "x",
      keyframes: [
        .init(id: "k/1", imageBase64: bytes.base64EncodedString(), frame: 0),
        .init(id: "k2", imagePath: "/img/k2.png", frame: 96),
      ])
    let written = try WarmServer.directorMaterializeAssets(&t, session: "abc", directory: dir.path)
    XCTAssertEqual(written.count, 1)
    let path = try XCTUnwrap(t.keyframes[0].imagePath)
    XCTAssertEqual((path as NSString).lastPathComponent, "director-abc-asset-k_1.png")
    XCTAssertNil(t.keyframes[0].imageBase64)
    XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: path)), bytes)
    XCTAssertEqual(t.keyframes[1].imagePath, "/img/k2.png")

    // Intermediates cleanup removes the session prefix only.
    let keep = dir.appendingPathComponent("director-abc.mp4")
    try Data("final".utf8).write(to: keep)
    try Data("chunk".utf8).write(to: dir.appendingPathComponent("director-abc-chunk0.mp4"))
    WarmServer.directorRemoveIntermediates(session: "abc", directory: dir.path, extra: [])
    let left = try FileManager.default.contentsOfDirectory(atPath: dir.path)
    XCTAssertEqual(left, ["director-abc.mp4"])
  }

  func testMaterializeRejectsInvalidBase64() {
    var t = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 289),
      globalPrompt: "x", keyframes: [.init(id: "k1", imageBase64: "!!!not base64!!!", frame: 0)])
    XCTAssertThrowsError(try WarmServer.directorMaterializeAssets(&t, session: "s", directory: NSTemporaryDirectory()))
  }

  // MARK: - Routes

  func testRouteIsIn405AllowListAndParsed() throws {
    let url = ControlSurfaceParser.warmServerSource(repoRoot: Self.repoRoot)
    let parsed = try ControlSurfaceParser.parse(fileAt: url, surface: .v1)
    XCTAssertTrue(parsed.problems.isEmpty, "\(parsed.problems)")
    XCTAssertTrue(parsed.routes.contains(RouteRef(method: "POST", path: "/v1/video/director")))
    XCTAssertTrue(parsed.routes.contains(RouteRef(method: "POST", path: "/v1/video/director/validate")))

    let source = try String(contentsOf: url, encoding: .utf8)
    let defaultArm = try XCTUnwrap(source.range(of: "return .error(.error(status: 405, message: \"Method not allowed\"))"))
    let head = source[..<defaultArm.lowerBound]
    let allowListStart = try XCTUnwrap(head.range(of: "default:", options: .backwards))
    let allowList = head[allowListStart.lowerBound...]
    XCTAssertTrue(allowList.contains("\"/v1/video/director\""))
    XCTAssertTrue(allowList.contains("\"/v1/video/director/validate\""))
  }

  func testDirectorErrorsMapToHTTPStatuses() {
    XCTAssertEqual(WarmServer.errorResponse(for: DirectorError.invalid([])).status, 400)
    XCTAssertEqual(WarmServer.errorResponse(for: DirectorError.unsupportedVersion(3)).status, 400)
    XCTAssertEqual(WarmServer.errorResponse(for: DirectorError.chunkFailed(chunk: 0, stage: "render", message: "x")).status, 500)
  }

  // MARK: - Status

  func testStatusCarriesPlanAndStages() throws {
    let plan = DirectorPlan(
      lengthFrames: 577, fps: 24, width: 576, height: 896, audioMode: "generated",
      chunks: [
        .init(index: 0, startFrame: 0, endFrame: 288, frames: 289, seed: 42, carryOver: false, audio: "generated"),
        .init(index: 1, startFrame: 288, endFrame: 576, frames: 289, seed: 43, carryOver: true, audio: "generated"),
      ],
      keyframeTicks: [.init(id: "k1", frame: 0)], boundaryFrames: [288], warnings: [])
    let status = VideoJobStatus(
      jobId: "j", status: .processing, mode: .director, backend: "ltx2-local",
      plan: plan, stageIndex: 1, stageCount: 2)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(status)) as? [String: Any])
    XCTAssertEqual(obj["mode"] as? String, "director")
    XCTAssertEqual(obj["stage_index"] as? Int, 1)
    XCTAssertEqual(obj["stage_count"] as? Int, 2)
    let p = try XCTUnwrap(obj["plan"] as? [String: Any])
    XCTAssertEqual(p["boundary_frames"] as? [Int], [288])
    XCTAssertEqual((p["chunks"] as? [[String: Any]])?[1]["carry_over"] as? Bool, true)

    // Decodes back (desktop / MCP clients use convertFromSnakeCase).
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let round = try decoder.decode(VideoJobStatus.self, from: encoder.encode(status))
    XCTAssertEqual(round.plan, plan)
    XCTAssertEqual(round.stageIndex, 1)

    // Old shape: fields omitted when nil.
    let legacy = VideoJobStatus(jobId: "j", status: .queued, mode: .i2v, backend: "ltx2-local")
    let legacyObj = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(legacy)) as? [String: Any])
    XCTAssertNil(legacyObj["plan"])
    XCTAssertNil(legacyObj["stage_index"])
    XCTAssertNil(legacyObj["stage_count"])
  }
}
