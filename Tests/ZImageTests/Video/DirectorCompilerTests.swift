import XCTest
@testable import ZImage

/// WP2b: the pure compiler — validated timeline -> one single-pass
/// /v1/video/generate body per chunk + the full DirectorPlan.
final class DirectorCompilerTests: XCTestCase {

  private func timeline(
    length: Int = 289, fps: Int = 24, width: Int = 576, height: Int = 896,
    prompt: String = "a woman walks through a market",
    seed: UInt64 = 771144,
    keyframes: [DirectorTimeline.Keyframe] = [.init(id: "k1", imagePath: "/img/k1.png", frame: 0)],
    segments: [DirectorTimeline.PromptSegment] = [],
    mode: DirectorTimeline.AudioMode = .generated,
    preset: String? = nil,
    loras: [DirectorTimeline.LoRARef] = [],
    negative: String? = nil,
    steps: Int? = nil,
    character: String? = nil
  ) -> DirectorTimeline {
    DirectorTimeline(
      settings: .init(
        fps: fps, width: width, height: height, lengthFrames: length, seed: seed,
        preset: preset, loras: loras, negativePrompt: negative, steps: steps, character: character),
      globalPrompt: prompt, keyframes: keyframes, promptSegments: segments,
      audio: .init(mode: mode))
  }

  private func compile(
    _ t: DirectorTimeline, session: String = "s1", source: String = "test"
  ) throws -> DirectorCompilation {
    let v = DirectorValidator.validate(t, fileExists: { _ in true })
    XCTAssertTrue(v.ok, "\(v.issues)")
    return try DirectorCompiler.compile(v, session: session, source: source)
  }

  private func keyframeEntries(_ body: [String: Any]) -> [(path: String, frame: Int, strength: Float)] {
    (body["keyframes"] as? [[String: Any]] ?? []).map {
      ($0["image_path"] as? String ?? "", $0["frame"] as? Int ?? -1, ($0["strength"] as? NSNumber)?.floatValue ?? -1)
    }
  }

  private func serialized(_ body: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
  }

  private func decodeLocal(_ body: [String: Any]) throws -> WarmServer.LocalVideoRequest {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(WarmServer.LocalVideoRequest.self, from: serialized(body))
  }

  // MARK: - Keyframe placement

  func testSingleChunkFFLF() throws {
    let t = timeline(keyframes: [
      .init(id: "k1", imagePath: "/img/first.png", frame: 0),
      .init(id: "k2", imagePath: "/img/last.png", frame: 0, isEndFrame: true),
    ])
    let c = try compile(t)
    XCTAssertEqual(c.chunks.count, 1)
    let chunk = c.chunks[0]
    let body = chunk.body
    XCTAssertEqual(chunk.index, 0)
    XCTAssertNil(chunk.carryOverFromChunk)
    XCTAssertEqual(chunk.outputName, "director-s1-chunk0.mp4")
    XCTAssertEqual(chunk.lastFrameName, "director-s1-chunk0-lastframe.png")
    XCTAssertTrue(chunk.wantsAudio)

    let kfs = keyframeEntries(body)
    XCTAssertEqual(kfs.map(\.path), ["/img/first.png", "/img/last.png"])
    XCTAssertEqual(kfs.map(\.frame), [0, 288])
    XCTAssertEqual(kfs.map(\.strength), [1.0, 1.0])
    XCTAssertEqual(body["frames"] as? Int, 289)
    XCTAssertEqual((body["extend_to_seconds"] as? NSNumber)?.doubleValue, 0)
    XCTAssertEqual((body["identity_anchor_strength"] as? NSNumber)?.doubleValue, 0)
    XCTAssertNil(body["image_path"])
    XCTAssertNil(body["image_base64"])
    XCTAssertNil(body["strength"])
    XCTAssertNil(body["duration"])
    XCTAssertEqual(body["audio"] as? Bool, true)
    XCTAssertEqual(body["enhance"] as? Bool, false)
    XCTAssertEqual(body["skip_character_injection"] as? Bool, true)
    XCTAssertNil(body["character"])
    XCTAssertEqual((body["seed"] as? NSNumber)?.uint64Value, 771144)
    XCTAssertEqual(body["width"] as? Int, 576)
    XCTAssertEqual(body["height"] as? Int, 896)
    XCTAssertEqual(body["fps"] as? Int, 24)
    XCTAssertEqual(body["prompt"] as? String, "a woman walks through a market")
    XCTAssertEqual(body["output_path"] as? String, "director-s1-chunk0.mp4")
    XCTAssertEqual(body["source"] as? String, "test")
    XCTAssertNil(body["beat_schedule"])
    XCTAssertNil(body["preset"])
    XCTAssertNil(body["loras"])
    XCTAssertNil(body["steps"])
    XCTAssertNil(body["negative_prompt"])

    XCTAssertEqual(c.plan.chunks.count, 1)
    XCTAssertEqual(c.plan.chunks[0].keyframes, [
      .init(id: "k1", localFrame: 0, strength: 1.0),
      .init(id: "k2", localFrame: 288, strength: 1.0),
    ])
    XCTAssertEqual(c.plan.chunks[0].seed, 771144)
    XCTAssertEqual(c.plan.boundaryFrames, [])
    XCTAssertEqual(c.plan.keyframeTicks, [.init(id: "k1", frame: 0), .init(id: "k2", frame: 288)])
  }

  func testTwoChunkKeyframeOnBoundary() throws {
    let t = timeline(length: 577, keyframes: [
      .init(id: "k1", imagePath: "/img/a.png", frame: 0),
      .init(id: "k2", imagePath: "/img/b.png", frame: 288),
      .init(id: "k3", imagePath: "/img/c.png", frame: 576),
    ])
    let c = try compile(t)
    XCTAssertEqual(c.chunks.count, 2)

    let c0 = c.chunks[0], c1 = c.chunks[1]
    XCTAssertEqual(keyframeEntries(c0.body).map(\.frame), [0, 288])
    XCTAssertEqual(keyframeEntries(c0.body).map(\.path), ["/img/a.png", "/img/b.png"])
    XCTAssertNil(c0.carryOverFromChunk)

    // Boundary keyframe k2 conditions chunk 0's LAST frame only; chunk 1
    // starts from the rendered carry-over (no frame-0 entry in the body).
    XCTAssertEqual(keyframeEntries(c1.body).map(\.frame), [288])
    XCTAssertEqual(keyframeEntries(c1.body).map(\.path), ["/img/c.png"])
    XCTAssertEqual(c1.carryOverFromChunk, 0)
    XCTAssertEqual((c1.body["seed"] as? NSNumber)?.uint64Value, 771145)
    XCTAssertEqual(c1.body["frames"] as? Int, 289)
    XCTAssertEqual(c1.outputName, "director-s1-chunk1.mp4")
    XCTAssertEqual(c1.lastFrameName, "director-s1-chunk1-lastframe.png")

    XCTAssertEqual(c.plan.boundaryFrames, [288])
    XCTAssertEqual(c.plan.chunks.map(\.carryOver), [false, true])
    XCTAssertEqual(c.plan.chunks[0].keyframes, [
      .init(id: "k1", localFrame: 0, strength: 1.0),
      .init(id: "k2", localFrame: 288, strength: 1.0),
    ])
    XCTAssertEqual(c.plan.chunks[1].keyframes, [.init(id: "k3", localFrame: 288, strength: 1.0)])
    XCTAssertTrue(c.plan.warnings.contains { $0.code == "keyframe_on_chunk_boundary" && $0.ids == ["k2"] })
  }

  func testBodyWithCarryOverPrependsFrameZero() throws {
    let t = timeline(length: 577, keyframes: [
      .init(id: "k1", imagePath: "/img/a.png", frame: 0),
      .init(id: "k3", imagePath: "/img/c.png", frame: 576, strength: 0.75),
    ])
    let c = try compile(t)
    let c1 = c.chunks[1]
    XCTAssertEqual(keyframeEntries(c1.body).map(\.frame), [288])

    let carried = c1.bodyWithCarryOver(imagePath: "/x.png")
    let kfs = keyframeEntries(carried)
    XCTAssertEqual(kfs.map(\.path), ["/x.png", "/img/c.png"])
    XCTAssertEqual(kfs.map(\.frame), [0, 288])
    XCTAssertEqual(kfs.map(\.strength), [1.0, 0.75])
    // The original body is untouched (pure).
    XCTAssertEqual(keyframeEntries(c1.body).map(\.frame), [288])
    // Every other key is preserved verbatim.
    var withoutKeyframes = carried
    withoutKeyframes["keyframes"] = nil
    var originalWithoutKeyframes = c1.body
    originalWithoutKeyframes["keyframes"] = nil
    XCTAssertEqual(try serialized(withoutKeyframes), try serialized(originalWithoutKeyframes))

    // A chunk with no keyframes at all gets the array created.
    let bare = try compile(timeline(length: 577, keyframes: [.init(id: "k1", imagePath: "/img/a.png", frame: 0)]))
    XCTAssertNil(bare.chunks[1].body["keyframes"])
    let bareCarried = bare.chunks[1].bodyWithCarryOver(imagePath: "/y.png")
    XCTAssertEqual(keyframeEntries(bareCarried).map(\.path), ["/y.png"])
    XCTAssertEqual(keyframeEntries(bareCarried).map(\.frame), [0])
    XCTAssertEqual(keyframeEntries(bareCarried).map(\.strength), [1.0])
    XCTAssertNil(bareCarried["image_path"])
    XCTAssertNil(bareCarried["strength"])
  }

  // MARK: - Prompts and beats

  func testPromptComposedAndBeatsSplitAtBoundary() throws {
    let global = "a woman walks through a market"
    let t = timeline(length: 577, prompt: global, segments: [
      .init(id: "p1", startFrame: 0, lengthFrames: 200, prompt: "she inspects the fruit"),
      .init(id: "p2", startFrame: 200, lengthFrames: 200, prompt: "she turns toward the camera"),
    ])
    let c = try compile(t)
    let b0 = c.chunks[0].body, b1 = c.chunks[1].body

    XCTAssertEqual(b0["prompt"] as? String, global + "\nshe inspects the fruit\nshe turns toward the camera")
    XCTAssertEqual(b1["prompt"] as? String, global + "\nshe turns toward the camera")

    let beats0 = c.plan.chunks[0].beatSchedule
    XCTAssertEqual(beats0.map(\.text), ["she inspects the fruit", "she turns toward the camera"])
    XCTAssertEqual(beats0[0].startFrac, 0, accuracy: 1e-6)
    XCTAssertEqual(beats0[0].endFrac, 200.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(beats0[1].startFrac, 200.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(beats0[1].endFrac, 1.0, accuracy: 1e-6)

    let beats1 = c.plan.chunks[1].beatSchedule
    XCTAssertEqual(beats1.map(\.text), ["she turns toward the camera"])
    XCTAssertEqual(beats1[0].startFrac, 0, accuracy: 1e-6)
    XCTAssertEqual(beats1[0].endFrac, 112.0 / 289.0, accuracy: 1e-6)

    XCTAssertEqual(c.plan.chunks[0].promptSegments, ["p1", "p2"])
    XCTAssertEqual(c.plan.chunks[1].promptSegments, ["p2"])

    // The body's beat_schedule decodes to the same values.
    let r0 = try decodeLocal(b0)
    let wire0 = try XCTUnwrap(r0.beatSchedule)
    XCTAssertEqual(wire0.map(\.text), beats0.map(\.text))
    XCTAssertEqual(wire0[0].endFrac, 200.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(wire0[1].startFrac, 200.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(wire0[1].endFrac, 1.0, accuracy: 1e-6)
    let r1 = try decodeLocal(b1)
    let wire1 = try XCTUnwrap(r1.beatSchedule)
    XCTAssertEqual(wire1.count, 1)
    XCTAssertEqual(wire1[0].startFrac, 0, accuracy: 1e-6)
    XCTAssertEqual(wire1[0].endFrac, 112.0 / 289.0, accuracy: 1e-6)
    XCTAssertNil(wire1[0].strength)
  }

  func testSegmentStartingOnBoundaryDoesNotLeakIntoPreviousChunk() throws {
    let global = "a woman walks through a market"
    let t = timeline(length: 577, prompt: global, segments: [
      .init(id: "p1", startFrame: 0, lengthFrames: 288, prompt: "she inspects the fruit"),
      .init(id: "p2", startFrame: 288, lengthFrames: 200, prompt: "she turns toward the camera"),
    ])
    let c = try compile(t)
    let b0 = c.chunks[0].body, b1 = c.chunks[1].body
    XCTAssertEqual(c.plan.chunks[0].promptSegments, ["p1"])
    XCTAssertEqual(c.plan.chunks[0].beatSchedule.map(\.text), ["she inspects the fruit"])
    XCTAssertEqual(b0["prompt"] as? String, global + "\nshe inspects the fruit")
    XCTAssertEqual((b0["beat_schedule"] as? [[String: Any]])?.count, 1)
    // p1 ends at 288 = start_1 exactly: it covers chunk 0 only, and p1's
    // [0, 288) does not reach chunk 1 at all.
    XCTAssertEqual(c.plan.chunks[1].promptSegments, ["p2"])
    XCTAssertEqual(b1["prompt"] as? String, global + "\nshe turns toward the camera")

    // Mirror: a segment ending at start_1 + 1 puts no sliver into chunk 1.
    let mirror = try compile(timeline(length: 577, segments: [
      .init(id: "p1", startFrame: 96, lengthFrames: 193, prompt: "she inspects the fruit"),
    ]))
    XCTAssertEqual(mirror.plan.chunks[0].promptSegments, ["p1"])
    XCTAssertEqual(mirror.plan.chunks[1].promptSegments, [])
    XCTAssertNil(mirror.chunks[1].body["beat_schedule"])

    // The timeline's LAST frame is nobody's boundary: a segment there stays.
    let tail = DirectorMath.beatFractions(
      segmentStart: 576, segmentLength: 1, chunk: DirectorMath.chunkLayout(lengthFrames: 577)[1],
      sharesEndFrame: false)
    XCTAssertNotNil(tail)
  }

  func testPresetOnlyTimelineSendsNoSeedOrFps() throws {
    var t = timeline(length: 577, preset: "kira-video")
    t.settings.seed = nil
    t.settings.fps = nil
    let c = try compile(t)
    for chunk in c.chunks {
      XCTAssertNil(chunk.body["seed"], "chunk \(chunk.index): the preset seed must apply")
      XCTAssertNil(chunk.body["fps"], "chunk \(chunk.index): the preset/config fps must apply")
      XCTAssertEqual(chunk.body["preset"] as? String, "kira-video")
    }
    XCTAssertEqual(c.plan.chunks.map(\.seed), [nil, nil])
    XCTAssertEqual(c.plan.fps, DirectorTimeline.Settings.defaultFps)

    // Once the server resolved chunk 0 (preset seed 9000, preset fps 30),
    // the recompiled bodies carry seed + k and that fps everywhere.
    let resolved = DirectorCompiler.resolvingDefaults(
      DirectorValidator.validate(t, fileExists: { _ in true }).snapped, fps: 30, seed: 9000)
    let rc = try compile(resolved)
    XCTAssertEqual(rc.chunks.map { ($0.body["seed"] as? NSNumber)?.uint64Value }, [9000, 9001])
    XCTAssertEqual(rc.chunks.map { $0.body["fps"] as? Int }, [30, 30])
    XCTAssertEqual(rc.plan.chunks.map(\.seed), [9000, 9001])
    XCTAssertEqual(rc.plan.fps, 30)

    // Explicit timeline values are never overridden by the resolved ones.
    let explicit = DirectorCompiler.resolvingDefaults(timeline(fps: 25, seed: 7), fps: 30, seed: 9000)
    XCTAssertEqual(explicit.settings.fps, 25)
    XCTAssertEqual(explicit.settings.seed, 7)
  }

  func testNoSegmentsOmitsBeatScheduleAndUsesGlobalPromptOnly() throws {
    let c = try compile(timeline(length: 577, prompt: "  a still lake at dawn"))
    for chunk in c.chunks {
      XCTAssertEqual(chunk.body["prompt"] as? String, "  a still lake at dawn")
      XCTAssertNil(chunk.body["beat_schedule"])
    }
    XCTAssertTrue(c.plan.chunks.allSatisfy { $0.beatSchedule.isEmpty && $0.promptSegments.isEmpty })
  }

  func testDuplicateSegmentTextsKeepTimeOrder() throws {
    let t = timeline(segments: [
      .init(id: "p2", startFrame: 96, lengthFrames: 96, prompt: "she smiles"),
      .init(id: "p1", startFrame: 0, lengthFrames: 96, prompt: "she smiles"),
    ])
    let c = try compile(t)
    let beats = c.plan.chunks[0].beatSchedule
    XCTAssertEqual(beats.map(\.text), ["she smiles", "she smiles"])
    XCTAssertEqual(beats[0].startFrac, 0, accuracy: 1e-6)
    XCTAssertEqual(beats[0].endFrac, 96.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(beats[1].startFrac, 96.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(beats[1].endFrac, 192.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(c.plan.chunks[0].promptSegments, ["p1", "p2"])
    XCTAssertEqual(c.chunks[0].body["prompt"] as? String, "a woman walks through a market\nshe smiles\nshe smiles")
  }

  // MARK: - Audio

  func testImportedModeSendsAudioFalseEverywhere() throws {
    let c = try compile(timeline(length: 577, mode: .imported))
    XCTAssertEqual(c.chunks.count, 2)
    for chunk in c.chunks {
      XCTAssertEqual(chunk.body["audio"] as? Bool, false)
      XCTAssertFalse(chunk.wantsAudio)
    }
    XCTAssertEqual(c.plan.audioMode, "imported")
    XCTAssertEqual(c.plan.chunks.map(\.audio), ["none", "none"])
  }

  func testGeneratedModeSendsAudioTrueEverywhere() throws {
    let c = try compile(timeline(length: 577, mode: .generated))
    XCTAssertEqual(c.chunks.count, 2)
    for chunk in c.chunks {
      XCTAssertEqual(chunk.body["audio"] as? Bool, true)
      XCTAssertTrue(chunk.wantsAudio)
    }
    XCTAssertEqual(c.plan.audioMode, "generated")
    XCTAssertEqual(c.plan.chunks.map(\.audio), ["generated", "generated"])
    XCTAssertTrue(c.plan.warnings.contains { $0.code == "generated_audio_seams" })
  }

  // MARK: - Settings forwarding

  func testPresetAndLorasForwardedAndExtendPinned() throws {
    let plain = try compile(timeline(length: 577, preset: "kira-video-avocado", negative: "blurry", steps: 12))
    for chunk in plain.chunks {
      XCTAssertEqual(chunk.body["preset"] as? String, "kira-video-avocado")
      XCTAssertNil(chunk.body["loras"])
      XCTAssertEqual(chunk.body["negative_prompt"] as? String, "blurry")
      XCTAssertEqual(chunk.body["steps"] as? Int, 12)
      XCTAssertEqual((chunk.body["extend_to_seconds"] as? NSNumber)?.doubleValue, 0)
      XCTAssertNil(chunk.body["duration"])
    }

    let withLoras = try compile(timeline(loras: [
      .init(path: "a.safetensors", scale: 0.6),
      .init(path: "b.safetensors", scale: 1.0, role: "accel"),
    ]))
    let loras = try XCTUnwrap(withLoras.chunks[0].body["loras"] as? [[String: Any]])
    XCTAssertEqual(loras.count, 2)
    XCTAssertEqual(loras[0]["path"] as? String, "a.safetensors")
    XCTAssertEqual((loras[0]["scale"] as? NSNumber)?.floatValue, 0.6)
    XCTAssertNil(loras[0]["role"])
    XCTAssertEqual(loras[1]["path"] as? String, "b.safetensors")
    XCTAssertEqual(loras[1]["role"] as? String, "accel")
    let req = try decodeLocal(withLoras.chunks[0].body)
    XCTAssertEqual(req.loras?.map(\.path), ["a.safetensors", "b.safetensors"])
    XCTAssertEqual(req.loras?[1].role, "accel")
    XCTAssertEqual(req.extendToSeconds, 0)
  }

  func testCharacterForwardedDisablesSkip() throws {
    let c = try compile(timeline(character: "kira"))
    XCTAssertEqual(c.chunks[0].body["character"] as? String, "kira")
    XCTAssertEqual(c.chunks[0].body["skip_character_injection"] as? Bool, false)
  }

  // MARK: - Guards and determinism

  func testCompileRejectsInvalidValidation() {
    let bad = timeline(keyframes: [.init(id: "k1", imagePath: "/i.png", frame: 100)])
    let v = DirectorValidator.validate(bad, fileExists: { _ in true })
    XCTAssertFalse(v.ok)
    XCTAssertThrowsError(try DirectorCompiler.compile(v, session: "s", source: "test")) { error in
      guard case DirectorError.invalid(let issues) = error else {
        return XCTFail("expected DirectorError.invalid, got \(error)")
      }
      XCTAssertTrue(issues.contains { $0.code == "keyframe_off_grid" })
    }
  }

  func testCompileRejectsUnmaterializedBase64Keyframe() {
    // The route materializes image_base64 to a file BEFORE compile; the
    // compiler refuses to emit a body without a path rather than guessing.
    let t = timeline(keyframes: [.init(id: "k1", imageBase64: "aGVsbG8=", frame: 0)])
    let v = DirectorValidator.validate(t, fileExists: { _ in true })
    XCTAssertTrue(v.ok, "\(v.issues)")
    XCTAssertThrowsError(try DirectorCompiler.compile(v, session: "s", source: "test")) { error in
      guard case DirectorError.invalid(let issues) = error else {
        return XCTFail("expected DirectorError.invalid, got \(error)")
      }
      XCTAssertEqual(issues.map(\.code), ["keyframe_image_missing"])
      XCTAssertEqual(issues.first?.ids, ["k1"])
    }
  }

  func testDeterministic() throws {
    let t = timeline(length: 577, keyframes: [
      .init(id: "k1", imagePath: "/img/a.png", frame: 0),
      .init(id: "k2", imagePath: "/img/b.png", frame: 304, strength: 0.5),
      .init(id: "k3", imagePath: "/img/c.png", frame: 0, isEndFrame: true),
    ], segments: [
      .init(id: "p1", startFrame: 0, lengthFrames: 300, prompt: "one"),
      .init(id: "p2", startFrame: 300, lengthFrames: 277, prompt: "two"),
    ], preset: "p", loras: [.init(path: "l.safetensors", scale: 0.8)])
    let a = try compile(t, session: "same", source: "api")
    let b = try compile(t, session: "same", source: "api")
    XCTAssertEqual(a.plan, b.plan)
    XCTAssertEqual(a.chunks.count, b.chunks.count)
    for (x, y) in zip(a.chunks, b.chunks) {
      XCTAssertEqual(try serialized(x.body), try serialized(y.body))
      XCTAssertEqual(x.outputName, y.outputName)
      XCTAssertEqual(x.lastFrameName, y.lastFrameName)
      XCTAssertEqual(x.carryOverFromChunk, y.carryOverFromChunk)
      XCTAssertEqual(x.span, y.span)
      XCTAssertEqual(x.wantsAudio, y.wantsAudio)
    }
    // The plan the validator returns is the compiler's plan.
    let v = DirectorValidator.validate(t, fileExists: { _ in true })
    XCTAssertEqual(v.plan, a.plan)
  }

  func testEveryBodyDecodesAsLocalVideoRequestAndResolvesKeyframes() throws {
    let t = timeline(length: 577, keyframes: [
      .init(id: "k1", imagePath: "/img/a.png", frame: 0),
      .init(id: "k2", imagePath: "/img/b.png", frame: 160, strength: 0.5),
      .init(id: "k3", imagePath: "/img/c.png", frame: 304),
      .init(id: "k4", imagePath: "/img/d.png", frame: 0, isEndFrame: true),
    ], segments: [
      .init(id: "p1", startFrame: 0, lengthFrames: 300, prompt: "she looks up"),
      .init(id: "p2", startFrame: 300, lengthFrames: 277, prompt: "she looks down"),
    ], preset: "kira-video-avocado", character: "kira")
    let c = try compile(t)
    XCTAssertEqual(c.chunks.count, 2)

    for chunk in c.chunks {
      // Chunk 0 never receives a carry-over (it has the real frame-0
      // keyframe); every later chunk is rendered WITH one.
      let variants = chunk.carryOverFromChunk == nil
        ? [chunk.body] : [chunk.body, chunk.bodyWithCarryOver(imagePath: "/carry.png")]
      for body in variants {
        XCTAssertTrue(JSONSerialization.isValidJSONObject(body))
        let req = try decodeLocal(body)
        let resolved = try WarmServer.resolveKeyframes(req)
        let frames = try XCTUnwrap(req.frames)
        XCTAssertTrue(LTX2VideoGenerator.isValidFrameCount(frames))
        XCTAssertEqual(frames, chunk.span.frames)
        XCTAssertLessThanOrEqual(frames, DirectorMath.maxChunkFrames)
        for extra in resolved.extras {
          XCTAssertGreaterThan(extra.frame, 0)
          XCTAssertLessThan(extra.frame, frames)
          XCTAssertEqual(extra.frame % 8, 0)
        }
        XCTAssertFalse(DirectorMath.keyframeBucketsCollide(resolved.extras.map(\.frame)))
        XCTAssertEqual(req.identityAnchorStrength, 0)
        XCTAssertEqual(req.extendToSeconds, 0)
        XCTAssertNil(req.duration)
        XCTAssertEqual(req.enhance, false)
        XCTAssertEqual(req.skipCharacterInjection, false)
        XCTAssertEqual(req.character, "kira")
        XCTAssertEqual(req.preset, "kira-video-avocado")
        let beats = WarmServer.resolveVideoBeatSchedule(
          requested: req.beatSchedule, isT2V: resolved.initImagePath == nil, totalChunks: 1)
        XCTAssertNil(beats.ignoredReason)
        XCTAssertEqual(beats.effective?.count, req.beatSchedule?.count)
      }
    }
    // Chunk 0 has a real frame-0 keyframe; chunk 1 gets it only via carry-over.
    XCTAssertEqual(try WarmServer.resolveKeyframes(try decodeLocal(c.chunks[0].body)).initImagePath, "/img/a.png")
    XCTAssertNil(try WarmServer.resolveKeyframes(try decodeLocal(c.chunks[1].body)).initImagePath)
    let carried = try WarmServer.resolveKeyframes(try decodeLocal(c.chunks[1].bodyWithCarryOver(imagePath: "/carry.png")))
    XCTAssertEqual(carried.initImagePath, "/carry.png")
    XCTAssertEqual(carried.strength, 1.0)
    XCTAssertEqual(carried.extras.map(\.frame), [16, 288])
  }
}
