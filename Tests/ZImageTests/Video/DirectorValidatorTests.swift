import XCTest
@testable import ZImage

/// WP1: the pure validator. File existence and the audio probe are injected
/// so every issue code is reachable without touching disk.
final class DirectorValidatorTests: XCTestCase {

  private func timeline(
    length: Int = 289, fps: Int = 24, width: Int = 576, height: Int = 896,
    prompt: String = "a woman walks",
    keyframes: [DirectorTimeline.Keyframe] = [.init(id: "k1", imagePath: "/img/k1.png", frame: 0)],
    segments: [DirectorTimeline.PromptSegment] = [],
    clips: [DirectorTimeline.AudioClip] = [],
    mode: DirectorTimeline.AudioMode = .generated,
    reference: [DirectorTimeline.ReferenceClip] = [],
    retake: DirectorTimeline.Retake = .init()
  ) -> DirectorTimeline {
    DirectorTimeline(
      settings: .init(fps: fps, width: width, height: height, lengthFrames: length),
      globalPrompt: prompt, keyframes: keyframes, promptSegments: segments,
      audioClips: clips, audio: .init(mode: mode), referenceClips: reference, retake: retake)
  }

  private func validate(
    _ t: DirectorTimeline, exists: @escaping (String) -> Bool = { _ in true },
    probe: @escaping (String) -> AudioProbe? = { _ in nil }
  ) -> DirectorValidation {
    DirectorValidator.validate(t, fileExists: exists, audioProbe: probe)
  }

  private func codes(_ v: DirectorValidation, _ severity: DirectorIssue.Severity? = nil) -> [String] {
    v.issues.filter { severity == nil || $0.severity == severity! }.map(\.code)
  }

  func testSnapsLengthAndWarns() {
    let v = validate(timeline(length: 290))
    XCTAssertEqual(v.snapped.settings.lengthFrames, 297)
    XCTAssertTrue(codes(v, .warning).contains("length_snapped"))
    XCTAssertTrue(v.ok)
    XCTAssertEqual(v.plan?.lengthFrames, 297)
  }

  func testRejectsOffGridKeyframe() {
    let v = validate(timeline(keyframes: [.init(id: "k1", imagePath: "/i.png", frame: 100)]))
    XCTAssertFalse(v.ok)
    XCTAssertNil(v.plan)
    let issue = v.issues.first { $0.code == "keyframe_off_grid" }
    XCTAssertEqual(issue?.severity, .error)
    XCTAssertEqual(issue?.ids, ["k1"])
  }

  func testRejectsCollidingKeyframes() {
    let v = validate(timeline(keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 8),
      .init(id: "k2", imagePath: "/i.png", frame: 8),
    ]))
    XCTAssertFalse(v.ok)
    let issue = v.issues.first { $0.code == "keyframes_collide" }
    XCTAssertEqual(issue?.severity, .error)
    XCTAssertEqual(Set(issue?.ids ?? []), ["k1", "k2"])
  }

  func testWarnsCloseKeyframes() {
    let v = validate(timeline(keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0),
      .init(id: "k2", imagePath: "/i.png", frame: 16),
    ]))
    XCTAssertTrue(v.ok)
    let issue = v.issues.first { $0.code == "keyframes_close" }
    XCTAssertEqual(issue?.severity, .warning)
    XCTAssertEqual(Set(issue?.ids ?? []), ["k1", "k2"])
    XCTAssertFalse(codes(validate(timeline(keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0),
      .init(id: "k2", imagePath: "/i.png", frame: 24),
    ]))).contains("keyframes_close"))
  }

  func testEndFramePinsToLastFrame() {
    let v = validate(timeline(length: 289, keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0),
      .init(id: "k2", imagePath: "/i.png", frame: 0, isEndFrame: true),
    ]))
    XCTAssertTrue(v.ok, "\(v.issues)")
    XCTAssertEqual(v.snapped.keyframes[1].frame, 288)
    XCTAssertEqual(v.plan?.keyframeTicks.map(\.frame), [0, 288])
  }

  func testMultipleEndFramesRejected() {
    let v = validate(timeline(keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0, isEndFrame: true),
      .init(id: "k2", imagePath: "/i.png", frame: 8, isEndFrame: true),
    ]))
    XCTAssertFalse(v.ok)
    XCTAssertTrue(codes(v, .error).contains("multiple_end_frames"))
  }

  func testBoundaryKeyframeWarns() {
    let layout = DirectorMath.chunkLayout(lengthFrames: 577)
    let boundary = layout[1].startFrame
    let full = validate(timeline(length: 577, keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0),
      .init(id: "k2", imagePath: "/i.png", frame: boundary),
    ]))
    XCTAssertTrue(full.ok)
    XCTAssertTrue(codes(full, .warning).contains("keyframe_on_chunk_boundary"))
    XCTAssertFalse(codes(full).contains("keyframe_partial_strength_on_boundary"))
    XCTAssertEqual(full.issues.first { $0.code == "keyframe_on_chunk_boundary" }?.ids, ["k2"])

    let partial = validate(timeline(length: 577, keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0),
      .init(id: "k2", imagePath: "/i.png", frame: boundary, strength: 0.6),
    ]))
    XCTAssertTrue(partial.ok)
    XCTAssertTrue(codes(partial, .warning).contains("keyframe_on_chunk_boundary"))
    XCTAssertTrue(codes(partial, .warning).contains("keyframe_partial_strength_on_boundary"))
  }

  func testShortSegmentWarnings() {
    let v = validate(timeline(segments: [
      .init(id: "p1", startFrame: 0, lengthFrames: 7, prompt: "a"),
      .init(id: "p2", startFrame: 8, lengthFrames: 24, prompt: "b"),
      .init(id: "p3", startFrame: 40, lengthFrames: 96, prompt: "c"),
    ]))
    XCTAssertTrue(v.ok)
    XCTAssertEqual(v.issues.first { $0.code == "segment_shorter_than_latent_frame" }?.ids, ["p1"])
    XCTAssertEqual(v.issues.first { $0.code == "segment_shorter_than_bias_window" }?.ids, ["p2"])
    XCTAssertFalse(codes(v).contains("segments_overlap"))
  }

  func testSegmentOutOfRangeAndEmptyPromptAndOverlap() {
    let v = validate(timeline(length: 289, segments: [
      .init(id: "p1", startFrame: 200, lengthFrames: 96, prompt: "a"),
      .init(id: "p2", startFrame: -1, lengthFrames: 8, prompt: "b"),
      .init(id: "p3", startFrame: 0, lengthFrames: 0, prompt: "c"),
      .init(id: "p4", startFrame: 0, lengthFrames: 48, prompt: "   "),
    ]))
    XCTAssertFalse(v.ok)
    let out = v.issues.filter { $0.code == "segment_out_of_range" }
    XCTAssertEqual(Set(out.flatMap(\.ids)), ["p1", "p2", "p3"])
    XCTAssertEqual(v.issues.first { $0.code == "segment_empty_prompt" }?.ids, ["p4"])

    let overlap = validate(timeline(segments: [
      .init(id: "p1", startFrame: 0, lengthFrames: 96, prompt: "a"),
      .init(id: "p2", startFrame: 48, lengthFrames: 96, prompt: "b"),
    ]))
    XCTAssertTrue(overlap.ok, "overlap is a warning")
    let issue = overlap.issues.first { $0.code == "segments_overlap" }
    XCTAssertEqual(issue?.severity, .warning)
    XCTAssertEqual(Set(issue?.ids ?? []), ["p1", "p2"])
  }

  func testInpaintAudioRejected() {
    let v = validate(timeline(mode: .inpaint))
    XCTAssertFalse(v.ok)
    XCTAssertTrue(codes(v, .error).contains("audio_mode_unsupported"))
  }

  func testReferenceAndRetakeRejected() {
    let v = validate(timeline(
      reference: [.init(id: "r1", videoPath: "/v.mp4", startFrame: 0, lengthFrames: 96)],
      retake: .init(enabled: true, videoPath: "/v.mp4", startFrame: 0, lengthFrames: 48)))
    XCTAssertFalse(v.ok)
    XCTAssertTrue(codes(v, .error).contains("reference_clips_unsupported"))
    XCTAssertTrue(codes(v, .error).contains("retake_unsupported"))
  }

  func testImportedModeMissingClipIsError() {
    let clip = DirectorTimeline.AudioClip(id: "a1", audioPath: "/missing.wav", startFrame: 0, lengthFrames: 96)
    let v = validate(timeline(clips: [clip], mode: .imported), exists: { $0.hasSuffix(".png") })
    XCTAssertFalse(v.ok)
    XCTAssertEqual(v.issues.first { $0.code == "audio_clip_missing" }?.ids, ["a1"])
    XCTAssertFalse(codes(v).contains("audio_clips_ignored"))
    XCTAssertFalse(codes(v).contains("generated_audio_seams"))
  }

  func testImportedModeProbeCodes() {
    let good = DirectorTimeline.AudioClip(id: "a1", audioPath: "/good.wav", startFrame: 0, lengthFrames: 96)
    let silent = DirectorTimeline.AudioClip(id: "a2", audioPath: "/video-only.mp4", startFrame: 96, lengthFrames: 48)
    let trimmed = DirectorTimeline.AudioClip(id: "a3", audioPath: "/short.wav", startFrame: 160, lengthFrames: 48, trimStartFrames: 96)
    let past = DirectorTimeline.AudioClip(id: "a4", audioPath: "/good.wav", startFrame: 240, lengthFrames: 96)
    let gone = DirectorTimeline.AudioClip(id: "a5", audioPath: "/good.wav", startFrame: 400, lengthFrames: 96)
    let v = validate(
      timeline(length: 289, clips: [good, silent, trimmed, past, gone], mode: .imported),
      probe: { path in
        switch path {
        case "/good.wav": return AudioProbe(durationSeconds: 10, hasAudioTrack: true)
        case "/video-only.mp4": return AudioProbe(durationSeconds: 10, hasAudioTrack: false)
        case "/short.wav": return AudioProbe(durationSeconds: 2, hasAudioTrack: true)
        default: return nil
        }
      })
    XCTAssertFalse(v.ok)
    XCTAssertEqual(v.issues.first { $0.code == "audio_clip_undecodable" }?.ids, ["a2"])
    XCTAssertEqual(v.issues.first { $0.code == "audio_trim_past_end" }?.ids, ["a3"])
    let pastIssue = v.issues.filter { $0.code == "audio_clip_past_end" }
    XCTAssertEqual(Set(pastIssue.flatMap(\.ids)), ["a4", "a5"])
    // Clips are clipped to the timeline in the snapped document; a clip that
    // starts past the end is dropped.
    XCTAssertEqual(v.snapped.audioClips.map(\.id), ["a1", "a2", "a3", "a4"])
    XCTAssertEqual(v.snapped.audioClips[3].lengthFrames, 49)
  }

  func testInvalidGainAndStrength() {
    let v = validate(timeline(
      keyframes: [.init(id: "k1", imagePath: "/i.png", frame: 0, strength: 0)],
      clips: [.init(id: "a1", audioPath: "/a.wav", startFrame: 0, lengthFrames: 8, gain: -1)],
      mode: .imported))
    XCTAssertFalse(v.ok)
    XCTAssertEqual(v.issues.first { $0.code == "invalid_strength" }?.ids, ["k1"])
    XCTAssertEqual(v.issues.first { $0.code == "invalid_gain" }?.ids, ["a1"])
    XCTAssertTrue(codes(validate(timeline(
      keyframes: [.init(id: "k1", imagePath: "/i.png", frame: 0, strength: 1.5)]))).contains("invalid_strength"))
  }

  func testGeneratedModeIgnoresClipsWithWarning() {
    let clip = DirectorTimeline.AudioClip(id: "a1", audioPath: "/missing.wav", startFrame: 0, lengthFrames: 96)
    let v = validate(timeline(clips: [clip], mode: .generated), exists: { _ in false })
    // Only the keyframe image is checked; the clip is ignored, not probed.
    XCTAssertFalse(codes(v).contains("audio_clip_missing"))
    let issue = v.issues.first { $0.code == "audio_clips_ignored" }
    XCTAssertEqual(issue?.severity, .warning)
    XCTAssertEqual(issue?.ids, ["a1"])
  }

  func testGeneratedMultiChunkWarnsSeams() {
    let multi = validate(timeline(length: 577))
    XCTAssertTrue(multi.ok)
    XCTAssertTrue(codes(multi, .warning).contains("generated_audio_seams"))
    let single = validate(timeline(length: 289))
    XCTAssertFalse(codes(single).contains("generated_audio_seams"))
    let imported = validate(timeline(length: 577, mode: .imported))
    XCTAssertFalse(codes(imported).contains("generated_audio_seams"))
  }

  func testTooShortAndTooLong() {
    let snapped = validate(timeline(length: 96))
    XCTAssertEqual(snapped.snapped.settings.lengthFrames, 97)
    XCTAssertTrue(snapped.ok, "96 snaps up to 97, the production floor")
    XCTAssertFalse(codes(snapped).contains("timeline_too_short"))

    let short = validate(timeline(length: 48))
    XCTAssertFalse(short.ok)
    XCTAssertTrue(codes(short, .error).contains("timeline_too_short"))
    XCTAssertNil(short.plan)

    let long = validate(timeline(length: 4617))
    XCTAssertFalse(long.ok)
    XCTAssertTrue(codes(long, .error).contains("timeline_too_long"))
    XCTAssertTrue(validate(timeline(length: 4609)).ok)
  }

  func testStructuralErrors() {
    let v = validate(timeline(fps: 0, width: 570, height: 896, prompt: "  "))
    XCTAssertFalse(v.ok)
    XCTAssertTrue(codes(v, .error).contains("missing_global_prompt"))
    XCTAssertTrue(codes(v, .error).contains("invalid_fps"))
    XCTAssertTrue(codes(v, .error).contains("invalid_dimensions"))
  }

  func testKeyframeImageCodes() {
    let v = validate(timeline(keyframes: [
      .init(id: "k1", frame: 0),
      .init(id: "k2", imagePath: "/nope.png", frame: 8),
      .init(id: "k3", imageBase64: "AAAA", frame: 16),
      .init(id: "k4", imagePath: "/nope.png", imageBase64: "AAAA", frame: 24),
    ]), exists: { _ in false })
    XCTAssertFalse(v.ok)
    XCTAssertEqual(v.issues.first { $0.code == "keyframe_image_missing" }?.ids, ["k1"])
    XCTAssertEqual(v.issues.first { $0.code == "keyframe_image_unreadable" }?.ids, ["k2"])
  }

  func testKeyframeOutOfRangeAndDuplicateIds() {
    let v = validate(timeline(length: 289, keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0),
      .init(id: "k1", imagePath: "/i.png", frame: 296),
      .init(id: "k2", imagePath: "/i.png", frame: -8),
    ], segments: [.init(id: "k2", startFrame: 0, lengthFrames: 48, prompt: "x")]))
    XCTAssertFalse(v.ok)
    let range = v.issues.filter { $0.code == "keyframe_out_of_range" }
    XCTAssertEqual(range.count, 2)
    let dup = v.issues.filter { $0.code == "duplicate_id" }
    XCTAssertEqual(Set(dup.flatMap(\.ids)), ["k1", "k2"])
  }

  func testNoKeyframesWarns() {
    let none = validate(timeline(keyframes: []))
    XCTAssertTrue(none.ok)
    XCTAssertTrue(codes(none, .warning).contains("no_keyframes"))
    let endOnly = validate(timeline(keyframes: [.init(id: "k1", imagePath: "/i.png", frame: 0, isEndFrame: true)]))
    XCTAssertTrue(codes(endOnly, .warning).contains("no_keyframes"), "no frame-0 keyframe: chunk 0 is T2V")
    XCTAssertFalse(codes(validate(timeline())).contains("no_keyframes"))
  }

  func testValidTimelineHasNoErrorsAndSkeletonPlan() {
    let t = timeline(
      length: 577,
      keyframes: [
        .init(id: "k1", imagePath: "/i.png", frame: 0),
        .init(id: "k2", imagePath: "/i.png", frame: 0, isEndFrame: true),
      ],
      segments: [
        .init(id: "p1", startFrame: 0, lengthFrames: 200, prompt: "a"),
        .init(id: "p2", startFrame: 200, lengthFrames: 200, prompt: "b"),
      ])
    let v = validate(t)
    XCTAssertTrue(v.ok, "\(v.issues)")
    XCTAssertTrue(v.issues.allSatisfy { $0.severity == .warning })
    let plan = try! XCTUnwrap(v.plan)
    XCTAssertEqual(plan.lengthFrames, 577)
    XCTAssertEqual(plan.fps, 24)
    XCTAssertEqual(plan.width, 576)
    XCTAssertEqual(plan.height, 896)
    XCTAssertEqual(plan.audioMode, "generated")
    XCTAssertEqual(plan.chunks.count, 2)
    XCTAssertEqual(plan.chunks.map(\.index), [0, 1])
    XCTAssertEqual(plan.chunks.map(\.startFrame), [0, 288])
    XCTAssertEqual(plan.chunks.map(\.endFrame), [288, 576])
    XCTAssertEqual(plan.chunks.map(\.frames), [289, 289])
    XCTAssertEqual(plan.chunks.map(\.seed), [42, 43])
    XCTAssertEqual(plan.chunks.map(\.carryOver), [false, true])
    XCTAssertEqual(plan.chunks.map(\.audio), ["generated", "generated"])
    XCTAssertEqual(plan.keyframeTicks, [.init(id: "k1", frame: 0), .init(id: "k2", frame: 576)])
    XCTAssertEqual(plan.boundaryFrames, [288])
    XCTAssertEqual(plan.warnings, v.issues.filter { $0.severity == .warning })
    XCTAssertTrue(plan.warnings.contains { $0.code == "generated_audio_seams" })
    XCTAssertEqual(validate(timeline(mode: .imported)).plan?.chunks.map(\.audio), ["none"])
  }

  func testValidationIsDeterministic() {
    let t = timeline(length: 577, keyframes: [
      .init(id: "k1", imagePath: "/i.png", frame: 0),
      .init(id: "k2", imagePath: "/i.png", frame: 16),
      .init(id: "k3", imagePath: "/i.png", frame: 288, strength: 0.5),
    ])
    XCTAssertEqual(validate(t), validate(t))
  }

  func testImportedClipOutOfRangeIsError() {
    let negative = DirectorTimeline.AudioClip(id: "a1", audioPath: "/a.wav", startFrame: -48, lengthFrames: 120)
    let empty = DirectorTimeline.AudioClip(id: "a2", audioPath: "/a.wav", startFrame: 0, lengthFrames: 0)
    let badTrim = DirectorTimeline.AudioClip(id: "a3", audioPath: "/a.wav", startFrame: 0, lengthFrames: 24, trimStartFrames: -1)
    let fine = DirectorTimeline.AudioClip(id: "a4", audioPath: "/a.wav", startFrame: 0, lengthFrames: 24)
    let v = validate(timeline(clips: [negative, empty, badTrim, fine], mode: .imported))
    XCTAssertFalse(v.ok)
    let ids = v.issues.filter { $0.code == "audio_clip_out_of_range" && $0.severity == .error }.flatMap(\.ids)
    XCTAssertEqual(ids, ["a1", "a2", "a3"])
    XCTAssertTrue(validate(timeline(clips: [fine], mode: .imported)).ok)
  }

  func testMissingFpsValidatesAgainstBuiltin() {
    var t = timeline()
    t.settings.fps = nil
    t.settings.seed = nil
    let v = validate(t)
    XCTAssertTrue(v.ok, "\(v.issues)")
    XCTAssertEqual(v.plan?.fps, DirectorTimeline.Settings.defaultFps)
    XCTAssertNil(v.plan?.chunks.first?.seed, "no seed is invented before the server resolves one")
  }
}
