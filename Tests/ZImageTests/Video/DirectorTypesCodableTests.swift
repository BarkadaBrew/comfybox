import XCTest
@testable import ZImage

/// WP1: the DirectorTimeline wire/file format. Every field marked "default"
/// in the schema decodes with that default under BOTH the shared
/// `DirectorJSON.decoder()` and the WarmServer route decoder (a plain
/// `JSONDecoder` with `.convertFromSnakeCase`) — they must agree byte for byte.
final class DirectorTypesCodableTests: XCTestCase {

  private static let minimal = """
  {
    "version": 1,
    "settings": { "width": 576, "height": 896, "length_frames": 289 },
    "global_prompt": "a woman walks through a rainy street",
    "keyframes": [ { "id": "k1", "image_path": "/tmp/k1.png", "frame": 0 } ]
  }
  """

  private func warmServerDecoder() -> JSONDecoder {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return d
  }

  private func assertDefaults(_ t: DirectorTimeline, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(t.version, 1, file: file, line: line)
    // No decode defaults: an absent fps/seed stays nil so the preset/config
    // value applies at render (request > preset > config > builtin).
    XCTAssertNil(t.settings.fps, file: file, line: line)
    XCTAssertNil(t.settings.seed, file: file, line: line)
    XCTAssertEqual(t.settings.width, 576, file: file, line: line)
    XCTAssertEqual(t.settings.height, 896, file: file, line: line)
    XCTAssertEqual(t.settings.lengthFrames, 289, file: file, line: line)
    XCTAssertEqual(t.settings.loras, [], file: file, line: line)
    XCTAssertNil(t.settings.preset, file: file, line: line)
    XCTAssertNil(t.settings.negativePrompt, file: file, line: line)
    XCTAssertNil(t.settings.steps, file: file, line: line)
    XCTAssertNil(t.settings.character, file: file, line: line)
    XCTAssertEqual(t.keyframes.count, 1, file: file, line: line)
    XCTAssertEqual(t.keyframes[0].id, "k1", file: file, line: line)
    XCTAssertEqual(t.keyframes[0].imagePath, "/tmp/k1.png", file: file, line: line)
    XCTAssertNil(t.keyframes[0].imageBase64, file: file, line: line)
    XCTAssertEqual(t.keyframes[0].frame, 0, file: file, line: line)
    XCTAssertEqual(t.keyframes[0].strength, 1.0, file: file, line: line)
    XCTAssertFalse(t.keyframes[0].isEndFrame, file: file, line: line)
    XCTAssertEqual(t.promptSegments, [], file: file, line: line)
    XCTAssertEqual(t.audioClips, [], file: file, line: line)
    XCTAssertEqual(t.audio.mode, .generated, file: file, line: line)
    XCTAssertEqual(t.referenceClips, [], file: file, line: line)
    XCTAssertFalse(t.retake.enabled, file: file, line: line)
    XCTAssertNil(t.retake.videoPath, file: file, line: line)
    XCTAssertEqual(t.retake.startFrame, 0, file: file, line: line)
    XCTAssertEqual(t.retake.lengthFrames, 0, file: file, line: line)
    XCTAssertEqual(t.retake.strength, 1.0, file: file, line: line)
  }

  func testMinimalDocumentDecodesWithDefaults() throws {
    let data = Data(Self.minimal.utf8)
    let shared = try DirectorJSON.decoder().decode(DirectorTimeline.self, from: data)
    let route = try warmServerDecoder().decode(DirectorTimeline.self, from: data)
    assertDefaults(shared)
    assertDefaults(route)
    XCTAssertEqual(shared, route)
  }

  func testAudioClipDefaultsTrimAndGain() throws {
    let json = """
    { "settings": { "width": 576, "height": 896, "length_frames": 289 },
      "global_prompt": "p",
      "audio_clips": [ { "id": "a1", "audio_path": "/tmp/a.wav", "start_frame": 0, "length_frames": 120 } ],
      "audio": { "mode": "imported" } }
    """
    let t = try DirectorJSON.decoder().decode(DirectorTimeline.self, from: Data(json.utf8))
    XCTAssertEqual(t.audioClips.count, 1)
    XCTAssertEqual(t.audioClips[0].trimStartFrames, 0)
    XCTAssertEqual(t.audioClips[0].gain, 1.0)
    XCTAssertEqual(t.audio.mode, .imported)
  }

  func testMissingVersionDecodesAsOne() throws {
    let json = """
    { "settings": { "width": 576, "height": 896, "length_frames": 289 }, "global_prompt": "p" }
    """
    let t = try DirectorJSON.decoder().decode(DirectorTimeline.self, from: Data(json.utf8))
    XCTAssertEqual(t.version, 1)
    let route = try warmServerDecoder().decode(DirectorTimeline.self, from: Data(json.utf8))
    XCTAssertEqual(route.version, 1)
  }

  func testFutureVersionRejected() {
    let json = """
    { "version": 2, "settings": { "width": 576, "height": 896, "length_frames": 289 }, "global_prompt": "p" }
    """
    XCTAssertThrowsError(try DirectorJSON.decoder().decode(DirectorTimeline.self, from: Data(json.utf8))) { error in
      guard case DirectorError.unsupportedVersion(let v) = error else {
        return XCTFail("expected unsupportedVersion, got \(error)")
      }
      XCTAssertEqual(v, 2)
    }
    XCTAssertThrowsError(try warmServerDecoder().decode(DirectorTimeline.self, from: Data(json.utf8))) { error in
      guard case DirectorError.unsupportedVersion(2) = error else {
        return XCTFail("expected unsupportedVersion(2), got \(error)")
      }
    }
  }

  func testUnknownKeysAreIgnored() throws {
    let json = """
    {
      "version": 1,
      "phase2_top_level": { "anything": [1, 2, 3] },
      "settings": { "width": 576, "height": 896, "length_frames": 289, "future_setting": "x" },
      "global_prompt": "p",
      "keyframes": [ { "id": "k1", "image_path": "/tmp/k1.png", "frame": 0, "compression": 12 } ],
      "prompt_segments": [ { "id": "p1", "start_frame": 0, "length_frames": 96, "prompt": "hi", "weight": 2 } ],
      "audio": { "mode": "generated", "master": true },
      "retake": { "enabled": false, "extra": 1 }
    }
    """
    let t = try DirectorJSON.decoder().decode(DirectorTimeline.self, from: Data(json.utf8))
    XCTAssertEqual(t.keyframes[0].id, "k1")
    XCTAssertEqual(t.promptSegments[0].prompt, "hi")
    XCTAssertEqual(t.audio.mode, .generated)
    XCTAssertFalse(t.retake.enabled)
  }

  func testRoundTripIsByteStableAndSnakeCase() throws {
    let t = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 577, seed: 771_144,
                      preset: "kira-video", loras: [.init(path: "a.safetensors", scale: 0.8)]),
      globalPrompt: "global",
      keyframes: [
        .init(id: "k1", imagePath: "/tmp/k1.png", frame: 0),
        .init(id: "k2", imagePath: "/tmp/k2.png", frame: 576, strength: 0.7, isEndFrame: true),
      ],
      promptSegments: [.init(id: "p1", startFrame: 0, lengthFrames: 96, prompt: "walks")],
      audioClips: [.init(id: "a1", audioPath: "/tmp/a.wav", startFrame: 0, lengthFrames: 120, trimStartFrames: 4, gain: 0.5)],
      audio: .init(mode: .imported))
    let first = try DirectorJSON.encoder().encode(t)
    let decoded = try DirectorJSON.decoder().decode(DirectorTimeline.self, from: first)
    XCTAssertEqual(decoded, t)
    let second = try DirectorJSON.encoder().encode(decoded)
    XCTAssertEqual(first, second, "encode(decode(x)) must be byte-identical to encode(x)")

    let text = String(decoding: first, as: UTF8.self)
    for key in ["length_frames", "is_end_frame", "trim_start_frames", "global_prompt",
                "prompt_segments", "audio_clips", "image_path", "start_frame", "reference_clips"] {
      XCTAssertTrue(text.contains("\"\(key)\""), "missing snake_case key \(key)")
    }
    for camel in ["lengthFrames", "isEndFrame", "trimStartFrames", "globalPrompt",
                  "promptSegments", "audioClips", "imagePath", "startFrame", "imageBase64"] {
      XCTAssertFalse(text.contains("\"\(camel)\""), "camelCase key leaked: \(camel)")
    }
    // The compact encoder is the same document without whitespace.
    let compact = try DirectorJSON.encoder(pretty: false).encode(t)
    XCTAssertFalse(String(decoding: compact, as: UTF8.self).contains("\n"))
    XCTAssertEqual(try DirectorJSON.decoder().decode(DirectorTimeline.self, from: compact), t)
  }

  func testIssueAndPlanEncodeSnakeCase() throws {
    let issue = DirectorIssue(severity: .warning, code: "length_snapped", message: "m", ids: ["k1"])
    let text = String(decoding: try DirectorJSON.encoder(pretty: false).encode(issue), as: UTF8.self)
    XCTAssertTrue(text.contains("\"severity\":\"warning\""))
    let plan = DirectorPlan(
      lengthFrames: 289, fps: 24, width: 576, height: 896, audioMode: "generated",
      chunks: [.init(index: 0, startFrame: 0, endFrame: 288, frames: 289, seed: 42, carryOver: false,
                     keyframes: [.init(id: "k1", localFrame: 0, strength: 1)],
                     promptSegments: ["p1"],
                     beatSchedule: [.init(text: "t", startFrac: 0, endFrac: 0.5)],
                     audio: "generated")],
      keyframeTicks: [.init(id: "k1", frame: 0)], boundaryFrames: [], warnings: [issue])
    let planText = String(decoding: try DirectorJSON.encoder(pretty: false).encode(plan), as: UTF8.self)
    for key in ["length_frames", "audio_mode", "start_frame", "end_frame", "carry_over", "local_frame",
                "prompt_segments", "beat_schedule", "start_frac", "end_frac", "keyframe_ticks", "boundary_frames"] {
      XCTAssertTrue(planText.contains("\"\(key)\""), "plan missing \(key)")
    }
    XCTAssertEqual(try DirectorJSON.decoder().decode(DirectorPlan.self, from: Data(planText.utf8)), plan)
  }

  func testDirectorErrorDescriptions() {
    XCTAssertEqual(DirectorError.chunkFailed(chunk: 1, stage: "render", message: "x").description,
                   "chunk 2 (render): x")
    XCTAssertTrue(DirectorError.invalid([
      DirectorIssue(severity: .error, code: "keyframe_off_grid", message: "m", ids: [])
    ]).description.contains("keyframe_off_grid"))
    XCTAssertNotNil(DirectorError.stitchFailed("s").errorDescription)
    XCTAssertNotNil(DirectorError.fileUnreadable("f").errorDescription)
    XCTAssertNotNil(DirectorError.importFailed("i").errorDescription)
  }
}
