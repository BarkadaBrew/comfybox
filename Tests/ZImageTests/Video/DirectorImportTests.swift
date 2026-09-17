import XCTest
@testable import ZImage

/// WP1: clean-room import of the upstream LTX Director `timeline_data`
/// shape (field names only — behaviour mapping, no upstream code).
final class DirectorImportTests: XCTestCase {

  private let settings = DirectorTimeline.Settings(width: 576, height: 896, lengthFrames: 97)

  private func imported(_ json: String) throws -> (timeline: DirectorTimeline, warnings: [DirectorIssue]) {
    try DirectorImport.fromUpstream(timelineData: Data(json.utf8), settings: settings)
  }

  func testImageSegmentsBecomeKeyframesAndPromptSegments() throws {
    let r = try imported("""
    {
      "global_prompt": "a long walk",
      "frame_rate": 24,
      "start_frame": 0,
      "end_frame": 200,
      "segments": [
        { "type": "image", "start": 0, "length": 96, "imageFile": "/img/a.png", "prompt": "she waves", "strength": 1.0 },
        { "type": "image", "start": 96, "length": 100, "imageFile": "/img/b.png", "strength": 0.8 },
        { "type": "image", "start": 200, "length": 48, "imageData": "data:image/png;base64,QUJD", "prompt": "" }
      ]
    }
    """)
    let t = r.timeline
    XCTAssertEqual(t.globalPrompt, "a long walk")
    XCTAssertEqual(t.settings.fps, 24)
    XCTAssertEqual(t.settings.width, 576)
    XCTAssertEqual(t.settings.lengthFrames, DirectorMath.snapLengthUp(248))
    XCTAssertEqual(t.keyframes.count, 3)
    XCTAssertEqual(t.keyframes.map(\.frame), [0, 96, 200])
    XCTAssertEqual(t.keyframes.map(\.imagePath), ["/img/a.png", "/img/b.png", nil])
    XCTAssertEqual(t.keyframes[1].strength, 0.8)
    XCTAssertEqual(t.keyframes[2].imageBase64, "QUJD", "data URL prefix stripped")
    XCTAssertEqual(t.promptSegments.count, 1)
    XCTAssertEqual(t.promptSegments[0].startFrame, 0)
    XCTAssertEqual(t.promptSegments[0].lengthFrames, 96)
    XCTAssertEqual(t.promptSegments[0].prompt, "she waves")
    XCTAssertEqual(Set(t.keyframes.map(\.id)).count, 3, "ids unique")
    XCTAssertEqual(t.audio.mode, .generated)
    XCTAssertTrue(r.warnings.isEmpty)
    XCTAssertEqual(t.version, 1)
  }

  func testAudioSegmentsBecomeImportedClips() throws {
    let r = try imported("""
    {
      "global_prompt": "p",
      "segments": [
        { "type": "image", "start": 0, "length": 96, "imageFile": "/img/a.png" },
        { "type": "audio", "start": 8, "length": 40, "audioFile": "/aud/in-segments.wav", "trimStart": 4, "gain": 0.5 }
      ],
      "audioSegments": [
        { "start": 48, "length": 48, "audioFile": "/aud/voice.wav", "trimStart": 12, "gain": 0.75 },
        { "start": 100, "length": 20, "audioFile": "/aud/fx.mp3" }
      ]
    }
    """)
    let t = r.timeline
    XCTAssertEqual(t.audio.mode, .imported)
    XCTAssertEqual(t.audioClips.count, 3)
    XCTAssertEqual(t.audioClips.map(\.audioPath), ["/aud/in-segments.wav", "/aud/voice.wav", "/aud/fx.mp3"])
    XCTAssertEqual(t.audioClips.map(\.startFrame), [8, 48, 100])
    XCTAssertEqual(t.audioClips.map(\.lengthFrames), [40, 48, 20])
    XCTAssertEqual(t.audioClips.map(\.trimStartFrames), [4, 12, 0])
    XCTAssertEqual(t.audioClips.map(\.gain), [0.5, 0.75, 1.0])
    XCTAssertEqual(Set(t.audioClips.map(\.id)).count, 3)
    XCTAssertEqual(t.settings.lengthFrames, DirectorMath.snapLengthUp(120))
  }

  func testVideoSegmentWarnsUnsupported() throws {
    let r = try imported("""
    {
      "global_prompt": "p",
      "segments": [
        { "type": "image", "start": 0, "length": 96, "imageFile": "/img/a.png" },
        { "type": "video", "start": 0, "length": 96, "videoFile": "/vid/ref.mp4" }
      ],
      "motionSegments": [ { "start": 0, "length": 48, "videoFile": "/vid/motion.mp4" } ],
      "retakeMode": true, "retakeVideo": "/vid/base.mp4", "retakeStart": 0, "retakeLength": 48
    }
    """)
    XCTAssertEqual(r.timeline.keyframes.count, 1)
    XCTAssertTrue(r.timeline.referenceClips.isEmpty)
    XCTAssertFalse(r.timeline.retake.enabled)
    let unsupported = r.warnings.filter { $0.code == "unsupported_upstream_segment" }
    XCTAssertEqual(unsupported.count, 3, "video segment, motion segment, retake block")
    XCTAssertTrue(unsupported.allSatisfy { $0.severity == .warning })
  }

  func testEndFrameFlagCarries() throws {
    let r = try imported("""
    {
      "global_prompt": "p", "end_frame": 289,
      "segments": [
        { "type": "image", "start": 0, "length": 96, "imageFile": "/img/a.png" },
        { "type": "image", "start": 288, "length": 1, "imageFile": "/img/z.png", "isEndFrame": true, "strength": 0.9 }
      ]
    }
    """)
    XCTAssertEqual(r.timeline.keyframes[1].isEndFrame, true)
    XCTAssertEqual(r.timeline.keyframes[1].frame, 288)
    XCTAssertEqual(r.timeline.keyframes[1].strength, 0.9)
    XCTAssertEqual(r.timeline.keyframes[0].isEndFrame, false)
    XCTAssertEqual(r.timeline.settings.lengthFrames, 289)
  }

  func testFrameRateMapsToFps() throws {
    let r = try imported("""
    { "global_prompt": "p", "frame_rate": 30.0, "segments": [ { "type": "image", "start": 0, "length": 60, "imageFile": "/a.png" } ] }
    """)
    XCTAssertEqual(r.timeline.settings.fps, 30)
    let none = try imported("""
    { "global_prompt": "p", "segments": [ { "type": "image", "start": 0, "length": 60, "imageFile": "/a.png" } ] }
    """)
    XCTAssertEqual(none.timeline.settings.fps, settings.fps, "falls back to the caller's settings")
    XCTAssertEqual(none.timeline.settings.seed, settings.seed)
  }

  func testGarbageThrowsImportFailed() {
    XCTAssertThrowsError(try imported("not json")) { error in
      guard case DirectorError.importFailed = error else { return XCTFail("\(error)") }
    }
    XCTAssertThrowsError(try imported("[1, 2, 3]")) { error in
      guard case DirectorError.importFailed = error else { return XCTFail("\(error)") }
    }
    // A timeline with no global prompt and no segments is still garbage.
    XCTAssertThrowsError(try imported("{ \"unrelated\": 1 }")) { error in
      guard case DirectorError.importFailed = error else { return XCTFail("\(error)") }
    }
  }

  func testLenientNumbersAndUnknownKeys() throws {
    let r = try imported("""
    { "global_prompt": "p", "frame_rate": 24, "future": { "x": 1 },
      "segments": [ { "type": "image", "start": 8.0, "length": 96.0, "imageFile": "/a.png", "strength": 0.5, "guide_strength": 2 } ] }
    """)
    XCTAssertEqual(r.timeline.keyframes[0].frame, 8)
    XCTAssertEqual(r.timeline.promptSegments.count, 0)
    XCTAssertEqual(r.timeline.settings.lengthFrames, DirectorMath.snapLengthUp(104))
  }
}
