import Foundation
import MLX
import XCTest

@testable import ZImage

/// Audio-driven chunks: a monologue that keeps its lips (FDD §4.7, WP11).
///
/// The contract: a clip marked `drives_video` conditions every chunk it covers
/// on that chunk's own slice of the voice, and the final assembly still muxes
/// the untouched master track.
final class DirectorAudioDrivenTests: XCTestCase {

  private func timeline(
    lengthFrames: Int = 577, driving: Bool = true, clipStart: Int = 0, trim: Int = 0
  ) -> DirectorTimeline {
    DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: lengthFrames),
      globalPrompt: "she talks to camera",
      keyframes: [.init(id: "k1", imagePath: "/img/a.png", frame: 0)],
      audioClips: [
        .init(
          id: "a1", audioPath: "/audio/voice.wav", startFrame: clipStart,
          lengthFrames: lengthFrames - clipStart, trimStartFrames: trim, gain: 1.0,
          drivesVideo: driving)
      ],
      audio: .init(mode: .imported))
  }

  private func compile(_ timeline: DirectorTimeline) throws -> DirectorCompilation {
    let validation = DirectorValidator.validate(timeline, fileExists: { _ in true })
    XCTAssertTrue(validation.ok, "\(validation.issues)")
    return try DirectorCompiler.compile(validation, session: "TEST", source: "test")
  }

  // MARK: the wire

  func testADrivingClipTurnsOnAudioAndNamesTheVoiceForEveryChunk() throws {
    let compilation = try compile(timeline())
    XCTAssertEqual(compilation.chunks.count, 2, "577 frames is two chunks")
    for chunk in compilation.chunks {
      XCTAssertEqual(chunk.body["audio"] as? Bool, true, "chunk \(chunk.index) must run its audio stream")
      XCTAssertEqual(chunk.body["audio_condition_path"] as? String, "/audio/voice.wav")
    }
  }

  func testEachChunkGetsItsOwnSliceOfTheOneVoice() throws {
    let compilation = try compile(timeline())
    let starts = compilation.chunks.map { $0.body["audio_condition_start_frame"] as? Int }
    XCTAssertEqual(starts.first, 0, "chunk 0 starts at the top of the track")
    XCTAssertEqual(
      starts.last, compilation.chunks[1].span.startFrame,
      "chunk 1 starts exactly where it sits on the timeline — the voice does not restart")
  }

  func testAClipTrimAtTheHeadShiftsEveryChunkEqually() throws {
    let compilation = try compile(timeline(trim: 48))
    let starts = compilation.chunks.compactMap { $0.body["audio_condition_start_frame"] as? Int }
    XCTAssertEqual(starts.first, 48, "a trimmed head is an offset into the FILE, not the timeline")
    XCTAssertEqual(starts.last, compilation.chunks[1].span.startFrame + 48)
  }

  func testAChunkBeforeTheVoiceStartsIsNotConditioned() throws {
    // The voice starts at frame 300, inside chunk 1 — chunk 0 has nothing to follow.
    let compilation = try compile(timeline(clipStart: 300))
    XCTAssertNil(compilation.chunks[0].body["audio_condition_path"])
    XCTAssertEqual(compilation.chunks[1].body["audio_condition_path"] as? String, "/audio/voice.wav")
  }

  func testAnOrdinaryImportedClipChangesNothing() throws {
    let compilation = try compile(timeline(driving: false))
    for chunk in compilation.chunks {
      XCTAssertNil(chunk.body["audio_condition_path"])
      XCTAssertEqual(
        chunk.body["audio"] as? Bool, false,
        "imported audio still renders chunks silent and muxes the mix once")
    }
  }

  func testTheFlagSurvivesTheWire() throws {
    let json = """
      {"version":1,"settings":{"width":576,"height":896,"length_frames":289},
       "global_prompt":"x","keyframes":[],"prompt_segments":[],
       "audio_clips":[{"id":"a1","audio_path":"/v.wav","start_frame":0,
       "length_frames":289,"drives_video":true}],"audio":{"mode":"imported"}}
      """
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let timeline = try decoder.decode(DirectorTimeline.self, from: Data(json.utf8))
    XCTAssertTrue(timeline.audioClips.first?.drivesVideo == true)

    // And a clip without the key is not driving.
    let plain = json.replacingOccurrences(of: ",\"drives_video\":true", with: "")
    let without = try decoder.decode(DirectorTimeline.self, from: Data(plain.utf8))
    XCTAssertFalse(without.audioClips.first?.drivesVideo == true)
  }

  // MARK: fitting a slice to a chunk

  func testConditioningIsTrimmedOrPaddedToTheChunkLength() {
    let latents = MLX.zeros([1, 8, 40, 16], dtype: .float32)
    XCTAssertEqual(LTX2AVDenoiseState.fit(latents, toFrames: 40).dim(2), 40)
    XCTAssertEqual(
      LTX2AVDenoiseState.fit(latents, toFrames: 25).dim(2), 25,
      "a longer slice is trimmed, never left to run past the chunk")
    let padded = LTX2AVDenoiseState.fit(latents, toFrames: 60)
    XCTAssertEqual(padded.dim(2), 60, "a short tail is silence, so the voice cannot drift")
    XCTAssertEqual(padded[0, 0, 59, 0].item(Float.self), 0)
  }
}
