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
    // A driving timeline chunks at the AUDIO ceiling, not the 289 default
    // (WP11) — so assert the property, not a frame count that moves with it.
    XCTAssertGreaterThan(compilation.chunks.count, 1)
    for chunk in compilation.chunks {
      XCTAssertLessThanOrEqual(chunk.span.frames, DirectorMath.audioDrivenChunkFrames)
      XCTAssertEqual(chunk.body["audio"] as? Bool, true, "chunk \(chunk.index) must run its audio stream")
      XCTAssertEqual(chunk.body["audio_condition_path"] as? String, "/audio/voice.wav")
    }
  }

  func testEachChunkGetsItsOwnSliceOfTheOneVoice() throws {
    let compilation = try compile(timeline())
    // EVERY chunk reads the track at its own timeline position — the voice is
    // continuous across the seams rather than restarting at each one.
    for chunk in compilation.chunks {
      XCTAssertEqual(
        chunk.body["audio_condition_start_frame"] as? Int, chunk.span.startFrame,
        "chunk \(chunk.index) must read the track where it sits on the timeline")
    }
    XCTAssertEqual(
      compilation.chunks.first?.body["audio_condition_start_frame"] as? Int, 0,
      "chunk 0 starts at the top of the track")
  }

  func testAClipTrimAtTheHeadShiftsEveryChunkEqually() throws {
    let compilation = try compile(timeline(trim: 48))
    for chunk in compilation.chunks {
      XCTAssertEqual(
        chunk.body["audio_condition_start_frame"] as? Int, chunk.span.startFrame + 48,
        "a trimmed head is a constant offset into the FILE, not the timeline")
    }
  }

  func testAChunkBeforeTheVoiceStartsIsNotConditioned() throws {
    // The voice starts at frame 300: every chunk that ENDS before it has
    // nothing to follow, and every chunk the clip covers is conditioned.
    // Stated against the span so it survives a change of chunk ceiling.
    let compilation = try compile(timeline(clipStart: 300))
    var sawUnconditioned = false
    var sawConditioned = false
    for chunk in compilation.chunks {
      let path = chunk.body["audio_condition_path"] as? String
      if chunk.span.startFrame + chunk.span.frames <= 300 {
        XCTAssertNil(path, "chunk \(chunk.index) ends before the voice starts")
        sawUnconditioned = true
      } else {
        XCTAssertEqual(path, "/audio/voice.wav", "chunk \(chunk.index) is covered by the clip")
        sawConditioned = true
      }
    }
    XCTAssertTrue(sawUnconditioned && sawConditioned, "the fixture must exercise both sides")
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

  /// The crash of gate run 2: a latent with the right time axis but the wrong
  /// feature width patchified to a 1208-wide token, and MLX answered `addmm`
  /// with `fatalError` — killing the engine, not the job.
  func testAMisshapenConditioningLatentIsRefusedRatherThanFatal() {
    let stream = MLX.zeros([1, 8, 152, 16], dtype: .float32)
    // What the old encode produced: frequency on the causal axis.
    let transposed = MLX.zeros([1, 8, 16, 151], dtype: .float32)
    XCTAssertNil(
      LTX2AVDenoiseState.conditioning(transposed, like: stream),
      "a wrong feature width must degrade to generated audio, never reach addmm")
    XCTAssertNil(LTX2AVDenoiseState.conditioning(MLX.zeros([1, 16, 152, 16]), like: stream))
    XCTAssertNil(LTX2AVDenoiseState.conditioning(MLX.zeros([8, 152, 16]), like: stream))
  }

  func testAWellShapedConditioningLatentIsAcceptedAndFitted() {
    let stream = MLX.zeros([1, 8, 152, 16], dtype: .float32)
    let given = MLX.zeros([1, 8, 140, 16], dtype: .float32)
    let fitted = LTX2AVDenoiseState.conditioning(given, like: stream)
    XCTAssertEqual(fitted?.shape, [1, 8, 152, 16], "short slices pad, they do not refuse")
  }

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
