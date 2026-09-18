import Foundation
import XCTest

@testable import ZImage

/// An audio-driven timeline chunks SHORTER (WP11, FDD §4.7).
///
/// Measured 2026-09-18 on one seed/keyframe/prompt, as the ratio of mouth
/// motion during speech to mouth motion during silence — "does the mouth go
/// still when the voice stops", which unlike a correlation does not depend on
/// how much of the track is actually speech:
///
///     145-frame chunks   1.92 (one voice), 1.73 (another)
///     273-frame chunks   1.03, 1.27, 1.30  (the three chunks of a 34 s take)
///
/// Chunk 0's 1.03 is no response at all. The voice barely matters; the chunk
/// length is the lever, so `drives_video` buys shorter chunks — more seams,
/// which tone-match and exact-last-frame have made invisible, for a mouth that
/// follows the words.
final class DirectorAudioChunkCeilingTests: XCTestCase {

  private func timeline(lengthFrames: Int, driving: Bool) -> DirectorTimeline {
    DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: lengthFrames),
      globalPrompt: "she talks to camera",
      keyframes: [.init(id: "k1", imagePath: "/img/a.png", frame: 0)],
      audioClips: [
        .init(
          id: "a1", audioPath: "/audio/v.wav", startFrame: 0,
          lengthFrames: lengthFrames, drivesVideo: driving)
      ],
      audio: .init(mode: .imported))
  }

  // MARK: the ceiling

  func testADrivingTimelineChunksAtTheAudioCeiling() {
    let driven = timeline(lengthFrames: 817, driving: true)
    XCTAssertTrue(driven.isAudioDriven)
    XCTAssertEqual(driven.chunkCeilingFrames, DirectorMath.audioDrivenChunkFrames)

    let layout = DirectorMath.chunkLayout(
      lengthFrames: 817, maxFrames: driven.chunkCeilingFrames)
    XCTAssertGreaterThan(layout.count, 3, "817 frames is 3 chunks at 289, more at 145")
    for span in layout {
      XCTAssertLessThanOrEqual(
        span.frames, DirectorMath.audioDrivenChunkFrames,
        "no chunk may exceed the ceiling — 273 frames scored 1.03, no response at all")
    }
  }

  func testAnOrdinaryTimelineIsUnCHANGED() {
    // The whole point of gating on `drives_video`: nothing else re-chunks.
    let plain = timeline(lengthFrames: 817, driving: false)
    XCTAssertFalse(plain.isAudioDriven)
    XCTAssertEqual(plain.chunkCeilingFrames, DirectorMath.maxChunkFrames)
    XCTAssertEqual(
      DirectorMath.chunkLayout(lengthFrames: 817, maxFrames: plain.chunkCeilingFrames).map(\.frames),
      DirectorMath.chunkLayout(lengthFrames: 817).map(\.frames))
  }

  func testATimelineWithNoAudioAtAllIsUnchanged() {
    let bare = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 817), globalPrompt: "x")
    XCTAssertFalse(bare.isAudioDriven)
    XCTAssertEqual(bare.chunkCeilingFrames, DirectorMath.maxChunkFrames)
  }

  // MARK: the layout still holds its invariants

  func testTheShorterChunksStillTileTheTimelineExactly() {
    for length in [145, 289, 577, 817, 1153] {
      let layout = DirectorMath.chunkLayout(
        lengthFrames: length, maxFrames: DirectorMath.audioDrivenChunkFrames)
      // Neighbours share one boundary frame, so the frames sum to L + (n-1).
      let covered = layout.reduce(0) { $0 + $1.frames } - (layout.count - 1)
      XCTAssertEqual(covered, length, "chunks must tile \(length) exactly")
      for span in layout {
        XCTAssertEqual((span.frames - 1) % 8, 0, "every chunk stays 1 + 8k")
      }
      XCTAssertEqual(layout.first?.startFrame, 0)
    }
  }

  // MARK: the compiler and validator agree

  func testTheCompilerEmitsTheShorterChunks() throws {
    let driven = timeline(lengthFrames: 817, driving: true)
    let validation = DirectorValidator.validate(driven, fileExists: { _ in true })
    XCTAssertTrue(validation.ok, "\(validation.issues)")
    let compilation = try DirectorCompiler.compile(validation, session: "T", source: "test")
    XCTAssertGreaterThan(compilation.chunks.count, 3)
    for chunk in compilation.chunks {
      XCTAssertLessThanOrEqual(chunk.span.frames, DirectorMath.audioDrivenChunkFrames)
      XCTAssertEqual(chunk.body["audio_condition_path"] as? String, "/audio/v.wav")
    }
  }

  func testADrivenTimelineHitsTheChunkBudgetSoonerAndSaysWhy() {
    // 16 chunks x 144 steps of timeline: a driven timeline runs out of chunk
    // budget at a shorter DURATION, and must say so rather than silently
    // producing a 17th chunk.
    let tooLong = timeline(lengthFrames: DirectorMath.maxTimelineFrames, driving: true)
    let validation = DirectorValidator.validate(tooLong, fileExists: { _ in true })
    XCTAssertFalse(validation.ok)
    let issue = validation.issues.first { $0.code == "timeline_too_long" }
    XCTAssertNotNil(issue, "\(validation.issues)")
    XCTAssertTrue(
      issue?.message.contains("audio-driven") == true,
      "the message must name the reason: \(issue?.message ?? "-")")

    // The same length WITHOUT driving is still fine.
    let plain = timeline(lengthFrames: DirectorMath.maxTimelineFrames, driving: false)
    XCTAssertTrue(DirectorValidator.validate(plain, fileExists: { _ in true }).ok)
  }
}
