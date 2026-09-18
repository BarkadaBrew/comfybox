import Foundation
import XCTest

@testable import ZImage

/// The audio chunk ceiling was REVERTED 2026-09-18 (see
/// `DirectorTimeline.chunkCeilingFrames`). These tests now pin the revert: a
/// driving timeline chunks exactly like any other.
///
/// Why: the 145-frame ceiling rested on a loud-vs-quiet mouth-motion metric
/// that counted the one-time transient of the subject raising her head from
/// the opening keyframe as the voice starts. That transient is ~40 frames, so
/// it is a much larger FRACTION of a short chunk — enough on its own to
/// produce the "1.03 -> 3.05" that justified the change. Excluding the first
/// 48 frames collapses it (2.45 -> 1.05), and the clean comparison
/// (continuation chunks, which have no onset) runs the other way: 273-frame
/// chunks 1.31, 137-frame chunks 1.10.
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

  func testADrivingTimelineChunksLikeANYOtherNow() {
    // The revert: `drives_video` still marks the timeline, but it no longer
    // changes how it is split.
    let driven = timeline(lengthFrames: 817, driving: true)
    XCTAssertTrue(driven.isAudioDriven, "the flag still means what it meant")
    XCTAssertEqual(
      driven.chunkCeilingFrames, DirectorMath.maxChunkFrames,
      "the 145 ceiling was reverted — its evidence was an onset artifact")
    XCTAssertEqual(
      DirectorMath.chunkLayout(lengthFrames: 817, maxFrames: driven.chunkCeilingFrames).map(\.frames),
      DirectorMath.chunkLayout(lengthFrames: 817).map(\.frames))
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

  func testTheCompilerEmitsTheSAMEChunksForDrivenAndPlain() throws {
    // The revert in one assertion: conditioning is still wired to every
    // chunk, but the SPANS are the ordinary ones.
    let driven = timeline(lengthFrames: 817, driving: true)
    let validation = DirectorValidator.validate(driven, fileExists: { _ in true })
    XCTAssertTrue(validation.ok, "\(validation.issues)")
    let compilation = try DirectorCompiler.compile(validation, session: "T", source: "test")
    XCTAssertEqual(
      compilation.chunks.map(\.span.frames),
      DirectorMath.chunkLayout(lengthFrames: 817).map(\.frames))
    for chunk in compilation.chunks {
      XCTAssertEqual(chunk.body["audio_condition_path"] as? String, "/audio/v.wav")
    }
  }

  func testADrivenTimelineNoLongerHitsTheBudgetSooner() {
    // 16 chunks x 144 steps of timeline: a driven timeline runs out of chunk
    // budget at a shorter DURATION, and must say so rather than silently
    // producing a 17th chunk.
    // Both reach the same limit again, because both chunk the same way: a
    // driving flag no longer costs timeline length.
    let atLimit = DirectorMath.maxTimelineFrames
    let driven = DirectorValidator.validate(
      timeline(lengthFrames: atLimit, driving: true), fileExists: { _ in true })
    let plain = DirectorValidator.validate(
      timeline(lengthFrames: atLimit, driving: false), fileExists: { _ in true })
    XCTAssertEqual(
      driven.issues.contains { $0.code == "timeline_too_long" },
      plain.issues.contains { $0.code == "timeline_too_long" },
      "driving must not change whether the length is accepted")
    XCTAssertEqual(driven.plan?.chunks.count, plain.plan?.chunks.count)
  }
}
