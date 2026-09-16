import XCTest
@testable import ZImage

/// WP1: the pure frame arithmetic shared by the engine, the CLI and the
/// desktop ruler. Cross-checked against the engine's own planning helpers so
/// the plan the client sees is what the generator renders.
final class DirectorMathTests: XCTestCase {

  func testSnapLengthUp() {
    XCTAssertEqual(DirectorMath.snapLengthUp(96), 97)
    XCTAssertEqual(DirectorMath.snapLengthUp(97), 97)
    XCTAssertEqual(DirectorMath.snapLengthUp(98), 105)
    XCTAssertEqual(DirectorMath.snapLengthUp(289), 289)
    XCTAssertEqual(DirectorMath.snapLengthUp(290), 297)
    XCTAssertEqual(DirectorMath.snapLengthUp(1), 9)
    XCTAssertEqual(DirectorMath.snapLengthUp(0), 9)
    XCTAssertEqual(DirectorMath.snapLengthUp(-5), 9)
    XCTAssertEqual(DirectorMath.snapLengthUp(9), 9)
    XCTAssertEqual(DirectorMath.snapLengthUp(10), 17)
    for l in 2...5000 {
      let s = DirectorMath.snapLengthUp(l)
      XCTAssertTrue(DirectorMath.isValidLength(s), "\(l) -> \(s)")
      XCTAssertGreaterThanOrEqual(s, l)
      XCTAssertLessThan(s - l, 8)
    }
    XCTAssertFalse(DirectorMath.isValidLength(1))
    XCTAssertFalse(DirectorMath.isValidLength(8))
    XCTAssertTrue(DirectorMath.isValidLength(9))
  }

  func testLatentIndexFloorsLikeThePipeline() {
    XCTAssertEqual(DirectorMath.latentIndex(frame: 0, latF: 37), 0)
    XCTAssertEqual(DirectorMath.latentIndex(frame: 7, latF: 37), 0)
    XCTAssertEqual(DirectorMath.latentIndex(frame: 8, latF: 37), 1)
    XCTAssertEqual(DirectorMath.latentIndex(frame: 15, latF: 37), 1)
    XCTAssertEqual(DirectorMath.latentIndex(frame: 288, latF: 37), 36)
    XCTAssertEqual(DirectorMath.latentIndex(frame: 500, latF: 37), 36, "clamped to latF-1")
    XCTAssertEqual(DirectorMath.latentFrames(for: 289), 37)
    XCTAssertEqual(DirectorMath.latentFrames(for: 97), 13)
    XCTAssertEqual(DirectorMath.latentFrames(for: 1), 1)
    // Grid-snapped frames map onto exactly one latent row each; ceil and
    // floor agree on the grid, which is why the validator forbids off-grid.
    for k in 0..<37 {
      XCTAssertEqual(DirectorMath.latentIndex(frame: k * 8, latF: 37), k)
    }
  }

  func testSnapToGrid() {
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 0, length: 289), 0)
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 3, length: 289), 0)
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 4, length: 289), 0, "ties round down")
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 5, length: 289), 8)
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 100, length: 289), 96, "tie rounds down (WP3 contract: 100 -> 96)")
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 96, length: 289), 96)
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 101, length: 289), 104)
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 288, length: 289), 288)
    XCTAssertEqual(DirectorMath.snapToGrid(frame: 400, length: 289), 288, "clamped to L-1")
    XCTAssertEqual(DirectorMath.snapToGrid(frame: -9, length: 289), 0)
  }

  func testFramesSecondsConversion() {
    XCTAssertEqual(DirectorMath.frames(seconds: 12.0, fps: 24), 288)
    XCTAssertEqual(DirectorMath.frames(seconds: 4.0416, fps: 24), 97)
    XCTAssertEqual(DirectorMath.seconds(frames: 289, fps: 24), 289.0 / 24.0, accuracy: 1e-9)
    XCTAssertEqual(DirectorMath.seconds(frames: 0, fps: 24), 0)
  }

  func testChunkLayoutSingle() {
    for l in [97, 145, 289] {
      let layout = DirectorMath.chunkLayout(lengthFrames: l)
      XCTAssertEqual(layout.count, 1, "\(l)")
      XCTAssertEqual(layout[0].index, 0)
      XCTAssertEqual(layout[0].startFrame, 0)
      XCTAssertEqual(layout[0].endFrame, l - 1)
      XCTAssertEqual(layout[0].frames, l)
    }
  }

  func testChunkLayoutBalanced() {
    let a = DirectorMath.chunkLayout(lengthFrames: 297)
    XCTAssertEqual(a.map(\.frames), [153, 145])
    XCTAssertEqual(a.map(\.startFrame), [0, 152])
    XCTAssertEqual(a.map(\.endFrame), [152, 296])

    let b = DirectorMath.chunkLayout(lengthFrames: 577)
    XCTAssertEqual(b.map(\.frames), [289, 289])
    XCTAssertEqual(b.map(\.startFrame), [0, 288])
    XCTAssertEqual(b.map(\.endFrame), [288, 576])

    let snapped = DirectorMath.snapLengthUp(578)
    XCTAssertEqual(snapped, 585)
    let c = DirectorMath.chunkLayout(lengthFrames: snapped)
    XCTAssertEqual(c.count, 3)
    for span in c {
      XCTAssertTrue(DirectorMath.isValidLength(span.frames))
      XCTAssertGreaterThanOrEqual(span.frames, 145)
      XCTAssertLessThanOrEqual(span.frames, 289)
      XCTAssertTrue(LTX2VideoGenerator.isValidFrameCount(span.frames))
    }
    for i in 1..<c.count {
      XCTAssertEqual(c[i].startFrame, c[i - 1].endFrame, "shared boundary")
    }
    XCTAssertEqual(c.map { $0.frames - 1 }.reduce(0, +) + 1, snapped)

    // Every legal timeline length: spans are 1+8k, within the window, shared
    // boundaries, and sum back to L.
    var l = 97
    while l <= DirectorMath.maxTimelineFrames {
      let layout = DirectorMath.chunkLayout(lengthFrames: l)
      XCTAssertFalse(layout.isEmpty)
      XCTAssertLessThanOrEqual(layout.count, DirectorMath.maxChunks)
      XCTAssertEqual(layout[0].startFrame, 0)
      XCTAssertEqual(layout.last!.endFrame, l - 1)
      for (i, span) in layout.enumerated() {
        XCTAssertEqual(span.index, i)
        XCTAssertEqual(span.endFrame, span.startFrame + span.frames - 1)
        XCTAssertTrue(DirectorMath.isValidLength(span.frames), "L=\(l) span \(span.frames)")
        XCTAssertLessThanOrEqual(span.frames, DirectorMath.maxChunkFrames)
        XCTAssertGreaterThanOrEqual(span.frames, layout.count == 1 ? 97 : 145, "L=\(l)")
        if i > 0 { XCTAssertEqual(span.startFrame, layout[i - 1].endFrame) }
      }
      XCTAssertEqual(layout.map { $0.frames - 1 }.reduce(0, +) + 1, l)
      l += 8
    }
    XCTAssertEqual(DirectorMath.chunkLayout(lengthFrames: DirectorMath.maxTimelineFrames).count, 16)
  }

  func testChunkLayoutAgreesWithEngineChunkPlanEqualChunks() {
    let plan = LTX2VideoGenerator.chunkPlan(
      framesPerChunk: 289, extendToSeconds: Float(577) / 24, fps: 24)
    XCTAssertEqual(plan.totalChunks, 2)
    XCTAssertEqual(plan.totalFrames, 577)
    XCTAssertEqual(DirectorMath.chunkLayout(lengthFrames: 577).map(\.frames), [289, 289])

    var l = 97
    while l <= DirectorMath.maxTimelineFrames {
      for span in DirectorMath.chunkLayout(lengthFrames: l) {
        let single = LTX2VideoGenerator.chunkPlan(
          framesPerChunk: span.frames, extendToSeconds: 0, fps: 24)
        XCTAssertEqual(single.totalChunks, 1, "L=\(l) span \(span.frames)")
        XCTAssertEqual(single.totalFrames, span.frames)
        XCTAssertEqual(
          WarmServer.resolvedLTX2Frames(
            requestFrames: span.frames, videoConfigDefaults: VideoDefaultValues(), diagnostic: false),
          span.frames, "server fold must be the identity on director spans")
      }
      l += 8
    }
  }

  func testBeatFractionsSplitAtBoundary() {
    let layout = DirectorMath.chunkLayout(lengthFrames: 577)
    let c0 = try! XCTUnwrap(DirectorMath.beatFractions(segmentStart: 200, segmentLength: 200, chunk: layout[0]))
    XCTAssertEqual(Double(c0.startFrac), 200.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(Double(c0.endFrac), 1.0, accuracy: 1e-6)
    let c1 = try! XCTUnwrap(DirectorMath.beatFractions(segmentStart: 200, segmentLength: 200, chunk: layout[1]))
    XCTAssertEqual(Double(c1.startFrac), 0.0, accuracy: 1e-6)
    XCTAssertEqual(Double(c1.endFrac), 112.0 / 289.0, accuracy: 1e-6)
    XCTAssertNil(DirectorMath.beatFractions(segmentStart: 0, segmentLength: 96, chunk: layout[1]))
    XCTAssertNil(DirectorMath.beatFractions(segmentStart: 400, segmentLength: 100, chunk: layout[0]))
    // A segment that ends exactly on the boundary frame belongs to chunk 0 only.
    XCTAssertNotNil(DirectorMath.beatFractions(segmentStart: 100, segmentLength: 188, chunk: layout[0]))
    XCTAssertNil(DirectorMath.beatFractions(segmentStart: 100, segmentLength: 188, chunk: layout[1]))
  }

  func testLocalFrameAndBoundaryOwnership() {
    let layout = DirectorMath.chunkLayout(lengthFrames: 577)
    XCTAssertEqual(DirectorMath.localFrame(global: 288, chunk: layout[0]), 288)
    XCTAssertEqual(DirectorMath.localFrame(global: 288, chunk: layout[1]), 0)
    XCTAssertTrue(DirectorMath.chunkOwnsKeyframe(layout[0], frame: 288))
    XCTAssertFalse(DirectorMath.chunkOwnsKeyframe(layout[1], frame: 288), "boundary keyframe is chunk 0's last frame")
    XCTAssertTrue(DirectorMath.chunkOwnsKeyframe(layout[0], frame: 0))
    XCTAssertTrue(DirectorMath.chunkOwnsKeyframe(layout[1], frame: 296))
    XCTAssertTrue(DirectorMath.chunkOwnsKeyframe(layout[1], frame: 576))
    XCTAssertFalse(DirectorMath.chunkOwnsKeyframe(layout[0], frame: 296))
    XCTAssertEqual(DirectorMath.owningChunk(frame: 288, layout: layout)?.index, 0)
    XCTAssertEqual(DirectorMath.owningChunk(frame: 289, layout: layout)?.index, 1)
    XCTAssertEqual(DirectorMath.owningChunk(frame: 0, layout: layout)?.index, 0)
    XCTAssertNil(DirectorMath.owningChunk(frame: 577, layout: layout))
    XCTAssertNil(DirectorMath.owningChunk(frame: -1, layout: layout))
  }

  func testKeyframeBucketsCollide() {
    XCTAssertTrue(DirectorMath.keyframeBucketsCollide([8, 8]))
    XCTAssertTrue(DirectorMath.keyframeBucketsCollide([8, 12]))
    XCTAssertFalse(DirectorMath.keyframeBucketsCollide([0, 8, 16]))
    XCTAssertFalse(DirectorMath.keyframeBucketsCollide([]))
    XCTAssertFalse(DirectorMath.keyframeBucketsCollide([7]))
    XCTAssertTrue(DirectorMath.keyframeBucketsCollide([0, 7]))
  }
}
