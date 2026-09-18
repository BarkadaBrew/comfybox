import Foundation
import XCTest

@testable import ZImage

/// Pause-snapped chunk boundaries (WP11's spec line, root-caused 2026-09-18).
///
/// A continuation chunk starts from the previous chunk's last frame. A
/// boundary landing mid-word makes that frame hold an OPEN mouth, and the pose
/// fights the voice for the rest of the chunk — measured 0.68 against 2.45 for
/// a closed-mouth start, with pin strength ruled out. Landing the boundary in
/// a pause is what the design asked for and what the measurement now supports.
final class DirectorPauseSnapTests: XCTestCase {

  private let length = 817
  private let ceiling = DirectorMath.audioDrivenChunkFrames

  private var layout: [DirectorMath.ChunkSpan] {
    DirectorMath.chunkLayout(lengthFrames: length, maxFrames: ceiling)
  }

  /// Every chunk must stay 1 + 8k and within the ceiling, and they must tile
  /// the timeline exactly — the invariants a snap must not break.
  private func assertLegal(
    _ spans: [DirectorMath.ChunkSpan], file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertFalse(spans.isEmpty, file: file, line: line)
    for span in spans {
      XCTAssertEqual((span.frames - 1) % 8, 0, "chunk \(span.index) is not 1+8k", file: file, line: line)
      XCTAssertLessThanOrEqual(span.frames, ceiling, "chunk \(span.index) exceeds the ceiling", file: file, line: line)
      XCTAssertGreaterThanOrEqual(span.frames, 9, file: file, line: line)
    }
    XCTAssertEqual(spans.first?.startFrame, 0, file: file, line: line)
    let covered = spans.reduce(0) { $0 + $1.frames } - (spans.count - 1)
    XCTAssertEqual(covered, length, "chunks must tile the timeline exactly", file: file, line: line)
    for (a, b) in zip(spans, spans.dropFirst()) {
      XCTAssertEqual(b.startFrame, a.startFrame + a.frames - 1, "neighbours share one frame", file: file, line: line)
    }
  }

  // MARK: identity

  func testNoQuietFrameAnywhereLeavesTheLayoutEXACTLYAsItWas() {
    // The safety property: a track with no pauses (or an unreadable one) must
    // reproduce `chunkLayout` byte for byte, never an illegal chunk.
    let snapped = DirectorMath.pauseSnapped(
      layout, lengthFrames: length, maxFrames: ceiling, isQuiet: { _ in false })
    XCTAssertEqual(snapped.map(\.frames), layout.map(\.frames))
    XCTAssertEqual(snapped.map(\.startFrame), layout.map(\.startFrame))
    assertLegal(snapped)
  }

  func testQuietEverywhereKeepsTheNOMINALBoundaries() {
    // When every frame qualifies, the nearest quiet frame IS the nominal one,
    // so nothing should move.
    let snapped = DirectorMath.pauseSnapped(
      layout, lengthFrames: length, maxFrames: ceiling, isQuiet: { _ in true })
    XCTAssertEqual(snapped.map(\.startFrame), layout.map(\.startFrame))
    assertLegal(snapped)
  }

  func testASingleChunkTimelineHasNoBoundaryToMove() {
    let single = DirectorMath.chunkLayout(lengthFrames: 137, maxFrames: ceiling)
    XCTAssertEqual(single.count, 1)
    XCTAssertEqual(
      DirectorMath.pauseSnapped(single, lengthFrames: 137, maxFrames: ceiling, isQuiet: { _ in true }),
      single)
  }

  // MARK: snapping

  func testABoundaryMovesToTheNEARESTQuietFrame() {
    let nominal = layout[1].startFrame
    // One quiet frame, 8 before the nominal boundary.
    let target = nominal - 8
    let snapped = DirectorMath.pauseSnapped(
      layout, lengthFrames: length, maxFrames: ceiling, isQuiet: { $0 == target })
    XCTAssertEqual(snapped[1].startFrame, target, "the first boundary moved into the pause")
    assertLegal(snapped)
  }

  func testOnlyMultiplesOfTheLatentStrideAreReachable() {
    // A pause at a non-multiple-of-8 offset cannot be used: the chunk would
    // stop being 1 + 8k. The boundary must stay put rather than go illegal.
    let nominal = layout[1].startFrame
    let snapped = DirectorMath.pauseSnapped(
      layout, lengthFrames: length, maxFrames: ceiling, isQuiet: { $0 == nominal - 3 })
    XCTAssertEqual(snapped[1].startFrame, nominal, "an unreachable pause is ignored")
    assertLegal(snapped)
  }

  func testASnapNeverPushesANeighbourPastTheCeiling() {
    // Quiet ONLY very early: moving there would leave the next chunk longer
    // than the ceiling, so the boundary must refuse.
    let snapped = DirectorMath.pauseSnapped(
      layout, lengthFrames: length, maxFrames: ceiling, isQuiet: { $0 == 8 })
    assertLegal(snapped)
    XCTAssertEqual(snapped.map(\.startFrame), layout.map(\.startFrame), "no legal move exists")
  }

  func testEveryBoundaryIsConsideredNotJustTheFirst() {
    let targets = Set(layout.dropFirst().dropLast().map { $0.startFrame - 8 })
    XCTAssertGreaterThan(targets.count, 1)
    let snapped = DirectorMath.pauseSnapped(
      layout, lengthFrames: length, maxFrames: ceiling, isQuiet: { targets.contains($0) })
    assertLegal(snapped)
    let movedCount = zip(snapped, layout).filter { $0.startFrame != $1.startFrame }.count
    XCTAssertGreaterThan(movedCount, 1, "more than one boundary found its pause")
  }

  func testTheSearchWindowIsBounded() {
    // A pause far outside the window must not drag a boundary across the clip.
    let nominal = layout[1].startFrame
    let snapped = DirectorMath.pauseSnapped(
      layout, lengthFrames: length, maxFrames: ceiling, searchFrames: 24,
      isQuiet: { $0 == nominal - 80 })
    XCTAssertEqual(snapped[1].startFrame, nominal, "80 frames is outside a 24-frame search")
    assertLegal(snapped)
  }
}
