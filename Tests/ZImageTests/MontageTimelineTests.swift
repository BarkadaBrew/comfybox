import XCTest
@testable import ZImage

final class MontageTimelineTests: XCTestCase {

  func testDurationMathWithTransitions() throws {
    // 3 + 4 + 3 seconds with 0.5s dissolve + 1s fade → 10 − 1.5 = 8.5s
    let t = try MontageTimeline.resolve(
      segmentDurations: [3, 4, 3],
      transitions: [
        MontageTransition(kind: .dissolve, durationS: 0.5),
        MontageTransition(kind: .fade, durationS: 1.0),
      ])
    XCTAssertEqual(t.totalDuration, 8.5, accuracy: 1e-9)
    XCTAssertEqual(t.placed[0].start, 0)
    XCTAssertEqual(t.placed[1].start, 2.5, accuracy: 1e-9)  // 3 − 0.5
    XCTAssertEqual(t.placed[2].start, 5.5, accuracy: 1e-9)  // 2.5 + 4 − 1
  }

  func testEmptyTransitionsMeansAllCuts() throws {
    let t = try MontageTimeline.resolve(segmentDurations: [2, 2, 2], transitions: [])
    XCTAssertEqual(t.totalDuration, 6, accuracy: 1e-9)
    XCTAssertEqual(t.transitions.count, 2)
    XCTAssertTrue(t.transitions.allSatisfy { $0.kind == .cut && $0.durationS == 0 })
  }

  func testValidationErrors() {
    XCTAssertThrowsError(try MontageTimeline.resolve(segmentDurations: [], transitions: []))
    // Wrong transition count.
    XCTAssertThrowsError(try MontageTimeline.resolve(
      segmentDurations: [2, 2, 2],
      transitions: [MontageTransition(kind: .cut)])) { error in
      guard case MontageError.badTransitionCount = error else {
        return XCTFail("expected badTransitionCount, got \(error)")
      }
    }
    // Transition as long as a neighbor.
    XCTAssertThrowsError(try MontageTimeline.resolve(
      segmentDurations: [2, 1],
      transitions: [MontageTransition(kind: .dissolve, durationS: 1.0)])) { error in
      guard case MontageError.transitionTooLong = error else {
        return XCTFail("expected transitionTooLong, got \(error)")
      }
    }
    // Non-positive segment duration.
    XCTAssertThrowsError(try MontageTimeline.resolve(segmentDurations: [2, 0], transitions: []))
  }

  func testFrameStateInsideAndOutsideOverlap() throws {
    let t = try MontageTimeline.resolve(
      segmentDurations: [3, 3],
      transitions: [MontageTransition(kind: .dissolve, durationS: 1.0)])
    // t=1: solidly inside segment 0.
    let solo = t.frameState(at: 1.0)
    XCTAssertEqual(solo.primary.index, 0)
    XCTAssertEqual(solo.primary.localT, 1.0, accuracy: 1e-9)
    XCTAssertNil(solo.blend)
    // Overlap zone is [2.0, 3.0): at t=2.5 progress is 0.5.
    let mid = t.frameState(at: 2.5)
    XCTAssertEqual(mid.primary.index, 0)
    let blend = try XCTUnwrap(mid.blend)
    XCTAssertEqual(blend.index, 1)
    XCTAssertEqual(blend.progress, 0.5, accuracy: 1e-9)
    XCTAssertEqual(blend.kind, .dissolve)
    // Past the overlap: segment 1 alone.
    let after = t.frameState(at: 3.5)
    XCTAssertEqual(after.primary.index, 1)
    XCTAssertNil(after.blend)
    // Clamped past the end: still segment 1.
    XCTAssertEqual(t.frameState(at: 99).primary.index, 1)
  }

  func testCutHasNoBlendState() throws {
    let t = try MontageTimeline.resolve(segmentDurations: [2, 2], transitions: [])
    XCTAssertEqual(t.frameState(at: 1.999).primary.index, 0)
    let s = t.frameState(at: 2.0)
    XCTAssertEqual(s.primary.index, 1)
    XCTAssertNil(s.blend)
  }
}
