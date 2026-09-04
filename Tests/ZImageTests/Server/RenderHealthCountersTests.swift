import XCTest
@testable import ZImage

/// comfybox#308: `/health.render_count` and `last_render_duration_ms` stayed
/// 0 through an entire HQ two-pass soak — the watchdog confirmed renders
/// completing while the counters never moved. Root cause: the local-video
/// completion path never called anything equivalent to what the six image
/// `run*Generate` methods do inline. Pins the extracted, pure counter step.
final class RenderHealthCountersTests: XCTestCase {

  func testSucceededIncrementsSuccessAndStampsDuration() {
    var counters = RenderHealthCounters()
    counters.apply(.succeeded(durationMs: 4200))
    XCTAssertEqual(counters.successCount, 1)
    XCTAssertEqual(counters.failedCount, 0)
    XCTAssertEqual(counters.lastDurationMs, 4200)
  }

  func testFailedIncrementsFailedAndLeavesDurationUntouched() {
    var counters = RenderHealthCounters(successCount: 2, failedCount: 0, lastDurationMs: 999)
    counters.apply(.failed)
    XCTAssertEqual(counters.successCount, 2, "unrelated counter must not move")
    XCTAssertEqual(counters.failedCount, 1)
    XCTAssertEqual(counters.lastDurationMs, 999, "a failed render has no duration worth reporting as 'last'")
  }

  /// A later success always overwrites `lastDurationMs` — it tracks the most
  /// recent completed render, image or video, matching what `/health`
  /// documents ("last render duration"), not "last successful video render".
  func testLatestSuccessOverwritesPreviousDuration() {
    var counters = RenderHealthCounters()
    counters.apply(.succeeded(durationMs: 1000))
    counters.apply(.succeeded(durationMs: 5000))
    XCTAssertEqual(counters.successCount, 2)
    XCTAssertEqual(counters.lastDurationMs, 5000)
  }

  func testMixedSequenceOfCompletions() {
    var counters = RenderHealthCounters()
    counters.apply(.succeeded(durationMs: 1000))
    counters.apply(.failed)
    counters.apply(.succeeded(durationMs: 3000))
    XCTAssertEqual(counters.successCount, 2)
    XCTAssertEqual(counters.failedCount, 1)
    XCTAssertEqual(counters.lastDurationMs, 3000)
  }

  // MARK: - comfybox#308 (review r1, item 3b): the `.localVideo` WIRING —
  // which of the three real completion outcomes maps to which event. Pins
  // `RenderCompletionEvent.forLocalVideoCompletion`, the exact function
  // `WarmServerCoordinator`'s `.localVideo` case calls at each of its three
  // exit points (success / thrown error / memory-admission refusal), so this
  // is the SELECTION logic under test, not only `RenderHealthCounters.apply`.

  func testLocalVideoSuccessMapsToSucceededWithMillisecondDuration() {
    var counters = RenderHealthCounters()
    counters.apply(.forLocalVideoCompletion(.succeeded(elapsedSeconds: 12.5)))
    XCTAssertEqual(counters.successCount, 1)
    XCTAssertEqual(counters.failedCount, 0)
    XCTAssertEqual(counters.lastDurationMs, 12_500, "elapsedSeconds must convert to whole milliseconds")
  }

  func testLocalVideoThrownErrorMapsToFailed() {
    var counters = RenderHealthCounters()
    counters.apply(.forLocalVideoCompletion(.threw))
    XCTAssertEqual(counters.successCount, 0)
    XCTAssertEqual(counters.failedCount, 1)
    XCTAssertNil(counters.lastDurationMs)
  }

  /// The memory-admission refusal is a real completion (the job reached the
  /// front of the queue and was refused) — NOT a queue-full rejection (which
  /// never dequeues at all, and so never reaches this mapping).
  func testLocalVideoAdmissionRefusalMapsToFailed() {
    var counters = RenderHealthCounters()
    counters.apply(.forLocalVideoCompletion(.admissionRefused))
    XCTAssertEqual(counters.successCount, 0)
    XCTAssertEqual(counters.failedCount, 1)
    XCTAssertNil(counters.lastDurationMs)
  }

  /// One render of each of the three real outcomes, in the order production
  /// code could plausibly hit them — the end-to-end soak scenario the issue
  /// describes, expressed purely.
  func testAllThreeLocalVideoOutcomesInSequence() {
    var counters = RenderHealthCounters()
    counters.apply(.forLocalVideoCompletion(.admissionRefused))
    counters.apply(.forLocalVideoCompletion(.succeeded(elapsedSeconds: 30.0)))
    counters.apply(.forLocalVideoCompletion(.threw))
    XCTAssertEqual(counters.successCount, 1)
    XCTAssertEqual(counters.failedCount, 2)
    XCTAssertEqual(counters.lastDurationMs, 30_000)
  }
}
