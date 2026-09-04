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

  /// The exact soak scenario from the issue: renders complete but nothing
  /// ever called an equivalent of `apply` — expressed here as "counters
  /// start at zero and STAY zero unless something drives them", the inverse
  /// pin of the bug (the real fix is wiring `.localVideo`'s completion into
  /// this type — this test just documents what "never wired" looks like).
  func testUnappliedCountersStayZero() {
    let counters = RenderHealthCounters()
    XCTAssertEqual(counters.successCount, 0)
    XCTAssertEqual(counters.failedCount, 0)
    XCTAssertNil(counters.lastDurationMs)
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
}
