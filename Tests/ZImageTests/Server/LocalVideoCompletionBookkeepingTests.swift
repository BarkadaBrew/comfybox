import Foundation
import XCTest

@testable import ZImage

/// comfybox#308 (review r2, item 2b): the `.localVideo` case's three exit
/// points (success / thrown error / memory-admission refusal) must all
/// reach `/health`'s render counters. `RenderCompletionEvent
/// .forLocalVideoCompletion` and `RenderHealthCounters.apply` were already
/// pinned in isolation, but nothing proved the ACTOR actually calls them —
/// a deleted call site at one of the three real exit points would leave
/// every existing test green.
///
/// This drives `WarmServerCoordinator.finishLocalVideo` through the
/// `WarmServerQueueProbe` `#if DEBUG` seam (`ModelOperationQueueTests.swift`
/// established the pattern: the coordinator's queue/admission machinery is
/// file-private and does real system memory probing + GPU calls, so a real
/// end-to-end `.localVideo` render can't run in a unit test — see
/// `vacateImageModelsAndAdmitVideo`). The seam calls the SAME
/// `finishLocalVideo` function all three production call sites call, so
/// this is the strongest wiring proof available without a live engine
/// (intent.md: agents run unit tests only).
final class LocalVideoCompletionBookkeepingTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    try isolateComfyBoxStateDirectory()
  }

  func testSuccessIncrementsSuccessCountStampsDurationAndClearsLastError() async throws {
    let probe = makeQueueProbe()
    let result = await probe.finishLocalVideo(.succeeded(elapsedSeconds: 4.2), lastError: nil)
    XCTAssertEqual(result.successCount, 1)
    XCTAssertEqual(result.failedCount, 0)
    XCTAssertEqual(result.lastDurationMs, 4200)
    XCTAssertNil(result.lastError)
  }

  func testThrownErrorIncrementsFailedCountAndSetsLastError() async throws {
    let probe = makeQueueProbe()
    let result = await probe.finishLocalVideo(.threw, lastError: "boom")
    XCTAssertEqual(result.successCount, 0)
    XCTAssertEqual(result.failedCount, 1)
    XCTAssertEqual(result.lastError, "boom")
  }

  /// The admission-refusal path — a REAL completion (the job reached the
  /// front of the queue and was refused), not a queue-full rejection.
  func testAdmissionRefusalIncrementsFailedCountAndSetsLastError() async throws {
    let probe = makeQueueProbe()
    let message = "Insufficient memory for LTX-2 video: only 1000MB free after evicting image models (need ~65000MB)"
    let result = await probe.finishLocalVideo(.admissionRefused, lastError: message)
    XCTAssertEqual(result.successCount, 0)
    XCTAssertEqual(result.failedCount, 1)
    XCTAssertEqual(result.lastError, message)
  }

  /// All three, in sequence, on the SAME coordinator — the soak scenario the
  /// issue describes, driven through the actor for real.
  func testAllThreeOutcomesAccumulateOnTheSameCoordinator() async throws {
    let probe = makeQueueProbe()
    _ = await probe.finishLocalVideo(.admissionRefused, lastError: "refused")
    _ = await probe.finishLocalVideo(.succeeded(elapsedSeconds: 30.0), lastError: nil)
    let result = await probe.finishLocalVideo(.threw, lastError: "boom")
    XCTAssertEqual(result.successCount, 1)
    XCTAssertEqual(result.failedCount, 2)
    XCTAssertEqual(result.lastDurationMs, 30_000, "duration persists from the earlier success")
    XCTAssertEqual(result.lastError, "boom", "the latest completion's error wins")
  }

  // MARK: - comfybox#308/#322 (review r3): the generic catch must not
  // double-count an operator interrupt (bare OR wrapped) as a failed render.

  func testLocalVideoCatchOutcomeIsNilForABareCancellation() {
    XCTAssertNil(localVideoCatchOutcome(for: CancellationError()))
  }

  func testLocalVideoCatchOutcomeIsNilForTheNamedInterruptError() {
    XCTAssertNil(localVideoCatchOutcome(for: WarmServerError.renderInterrupted))
  }

  /// The exact regression: a cancellation wrapped inside a load/reload
  /// error, the shape a resume's model reload throws.
  func testLocalVideoCatchOutcomeIsNilForAWrappedCancellation() {
    XCTAssertNil(localVideoCatchOutcome(
      for: ModelPoolError.loadFailed("krea2", CancellationError())))
    XCTAssertNil(localVideoCatchOutcome(
      for: ZImageWeightsMapping.WeightApplicationError.updateFailed(
        prefix: "transformer", underlying: CancellationError())))
  }

  func testLocalVideoCatchOutcomeIsThrewForAGenuineFailure() {
    XCTAssertEqual(localVideoCatchOutcome(for: WarmServerError.invalidRequest(message: "bad dims")), .threw)
    XCTAssertEqual(localVideoCatchOutcome(for: LTX2VideoError.weightsMissing("/nowhere")), .threw)
    XCTAssertEqual(
      localVideoCatchOutcome(for: ModelPoolError.loadFailed("krea2", LTX2VideoError.unsupportedPlatform)),
      .threw,
      "a wrapper around a genuine failure is a genuine failure")
  }

  /// The seam: a WRAPPED cancellation, driven through the REAL coordinator's
  /// `handleLocalVideoCatch` (the exact function the production `.localVideo`
  /// generic catch calls) — counters and lastError must not move.
  func testSeamWrappedCancellationDoesNotIncrementFailedCount() async throws {
    let probe = makeQueueProbe()
    let result = await probe.handleLocalVideoCatch(
      ModelPoolError.loadFailed("krea2 reload on resume", CancellationError()))
    XCTAssertEqual(result.successCount, 0)
    XCTAssertEqual(result.failedCount, 0, "a wrapped cancellation must not count as a failed render")
    XCTAssertNil(result.lastError)
  }

  func testSeamBareCancellationDoesNotIncrementFailedCount() async throws {
    let probe = makeQueueProbe()
    let result = await probe.handleLocalVideoCatch(CancellationError())
    XCTAssertEqual(result.failedCount, 0)
    XCTAssertNil(result.lastError)
  }

  /// The counterpart: a genuine failure through the SAME seam still counts.
  func testSeamGenuineFailureIncrementsFailedCountAndSetsLastError() async throws {
    let probe = makeQueueProbe()
    let result = await probe.handleLocalVideoCatch(WarmServerError.invalidRequest(message: "bad dims"))
    XCTAssertEqual(result.successCount, 0)
    XCTAssertEqual(result.failedCount, 1)
    XCTAssertEqual(result.lastError, "bad dims")
  }

  /// A wrapped cancellation followed by a genuine failure on the same
  /// coordinator — only the genuine one counts.
  func testSeamMixedWrappedInterruptThenGenuineFailure() async throws {
    let probe = makeQueueProbe()
    _ = await probe.handleLocalVideoCatch(ModelPoolError.loadFailed("krea2", CancellationError()))
    let result = await probe.handleLocalVideoCatch(LTX2VideoError.weightsMissing("/nowhere"))
    XCTAssertEqual(result.failedCount, 1, "only the genuine failure counts")
  }
}
