import XCTest
@testable import ZImage

/// WP-E10 "E9b" (FDD D22 / AC-18; Addendum A.2, E4 review MAJOR): the id a
/// `POST /v1/generate/async` caller receives must be THE id the queue
/// persists and replays under — `recordFailedReplay` used to key on the
/// coordinator's private `PendingJob.id`, a second UUID the caller never saw,
/// so a job that failed replay after a restart was recorded under a name
/// nobody could poll. The tracker now hands its own id to the enqueue seam.
final class AsyncJobIdTests: XCTestCase {

  private final class Captured: @unchecked Sendable {
    private let lock = NSLock()
    private var ids: [String] = []
    func record(_ id: String) { lock.lock(); ids.append(id); lock.unlock() }
    var all: [String] { lock.lock(); defer { lock.unlock() }; return ids }
  }

  private func waitUntilDone(_ tracker: ImageJobTracker, _ jobId: String) -> ImageJobStatus? {
    for _ in 0..<200 {
      if let s = tracker.status(jobId: jobId), s.status == .succeeded || s.status == .failed { return s }
      Thread.sleep(forTimeInterval: 0.01)
    }
    return tracker.status(jobId: jobId)
  }

  func testSubmitHandsItsOwnIdToTheQueue() throws {
    let tracker = ImageJobTracker()
    let captured = Captured()
    let payload = GeneratePayload(prompt: "x")
    let status = tracker.submit(payload, source: "api", rawBody: nil) { jobId in
      captured.record(jobId)
      return GenerateResponse(success: true, outputPath: "/tmp/x.png", durationMs: 1)
    }
    // Not `.queued`: the detached worker may already have flipped it to
    // `.processing` by the time submit returns. The claim under test is the
    // id, and that it is pollable from the moment the caller has it.
    XCTAssertFalse(status.jobId.isEmpty)
    XCTAssertNotNil(tracker.status(jobId: status.jobId))
    let done = try XCTUnwrap(waitUntilDone(tracker, status.jobId))
    XCTAssertEqual(done.status, .succeeded)
    XCTAssertEqual(captured.all, [status.jobId], "the queue saw exactly the client-visible id")
  }

  /// The preempting submit path is the same seam: whatever the preemptor
  /// decides, the enqueue (if it happens) carries the client-visible id.
  func testSubmitPreemptingHandsItsOwnIdToTheQueueAndThePreemptor() throws {
    let tracker = ImageJobTracker()
    let enqueued = Captured()
    let preempted = Captured()
    let payload = GeneratePayload(prompt: "x")
    let status = tracker.submitPreempting(
      payload, source: "api", rawBody: nil,
      preemptor: { jobId in preempted.record(jobId); return .refused(eta: 12) },
      enqueue: { jobId in
        enqueued.record(jobId)
        return GenerateResponse(success: true, outputPath: "/tmp/x.png", durationMs: 1)
      })
    let done = try XCTUnwrap(waitUntilDone(tracker, status.jobId))
    XCTAssertEqual(done.status, .succeeded)
    XCTAssertEqual(done.preemptRefused, true)
    XCTAssertEqual(done.etaSec, 12)
    XCTAssertEqual(preempted.all, [status.jobId])
    XCTAssertEqual(enqueued.all, [status.jobId])
  }

  /// A failed replay recorded under that id is what the status route reports.
  func testFailedReplayIsQueryableUnderTheClientVisibleId() throws {
    let tracker = ImageJobTracker()
    let clientVisible = "A1B2C3D4-CLIENT-VISIBLE"
    tracker.recordFailedReplay(
      jobId: clientVisible, source: "bree",
      error: WarmServerError.unknownSampler(name: "uni_pc", valid: ["euler"]))
    let status = try XCTUnwrap(tracker.status(jobId: clientVisible))
    XCTAssertEqual(status.jobId, clientVisible)
    XCTAssertEqual(status.status, .failed)
    XCTAssertTrue(try XCTUnwrap(status.error).contains("uni_pc"))
  }
}
