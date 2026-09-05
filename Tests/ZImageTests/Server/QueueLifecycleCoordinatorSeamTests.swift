// QueueLifecycleCoordinatorSeamTests.swift — comfybox#283/#217: the DEBUG
// seam proof that an enqueue→admit→start→complete cycle through the REAL
// `WarmServerCoordinator` (not a mock) produces the expected lifecycle
// events. Drives `WarmServerQueueProbe`'s `.synthetic` job kind — the same
// seam `ControlPlaneTests`/`ModelOperationQueueTests` use to occupy the real
// queue without any model weights — and reads the events back through
// `WarmServerQueueProbe.lifecycleEvents(jobId:)`.

import Foundation
import XCTest

@testable import ZImage

final class QueueLifecycleCoordinatorSeamTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    // The probe drives a REAL coordinator, which persists a queue snapshot,
    // reads a pause sentinel, and (as of this instrument) appends to
    // `queue-lifecycle.jsonl` — none of which is this test's to touch.
    try isolateComfyBoxStateDirectory()
  }

  private func waitUntil(
    _ description: String, timeout: TimeInterval = 6, _ predicate: @escaping () -> Bool,
    file: StaticString = #filePath, line: UInt = #line
  ) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while !predicate() && Date() < deadline {
      try await Task.sleep(nanoseconds: 5_000_000)
    }
    XCTAssertTrue(predicate(), "waitUntil timed out: \(description)", file: file, line: line)
  }

  /// The core deliverable: a full enqueue→admit→start→complete cycle through
  /// the real coordinator, with no mock in between, produces exactly the
  /// expected event kinds in order, all under the SAME job id, all in the
  /// SAME boot.
  func testEnqueueStartCompleteCycleProducesTheExpectedEvents() async throws {
    let probe = makeQueueProbe()
    let jobId = "seam-\(UUID().uuidString)"

    let finished = try await probe.enqueueSynthetic(durationMs: 50, id: jobId)
    XCTAssertTrue(finished)

    let events = probe.lifecycleEvents(jobId: jobId)
    XCTAssertEqual(
      events.map { $0.kind }, [.enqueued, .admitted, .started, .completed],
      "a synthetic job's full lifecycle, in order, through the real coordinator")
    XCTAssertTrue(events.allSatisfy { $0.jobId == jobId })
    // One boot for the whole test process — every event shares it.
    let bootIds = Set(events.map { $0.bootId })
    XCTAssertEqual(bootIds.count, 1)
    // Sequence numbers strictly increase within this job's own history.
    let sequences = events.map { $0.sequence }
    XCTAssertEqual(sequences, sequences.sorted())
    XCTAssertEqual(Set(sequences).count, sequences.count, "no duplicate sequence numbers")
  }

  /// A job still sitting in `pending` when cancelled is `.dropped`, never
  /// `.interrupted` (that kind is reserved for a job already admitted) and
  /// never `.completed`/`.failed` (its continuation never ran).
  func testCancellingAPendingJobRecordsDroppedNotInterrupted() async throws {
    let probe = makeQueueProbe()
    let occupyingId = "seam-occupy-\(UUID().uuidString)"
    let pendingId = "seam-pending-\(UUID().uuidString)"

    async let occupying: Bool = probe.enqueueSynthetic(durationMs: 800, id: occupyingId)
    try await waitUntil("occupying job running") { probe.activeJobSummary != nil }

    async let pendingResult: Bool = probe.enqueueSynthetic(durationMs: 10, id: pendingId)
    try await waitUntil("second job parked behind the first") { probe.pendingCount >= 1 }

    let cancelled = await probe.cancelPending(id: pendingId)
    XCTAssertTrue(cancelled)

    let pendingEvents = probe.lifecycleEvents(jobId: pendingId)
    XCTAssertEqual(pendingEvents.map { $0.kind }, [.enqueued, .dropped])

    // The occupying job still finishes normally — dropping its neighbor must
    // not affect it.
    let occupyingFinished = try await occupying
    XCTAssertTrue(occupyingFinished)
    XCTAssertEqual(probe.lifecycleEvents(jobId: occupyingId).map { $0.kind }, [.enqueued, .admitted, .started, .completed])

    // The cancelled job's own continuation was resumed with `.cancelled` —
    // proven indirectly: it never reaches `.completed`, and awaiting it
    // throws (draining the `async let` so teardown's drain guard is honest).
    do {
      _ = try await pendingResult
      XCTFail("a cancelled pending job's continuation must throw")
    } catch {
      // Expected — WarmServerCoordinator.ServerError.cancelled, but that type
      // is file-private to WarmServer.swift so this test only asserts it threw.
    }
  }

  /// `/v1/queue/lifecycle`'s `limit` behavior at the ledger level: with
  /// several jobs recorded, filtering by id returns only that id's own
  /// events.
  func testLifecycleEventsCanBeFilteredByJobIdThroughTheProbe() async throws {
    let probe = makeQueueProbe()
    let idA = "seam-a-\(UUID().uuidString)"
    let idB = "seam-b-\(UUID().uuidString)"
    _ = try await probe.enqueueSynthetic(durationMs: 10, id: idA)
    _ = try await probe.enqueueSynthetic(durationMs: 10, id: idB)

    let aEvents = probe.lifecycleEvents(jobId: idA)
    let bEvents = probe.lifecycleEvents(jobId: idB)
    XCTAssertTrue(aEvents.allSatisfy { $0.jobId == idA })
    XCTAssertTrue(bEvents.allSatisfy { $0.jobId == idB })
    XCTAssertEqual(aEvents.map { $0.kind }, [.enqueued, .admitted, .started, .completed])
    XCTAssertEqual(bEvents.map { $0.kind }, [.enqueued, .admitted, .started, .completed])
  }
}
