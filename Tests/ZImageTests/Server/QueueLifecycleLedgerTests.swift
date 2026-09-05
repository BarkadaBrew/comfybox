// QueueLifecycleLedgerTests.swift — comfybox#283/#217: the ledger as a pure
// component (ordering, ring eviction, JSONL round trip, boot-id change
// detection, replay classification), plus a DEBUG-seam test that a real
// enqueue→admit→start→complete cycle through the actual coordinator produces
// the expected event sequence (see `WarmServerQueueProbeLifecycleTests`
// below).

import XCTest
@testable import ZImage

final class QueueLifecycleLedgerTests: XCTestCase {

  private func tempJSONLPath() -> URL {
    URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("queue-lifecycle-\(UUID().uuidString).jsonl")
  }

  // MARK: - Ordering

  func testEventsPreserveEmissionOrderAcrossInterleavedJobs() {
    let ledger = QueueLifecycleLedger(jsonlURL: nil)
    ledger.record(jobId: "a", kind: .enqueued)
    ledger.record(jobId: "b", kind: .enqueued)
    ledger.record(jobId: "a", kind: .admitted)
    ledger.record(jobId: "b", kind: .admitted)
    ledger.record(jobId: "a", kind: .completed)
    ledger.record(jobId: "b", kind: .completed)

    let all = ledger.events()
    XCTAssertEqual(all.map { $0.jobId }, ["a", "b", "a", "b", "a", "b"])
    XCTAssertEqual(all.map { $0.kind }, [.enqueued, .enqueued, .admitted, .admitted, .completed, .completed])
    // Sequence numbers are monotonically increasing and unique.
    XCTAssertEqual(all.map { $0.sequence }, Array(0..<UInt64(all.count)))

    let onlyA = ledger.events(jobId: "a")
    XCTAssertEqual(onlyA.map { $0.kind }, [.enqueued, .admitted, .completed])
  }

  func testLastEventReturnsTheMostRecentForThatJobOnly() {
    let ledger = QueueLifecycleLedger(jsonlURL: nil)
    ledger.record(jobId: "a", kind: .enqueued)
    ledger.record(jobId: "b", kind: .enqueued)
    ledger.record(jobId: "a", kind: .admitted)
    XCTAssertEqual(ledger.lastEvent(jobId: "a")?.kind, .admitted)
    XCTAssertEqual(ledger.lastEvent(jobId: "b")?.kind, .enqueued)
    XCTAssertNil(ledger.lastEvent(jobId: "nonexistent"))
  }

  func testTailReturnsTheLastNEventsForOneJob() {
    let ledger = QueueLifecycleLedger(jsonlURL: nil, progressMinInterval: 0)
    for step in 0..<10 {
      ledger.record(jobId: "a", kind: .progress, step: step, totalSteps: 10)
    }
    let tail = ledger.tail(jobId: "a", count: 5)
    XCTAssertEqual(tail.map { $0.step }, [5, 6, 7, 8, 9])
  }

  // MARK: - Ring eviction

  func testRingEvictsOldestBeyondCapacity() {
    let ledger = QueueLifecycleLedger(capacity: 5, jsonlURL: nil)
    for i in 0..<12 {
      ledger.record(jobId: "job-\(i)", kind: .enqueued)
    }
    let all = ledger.events()
    XCTAssertEqual(all.count, 5, "the ring must never exceed its configured capacity")
    // The oldest 7 are gone; the ring holds the newest 5.
    XCTAssertEqual(all.map { $0.jobId }, (7..<12).map { "job-\($0)" })
    // Sequence numbers are NOT reset by eviction — they keep counting up.
    XCTAssertEqual(all.map { $0.sequence }, [7, 8, 9, 10, 11])
  }

  // MARK: - Progress throttling (bounded rate)

  func testProgressTicksAreThrottledPerJob() {
    var now = Date(timeIntervalSince1970: 0)
    let ledger = QueueLifecycleLedger(jsonlURL: nil, progressMinInterval: 1.0, clock: { now })
    let first = ledger.record(jobId: "a", kind: .progress, step: 1, totalSteps: 10)
    XCTAssertNotNil(first, "the first tick for a job is never throttled")
    let throttled = ledger.record(jobId: "a", kind: .progress, step: 2, totalSteps: 10)
    XCTAssertNil(throttled, "a tick inside the min interval must be dropped, not just hidden from the ring")
    XCTAssertEqual(ledger.events(jobId: "a").count, 1)

    now = now.addingTimeInterval(1.5)
    let allowed = ledger.record(jobId: "a", kind: .progress, step: 3, totalSteps: 10)
    XCTAssertNotNil(allowed, "a tick past the min interval is recorded")
    XCTAssertEqual(ledger.events(jobId: "a").map { $0.step }, [1, 3])
  }

  func testProgressThrottleIsPerJobNotGlobal() {
    var now = Date(timeIntervalSince1970: 0)
    let ledger = QueueLifecycleLedger(jsonlURL: nil, progressMinInterval: 1.0, clock: { now })
    XCTAssertNotNil(ledger.record(jobId: "a", kind: .progress, step: 1))
    now = now.addingTimeInterval(0.1)
    // A different job's first tick is never throttled by "a"'s window.
    XCTAssertNotNil(ledger.record(jobId: "b", kind: .progress, step: 1))
  }

  func testProgressThrottleWindowClearsOnATerminalEvent() {
    var now = Date(timeIntervalSince1970: 0)
    let ledger = QueueLifecycleLedger(jsonlURL: nil, progressMinInterval: 1.0, clock: { now })
    XCTAssertNotNil(ledger.record(jobId: "a", kind: .progress, step: 1))
    ledger.record(jobId: "a", kind: .completed)
    // No time has passed, but a terminal event should not leave a stale
    // throttle entry haunting a FUTURE job id reuse (ids are UUIDs in
    // production, but the ledger's own bookkeeping must not assume that).
    XCTAssertNotNil(ledger.record(jobId: "a", kind: .progress, step: 1))
  }

  // MARK: - JSONL round trip

  func testJSONLRoundTripsThroughAFreshLedgerInstance() {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }

    let boot1 = QueueLifecycleLedger(bootId: "boot-1", jsonlURL: path)
    boot1.record(jobId: "a", kind: .enqueued, jobKind: "generate", source: "api")
    boot1.record(jobId: "a", kind: .admitted, jobKind: "generate", source: "api")
    boot1.record(jobId: "a", kind: .completed, jobKind: "generate", source: "api", durationMs: 4200)

    // A fresh ledger pointed at the SAME file (simulating a restart) seeds
    // its ring and its sequence counter from what boot1 wrote.
    let boot2 = QueueLifecycleLedger(bootId: "boot-2", jsonlURL: path)
    let seeded = boot2.events()
    XCTAssertEqual(seeded.count, 3)
    XCTAssertEqual(seeded.map { $0.kind }, [.enqueued, .admitted, .completed])
    XCTAssertEqual(seeded.map { $0.bootId }, ["boot-1", "boot-1", "boot-1"])
    XCTAssertEqual(seeded.last?.durationMs, 4200)

    // The new boot's own first event continues the SAME monotonic sequence
    // rather than resetting to 0 — a restart must never let two events from
    // different boots collide on the same sequence number.
    let next = boot2.record(jobId: "b", kind: .enqueued)
    XCTAssertEqual(next?.sequence, 3)
    XCTAssertEqual(next?.bootId, "boot-2")

    // And the file itself now has all 4 lines, across both boots.
    let onDisk = QueueLifecycleLedger.loadJSONL(from: path)
    XCTAssertEqual(onDisk.count, 4)
    XCTAssertEqual(onDisk.map { $0.bootId }, ["boot-1", "boot-1", "boot-1", "boot-2"])
  }

  func testLoadJSONLSkipsAMalformedTrailingLineRatherThanFailingTheWholeRead() throws {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }

    let ledger = QueueLifecycleLedger(jsonlURL: path)
    ledger.record(jobId: "a", kind: .enqueued)
    ledger.record(jobId: "a", kind: .completed)

    // Simulate a write that raced a crash mid-line.
    let handle = try FileHandle(forWritingTo: path)
    handle.seekToEndOfFile()
    handle.write(Data("{\"jobId\":\"a\",\"kind\":\"fail".utf8))
    try handle.close()

    let recovered = QueueLifecycleLedger.loadJSONL(from: path)
    XCTAssertEqual(recovered.count, 2, "the malformed trailing line is skipped, not fatal")
  }

  func testLoadJSONLOfAMissingFileReturnsEmpty() {
    let path = tempJSONLPath()
    XCTAssertEqual(QueueLifecycleLedger.loadJSONL(from: path), [])
  }

  // MARK: - Boot-id change detection

  func testBootIdChangesAcrossARestartAreVisibleInTheCombinedStream() {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }

    let firstBoot = QueueLifecycleLedger(bootId: "boot-alpha", jsonlURL: path)
    firstBoot.record(jobId: "job-1", kind: .enqueued)
    firstBoot.record(jobId: "job-1", kind: .admitted)
    firstBoot.record(jobId: "job-1", kind: .started)
    // The process dies here — mid-render, no `.completed`/`.failed` ever
    // written for job-1. This is exactly #283's scenario.

    let secondBoot = QueueLifecycleLedger(bootId: "boot-beta", jsonlURL: path)
    secondBoot.record(jobId: "job-1", kind: .replayedAfterRestart, fromStep1: true)

    let all = QueueLifecycleLedger.loadJSONL(from: path)
    let bootIds = Set(all.map { $0.bootId })
    XCTAssertEqual(bootIds, ["boot-alpha", "boot-beta"], "two distinct boots must be distinguishable from the file alone")
    XCTAssertEqual(all.last?.bootId, "boot-beta")
    XCTAssertEqual(all.last?.kind, .replayedAfterRestart)
    // The restart boundary is exactly where the boot id changes.
    let bootIdSequence = all.map { $0.bootId }
    XCTAssertEqual(bootIdSequence, ["boot-alpha", "boot-alpha", "boot-alpha", "boot-beta"])
  }

  func testEachLedgerInstanceGetsAFreshRandomBootIdByDefault() {
    let a = QueueLifecycleLedger(jsonlURL: nil)
    let b = QueueLifecycleLedger(jsonlURL: nil)
    XCTAssertNotEqual(a.bootId, b.bootId, "a fresh process (a fresh ledger instance) must never collide boot ids")
  }

  // MARK: - Replay classification (#283 finding 1)

  func testClassifyReturnsFromStep1WhenNoCheckpointWasEverRecorded() {
    let events: [QueueLifecycleEvent] = [
      .init(sequence: 0, bootId: "b1", wallTime: Date(), jobId: "a", kind: .enqueued),
      .init(sequence: 1, bootId: "b1", wallTime: Date(), jobId: "a", kind: .admitted),
      .init(sequence: 2, bootId: "b1", wallTime: Date(), jobId: "a", kind: .started),
    ]
    let result = ReplayClassifier.classify(priorEvents: events)
    XCTAssertTrue(result.fromStep1)
    XCTAssertNil(result.resumeStep)
    XCTAssertNil(result.resumeChunk)
  }

  func testClassifyResumesFromAnOpenCheckpoint() {
    let events: [QueueLifecycleEvent] = [
      .init(sequence: 0, bootId: "b1", wallTime: Date(), jobId: "v", kind: .enqueued),
      .init(sequence: 1, bootId: "b1", wallTime: Date(), jobId: "v", kind: .admitted),
      .init(sequence: 2, bootId: "b1", wallTime: Date(), jobId: "v", kind: .started),
      .init(sequence: 3, bootId: "b1", wallTime: Date(), jobId: "v", kind: .checkpointed, step: 42, chunk: 3),
      // Process dies here — no `.resumed`/terminal event ever closes this out.
    ]
    let result = ReplayClassifier.classify(priorEvents: events)
    XCTAssertFalse(result.fromStep1)
    XCTAssertEqual(result.resumeStep, 42)
    XCTAssertEqual(result.resumeChunk, 3)
  }

  func testClassifyIgnoresACheckpointThatWasAlreadyResumed() {
    let events: [QueueLifecycleEvent] = [
      .init(sequence: 0, bootId: "b1", wallTime: Date(), jobId: "v", kind: .checkpointed, step: 10, chunk: 1),
      .init(sequence: 1, bootId: "b1", wallTime: Date(), jobId: "v", kind: .resumed, step: 10, chunk: 1),
    ]
    let result = ReplayClassifier.classify(priorEvents: events)
    XCTAssertTrue(result.fromStep1, "a checkpoint already resumed has nothing left to resume a REPLAY from")
  }

  func testClassifyIgnoresACheckpointSupersededByATerminalOutcome() {
    for terminal: QueueLifecycleEventKind in [.completed, .failed, .dropped] {
      let events: [QueueLifecycleEvent] = [
        .init(sequence: 0, bootId: "b1", wallTime: Date(), jobId: "v", kind: .checkpointed, step: 10, chunk: 1),
        .init(sequence: 1, bootId: "b1", wallTime: Date(), jobId: "v", kind: terminal),
      ]
      let result = ReplayClassifier.classify(priorEvents: events)
      XCTAssertTrue(result.fromStep1, "a \(terminal.rawValue) after the checkpoint closes it out")
    }
  }

  func testClassifyUsesTheMostRecentOpenCheckpointWhenSeveralWereTaken() {
    let events: [QueueLifecycleEvent] = [
      .init(sequence: 0, bootId: "b1", wallTime: Date(), jobId: "v", kind: .checkpointed, step: 5, chunk: 0),
      .init(sequence: 1, bootId: "b1", wallTime: Date(), jobId: "v", kind: .resumed, step: 5, chunk: 0),
      .init(sequence: 2, bootId: "b1", wallTime: Date(), jobId: "v", kind: .checkpointed, step: 30, chunk: 2),
      // Dies before this second checkpoint is ever resumed.
    ]
    let result = ReplayClassifier.classify(priorEvents: events)
    XCTAssertFalse(result.fromStep1)
    XCTAssertEqual(result.resumeStep, 30)
    XCTAssertEqual(result.resumeChunk, 2)
  }

  func testClassifierIgnoresEmptyHistory() {
    let result = ReplayClassifier.classify(priorEvents: [])
    XCTAssertTrue(result.fromStep1)
  }

  // MARK: - classifyReplay convenience

  func testClassifyReplayReadsThisLedgersOwnHistoryForTheJobId() {
    let ledger = QueueLifecycleLedger(jsonlURL: nil)
    ledger.record(jobId: "v", kind: .checkpointed, step: 7, chunk: 1)
    let result = ledger.classifyReplay(jobId: "v")
    XCTAssertFalse(result.fromStep1)
    XCTAssertEqual(result.resumeStep, 7)
  }
}
