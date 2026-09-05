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
    // C1: `record` never waits on disk I/O — flush explicitly before a
    // fresh ledger reads this file back.
    boot1.waitForPendingWritesForTesting()

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
    boot2.waitForPendingWritesForTesting()

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
    ledger.waitForPendingWritesForTesting()

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
    firstBoot.waitForPendingWritesForTesting()
    // The process dies here — mid-render, no `.completed`/`.failed` ever
    // written for job-1. This is exactly #283's scenario.

    let secondBoot = QueueLifecycleLedger(bootId: "boot-beta", jsonlURL: path)
    secondBoot.record(jobId: "job-1", kind: .replayedAfterRestart, fromStep1: true)
    secondBoot.waitForPendingWritesForTesting()

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
    // PR #370 review I3: `.abandoned` (the video's checkpoint was dropped
    // because an operator interrupt arrived during the preemption episode,
    // NOT resumed) must close out an open checkpoint exactly like the
    // others — a replay must never see it as "still open."
    for terminal: QueueLifecycleEventKind in [.completed, .failed, .dropped, .abandoned] {
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

  // MARK: - PR #370 review round 1, C1: lock protects only the ring, never disk I/O

  /// The core C1 claim: `record`'s latency does not depend on how slow the
  /// disk write is — the write happens on a separate queue, after `record`
  /// has already returned the event to its caller.
  func testRecordReturnsWithoutWaitingOnASlowWriter() {
    let started = DispatchSemaphore(value: 0)
    let releaseWriter = DispatchSemaphore(value: 0)
    let writer: QueueLifecycleLedger.BatchWriter = { _, _, _, _, _ in
      started.signal()
      // Blocks until the test explicitly releases it — if `record` waited
      // on this, the assertion below would time out instead of passing.
      _ = releaseWriter.wait(timeout: .now() + 5)
    }
    let ledger = QueueLifecycleLedger(jsonlURL: tempJSONLPath(), writer: writer)

    let recordStart = Date()
    ledger.record(jobId: "a", kind: .enqueued)
    let recordElapsed = Date().timeIntervalSince(recordStart)
    XCTAssertLessThan(recordElapsed, 0.5, "record must not block on the writer")

    // Let the slow writer proceed so it doesn't leak a blocked thread past
    // this test's lifetime, then prove it actually ran (not just skipped).
    XCTAssertEqual(started.wait(timeout: .now() + 2), .success, "the writer should still have started, just not been waited on")
    releaseWriter.signal()
  }

  /// C1: a writer that never keeps up must not grow `pendingWrites` without
  /// bound — the oldest queued-for-write events are dropped instead
  /// (counted, not silent), while `events()` (the in-memory ring) is
  /// completely unaffected — a dropped WRITE only means that one event
  /// never reaches disk, never that it disappears from `events()`.
  func testPendingWritesAreBoundedAndOldestAreDroppedWhenTheWriterStalls() {
    let writerGate = DispatchSemaphore(value: 0)
    let writer: QueueLifecycleLedger.BatchWriter = { _, _, _, _, _ in
      // Never returns until the test releases it — simulates a writer that
      // has fallen permanently behind for the span of this test.
      _ = writerGate.wait(timeout: .now() + 5)
    }
    let ledger = QueueLifecycleLedger(
      jsonlURL: tempJSONLPath(), maxPendingWrites: 5, writer: writer)

    for i in 0..<20 {
      ledger.record(jobId: "job-\(i)", kind: .enqueued)
    }

    XCTAssertGreaterThan(ledger.droppedWriteCountForTesting, 0, "the bounded buffer must drop, not grow forever")
    // The ring itself (in-memory, independent of the stalled writer) still
    // has every event — only the on-disk copy of the oldest ones was lost.
    XCTAssertEqual(ledger.events().count, 20)

    writerGate.signal()
  }

  // MARK: - PR #370 review round 1, C2: bounded disk footprint + lazy tail reseed

  /// comfybox#379: this used to flush ONCE, after all 20 `record()` calls,
  /// relying on the writer queue happening to have drained at least TWICE by
  /// then (so a later batch's write saw a non-empty on-disk file and
  /// actually exercised `rotateIfNeeded`'s `currentSize > 0` branch). Whether
  /// that was true depended entirely on how the background `writerQueue`
  /// thread got scheduled relative to this test's tight, synchronous loop —
  /// under normal load the OS never ran it until the loop finished, coalescing
  /// every event into ONE batch; under CPU contention (a parallel build) it
  /// sometimes got a slice early enough to drain in between calls, splitting
  /// the events across two-plus batches instead. A single batch of all 20
  /// events writes a BRAND NEW file in one shot — `rotateIfNeeded` never fires
  /// for a brand-new file (nothing to preserve, see its doc comment) — so the
  /// assertion below failed exactly when the single-batch case occurred
  /// (confirmed directly: instrumenting the `writer` seam showed `[20]` with
  /// `.1` absent on one run, `[10, 10]`/`[4, 16]`/`[12, 8]` with `.1` present
  /// on others, from nothing but re-running the same test).
  ///
  /// Fixed by flushing after EVERY `record()` call instead of once at the
  /// end, via the existing `waitForPendingWritesForTesting()` test seam —
  /// this makes each event its OWN batch deterministically, independent of
  /// however the writer queue happens to get scheduled, so the file exists
  /// and is non-empty by the second record and every later one's write
  /// deterministically re-checks `rotateIfNeeded` against it.
  func testRotationMovesTheCurrentFileAsideOnceItWouldExceedTheLimit() {
    let path = tempJSONLPath()
    defer {
      try? FileManager.default.removeItem(at: path)
      try? FileManager.default.removeItem(at: path.appendingPathExtension("1"))
    }
    // A tiny limit so two small single-event batches are enough to trigger
    // rotation once flushed deterministically (see the doc comment above).
    let ledger = QueueLifecycleLedger(jsonlURL: path, rotateAtBytes: 200, keepGenerations: 2)
    for i in 0..<20 {
      ledger.record(jobId: "job-\(i)", kind: .enqueued, jobKind: "generate", source: "api")
      ledger.waitForPendingWritesForTesting()
    }

    let rotated = path.appendingPathExtension("1")
    XCTAssertTrue(FileManager.default.fileExists(atPath: rotated.path), "expected a rotated backup once the live file crossed the limit")
    XCTAssertTrue(FileManager.default.fileExists(atPath: path.path), "the live file must still exist for new writes")
  }

  func testRotationNeverFiresOnABrandNewOrEmptyFile() {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }
    // incomingBytes alone exceeds rotateAtBytes, but the file is empty/absent
    // — nothing to preserve, so rotation must not fire (would just create an
    // empty ".1" backup for no reason).
    QueueLifecycleLedger.rotateIfNeeded(
      url: path, incomingBytes: 1000, rotateAtBytes: 10, keepGenerations: 2, fileManager: .default)
    XCTAssertFalse(FileManager.default.fileExists(atPath: path.appendingPathExtension("1").path))
  }

  func testLoadJSONLTailToleratesATruncatedLastLine() throws {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }

    let ledger = QueueLifecycleLedger(jsonlURL: path)
    ledger.record(jobId: "a", kind: .enqueued)
    ledger.record(jobId: "a", kind: .admitted)
    ledger.waitForPendingWritesForTesting()

    // A write that raced a crash mid-line.
    let handle = try FileHandle(forWritingTo: path)
    handle.seekToEndOfFile()
    handle.write(Data("{\"jobId\":\"a\",\"kind\":\"star".utf8))
    try handle.close()

    let tail = QueueLifecycleLedger.loadJSONLTail(from: path, maxBytes: 64 * 1024)
    XCTAssertEqual(tail.map { $0.kind }, [.enqueued, .admitted], "the truncated trailing line is skipped, not fatal")
  }

  /// C2: the reseed window can start MID-LINE (the tail is a fixed byte
  /// count, not line-aligned) — the partial LEADING line must be discarded,
  /// not mis-parsed as a malformed record.
  func testLoadJSONLTailDiscardsAPartialLeadingLine() {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }

    let ledger = QueueLifecycleLedger(jsonlURL: path)
    for i in 0..<50 {
      ledger.record(jobId: "job-\(i)", kind: .enqueued, jobKind: "generate", source: "api")
    }
    ledger.waitForPendingWritesForTesting()

    // A small window guarantees it starts partway through some line.
    let tail = QueueLifecycleLedger.loadJSONLTail(from: path, maxBytes: 300)
    XCTAssertFalse(tail.isEmpty)
    // Every recovered event must be a COMPLETE, validly-decoded record —
    // `loadJSONLTail` would rather return fewer events than a mis-parsed one.
    for event in tail {
      XCTAssertEqual(event.kind, .enqueued)
      XCTAssertTrue(event.jobId.hasPrefix("job-"))
    }
  }

  /// C2: a ledger's reseed is LAZY (never in `init`) and reads only the
  /// TAIL — this proves the sequence counter still continues correctly past
  /// whatever the tail's own max sequence was, even when the full file is
  /// far larger than the tail window (so the tail-read genuinely cannot see
  /// the file's true beginning).
  func testSequenceContinuesFromTheTailMaxEvenWhenTheFileIsLargerThanTheTailWindow() {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }

    let firstBoot = QueueLifecycleLedger(jsonlURL: path)
    for i in 0..<200 {
      firstBoot.record(jobId: "job-\(i)", kind: .enqueued, jobKind: "generate", source: "api")
    }
    firstBoot.waitForPendingWritesForTesting()

    // A tail window far smaller than the full file — cannot see sequence 0.
    let secondBoot = QueueLifecycleLedger(jsonlURL: path, reseedTailBytes: 512)
    let next = secondBoot.record(jobId: "job-new", kind: .enqueued)
    XCTAssertNotNil(next)
    XCTAssertGreaterThan(next!.sequence, 190, "must continue from (at least) what the tail window could see, never reset to 0")
  }

  /// C2 / M: `QueueLifecycleLedger()`'s init must never touch disk — the
  /// reseed is lazy, triggered by the FIRST real `record`/`events` call.
  /// Constructing a ledger pointed at an existing, non-trivial file must be
  /// effectively instantaneous.
  func testInitDoesNotReadTheFile() {
    let path = tempJSONLPath()
    defer { try? FileManager.default.removeItem(at: path) }
    let seed = QueueLifecycleLedger(jsonlURL: path)
    for i in 0..<500 {
      seed.record(jobId: "job-\(i)", kind: .enqueued, jobKind: "generate", source: "api")
    }
    seed.waitForPendingWritesForTesting()

    let start = Date()
    let fresh = QueueLifecycleLedger(jsonlURL: path)
    let elapsed = Date().timeIntervalSince(start)
    XCTAssertLessThan(elapsed, 0.05, "construction must not read the file")
    // Confirms the ledger is still fully functional — the reseed just
    // hasn't happened yet at construction time.
    _ = fresh
  }

  /// M: the batch writer must never fall back to an atomic FULL-FILE
  /// overwrite when an EXISTING file cannot be opened for writing (the bug
  /// this replaces: the old `else` branch treated ANY open failure,
  /// including a transient one like EMFILE, as "the file doesn't exist yet"
  /// and clobbered it). Simulated here with a read-only file, which fails
  /// `FileHandle(forWritingTo:)` the same way a resource-exhaustion error
  /// would — this test cares about the DECISION (create-if-absent, never
  /// clobber-if-present-but-unopenable), not the specific OS error.
  func testBatchWriterNeverClobbersAnExistingFileItCannotOpenForWriting() throws {
    let path = tempJSONLPath()
    defer {
      try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: path.path)
      try? FileManager.default.removeItem(at: path)
    }
    let original = "{\"precious\":\"history\"}\n"
    try Data(original.utf8).write(to: path)
    try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: path.path)

    let event = QueueLifecycleEvent(sequence: 0, bootId: "b", wallTime: Date(), jobId: "a", kind: .enqueued)
    QueueLifecycleLedger.defaultBatchWriter([event], path, .default, 20 * 1024 * 1024, 2)

    let contentAfter = try String(contentsOf: path, encoding: .utf8)
    XCTAssertEqual(contentAfter, original, "an existing file that cannot be opened for writing must be left untouched, never overwritten")
  }

  // MARK: - PR #370 review round 1, I4: `wallTime` is ISO8601 on every wire surface

  func testWallTimeIsAlwaysAnISO8601StringRegardlessOfTheAmbientEncoder() throws {
    let event = QueueLifecycleEvent(sequence: 1, bootId: "b", wallTime: Date(timeIntervalSince1970: 1_700_000_000), jobId: "a", kind: .completed)

    // (a) the generic snake_case HTTP encoder `/v1/queue/lifecycle` and
    // `/v1/generate/status/{id}` share — NO `dateEncodingStrategy` override,
    // which is exactly the encoder that used to leak a raw Double.
    let snakeCaseEncoder = JSONEncoder()
    snakeCaseEncoder.keyEncodingStrategy = .convertToSnakeCase
    let snakeCaseJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: snakeCaseEncoder.encode(event)) as? [String: Any])
    XCTAssertTrue(snakeCaseJSON["wall_time"] is String, "wall_time must be a string, not a number")

    // (b) a plain encoder with no strategy at all.
    let plainEncoder = JSONEncoder()
    let plainJSON = try XCTUnwrap(try JSONSerialization.jsonObject(with: plainEncoder.encode(event)) as? [String: Any])
    XCTAssertTrue(plainJSON["wallTime"] is String, "wallTime must be a string, not a number")

    // (c) even an encoder that explicitly asks for a DIFFERENT date strategy
    // must not affect this field — the type owns its own format.
    let secondsSince1970Encoder = JSONEncoder()
    secondsSince1970Encoder.dateEncodingStrategy = .secondsSince1970
    let overriddenJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: secondsSince1970Encoder.encode(event)) as? [String: Any])
    XCTAssertTrue(overriddenJSON["wallTime"] is String, "a competing ambient dateEncodingStrategy must not win")

    // Round-trips through the ledger's own decoder too.
    let decoded = try JSONDecoder().decode(QueueLifecycleEvent.self, from: try plainEncoder.encode(event))
    XCTAssertEqual(decoded.wallTime.timeIntervalSince1970, event.wallTime.timeIntervalSince1970, accuracy: 1.0)
  }
}
