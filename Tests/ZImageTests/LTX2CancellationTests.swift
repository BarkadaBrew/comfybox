import XCTest
@testable import ZImage

/// comfybox#322 — mid-render cancellation for the LTX-2 video path.
///
/// Before this the whole LTX-2 render had zero `Task.checkCancellation()`
/// sites, so `/v1/queue/interrupt` could only stop the NEXT queue item: a bad
/// 30-minute clip had to burn to completion. The image path was fixed the same
/// way in comfybox#304 (`Krea2DenoiseLoop.run`, `ChromaPipeline.denoise`) and
/// these tests mirror `Krea2DenoiseLoopTests`' two cancellation cases.
///
/// LTX-2 needs one thing the image path does not: its loops ALSO carry the
/// #1479 preemption signal, and the two mechanisms mean opposite things — a
/// yield parks a checkpoint so the render RESUMES, a cancel abandons it. The
/// ordering (cancel first) is therefore a correctness property, which is why
/// it lives in `LTX2LoopBoundary` as a pure function rather than being spelled
/// out at each of the ~8 boundaries. No weights, no pipeline, no GPU.
final class LTX2CancellationTests: XCTestCase {

  // MARK: - The pure boundary decision

  func testProceedsWhenNeitherCancelledNorRaised() throws {
    XCTAssertEqual(
      try LTX2LoopBoundary.decide(cancelled: false, preemptionRaised: false), .proceed)
  }

  func testYieldsWhenPreemptionRaised() throws {
    XCTAssertEqual(
      try LTX2LoopBoundary.decide(cancelled: false, preemptionRaised: true), .yield)
  }

  func testCancellationThrowsWithNoSignalRaised() {
    XCTAssertThrowsError(
      try LTX2LoopBoundary.decide(cancelled: true, preemptionRaised: false)
    ) { XCTAssertTrue($0 is CancellationError, "must be CancellationError, unmodified") }
  }

  /// The load-bearing case from the issue: "a cancel during a preemption
  /// handoff should cancel cleanly, not resume." If the yield won here, an
  /// interrupted render would bank a checkpoint and the coordinator would
  /// dutifully resume the very clip the operator asked to kill.
  func testCancellationBeatsAPreemptionYield() {
    XCTAssertThrowsError(
      try LTX2LoopBoundary.decide(cancelled: true, preemptionRaised: true)
    ) { XCTAssertTrue($0 is CancellationError, "cancel must win over yield") }
  }

  // MARK: - The production entry, under a real Task

  /// `decide(preemption:)` reads the AMBIENT task's cancellation flag — the
  /// same read every LTX-2 loop makes. A task cancelled before it starts must
  /// throw at its first boundary, having done no work at all.
  func testAmbientTaskCancelledBeforeStartThrowsAtFirstBoundary() async {
    var steps = 0
    let signal = PreemptionSignal()
    let task = Task<Void, Error> {
      for _ in 0..<10 {
        _ = try LTX2LoopBoundary.decide(preemption: signal)
        steps += 1
      }
    }
    task.cancel()
    do {
      _ = try await task.value
      XCTFail("expected CancellationError")
    } catch is CancellationError {
      XCTAssertEqual(steps, 0, "cancelled-before-start must not run a single step")
    } catch {
      XCTFail("expected CancellationError, got \(error)")
    }
  }

  /// The abort-latency claim, stated as a test: a token that flips after N
  /// steps stops the loop within ONE step — not at the end of the run.
  ///
  /// N = 3 of 10 stands in for the production shape (an interrupt at ~30% of a
  /// 10-step LTX-2 stage): the loop must have executed exactly 3 bodies and
  /// thrown at the top of the 4th.
  func testCancellationMidLoopStopsWithinOneStep() async {
    var steps = 0
    let signal = PreemptionSignal()
    var capture: Task<Void, Error>?
    let task = Task<Void, Error> {
      for _ in 0..<10 {
        _ = try LTX2LoopBoundary.decide(preemption: signal)
        steps += 1
        if steps == 3 { capture?.cancel() }
      }
    }
    capture = task
    do {
      _ = try await task.value
      XCTFail("expected CancellationError after step 3 of 10")
    } catch is CancellationError {
      XCTAssertEqual(
        steps, 3,
        "loop must stop at the boundary right after cancellation, not run to completion")
    } catch {
      XCTFail("expected CancellationError, got \(error)")
    }
  }

  /// Same loop, but the #1479 signal is raised at the same moment the task is
  /// cancelled — the preemption-handoff race. The loop must throw, never take
  /// the `.yield` branch that banks a resumable checkpoint.
  func testCancellationDuringPreemptionHandoffThrowsRatherThanYielding() async {
    var steps = 0
    var yielded = false
    let signal = PreemptionSignal()
    var capture: Task<Void, Error>?
    let task = Task<Void, Error> {
      for _ in 0..<10 {
        if try LTX2LoopBoundary.decide(preemption: signal) == .yield {
          yielded = true
          return
        }
        steps += 1
        if steps == 3 {
          signal.raise()      // a preempting image job arrives…
          capture?.cancel()   // …and the operator interrupts in the same window
        }
      }
    }
    capture = task
    do {
      _ = try await task.value
      XCTFail("expected CancellationError")
    } catch is CancellationError {
      XCTAssertFalse(yielded, "must NOT bank a checkpoint the coordinator would resume")
      XCTAssertEqual(steps, 3)
    } catch {
      XCTFail("expected CancellationError, got \(error)")
    }
  }

  /// A raised signal with NO cancellation still yields — the #1479 preemption
  /// path is unchanged by this fix.
  func testRaisedSignalAloneStillYields() async throws {
    var steps = 0
    var yielded = false
    let signal = PreemptionSignal()
    let task = Task<Void, Error> {
      for _ in 0..<10 {
        if try LTX2LoopBoundary.decide(preemption: signal) == .yield {
          yielded = true
          return
        }
        steps += 1
        if steps == 3 { signal.raise() }
      }
    }
    try await task.value
    XCTAssertTrue(yielded, "preemption must still checkpoint when nothing is cancelled")
    XCTAssertEqual(steps, 3)
  }

  /// A nil signal is the non-preemptible `generate()` path (chunk loop, decode,
  /// audio). It must never yield, and must still honour cancellation.
  func testNilSignalNeverYieldsButStillCancels() async {
    var steps = 0
    var capture: Task<Void, Error>?
    let task = Task<Void, Error> {
      for _ in 0..<10 {
        // NOT XCTAssertEqual(try …): the autoclosure catches the throw and
        // records it as a failure instead of letting it out of the loop.
        let decision = try LTX2LoopBoundary.decide(preemption: nil)
        XCTAssertEqual(decision, .proceed)
        steps += 1
        if steps == 2 { capture?.cancel() }
      }
    }
    capture = task
    do {
      _ = try await task.value
      XCTFail("expected CancellationError")
    } catch is CancellationError {
      XCTAssertEqual(steps, 2)
    } catch {
      XCTFail("expected CancellationError, got \(error)")
    }
  }

  // MARK: - Interrupt vs. failure classification

  /// Both spellings of an interrupt must be recognised: the raw
  /// `CancellationError` a pipeline loop throws (the #304 propagate-unmodified
  /// contract) and the named `WarmServerError.renderInterrupted` the video
  /// queue case substitutes so the client sees a sentence.
  func testInterruptClassification() {
    XCTAssertTrue(isRenderInterruption(CancellationError()))
    XCTAssertTrue(isRenderInterruption(WarmServerError.renderInterrupted))
    XCTAssertFalse(isRenderInterruption(WarmServerError.invalidRequest(message: "nope")))
    XCTAssertFalse(isRenderInterruption(LTX2VideoError.weightsMissing("/nowhere")))
    XCTAssertFalse(
      isRenderInterruption(LTX2ResumeError.stepOutOfRange(step: 9, steps: 4)),
      "a refused resume is a real failure, not an interrupt")
  }

  // MARK: - Job reporting

  /// An interrupted render is terminal but NOT a failure. `status` stays
  /// `failed` so every existing polling client still sees a terminal state it
  /// knows (Desktop's `isTerminal`, the daemon's decoder); the additive
  /// `interrupted` flag and a plain-English `error` carry the real outcome.
  func testInterruptedRenderReportsInterruptedNotFailed() {
    let tracker = VideoJobTracker()
    let (jobId, queued) = tracker.register(source: "test", mode: .t2v)
    XCTAssertNil(queued.interrupted, "a queued job carries no interrupted flag")

    tracker.markProcessing(jobId)
    tracker.markFailed(jobId, error: CancellationError())

    let status = tracker.status(jobId: jobId)
    XCTAssertEqual(status?.interrupted, true, "additive flag must say interrupted")
    XCTAssertEqual(status?.status, .failed, "status stays a state existing clients know")
    XCTAssertEqual(status?.error, "Render interrupted by /v1/queue/interrupt")
    XCTAssertNil(status?.outputPath, "an interrupted render produces no clip")
  }

  /// The named error takes the same route — the video queue case substitutes
  /// it for `CancellationError` before the continuation sees it.
  func testNamedInterruptErrorReportsInterrupted() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "test", mode: .i2v)
    tracker.markFailed(jobId, error: WarmServerError.renderInterrupted)
    XCTAssertEqual(tracker.status(jobId: jobId)?.interrupted, true)
  }

  /// A real failure is untouched by this change: no flag, and the original
  /// message survives.
  func testGenuineFailureStillReportsFailed() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "test", mode: .t2v)
    tracker.markFailed(jobId, error: WarmServerError.invalidRequest(message: "bad dims"))
    let status = tracker.status(jobId: jobId)
    XCTAssertEqual(status?.status, .failed)
    XCTAssertNil(status?.interrupted, "a genuine failure must not be relabelled")
    XCTAssertEqual(status?.error, "bad dims")
  }

  /// The additive field is absent from the JSON on every non-interrupted
  /// outcome, so nothing changes on the wire for existing clients.
  func testInterruptedFieldIsAbsentUnlessInterrupted() throws {
    let encoder = JSONEncoder()
    let plain = VideoJobStatus(jobId: "j", status: .processing)
    let json = String(data: try encoder.encode(plain), encoding: .utf8) ?? ""
    XCTAssertFalse(json.contains("interrupted"), "field must be omitted when nil: \(json)")

    let stopped = VideoJobStatus(jobId: "j", status: .failed, interrupted: true)
    let stoppedJSON = String(data: try encoder.encode(stopped), encoding: .utf8) ?? ""
    XCTAssertTrue(stoppedJSON.contains("\"interrupted\""), stoppedJSON)
  }
}
