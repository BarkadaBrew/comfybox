import XCTest
@testable import ZImage

/// Unit tests for the async LOCAL LTX-2 video job tracker (Phase A of the
/// JoyAI-Echo port). The state machine is exercised in isolation — no
/// coordinator, no real render — via the tracker's transition surface, exactly
/// how the production `submit(...)` drives it. Mirrors the image job pattern.
final class VideoJobTrackerTests: XCTestCase {

  private func result(
    path: String = "/out/ltx2.mp4", frames: Int = 97,
    duration: Float = 4.0, elapsed: Double = 12.5
  ) -> LTX2VideoResult {
    LTX2VideoResult(
      outputPath: path, frameCount: frames,
      durationSeconds: duration, elapsedSeconds: elapsed)
  }

  // MARK: - Unknown id

  func testUnknownIdReturnsNil() {
    let tracker = VideoJobTracker()
    XCTAssertNil(tracker.status(jobId: "does-not-exist"))
    XCTAssertNil(tracker.status(jobId: UUID().uuidString))
  }

  // MARK: - submit → queued

  func testRegisterStartsQueued() {
    let tracker = VideoJobTracker()
    let (jobId, status) = tracker.register(source: "desktop", mode: .t2v)

    XCTAssertFalse(jobId.isEmpty)
    XCTAssertEqual(status.jobId, jobId)
    XCTAssertEqual(status.status, .queued)
    XCTAssertEqual(status.backend, "ltx2-local")
    XCTAssertEqual(status.mode, .t2v)
    XCTAssertNil(status.outputPath)
    XCTAssertNil(status.error)
    XCTAssertNil(status.progressPercent)
    XCTAssertGreaterThanOrEqual(status.elapsedMs ?? -1, 0)

    // Round-trips through status(jobId:).
    XCTAssertEqual(tracker.status(jobId: jobId)?.status, .queued)
  }

  func testModeIsPreserved() {
    let tracker = VideoJobTracker()
    let (i2vId, _) = tracker.register(source: "api", mode: .i2v)
    XCTAssertEqual(tracker.status(jobId: i2vId)?.mode, .i2v)
  }

  // MARK: - queued → processing

  func testMarkProcessing() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markProcessing(jobId)
    XCTAssertEqual(tracker.status(jobId: jobId)?.status, .processing)
  }

  // MARK: - progress updates

  func testProgressUpdatesAndClamps() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markProcessing(jobId)

    tracker.setProgress(jobId, 42)
    XCTAssertEqual(tracker.status(jobId: jobId)?.progressPercent, 42)

    tracker.setProgress(jobId, 150)   // clamps high
    XCTAssertEqual(tracker.status(jobId: jobId)?.progressPercent, 100)

    tracker.setProgress(jobId, -5)    // clamps low
    XCTAssertEqual(tracker.status(jobId: jobId)?.progressPercent, 0)
  }

  func testSetProgressOnUnknownIdIsNoOp() {
    let tracker = VideoJobTracker()
    tracker.setProgress("nope", 50)   // must not crash
    XCTAssertNil(tracker.status(jobId: "nope"))
  }

  // MARK: - processing → succeeded

  func testMarkSucceeded() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .i2v)
    tracker.markProcessing(jobId)
    tracker.setProgress(jobId, 60)

    tracker.markSucceeded(jobId, result: result(path: "/out/final.mp4", duration: 6.0, elapsed: 30.0))

    let s = tracker.status(jobId: jobId)
    XCTAssertEqual(s?.status, .succeeded)
    XCTAssertEqual(s?.outputPath, "/out/final.mp4")
    XCTAssertEqual(s?.videoDurationSeconds, 6)
    XCTAssertEqual(s?.durationMs, 30_000)
    // Completion forces progress to 100 regardless of the last streamed value.
    XCTAssertEqual(s?.progressPercent, 100)
    XCTAssertNil(s?.error)
  }

  func testElapsedFreezesAfterCompletion() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markSucceeded(jobId, result: result())
    let first = tracker.status(jobId: jobId)?.elapsedMs
    Thread.sleep(forTimeInterval: 0.02)
    let second = tracker.status(jobId: jobId)?.elapsedMs
    XCTAssertEqual(first, second, "elapsedMs must freeze once the job completes")
  }

  // MARK: - processing → failed

  func testMarkFailed() {
    struct Boom: LocalizedError { var errorDescription: String? { "render blew up" } }
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markProcessing(jobId)

    tracker.markFailed(jobId, error: Boom())

    let s = tracker.status(jobId: jobId)
    XCTAssertEqual(s?.status, .failed)
    XCTAssertEqual(s?.error, "render blew up")
    XCTAssertNil(s?.outputPath)
  }

  // MARK: - prune

  func testPruneRemovesCompletedButKeepsRunning() {
    let tracker = VideoJobTracker()
    let (doneId, _) = tracker.register(source: "api", mode: .t2v)
    let (runningId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markSucceeded(doneId, result: result())
    tracker.markProcessing(runningId)

    // Negative ttl => cutoff in the future => every already-completed job drops;
    // the still-running job (no completedAt) survives.
    tracker.pruneCompleted(olderThan: -1)

    XCTAssertNil(tracker.status(jobId: doneId), "completed job should be pruned")
    XCTAssertEqual(tracker.status(jobId: runningId)?.status, .processing,
                   "running job must survive prune")
  }

  func testPruneKeepsRecentCompleted() {
    let tracker = VideoJobTracker()
    let (doneId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markSucceeded(doneId, result: result())
    // Large ttl => recent completion stays.
    tracker.pruneCompleted(olderThan: 3600)
    XCTAssertEqual(tracker.status(jobId: doneId)?.status, .succeeded)
  }

  // MARK: - progress percent mapping (pure helper reused by both local paths)

  func testProgressPercentMapping() {
    XCTAssertEqual(WarmServer.localVideoProgressPercent(chunk: 0, totalChunks: 4, step: 0, totalSteps: 8), 0)
    XCTAssertEqual(WarmServer.localVideoProgressPercent(chunk: 4, totalChunks: 4, step: 0, totalSteps: 8), 100)
    XCTAssertEqual(WarmServer.localVideoProgressPercent(chunk: 1, totalChunks: 2, step: 4, totalSteps: 8), 75)
    XCTAssertEqual(WarmServer.localVideoProgressPercent(chunk: 0, totalChunks: 1, step: 4, totalSteps: 8), 50)
  }

  func testProgressPercentGuardsAgainstZeroTotals() {
    // Never divide by zero; never exceed 0-100.
    XCTAssertEqual(WarmServer.localVideoProgressPercent(chunk: 0, totalChunks: 0, step: 0, totalSteps: 0), 0)
    XCTAssertEqual(WarmServer.localVideoProgressPercent(chunk: 5, totalChunks: 0, step: 9, totalSteps: 0), 100)
    XCTAssertEqual(WarmServer.localVideoProgressPercent(chunk: -3, totalChunks: 4, step: -2, totalSteps: 8), 0)
  }
}
