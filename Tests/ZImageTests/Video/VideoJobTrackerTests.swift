import XCTest
@testable import ZImage

/// Unit tests for the async LOCAL LTX-2 video job tracker (Phase A of the
/// JoyAI-Echo port). The state machine is exercised in isolation — no
/// coordinator, no real render — via the tracker's transition surface, exactly
/// how the production `submit(...)` drives it. Mirrors the image job pattern.
final class VideoJobTrackerTests: XCTestCase {

  private func result(
    path: String = "/out/ltx2.mp4", frames: Int = 97,
    duration: Float = 4.0, elapsed: Double = 12.5,
    refineSkippedReason: String? = nil
  ) -> LTX2VideoResult {
    LTX2VideoResult(
      outputPath: path, frameCount: frames,
      durationSeconds: duration, elapsedSeconds: elapsed,
      refineSkippedReason: refineSkippedReason)
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

  func testAcceptedRecipeHashSurvivesEveryStatusTransition() {
    let tracker = VideoJobTracker()
    let (jobId, accepted) = tracker.register(
      source: "api", mode: .i2v, recipeHash: "recipe-sha")
    XCTAssertEqual(accepted.recipeHash, "recipe-sha")
    tracker.markProcessing(jobId)
    XCTAssertEqual(tracker.status(jobId: jobId)?.recipeHash, "recipe-sha")
    tracker.markSucceeded(jobId, result: result())
    XCTAssertEqual(tracker.status(jobId: jobId)?.recipeHash, "recipe-sha")
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

  /// comfybox#307: `refine_skipped` must be nil on a normal (or non-two-stage)
  /// success — additive, no change for every existing caller/test.
  func testMarkSucceededWithNoRefineSkipLeavesFieldNil() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .i2v)
    tracker.markSucceeded(jobId, result: result())
    XCTAssertNil(tracker.status(jobId: jobId)?.refineSkipped)
  }

  /// The phantom-refine case the issue describes: `two_stage` requested, the
  /// volume gate skipped it, the render still "succeeds" — but the status
  /// must say so.
  func testMarkSucceededSurfacesRefineSkippedReason() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .i2v)
    tracker.markSucceeded(jobId, result: result(
      refineSkippedReason: "volume_gate (pre-refine volume 30000 > refine_max_vol 26000)"))
    XCTAssertEqual(
      tracker.status(jobId: jobId)?.refineSkipped,
      "volume_gate (pre-refine volume 30000 > refine_max_vol 26000)")
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

  // MARK: - activeJobCount (#298 review finding 5: terminal jobs must not
  // inflate /health.video.active_jobs during their TTL retention window)

  func testActiveJobCountExcludesTerminalJobs() {
    struct Boom: LocalizedError { var errorDescription: String? { "render blew up" } }
    let tracker = VideoJobTracker()

    XCTAssertEqual(tracker.activeJobCount, 0, "an empty tracker has no active jobs")

    let (queuedId, _) = tracker.register(source: "api", mode: .t2v)
    XCTAssertEqual(tracker.activeJobCount, 1, "a freshly-queued job is active")

    let (processingId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markProcessing(processingId)
    XCTAssertEqual(tracker.activeJobCount, 2, "queued + processing both count")

    let (succeededId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markSucceeded(succeededId, result: result())
    XCTAssertEqual(tracker.activeJobCount, 2, "a succeeded (terminal) job drops out immediately")

    let (failedId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markFailed(failedId, error: Boom())
    XCTAssertEqual(tracker.activeJobCount, 2, "a failed (terminal) job drops out immediately")

    // The terminal jobs are still queryable by id (TTL retention) even
    // though they no longer count as active.
    XCTAssertEqual(tracker.status(jobId: succeededId)?.status, .succeeded)
    XCTAssertEqual(tracker.status(jobId: failedId)?.status, .failed)
    XCTAssertEqual(tracker.status(jobId: queuedId)?.status, .queued)
    XCTAssertEqual(tracker.status(jobId: processingId)?.status, .processing)
  }

  // MARK: - Director (WP2c)

  private func directorPlan() -> DirectorPlan {
    DirectorPlan(
      lengthFrames: 577, fps: 24, width: 576, height: 896, audioMode: "generated",
      chunks: [
        .init(index: 0, startFrame: 0, endFrame: 288, frames: 289, seed: 42, carryOver: false, audio: "generated"),
        .init(index: 1, startFrame: 288, endFrame: 576, frames: 289, seed: 43, carryOver: true, audio: "generated"),
      ],
      keyframeTicks: [.init(id: "k1", frame: 0)], boundaryFrames: [288], warnings: [])
  }

  func testSubmitOrchestratedDirectorRegistersPlanConfigHashAndStages() throws {
    let traceDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("director-trace-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: traceDir) }
    let tracker = VideoJobTracker()
    tracker.traceStore = RenderTraceStore(directory: traceDir)
    let plan = directorPlan()
    let config = [LTX2ResolvedParam(
      name: "two_stage", envKey: nil, tier: "A", value: "false", source: .builtin, valid: true, note: nil)]

    let started = expectation(description: "work started")
    let release = DispatchSemaphore(value: 0)
    let seenJobId = LockedBox<String?>(nil)
    let queued = tracker.submitOrchestrated(
      source: "desktop", mode: .director,
      resolvedConfig: config, recipeHash: "hash-0",
      tracePayload: ["director_plan": "{\"length_frames\":577}", "director_chunks": "2"],
      plan: plan, stageCount: 2
    ) { jobId, report in
      seenJobId.set(jobId)
      tracker.setStage(jobId, index: 1, count: 2)
      report(50)
      started.fulfill()
      release.wait()
      return LTX2VideoResult(outputPath: "/out/director.mp4", frameCount: 577, durationSeconds: 24, elapsedSeconds: 1)
    }

    XCTAssertEqual(queued.status, .queued)
    XCTAssertEqual(queued.mode, .director)
    XCTAssertEqual(queued.plan, plan)
    XCTAssertEqual(queued.stageCount, 2)
    XCTAssertNil(queued.stageIndex)
    XCTAssertEqual(queued.recipeHash, "hash-0")
    XCTAssertEqual(queued.resolvedConfig, config)

    wait(for: [started], timeout: 5)
    XCTAssertEqual(seenJobId.get(), queued.jobId, "the work closure receives the tracker job id")
    let mid = try XCTUnwrap(tracker.status(jobId: queued.jobId))
    XCTAssertEqual(mid.stageIndex, 1)
    XCTAssertEqual(mid.stageCount, 2)
    XCTAssertEqual(mid.progressPercent, 50)
    release.signal()

    let deadline = Date().addingTimeInterval(5)
    while tracker.status(jobId: queued.jobId)?.status != .succeeded, Date() < deadline {
      usleep(10_000)
    }
    let done = try XCTUnwrap(tracker.status(jobId: queued.jobId))
    XCTAssertEqual(done.status, .succeeded)
    XCTAssertEqual(done.mode, .director)
    XCTAssertEqual(done.plan, plan)

    tracker.traceStore?.flush()
    let submitted = try XCTUnwrap(tracker.traceStore?.events(renderId: queued.jobId).first { $0.event == .submitted })
    XCTAssertEqual(submitted.payload["mode"], "director")
    XCTAssertEqual(submitted.payload["director_chunks"], "2")
    XCTAssertNotNil(submitted.payload["director_plan"])
  }

  func testDirectorChunkFailedErrorStringFormat() {
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(source: "api", mode: .director, plan: directorPlan(), stageCount: 3)
    tracker.markProcessing(jobId)
    tracker.markFailed(jobId, error: DirectorError.chunkFailed(chunk: 1, stage: "render", message: "x"))
    let s = tracker.status(jobId: jobId)
    XCTAssertEqual(s?.status, .failed)
    XCTAssertEqual(s?.error, "chunk 2/3 (render): x")
    XCTAssertNil(s?.interrupted)

    let (stitchId, _) = tracker.register(source: "api", mode: .director, stageCount: 3)
    tracker.markFailed(stitchId, error: DirectorError.stitchFailed("writer died"))
    XCTAssertEqual(tracker.status(jobId: stitchId)?.error, "chunk 3/3 (stitch): writer died")

    // Cancellation still classifies as an interrupt, not a chunk failure.
    let (cancelId, _) = tracker.register(source: "api", mode: .director, stageCount: 3)
    tracker.markFailed(cancelId, error: CancellationError())
    XCTAssertEqual(tracker.status(jobId: cancelId)?.interrupted, true)

    // Non-director jobs keep localizedDescription.
    let (plainId, _) = tracker.register(source: "api", mode: .t2v)
    tracker.markFailed(plainId, error: DirectorError.chunkFailed(chunk: 1, stage: "render", message: "x"))
    XCTAssertEqual(tracker.status(jobId: plainId)?.error, "chunk 2 (render): x")
  }
}

private final class LockedBox<T>: @unchecked Sendable {
  private let lock = NSLock()
  private var value: T
  init(_ value: T) { self.value = value }
  func set(_ v: T) { lock.lock(); value = v; lock.unlock() }
  func get() -> T { lock.lock(); defer { lock.unlock() }; return value }
}
