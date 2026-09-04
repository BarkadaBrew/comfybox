import XCTest

@testable import ZImage

/// comfybox#289 — ONE job model. A single `job_id` namespace and a single
/// `get_job` tool map every engine status shape (image, video, swap,
/// storyboard) onto `{job_id, kind, state, progress, result?, error?,
/// retry_after_seconds?}`.
///
/// The per-kind polling tools (`video_status`, `workflow_run_status`) keep
/// their existing shapes — the daemon contract is production (intent.md);
/// this is additive on top, never a replacement.
final class MCPJobModelTests: XCTestCase {

  // MARK: - Kind → status route

  func testImageAndSwapPollTheImageStatusRoute() {
    XCTAssertEqual(MCPJobKind.image.statusPathTemplate, "/v1/generate/status/{id}")
    XCTAssertEqual(MCPJobKind.swap.statusPathTemplate, "/v1/generate/status/{id}")
    XCTAssertEqual(MCPJobKind.image.statusPath(jobId: "J-1"), "/v1/generate/status/J-1")
  }

  /// Storyboard renders submit through `/v1/storyboard/render` and are tracked
  /// as VIDEO jobs (MCPToolExecutor.executeRenderStoryboard: "202 + job id;
  /// poll video_status"), so the unified model polls the video route for them.
  func testVideoAndStoryboardPollTheVideoStatusRoute() {
    XCTAssertEqual(MCPJobKind.video.statusPathTemplate, "/v1/video/status/{id}")
    XCTAssertEqual(MCPJobKind.storyboard.statusPathTemplate, "/v1/video/status/{id}")
    XCTAssertEqual(MCPJobKind.storyboard.statusPath(jobId: "J-2"), "/v1/video/status/J-2")
  }

  func testProbeOrderIsImageThenVideo() {
    XCTAssertEqual(MCPJobModel.probeOrder, [.image, .video])
  }

  // MARK: - Engine state vocabulary → one state vocabulary

  func testEngineStatesMapOntoOneVocabulary() {
    XCTAssertEqual(MCPJobModel.state(fromEngine: "queued"), .queued)
    XCTAssertEqual(MCPJobModel.state(fromEngine: "processing"), .running)
    XCTAssertEqual(MCPJobModel.state(fromEngine: "running"), .running)
    XCTAssertEqual(MCPJobModel.state(fromEngine: "succeeded"), .completed)
    // mapHTTPResponse already normalizes "succeeded" -> "completed" for the
    // per-kind tools; a caller feeding that back must land in the same state.
    XCTAssertEqual(MCPJobModel.state(fromEngine: "completed"), .completed)
    XCTAssertEqual(MCPJobModel.state(fromEngine: "failed"), .failed)
  }

  /// #1479: a video render checkpointed so a preempting image job can run is
  /// NOT terminal — a client that stops polling here waits forever.
  func testPausedForPreemptionIsRunningNotTerminal() {
    XCTAssertEqual(MCPJobModel.state(fromEngine: "pausedForPreemption"), .running)
    XCTAssertEqual(MCPJobModel.state(fromEngine: "paused_for_preemption"), .running)
    XCTAssertFalse(MCPJobState.running.isTerminal)
    XCTAssertTrue(MCPJobState.completed.isTerminal)
    XCTAssertTrue(MCPJobState.failed.isTerminal)
  }

  func testUnknownEngineStateIsNil() {
    XCTAssertNil(MCPJobModel.state(fromEngine: "banana"))
  }

  // MARK: - Image job mapping

  func testUnifyImageSucceeded() throws {
    let status: [String: Any] = [
      "job_id": "J-1", "status": "succeeded", "source": "mcp",
      "output_path": "/tmp/a.png", "duration_ms": 4200, "elapsed_ms": 5000,
    ]
    let out = MCPJobModel.unify(kind: .image, jobId: "J-1", status: status)
    XCTAssertEqual(out["job_id"] as? String, "J-1")
    XCTAssertEqual(out["kind"] as? String, "image")
    XCTAssertEqual(out["state"] as? String, "completed")
    XCTAssertEqual(out["progress"] as? Int, 100)
    let result = try XCTUnwrap(out["result"] as? [String: Any])
    XCTAssertEqual(result["output_path"] as? String, "/tmp/a.png")
    XCTAssertEqual(result["duration_ms"] as? Int, 4200)
    XCTAssertNil(out["error"])
    XCTAssertNil(out["retry_after_seconds"])
  }

  func testUnifyImageQueuedHasZeroProgressAndNoResult() {
    let status: [String: Any] = ["job_id": "J-1", "status": "queued", "elapsed_ms": 12]
    let out = MCPJobModel.unify(kind: .image, jobId: "J-1", status: status)
    XCTAssertEqual(out["state"] as? String, "queued")
    XCTAssertEqual(out["progress"] as? Int, 0)
    XCTAssertNil(out["result"])
  }

  /// The image tracker has no per-job percent (ImageJobStatus carries none),
  /// so a running image job's live percent comes from the lock-based queue
  /// snapshot (`GET /v1/queue.progress_percent`) — and ONLY when that snapshot
  /// says this job is the active render.
  func testUnifyImageRunningTakesQueueProgressWhenActive() {
    let status: [String: Any] = ["job_id": "J-1", "status": "processing", "elapsed_ms": 9000]
    let out = MCPJobModel.unify(kind: .image, jobId: "J-1", status: status, queueProgressPercent: 42)
    XCTAssertEqual(out["state"] as? String, "running")
    XCTAssertEqual(out["progress"] as? Int, 42)
  }

  func testUnifyImageRunningWithoutQueueProgressReportsZero() {
    let status: [String: Any] = ["job_id": "J-1", "status": "processing", "elapsed_ms": 9000]
    let out = MCPJobModel.unify(kind: .image, jobId: "J-1", status: status)
    XCTAssertEqual(out["progress"] as? Int, 0)
  }

  func testUnifyImageFailedCarriesError() {
    let status: [String: Any] = [
      "job_id": "J-1", "status": "failed", "error": "VRAM exhausted", "elapsed_ms": 900,
    ]
    let out = MCPJobModel.unify(kind: .image, jobId: "J-1", status: status)
    XCTAssertEqual(out["state"] as? String, "failed")
    XCTAssertEqual(out["error"] as? String, "VRAM exhausted")
    XCTAssertNil(out["result"])
  }

  /// A LoRA-swap job replayed after a restart is tracked BY ID in the image
  /// tracker (WarmServer.recordFailedReplay), so the swap kind reads the same
  /// route and shape — only the echoed `kind` differs.
  func testUnifySwapUsesImageShapeAndEchoesSwapKind() {
    let status: [String: Any] = ["job_id": "S-1", "status": "failed", "error": "replay failed"]
    let out = MCPJobModel.unify(kind: .swap, jobId: "S-1", status: status)
    XCTAssertEqual(out["kind"] as? String, "swap")
    XCTAssertEqual(out["state"] as? String, "failed")
    XCTAssertEqual(out["error"] as? String, "replay failed")
  }

  // MARK: - Video job mapping

  func testUnifyVideoProcessingUsesEngineProgressPercent() {
    let status: [String: Any] = [
      "job_id": "V-1", "status": "processing", "backend": "ltx2", "progress_percent": 37,
      "estimated_seconds": 240,
    ]
    let out = MCPJobModel.unify(kind: .video, jobId: "V-1", status: status)
    XCTAssertEqual(out["kind"] as? String, "video")
    XCTAssertEqual(out["state"] as? String, "running")
    XCTAssertEqual(out["progress"] as? Int, 37)
  }

  func testUnifyVideoSucceededCarriesVideoResultFields() throws {
    let status: [String: Any] = [
      "job_id": "V-1", "status": "succeeded", "backend": "ltx2",
      "output_path": "/tmp/clip.mp4", "duration_ms": 900_000,
      "file_size_bytes": 12_345_678, "video_duration_seconds": 8, "frame_count": 193,
    ]
    let out = MCPJobModel.unify(kind: .video, jobId: "V-1", status: status)
    XCTAssertEqual(out["state"] as? String, "completed")
    XCTAssertEqual(out["progress"] as? Int, 100)
    let result = try XCTUnwrap(out["result"] as? [String: Any])
    XCTAssertEqual(result["output_path"] as? String, "/tmp/clip.mp4")
    XCTAssertEqual(result["duration_ms"] as? Int, 900_000)
    XCTAssertEqual(result["file_size_bytes"] as? Int, 12_345_678)
    XCTAssertEqual(result["video_duration_seconds"] as? Int, 8)
    XCTAssertEqual(result["frame_count"] as? Int, 193)
  }

  func testUnifyStoryboardEchoesStoryboardKindOnAVideoShape() {
    let status: [String: Any] = [
      "job_id": "SB-1", "status": "processing", "backend": "ltx2", "progress_percent": 5,
    ]
    let out = MCPJobModel.unify(kind: .storyboard, jobId: "SB-1", status: status)
    XCTAssertEqual(out["kind"] as? String, "storyboard")
    XCTAssertEqual(out["state"] as? String, "running")
    XCTAssertEqual(out["progress"] as? Int, 5)
  }

  /// comfybox#322: an interrupted render still reports `failed` — terminal —
  /// with a flag and a plain-English reason. The unified model must not invent
  /// a state name no polling client knows.
  func testUnifyVideoInterruptedStaysTerminalFailed() {
    let status: [String: Any] = [
      "job_id": "V-2", "status": "failed", "backend": "ltx2",
      "error": "Render interrupted by operator", "interrupted": true,
    ]
    let out = MCPJobModel.unify(kind: .video, jobId: "V-2", status: status)
    XCTAssertEqual(out["state"] as? String, "failed")
    XCTAssertEqual(out["error"] as? String, "Render interrupted by operator")
  }

  // MARK: - Queue-recovery envelope

  /// The engine can refuse while it replays its persisted queue after a
  /// restart (QueueRecoveryGate: 503 + retry_after_seconds). get_job reports
  /// that as a still-queued job with the server's own hint, so the caller
  /// reschedules instead of treating it as a failure.
  func testRecoveryEnvelopeIsQueuedWithRetryHint() {
    let out = MCPJobModel.recoveryEnvelope(kind: .video, jobId: "V-3", retryAfterSeconds: 45)
    XCTAssertEqual(out["job_id"] as? String, "V-3")
    XCTAssertEqual(out["kind"] as? String, "video")
    XCTAssertEqual(out["state"] as? String, "queued")
    XCTAssertEqual(out["progress"] as? Int, 0)
    XCTAssertEqual(out["retry_after_seconds"] as? Int, 45)
  }

  func testUnifyPassesThroughRetryAfterSecondsWhenPresent() {
    let status: [String: Any] = [
      "job_id": "V-4", "status": "queued", "retry_after_seconds": 30,
    ]
    let out = MCPJobModel.unify(kind: .video, jobId: "V-4", status: status)
    XCTAssertEqual(out["retry_after_seconds"] as? Int, 30)
  }

  // MARK: - Submit envelope (#288)

  func testSubmitEnvelopeFromAsyncAcceptBody() {
    let accepted: [String: Any] = ["job_id": "J-9", "status": "queued", "source": "mcp", "elapsed_ms": 1]
    let out = MCPJobModel.submitEnvelope(kind: .image, status: accepted)
    XCTAssertEqual(out["job_id"] as? String, "J-9")
    XCTAssertEqual(out["kind"] as? String, "image")
    XCTAssertEqual(out["state"] as? String, "queued")
    XCTAssertEqual(out["progress"] as? Int, 0)
    XCTAssertEqual(out["poll_with"] as? String, "get_job")
  }

  func testSubmitEnvelopeWithoutJobIdIsNil() {
    XCTAssertNil(MCPJobModel.submitEnvelope(kind: .image, status: ["status": "queued"]))
  }

  // MARK: - get_job flow (injected transport, no networking)

  private struct Call: Equatable { let method: String; let path: String }

  private actor CallLog {
    private(set) var calls: [String] = []
    func record(_ method: String, _ path: String) { calls.append("\(method) \(path)") }
    func all() -> [String] { calls }
  }

  private func json(_ obj: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
  }

  func testGetJobWithExplicitVideoKindCallsOnlyTheVideoRoute() async throws {
    let log = CallLog()
    let result = try await MCPToolExecutor.runGetJob(jobId: "V-1", kind: .video) { method, path in
      await log.record(method, path)
      return (200, self.json(["job_id": "V-1", "status": "processing", "progress_percent": 12]))
    }
    XCTAssertFalse(result.isError)
    let calls = await log.all()
    XCTAssertEqual(calls, ["GET /v1/video/status/V-1"])
    let structured = try XCTUnwrap(result.structuredJSON)
    let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: structured) as? [String: Any])
    XCTAssertEqual(obj["kind"] as? String, "video")
    XCTAssertEqual(obj["progress"] as? Int, 12)
  }

  func testGetJobWithoutKindProbesImageThenVideo() async throws {
    let log = CallLog()
    let result = try await MCPToolExecutor.runGetJob(jobId: "V-1", kind: nil) { method, path in
      await log.record(method, path)
      if path.hasPrefix("/v1/generate/status/") {
        return (404, self.json(["error": "Image job not found: V-1"]))
      }
      return (200, self.json(["job_id": "V-1", "status": "succeeded", "output_path": "/tmp/c.mp4"]))
    }
    XCTAssertFalse(result.isError)
    let calls = await log.all()
    XCTAssertEqual(calls, ["GET /v1/generate/status/V-1", "GET /v1/video/status/V-1"])
    let obj = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(result.structuredJSON)) as? [String: Any])
    XCTAssertEqual(obj["kind"] as? String, "video")
    XCTAssertEqual(obj["state"] as? String, "completed")
  }

  func testGetJobNotFoundAnywhereIsACleanError() async throws {
    let result = try await MCPToolExecutor.runGetJob(jobId: "nope", kind: nil) { _, _ in
      (404, self.json(["error": "not found"]))
    }
    XCTAssertTrue(result.isError)
    let text = try XCTUnwrap(result.content.first?.text)
    XCTAssertTrue(text.contains("nope"), text)
  }

  /// A running IMAGE job's percent needs the second (lock-based, never
  /// actor-blocked) queue read — and only that case, so a completed or queued
  /// job costs exactly one HTTP call.
  func testGetJobReadsQueueProgressOnlyForARunningImageJob() async throws {
    let log = CallLog()
    let result = try await MCPToolExecutor.runGetJob(jobId: "J-1", kind: .image) { method, path in
      await log.record(method, path)
      if path == "/v1/queue" {
        return (200, self.json(["active_job_id": "J-1", "progress_percent": 66, "is_rendering": true]))
      }
      return (200, self.json(["job_id": "J-1", "status": "processing"]))
    }
    let calls = await log.all()
    XCTAssertEqual(calls, ["GET /v1/generate/status/J-1", "GET /v1/queue"])
    let obj = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(result.structuredJSON)) as? [String: Any])
    XCTAssertEqual(obj["progress"] as? Int, 66)
  }

  func testGetJobIgnoresQueueProgressForADifferentActiveJob() async throws {
    let result = try await MCPToolExecutor.runGetJob(jobId: "J-1", kind: .image) { _, path in
      if path == "/v1/queue" {
        return (200, self.json(["active_job_id": "OTHER", "progress_percent": 66]))
      }
      return (200, self.json(["job_id": "J-1", "status": "processing"]))
    }
    let obj = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(result.structuredJSON)) as? [String: Any])
    XCTAssertEqual(obj["progress"] as? Int, 0)
  }

  func testGetJobCompletedImageMakesOneCall() async throws {
    let log = CallLog()
    _ = try await MCPToolExecutor.runGetJob(jobId: "J-1", kind: .image) { method, path in
      await log.record(method, path)
      return (200, self.json(["job_id": "J-1", "status": "succeeded", "output_path": "/tmp/a.png"]))
    }
    XCTAssertEqual(await log.all(), ["GET /v1/generate/status/J-1"])
  }

  func testGetJobSurfacesQueueRecoveryRefusalAsQueuedWithRetryHint() async throws {
    let result = try await MCPToolExecutor.runGetJob(jobId: "V-5", kind: .video) { _, _ in
      (503, self.json([
        "error": "Queue recovery in progress",
        "error_code": QueueRecoveryGate.errorCode,
        "retry_after_seconds": 20,
      ]))
    }
    XCTAssertFalse(result.isError)
    let obj = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(result.structuredJSON)) as? [String: Any])
    XCTAssertEqual(obj["state"] as? String, "queued")
    XCTAssertEqual(obj["retry_after_seconds"] as? Int, 20)
  }

  /// A 503 that is NOT the recovery gate (e.g. "LTX-2 not configured") is a
  /// real error — retrying can never fix it, so it must not be dressed up as
  /// a queued job.
  func testGetJobNonRecovery503IsAnError() async throws {
    let result = try await MCPToolExecutor.runGetJob(jobId: "V-6", kind: .video) { _, _ in
      (503, self.json(["error": "LTX-2 not configured"]))
    }
    XCTAssertTrue(result.isError)
  }

  // MARK: - Async submit (#288)

  func testSubmitImageJobPostsTheAsyncRouteAndReturnsAJobId() async throws {
    let log = CallLog()
    let result = try await MCPToolExecutor.runSubmitImageJob(body: Data("{}".utf8)) { method, path, _ in
      await log.record(method, path)
      return (202, self.json(["job_id": "J-42", "status": "queued", "source": "mcp", "elapsed_ms": 0]))
    }
    XCTAssertFalse(result.isError)
    XCTAssertEqual(await log.all(), ["POST /v1/generate/async"])
    let obj = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(result.structuredJSON)) as? [String: Any])
    XCTAssertEqual(obj["job_id"] as? String, "J-42")
    XCTAssertEqual(obj["kind"] as? String, "image")
    XCTAssertEqual(obj["state"] as? String, "queued")
    XCTAssertEqual(obj["poll_with"] as? String, "get_job")
  }

  func testSubmitImageJobMapsAnErrorStatusThrough() async throws {
    let result = try await MCPToolExecutor.runSubmitImageJob(body: Data("{}".utf8)) { _, _, _ in
      (400, self.json(["error": "width must be divisible by 16"]))
    }
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("divisible") == true)
  }
}
