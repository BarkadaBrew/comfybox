import XCTest
@testable import ZImage

/// comfybox#362 — during a #1479 preemption episode, `/health` and
/// `/v1/queue` show the preempting IMAGE job as active, but before this fix
/// `activeRenderTask` (and the sync `liveHealth` handle `/v1/queue/interrupt`
/// actually cancels) still pointed at the checkpointed VIDEO for the whole
/// episode — so a plain interrupt abandoned the invisible video while the
/// visibly-active image render kept going.
///
/// The fix has two parts, tested here:
///
///   1. `runAsPublishedActiveRender` (`WarmServer.swift`) republishes the
///      PREEMPTOR's own task as the active render for the episode's duration,
///      restoring the video's task afterward — `InterruptTargetTests`'s
///      `testRunAsPublishedActiveRender…` tests pin this directly, using fake
///      tasks instead of a real checkpoint/render (the #218 admission gate
///      and a real LTX-2 resume both need real weights, out of a unit test's
///      reach — same rationale `LTX2CancellationBoundaryTests` documents for
///      pinning boundary placement rather than behaviour).
///   2. `/v1/queue/interrupt` gains an additive `target` ("active" default |
///      "video" | a job id), resolved by the pure `InterruptTarget.resolve`
///      and applied identically by both implementations
///      (`LiveHealthState.cancelActiveRender`, the sync no-actor-hop path,
///      and `WarmServerCoordinator.cancelActiveRender`, the async fallback).
///
/// Also covers comfybox#362's second finding: `bypassVideoAdmissionForTests`
/// (DEBUG-only) previously had no way to be turned back off within a test.
final class InterruptTargetResolutionTests: XCTestCase {

  func testNilAndTheLiteralActiveResolveToActive() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: nil, activeJobId: "a", checkpointedVideoJobId: "v"), .active)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "active", activeJobId: "a", checkpointedVideoJobId: "v"), .active)
    // Even with nothing running, the default target is still "active" — the
    // resolution is a pure name, not a "does anything match" check.
    XCTAssertEqual(
      InterruptTarget.resolve(target: nil, activeJobId: nil, checkpointedVideoJobId: nil), .active)
  }

  func testVideoResolvesToVideoRegardlessOfWhatIsCheckpointed() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: "video", activeJobId: "a", checkpointedVideoJobId: "v"), .video)
    // No episode in progress (checkpointedVideoJobId nil) still resolves to
    // "video" — the per-host cancel function decides how to reach it (the
    // active render itself, if IT is the video).
    XCTAssertEqual(
      InterruptTarget.resolve(target: "video", activeJobId: "a", checkpointedVideoJobId: nil), .video)
  }

  func testAJobIdMatchingTheActiveJobResolvesToActive() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: "a", activeJobId: "a", checkpointedVideoJobId: "v"), .active)
  }

  func testAJobIdMatchingTheCheckpointedVideoResolvesToVideo() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: "v", activeJobId: "a", checkpointedVideoJobId: "v"), .video)
  }

  func testAnUnrecognisedJobIdIsUnknown() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: "no-such-job", activeJobId: "a", checkpointedVideoJobId: "v"),
      .unknownJobId)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "no-such-job", activeJobId: nil, checkpointedVideoJobId: nil),
      .unknownJobId)
  }
}

final class InterruptRouteResponseTests: XCTestCase {

  func testCancelledBuildsA200WithTheAdditiveFieldsPopulated() {
    let (status, body) = InterruptRouteResponse.build(from: .cancelled(jobId: "j1", kind: "video"))
    XCTAssertEqual(status, 200)
    XCTAssertTrue(body.success)
    XCTAssertTrue(body.interrupted)
    XCTAssertEqual(body.interruptedJobId, "j1")
    XCTAssertEqual(body.interruptedKind, "video")
  }

  func testNothingToCancelBuildsA200WithInterruptedFalseAndNoAdditiveFields() {
    let (status, body) = InterruptRouteResponse.build(from: .nothingToCancel)
    XCTAssertEqual(status, 200)
    XCTAssertTrue(body.success, "nothing running there is not an error — matches pre-#362 behaviour")
    XCTAssertFalse(body.interrupted)
    XCTAssertNil(body.interruptedJobId)
    XCTAssertNil(body.interruptedKind)
  }

  func testUnknownTargetBuildsA404() {
    let (status, body) = InterruptRouteResponse.build(from: .unknownTarget)
    XCTAssertEqual(status, 404)
    XCTAssertFalse(body.success)
    XCTAssertFalse(body.interrupted)
  }
}

/// Integration-level: the pure resolver wired through the REAL
/// `LiveHealthState`/`WarmServerCoordinator` state, via `WarmServerQueueProbe`.
final class InterruptTargetIntegrationTests: XCTestCase {

  /// Cross-task state for a synchronous/spin-loop fake render. Same shape as
  /// `LocalVideoInterruptTests.RenderProbe` — duplicated per-file convention
  /// already used across this test target (`ControlPlaneTests`,
  /// `AsyncJobIdTests`, etc. each keep their own `waitUntil` too).
  private final class RenderProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var _running = false
    private var _observedCancellation = false

    var running: Bool {
      get { lock.lock(); defer { lock.unlock() }; return _running }
      set { lock.lock(); _running = newValue; lock.unlock() }
    }
    var observedCancellation: Bool {
      get { lock.lock(); defer { lock.unlock() }; return _observedCancellation }
      set { lock.lock(); _observedCancellation = newValue; lock.unlock() }
    }
  }

  private static let dummyResult = LTX2VideoResult(
    outputPath: "/dev/null", frameCount: 0, durationSeconds: 0, elapsedSeconds: 0)

  /// Spin until `condition`, or fail. `controlInterrupt()` itself is
  /// idempotent-safe to poll: it only ever cancels an already-published task,
  /// so calling it repeatedly while waiting for a handle to be published
  /// cannot double-fire anything harmful.
  private func waitUntil(
    _ description: String, timeout: TimeInterval = 10,
    _ condition: @Sendable () -> Bool
  ) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() {
      if Date() >= deadline { XCTFail("timed out waiting for \(description)"); return }
      try await Task.sleep(nanoseconds: 5_000_000)
    }
  }

  // MARK: - Default target agrees with health

  /// The controller's ruling, checked directly: a plain interrupt (no
  /// `target`) cancels whatever `/health` currently reports as active, and
  /// the response's additive `interrupted_job_id`/`interrupted_kind` name
  /// exactly that job — not a copy, the SAME id `/health` was showing.
  func testDefaultTargetCancelsWhateverHealthShowsAsActiveAndNamesIt() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let jobId = "synthetic-\(UUID().uuidString)"
    let job = Task { try await probe.enqueueSynthetic(durationMs: 3000, id: jobId) }
    try await waitUntil("the synthetic job to become active") { probe.activeJobId == jobId }

    // Health and the interrupt target must agree BEFORE we touch anything.
    XCTAssertEqual(probe.activeJobId, jobId)

    let result = probe.controlInterrupt(target: nil)
    XCTAssertTrue(result.interrupted)
    XCTAssertFalse(result.unknownTarget)
    XCTAssertEqual(result.jobId, jobId, "interrupted_job_id must equal health's active job id")
    XCTAssertEqual(result.kind, "synthetic", "interrupted_kind must name what was actually running")

    let succeeded = try await job.value
    XCTAssertFalse(succeeded, "the active job must have observed the cancellation")
  }

  /// The async fallback path (`ControlPlaneSyncFlag` off) must agree with the
  /// sync one — not assumed, checked: both resolve the SAME job/kind for the
  /// SAME default target.
  func testAsyncFallbackInterruptAgreesWithTheSyncPath() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let jobId = "synthetic-async-\(UUID().uuidString)"
    let job = Task { try? await probe.enqueueSynthetic(durationMs: 3000, id: jobId) }
    try await waitUntil("the synthetic job to become active") { probe.activeJobId == jobId }

    let result = await probe.coordinatorInterrupt()
    XCTAssertTrue(result.interrupted)
    XCTAssertEqual(result.jobId, jobId)
    XCTAssertEqual(result.kind, "synthetic")
    _ = await job.value
  }

  func testAnUnknownJobIdTargetIsReportedAsUnknownNotAsNothingToCancel() throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let result = probe.controlInterrupt(target: "no-such-job-anywhere")
    XCTAssertFalse(result.interrupted)
    XCTAssertTrue(result.unknownTarget, "an unrecognised job id must 404, not silently report false")
  }

  // MARK: - `target: "video"` reaches the checkpointed video specifically

  /// The other half of the controller's ruling: while a preemption episode
  /// has swapped the active render to the preempting image job, `target:
  /// "video"` must still reach the checkpointed video — and must NOT touch
  /// the active image job. Uses `setCheckpointedVideo` (the same publish
  /// `runPreemptionEpisode` performs) with a fake task, and a real
  /// `.synthetic` job standing in for the preemptor, so no model weights are
  /// needed anywhere.
  func testVideoTargetCancelsOnlyTheCheckpointedVideoDuringAnEpisode() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let videoProbe = RenderProbe()
    let videoTask = Task {
      videoProbe.running = true
      let deadline = Date().addingTimeInterval(10)
      while !Task.isCancelled && Date() < deadline { usleep(2_000) }
      if Task.isCancelled { videoProbe.observedCancellation = true }
    }
    try await waitUntil("the fake checkpointed-video task to start") { videoProbe.running }
    await probe.setCheckpointedVideo(task: videoTask, jobId: "video-job-1")

    // Stand-in for the preemptor: a real queued job, so `/health` genuinely
    // shows IT as active while the fake video sits checkpointed underneath.
    let preemptorJobId = "preemptor-job-1"
    let job = Task { try await probe.enqueueSynthetic(durationMs: 3000, id: preemptorJobId) }
    try await waitUntil("the preemptor stand-in to become active") { probe.activeJobId == preemptorJobId }

    let result = probe.controlInterrupt(target: "video")
    XCTAssertTrue(result.interrupted)
    XCTAssertEqual(result.jobId, "video-job-1")
    XCTAssertEqual(result.kind, "video")

    _ = await videoTask.value
    XCTAssertTrue(videoProbe.observedCancellation, "target: video must reach the checkpointed video")

    // The active (preempting) job must be untouched — it completes normally.
    let succeeded = try await job.value
    XCTAssertTrue(succeeded, "target: video must not cancel the active preempting job")
  }

  /// A job id target resolves identically to the reserved words: the video's
  /// own id behaves exactly like `target: "video"`.
  func testTheCheckpointedVideosOwnJobIdResolvesLikeTheVideoTarget() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let videoProbe = RenderProbe()
    let videoTask = Task {
      videoProbe.running = true
      let deadline = Date().addingTimeInterval(10)
      while !Task.isCancelled && Date() < deadline { usleep(2_000) }
      if Task.isCancelled { videoProbe.observedCancellation = true }
    }
    try await waitUntil("the fake checkpointed-video task to start") { videoProbe.running }
    await probe.setCheckpointedVideo(task: videoTask, jobId: "video-job-2")

    let preemptorJobId = "preemptor-job-2"
    let job = Task { try await probe.enqueueSynthetic(durationMs: 3000, id: preemptorJobId) }
    try await waitUntil("the preemptor stand-in to become active") { probe.activeJobId == preemptorJobId }

    let result = probe.controlInterrupt(target: "video-job-2")
    XCTAssertTrue(result.interrupted)
    XCTAssertEqual(result.kind, "video")

    _ = await videoTask.value
    XCTAssertTrue(videoProbe.observedCancellation)
    let succeeded = try await job.value
    XCTAssertTrue(succeeded)
  }

  /// No episode in progress: `target: "video"` still reaches a REAL video
  /// render (the `.localVideo` queue case) when IT is directly the active
  /// render — falling back to `activeRenderTask` rather than requiring a
  /// separately-published checkpoint handle.
  func testVideoTargetHitsTheVideoDirectlyWhenNoEpisodeIsInProgress() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()
    await probe.bypassVideoAdmission()

    let render = RenderProbe()
    let job = Task { () -> Result<LTX2VideoResult, Error> in
      do {
        return .success(try await probe.enqueueLocalVideo { _ in
          render.running = true
          let deadline = Date().addingTimeInterval(10)
          while !Task.isCancelled && Date() < deadline { usleep(2_000) }
          if Task.isCancelled {
            render.observedCancellation = true
            throw CancellationError()
          }
          return .completed(Self.dummyResult)
        })
      } catch { return .failure(error) }
    }
    try await waitUntil("the video body to start") { render.running }

    let result = probe.controlInterrupt(target: "video")
    XCTAssertTrue(result.interrupted)
    XCTAssertEqual(result.kind, "video")

    let outcome = await job.value
    guard case .failure(let error) = outcome else {
      return XCTFail("an interrupted render must not report success")
    }
    XCTAssertTrue(WarmServerQueueProbe.isInterrupted(error))
    XCTAssertTrue(render.observedCancellation)
  }

  // MARK: - `runAsPublishedActiveRender`: the task-publish fix itself

  /// This is the direct pin for comfybox#362's root cause: publishes the
  /// shielded work's own task as the active render for its duration, then
  /// restores the video's task — proven with fake tasks (the #218 admission
  /// gate and a real LTX-2 resume both need real memory/weights a unit test
  /// cannot supply).
  func testRunAsPublishedActiveRenderPublishesWorkThenRestoresTheVideo() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let videoProbe = RenderProbe()
    let videoTask = Task {
      videoProbe.running = true
      let deadline = Date().addingTimeInterval(10)
      while !Task.isCancelled && Date() < deadline { usleep(2_000) }
      if Task.isCancelled { videoProbe.observedCancellation = true }
    }
    try await waitUntil("the fake video task to start") { videoProbe.running }

    let workProbe = RenderProbe()
    let episode = Task {
      await probe.runAsPublishedActiveRender(restoringTo: videoTask) {
        workProbe.running = true
        let deadline = Date().addingTimeInterval(10)
        while !Task.isCancelled && Date() < deadline { usleep(2_000) }
        if Task.isCancelled { workProbe.observedCancellation = true }
      }
    }

    // Poll a DEFAULT-target interrupt until it lands — it can only succeed
    // once the episode has actually published `work`'s task, so a bounded
    // poll (rather than a fixed sleep) is what makes this deterministic.
    try await waitUntil("the default-target interrupt to land on the shielded work") {
      probe.controlInterrupt()
    }
    await episode.value

    XCTAssertTrue(workProbe.observedCancellation, "the shielded work must be reachable as the default target")
    XCTAssertFalse(
      videoProbe.observedCancellation,
      "the video's own task must NOT be touched while its handle isn't published as active")

    // After the episode, the video's own handle is republished — a further
    // default-target interrupt now reaches the video.
    XCTAssertTrue(probe.controlInterrupt(), "the video's handle must be restored as active once the episode ends")
    _ = await videoTask.value
    XCTAssertTrue(videoProbe.observedCancellation)
  }

  // MARK: - comfybox#362 second finding: the DEBUG admission bypass can be reset

  func testBypassVideoAdmissionCanBeToggledBackOffWithinATest() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    var value = await probe.bypassVideoAdmissionValueForTest()
    XCTAssertFalse(value, "a fresh probe must start with admission NOT bypassed")

    await probe.bypassVideoAdmission()
    value = await probe.bypassVideoAdmissionValueForTest()
    XCTAssertTrue(value)

    await probe.bypassVideoAdmission(false)
    value = await probe.bypassVideoAdmissionValueForTest()
    XCTAssertFalse(value, "the bypass must be resettable within a single test/coordinator lifetime")
  }
}
