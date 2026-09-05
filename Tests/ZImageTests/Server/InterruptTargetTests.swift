import XCTest
@testable import ZImage

/// comfybox#362 — during a #1479 preemption episode, `/health` and
/// `/v1/queue` show the preempting IMAGE job as active, but before this fix
/// the published render task (and the sync `liveHealth` handle
/// `/v1/queue/interrupt` actually cancels) still pointed at the checkpointed
/// VIDEO for the whole episode — so a plain interrupt abandoned the invisible
/// video while the visibly-active image render kept going.
///
/// The fix has three parts, tested here:
///
///   1. `runAsPublishedActiveRender` (`WarmServer.swift`) republishes the
///      PREEMPTOR's own task as the active render for the episode's duration,
///      restoring the video's afterward — `InterruptTargetTests`'s
///      `testRunAsPublishedActiveRender…` test pins this directly, using fake
///      tasks instead of a real checkpoint/render (the #218 admission gate
///      and a real LTX-2 resume both need real weights, out of a unit test's
///      reach — same rationale `LTX2CancellationBoundaryTests` documents for
///      pinning boundary placement rather than behaviour).
///   2. Task and identity are published as ONE value, `PublishedRender`
///      (review r1, finding 1), so the job the interrupt cancels and the id it
///      reports can never come from different jobs — and both ids a video
///      answers to (queue and `/v1/video/status/{id}`, comfybox#283) ride
///      along in it (finding 3).
///   3. `/v1/queue/interrupt` gains an additive `target` ("active" default |
///      "video" | either id of either job), resolved by the pure
///      `InterruptTarget.resolve` and executed by the single shared
///      `InterruptExecutor` that BOTH implementations call
///      (`LiveHealthState.cancelActiveRender`, the sync no-actor-hop path, and
///      `WarmServerCoordinator.cancelActiveRender`, the async fallback, which
///      delegates to it — finding 2).
///
/// Also covers comfybox#362's second finding: `bypassVideoAdmissionForTests`
/// (DEBUG-only) previously had no way to be turned back off within a test.
final class InterruptTargetResolutionTests: XCTestCase {

  /// Identity-only publications — the resolver never looks at the task.
  private func active(_ id: String?, statusJobId: String? = nil, kind: String? = "generate")
    -> PublishedRender
  {
    PublishedRender(task: nil, jobId: id, statusJobId: statusJobId, kind: kind)
  }
  private func video(_ id: String?, statusJobId: String? = nil) -> PublishedRender {
    PublishedRender(task: nil, jobId: id, statusJobId: statusJobId, kind: "video")
  }

  func testNilAndTheLiteralActiveResolveToActive() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: nil, active: active("a"), checkpointedVideo: video("v")),
      .active)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "active", active: active("a"), checkpointedVideo: video("v")),
      .active)
    // Even with nothing running, the default target is still "active" — the
    // resolution is a pure name, not a "does anything match" check.
    XCTAssertEqual(
      InterruptTarget.resolve(target: nil, active: .none, checkpointedVideo: .none), .active)
  }

  /// Review r1, finding 6: the reserved words are normalised — trimmed and
  /// case-insensitive — and `""` is the DEFAULT target, not a job id that can
  /// never match (which would have 404'd).
  func testTheReservedWordsAreTrimmedAndCaseInsensitiveAndEmptyMeansActive() {
    let a = active("a")
    let v = video("v")
    XCTAssertEqual(InterruptTarget.resolve(target: "", active: a, checkpointedVideo: v), .active)
    XCTAssertEqual(InterruptTarget.resolve(target: "   ", active: a, checkpointedVideo: v), .active)
    XCTAssertEqual(
      InterruptTarget.resolve(target: " ACTIVE ", active: a, checkpointedVideo: v), .active)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "Video", active: a, checkpointedVideo: v), .video)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "\n video \t", active: a, checkpointedVideo: v), .video)
  }

  /// …but a job id is NOT case-folded: ids are opaque, and Foundation renders
  /// a `UUID` upper-case, so folding them would make two different jobs
  /// collide. Only surrounding whitespace is trimmed.
  func testAJobIdIsMatchedExactlyApartFromSurroundingWhitespace() {
    let id = "9F3C1B2A-0000-0000-0000-000000000001"
    let a = active(id)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "  \(id)  ", active: a, checkpointedVideo: .none), .active)
    XCTAssertEqual(
      InterruptTarget.resolve(target: id.lowercased(), active: a, checkpointedVideo: .none),
      .unknownJobId)
  }

  func testVideoResolvesToVideoRegardlessOfWhatIsCheckpointed() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: "video", active: active("a"), checkpointedVideo: video("v")),
      .video)
    // No episode in progress (nothing in the checkpointed slot) still
    // resolves to "video" — `InterruptExecutor` decides how to reach it (the
    // active render itself, if IT is the video) or that it is not there.
    XCTAssertEqual(
      InterruptTarget.resolve(target: "video", active: active("a"), checkpointedVideo: .none),
      .video)
  }

  func testAJobIdMatchingTheActiveJobResolvesToActive() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: "a", active: active("a"), checkpointedVideo: video("v")),
      .active)
  }

  func testAJobIdMatchingTheCheckpointedVideoResolvesToVideo() {
    XCTAssertEqual(
      InterruptTarget.resolve(target: "v", active: active("a"), checkpointedVideo: video("v")),
      .video)
  }

  /// Review r1, finding 3: the checkpointed video answers to BOTH of its ids
  /// — the queue id `/v1/queue` shows and the `/v1/video/status/{id}` id the
  /// client that submitted it actually holds (comfybox#283: they differ).
  func testEitherOfTheVideosTwoIdsResolvesToTheVideo() {
    let v = video("queue-id-1", statusJobId: "status-id-1")
    XCTAssertEqual(
      InterruptTarget.resolve(target: "queue-id-1", active: active("a"), checkpointedVideo: v),
      .video)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "status-id-1", active: active("a"), checkpointedVideo: v),
      .video)
  }

  /// The same two-id rule applies to a video that is directly the active
  /// render (no episode) — it is published with both ids too.
  func testEitherIdOfAnActiveVideoResolvesToActive() {
    let a = active("queue-id-2", statusJobId: "status-id-2", kind: "video")
    XCTAssertEqual(
      InterruptTarget.resolve(target: "status-id-2", active: a, checkpointedVideo: .none), .active)
  }

  func testAnUnrecognisedJobIdIsUnknown() {
    XCTAssertEqual(
      InterruptTarget.resolve(
        target: "no-such-job", active: active("a"), checkpointedVideo: video("v")),
      .unknownJobId)
    XCTAssertEqual(
      InterruptTarget.resolve(target: "no-such-job", active: .none, checkpointedVideo: .none),
      .unknownJobId)
  }
}

/// `InterruptExecutor` is the ONE cancel implementation both route arms run
/// (review r1, finding 2). These pin the decisions it makes on top of the
/// resolver, without a coordinator anywhere near them.
final class InterruptExecutorTests: XCTestCase {

  /// A task that is already finished — `cancel()` on it is a no-op, which is
  /// all these tests need (they assert on the OUTCOME, not on cancellation
  /// propagation; the integration suite below covers that).
  private func finishedTask() -> Task<Void, Never> { Task {} }

  func testTheDefaultTargetWithNothingRenderingIsNothingToCancelNotA404() {
    XCTAssertEqual(
      InterruptExecutor.cancel(target: nil, active: .none, checkpointedVideo: .none),
      .nothingToCancel,
      "the pre-#362 wire shape must survive: no body, nothing running, 200 + interrupted:false")
    XCTAssertEqual(
      InterruptExecutor.cancel(target: "active", active: .none, checkpointedVideo: .none),
      .nothingToCancel)
  }

  /// Review r1, finding 5: an EXPLICIT `"video"` that names no video is a
  /// 404, exactly like an unrecognised job id. Only the DEFAULT target keeps
  /// the legacy `interrupted: false` body — no pre-#362 client sends `target`
  /// at all, so nothing that exists today can see this status.
  func testAnExplicitVideoTargetWithNoVideoAnywhereIs404() {
    XCTAssertEqual(
      InterruptExecutor.cancel(target: "video", active: .none, checkpointedVideo: .none),
      .unknownTarget)
    let image = PublishedRender(task: finishedTask(), jobId: "img", kind: "generate")
    XCTAssertEqual(
      InterruptExecutor.cancel(target: "video", active: image, checkpointedVideo: .none),
      .unknownTarget,
      "an image render is not a video — target: video must not fall through to it")
  }

  func testAVideoTargetReachesTheCheckpointedSlotFirst() {
    let preemptor = PublishedRender(task: finishedTask(), jobId: "img", kind: "generate")
    let video = PublishedRender(
      task: finishedTask(), jobId: "queue-id", statusJobId: "status-id", kind: "video")
    XCTAssertEqual(
      InterruptExecutor.cancel(target: "video", active: preemptor, checkpointedVideo: video),
      .cancelled(jobId: "queue-id", kind: "video"),
      "interrupted_job_id reports the QUEUE id, the one /v1/queue and /health show")
  }

  /// No episode: `target: "video"` falls back to the active render when IT is
  /// the video.
  func testAVideoTargetFallsBackToAnActiveVideoRender() {
    let video = PublishedRender(
      task: finishedTask(), jobId: "queue-id", statusJobId: "status-id", kind: "video")
    XCTAssertEqual(
      InterruptExecutor.cancel(target: "video", active: video, checkpointedVideo: .none),
      .cancelled(jobId: "queue-id", kind: "video"))
  }

  /// The triple is consumed as one value (review r1, finding 1): the id
  /// reported is the id published ALONGSIDE the task that was cancelled — the
  /// two cannot come from different jobs.
  func testTheReportedIdentityComesFromTheSameTripleAsTheCancelledTask() {
    let preemptor = PublishedRender(task: finishedTask(), jobId: "preemptor", kind: "generate")
    let video = PublishedRender(task: finishedTask(), jobId: "video-job", kind: "video")
    XCTAssertEqual(
      InterruptExecutor.cancel(target: nil, active: preemptor, checkpointedVideo: video),
      .cancelled(jobId: "preemptor", kind: "generate"),
      "the default target cancels the preemptor and must report the PREEMPTOR's id")
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

  /// Review r1, finding 2: the two implementations must source the job id and
  /// kind from the SAME published triple, not from two different places.
  ///
  /// `runGenerate`'s `defer` nils the actor's `activeJobId` WITHOUT a
  /// `publishHealth()`, so for the window between a render finishing and the
  /// queue loop's own `defer` running, the actor field and the published
  /// snapshot disagree. When the sync path read `snapshot.activeJobId` and the
  /// async fallback read the actor's `activeJobId`, a job-id `target` in that
  /// window got two different answers: cancelled-and-named vs a 404.
  func testSyncAndAsyncAgreeOnAJobIdTargetImmediatelyAfterAJobEnds() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let jobId = "synthetic-window-\(UUID().uuidString)"
    let job = Task { try? await probe.enqueueSynthetic(durationMs: 3000, id: jobId) }
    try await waitUntil("the synthetic job to become active") { probe.activeJobId == jobId }

    // The exact state `runGenerate`'s defer leaves behind.
    await probe.clearActiveJobIdWithoutPublishing()

    let sync = probe.controlInterrupt(target: jobId)
    let async = await probe.coordinatorInterrupt(target: jobId)
    XCTAssertEqual(
      sync.interrupted, async.interrupted,
      "sync and async /v1/queue/interrupt must agree on whether a job-id target was interrupted")
    XCTAssertEqual(sync.unknownTarget, async.unknownTarget, "…and on whether it was a 404")
    XCTAssertEqual(sync.jobId, async.jobId, "…and on interrupted_job_id")
    XCTAssertEqual(sync.kind, async.kind, "…and on interrupted_kind")
    XCTAssertEqual(sync.jobId, jobId)
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
  ///
  /// Review r1, finding 6: the video's task is PUBLISHED as the active render
  /// first, exactly as the `.localVideo` queue case does before
  /// `runPreemptionEpisode` runs. Without that the "the video was not
  /// cancelled" assertion was vacuous — nothing could have reached the video,
  /// because the active slot was empty for the whole test. With a real
  /// published active task, a default-target interrupt genuinely COULD have
  /// hit the video (that is precisely the #362 bug), so the assertion has
  /// something to prove.
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

    // The state a real episode starts from: the VIDEO is the published active
    // render, under both of its ids (comfybox#283).
    await probe.setActiveRender(
      task: videoTask, jobId: "video-queue-id", statusJobId: "video-status-id", kind: "video")
    let videoPublication = await probe.publishedActiveRender
    XCTAssertEqual(videoPublication.jobId, "video-queue-id")

    // Sanity: with the video published as active, a default interrupt WOULD
    // reach it — so the assertion further down is not vacuous.
    XCTAssertEqual(
      probe.controlInterrupt(target: nil).jobId, "video-queue-id",
      "precondition: the video is genuinely reachable as the default target before the episode")
    _ = await videoTask.value
    XCTAssertTrue(
      videoProbe.observedCancellation,
      "precondition: a default interrupt really does cancel the video when the video is the "
        + "published active render — which is what makes the episode assertion below meaningful")

    // Re-stage: a fresh video task, published exactly as before.
    let video2 = RenderProbe()
    let videoTask2 = Task {
      video2.running = true
      let deadline = Date().addingTimeInterval(10)
      while !Task.isCancelled && Date() < deadline { usleep(2_000) }
      if Task.isCancelled { video2.observedCancellation = true }
    }
    try await waitUntil("the second fake video task to start") { video2.running }
    await probe.setActiveRender(
      task: videoTask2, jobId: "video-queue-id", statusJobId: "video-status-id", kind: "video")
    let publication2 = await probe.publishedActiveRender

    let workProbe = RenderProbe()
    let episode = Task {
      await probe.runAsPublishedActiveRender(restoringTo: publication2) {
        workProbe.running = true
        let deadline = Date().addingTimeInterval(10)
        while !Task.isCancelled && Date() < deadline { usleep(2_000) }
        if Task.isCancelled { workProbe.observedCancellation = true }
      }
    }

    // Poll a DEFAULT-target interrupt until it lands on the shielded work —
    // the episode publishes the work's own task, so once the interrupt
    // reports anything other than the video's id it has taken effect. A
    // bounded poll (rather than a fixed sleep) is what makes this
    // deterministic.
    try await waitUntil("the default-target interrupt to land on the shielded work") {
      workProbe.running && probe.controlInterrupt(target: nil).jobId != "video-queue-id"
    }
    await episode.value

    XCTAssertTrue(
      workProbe.observedCancellation, "the shielded work must be reachable as the default target")
    XCTAssertFalse(
      video2.observedCancellation,
      "the video must NOT be cancelled by a default-target interrupt during the episode — this is "
        + "comfybox#362's bug, and the video WAS reachable that way moments earlier")

    // After the episode, the video's own triple is republished — a further
    // default-target interrupt now reaches the video, and names IT.
    let after = probe.controlInterrupt(target: nil)
    XCTAssertTrue(after.interrupted, "the video's handle must be restored once the episode ends")
    XCTAssertEqual(after.jobId, "video-queue-id")
    XCTAssertEqual(after.kind, "video")
    _ = await videoTask2.value
    XCTAssertTrue(video2.observedCancellation)
  }

  /// Review r1, finding 3, end to end through the real publication path: a
  /// checkpointed video is reachable by its `/v1/video/status/{id}` id, not
  /// just by the queue id `/v1/queue` shows.
  func testTheCheckpointedVideosStatusIdAlsoReachesIt() async throws {
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
    await probe.setCheckpointedVideo(
      task: videoTask, jobId: "video-queue-3", statusJobId: "video-status-3")

    let preemptorJobId = "preemptor-job-3"
    let job = Task { try await probe.enqueueSynthetic(durationMs: 3000, id: preemptorJobId) }
    try await waitUntil("the preemptor stand-in to become active") {
      probe.activeJobId == preemptorJobId
    }

    // The id a client that submitted the video actually holds.
    let result = probe.controlInterrupt(target: "video-status-3")
    XCTAssertTrue(result.interrupted, "the /v1/video/status/{id} id must reach the video too")
    XCTAssertEqual(
      result.jobId, "video-queue-3",
      "interrupted_job_id reports the QUEUE id regardless of which id was used to target")
    XCTAssertEqual(result.kind, "video")

    _ = await videoTask.value
    XCTAssertTrue(videoProbe.observedCancellation)
    let succeeded = try await job.value
    XCTAssertTrue(succeeded, "the active preempting job must be untouched")
  }

  // MARK: - Route level: request bytes in, response bytes out

  /// Review r1, finding 5: the sync `/v1/queue/interrupt` route itself —
  /// `InterruptRoute.decodeTarget` → `LiveHealthState.cancelActiveRender` →
  /// `InterruptRoute.response`, the exact chain `syncInterruptResponse` runs.
  /// Asserted on the SERIALISED JSON, because the snake_case conversion and
  /// the status code are the part of the contract clients actually see.
  private func routeJSON(_ response: HTTPResponse) throws -> [String: Any] {
    try XCTUnwrap(
      JSONSerialization.jsonObject(with: response.body) as? [String: Any],
      "interrupt response body was not a JSON object")
  }

  /// The pre-#362 shape: no body at all, nothing running. This body must be
  /// byte-identical to what the route returned before `target` existed —
  /// every existing client (none of which sends `target`) sees no change.
  func testTheNothingToCancelBodyIsByteIdenticalToPre362() throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    // The EXACT payload type the route encoded before #362 existed
    // (`struct InterruptResult { success; interrupted }`), through the same
    // `HTTPResponse.json` encoder — so this compares real bytes to real
    // bytes rather than to a hand-written string whose key order could drift
    // with Foundation.
    struct Pre362InterruptResult: Encodable {
      let success: Bool
      let interrupted: Bool
    }
    let legacy = HTTPResponse.json(
      status: 200, payload: Pre362InterruptResult(success: true, interrupted: false))

    let response = probe.syncInterruptRoute(body: Data())
    XCTAssertEqual(response.status, legacy.status)
    XCTAssertEqual(
      response.body, legacy.body,
      "nothing-to-cancel must be byte-identical to pre-#362: the additive fields ABSENT (not "
        + "null), so a client that only reads success/interrupted sees no change. Got "
        + String(decoding: response.body, as: UTF8.self))
  }

  /// Body decode: an absent body, `{}`, `{"target": null}` and a body that is
  /// not JSON at all must all mean the default target — the route has never
  /// required a body and still does not.
  func testEveryFormOfAbsentTargetDecodesToTheDefault() throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    for body in [Data(), Data("{}".utf8), Data(#"{"target":null}"#.utf8), Data("not json".utf8)] {
      let response = probe.syncInterruptRoute(body: body)
      XCTAssertEqual(
        response.status, 200,
        "body \(String(decoding: body, as: UTF8.self).debugDescription) must resolve to the "
          + "default target (200 + interrupted:false), not a 404")
      let json = try routeJSON(response)
      XCTAssertEqual(json["interrupted"] as? Bool, false)
    }
    XCTAssertNil(InterruptRoute.decodeTarget(from: Data(#"{"target":null}"#.utf8)))
    XCTAssertEqual(InterruptRoute.decodeTarget(from: Data(#"{"target":"video"}"#.utf8)), "video")
  }

  /// `target: "video"` with no episode and no video running at all: a 404,
  /// not a silent `interrupted: false`.
  func testAVideoTargetWithNoEpisodeIs404OnTheWire() throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let response = probe.syncInterruptRoute(body: Data(#"{"target":"video"}"#.utf8))
    XCTAssertEqual(response.status, 404)
    let json = try routeJSON(response)
    XCTAssertEqual(json["success"] as? Bool, false)
    XCTAssertEqual(json["interrupted"] as? Bool, false)
  }

  func testAnUnknownJobIdIs404OnTheWire() throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let response = probe.syncInterruptRoute(body: Data(#"{"target":"no-such-job"}"#.utf8))
    XCTAssertEqual(response.status, 404)
    XCTAssertEqual(try routeJSON(response)["interrupted"] as? Bool, false)
  }

  /// The additive fields go out SNAKE_CASE (`HTTPResponse.json` applies
  /// `.convertToSnakeCase`) — a camelCase leak here would silently break
  /// every client reading them.
  func testTheAdditiveFieldsAreSnakeCaseOnTheWire() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let jobId = "synthetic-wire-\(UUID().uuidString)"
    let job = Task { try? await probe.enqueueSynthetic(durationMs: 3000, id: jobId) }
    try await waitUntil("the synthetic job to become active") { probe.activeJobId == jobId }

    let response = probe.syncInterruptRoute(body: Data(#"{"target":"active"}"#.utf8))
    XCTAssertEqual(response.status, 200)
    XCTAssertEqual(response.contentType, "application/json")
    let json = try routeJSON(response)
    XCTAssertEqual(json["success"] as? Bool, true)
    XCTAssertEqual(json["interrupted"] as? Bool, true)
    XCTAssertEqual(json["interrupted_job_id"] as? String, jobId)
    XCTAssertEqual(json["interrupted_kind"] as? String, "synthetic")
    XCTAssertNil(json["interruptedJobId"], "the wire is snake_case, not camelCase")
    XCTAssertNil(json["interruptedKind"])
    _ = await job.value
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
