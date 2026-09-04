import XCTest
@testable import ZImage

/// comfybox#322 — the coordinator half of mid-render video cancellation,
/// pinned through the real `.localVideo` queue case (review r1, items 1 + 2a).
///
/// These are the tests that would have caught the two root causes:
///
///   1. `.localVideo` was the ONLY render case that never published a retained
///      render task, so `activeRenderTask` stayed nil for the whole 5-60 minute
///      clip and `/v1/queue/interrupt` answered `interrupted: false`.
///      `testInterruptIsANoOpBeforeTheFixIsPresent…` pins that
///      `controlInterrupt()` returns TRUE during a video job and that the body
///      actually observes the cancellation.
///
///   2. Interrupting the video used to cancel the preempting IMAGE job as
///      collateral, because the preemption episode was awaited inside the
///      video's render task and `runGenerate` is cancellation-aware since #304.
///      `testShieldedWorkIgnoresTheCallersCancellation` exercises the exact
///      production shield the episode wraps its image job in.
///
/// The #218 admission gate wants ~65-80GB of genuinely free RAM, which no unit
/// test can arrange on a machine that is also serving production, so these use
/// the DEBUG `bypassVideoAdmission()` seam. Admission itself is covered with
/// injected byte figures by `HeavyModelAdmissionTests`; what is under test here
/// is the queue case's cancellation wiring.
final class LocalVideoInterruptTests: XCTestCase {

  /// Cross-task state for a synchronous render body. `@unchecked Sendable` +
  /// a lock, the same shape the coordinator's own trackers use.
  private final class RenderProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var _running = false
    private var _observedCancellation = false
    private var _finishedNormally = false

    var running: Bool {
      get { lock.lock(); defer { lock.unlock() }; return _running }
      set { lock.lock(); _running = newValue; lock.unlock() }
    }
    var observedCancellation: Bool {
      get { lock.lock(); defer { lock.unlock() }; return _observedCancellation }
      set { lock.lock(); _observedCancellation = newValue; lock.unlock() }
    }
    var finishedNormally: Bool {
      get { lock.lock(); defer { lock.unlock() }; return _finishedNormally }
      set { lock.lock(); _finishedNormally = newValue; lock.unlock() }
    }
  }

  private static let dummyResult = LTX2VideoResult(
    outputPath: "/dev/null", frameCount: 0, durationSeconds: 0, elapsedSeconds: 0)

  /// Poll until `condition`, or fail. Renders here are synchronous bodies on
  /// the coordinator's own serial executor, so the test task must poll rather
  /// than await anything the render owns.
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

  // MARK: - Root cause 1: the interrupt had no handle to pull

  /// The pin. Before comfybox#322 every assertion here failed the same way:
  /// `controlInterrupt()` returned false because `.localVideo` never published
  /// a render task, and the body ran to completion with `Task.isCancelled`
  /// never true.
  func testLocalVideoPublishesACancellableRenderTask() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()
    await probe.bypassVideoAdmission()

    XCTAssertFalse(
      probe.controlInterrupt(),
      "nothing is rendering yet — there is no task to cancel")

    let render = RenderProbe()
    let job = Task { () -> Result<LTX2VideoResult, Error> in
      do {
        return .success(try await probe.enqueueLocalVideo { _ in
          render.running = true
          // Stand in for a denoise loop: spin at a "step boundary" until the
          // ambient task is cancelled, exactly as the LTX-2 loops now do.
          let deadline = Date().addingTimeInterval(10)
          while !Task.isCancelled && Date() < deadline {
            usleep(2_000)
          }
          if Task.isCancelled {
            render.observedCancellation = true
            throw CancellationError()
          }
          render.finishedNormally = true
          return .completed(Self.dummyResult)
        })
      } catch {
        return .failure(error)
      }
    }

    try await waitUntil("the video body to start") { render.running }

    // ROOT CAUSE 1: this returned false for the whole render before the fix.
    XCTAssertTrue(
      probe.controlInterrupt(),
      "a running .localVideo job must publish a cancellable render task")

    let outcome = await job.value
    switch outcome {
    case .success:
      XCTFail("an interrupted render must not report success")
    case .failure(let error):
      XCTAssertTrue(
        WarmServerQueueProbe.isInterrupted(error),
        "expected the named interrupt, got \(error)")
    }
    XCTAssertTrue(
      render.observedCancellation,
      "the render body must observe Task.isCancelled — this is what the LTX-2 step boundaries read")
    XCTAssertFalse(render.finishedNormally, "the render must not have run to completion")
  }

  /// The interrupt is not a permanent state: the queue keeps going, and the
  /// NEXT video job runs to completion uncancelled. (An interrupt that leaked
  /// into the following job would look exactly like the wedge #339 chases.)
  func testQueueProceedsAfterAnInterruptedVideo() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()
    await probe.bypassVideoAdmission()

    let first = RenderProbe()
    let firstJob = Task { () -> Error? in
      do {
        _ = try await probe.enqueueLocalVideo { _ in
          first.running = true
          let deadline = Date().addingTimeInterval(10)
          while !Task.isCancelled && Date() < deadline { usleep(2_000) }
          if Task.isCancelled { throw CancellationError() }
          return .completed(Self.dummyResult)
        }
        return nil
      } catch { return error }
    }
    try await waitUntil("the first video body to start") { first.running }
    XCTAssertTrue(probe.controlInterrupt())
    let firstError = await firstJob.value
    XCTAssertTrue(WarmServerQueueProbe.isInterrupted(firstError ?? CancellationError()))

    let second = RenderProbe()
    let result = try await probe.enqueueLocalVideo { _ in
      second.running = true
      XCTAssertFalse(
        Task.isCancelled,
        "the previous job's interrupt must not carry into the next render task")
      second.finishedNormally = true
      return .completed(Self.dummyResult)
    }
    XCTAssertEqual(result.outputPath, Self.dummyResult.outputPath)
    XCTAssertTrue(second.finishedNormally)
  }

  // MARK: - Root cause 2 (Critical): the preemptor must not be collateral

  /// `runShieldedFromCancellation` is the PRODUCTION function the preemption
  /// episode wraps its image job in — this calls that same function, not a
  /// copy of its shape.
  ///
  /// Before the r1 fix the episode used a plain `await runGenerate(…)`, which
  /// is a structured call inside the video's render task: cancelling the video
  /// cancelled the image render mid-denoise, which is exactly the "died as
  /// cancel collateral with an opaque error" incident #322 exists to end.
  func testShieldedWorkIgnoresTheCallersCancellation() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let work = RenderProbe()
    var capture: Task<Void, Never>?
    let caller = Task {
      await probe.runShieldedFromCancellation {
        work.running = true
        // The "image render" starts, then the operator interrupts the VIDEO.
        capture?.cancel()
        // A shielded task inherits no cancellation, so this must stay false
        // for every one of these checks — the same read `Krea2DenoiseLoop`
        // makes at each of its step boundaries since #304.
        for _ in 0..<50 {
          if Task.isCancelled { work.observedCancellation = true }
          try? await Task.sleep(nanoseconds: 1_000_000)
        }
        work.finishedNormally = true
      }
    }
    capture = caller
    await caller.value

    XCTAssertTrue(work.running)
    XCTAssertFalse(
      work.observedCancellation,
      "the preempting image job must NEVER see the video's cancellation")
    XCTAssertTrue(
      work.finishedNormally,
      "the shield must wait for the preemptor to finish, not abandon it")
    XCTAssertTrue(caller.isCancelled, "…while the caller itself really was cancelled")
  }

  /// The other half of the Critical ruling: once the (protected) preemptor has
  /// finished, an interrupted video is ABANDONED, not resumed. Resuming a clip
  /// the operator asked to kill would leave the render running for another 20
  /// minutes and defeat the interrupt entirely.
  func testEpisodeDispositionAbandonsAnInterruptedVideo() {
    XCTAssertEqual(
      LTX2PreemptionEpisode.disposition(videoInterrupted: false), .resumeVideo,
      "#1479 preemption is unchanged when nothing was interrupted")
    XCTAssertEqual(
      LTX2PreemptionEpisode.disposition(videoInterrupted: true), .abandonVideo,
      "an interrupted video must not come back from the dead after the preemptor finishes")
  }
}
