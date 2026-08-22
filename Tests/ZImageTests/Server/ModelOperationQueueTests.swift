// ModelOperationQueueTests.swift — K-FIX-1 / Codex engine review C2.
//
// `/v1/model/load`, `/v1/model/activate` and `/v1/model/unload` called the
// coordinator's pool methods DIRECTLY, and the `wait: false` arm additionally
// started a detached `Task`. Actor isolation does not serialize an async
// method across its awaits: while `poolLoad` awaited `ModelPool.load`, the
// coordinator was free to dequeue and start a render, and the load could then
// allocate a second ~22 GB transformer, release `active.box` and call
// `GPU.clearCache()` UNDER it — use-after-release, corrupted output, Metal
// failure or OOM. `ModelPool.evictIfNeeded`'s own comment asserts that
// "callers are serialized with renders on the coordinator queue"; these three
// routes were the callers that were not.
//
// Every mutating pool operation now goes through the SAME FIFO as renders,
// LoRA swaps and the ComfyBridge model switch. These tests are the barrier
// proof: with a job occupying the queue, a model operation does not BEGIN
// until that job exits.
//
// The probe (`WarmServerQueueProbe`) drives the real coordinator, so it
// persists a queue snapshot and reads a pause sentinel — `isolateComfyBoxStateDirectory()`
// points both at a per-test temp directory so the live engine's are untouched.

import Foundation
import XCTest

@testable import ZImage

private final class Recorder: @unchecked Sendable {
  private let lock = NSLock()
  private var events: [String] = []
  func record(_ event: String) { lock.lock(); events.append(event); lock.unlock() }
  var all: [String] { lock.lock(); defer { lock.unlock() }; return events }
  var count: Int { lock.lock(); defer { lock.unlock() }; return events.count }
}

/// A one-shot gate the test opens when it is ready.
private final class Gate: @unchecked Sendable {
  private let semaphore = DispatchSemaphore(value: 0)
  func open() { semaphore.signal() }
  func waitUntilOpen() { semaphore.wait() }
}

/// Holds the shutdown call's result across the polling loop.
private actor ShutdownOutcome {
  private(set) var value: Bool?
  func record(_ response: Bool?) { value = response }
}

final class ModelOperationQueueTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    // Precondition for the whole file: the probe drives a REAL coordinator,
    // which persists a queue snapshot and reads a pause sentinel — neither of
    // which is the test's to touch. The shared helper redirects both and
    // asserts the redirection took effect.
    try isolateComfyBoxStateDirectory()
  }

  /// An operation on an empty pool fails fast with `modelNotInPool` — the
  /// cheapest observable "this ran", with no weights and no GPU.
  private func expectNotInPool(_ error: Error, _ message: String = "") {
    guard case ModelPoolError.modelNotInPool = error else {
      return XCTFail("expected modelNotInPool, got \(error). \(message)")
    }
  }

  // MARK: - The barrier

  /// THE C2 proof: while a job occupies the queue, a `/v1/model/load` does not
  /// begin. Before the fix the route called `poolLoad` directly and the load
  /// would have started immediately, concurrently with the render.
  func testAModelOperationDoesNotBeginUntilTheActiveJobExits() async throws {
    let probe = makeQueueProbe()
    let order = Recorder()
    let renderStarted = Gate()
    let releaseRender = Gate()

    // A job occupying the queue, standing in for a render.
    async let renderDone: Bool = probe.enqueueFakeRender {
      order.record("render-start")
      renderStarted.open()
      await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async {
          releaseRender.waitUntilOpen()
          c.resume()
        }
      }
      order.record("render-end")
      return true
    }

    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async { renderStarted.waitUntilOpen(); c.resume() }
    }
    XCTAssertEqual(order.all, ["render-start"])

    // Now post the model operation. It must not run.
    let operationTask = Task {
      do {
        _ = try await probe.enqueueModelOperation(.activate(modelId: "no-such-model"))
        order.record("model-op-succeeded")
      } catch {
        order.record("model-op-failed")
      }
    }

    // Give it every chance to run concurrently: the unfixed path reached
    // `ModelPool.activate` on the first await and threw immediately.
    try await Task.sleep(nanoseconds: 300_000_000)
    XCTAssertEqual(
      order.all, ["render-start"],
      "a pool mutation began while a job was active — the C2 race")

    releaseRender.open()
    _ = try await renderDone
    await operationTask.value

    XCTAssertEqual(order.all, ["render-start", "render-end", "model-op-failed"])
  }

  /// The operation still RUNS once the queue drains — the barrier is not a
  /// silent drop. The error is the pool's own, propagated to the caller so the
  /// route can map it to a status code.
  func testTheOperationRunsAndItsErrorReachesTheCaller() async throws {
    let probe = makeQueueProbe()
    var caught: Error?
    do {
      _ = try await probe.enqueueModelOperation(.unload(modelId: "no-such-model"))
    } catch {
      caught = error
    }
    expectNotInPool(try XCTUnwrap(caught))
  }

  /// All three mutating operations go through the queue — load and activate
  /// were the ones Codex named, unload releases weights the same way.
  func testLoadActivateAndUnloadAllQueue() async throws {
    let probe = makeQueueProbe()
    let order = Recorder()
    let renderStarted = Gate()
    let releaseRender = Gate()

    async let renderDone: Bool = probe.enqueueFakeRender {
      renderStarted.open()
      await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async { releaseRender.waitUntilOpen(); c.resume() }
      }
      order.record("render-end")
      return true
    }
    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async { renderStarted.waitUntilOpen(); c.resume() }
    }

    let operations: [ModelOperation] = [
      .load(modelSpec: "/definitely/not/a/model", quantization: nil, activate: true),
      .activate(modelId: "no-such-model"),
      .unload(modelId: "no-such-model"),
    ]
    let tasks = operations.map { op in
      Task {
        _ = try? await probe.enqueueModelOperation(op)
        order.record("done:\(op.kind)")
      }
    }

    try await Task.sleep(nanoseconds: 250_000_000)
    XCTAssertTrue(order.all.isEmpty, "no pool mutation may begin under an active job: \(order.all)")

    releaseRender.open()
    _ = try await renderDone
    for task in tasks { await task.value }

    XCTAssertEqual(order.all.first, "render-end", "the active job finished first")
    XCTAssertEqual(Set(order.all.dropFirst()),
                   ["done:model_load", "done:model_activate", "done:model_unload"])
  }

  // MARK: - `wait: false` is a tracked job, not a detached Task

  /// The worst version of the race: `/v1/model/load` with `wait: false` used
  /// to start a detached `Task` that ran `poolLoad` OUTSIDE the queue
  /// entirely, so nothing in the system knew it was running. It is now an
  /// ordinary FIFO job with an id the caller gets back and `/v1/queue` lists.
  func testAsyncLoadIsAQueuedJobWithAnIdRatherThanADetachedTask() async throws {
    let probe = makeQueueProbe()
    let renderStarted = Gate()
    let releaseRender = Gate()

    async let renderDone: Bool = probe.enqueueFakeRender {
      renderStarted.open()
      await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async { releaseRender.waitUntilOpen(); c.resume() }
      }
      return true
    }
    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async { renderStarted.waitUntilOpen(); c.resume() }
    }

    let jobId = try await probe.enqueueModelOperationDetached(
      .load(modelSpec: "/definitely/not/a/model", quantization: nil, activate: false))
    XCTAssertFalse(jobId.isEmpty)

    // It is WAITING — visible in the queue, behind the active job.
    XCTAssertEqual(probe.pendingJobKinds(), ["model_load"])

    releaseRender.open()
    _ = try await renderDone

    // A detached operation has no continuation to await, so the test must
    // drain the queue itself before returning (New-2): teardown unsets
    // COMFYBOX_STATE_DIR, and a job still executing would then let the loop's
    // persistQueueState() delete the LIVE snapshot. `enqueueModelOperation` is
    // FIFO-behind everything already queued, so awaiting one drains the rest.
    // (`/definitely/not/a/model` goes through Hub resolution — the window is
    // seconds wide, not microseconds.)
    try await drain(probe)
    XCTAssertTrue(probe.isDrained)
  }

  /// The queue lists model operations under their own kinds, so an operator
  /// looking at `/v1/queue` can see (and cancel) one.
  func testQueueListsEachOperationKind() async throws {
    let probe = makeQueueProbe()
    let renderStarted = Gate()
    let releaseRender = Gate()

    async let renderDone: Bool = probe.enqueueFakeRender {
      renderStarted.open()
      await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async { releaseRender.waitUntilOpen(); c.resume() }
      }
      return true
    }
    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async { renderStarted.waitUntilOpen(); c.resume() }
    }

    _ = try await probe.enqueueModelOperationDetached(
      .load(modelSpec: "/x", quantization: nil, activate: true))
    _ = try await probe.enqueueModelOperationDetached(.activate(modelId: "y"))
    _ = try await probe.enqueueModelOperationDetached(.unload(modelId: "z"))

    XCTAssertEqual(probe.pendingJobKinds(), ["model_load", "model_activate", "model_unload"])

    releaseRender.open()
    _ = try await renderDone
    try await drain(probe)                       // New-2 — see above.
    XCTAssertTrue(probe.isDrained)
  }

  // MARK: - Summaries

  /// Wait for everything already queued by enqueueing one more operation
  /// behind it: the FIFO cannot reach this one until the rest have run.
  private func drain(_ probe: WarmServerQueueProbe) async throws {
    _ = try? await probe.enqueueModelOperation(.unload(modelId: "drain-sentinel"))
  }

  /// The predicate the teardown guard blocks on. If `isDrained` were true
  /// while work is in flight, the guard would unset `COMFYBOX_STATE_DIR`
  /// under a running loop and the next `persistQueueState()` would delete the
  /// LIVE snapshot — so the predicate is pinned here, not assumed (New-2).
  func testIsDrainedIsFalseWhileWorkIsInFlightAndTrueAfter() async throws {
    let probe = makeQueueProbe()
    XCTAssertTrue(probe.isDrained, "a fresh queue is drained")

    let renderStarted = Gate()
    let releaseRender = Gate()
    async let renderDone: Bool = probe.enqueueFakeRender {
      renderStarted.open()
      await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async { releaseRender.waitUntilOpen(); c.resume() }
      }
      return true
    }
    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async { renderStarted.waitUntilOpen(); c.resume() }
    }

    XCTAssertFalse(probe.isDrained, "a job is ACTIVE — teardown must not unset the override yet")
    _ = try await probe.enqueueModelOperationDetached(.unload(modelId: "queued-behind"))
    XCTAssertFalse(probe.isDrained, "…and one is pending behind it")
    XCTAssertEqual(probe.pendingCount, 1)

    releaseRender.open()
    _ = try await renderDone
    try await drain(probe)
    XCTAssertTrue(probe.isDrained, "everything ran — now the override may be unset")
  }

  // MARK: - Pause (K-FIX-1 round 2, New-1)

  /// A PAUSED queue must not wedge `/v1/model/load|activate|unload`.
  ///
  /// Routing model ops through the FIFO (C2) put them behind a `processLoop`
  /// that returns immediately while `isPaused`, and `enqueueModelOperation`
  /// parks a continuation with no timeout — so on a paused engine (which is
  /// how Todd frees the GPU for a deploy, and the pause SURVIVES restarts via
  /// the sentinel) every synchronous model op hung until the client gave up,
  /// and `wait: false` returned a job id for a load that never started. Before
  /// this wave those routes worked while paused, because they called the pool
  /// directly.
  ///
  /// Ruling: "pause" means no RENDERS. A model op runs while paused; the FIFO
  /// still serialises it against any in-flight render, which is C2's actual
  /// guarantee.
  func testAModelOperationCompletesWhileTheQueueIsPaused() async throws {
    let probe = makeQueueProbe()
    await probe.setPaused(true)

    var caught: Error?
    do {
      _ = try await probe.enqueueModelOperation(.unload(modelId: "no-such-model"))
    } catch {
      caught = error
    }
    expectNotInPool(try XCTUnwrap(caught, "the model op must RUN while paused, not park"))

    // Still paused — running the op did not resume the queue as a side effect.
    XCTAssertTrue(probe.isPaused)
    await probe.setPaused(false)
    try await drain(probe)
  }

  /// The other half of the ruling: a render (anything that is not a model
  /// operation) stays parked while paused, and runs on resume.
  func testARenderStaysParkedWhileTheQueueIsPaused() async throws {
    let probe = makeQueueProbe()
    let order = Recorder()
    await probe.setPaused(true)

    let renderTask = Task {
      _ = try? await probe.enqueueFakeRender {
        order.record("render")
        return true
      }
    }

    // The model op runs straight past the parked render.
    _ = try? await probe.enqueueModelOperation(.unload(modelId: "no-such-model"))
    order.record("model-op")

    try await Task.sleep(nanoseconds: 250_000_000)
    XCTAssertEqual(order.all, ["model-op"], "a render must NOT run while paused")
    XCTAssertEqual(probe.pendingJobKinds(), ["model_switch"], "it is still queued, not dropped")

    await probe.setPaused(false)
    await renderTask.value
    XCTAssertEqual(order.all, ["model-op", "render"])
    try await drain(probe)
  }

  /// Parked renders keep their relative order across the pause: a model
  /// operation passing them does not reshuffle the queue.
  func testParkedRendersKeepTheirOrderWhenUnpaused() async throws {
    let probe = makeQueueProbe()
    let order = Recorder()
    await probe.setPaused(true)

    let first = Task { _ = try? await probe.enqueueFakeRender { order.record("A"); return true } }
    try await Task.sleep(nanoseconds: 50_000_000)
    let second = Task { _ = try? await probe.enqueueFakeRender { order.record("B"); return true } }
    try await Task.sleep(nanoseconds: 50_000_000)
    XCTAssertEqual(probe.pendingJobKinds(), ["model_switch", "model_switch"])

    _ = try? await probe.enqueueModelOperation(.unload(modelId: "no-such-model"))
    XCTAssertEqual(order.all, [], "neither render ran")
    XCTAssertEqual(probe.pendingJobKinds(), ["model_switch", "model_switch"], "both still queued")

    await probe.setPaused(false)
    await first.value
    await second.value
    XCTAssertEqual(order.all, ["A", "B"], "FIFO order across the pause")
    try await drain(probe)
  }

  // MARK: - Pause exemption for shutdown + the capacity gate (WP-E8 hygiene)

  /// `/v1/shutdown` on a PAUSED engine bricked it.
  ///
  /// `enqueueShutdown` sets `shuttingDown = true` BEFORE appending the job,
  /// and `.shutdown` was not pause-exempt — so the job parked behind the
  /// pause gate, the caller's continuation never resumed, and every later
  /// enqueue threw `.shuttingDown`. The engine could then only be recovered
  /// with SIGKILL. The handler is a bare continuation resume with no GPU
  /// work, so "pause means no RENDERS" covers it exactly as it covers a
  /// model operation.
  func testShutdownCompletesWhileTheQueueIsPaused() async throws {
    let probe = makeQueueProbe()
    await probe.setPaused(true)

    let outcome = ShutdownOutcome()
    let posted = Task { await outcome.record(try? await probe.enqueueShutdown()) }

    // Bounded: a parked continuation never resumes, so poll rather than await.
    let deadline = Date().addingTimeInterval(3)
    while await outcome.value == nil, Date() < deadline {
      try await Task.sleep(nanoseconds: 20_000_000)
    }
    // Deliberately NOT `await posted.value`: with the bug present that task
    // is parked on a continuation that never resumes, and awaiting it would
    // hang the whole suite instead of failing this test.
    let recorded = await outcome.value
    if recorded != nil { _ = await posted.result }
    let success = try XCTUnwrap(
      recorded,
      "/v1/shutdown parked under the pause gate — the caller hangs forever and "
        + "every later enqueue throws .shuttingDown (only SIGKILL recovers)")
    XCTAssertTrue(success)

    // Pausing is unchanged by it, and the queue is empty.
    XCTAssertTrue(probe.isPaused)
    XCTAssertTrue(probe.isDrained)
    await probe.setPaused(false)
  }

  /// A shutdown posted while paused does not resurrect the parked renders:
  /// they stay queued (the process is going away, not draining).
  func testShutdownWhilePausedDoesNotRunParkedRenders() async throws {
    let probe = makeQueueProbe()
    let order = Recorder()
    await probe.setPaused(true)

    let render = Task { _ = try? await probe.enqueueFakeRender { order.record("render"); return true } }
    try await Task.sleep(nanoseconds: 100_000_000)
    XCTAssertEqual(probe.pendingJobKinds(), ["model_switch"])

    _ = try? await probe.enqueueShutdown()
    XCTAssertEqual(order.all, [], "a render must not run because a shutdown passed it")
    XCTAssertEqual(probe.pendingJobKinds(), ["model_switch"], "still parked, not dropped")

    // Clean up: resume so the parked render finishes before teardown.
    await probe.setPaused(false)
    await render.value
    XCTAssertEqual(order.all, ["render"])
  }

  /// A paused queue holding `maxPendingRequests` parked renders answered
  /// `/v1/model/unload` with `queueFull` — the same "cannot free the GPU
  /// while paused" wedge as the parked continuation, one layer up. Model
  /// operations are operator actions, bounded in number; they bypass the
  /// capacity gate.
  func testModelOperationsBypassTheCapacityGate() async throws {
    let probe = makeQueueProbe(maxPendingRequests: 10)
    let order = Recorder()
    await probe.setPaused(true)

    var renders: [Task<Void, Never>] = []
    for index in 0..<10 {
      renders.append(Task {
        _ = try? await probe.enqueueFakeRender { order.record("r\(index)"); return true }
      })
    }
    let deadline = Date().addingTimeInterval(5)
    while probe.pendingCount < 10, Date() < deadline {
      try await Task.sleep(nanoseconds: 20_000_000)
    }
    XCTAssertEqual(probe.pendingCount, 10, "the queue must be at capacity for this test to mean anything")

    // Synchronous seam (/v1/model/unload, /v1/model/activate, wait:true load).
    var caught: Error?
    do {
      _ = try await probe.enqueueModelOperation(.unload(modelId: "no-such-model"))
    } catch {
      caught = error
    }
    let error = try XCTUnwrap(caught)
    if WarmServerQueueProbe.isQueueFull(error) {
      XCTFail("a model operation was refused with queueFull on a full, paused queue — "
        + "the operator cannot free the GPU")
    }
    expectNotInPool(error, "the operation must RUN, not be refused")

    // Detached seam (/v1/model/load, wait:false).
    let jobId = try await probe.enqueueModelOperationDetached(.unload(modelId: "no-such-model-2"))
    XCTAssertFalse(jobId.isEmpty)

    XCTAssertEqual(order.all, [], "no parked render ran")

    await probe.setPaused(false)
    for render in renders { await render.value }
    try await drain(probe)
    XCTAssertEqual(order.all.count, 10, "every parked render still ran on resume")
  }

  /// The gate stays on for RENDERS — bypassing it for model ops must not
  /// turn the queue into an unbounded buffer for generate traffic.
  func testTheCapacityGateStillRefusesRendersWhenFull() async throws {
    let probe = makeQueueProbe(maxPendingRequests: 2)
    await probe.setPaused(true)

    var renders: [Task<Void, Never>] = []
    for _ in 0..<2 {
      renders.append(Task { _ = try? await probe.enqueueFakeRender { true } })
    }
    let deadline = Date().addingTimeInterval(5)
    while probe.pendingCount < 2, Date() < deadline {
      try await Task.sleep(nanoseconds: 20_000_000)
    }
    XCTAssertEqual(probe.pendingCount, 2)

    var caught: Error?
    do {
      _ = try await probe.enqueueFakeRender { true }
    } catch {
      caught = error
    }
    XCTAssertTrue(
      WarmServerQueueProbe.isQueueFull(try XCTUnwrap(caught)),
      "a render past capacity must still be refused with queueFull, got \(caught as Any)")

    await probe.setPaused(false)
    for render in renders { await render.value }
    try await drain(probe)
  }

  func testOperationSummariesNameTheModel() {
    XCTAssertTrue(
      ModelOperation.load(modelSpec: "krea2-raw", quantization: "q8", activate: true)
        .summary.contains("krea2-raw"))
    XCTAssertTrue(
      ModelOperation.load(modelSpec: "krea2-raw", quantization: nil, activate: true)
        .summary.contains("activate"))
    XCTAssertTrue(ModelOperation.activate(modelId: "kroma-v0.2").summary.contains("kroma-v0.2"))
    XCTAssertTrue(ModelOperation.unload(modelId: "kroma-v0.2").summary.contains("kroma-v0.2"))
  }
}
