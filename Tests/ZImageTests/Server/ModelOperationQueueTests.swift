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
// persists a queue snapshot and reads a pause sentinel — `COMFYBOX_STATE_DIR`
// points both at a temp directory so the live engine's are untouched.

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

final class ModelOperationQueueTests: XCTestCase {

  private var stateDir: URL!

  override func setUpWithError() throws {
    stateDir = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("comfybox-queue-probe-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: stateDir, withIntermediateDirectories: true)
    setenv("COMFYBOX_STATE_DIR", stateDir.path, 1)
    // Precondition for the whole file: the probe must not be reading (or
    // clearing) the live engine's state.
    XCTAssertEqual(QueueStateStore.stateDirectory.path, stateDir.path)
  }

  override func tearDownWithError() throws {
    unsetenv("COMFYBOX_STATE_DIR")
    try? FileManager.default.removeItem(at: stateDir)
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
    let probe = WarmServerQueueProbe()
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
    let probe = WarmServerQueueProbe()
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
    let probe = WarmServerQueueProbe()
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
    let probe = WarmServerQueueProbe()
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
  }

  /// The queue lists model operations under their own kinds, so an operator
  /// looking at `/v1/queue` can see (and cancel) one.
  func testQueueListsEachOperationKind() async throws {
    let probe = WarmServerQueueProbe()
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
  }

  // MARK: - Summaries

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
