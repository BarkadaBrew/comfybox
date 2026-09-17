import MLX
import XCTest

@testable import ZImage

/// Codex review of #465 (finding 2): the PRODUCTION hold wiring — the
/// `raiseHold` closure (preemption flag + #1479 signal + watchdog) and the
/// table's `onEmpty` cleanup — driven through the slot route exactly as
/// `POST/DELETE /v1/queue/inference-slot` run it, so a stranded
/// `preemptionInFlight` or a stale signal cannot regress silently.
final class InferenceSlotWiringTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    try isolateComfyBoxStateDirectory()
  }

  final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var v = false
    func set() { lock.lock(); v = true; lock.unlock() }
    var value: Bool { lock.lock(); defer { lock.unlock() }; return v }
  }

  private func waitUntil(_ what: String, timeout: TimeInterval = 10, _ cond: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while !cond() {
      if Date() > deadline { XCTFail("timed out waiting for \(what)"); return }
      try await Task.sleep(nanoseconds: 10_000_000)
    }
  }

  private func acquire(_ ctx: InferenceSlotRouteContext, holder: String) -> String? {
    let body = try? JSONSerialization.data(withJSONObject: ["holder": holder, "ttl_sec": 60])
    let r = InferenceSlotRoute.handle(method: "POST", path: "/v1/queue/inference-slot", body: body ?? Data(), ctx: ctx)
    return r.payload["slot_id"] as? String
  }

  private func release(_ ctx: InferenceSlotRouteContext, _ id: String) -> Bool {
    let r = InferenceSlotRoute.handle(method: "DELETE", path: "/v1/queue/inference-slot/\(id)", body: Data(), ctx: ctx)
    return r.payload["released"] as? Bool ?? false
  }

  private static func checkpoint() -> LTX2ResumeState {
    LTX2ResumeState(
      videoLatents: MLXArray.zeros([1]), stepIndex: 1, sigmas: [1.0, 0.5, 0.0],
      phase: .baseDenoise, chunkIndex: 0, seed: 1,
      audioLatents: nil, audioNoiseKey: nil,
      configFingerprint: "test", refineCleanLatents: nil)
  }

  func testReleasingBeforeTheVideoYieldsClearsHoldFlagAndSignal() async throws {
    let probe = makeQueueProbe()
    let ctx = probe.slotRouteContext(videoRendering: { true }, watchdogSec: 60)

    let id = try XCTUnwrap(acquire(ctx, holder: "test:chat"))
    XCTAssertTrue(probe.inferenceHold.isRequested, "a slot during a video requests the hold")
    XCTAssertTrue(probe.preemptionInFlight.get(), "the hold owns the one preemption")
    XCTAssertTrue(probe.preemptionSignal.isRaised, "the video is asked to checkpoint")

    XCTAssertTrue(release(ctx, id))
    XCTAssertFalse(probe.inferenceHold.isRequested, "an unclaimed request is withdrawn on the last release")
    XCTAssertFalse(probe.preemptionInFlight.get(), "no stranded preemption flag")
    XCTAssertFalse(probe.preemptionSignal.isRaised, "no stale signal for the video to trip over later")
  }

  func testWatchdogWithdrawsARequestTheVideoNeverClaims() async throws {
    let probe = makeQueueProbe()
    let ctx = probe.slotRouteContext(videoRendering: { true }, watchdogSec: 0.2)

    _ = try XCTUnwrap(acquire(ctx, holder: "test:vision"))
    XCTAssertTrue(probe.inferenceHold.isRequested)
    try await waitUntil("the watchdog to fire", timeout: 5) { !probe.inferenceHold.isRequested }
    XCTAssertFalse(probe.preemptionInFlight.get())
    XCTAssertFalse(probe.preemptionSignal.isRaised)
    XCTAssertTrue(probe.inferenceSlots.isHeld(), "the slot itself stays; it now waits for the render to end")
  }

  func testNoVideoMeansNoHoldAndNoFlag() async throws {
    let probe = makeQueueProbe()
    let ctx = probe.slotRouteContext(videoRendering: { false }, watchdogSec: 60)
    let id = try XCTUnwrap(acquire(ctx, holder: "test:chat"))
    XCTAssertFalse(probe.inferenceHold.isRequested)
    XCTAssertFalse(probe.preemptionInFlight.get())
    XCTAssertFalse(probe.preemptionSignal.isRaised)
    XCTAssertTrue(release(ctx, id))
  }

  func testCancellingAParkedVideoUnparksAndLeaksNoSlot() async throws {
    let probe = makeQueueProbe()
    await probe.bypassVideoAdmission()
    let ctx = probe.slotRouteContext(videoRendering: { true }, watchdogSec: 60)

    let started = Flag()
    let proceed = DispatchSemaphore(value: 0)
    let job = Task { () -> Error? in
      do {
        _ = try await probe.enqueueLocalVideo { _ in
          started.set()
          proceed.wait()
          // The slot was acquired before `proceed`: the render observed the signal.
          return .yielded(Self.checkpoint())
        }
        return nil
      } catch { return error }
    }

    try await waitUntil("the video body to start") { started.value }
    let id = try XCTUnwrap(acquire(ctx, holder: "test:glimmer"))
    proceed.signal()
    try await waitUntil("the episode to park") { probe.inferenceHold.isParked }

    let interrupt = await probe.coordinatorInterrupt()
    XCTAssertTrue(interrupt.interrupted, "the parked video is the active render")
    let error = await job.value
    XCTAssertNotNil(error, "an interrupted parked video does not resume")
    XCTAssertFalse(probe.inferenceHold.isParked, "cancel while parked unparks")
    XCTAssertFalse(probe.preemptionInFlight.get())
    XCTAssertFalse(probe.preemptionSignal.isRaised)

    XCTAssertTrue(probe.inferenceSlots.isHeld(), "the client's slot is the client's to release")
    XCTAssertTrue(release(ctx, id))
    XCTAssertFalse(probe.inferenceSlots.isHeld(), "and releasing it leaves nothing behind")
  }
}
