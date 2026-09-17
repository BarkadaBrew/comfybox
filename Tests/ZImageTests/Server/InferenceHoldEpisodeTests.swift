import MLX
import XCTest

@testable import ZImage

/// FDD-glimmer-gpu-slot §3.1 (3): an in-flight LTX-2 video that yields while an
/// inference hold is requested parks — weights resident, no image preemptor —
/// until every slot is released, then goes on to resume.
final class InferenceHoldEpisodeTests: XCTestCase {

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

  private static func checkpoint() -> LTX2ResumeState {
    LTX2ResumeState(
      videoLatents: MLXArray.zeros([1]), stepIndex: 1, sigmas: [1.0, 0.5, 0.0],
      phase: .baseDenoise, chunkIndex: 0, seed: 1,
      audioLatents: nil, audioNoiseKey: nil,
      configFingerprint: "test", refineCleanLatents: nil)
  }

  func testRequestedHoldParksTheVideoUntilSlotsRelease() async throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()
    await probe.bypassVideoAdmission()

    let started = Flag()
    let proceed = DispatchSemaphore(value: 0)
    let finished = Flag()
    let job = Task { () -> Error? in
      defer { finished.set() }
      do {
        _ = try await probe.enqueueLocalVideo { _ in
          started.set()
          proceed.wait()
          // The render observed the #1479 signal at a step boundary.
          return .yielded(Self.checkpoint())
        }
        return nil
      } catch { return error }
    }

    try await waitUntil("the video body to start") { started.value }
    // What POST /v1/queue/inference-slot does while a video renders.
    let slot = probe.inferenceSlots.acquire(holder: "test:glimmer", ttl: 60)
    XCTAssertTrue(probe.inferenceHold.request())
    proceed.signal()

    try await waitUntil("the episode to park for the slot") { probe.inferenceHold.isParked }
    try await Task.sleep(nanoseconds: 300_000_000)
    XCTAssertFalse(finished.value, "the video stays parked while a slot is held")
    XCTAssertTrue(probe.inferenceHold.isParked)

    XCTAssertTrue(probe.inferenceSlots.release(id: slot.id))
    _ = await job.value
    XCTAssertFalse(probe.inferenceHold.isParked, "releasing the last slot unparks the video")
    XCTAssertTrue(finished.value, "the episode went on to resume (in a unit test the cold reload has no weights)")
  }
}
