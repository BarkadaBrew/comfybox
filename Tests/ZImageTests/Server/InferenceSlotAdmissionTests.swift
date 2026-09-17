import XCTest

@testable import ZImage

/// FDD-glimmer-gpu-slot §3.1 (1): while an inference slot is held the job loop
/// starts no render; releasing the last slot wakes the loop; the operator's
/// manual pause is independent and wins.
final class InferenceSlotAdmissionTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    try isolateComfyBoxStateDirectory()
  }

  func testHeldSlotParksARenderUntilReleased() async throws {
    let probe = makeQueueProbe()
    let slot = probe.inferenceSlots.acquire(holder: "test:chat", ttl: 60)
    let ran = LockedCounter()
    let job = Task { try await probe.enqueueFakeRender { ran.increment(); return true } }

    try await Task.sleep(nanoseconds: 300_000_000)
    XCTAssertEqual(ran.value, 0, "no render starts while a slot is held")
    XCTAssertEqual(probe.pendingCount, 1)

    XCTAssertTrue(probe.inferenceSlots.release(id: slot.id))
    _ = try await job.value
    XCTAssertEqual(ran.value, 1, "releasing the last slot wakes the loop")
  }

  func testSlotExpiryAlsoWakesTheLoop() async throws {
    let probe = makeQueueProbe()
    _ = probe.inferenceSlots.acquire(holder: "test:crashed-client", ttl: 1)
    let ran = LockedCounter()
    let job = Task { try await probe.enqueueFakeRender { ran.increment(); return true } }
    try await Task.sleep(nanoseconds: 300_000_000)
    XCTAssertEqual(ran.value, 0)
    _ = try await job.value
    XCTAssertEqual(ran.value, 1, "an expired lease cannot pin the GPU")
  }

  func testManualPauseStillParksAfterSlotRelease() async throws {
    let probe = makeQueueProbe()
    await probe.setPaused(true)
    let slot = probe.inferenceSlots.acquire(holder: "test:vision", ttl: 60)
    let ran = LockedCounter()
    let job = Task { try await probe.enqueueFakeRender { ran.increment(); return true } }

    try await Task.sleep(nanoseconds: 200_000_000)
    XCTAssertTrue(probe.inferenceSlots.release(id: slot.id))
    try await Task.sleep(nanoseconds: 300_000_000)
    XCTAssertEqual(ran.value, 0, "releasing slots never un-pauses a manual pause")
    XCTAssertEqual(probe.pendingCount, 1)

    await probe.setPaused(false)
    _ = try await job.value
    XCTAssertEqual(ran.value, 1)
  }
}
