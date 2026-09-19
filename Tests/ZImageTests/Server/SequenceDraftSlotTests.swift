import Foundation
import XCTest

@testable import ZImage

/// Drafting a sequence is Glimmer work on the GPU this Mac also renders with,
/// so it holds a top-priority inference slot (Todd 2026-09-17: "any glimmer
/// action requires a scheduled GPU lease slot with top priority").
///
/// Without one the drafter competed with an in-flight render. Measured
/// 2026-09-19: Glimmer takes ~83 s on an IDLE GPU against a 120 s default
/// timeout — contention was not eating a margin, it was eating the whole
/// margin. And the CPU fallback is not an answer: dolphin 8B returns a valid
/// timeline in 6 s and the wrong film.
final class SequenceDraftSlotTests: XCTestCase {

  func testTheBudgetsLeaveRoomForAContendedGlimmer() {
    // 83 s idle is the measurement these are sized against.
    XCTAssertGreaterThan(
      WarmServer.sequenceDraftTimeout, 83 * 2,
      "a per-call timeout must survive a Glimmer slowed by contention")
    XCTAssertGreaterThan(
      WarmServer.sequenceDraftSlotTTL, WarmServer.sequenceDraftTimeout,
      "the lease must outlive a single call, or the slot lapses mid-draft")
    XCTAssertLessThanOrEqual(
      WarmServer.sequenceDraftSlotTTL, InferenceSlotTable.maxTTL,
      "the table clamps beyond this, so asking for more is a silent shortening")
  }

  func testHoldingASlotStOPSTheQueueStartingRenders() {
    // The whole mechanism in one assertion: while a slot is held the job loop
    // starts nothing, independent of the operator's manual pause.
    let table = InferenceSlotTable()
    XCTAssertFalse(table.isHeld())
    let slot = table.acquire(holder: "comfybox:sequence-draft", ttl: 60)
    XCTAssertTrue(table.isHeld(), "a draft in progress must hold the GPU")
    XCTAssertTrue(table.release(id: slot.id))
    XCTAssertFalse(table.isHeld(), "and give it back the moment it is done")
  }

  func testARenewedSlotSurvivesTheRepairRound() {
    // The draft makes TWO calls. Losing the lease between them would hand the
    // GPU back to a render in the middle of the job the slot protects.
    var clock = Date()
    let table = InferenceSlotTable(now: { clock })
    let slot = table.acquire(holder: "comfybox:sequence-draft", ttl: 300)
    clock = clock.addingTimeInterval(250)
    XCTAssertNotNil(table.renew(id: slot.id, ttl: 300), "renew before the repair")
    clock = clock.addingTimeInterval(250)
    XCTAssertTrue(table.isHeld(), "still held 500s in, because it was renewed")
  }

  func testALeakedSlotEXPIRESRatherThanPinningTheGPU() {
    // The defer releases on every path, but a crash between acquire and defer
    // must not stop rendering forever. TTL is the backstop.
    var clock = Date()
    let table = InferenceSlotTable(now: { clock })
    _ = table.acquire(holder: "comfybox:sequence-draft", ttl: 60)
    XCTAssertTrue(table.isHeld())
    clock = clock.addingTimeInterval(61)
    XCTAssertFalse(table.isHeld(), "a crashed drafter cannot pin the GPU")
  }
}
