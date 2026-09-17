import XCTest

@testable import ZImage

/// FDD-glimmer-gpu-slot §3.1 (3)/(4): the hold state machine and the grant rule.
final class InferenceHoldTests: XCTestCase {

  func testRequestTakeUnparkCycle() {
    let hold = InferenceHold()
    XCTAssertFalse(hold.isRequested)
    XCTAssertFalse(hold.isParked)
    XCTAssertTrue(hold.request())
    XCTAssertFalse(hold.request(), "a second request while one is pending is refused")
    XCTAssertTrue(hold.isRequested)
    XCTAssertTrue(hold.takeRequest())
    XCTAssertFalse(hold.isRequested)
    XCTAssertTrue(hold.isParked)
    XCTAssertFalse(hold.takeRequest(), "nothing left to take")
    XCTAssertFalse(hold.request(), "cannot request while parked")
    hold.unpark()
    XCTAssertFalse(hold.isParked)
    XCTAssertTrue(hold.request(), "idle again")
  }

  func testCancelOnlyAffectsAPendingRequest() {
    let hold = InferenceHold()
    XCTAssertFalse(hold.cancelIfRequested(), "idle: nothing to cancel")
    XCTAssertTrue(hold.request())
    XCTAssertTrue(hold.cancelIfRequested())
    XCTAssertFalse(hold.isRequested)
    XCTAssertTrue(hold.request())
    XCTAssertTrue(hold.takeRequest())
    XCTAssertFalse(hold.cancelIfRequested(), "a parked hold is not cancellable by the watchdog")
    XCTAssertTrue(hold.isParked)
  }

  func testGrantState() {
    XCTAssertEqual(InferenceSlotGrant.state(isParked: true, isRequested: false, gpuBusy: true), .granted)
    XCTAssertEqual(InferenceSlotGrant.state(isParked: false, isRequested: false, gpuBusy: false), .granted)
    XCTAssertEqual(InferenceSlotGrant.state(isParked: false, isRequested: true, gpuBusy: true), .preempting)
    XCTAssertEqual(InferenceSlotGrant.state(isParked: false, isRequested: false, gpuBusy: true), .waiting)
  }

  func testEta() {
    let now = Date(timeIntervalSince1970: 2_000)
    let started = now.addingTimeInterval(-60)
    XCTAssertEqual(InferenceSlotGrant.etaSec(progressPct: 50, renderStartedAt: started, now: now) ?? -1, 60, accuracy: 0.001)
    XCTAssertEqual(InferenceSlotGrant.etaSec(progressPct: 75, renderStartedAt: started, now: now) ?? -1, 20, accuracy: 0.001)
    XCTAssertNil(InferenceSlotGrant.etaSec(progressPct: 0, renderStartedAt: started, now: now))
    XCTAssertNil(InferenceSlotGrant.etaSec(progressPct: 100, renderStartedAt: started, now: now))
    XCTAssertNil(InferenceSlotGrant.etaSec(progressPct: nil, renderStartedAt: started, now: now))
    XCTAssertNil(InferenceSlotGrant.etaSec(progressPct: 40, renderStartedAt: nil, now: now))
  }
}
