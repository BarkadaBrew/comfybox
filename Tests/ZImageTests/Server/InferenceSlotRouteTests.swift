import XCTest

@testable import ZImage

/// FDD-glimmer-gpu-slot §3.1: the `/v1/queue/inference-slot` protocol, driven
/// through the pure route handler with injected GPU state.
final class InferenceSlotRouteTests: XCTestCase {

  final class World: @unchecked Sendable {
    let table = InferenceSlotTable()
    let hold = InferenceHold()
    var gpuBusy = false
    var videoRendering = false
    var progressPct: Int?
    var renderStartedAt: Date?
    var raises = 0
    let now = Date(timeIntervalSince1970: 5_000)

    lazy var ctx = InferenceSlotRouteContext(
      table: table, hold: hold,
      gpuBusy: { [unowned self] in self.gpuBusy },
      videoRendering: { [unowned self] in self.videoRendering },
      progress: { [unowned self] in (self.progressPct, self.renderStartedAt) },
      raiseHold: { [unowned self] in
        // The server's closure: request the hold (the real one also raises the
        // #1479 signal and arms a watchdog).
        if self.hold.request() { self.raises += 1 }
      },
      now: { [unowned self] in self.now })
  }

  private func call(_ w: World, _ method: String, _ path: String, _ body: [String: Any]? = nil) -> (Int, [String: Any]) {
    let data = body.flatMap { try? JSONSerialization.data(withJSONObject: $0) } ?? Data()
    let r = InferenceSlotRoute.handle(method: method, path: path, body: data, ctx: w.ctx)
    return (r.status, r.payload)
  }

  func testMatches() {
    XCTAssertTrue(InferenceSlotRoute.matches(method: "POST", path: "/v1/queue/inference-slot"))
    XCTAssertTrue(InferenceSlotRoute.matches(method: "GET", path: "/v1/queue/inference-slot"))
    XCTAssertTrue(InferenceSlotRoute.matches(method: "GET", path: "/v1/queue/inference-slot/abc"))
    XCTAssertTrue(InferenceSlotRoute.matches(method: "DELETE", path: "/v1/queue/inference-slot/abc"))
    XCTAssertTrue(InferenceSlotRoute.matches(method: "POST", path: "/v1/queue/inference-slot/abc/renew"))
    XCTAssertFalse(InferenceSlotRoute.matches(method: "DELETE", path: "/v1/queue/abc"), "the generic cancel route is untouched")
    XCTAssertFalse(InferenceSlotRoute.matches(method: "POST", path: "/v1/queue/abc/move"))
  }

  func testAcquireRequiresHolder() {
    let w = World()
    let (status, payload) = call(w, "POST", "/v1/queue/inference-slot", [:])
    XCTAssertEqual(status, 400)
    XCTAssertNotNil(payload["error"])
    XCTAssertFalse(w.table.isHeld())
  }

  func testAcquireGrantedWhenIdle() {
    let w = World()
    let (status, payload) = call(w, "POST", "/v1/queue/inference-slot", ["holder": "kira:chat", "ttl_sec": 60])
    XCTAssertEqual(status, 200)
    XCTAssertEqual(payload["state"] as? String, "granted")
    XCTAssertEqual(payload["holder"] as? String, "kira:chat")
    XCTAssertEqual(payload["active_slots"] as? Int, 1)
    XCTAssertNotNil(payload["slot_id"] as? String)
    XCTAssertNotNil(payload["expires_at"] as? String)
    XCTAssertEqual(w.raises, 0, "nothing to preempt")
  }

  func testAcquireWaitsBehindAnImageRenderWithEta() {
    let w = World()
    w.gpuBusy = true
    w.progressPct = 50
    w.renderStartedAt = w.now.addingTimeInterval(-40)
    let (status, payload) = call(w, "POST", "/v1/queue/inference-slot", ["holder": "bree:vision"])
    XCTAssertEqual(status, 200)
    XCTAssertEqual(payload["state"] as? String, "waiting")
    XCTAssertEqual(payload["eta_sec"] as? Double ?? -1, 40, accuracy: 0.001)
    XCTAssertEqual(w.raises, 0, "an image render cannot be preempted")
  }

  func testAcquirePreemptsAnInFlightVideo() {
    let w = World()
    w.gpuBusy = true
    w.videoRendering = true
    let (_, first) = call(w, "POST", "/v1/queue/inference-slot", ["holder": "kira:t2v"])
    XCTAssertEqual(first["state"] as? String, "preempting")
    XCTAssertEqual(w.raises, 1)
    let (_, second) = call(w, "POST", "/v1/queue/inference-slot", ["holder": "kira:vision"])
    XCTAssertEqual(second["state"] as? String, "preempting")
    XCTAssertEqual(w.raises, 1, "one hold request covers every slot")

    // The video checkpoints and parks → both slots read granted.
    XCTAssertTrue(w.hold.takeRequest())
    let id = first["slot_id"] as! String
    let (status, polled) = call(w, "GET", "/v1/queue/inference-slot/\(id)")
    XCTAssertEqual(status, 200)
    XCTAssertEqual(polled["state"] as? String, "granted")
  }

  func testRenewReleaseAndUnknownIds() {
    let w = World()
    let (_, acq) = call(w, "POST", "/v1/queue/inference-slot", ["holder": "chat", "ttl_sec": 30])
    let id = acq["slot_id"] as! String
    let (renewStatus, renewed) = call(w, "POST", "/v1/queue/inference-slot/\(id)/renew", ["ttl_sec": 120])
    XCTAssertEqual(renewStatus, 200)
    XCTAssertEqual(renewed["slot_id"] as? String, id)

    XCTAssertEqual(call(w, "GET", "/v1/queue/inference-slot/nope").0, 404)
    XCTAssertEqual(call(w, "POST", "/v1/queue/inference-slot/nope/renew", [:]).0, 404)

    let (listStatus, list) = call(w, "GET", "/v1/queue/inference-slot")
    XCTAssertEqual(listStatus, 200)
    XCTAssertEqual(list["active_slots"] as? Int, 1)
    XCTAssertEqual((list["slots"] as? [[String: Any]])?.count, 1)

    let (delStatus, deleted) = call(w, "DELETE", "/v1/queue/inference-slot/\(id)")
    XCTAssertEqual(delStatus, 200)
    XCTAssertEqual(deleted["released"] as? Bool, true)
    XCTAssertEqual(deleted["active_slots"] as? Int, 0)
    let (_, again) = call(w, "DELETE", "/v1/queue/inference-slot/\(id)")
    XCTAssertEqual(again["released"] as? Bool, false, "release is idempotent")
  }
}
