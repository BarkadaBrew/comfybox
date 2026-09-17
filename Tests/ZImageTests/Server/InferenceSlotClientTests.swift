import XCTest

@testable import ZImage

/// Codex review of #465 (finding 1): the client engine-local Glimmer callers use
/// to hold a top-priority inference slot — deadline-bounded, cancellable,
/// renewed from the moment a slot id exists, released on every path, and
/// reporting each wait so a UI can show it.
final class InferenceSlotClientTests: XCTestCase {

  /// Records every request; answers from a scripted handler.
  final class Stub: WarmServerTransport, @unchecked Sendable {
    typealias Handler = @Sendable (_ method: String, _ path: String) throws -> (Int, Data)
    private let lock = NSLock()
    private var log: [String] = []
    private let handler: Handler
    init(_ handler: @escaping Handler) { self.handler = handler }
    var calls: [String] { lock.lock(); defer { lock.unlock() }; return log }
    func count(_ prefix: String) -> Int { calls.filter { $0.hasPrefix(prefix) }.count }
    private func run(_ m: String, _ p: String) throws -> (Int, Data) {
      lock.lock(); log.append("\(m) \(p)"); lock.unlock()
      return try handler(m, p)
    }
    func get(_ path: String) async throws -> (Int, Data) { try run("GET", path) }
    func post(_ path: String, body: Data) async throws -> (Int, Data) { try run("POST", path) }
    func put(_ path: String, body: Data) async throws -> (Int, Data) { try run("PUT", path) }
    func patch(_ path: String, body: Data) async throws -> (Int, Data) { try run("PATCH", path) }
    func delete(_ path: String) async throws -> (Int, Data) { try run("DELETE", path) }
    func send(method: String, path: String, body: Data, headers: [String: String]) async throws -> (Int, Data, [String: String]) {
      let (s, d) = try run(method, path); return (s, d, [:])
    }
  }

  /// Fake clock that advances only when the client sleeps.
  final class Clock: @unchecked Sendable {
    private let lock = NSLock()
    private var t = Date(timeIntervalSince1970: 10_000)
    func now() -> Date { lock.lock(); defer { lock.unlock() }; return t }
    func advance(_ s: TimeInterval) { lock.lock(); t = t.addingTimeInterval(s); lock.unlock() }
  }

  final class Box<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []
    func append(_ v: T) { lock.lock(); items.append(v); lock.unlock() }
    var all: [T] { lock.lock(); defer { lock.unlock() }; return items }
  }

  static func json(_ obj: [String: Any]) -> Data { (try? JSONSerialization.data(withJSONObject: obj)) ?? Data() }
  static let glimmer = "http://127.0.0.1:11234/v1"

  private func client(_ stub: Stub, _ clock: Clock, ttl: TimeInterval = 180, maxWait: TimeInterval = 900) -> InferenceSlotClient {
    InferenceSlotClient(
      transport: stub, endpoints: InferenceSlotClient.defaultGlimmerEndpoints,
      ttlSec: ttl, pollSec: 1, maxWaitSec: maxWait,
      now: { clock.now() },
      sleep: { s in clock.advance(s); await Task.yield() })
  }

  /// POST → `acquire`; GET → the next poll state; renew/DELETE → 200.
  private func scripted(acquire: [String: Any], polls: [[String: Any]]) -> Stub {
    let queue = Box<[String: Any]>()
    polls.forEach { queue.append($0) }
    let index = Box<Int>()
    return Stub { method, path in
      switch (method, path) {
      case ("POST", "/v1/queue/inference-slot"):
        return (200, Self.json(acquire))
      case ("GET", _):
        index.append(0)
        let i = min(index.all.count, queue.all.count) - 1
        return (200, Self.json(queue.all[max(0, i)]))
      default:
        return (200, Self.json(["released": true]))
      }
    }
  }

  // MARK: - Endpoint detection

  func testGlimmerEndpointDetection() {
    let list = InferenceSlotClient.defaultGlimmerEndpoints
    XCTAssertTrue(InferenceSlotClient.isGlimmerEndpoint("http://127.0.0.1:11234/v1", endpoints: list))
    XCTAssertTrue(InferenceSlotClient.isGlimmerEndpoint("http://localhost:11234", endpoints: list))
    XCTAssertTrue(InferenceSlotClient.isGlimmerEndpoint("http://10.0.100.134:11234/v1/", endpoints: list))
    XCTAssertFalse(InferenceSlotClient.isGlimmerEndpoint("http://127.0.0.1:1234/v1", endpoints: list), "LM Studio is not Glimmer")
    XCTAssertFalse(InferenceSlotClient.isGlimmerEndpoint("https://openrouter.ai/api/v1", endpoints: list))
    XCTAssertEqual(
      InferenceSlotClient.glimmerEndpoints(environment: ["COMFYBOX_GLIMMER_ENDPOINTS": "a:1, b:2"]), ["a:1", "b:2"])
    XCTAssertEqual(InferenceSlotClient.glimmerEndpoints(environment: [:]), list)
  }

  func testNonGlimmerEndpointRunsWithoutTouchingTheEngine() async throws {
    let stub = Stub { _, _ in XCTFail("no engine call expected"); return (500, Data()) }
    let out = try await client(stub, Clock()).withSlot(
      holder: "t", baseURL: "http://127.0.0.1:1234/v1", deadline: nil, reserveSec: 0, onWait: nil) { 7 }
    XCTAssertEqual(out, 7)
    XCTAssertTrue(stub.calls.isEmpty)
  }

  // MARK: - Grant, release

  func testGrantedImmediatelyRunsAndReleases() async throws {
    let stub = scripted(acquire: ["slot_id": "s1", "state": "granted"], polls: [])
    let out = try await client(stub, Clock()).withSlot(
      holder: "comfybox:test", baseURL: Self.glimmer, deadline: nil, reserveSec: 0, onWait: nil) { "ok" }
    XCTAssertEqual(out, "ok")
    XCTAssertEqual(stub.calls.first, "POST /v1/queue/inference-slot")
    XCTAssertEqual(stub.calls.last, "DELETE /v1/queue/inference-slot/s1")
  }

  func testBodyErrorStillReleases() async {
    struct Boom: Error {}
    let stub = scripted(acquire: ["slot_id": "s1", "state": "granted"], polls: [])
    do {
      _ = try await client(stub, Clock()).withSlot(
        holder: "t", baseURL: Self.glimmer, deadline: nil, reserveSec: 0, onWait: nil) { () -> Int in throw Boom() }
      XCTFail("expected the body's error")
    } catch {
      XCTAssertTrue(error is Boom)
    }
    XCTAssertEqual(stub.count("DELETE"), 1)
  }

  // MARK: - Waiting: show it, renew through it

  func testWaitsReportsEachPollAndRenewsBeforeGrant() async throws {
    let stub = scripted(
      acquire: ["slot_id": "s1", "state": "waiting", "eta_sec": 12],
      polls: [
        ["state": "waiting", "eta_sec": 11], ["state": "preempting"], ["state": "waiting", "eta_sec": 9],
        ["state": "waiting", "eta_sec": 8], ["state": "granted"],
      ])
    let waits = Box<InferenceSlotClient.Wait>()
    let out = try await client(stub, Clock(), ttl: 4).withSlot(
      holder: "t", baseURL: Self.glimmer, deadline: nil, reserveSec: 0, onWait: { waits.append($0) }) { 1 }
    XCTAssertEqual(out, 1)
    XCTAssertEqual(waits.all.first?.state, "waiting")
    XCTAssertEqual(waits.all.first?.etaSec, 12)
    XCTAssertTrue(waits.all.contains { $0.state == "preempting" })
    XCTAssertGreaterThanOrEqual(stub.count("POST /v1/queue/inference-slot/s1/renew"), 1,
      "a waiting slot is renewed (ttl 4 s, 5 s of polling) so it cannot expire before the grant")
    XCTAssertEqual(stub.count("DELETE"), 1)
  }

  func testStatusText() {
    XCTAssertEqual(InferenceSlotClient.Wait(state: "waiting", etaSec: 41.6, waitedSec: 3).statusText, "Waiting for the GPU (~42s)")
    XCTAssertEqual(InferenceSlotClient.Wait(state: "waiting", etaSec: nil, waitedSec: 3).statusText, "Waiting for the GPU…")
    XCTAssertEqual(InferenceSlotClient.Wait(state: "preempting", etaSec: nil, waitedSec: 1).statusText, "Waiting for the GPU (pausing the video render…)")
  }

  // MARK: - Deadline

  func testEtaBeyondTheBudgetFailsEarlyAndReleases() async {
    let clock = Clock()
    let stub = scripted(acquire: ["slot_id": "s1", "state": "waiting", "eta_sec": 300], polls: [])
    let ran = Box<Int>()
    do {
      _ = try await client(stub, clock).withSlot(
        holder: "t", baseURL: Self.glimmer, deadline: clock.now().addingTimeInterval(180), reserveSec: 120,
        onWait: nil) { ran.append(1) }
      XCTFail("expected gpuBusy")
    } catch let error as InferenceSlotWaitError {
      XCTAssertEqual(error, .gpuBusy(etaSec: 300))
      XCTAssertEqual(error.localizedDescription, "The GPU is busy rendering (about 300 s left).")
    } catch {
      XCTFail("unexpected \(error)")
    }
    XCTAssertTrue(ran.all.isEmpty, "the generation never started")
    XCTAssertEqual(stub.count("GET"), 0, "no pointless polling once the eta says no")
    XCTAssertEqual(stub.count("DELETE"), 1)
  }

  func testDeadlinePassesWhileWaitingThrowsBusyAndReleases() async {
    let clock = Clock()
    let stub = scripted(acquire: ["slot_id": "s1", "state": "waiting"], polls: [["state": "waiting"]])
    do {
      _ = try await client(stub, clock).withSlot(
        holder: "t", baseURL: Self.glimmer, deadline: clock.now().addingTimeInterval(5), reserveSec: 2,
        onWait: nil) { 1 }
      XCTFail("expected gpuBusy")
    } catch let error as InferenceSlotWaitError {
      XCTAssertEqual(error, .gpuBusy(etaSec: nil))
      XCTAssertEqual(error.localizedDescription, "The GPU is busy rendering.")
    } catch {
      XCTFail("unexpected \(error)")
    }
    XCTAssertLessThanOrEqual(stub.count("GET"), 4, "3 s of wait budget at 1 s polls")
    XCTAssertEqual(stub.count("DELETE"), 1)
  }

  // MARK: - Cancellation

  func testCancellationWhileWaitingReleasesAndThrows() async {
    let stub = Stub { method, path in
      if method == "POST" && path == "/v1/queue/inference-slot" { return (200, Self.json(["slot_id": "s1", "state": "waiting"])) }
      if method == "GET" { return (200, Self.json(["state": "waiting"])) }
      return (200, Self.json([:]))
    }
    let slotClient = InferenceSlotClient(
      transport: stub, endpoints: InferenceSlotClient.defaultGlimmerEndpoints,
      ttlSec: 180, pollSec: 0.02, maxWaitSec: 900)
    let task = Task {
      try await slotClient.withSlot(holder: "t", baseURL: Self.glimmer, deadline: nil, reserveSec: 0, onWait: nil) { 1 }
    }
    try? await Task.sleep(nanoseconds: 100_000_000)
    task.cancel()
    do {
      _ = try await task.value
      XCTFail("expected cancellation")
    } catch let error as InferenceSlotWaitError {
      XCTAssertEqual(error, .cancelled)
    } catch {
      XCTFail("unexpected \(error)")
    }
    XCTAssertEqual(stub.count("DELETE /v1/queue/inference-slot/s1"), 1, "a cancelled wait still releases its slot")
  }

  // MARK: - Engines without slots, unreachable engines, lost slots

  func testEngineWithoutSlotRoutesRunsUnslotted() async throws {
    for status in [404, 405] {
      let stub = Stub { _, _ in (status, Data()) }
      let out = try await client(stub, Clock()).withSlot(
        holder: "t", baseURL: Self.glimmer, deadline: nil, reserveSec: 0, onWait: nil) { status }
      XCTAssertEqual(out, status)
      XCTAssertEqual(stub.calls, ["POST /v1/queue/inference-slot"])
    }
  }

  func testUnreachableEngineFailsOpen() async throws {
    struct Refused: Error {}
    let stub = Stub { _, _ in throw Refused() }
    let out = try await client(stub, Clock()).withSlot(
      holder: "t", baseURL: Self.glimmer, deadline: nil, reserveSec: 0, onWait: nil) { "ran" }
    XCTAssertEqual(out, "ran")
  }

  func testLostSlotIsReacquired() async throws {
    let acquires = Box<Int>()
    let stub = Stub { method, path in
      if method == "POST" && path == "/v1/queue/inference-slot" {
        acquires.append(1)
        return (200, Self.json(["slot_id": "s\(acquires.all.count)", "state": "waiting"]))
      }
      if method == "GET" && path.hasSuffix("/s1") { return (404, Data()) }
      if method == "GET" { return (200, Self.json(["state": "granted"])) }
      return (200, Self.json([:]))
    }
    let out = try await client(stub, Clock()).withSlot(
      holder: "t", baseURL: Self.glimmer, deadline: nil, reserveSec: 0, onWait: nil) { 5 }
    XCTAssertEqual(out, 5)
    XCTAssertEqual(acquires.all.count, 2, "an expired or forgotten slot is taken again, not skipped")
    XCTAssertEqual(stub.calls.last, "DELETE /v1/queue/inference-slot/s2")
  }
}
