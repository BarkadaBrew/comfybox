import XCTest

@testable import ZImage

/// Codex review of #465 (finding 1): `repair_image`'s vision diagnosis on a
/// Glimmer endpoint holds a top-priority inference slot, bounded by the tool's
/// deadline, and releases it before the repair render is queued.
final class MCPRepairImageSlotTests: XCTestCase {

  private final class Stub: WarmServerTransport, @unchecked Sendable {
    typealias Handler = @Sendable (_ method: String, _ path: String) -> (Int, Data)
    private let lock = NSLock()
    private var recorded: [(String, Data)] = []
    private let handler: Handler
    init(_ handler: @escaping Handler) { self.handler = handler }
    var calls: [String] { lock.lock(); defer { lock.unlock() }; return recorded.map(\.0) }
    func body(of call: String) -> [String: Any]? {
      lock.lock(); defer { lock.unlock() }
      guard let data = recorded.first(where: { $0.0 == call })?.1 else { return nil }
      return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
    private func run(_ m: String, _ p: String, _ b: Data) -> (Int, Data) {
      lock.lock(); recorded.append(("\(m) \(p)", b)); lock.unlock()
      return handler(m, p)
    }
    func get(_ path: String) async throws -> (Int, Data) { run("GET", path, Data()) }
    func post(_ path: String, body: Data) async throws -> (Int, Data) { run("POST", path, body) }
    func put(_ path: String, body: Data) async throws -> (Int, Data) { run("PUT", path, body) }
    func patch(_ path: String, body: Data) async throws -> (Int, Data) { run("PATCH", path, body) }
    func delete(_ path: String) async throws -> (Int, Data) { run("DELETE", path, Data()) }
    func send(method: String, path: String, body: Data, headers: [String: String]) async throws -> (Int, Data, [String: String]) {
      let (s, d) = run(method, path, body); return (s, d, [:])
    }
  }

  /// Port 9 (discard) refuses fast — the vision call fails without touching a real model.
  private static let config = #"{"providers":{"vision":{"baseUrl":"http://127.0.0.1:9/v1","model":"glimmer"}}}"#

  private func imageFile() throws -> String {
    let path = NSTemporaryDirectory() + "repair-slot-\(UUID().uuidString).png"
    try Data([0x89, 0x50, 0x4E, 0x47]).write(to: URL(fileURLWithPath: path))
    addTeardownBlock { try? FileManager.default.removeItem(atPath: path) }
    return path
  }

  private func args(_ path: String) throws -> MCPParams {
    try JSONDecoder().decode(MCPParams.self, from: Data(#"{"image_path": "\#(path)"}"#.utf8))
  }

  func testDiagnosisHoldsASlotAndReleasesItBeforeTheRepairRender() async throws {
    let stub = Stub { method, path in
      switch (method, path) {
      case ("GET", "/v1/config"): return (200, Data(Self.config.utf8))
      case ("POST", "/v1/queue/inference-slot"): return (200, Data(#"{"slot_id":"s1","state":"granted"}"#.utf8))
      case ("POST", "/v1/generate"): return (500, Data(#"{"error":"stub"}"#.utf8))
      default: return (200, Data("{}".utf8))
      }
    }
    let executor = MCPToolExecutor(client: stub, glimmerEndpoints: ["127.0.0.1:9"])
    _ = await executor.execute(name: "repair_image", arguments: try args(imageFile()))

    let calls = stub.calls
    let acquire = try XCTUnwrap(calls.firstIndex(of: "POST /v1/queue/inference-slot"), "\(calls)")
    let release = try XCTUnwrap(calls.firstIndex(of: "DELETE /v1/queue/inference-slot/s1"), "\(calls)")
    let render = try XCTUnwrap(calls.firstIndex(of: "POST /v1/generate"), "\(calls)")
    XCTAssertLessThan(acquire, release)
    XCTAssertLessThan(release, render, "the slot is free before the repair render queues behind it")
    XCTAssertEqual(stub.body(of: "POST /v1/queue/inference-slot")?["holder"] as? String, "comfybox:repair_image")
  }

  func testBusyGpuSkipsTheDiagnosisButStillRepairs() async throws {
    let stub = Stub { method, path in
      switch (method, path) {
      case ("GET", "/v1/config"): return (200, Data(Self.config.utf8))
      case ("POST", "/v1/queue/inference-slot"):
        return (200, Data(#"{"slot_id":"s1","state":"waiting","eta_sec":3600}"#.utf8))
      case ("POST", "/v1/generate"): return (500, Data(#"{"error":"stub"}"#.utf8))
      default: return (200, Data("{}".utf8))
      }
    }
    let executor = MCPToolExecutor(client: stub, glimmerEndpoints: ["127.0.0.1:9"])
    _ = await executor.execute(name: "repair_image", arguments: try args(imageFile()))

    let calls = stub.calls
    XCTAssertTrue(calls.contains("DELETE /v1/queue/inference-slot/s1"), "\(calls)")
    XCTAssertFalse(calls.contains { $0.hasPrefix("GET /v1/queue/inference-slot") }, "an hour-long eta fails at once")
    XCTAssertTrue(calls.contains("POST /v1/generate"), "the repair still runs with the baseline defect negatives")
  }

  func testNonGlimmerVisionTakesNoSlot() async throws {
    let stub = Stub { method, path in
      switch (method, path) {
      case ("GET", "/v1/config"): return (200, Data(Self.config.utf8))
      default: return (500, Data("{}".utf8))
      }
    }
    let executor = MCPToolExecutor(client: stub, glimmerEndpoints: ["10.0.100.134:11234"])
    _ = await executor.execute(name: "repair_image", arguments: try args(imageFile()))
    XCTAssertFalse(stub.calls.contains { $0.contains("inference-slot") }, "\(stub.calls)")
  }
}
