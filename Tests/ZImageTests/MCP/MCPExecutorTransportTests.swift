import XCTest

@testable import ZImage

/// Executor-level coverage for the job-model composites (PR #367 review r1,
/// item 5). The earlier tests drove the `run*` helpers directly, which left
/// the WIRING untested: whether `execute(name:arguments:)` actually reads the
/// new parameters and dispatches to the right composite. These go through the
/// real dispatch switch with a stubbed transport — no server, no networking.
final class MCPExecutorTransportTests: XCTestCase {

  /// Records every request and answers from a closure.
  private final class StubTransport: WarmServerTransport, @unchecked Sendable {
    typealias Handler = @Sendable (_ method: String, _ path: String, _ body: Data) -> (Int, Data)

    private let handler: Handler
    private let lock = NSLock()
    private var recorded: [(method: String, path: String, body: Data)] = []

    init(handler: @escaping Handler) {
      self.handler = handler
    }

    var calls: [String] {
      lock.lock()
      defer { lock.unlock() }
      return recorded.map { "\($0.method) \($0.path)" }
    }

    func body(ofCallAt index: Int) -> [String: Any]? {
      lock.lock()
      defer { lock.unlock() }
      guard recorded.indices.contains(index) else { return nil }
      return try? JSONSerialization.jsonObject(with: recorded[index].body) as? [String: Any]
    }

    private func run(_ method: String, _ path: String, _ body: Data) -> (Int, Data) {
      lock.lock()
      recorded.append((method, path, body))
      lock.unlock()
      return handler(method, path, body)
    }

    func get(_ path: String) async throws -> (Int, Data) { run("GET", path, Data()) }
    func post(_ path: String, body: Data) async throws -> (Int, Data) { run("POST", path, body) }
    func put(_ path: String, body: Data) async throws -> (Int, Data) { run("PUT", path, body) }
    func patch(_ path: String, body: Data) async throws -> (Int, Data) { run("PATCH", path, body) }
    func delete(_ path: String) async throws -> (Int, Data) { run("DELETE", path, Data()) }
    func send(method: String, path: String, body: Data, headers: [String: String]) async throws
      -> (Int, Data, [String: String])
    {
      let (status, data) = run(method, path, body)
      return (status, data, [:])
    }
  }

  private func json(_ obj: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
  }

  private func tempPNG(bytes: Int) throws -> String {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("mcp-exec-\(UUID().uuidString).png")
    try Data(repeating: 0xCD, count: bytes).write(to: url)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url.path
  }

  // MARK: - #288: the async parameter is actually wired

  func testGenerateImageAsyncTrueSubmitsToTheAsyncRoute() async throws {
    let transport = StubTransport { _, _, _ in
      (202, self.json(["job_id": "J-42", "status": "queued", "source": "mcp", "elapsed_ms": 0]))
    }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "generate_image",
      arguments: MCPParams(["prompt": AnyCodable("a rose"), "async": AnyCodable(true)]))

    XCTAssertFalse(result.isError)
    XCTAssertEqual(transport.calls, ["POST /v1/generate/async"])
    XCTAssertEqual(transport.body(ofCallAt: 0)?["prompt"] as? String, "a rose")
    let obj = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(result.structuredJSON)) as? [String: Any])
    XCTAssertEqual(obj["job_id"] as? String, "J-42")
    XCTAssertEqual(obj["poll_with"] as? String, "get_job")
  }

  /// The default is the old behaviour, exactly: one POST to the synchronous
  /// route, and no progress polling (no token was supplied).
  func testGenerateImageDefaultsToTheSynchronousRouteWithNoExtraCalls() async {
    let transport = StubTransport { _, _, _ in
      (200, self.json(["success": true, "output_path": "/tmp/a.png", "duration_ms": 1200]))
    }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "generate_image", arguments: MCPParams(["prompt": AnyCodable("a rose")]))

    XCTAssertFalse(result.isError)
    XCTAssertEqual(transport.calls, ["POST /v1/generate"])
  }

  func testGenerateImageAsyncFalseIsAlsoSynchronous() async {
    let transport = StubTransport { _, _, _ in
      (200, self.json(["success": true, "output_path": "/tmp/a.png", "duration_ms": 1200]))
    }
    _ = await MCPToolExecutor(client: transport).execute(
      name: "generate_image",
      arguments: MCPParams(["prompt": AnyCodable("a rose"), "async": AnyCodable(false)]))
    XCTAssertEqual(transport.calls, ["POST /v1/generate"])
  }

  // MARK: - #294: the sync return_image path

  func testGenerateImageReturnImageAttachesTheRenderedPNG() async throws {
    let path = try tempPNG(bytes: 256)
    let transport = StubTransport { _, _, _ in
      (200, self.json(["success": true, "output_path": path, "duration_ms": 1200]))
    }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "generate_image",
      arguments: MCPParams([
        "prompt": AnyCodable("a rose"), "return_image": AnyCodable(true),
      ]))

    let images = result.content.filter { $0.type == "image" }
    XCTAssertEqual(images.count, 1)
    XCTAssertEqual(images.first?.mimeType, "image/png")
    XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(images.first?.data))?.count, 256)
    // The text block still carries the engine's own JSON, unchanged.
    XCTAssertTrue(try XCTUnwrap(result.content.first?.text).contains("output_path"))
  }

  func testGenerateImageWithoutReturnImageHasNoImageBlock() async throws {
    let path = try tempPNG(bytes: 256)
    let transport = StubTransport { _, _, _ in
      (200, self.json(["success": true, "output_path": path, "duration_ms": 1200]))
    }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "generate_image", arguments: MCPParams(["prompt": AnyCodable("a rose")]))
    XCTAssertEqual(result.content.count, 1)
    XCTAssertEqual(result.content.first?.type, "text")
  }

  /// A failed render must not try to attach anything, and must stay an error.
  func testFailedRenderWithReturnImageStaysAnError() async {
    let transport = StubTransport { _, _, _ in
      (400, self.json(["error": "width must be divisible by 16"]))
    }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "generate_image",
      arguments: MCPParams(["prompt": AnyCodable("x"), "return_image": AnyCodable(true)]))
    XCTAssertTrue(result.isError)
    XCTAssertEqual(result.content.filter { $0.type == "image" }.count, 0)
  }

  // MARK: - #289: get_job argument validation

  func testGetJobRejectsAnUnknownKindWithoutTouchingTheEngine() async throws {
    let transport = StubTransport { _, _, _ in (200, Data("{}".utf8)) }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "get_job",
      arguments: MCPParams(["job_id": AnyCodable("J-1"), "kind": AnyCodable("banana")]))

    XCTAssertTrue(result.isError)
    let text = try XCTUnwrap(result.content.first?.text)
    XCTAssertTrue(text.contains("banana"), text)
    XCTAssertTrue(text.contains("image"), "name the kinds that ARE valid: \(text)")
    XCTAssertEqual(transport.calls, [], "a bad argument must not reach the engine")
  }

  /// `swap` was removed from the advertised enum (review r1, item 3) — a
  /// caller who tries it gets told what to use instead, not a silent probe.
  func testGetJobRejectsTheRetiredSwapKind() async throws {
    let transport = StubTransport { _, _, _ in (200, Data("{}".utf8)) }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "get_job",
      arguments: MCPParams(["job_id": AnyCodable("S-1"), "kind": AnyCodable("swap")]))
    XCTAssertTrue(result.isError)
    XCTAssertEqual(transport.calls, [])
  }

  func testGetJobMissingJobIdIsACleanError() async {
    let transport = StubTransport { _, _, _ in (200, Data("{}".utf8)) }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "get_job", arguments: nil)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("job_id") == true)
    XCTAssertEqual(transport.calls, [])
  }

  /// A swap job id that only exists because queue recovery replayed it still
  /// resolves — through the image tracker, reported as `image`.
  func testSwapReplayIdStillResolvesThroughTheImageTracker() async throws {
    let transport = StubTransport { _, path, _ in
      if path.hasPrefix("/v1/generate/status/") {
        return (200, self.json(["job_id": "S-1", "status": "failed", "error": "replay failed"]))
      }
      return (404, self.json(["error": "not found"]))
    }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "get_job", arguments: MCPParams(["job_id": AnyCodable("S-1")]))

    XCTAssertFalse(result.isError)
    let obj = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(result.structuredJSON)) as? [String: Any])
    XCTAssertEqual(obj["kind"] as? String, "image")
    XCTAssertEqual(obj["state"] as? String, "failed")
    XCTAssertEqual(obj["error"] as? String, "replay failed")
  }
}
