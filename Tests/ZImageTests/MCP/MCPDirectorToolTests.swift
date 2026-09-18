import XCTest

@testable import ZImage

/// Director MCP tools (docs/FDD-ltx-director-tab.md §4.3, WP4):
/// `generate_director_video` -> POST /v1/video/director and
/// `validate_director_timeline` -> POST /v1/video/director/validate.
/// Registry pins + executor wiring through a stub transport (no server).
final class MCPDirectorToolTests: XCTestCase {

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

  private static let repoRoot: URL = {
    // <root>/Tests/ZImageTests/MCP/MCPDirectorToolTests.swift
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }()

  /// A minimal MCP-shaped timeline: what an agent would send.
  private static let timelineJSON = """
    {
      "settings": {"width": 576, "height": 896, "length_frames": 289, "seed": 7},
      "global_prompt": "a woman walks along a pier at dusk",
      "keyframes": [
        {"id": "k1", "image_path": "/tmp/a.png", "frame": 0},
        {"id": "k2", "image_path": "/tmp/b.png", "frame": 0, "is_end_frame": true}
      ],
      "prompt_segments": [{"id": "p1", "start_frame": 0, "length_frames": 96, "prompt": "she turns"}]
    }
    """

  private func params(_ json: String) throws -> MCPParams {
    try JSONDecoder().decode(MCPParams.self, from: Data(json.utf8))
  }

  private func text(_ result: MCPToolResult) -> String {
    result.content.compactMap(\.text).joined()
  }

  // MARK: - Registry

  func testToolsRegisteredWithRoutesAndAnnotations() throws {
    let generate = try XCTUnwrap(MCPToolRegistry.tool(named: "generate_director_video"))
    XCTAssertEqual(generate.annotations, .additive)
    XCTAssertEqual(generate.routes, [RouteRef(method: "POST", path: "/v1/video/director")])
    XCTAssertEqual(generate.inputSchema["required"] as? [String], ["timeline"])
    let generateProps = try XCTUnwrap(generate.inputSchema["properties"] as? [String: Any])
    XCTAssertEqual(Set(generateProps.keys), ["timeline", "output_path", "source"])

    let validate = try XCTUnwrap(MCPToolRegistry.tool(named: "validate_director_timeline"))
    XCTAssertEqual(validate.annotations, .readOnly)
    XCTAssertEqual(validate.annotations?.readOnlyHint, true)
    XCTAssertEqual(validate.annotations?.destructiveHint, false)
    XCTAssertEqual(validate.routes, [RouteRef(method: "POST", path: "/v1/video/director/validate")])
    XCTAssertEqual(validate.inputSchema["required"] as? [String], ["timeline"])

    XCTAssertEqual(MCPToolRegistry.tools.count, 78)
    let names = MCPToolRegistry.tools.map(\.name)
    XCTAssertTrue(names.contains("generate_director_video"))
    XCTAssertTrue(names.contains("validate_director_timeline"))
  }

  /// The timeline property documents the schema an agent must author: the
  /// required top-level fields, and the hints that matter for correctness
  /// (end frame pinning, the 8-frame grid, clips only in imported mode).
  func testTimelineSchemaDocumentsTheShape() throws {
    for name in ["generate_director_video", "validate_director_timeline"] {
      let tool = try XCTUnwrap(MCPToolRegistry.tool(named: name))
      let props = try XCTUnwrap(tool.inputSchema["properties"] as? [String: Any])
      let timeline = try XCTUnwrap(props["timeline"] as? [String: Any], name)
      XCTAssertEqual(timeline["type"] as? String, "object", name)
      XCTAssertEqual(timeline["required"] as? [String], ["settings", "global_prompt"], name)
      let fields = try XCTUnwrap(timeline["properties"] as? [String: Any], name)
      for key in ["version", "settings", "global_prompt", "keyframes", "prompt_segments",
                  "audio_clips", "audio"] {
        XCTAssertNotNil(fields[key], "\(name) timeline should document \(key)")
      }
      let settings = try XCTUnwrap(fields["settings"] as? [String: Any])
      XCTAssertEqual(settings["required"] as? [String], ["width", "height", "length_frames"], name)

      let description = (timeline["description"] as? String ?? "") + tool.description
      for hint in ["is_end_frame", "prompt_segments", "audio_clips", "multiple of 8", "imported"] {
        XCTAssertTrue(description.contains(hint), "\(name) should mention \(hint)")
      }
    }
  }

  // MARK: - Executor wiring

  func testExecutorForwardsTimelineVerbatim() async throws {
    let transport = StubTransport { _, _, _ in
      (202, Data(#"{"job_id":"D-1","status":"queued","mode":"director","stage_count":1}"#.utf8))
    }
    let args = try params(
      #"{"timeline": \#(Self.timelineJSON), "output_path": "pier.mp4", "source": "kira"}"#)
    let result = await MCPToolExecutor(client: transport).execute(
      name: "generate_director_video", arguments: args)

    XCTAssertFalse(result.isError, text(result))
    XCTAssertEqual(transport.calls, ["POST /v1/video/director"])
    let body = try XCTUnwrap(transport.body(ofCallAt: 0))
    XCTAssertEqual(body["output_path"] as? String, "pier.mp4")
    XCTAssertEqual(body["source"] as? String, "kira")
    let sent = try XCTUnwrap(body["timeline"] as? [String: Any])
    let expected = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(Self.timelineJSON.utf8)) as? [String: Any])
    XCTAssertEqual(NSDictionary(dictionary: sent), NSDictionary(dictionary: expected))
    // The forwarded timeline still decodes as the engine's DirectorTimeline.
    let reencoded = try JSONSerialization.data(withJSONObject: sent)
    let decoded = try DirectorJSON.decoder().decode(DirectorTimeline.self, from: reencoded)
    XCTAssertEqual(decoded.settings.lengthFrames, 289)
    XCTAssertTrue(text(result).contains("D-1"))
  }

  func testExecutorRequiresTimeline() async throws {
    let transport = StubTransport { _, _, _ in (202, Data("{}".utf8)) }
    let executor = MCPToolExecutor(client: transport)
    for name in ["generate_director_video", "validate_director_timeline"] {
      let result = await executor.execute(
        name: name, arguments: try params(#"{"output_path": "x.mp4"}"#))
      XCTAssertTrue(result.isError, name)
      XCTAssertTrue(text(result).contains("timeline"), name)
      let none = await executor.execute(name: name, arguments: nil)
      XCTAssertTrue(none.isError, name)
    }
    XCTAssertEqual(transport.calls, [], "no request without a timeline")
  }

  func testValidateExecutorPostsValidatePath() async throws {
    let reply = #"{"issues":[],"ok":true,"plan":{"chunks":[]},"snapped_length_frames":289}"#
    let transport = StubTransport { _, _, _ in (200, Data(reply.utf8)) }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "validate_director_timeline",
      arguments: try params(#"{"timeline": \#(Self.timelineJSON)}"#))

    XCTAssertFalse(result.isError, text(result))
    XCTAssertEqual(transport.calls, ["POST /v1/video/director/validate"])
    let body = try XCTUnwrap(transport.body(ofCallAt: 0))
    XCTAssertNotNil(body["timeline"] as? [String: Any])
    let out = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(text(result).utf8)) as? [String: Any])
    XCTAssertEqual(out["ok"] as? Bool, true)
    XCTAssertEqual(out["snapped_length_frames"] as? Int, 289)
    XCTAssertNotNil(result.structuredJSON)
  }

  func testValidateExecutorSurfacesServerError() async throws {
    let transport = StubTransport { _, _, _ in
      (400, Data(#"{"error":"timeline invalid","issues":[]}"#.utf8))
    }
    let result = await MCPToolExecutor(client: transport).execute(
      name: "validate_director_timeline",
      arguments: try params(#"{"timeline": {"version": 2}}"#))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(text(result).contains("timeline invalid"))
  }

  func testExecutorSourceContainsLiteralPaths() throws {
    let url = Self.repoRoot.appendingPathComponent("Sources/ZImage/MCP/MCPToolExecutor.swift")
    let source = try String(contentsOf: url, encoding: .utf8)
    XCTAssertTrue(source.contains(#"case "generate_director_video":"#))
    XCTAssertTrue(source.contains(#"case "validate_director_timeline":"#))
    XCTAssertTrue(source.contains(#"postWithQueueRecoveryRetry("/v1/video/director", body:"#))
    XCTAssertTrue(source.contains(#"client.post("/v1/video/director/validate", body:"#))
  }
}
