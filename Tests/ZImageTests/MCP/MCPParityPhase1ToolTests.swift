import XCTest
@testable import ZImage

/// Tests for the headless-parity Phase 1 gap-set tools (comfybox#300, FDD
/// §4.2): move_queue_job, update_lora_triggerwords, create_preset,
/// delete_preset, set_warm_preset, create_character, delete_character.
///
/// Scope note: this is the 6-item gap set only (§4.2's broader "New tools"
/// list — cancel_queue_job, import_loras, civitai_harvest,
/// promote_video_trace, rate_video_trace, delete_workflow,
/// generate_image_async, generate_video_async — and update_config, which
/// FDD §0 row 10 explicitly holds to Phase 3, are other work's territory).
final class MCPParityPhase1ToolTests: XCTestCase {

  private static let newToolNames = [
    "move_queue_job", "update_lora_triggerwords", "create_preset", "delete_preset",
    "set_warm_preset", "create_character", "delete_character",
  ]

  // MARK: - Registration

  func testNewToolsRegistered() {
    let names = Set(MCPToolRegistry.tools.map(\.name))
    for tool in Self.newToolNames {
      XCTAssertTrue(names.contains(tool), "tools array should contain '\(tool)'")
      XCTAssertNotNil(MCPToolRegistry.tool(named: tool), "'\(tool)' should be defined")
    }
  }

  /// Pinned so a silent registry change is caught immediately (same
  /// convention as MCPVideoToolTests.testTotalToolCount). Bumped 52 -> 55 by
  /// Phase 3 (comfybox#300, FDD §3.3/§4.4, 2026-08-30): get_config,
  /// patch_config, update_config.
  func testTotalToolCountIncludesGapSet() {
    XCTAssertEqual(MCPToolRegistry.tools.count, 55, "Expected 55 registered MCP tools after Phase 1 + Phase 3")
  }

  /// Every tool added in this session declares its route(s) (FDD §3.5 D5:
  /// `routes: [RouteRef]` populated for new tools).
  func testNewToolsDeclareRoutes() {
    for name in Self.newToolNames {
      guard let tool = MCPToolRegistry.tool(named: name) else {
        XCTFail("\(name) missing from registry"); continue
      }
      XCTAssertFalse(tool.routes.isEmpty, "\(name) should declare at least one RouteRef")
    }
  }

  func testMoveQueueJobRoute() {
    let tool = MCPToolRegistry.tool(named: "move_queue_job")!
    XCTAssertEqual(tool.routes, [RouteRef(method: "POST", path: "/v1/queue/{id}/move")])
  }

  func testUpdateLoraTriggerwordsRoute() {
    let tool = MCPToolRegistry.tool(named: "update_lora_triggerwords")!
    XCTAssertEqual(tool.routes, [RouteRef(method: "POST", path: "/v1/loras/{id}/update")])
  }

  func testCreatePresetRoute() {
    let tool = MCPToolRegistry.tool(named: "create_preset")!
    XCTAssertEqual(tool.routes, [RouteRef(method: "POST", path: "/v1/presets")])
  }

  func testDeletePresetRoute() {
    let tool = MCPToolRegistry.tool(named: "delete_preset")!
    XCTAssertEqual(tool.routes, [RouteRef(method: "DELETE", path: "/v1/presets/{id}")])
  }

  func testSetWarmPresetRoutes() {
    let tool = MCPToolRegistry.tool(named: "set_warm_preset")!
    XCTAssertEqual(tool.routes, [
      RouteRef(method: "POST", path: "/v1/model/activate"),
      RouteRef(method: "POST", path: "/v1/model/load"),
      RouteRef(method: "GET", path: "/v1/config"),
      RouteRef(method: "PUT", path: "/v1/config"),
    ])
  }

  func testCreateCharacterRoutes() {
    let tool = MCPToolRegistry.tool(named: "create_character")!
    XCTAssertEqual(tool.routes, [
      RouteRef(method: "POST", path: "/v1/characters"),
      RouteRef(method: "PUT", path: "/v1/characters"),
    ])
  }

  func testDeleteCharacterRoute() {
    let tool = MCPToolRegistry.tool(named: "delete_character")!
    XCTAssertEqual(tool.routes, [RouteRef(method: "DELETE", path: "/v1/characters/{id}")])
  }

  // MARK: - Schema: required fields

  func testRequiredParamsDeclared() {
    let requiredById: [String: [String]] = [
      "move_queue_job": ["id", "direction"],
      "update_lora_triggerwords": ["id", "triggerwords"],
      "create_preset": ["id", "name"],
      "delete_preset": ["id"],
      "set_warm_preset": ["model"],
      "create_character": ["name"],
      "delete_character": ["id"],
    ]
    for (tool, keys) in requiredById {
      guard let def = MCPToolRegistry.tool(named: tool),
            let schema = def.inputSchema as? [String: Any],
            let required = schema["required"] as? [String] else {
        XCTFail("\(tool) missing input schema / required")
        continue
      }
      for key in keys {
        XCTAssertTrue(required.contains(key), "\(tool) should require '\(key)'")
      }
    }
  }

  // MARK: - Schema: enum constraints

  func testMoveQueueJobDirectionEnum() {
    let tool = MCPToolRegistry.tool(named: "move_queue_job")!
    let properties = tool.inputSchema["properties"] as? [String: Any]
    let direction = properties?["direction"] as? [String: Any]
    XCTAssertEqual(direction?["enum"] as? [String], ["top", "up", "down"])
  }

  func testCreatePresetMediaKindEnum() {
    let tool = MCPToolRegistry.tool(named: "create_preset")!
    let properties = tool.inputSchema["properties"] as? [String: Any]
    let mediaKind = properties?["media_kind"] as? [String: Any]
    XCTAssertEqual(mediaKind?["enum"] as? [String], ["image", "video"])
  }

  func testCreateCharacterKindEnum() {
    let tool = MCPToolRegistry.tool(named: "create_character")!
    let properties = tool.inputSchema["properties"] as? [String: Any]
    let kind = properties?["kind"] as? [String: Any]
    XCTAssertEqual(kind?["enum"] as? [String], ["character", "scene"])
  }

  // MARK: - Executor: missing/invalid required params -> clean error

  private func unreachableExecutor() -> MCPToolExecutor {
    MCPToolExecutor(client: WarmServerClient(host: "127.0.0.1", port: 19999))
  }

  func testMoveQueueJobMissingIdReturnsError() async {
    let result = await unreachableExecutor().execute(
      name: "move_queue_job", arguments: MCPParams(["direction": AnyCodable("up")]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("id") == true)
  }

  func testMoveQueueJobMissingDirectionReturnsError() async {
    let result = await unreachableExecutor().execute(
      name: "move_queue_job", arguments: MCPParams(["id": AnyCodable("job-1")]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("direction") == true)
  }

  /// The clean-400 requirement (FDD scope note): an unrecognized direction
  /// must be rejected here, not forwarded to WarmServer, which currently
  /// no-ops (200, moved:false) rather than 400ing on a bad direction
  /// (WarmServer.swift movePending, default case).
  func testMoveQueueJobUnknownDirectionReturnsCleanError() async {
    let result = await unreachableExecutor().execute(
      name: "move_queue_job",
      arguments: MCPParams(["id": AnyCodable("job-1"), "direction": AnyCodable("sideways")]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("direction") == true)
    XCTAssertTrue(result.content.first?.text?.contains("sideways") == true)
  }

  func testUpdateLoraTriggerwordsMissingIdReturnsError() async {
    let result = await unreachableExecutor().execute(
      name: "update_lora_triggerwords",
      arguments: MCPParams(["triggerwords": AnyCodable([AnyCodable("foo")])]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("id") == true)
  }

  func testUpdateLoraTriggerwordsMissingTriggerwordsReturnsError() async {
    let result = await unreachableExecutor().execute(
      name: "update_lora_triggerwords", arguments: MCPParams(["id": AnyCodable("lora-1")]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("triggerwords") == true)
  }

  func testCreatePresetMissingIdReturnsError() async {
    let result = await unreachableExecutor().execute(
      name: "create_preset", arguments: MCPParams(["name": AnyCodable("My Preset")]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("id") == true)
  }

  func testCreatePresetMissingNameReturnsError() async {
    let result = await unreachableExecutor().execute(
      name: "create_preset", arguments: MCPParams(["id": AnyCodable("preset-1")]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("name") == true)
  }

  func testDeletePresetMissingIdReturnsError() async {
    let result = await unreachableExecutor().execute(name: "delete_preset", arguments: nil)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("id") == true)
  }

  func testSetWarmPresetMissingModelReturnsError() async {
    let result = await unreachableExecutor().execute(name: "set_warm_preset", arguments: nil)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("model") == true)
  }

  func testCreateCharacterMissingNameReturnsError() async {
    let result = await unreachableExecutor().execute(
      name: "create_character", arguments: MCPParams(["id": AnyCodable("char-1")]))
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("name") == true)
  }

  func testDeleteCharacterMissingIdReturnsError() async {
    let result = await unreachableExecutor().execute(name: "delete_character", arguments: nil)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("id") == true)
  }

  // MARK: - set_warm_preset composite: order + abort-on-failure (no networking)

  /// Records calls made through the injectable closure so tests can assert
  /// exact order without a real HTTP server.
  private actor CallLog {
    private(set) var calls: [(method: String, path: String)] = []
    func record(_ method: String, _ path: String) { calls.append((method, path)) }
    func methodsAndPaths() -> [(String, String)] { calls.map { ($0.method, $0.path) } }
  }

  func testSetWarmPresetActivateSucceeds_thenGetThenPutConfigInOrder() async throws {
    let log = CallLog()
    let configDoc: [String: Any] = ["modelSpec": "old-model", "otherField": "keep-me"]
    let configData = try JSONSerialization.data(withJSONObject: configDoc)

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (200, Data("{}".utf8))
      case ("GET", "/v1/config"): return (200, configData)
      case ("PUT", "/v1/config"): return (200, Data("{}".utf8))
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data())
      }
    }

    XCTAssertFalse(result.isError)
    let calls = await log.methodsAndPaths()
    XCTAssertEqual(calls.map(\.0), ["POST", "GET", "PUT"])
    XCTAssertEqual(calls.map(\.1), ["/v1/model/activate", "/v1/config", "/v1/config"])
  }

  /// The PUT body must carry the mutated modelSpec while preserving
  /// unrelated fields fetched from GET — a fetch-then-mutate-then-save round
  /// trip, not a bare `{modelSpec: ...}` document (which would delete every
  /// other config field per WarmServer's enumerated-keys-only encode).
  func testSetWarmPresetPutBodyPreservesUnrelatedFieldsAndSetsModelSpec() async throws {
    let configDoc: [String: Any] = ["modelSpec": "old-model", "otherField": "keep-me"]
    let configData = try JSONSerialization.data(withJSONObject: configDoc)
    var putBody: Data?

    _ = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, body in
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (200, Data("{}".utf8))
      case ("GET", "/v1/config"): return (200, configData)
      case ("PUT", "/v1/config"):
        putBody = body
        return (200, Data("{}".utf8))
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data())
      }
    }

    let putObj = try XCTUnwrap(putBody.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] })
    XCTAssertEqual(putObj["modelSpec"] as? String, "new-model")
    XCTAssertEqual(putObj["otherField"] as? String, "keep-me")
  }

  /// Mirrors PresetView.setAsWarm's do/catch: activation failure falls back
  /// to a load, and the composite proceeds to the config write on that
  /// fallback's success.
  func testSetWarmPresetActivateFails_loadFallbackSucceeds_thenConfigWritten() async throws {
    let log = CallLog()

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (500, Data("{\"error\":\"not in pool\"}".utf8))
      case ("POST", "/v1/model/load"): return (202, Data("{}".utf8))
      case ("GET", "/v1/config"): return (200, Data("{}".utf8))
      case ("PUT", "/v1/config"): return (200, Data("{}".utf8))
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data())
      }
    }

    XCTAssertFalse(result.isError)
    let calls = await log.methodsAndPaths().map { "\($0.0) \($0.1)" }
    XCTAssertEqual(calls, [
      "POST /v1/model/activate", "POST /v1/model/load", "GET /v1/config", "PUT /v1/config",
    ])
  }

  /// If BOTH activation and the load fallback fail, the config must never be
  /// touched — a partial apply here would report success while the server's
  /// warm-start default silently didn't change.
  func testSetWarmPresetActivateAndLoadBothFail_configNeverTouched() async throws {
    let log = CallLog()

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (500, Data("{\"error\":\"nope\"}".utf8))
      case ("POST", "/v1/model/load"): return (500, Data("{\"error\":\"still nope\"}".utf8))
      default:
        XCTFail("config must never be touched when activation fails entirely: \(method) \(path)")
        return (500, Data())
      }
    }

    XCTAssertTrue(result.isError, "composite should report failure, not partially apply")
    let calls = await log.methodsAndPaths().map { "\($0.0) \($0.1)" }
    XCTAssertEqual(calls, ["POST /v1/model/activate", "POST /v1/model/load"])
  }

  /// If GET /v1/config fails after a successful activation, PUT must not be
  /// attempted either — same "no partial apply" contract, one step later.
  func testSetWarmPresetGetConfigFails_putNeverAttempted() async throws {
    let log = CallLog()

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (200, Data("{}".utf8))
      case ("GET", "/v1/config"): return (500, Data("{\"error\":\"disk error\"}".utf8))
      default:
        XCTFail("PUT must never be attempted when GET /v1/config fails: \(method) \(path)")
        return (500, Data())
      }
    }

    XCTAssertTrue(result.isError)
    let calls = await log.methodsAndPaths().map { "\($0.0) \($0.1)" }
    XCTAssertEqual(calls, ["POST /v1/model/activate", "GET /v1/config"])
  }
}
