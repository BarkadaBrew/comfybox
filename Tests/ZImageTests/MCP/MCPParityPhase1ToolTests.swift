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
  /// patch_config, update_config. Bumped 55 -> 56 by the job-model cluster
  /// (comfybox#289): get_job.
  func testTotalToolCountIncludesGapSet() {
    XCTAssertEqual(MCPToolRegistry.tools.count, 56, "Expected 56 registered MCP tools after Phase 1 + Phase 3 + get_job")
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
    // Phase 4: the dispatch arm serves POST and PUT identically (upsert), so
    // the tool claims both — same posture as create_character.
    XCTAssertEqual(tool.routes, [
      RouteRef(method: "POST", path: "/v1/presets"),
      RouteRef(method: "PUT", path: "/v1/presets"),
    ])
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

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (200, Data("{}".utf8), [:])
      case ("GET", "/v1/config"): return (200, configData, [:])
      case ("PUT", "/v1/config"): return (200, Data("{}".utf8), [:])
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data(), [:])
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

    _ = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, body, _ in
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (200, Data("{}".utf8), [:])
      case ("GET", "/v1/config"): return (200, configData, [:])
      case ("PUT", "/v1/config"):
        putBody = body
        return (200, Data("{}".utf8), [:])
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data(), [:])
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

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (500, Data("{\"error\":\"not in pool\"}".utf8), [:])
      case ("POST", "/v1/model/load"): return (202, Data("{}".utf8), [:])
      case ("GET", "/v1/config"): return (200, Data("{}".utf8), [:])
      case ("PUT", "/v1/config"): return (200, Data("{}".utf8), [:])
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data(), [:])
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

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (500, Data("{\"error\":\"nope\"}".utf8), [:])
      case ("POST", "/v1/model/load"): return (500, Data("{\"error\":\"still nope\"}".utf8), [:])
      default:
        XCTFail("config must never be touched when activation fails entirely: \(method) \(path)")
        return (500, Data(), [:])
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

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _, _ in
      await log.record(method, path)
      switch (method, path) {
      case ("POST", "/v1/model/activate"): return (200, Data("{}".utf8), [:])
      case ("GET", "/v1/config"): return (500, Data("{\"error\":\"disk error\"}".utf8), [:])
      default:
        XCTFail("PUT must never be attempted when GET /v1/config fails: \(method) \(path)")
        return (500, Data(), [:])
      }
    }

    XCTAssertTrue(result.isError)
    let calls = await log.methodsAndPaths().map { "\($0.0) \($0.1)" }
    XCTAssertEqual(calls, ["POST /v1/model/activate", "GET /v1/config"])
  }

  /// The lost-update fix (adversarial review F2, 2026-08-30): a concurrent
  /// `PATCH /v1/config` landing between set_warm_preset's GET and PUT must
  /// not be clobbered. The composite sends the GET's ETag as If-Match, so
  /// the stale PUT is rejected with 409; the ONE bounded retry re-fetches
  /// the patched document and re-applies only the modelSpec mutation — the
  /// final PUT body carries BOTH the concurrent patch's field AND the new
  /// modelSpec.
  func testSetWarmPresetRetriesOn409PreservingConcurrentPatch() async throws {
    /// A tiny fake config server whose document is PATCHed by "someone else"
    /// immediately after the first GET is served — the exact interleaving
    /// that used to lose the patch.
    actor FakeConfigServer {
      var doc: [String: Any] = ["modelSpec": "old-model", "otherField": "keep-me"]
      var etag = "\"v1\""
      private(set) var getCount = 0
      private(set) var putAttempts: [(ifMatch: String?, body: Data)] = []

      func get() throws -> (Data, String) {
        getCount += 1
        let data = try JSONSerialization.data(withJSONObject: doc)
        let servedETag = etag
        if getCount == 1 {
          // The concurrent PATCH: lands right after our GET response is on
          // the wire, before our PUT arrives.
          doc["patchedField"] = "concurrent"
          etag = "\"v2\""
        }
        return (data, servedETag)
      }

      func put(ifMatch: String?, body: Data) throws -> Int {
        putAttempts.append((ifMatch, body))
        if let ifMatch, ifMatch != etag { return 409 }
        doc = try JSONSerialization.jsonObject(with: body) as! [String: Any]
        etag = "\"v3\""
        return 200
      }
    }

    let server = FakeConfigServer()
    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, body, headers in
      switch (method, path) {
      case ("POST", "/v1/model/activate"):
        return (200, Data("{}".utf8), [:])
      case ("GET", "/v1/config"):
        let (data, etag) = try await server.get()
        return (200, data, ["ETag": etag])
      case ("PUT", "/v1/config"):
        let status = try await server.put(ifMatch: headers["If-Match"], body: body)
        return (status, Data("{}".utf8), [:])
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data(), [:])
      }
    }

    XCTAssertFalse(result.isError, "the bounded retry should succeed")

    let attempts = await server.putAttempts
    XCTAssertEqual(attempts.count, 2, "exactly one retry after the 409")
    XCTAssertEqual(attempts[0].ifMatch, "\"v1\"", "first PUT must send the first GET's ETag as If-Match")
    XCTAssertEqual(attempts[1].ifMatch, "\"v2\"", "the retry must send the RE-FETCHED document's ETag")

    // The winning PUT body carries BOTH writes: the concurrent patch's field
    // and our modelSpec mutation (plus the untouched pre-existing field).
    let finalBody = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: attempts[1].body) as? [String: Any])
    XCTAssertEqual(finalBody["modelSpec"] as? String, "new-model")
    XCTAssertEqual(finalBody["patchedField"] as? String, "concurrent",
                    "the concurrent PATCH must survive the retry")
    XCTAssertEqual(finalBody["otherField"] as? String, "keep-me")

    // And the server's final document agrees.
    let finalDoc = await server.doc
    XCTAssertEqual(finalDoc["modelSpec"] as? String, "new-model")
    XCTAssertEqual(finalDoc["patchedField"] as? String, "concurrent")
  }

  /// The bounded half of the F2 fix: if a SECOND 409 follows the retry (a
  /// pathological write storm), the composite returns the 409 as a clean
  /// error instead of looping.
  func testSetWarmPresetSecondConflictReturns409() async throws {
    actor Counter {
      private(set) var puts = 0
      func bump() -> Int { puts += 1; return puts }
    }
    let counter = Counter()
    let configData = try JSONSerialization.data(withJSONObject: ["modelSpec": "old"])

    let result = try await MCPToolExecutor.runSetWarmPreset(model: "new-model") { method, path, _, headers in
      switch (method, path) {
      case ("POST", "/v1/model/activate"):
        return (200, Data("{}".utf8), [:])
      case ("GET", "/v1/config"):
        return (200, configData, ["ETag": "\"stale\""])
      case ("PUT", "/v1/config"):
        _ = await counter.bump()
        XCTAssertEqual(headers["If-Match"], "\"stale\"")
        return (409, Data("{\"error\":\"conflict\"}".utf8), [:])
      default:
        XCTFail("unexpected call \(method) \(path)")
        return (500, Data(), [:])
      }
    }

    XCTAssertTrue(result.isError, "a second conflict must surface as an error")
    let puts = await counter.puts
    XCTAssertEqual(puts, 2, "exactly two PUT attempts — never an unbounded loop")
  }
}
