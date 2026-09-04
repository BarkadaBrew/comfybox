import XCTest

@testable import ZImage

/// Registry surface for the job-model cluster (comfybox#288, #289, #292,
/// #294): the additive `generate_image` parameters and the additive `get_job`
/// tool, including the #297 safety annotations and the §3.5 route claims.
final class MCPJobToolSchemaTests: XCTestCase {

  private func properties(of tool: String) throws -> [String: Any] {
    let def = try XCTUnwrap(MCPToolRegistry.tool(named: tool))
    return try XCTUnwrap(def.inputSchema["properties"] as? [String: Any])
  }

  // MARK: - #288: additive async on generate_image

  func testGenerateImageExposesAsyncBooleanDefaultingToSynchronous() throws {
    let async = try XCTUnwrap(
      try properties(of: "generate_image")["async"] as? [String: Any],
      "generate_image must expose 'async'")
    XCTAssertEqual(async["type"] as? String, "boolean")
    XCTAssertEqual(async["default"] as? Bool, false, "the default must stay synchronous")
    let description = try XCTUnwrap(async["description"] as? String)
    XCTAssertTrue(description.contains("get_job"), "point callers at the polling tool")
  }

  func testGenerateImageStillOnlyRequiresPrompt() throws {
    let def = try XCTUnwrap(MCPToolRegistry.tool(named: "generate_image"))
    XCTAssertEqual(def.inputSchema["required"] as? [String], ["prompt"])
  }

  /// #288 claims the async route, so its ParityExemption (which said a
  /// dedicated async tool was "deferred") must be gone — a route cannot be
  /// both claimed and excused.
  func testGenerateImageClaimsBothTheSyncAndAsyncRoutes() throws {
    let def = try XCTUnwrap(MCPToolRegistry.tool(named: "generate_image"))
    XCTAssertTrue(def.routes.contains(RouteRef(method: "POST", path: "/v1/generate")))
    XCTAssertTrue(def.routes.contains(RouteRef(method: "POST", path: "/v1/generate/async")))
    XCTAssertNil(
      ParityExemptions.reason(for: RouteRef(method: "POST", path: "/v1/generate/async")),
      "POST /v1/generate/async is claimed by generate_image now — drop the exemption")
  }

  // MARK: - #289: one get_job tool

  func testGetJobIsRegisteredAndReadOnly() throws {
    let def = try XCTUnwrap(MCPToolRegistry.tool(named: "get_job"), "get_job must be registered")
    XCTAssertEqual(def.annotations?.readOnlyHint, true)
    XCTAssertEqual(def.annotations?.destructiveHint, false)
  }

  func testGetJobRequiresJobIdAndOffersEveryKind() throws {
    let def = try XCTUnwrap(MCPToolRegistry.tool(named: "get_job"))
    XCTAssertEqual(def.inputSchema["required"] as? [String], ["job_id"])
    let kind = try XCTUnwrap(try properties(of: "get_job")["kind"] as? [String: Any])
    XCTAssertEqual(kind["enum"] as? [String], MCPJobKind.allCases.map(\.rawValue))
    XCTAssertEqual(kind["enum"] as? [String], ["image", "video", "swap", "storyboard"])
  }

  func testGetJobClaimsTheThreeRoutesItReads() throws {
    let def = try XCTUnwrap(MCPToolRegistry.tool(named: "get_job"))
    XCTAssertEqual(
      Set(def.routes),
      Set([
        RouteRef(method: "GET", path: "/v1/generate/status/{id}"),
        RouteRef(method: "GET", path: "/v1/video/status/{id}"),
        RouteRef(method: "GET", path: "/v1/queue"),
      ]))
  }

  /// The existing per-kind polling tools are NOT removed (#289 proposed
  /// removing two; intent.md forbids silently changing the tool surface —
  /// version or shim). They keep their names, schemas and shapes.
  func testPerKindPollingToolsSurvive() {
    XCTAssertNotNil(MCPToolRegistry.tool(named: "video_status"))
    XCTAssertNotNil(MCPToolRegistry.tool(named: "workflow_run_status"))
  }

  // MARK: - #294: return_image

  func testReturnImageIsAdditiveAndDefaultsFalse() throws {
    for tool in ["generate_image", "get_job"] {
      let flag = try XCTUnwrap(
        try properties(of: tool)[MCPImageAttachment.parameterName] as? [String: Any],
        "\(tool) must expose return_image")
      XCTAssertEqual(flag["type"] as? String, "boolean")
      XCTAssertEqual(flag["default"] as? Bool, false)
      let description = try XCTUnwrap(flag["description"] as? String)
      XCTAssertTrue(
        description.lowercased().contains("base64"), "\(tool): say what the payload costs")
    }
    XCTAssertEqual(MCPImageAttachment.parameterName, "return_image")
  }

  // MARK: - Catalog pin

  func testToolCountGrewByExactlyOne() {
    XCTAssertEqual(
      MCPToolRegistry.tools.count, 56,
      "55 tools + get_job (#289). Update this pin and the two other pins in the same review.")
  }
}
