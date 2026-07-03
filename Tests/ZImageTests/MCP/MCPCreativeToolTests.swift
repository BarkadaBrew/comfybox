import XCTest
@testable import ZImage

/// Tests for the creative-layer / queue / nearline MCP tools added 2026-07,
/// keeping the registry and the executor's dispatch switch in sync.
final class MCPCreativeToolTests: XCTestCase {

  /// Every tool added this session must be both defined and in the tools list.
  func testNewToolsRegistered() {
    let expected = [
      "enhance_prompt", "list_characters", "list_presets", "import_legacy_presets",
      "queue_list", "interrupt_render", "cancel_job",
      "nearline_list", "nearline_scan", "nearline_stage", "nearline_evict",
    ]
    let names = Set(MCPToolRegistry.tools.map(\.name))
    for tool in expected {
      XCTAssertTrue(names.contains(tool), "tools array should contain '\(tool)'")
      XCTAssertNotNil(MCPToolRegistry.tool(named: tool), "'\(tool)' should be defined")
    }
  }

  func testToolNamesAreUnique() {
    let names = MCPToolRegistry.tools.map(\.name)
    XCTAssertEqual(names.count, Set(names).count, "MCP tool names must be unique")
  }

  func testRequiredParamsDeclared() {
    // Tools that need an argument declare it required so clients validate.
    let requiredById: [String: String] = [
      "enhance_prompt": "prompt",
      "cancel_job": "id",
      "nearline_stage": "name",
      "nearline_evict": "name",
    ]
    for (tool, key) in requiredById {
      guard let def = MCPToolRegistry.tool(named: tool),
            let schema = def.inputSchema as? [String: Any],
            let required = schema["required"] as? [String] else {
        XCTFail("\(tool) missing input schema / required")
        continue
      }
      XCTAssertTrue(required.contains(key), "\(tool) should require '\(key)'")
    }
  }

  /// Unknown tools still route to a clean error (dispatch switch has a default).
  func testUnknownToolReturnsError() async {
    let executor = MCPToolExecutor(client: WarmServerClient(host: "127.0.0.1", port: 59999))
    let result = await executor.execute(name: "does_not_exist", arguments: nil)
    XCTAssertTrue(result.isError)
  }
}
