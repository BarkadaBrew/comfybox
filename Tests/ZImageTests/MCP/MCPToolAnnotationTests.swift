import XCTest
@testable import ZImage

final class MCPToolAnnotationTests: XCTestCase {
  private let readOnlyTools: Set<String> = [
    "apply_style",
    "civitai_search",
    "get_config",
    "get_job",
    "list_characters",
    "list_loras",
    "list_models",
    "list_presets",
    "list_styles",
    "list_workflows",
    "lora_library",
    "model_pool",
    "nearline_list",
    "queue_list",
    "queue_status",
    "server_health",
    "system_stats",
    "video_status",
    "workflow_run_status",
  ]

  private let destructiveTools: Set<String> = [
    "cancel_job",
    "clear_queue",
    "delete_character",
    "delete_preset",
    "interrupt_render",
    "lora_quarantine",
    "nearline_evict",
    "shutdown_server",
  ]

  func testEveryToolDeclaresExplicitSafetyAnnotations() {
    for tool in MCPToolRegistry.tools {
      XCTAssertNotNil(tool.annotations, "\(tool.name) must declare MCP safety annotations")
    }
  }

  func testReadOnlyToolsAreAnnotatedReadOnlyAndNonDestructive() {
    for name in readOnlyTools {
      let annotations = MCPToolRegistry.tool(named: name)?.annotations
      XCTAssertEqual(annotations?.readOnlyHint, true, "\(name) should be read-only")
      XCTAssertEqual(annotations?.destructiveHint, false, "\(name) should be non-destructive")
    }
  }

  func testDestructiveToolsAreAnnotatedMutatingAndDestructive() {
    for name in destructiveTools {
      let annotations = MCPToolRegistry.tool(named: name)?.annotations
      XCTAssertEqual(annotations?.readOnlyHint, false, "\(name) mutates server state")
      XCTAssertEqual(annotations?.destructiveHint, true, "\(name) may destroy work or data")
    }
  }

  func testRemainingToolsAreAnnotatedAsAdditiveMutations() {
    for tool in MCPToolRegistry.tools
      where !readOnlyTools.contains(tool.name) && !destructiveTools.contains(tool.name)
    {
      XCTAssertEqual(tool.annotations?.readOnlyHint, false, "\(tool.name) mutates server state")
      XCTAssertEqual(tool.annotations?.destructiveHint, false, "\(tool.name) should be additive")
    }
  }

  func testToolsListJSONIncludesCamelCaseAnnotationHints() throws {
    let tool = try XCTUnwrap(MCPToolRegistry.tool(named: "shutdown_server"))
    let json = tool.responseJSON()
    let annotations = try XCTUnwrap(json["annotations"] as? [String: Any])

    XCTAssertEqual(annotations["readOnlyHint"] as? Bool, false)
    XCTAssertEqual(annotations["destructiveHint"] as? Bool, true)
  }
}
