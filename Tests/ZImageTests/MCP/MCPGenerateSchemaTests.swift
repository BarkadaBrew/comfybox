import XCTest

@testable import ZImage

/// WP-E4 (FDD-krea2-raw-recipe §3.4, AC-17) — the MCP `generate_image` schema
/// enumerates `SchedulerKind.allCases` / `SigmaScheduleKind.allCases` instead
/// of a free-text option list (which advertised `linear`, a schedule that does
/// not exist), so the schema cannot drift from the engine again.
final class MCPGenerateSchemaTests: XCTestCase {

  private func property(_ name: String) throws -> [String: Any] {
    let def = try XCTUnwrap(MCPToolRegistry.tool(named: "generate_image"))
    let props = try XCTUnwrap(def.inputSchema["properties"] as? [String: Any])
    return try XCTUnwrap(props[name] as? [String: Any], "generate_image has no '\(name)' property")
  }

  func testSchedulerEnumEqualsSchedulerKindAllCases() throws {
    let scheduler = try property("scheduler")
    let enumValues = try XCTUnwrap(scheduler["enum"] as? [String], "scheduler must carry an enum array")
    XCTAssertEqual(enumValues, SchedulerKind.allCases.map(\.rawValue))
    for v in enumValues { XCTAssertNotNil(try RecipeNameResolver.resolveSchedulerKind(v), v) }
  }

  func testSigmaScheduleEnumEqualsSigmaScheduleKindAllCases() throws {
    let schedule = try property("sigma_schedule")
    let enumValues = try XCTUnwrap(schedule["enum"] as? [String], "sigma_schedule must carry an enum array")
    XCTAssertEqual(enumValues, SigmaScheduleKind.allCases.map(\.rawValue))
    XCTAssertFalse(enumValues.contains("linear"))
    for v in enumValues { XCTAssertNotNil(try RecipeNameResolver.resolveSigmaScheduleKind(v), v) }
    let description = try XCTUnwrap(schedule["description"] as? String)
    XCTAssertFalse(description.contains("linear"), "description still names the phantom 'linear' schedule")
  }
}
