import XCTest

@testable import ZImage

/// comfybox#153 (re-scoped) — the MCP bridge NEVER starts a server. launchd
/// (com.barkadabrew.comfybox) owns the engine lifecycle; spawning a second
/// copy is exactly the orphaned-process risk intent.md rules out. The
/// bridge's only startup decision is: connect to whatever answers on the
/// port (healthy or not), or fail loudly if nothing does.
final class MCPBridgeStartupPolicyTests: XCTestCase {

  func testOccupiedPortConnects() {
    let decision = MCPBridgeStartupPolicy.decide(port: 7870, portOccupied: true)
    XCTAssertEqual(decision.action, .connect)
    XCTAssertTrue(decision.detail.contains("7870"), "detail should name the port: \(decision.detail)")
  }

  func testFreePortFailsLoudlyNamingPortAndLaunchctlCommand() {
    let decision = MCPBridgeStartupPolicy.decide(port: 7870, portOccupied: false)
    XCTAssertEqual(decision.action, .failLoudly)
    XCTAssertTrue(decision.detail.contains("7870"), "detail should name the port: \(decision.detail)")
    XCTAssertTrue(
      decision.detail.contains("launchctl kickstart"),
      "detail should tell the operator how to start the real engine: \(decision.detail)")
    XCTAssertTrue(
      decision.detail.contains("com.barkadabrew.comfybox"),
      "detail should name the managed launchd service: \(decision.detail)")
  }

  func testLaunchctlKickstartCommandNamesTheManagedService() {
    let command = MCPBridgeStartupPolicy.launchctlKickstartCommand()
    XCTAssertTrue(command.hasPrefix("launchctl kickstart"))
    XCTAssertTrue(command.contains("com.barkadabrew.comfybox"))
  }

  /// Structural guard: the action enum has exactly `connect` and
  /// `failLoudly` — no `spawn` case. If a future change reintroduces one,
  /// this test fails and names every case actually present, rather than
  /// relying on someone noticing a new `case` in review.
  func testActionEnumHasNoSpawnCaseStructurally() {
    XCTAssertEqual(Set(MCPBridgeStartupAction.allCases), [.connect, .failLoudly])
  }
}
