import XCTest

@testable import ZImage

/// comfybox#153 review round 2, point 1 — the bridge no longer exits when
/// nothing is listening at startup; it warns once and serves anyway. Every
/// tool call made while the engine is still down must fail with the SAME
/// message the startup warning printed (port + the launchctl kickstart
/// command), not the generic "WarmServer not running" text, so an MCP
/// client sees one consistent, actionable error either way.
final class MCPToolExecutorStartupErrorTests: XCTestCase {

  /// A transport that always throws `.connectionRefused`, standing in for
  /// "the engine has not come up yet" without touching a real socket.
  private struct RefusingTransport: WarmServerTransport {
    let host: String
    let port: UInt16

    private var error: WarmServerClientError { .connectionRefused(host: host, port: port) }

    func get(_ path: String) async throws -> (Int, Data) { throw error }
    func post(_ path: String, body: Data) async throws -> (Int, Data) { throw error }
    func put(_ path: String, body: Data) async throws -> (Int, Data) { throw error }
    func patch(_ path: String, body: Data) async throws -> (Int, Data) { throw error }
    func delete(_ path: String) async throws -> (Int, Data) { throw error }
    func send(method: String, path: String, body: Data, headers: [String: String]) async throws
      -> (Int, Data, [String: String])
    { throw error }
  }

  func testConnectionRefusedMapsToTheSharedNothingListeningMessage() async {
    let transport = RefusingTransport(host: "127.0.0.1", port: 7870)
    let result = await MCPToolExecutor(client: transport).execute(name: "server_health", arguments: nil)

    XCTAssertTrue(result.isError)
    let expected = MCPBridgeStartupPolicy.nothingListeningMessage(host: "127.0.0.1", port: 7870)
    let text = result.toResponseDict()["content"] as? [[String: Any]]
    let message = (text?.first?["text"] as? String) ?? ""
    XCTAssertTrue(
      message.contains(expected),
      "expected the shared nothing-listening message in: \(message)")
  }

  /// A transport that always throws `.timedOut`, standing in for a
  /// mid-boot engine whose port already accepts connections but hasn't
  /// answered `/health` yet.
  private struct TimingOutTransport: WarmServerTransport {
    let host: String
    let port: UInt16

    private var error: WarmServerClientError { .timedOut(host: host, port: port) }

    func get(_ path: String) async throws -> (Int, Data) { throw error }
    func post(_ path: String, body: Data) async throws -> (Int, Data) { throw error }
    func put(_ path: String, body: Data) async throws -> (Int, Data) { throw error }
    func patch(_ path: String, body: Data) async throws -> (Int, Data) { throw error }
    func delete(_ path: String) async throws -> (Int, Data) { throw error }
    func send(method: String, path: String, body: Data, headers: [String: String]) async throws
      -> (Int, Data, [String: String])
    { throw error }
  }

  /// comfybox#389: PR #382's review-round-2 fix special-cased only
  /// `.connectionRefused` — a mid-boot engine (socket accepting, slow to
  /// answer) throws `.timedOut` instead and got the generic
  /// "WarmServer not running" text rather than this actionable hint.
  func testTimedOutMapsToTheSharedNothingListeningMessageToo() async {
    let transport = TimingOutTransport(host: "127.0.0.1", port: 7870)
    let result = await MCPToolExecutor(client: transport).execute(name: "server_health", arguments: nil)

    XCTAssertTrue(result.isError)
    let expected = MCPBridgeStartupPolicy.nothingListeningMessage(
      for: .timedOut(host: "127.0.0.1", port: 7870))
    let text = result.toResponseDict()["content"] as? [[String: Any]]
    let message = (text?.first?["text"] as? String) ?? ""
    XCTAssertTrue(
      message.contains(expected ?? "<nil>"),
      "expected the shared nothing-listening message in: \(message)")
    XCTAssertTrue(
      message.contains(MCPBridgeStartupPolicy.launchctlKickstartCommand()),
      "expected the launchd hint in: \(message)")
  }
}
