// MCPBridgeStartupPolicy.swift — pure connect-or-warn decision for comfybox#153
//
// The MCP bridge (`ComfyBox mcp`) NEVER starts a server. Lifecycle
// ownership belongs to launchd (com.barkadabrew.comfybox) alone — spawning
// a second copy is exactly the orphaned-process risk intent.md rules out
// ("no orphaned processes"; "MCP and serve modes must exit when their
// parent dies and must not leak model memory across requests").
//
// The bridge's only job at startup is: is anything answering on the port?
// If yes, connect to it — healthy or not, an unhealthy or old server is
// still not the bridge's to replace. If no, the bridge does NOT exit
// (review round 2, point 1): launchd's RunAtLoad and the MCP host commonly
// race at login, so a bridge that starts before the engine must keep
// serving — print the warning once and let every tool call fail with the
// same message (see MCPToolExecutor) until the engine comes up. Exiting
// would make the MCP host mark the bridge dead until someone manually
// restarts it, which is worse than the original comfybox#153 bug.

import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// The only two things the bridge can decide to do at startup. There is no
/// `spawn` case, on purpose — see `MCPBridgeStartupPolicyTests
/// .testActionEnumHasNoSpawnCaseStructurally`, which pins this enum to
/// exactly these two cases so a future change can't quietly reintroduce
/// one without a failing test naming the new case.
public enum MCPBridgeStartupAction: String, CaseIterable, Equatable, Sendable {
  /// A server answered the port — connect to it, healthy or not.
  case connect
  /// Nothing answered the port. The bridge keeps running and serves
  /// anyway: it logs the warning once (port + the launchctl command to
  /// start the real engine) and starts `MCPServer` as normal. It does NOT
  /// spawn anything and does NOT exit — every tool call will fail with
  /// this same "nothing is listening" message (MCPToolExecutor maps
  /// `WarmServerClientError.connectionRefused` to it) until the engine
  /// answers.
  case warnAndServe
}

/// One startup decision: what to do, and the human-readable detail to log.
public struct MCPBridgeStartupDecision: Equatable, Sendable {
  public let action: MCPBridgeStartupAction
  public let detail: String

  public init(action: MCPBridgeStartupAction, detail: String) {
    self.action = action
    self.detail = detail
  }
}

public enum MCPBridgeStartupPolicy {
  /// The command that starts the managed engine — surfaced whenever
  /// nothing answers the port, so "nothing is listening, what do I do" has
  /// one obvious answer that is never "the bridge will start it for you".
  public static func launchctlKickstartCommand() -> String {
    "launchctl kickstart -k gui/\(getuid())/com.barkadabrew.comfybox"
  }

  /// The message shown when nothing answers `host:port` — at startup
  /// (this file) and on every subsequent tool call that hits
  /// `WarmServerClientError.connectionRefused` (`MCPToolExecutor`). Kept as
  /// one function so the two call sites can never drift apart.
  public static func nothingListeningMessage(host: String, port: UInt16) -> String {
    "Nothing is listening on \(host):\(port). This bridge never starts a server — " +
      "launchd (com.barkadabrew.comfybox) owns the engine lifecycle. Start it with: " +
      "\(launchctlKickstartCommand())"
  }

  /// Decide what the bridge should do. Pure function — the caller already
  /// resolved whether the port is occupied (via `MCPPortProbe`); this makes
  /// both branches reachable from a plain XCTest with no sockets touched.
  public static func decide(host: String, port: UInt16, portOccupied: Bool) -> MCPBridgeStartupDecision {
    if portOccupied {
      return MCPBridgeStartupDecision(
        action: .connect,
        detail: "\(host):\(port) has a server listening — connecting to it, not starting one.")
    }
    return MCPBridgeStartupDecision(
      action: .warnAndServe,
      detail: nothingListeningMessage(host: host, port: port))
  }
}
