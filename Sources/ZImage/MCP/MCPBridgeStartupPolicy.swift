// MCPBridgeStartupPolicy.swift — pure connect-or-fail decision for comfybox#153
//
// The MCP bridge (`ComfyBox mcp`) NEVER starts a server. Lifecycle
// ownership belongs to launchd (com.barkadabrew.comfybox) alone — spawning
// a second copy is exactly the orphaned-process risk intent.md rules out
// ("no orphaned processes"; "MCP and serve modes must exit when their
// parent dies and must not leak model memory across requests").
//
// The bridge's only job at startup is: is anything answering on the port?
// If yes, connect to it — healthy or not, an unhealthy or old server is
// still not the bridge's to replace. If no, fail loudly and tell the
// operator how to start the real engine, instead of silently doing nothing
// or (the original comfybox#153 failure mode) spawning a second one.

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
  case connect
  case failLoudly
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
  /// The command that starts the managed engine — surfaced in the failure
  /// message so "nothing is listening, what do I do" has one obvious
  /// answer that is never "the bridge will start it for you".
  public static func launchctlKickstartCommand() -> String {
    "launchctl kickstart -k gui/\(getuid())/com.barkadabrew.comfybox"
  }

  /// Decide what the bridge should do. Pure function — the caller already
  /// resolved whether the port is occupied (via `MCPPortProbe`); this makes
  /// both branches reachable from a plain XCTest with no sockets touched.
  public static func decide(port: UInt16, portOccupied: Bool) -> MCPBridgeStartupDecision {
    if portOccupied {
      return MCPBridgeStartupDecision(
        action: .connect,
        detail: "Port \(port) has a server listening — connecting to it, not starting one.")
    }
    return MCPBridgeStartupDecision(
      action: .failLoudly,
      detail: "Nothing is listening on port \(port). This bridge never starts a server — " +
        "launchd (com.barkadabrew.comfybox) owns the engine lifecycle. Start it with: " +
        "\(launchctlKickstartCommand())")
  }
}
