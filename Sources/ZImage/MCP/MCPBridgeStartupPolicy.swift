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
  /// this same "nothing is listening" message (MCPToolExecutor maps every
  /// `WarmServerClientError` that means "unreachable" — refused outright,
  /// or timed out while mid-boot, comfybox#389 — to it) until the engine
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

  /// Whether a transport failure means "the engine is unreachable or not
  /// yet answering" — gets the shared launchd hint — vs some other kind of
  /// failure that keeps its own message. comfybox#389: PR #382's review-
  /// round-2 fix special-cased only an outright refusal; a mid-boot engine
  /// (socket accepting, slow to answer) times out instead, which is the
  /// SAME actionable situation, just worded differently.
  public enum MCPUnreachableReason: Equatable, Sendable {
    /// Nothing answered the port at all.
    case refused
    /// The port answered (or DNS resolved) but no response came back in time.
    case timedOut
  }

  /// The message shown when nothing answers `host:port` — at startup
  /// (this file) and on every subsequent tool call that hits a
  /// `WarmServerClientError` classified by `nothingListeningMessage(for:)`
  /// (`MCPToolExecutor`). Kept as one function so every call site can
  /// never drift apart. `reason` only changes the opening sentence; the
  /// launchd hint itself is unconditional.
  public static func nothingListeningMessage(
    host: String, port: UInt16, reason: MCPUnreachableReason = .refused
  ) -> String {
    let situation: String
    switch reason {
    case .refused:
      situation = "Nothing is listening on \(host):\(port)."
    case .timedOut:
      situation = "\(host):\(port) did not answer in time (it may still be starting up)."
    }
    return situation + " This bridge never starts a server — " +
      "launchd (com.barkadabrew.comfybox) owns the engine lifecycle. Start it with: " +
      "\(launchctlKickstartCommand())"
  }

  /// Pure classification (comfybox#389): does this `WarmServerClientError`
  /// mean "the engine is unreachable or not answering yet" — in which
  /// case this returns the exact hint text to show, refused- or timed-out-
  /// worded as appropriate — or does it mean something else (a malformed
  /// URL, a generic network error unrelated to the engine simply not
  /// being up yet), in which case this returns nil and the caller keeps
  /// that error's own message. The ONE place that decides which transport
  /// errors get the launchd hint, so `MCPToolExecutor` never has to
  /// special-case individual error cases itself.
  public static func nothingListeningMessage(for error: WarmServerClientError) -> String? {
    switch error {
    case .connectionRefused(let host, let port):
      return nothingListeningMessage(host: host, port: port, reason: .refused)
    case .timedOut(let host, let port):
      return nothingListeningMessage(host: host, port: port, reason: .timedOut)
    case .invalidURL, .invalidResponse, .networkError:
      return nil
    }
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

  /// The full `ComfyBox mcp --help` text. comfybox#389: previously
  /// hand-duplicated in `Sources/ComfyBox/main.swift` with the kickstart
  /// command spelled out as a literal string, which could silently drift
  /// from `launchctlKickstartCommand()`. This is now the ONE source —
  /// `main.swift`'s `printMCPUsage()` just prints it — and it lives here
  /// (rather than in the `ComfyBox` executable target) so it's unit-
  /// testable from `ZImageTests` without building or running the CLI
  /// binary.
  public static func mcpHelpText() -> String {
    """
    Start MCP (Model Context Protocol) server mode.
    Bridges stdio JSON-RPC 2.0 to WarmServer HTTP API.

    Usage: ComfyBox mcp [options]
      --port                    WarmServer port to connect to (default: 7870)
      --host                    WarmServer host to connect to (default: 127.0.0.1)
      --help, -h                Show help

    The MCP server reads JSON-RPC requests from stdin and writes responses
    to stdout. All logging goes to stderr. Runs until stdin closes.

    This bridge NEVER starts a server (comfybox#153): launchd
    (com.barkadabrew.comfybox) owns the engine lifecycle. If a server is
    already listening on --port, healthy or not, the bridge connects to it.
    If nothing is listening, the bridge does NOT exit — launchd's
    RunAtLoad and the MCP host commonly race at login, so a bridge that
    starts before the engine must keep serving. It prints one warning to
    stderr (naming --port and the launchctl command below) and starts
    anyway; every tool call fails with that same message until the engine
    answers. Start the managed engine with:
      \(launchctlKickstartCommand())

    Registration:
      claude mcp add comfybox -- comfybox mcp --port 7870

    Tools:
      generate_image    Text-to-image / img2img generation
      swap_loras        Hot-swap active LoRA weights
      list_models       List supported model families
      list_styles       List style presets
      server_health     Server health and loaded model info
      queue_status      Generation queue status
      clear_queue       Cancel pending generation jobs
      list_loras        List available LoRA files
      shutdown_server   Graceful server shutdown
      system_stats      Hardware and system info
      apply_style       Apply style preset to prompt
    """
  }
}
