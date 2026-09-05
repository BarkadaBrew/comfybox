// MCPProgress.swift — MCP progress notifications during renders (comfybox#292)
//
// A 90-second render is indistinguishable from a hang: the MCP client sees
// silence until the result lands. The MCP spec's answer is
// `notifications/progress`, sent ONLY when the client asked for it by putting
// a `progressToken` in the request's `_meta`.
//
// Two rules shape everything here:
//
//   1. Poll the engine at a bounded cadence, and only while a progress token
//      exists. The Mac is shared with LM Studio and the embeddings service
//      (intent.md: memory is a shared resource); a poll nobody asked for is
//      pure waste.
//   2. Nothing outlives the request (intent.md: no orphaned processes). The
//      poller is a child task that is cancelled AND awaited before the tool
//      call returns, on success and on error alike.
//
// The source of truth for progress is `GET /v1/queue`, which is served from
// the lock-protected live-health snapshot rather than through the
// `WarmServerCoordinator` actor — so, unlike `/health`, it still answers
// while a render holds the actor (comfybox#217).

import Foundation

// MARK: - Reporter

/// Sink for MCP progress notifications. Implemented by `MCPServer` (writing a
/// JSON-RPC notification to stdout); tests substitute a recorder.
public protocol MCPProgressReporter: Sendable {
  func report(progress: Double, total: Double?, message: String?) async
}

// MARK: - Wire format

/// The `notifications/progress` JSON-RPC notification (MCP basic/utilities).
public enum MCPProgressNotification {
  public static let method = "notifications/progress"

  /// Build the notification object. `token` is echoed VERBATIM — MCP allows a
  /// string or an integer progress token, and a client that sent `7` must not
  /// get `"7"` back.
  public static func json(
    token: AnyCodable, progress: Double, total: Double?, message: String?
  ) -> [String: Any] {
    var params: [String: Any] = [
      "progressToken": token.rawValue,
      "progress": progress,
    ]
    if let total { params["total"] = total }
    if let message { params["message"] = message }
    return [
      "jsonrpc": "2.0",
      "method": method,
      "params": params,
    ]
  }
}

// MARK: - Scheduler (pure)

/// Cadence and termination rules for progress polling, as a pure function of
/// (elapsed, last emitted value, latest engine snapshot). No clock, no I/O —
/// the reason the rules below are unit-testable at all.
public enum MCPProgressScheduler {

  /// Poll cadence. Two seconds is the same cadence the Desktop app polls at,
  /// and is cheap: `GET /v1/queue` is a lock-based snapshot read.
  public static let defaultInterval: TimeInterval = 2.0

  /// Hard ceiling on how long one tool call will keep polling. A render that
  /// runs past this still completes and still returns its result — it just
  /// stops narrating. Bounded so a wedged render cannot leave a poller
  /// spinning for the life of the process.
  public static let maxDuration: TimeInterval = 1800

  /// What the engine's queue route says right now.
  public struct Snapshot: Sendable, Equatable {
    public let isRendering: Bool
    public let progressPercent: Int?
    public let pending: Int
    public let activeJobId: String?

    public init(isRendering: Bool, progressPercent: Int?, pending: Int, activeJobId: String?) {
      self.isRendering = isRendering
      self.progressPercent = progressPercent
      self.pending = pending
      self.activeJobId = activeJobId
    }

    /// Parse `GET /v1/queue`'s payload (`buildQueuePayloadData`). Returns nil
    /// for anything unparseable, which the scheduler treats as "skip this
    /// tick", never as "stop".
    public init?(queuePayload: Data) {
      guard let obj = try? JSONSerialization.jsonObject(with: queuePayload) as? [String: Any] else {
        return nil
      }
      self.isRendering = (obj["is_rendering"] as? Bool) ?? false
      self.progressPercent = (obj["progress_percent"] as? NSNumber)?.intValue
      self.pending = (obj["pending"] as? [Any])?.count ?? 0
      self.activeJobId = obj["active_job_id"] as? String
    }
  }

  public enum Decision: Sendable, Equatable {
    case emit(progress: Double, total: Double, message: String)
    case skip
    case stop(reason: String)
  }

  /// Decide what this tick should do.
  ///
  /// - A missing snapshot (poll failed, engine busy) is a SKIP, not a stop:
  ///   losing one progress tick must never end the narration of a render that
  ///   is still running.
  /// - Progress must increase monotonically per MCP; an unchanged or lower
  ///   percent emits nothing rather than repeating or going backwards (the
  ///   queue snapshot describes the ACTIVE render, which can roll over to a
  ///   different job mid-poll).
  public static func decide(
    elapsed: TimeInterval,
    maxDuration: TimeInterval = MCPProgressScheduler.maxDuration,
    lastEmitted: Double?,
    snapshot: Snapshot?
  ) -> Decision {
    if elapsed >= maxDuration {
      return .stop(reason: "progress polling deadline reached after \(Int(maxDuration))s")
    }
    guard let snapshot else { return .skip }

    let percent = Double(min(100, max(0, snapshot.progressPercent ?? 0)))
    if let lastEmitted, percent <= lastEmitted { return .skip }

    let message: String
    if snapshot.isRendering {
      message = snapshot.pending > 0
        ? "rendering — \(Int(percent))% (\(snapshot.pending) queued behind it)"
        : "rendering — \(Int(percent))%"
    } else if snapshot.pending > 0 {
      message = "waiting in the engine queue — \(snapshot.pending) job(s) pending"
    } else {
      message = "submitted — waiting for the engine"
    }
    return .emit(progress: percent, total: 100, message: message)
  }
}
