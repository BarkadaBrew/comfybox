// MCPJobModel.swift — ONE job model for every async render (comfybox#289)
//
// Before this, an MCP caller had three ways to ask "is it done yet?":
// `video_status` for video, `workflow_run_status` for workflow runs, and
// NOTHING for images (a `generate_image` call simply blocked until the render
// finished, or until the client's tool timeout killed it — comfybox#288).
//
// This file is the single mapping layer: one `job_id` namespace, one state
// vocabulary, one envelope shape
//
//     { job_id, kind, state, progress, result?, error?, retry_after_seconds? }
//
// for every kind of job the engine tracks. It is PURE — no HTTP, no clock —
// so each engine status shape is pinned by a unit test.
//
// The existing per-kind tools keep working and keep returning exactly what
// they return today (intent.md: the daemon contract is production — version
// or shim, never silently change). `get_job` is additive on top.

import Foundation

// MARK: - Kind

/// What produced a job id. Determines which status route answers for it.
///
/// - `image` / `swap`: `ImageJobTracker` (`GET /v1/generate/status/{id}`).
///   A LoRA-swap job replayed after a restart is recorded on its own id in
///   the SAME tracker (WarmServer `recordFailedReplay`, #339 review r4 item
///   3), so both read the same route; only the echoed `kind` differs.
/// - `video` / `storyboard`: `VideoJobTracker` (`GET /v1/video/status/{id}`).
///   Storyboard renders submit through `POST /v1/storyboard/render` and are
///   tracked as video jobs (see `executeRenderStoryboard`).
public enum MCPJobKind: String, CaseIterable, Sendable {
  case image
  case video
  case swap
  case storyboard

  /// The kinds a CALLER can name. `swap` is excluded: LoRA swaps are
  /// synchronous (`POST /v1/lora/swap` returns the result), so no caller ever
  /// holds a swap job id to poll. The only swap ids that exist come from
  /// queue replay, and they live in the image tracker keyed by id — so they
  /// resolve through the image probe and report as `image`. The case stays
  /// for that internal mapping; advertising it would be dead surface
  /// (PR #367 review r1, item 3).
  public static let selectableCases: [MCPJobKind] = [.image, .video, .storyboard]

  /// Status route template, in the parity parser's `{id}` normalization.
  public var statusPathTemplate: String {
    switch self {
    case .image, .swap: return "/v1/generate/status/{id}"
    case .video, .storyboard: return "/v1/video/status/{id}"
    }
  }

  /// Concrete status path for one job id.
  public func statusPath(jobId: String) -> String {
    let encoded = jobId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? jobId
    return statusPathTemplate.replacingOccurrences(of: "{id}", with: encoded)
  }

  /// Whether a job of this kind carries its live percent in its own status
  /// payload. Video does (`progress_percent`); the image tracker does not,
  /// so an image job's percent comes from the lock-based queue snapshot.
  public var carriesOwnProgressPercent: Bool {
    switch self {
    case .video, .storyboard: return true
    case .image, .swap: return false
    }
  }
}

// MARK: - State

/// The one state vocabulary. Deliberately four values: a polling client only
/// ever needs "keep polling" vs "stop, and why".
///
/// `completed` (not `succeeded`) because `MCPToolExecutor.mapHTTPResponse`
/// already normalizes `succeeded -> completed` for every existing tool — one
/// done-state across the whole surface.
public enum MCPJobState: String, Sendable {
  case queued
  case running
  case completed
  case failed
  /// The engine reported a state this build does not know (or none at all).
  ///
  /// TERMINAL on purpose. The old behaviour — coercing anything unrecognized
  /// to `running` — is precisely the infinite-poll failure this model exists
  /// to prevent: a client would poll a job that will never move again, and
  /// the reason would be invisible. Reporting it terminal stops the client
  /// and puts the raw engine value in `error` where a human can see it
  /// (PR #367 review r1, item 1).
  case unknown

  public var isTerminal: Bool {
    self == .completed || self == .failed || self == .unknown
  }
}

// MARK: - Mapping

public enum MCPJobModel {

  /// Order `get_job` probes when the caller did not name a kind: image first
  /// (the common case and the cheap tracker), then video.
  public static let probeOrder: [MCPJobKind] = [.image, .video]

  /// Map an engine state string onto the one vocabulary. Returns nil for a
  /// value neither tracker produces, so a future engine state surfaces as an
  /// explicit unknown rather than being silently coerced to "done".
  public static func state(fromEngine raw: String) -> MCPJobState? {
    switch raw {
    case "queued":
      return .queued
    case "processing", "running",
      // #1479: a video render checkpointed so a preempting image job can run.
      // NOT terminal — a client that stops polling here waits forever.
      "pausedForPreemption", "paused_for_preemption":
      return .running
    case "succeeded", "completed":
      return .completed
    case "failed":
      return .failed
    default:
      return nil
    }
  }

  /// The unified envelope for one job status payload.
  ///
  /// - Parameters:
  ///   - kind: echoed verbatim, so a caller who submitted a storyboard sees
  ///     `storyboard` even though the video tracker answered.
  ///   - status: the engine's own status JSON, already snake_cased on the wire.
  ///   - queueProgressPercent: the ACTIVE render's percent from
  ///     `GET /v1/queue`, supplied only when that snapshot says this job is
  ///     the active one (image jobs have no per-job percent of their own).
  public static func unify(
    kind: MCPJobKind, jobId: String, status: [String: Any], queueProgressPercent: Int? = nil
  ) -> [String: Any] {
    let rawState = (status["status"] as? String) ?? (status["state"] as? String) ?? ""
    let state = Self.state(fromEngine: rawState) ?? .unknown
    let ownPercent = kind.carriesOwnProgressPercent ? intValue(status["progress_percent"]) : nil

    var envelope: [String: Any] = [
      "job_id": (status["job_id"] as? String) ?? jobId,
      "kind": kind.rawValue,
      "state": state.rawValue,
      "progress": progress(state: state, percent: ownPercent ?? queueProgressPercent),
    ]

    if state == .completed, let result = result(kind: kind, status: status), !result.isEmpty {
      envelope["result"] = result
    }
    let engineError = (status["error"] as? String).flatMap { $0.isEmpty ? nil : $0 }
    if state == .unknown {
      // Name the raw value so the fix is one grep away, and keep whatever the
      // engine itself said alongside it.
      let raw = rawState.isEmpty ? "(absent)" : rawState
      envelope["error"] = engineError.map { "unmapped_state:\(raw) — \($0)" }
        ?? "unmapped_state:\(raw)"
    } else if let engineError {
      envelope["error"] = engineError
    }
    if let retry = intValue(status["retry_after_seconds"]) {
      envelope["retry_after_seconds"] = retry
    }
    return envelope
  }

  /// The envelope for a status read refused by the queue-recovery gate
  /// (`QueueRecoveryGate`: HTTP 503 + `retry_after_seconds`) while the engine
  /// replays its persisted queue after a restart. The job is not lost — it is
  /// queued — so this reports it as such with the server's own reschedule
  /// hint rather than as a failure.
  public static func recoveryEnvelope(
    kind: MCPJobKind, jobId: String, retryAfterSeconds: Int?
  ) -> [String: Any] {
    var envelope: [String: Any] = [
      "job_id": jobId,
      "kind": kind.rawValue,
      "state": MCPJobState.queued.rawValue,
      "progress": 0,
    ]
    if let retryAfterSeconds { envelope["retry_after_seconds"] = retryAfterSeconds }
    return envelope
  }

  /// The envelope an async SUBMIT returns (#288): the same shape as a poll,
  /// plus the name of the tool to poll with, so a small-model caller learns
  /// one pattern instead of three. Nil when the accept body carried no id —
  /// a submit without a job id is not a job.
  public static func submitEnvelope(kind: MCPJobKind, status: [String: Any]) -> [String: Any]? {
    guard let jobId = status["job_id"] as? String, !jobId.isEmpty else { return nil }
    var envelope = unify(kind: kind, jobId: jobId, status: status)
    envelope["poll_with"] = "get_job"
    return envelope
  }

  // MARK: - Private

  private static func progress(state: MCPJobState, percent: Int?) -> Int {
    switch state {
    case .completed: return 100
    case .queued: return 0
    case .running, .failed, .unknown: return clamp(percent ?? 0)
    }
  }

  private static func clamp(_ percent: Int) -> Int {
    min(100, max(0, percent))
  }

  /// The completion payload, per kind. Only the fields a caller acts on —
  /// the per-kind tools remain the way to get the full engine record
  /// (`applied_loras`, `resolved_config`, provenance, …).
  private static func result(kind: MCPJobKind, status: [String: Any]) -> [String: Any]? {
    var result: [String: Any] = [:]
    if let path = status["output_path"] as? String, !path.isEmpty {
      result["output_path"] = path
    }
    if let ms = intValue(status["duration_ms"]) { result["duration_ms"] = ms }
    switch kind {
    case .image, .swap:
      break
    case .video, .storyboard:
      if let bytes = intValue(status["file_size_bytes"]) { result["file_size_bytes"] = bytes }
      if let seconds = intValue(status["video_duration_seconds"]) {
        result["video_duration_seconds"] = seconds
      }
      if let frames = intValue(status["frame_count"]) { result["frame_count"] = frames }
    }
    return result
  }

  /// JSONSerialization hands back `NSNumber`; read it as Int without
  /// depending on which numeric type the encoder happened to emit.
  private static func intValue(_ any: Any?) -> Int? {
    if let i = any as? Int { return i }
    if let n = any as? NSNumber { return n.intValue }
    if let d = any as? Double { return Int(d) }
    return nil
  }
}
