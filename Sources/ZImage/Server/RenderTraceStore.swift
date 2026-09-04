import Foundation
import Logging

// Task #19 (specs/motion-tab-prompt-lab.md rev 2, Codex findings #1–3):
// render traces as APPEND-ONLY lifecycle events.
//
// - Identity: a stable `render_id` assigned before rendering. Paths are
//   mutable locators and never keys.
// - Self-describing: every line carries schema_version + task_kind so the
//   image follow-on reuses this store without migration.
// - Crash-visible: `submitted`/`started` are written when they happen; a
//   process death leaves an open trace that recovery marks `abandoned`.
//   Terminal-only logging cannot represent that (finding #3).
// - One serialized writer; monthly-rotated JSONL files.

public enum RenderTaskKind: String, Codable, Sendable {
  case videoRender = "video_render"
  case imageRender = "image_render"
  case img2img
  case inpaint
  case storyboard
}

public enum RenderTraceEventKind: String, Codable, Sendable {
  case submitted, started, terminal, abandoned
  /// Human quality verdict, appended post-hoc from the Gallery/Prompt Lab
  /// (task #19). Payload: axis, vote, and optional note.
  case rated
}

public struct RenderTraceEvent: Sendable {
  public let renderId: String
  public let event: RenderTraceEventKind
  public let taskKind: RenderTaskKind
  public let payload: [String: String]
  public let schemaVersion: Int
  public let ts: Date

  public init(
    renderId: String,
    event: RenderTraceEventKind,
    taskKind: RenderTaskKind,
    payload: [String: String],
    ts: Date = Date()
  ) {
    self.renderId = renderId
    self.event = event
    self.taskKind = taskKind
    self.payload = payload
    self.schemaVersion = 1
    self.ts = ts
  }
}

public final class RenderTraceStore: @unchecked Sendable {

  private let directory: URL
  private let queue = DispatchQueue(label: "comfybox.render-traces")  // the ONE writer
  private let logger = Logger(label: "z-image.render-traces")
  private let iso = ISO8601DateFormatter()

  public init(directory: URL? = nil) {
    self.directory = directory
      ?? FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".comfybox/traces")
    try? FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true)
  }

  // MARK: - Append

  public func append(_ event: RenderTraceEvent) {
    queue.async { self.write(event) }
  }

  /// Blocks until all queued appends hit disk. Tests + shutdown.
  public func flush() {
    queue.sync {}
  }

  private func write(_ event: RenderTraceEvent) {
    var obj: [String: Any] = [
      "schema_version": event.schemaVersion,
      "render_id": event.renderId,
      "event": event.event.rawValue,
      "task_kind": event.taskKind.rawValue,
      "ts": iso.string(from: event.ts),
    ]
    if !event.payload.isEmpty { obj["payload"] = event.payload }
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
          let line = String(data: data, encoding: .utf8) else {
      logger.error("trace event for \(event.renderId) failed to serialize — dropped")
      return
    }
    let url = currentFile()
    do {
      if FileManager.default.fileExists(atPath: url.path) {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data((line + "\n").utf8))
      } else {
        try (line + "\n").write(to: url, atomically: true, encoding: .utf8)
      }
    } catch {
      logger.error("trace append failed: \(error)")
    }
  }

  private func currentFile() -> URL {
    let fmt = DateFormatter()
    fmt.dateFormat = "yyyyMM"
    return directory.appendingPathComponent("renders-\(fmt.string(from: Date())).jsonl")
  }

  // MARK: - Read

  public func events(renderId: String) -> [RenderTraceEvent] {
    allEvents().filter { $0.renderId == renderId }
  }

  private func allEvents() -> [RenderTraceEvent] {
    let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil))?
      .filter { $0.pathExtension == "jsonl" }.sorted { $0.path < $1.path } ?? []
    var out: [RenderTraceEvent] = []
    for f in files {
      guard let text = try? String(contentsOf: f, encoding: .utf8) else { continue }
      for line in text.split(separator: "\n") {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              let id = obj["render_id"] as? String,
              let ev = (obj["event"] as? String).flatMap(RenderTraceEventKind.init(rawValue:)),
              let kind = (obj["task_kind"] as? String).flatMap(RenderTaskKind.init(rawValue:))
        else { continue }
        let ts = (obj["ts"] as? String).flatMap { iso.date(from: $0) } ?? Date.distantPast
        let payload = (obj["payload"] as? [String: String]) ?? [:]
        out.append(RenderTraceEvent(renderId: id, event: ev, taskKind: kind, payload: payload, ts: ts))
      }
    }
    return out
  }

  /// Newest-first summaries grouped by render_id — the Prompt Lab feed.
  public struct TraceSummary: Codable, Sendable {
    public let renderId: String
    public let taskKind: String
    public let events: [String]
    public let submittedAt: Date?
    public let status: String       // running | succeeded | failed | abandoned | rated:<axis>
    public let prompt: String?
    public let outputPath: String?
    public let optimizationAttemptId: String?
    public let config: String?
    public let error: String?
    public let rating: String?
    /// comfybox#328 (Codex round 1, finding 2): non-nil (`"beat_schedule"`)
    /// when this render's prompt enhancement was skipped to let a
    /// `beat_schedule` locate verbatim — GET /v1/video/traces was claiming
    /// to surface this via the raw submitted payload, but this summary type
    /// is what that endpoint actually returns, and it dropped the field.
    public let enhancementSkipped: String?
    /// comfybox#328: non-nil (`"i2v_unsupported"`) when a `beat_schedule`
    /// on this render's (I2V) request was dropped before reaching the
    /// generator.
    public let beatScheduleIgnored: String?
  }

  public func recentSummaries(limit: Int = 50) -> [TraceSummary] {
    var grouped: [String: [RenderTraceEvent]] = [:]
    for e in allEvents() { grouped[e.renderId, default: []].append(e) }
    let summaries: [TraceSummary] = grouped.map { id, events in
      let submitted = events.first { $0.event == .submitted }
      let terminal = events.last { $0.event == .terminal }
      let rated = events.last { $0.event == .rated }
      let status: String
      if events.contains(where: { $0.event == .abandoned }) { status = "abandoned" }
      else if let t = terminal { status = t.payload["status"] ?? "succeeded" }
      else { status = "running" }
      return TraceSummary(
        renderId: id,
        taskKind: events.first?.taskKind.rawValue ?? "video_render",
        events: events.map(\.event.rawValue),
        submittedAt: submitted?.ts ?? events.first?.ts,
        status: status,
        prompt: submitted?.payload["prompt"] ?? submitted?.payload["intent"],
        outputPath: terminal?.payload["output_path"],
        optimizationAttemptId: submitted?.payload["optimization_attempt_id"],
        config: submitted?.payload["config"],
        error: terminal?.payload["error"],
        rating: rated.map { "\($0.payload["axis"] ?? "overall"):\($0.payload["vote"] ?? "?")" },
        enhancementSkipped: submitted?.payload["enhancement_skipped"],
        beatScheduleIgnored: submitted?.payload["beat_schedule_ignored"]
      )
    }
    return summaries
      .sorted { ($0.submittedAt ?? .distantPast) > ($1.submittedAt ?? .distantPast) }
      .prefix(limit).map { $0 }
  }

  // MARK: - Recovery

  /// Mark every trace with no terminal/abandoned event as `abandoned`.
  /// Call once at server startup. Returns the number marked.
  @discardableResult
  public func markAbandonedOpenTraces() -> Int {
    var lastKind: [String: (RenderTraceEventKind, RenderTaskKind)] = [:]
    for e in allEvents() {
      lastKind[e.renderId] = (e.event, e.taskKind)
    }
    var marked = 0
    for (id, (kind, task)) in lastKind where kind == .submitted || kind == .started {
      append(RenderTraceEvent(
        renderId: id, event: .abandoned, taskKind: task,
        payload: ["reason": "no terminal event at startup — process died mid-render"]))
      marked += 1
    }
    flush()
    if marked > 0 { logger.warning("marked \(marked) open render trace(s) abandoned") }
    return marked
  }
}
