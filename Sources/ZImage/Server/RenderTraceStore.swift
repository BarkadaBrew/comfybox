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
