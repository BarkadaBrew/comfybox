// AuditLog.swift — Append-only JSONL audit trail (~/.comfybox/audit-log.jsonl).
//
// Records notable server events (generation submitted/completed/failed, model
// load/unload, config change) so the desktop UI can surface an activity history.
// Ported for parity with the Coffee Shop image service audit log
// (src/service.ts `appendAuditLog` / `getAuditLog`): one JSON object per line,
// appended synchronously, read back most-recent-first with a limit.
//
// Writes are serialized on a private queue so concurrent `append` calls from the
// server's request handlers never interleave partial lines in the file.

import Foundation

/// The category of an audited event. A wrapper over a string (rather than a closed
/// enum) so new event kinds decode cleanly on older builds — the schema can grow
/// without breaking existing `audit-log.jsonl` files.
public struct AuditKind: RawRepresentable, Codable, Equatable, Hashable, Sendable, ExpressibleByStringLiteral {
  public let rawValue: String

  public init(rawValue: String) { self.rawValue = rawValue }
  public init(_ rawValue: String) { self.rawValue = rawValue }
  public init(stringLiteral value: String) { self.rawValue = value }

  public init(from decoder: Decoder) throws {
    rawValue = try decoder.singleValueContainer().decode(String.self)
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.singleValueContainer()
    try c.encode(rawValue)
  }

  // Well-known kinds. Callers may also construct arbitrary kinds via string literal.
  public static let generationSubmitted: AuditKind = "generation.submitted"
  public static let generationCompleted: AuditKind = "generation.completed"
  public static let generationFailed: AuditKind = "generation.failed"
  public static let generationCancelled: AuditKind = "generation.cancelled"
  public static let modelLoad: AuditKind = "model.load"
  public static let modelUnload: AuditKind = "model.unload"
  public static let configChange: AuditKind = "config.change"
}

/// One line of the audit log.
public struct AuditEntry: Codable, Equatable, Sendable {
  /// When the event occurred.
  public var timestamp: Date
  /// What kind of event this is.
  public var kind: AuditKind
  /// Human-readable description of the event.
  public var message: String
  /// Optional structured detail (e.g. `["jobId": "abc", "durationMs": "1234"]`).
  public var metadata: [String: String]?

  public init(
    timestamp: Date = Date(),
    kind: AuditKind,
    message: String,
    metadata: [String: String]? = nil
  ) {
    self.timestamp = timestamp
    self.kind = kind
    self.message = message
    self.metadata = metadata
  }

  private enum CodingKeys: String, CodingKey {
    case timestamp, kind, message, metadata
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    timestamp = try c.decodeIfPresent(Date.self, forKey: .timestamp) ?? Date(timeIntervalSince1970: 0)
    kind = try c.decodeIfPresent(AuditKind.self, forKey: .kind) ?? AuditKind("unknown")
    message = try c.decodeIfPresent(String.self, forKey: .message) ?? ""
    metadata = try c.decodeIfPresent([String: String].self, forKey: .metadata)
  }
}

/// Append-only JSONL audit log. Thread-safe: all file mutations run on a private
/// serial queue so appends never interleave.
public final class AuditLog: @unchecked Sendable {
  /// `~/.comfybox/audit-log.jsonl`, or `$COMFYBOX_STATE_DIR/audit-log.jsonl`
  /// (K-FIX-1: same override every other `.comfybox` path honors, so a test
  /// that constructs a default `AuditLog()` never appends to the LIVE file).
  public static func defaultPath() -> URL {
    ComfyBoxServerConfig.stateDirectory().appendingPathComponent("audit-log.jsonl")
  }

  private let path: URL
  private let fileManager: FileManager
  private let queue = DispatchQueue(label: "com.comfybox.audit-log")

  private static func makeEncoder() -> JSONEncoder {
    let e = JSONEncoder()
    e.dateEncodingStrategy = .iso8601
    e.outputFormatting = [.sortedKeys] // single line — no .prettyPrinted
    return e
  }

  private static func makeDecoder() -> JSONDecoder {
    let d = JSONDecoder()
    d.dateDecodingStrategy = .iso8601
    return d
  }

  public init(path: URL = AuditLog.defaultPath(), fileManager: FileManager = .default) {
    self.path = path
    self.fileManager = fileManager
  }

  /// Append one event to the log. Serialized against other appends. Failures to
  /// write are swallowed — audit logging must never crash the server or a request.
  public func append(_ entry: AuditEntry) {
    queue.sync {
      guard let line = try? Self.makeEncoder().encode(entry) else { return }
      var data = line
      data.append(0x0A) // '\n'

      let dir = path.deletingLastPathComponent()
      try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)

      if let handle = try? FileHandle(forWritingTo: path) {
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
      } else {
        // File doesn't exist yet — create it with this first line.
        try? data.write(to: path, options: .atomic)
      }
    }
  }

  /// Convenience: build and append an entry in one call.
  public func append(kind: AuditKind, message: String, metadata: [String: String]? = nil) {
    append(AuditEntry(kind: kind, message: message, metadata: metadata))
  }

  /// Return up to `limit` most-recent entries, newest first. Malformed lines are
  /// skipped. Returns an empty array if the log does not yet exist.
  public func recent(limit: Int = 100) -> [AuditEntry] {
    queue.sync {
      guard limit > 0,
            fileManager.fileExists(atPath: path.path),
            let text = try? String(contentsOf: path, encoding: .utf8) else {
        return []
      }
      let decoder = Self.makeDecoder()
      let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
      var result: [AuditEntry] = []
      result.reserveCapacity(min(limit, lines.count))
      // Walk from the end (most recent) backwards until we have `limit` entries.
      for line in lines.reversed() {
        guard let data = line.data(using: .utf8),
              let entry = try? decoder.decode(AuditEntry.self, from: data) else {
          continue
        }
        result.append(entry)
        if result.count >= limit { break }
      }
      return result
    }
  }
}
