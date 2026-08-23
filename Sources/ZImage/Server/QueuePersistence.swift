// QueuePersistence.swift — durable FIFO render queue.
//
// The render queue previously lived only in the WarmServerCoordinator
// actor's memory: a crash (jetsam, the SIGTERM-mid-render shutdown bug,
// a future bug) silently dropped every pending and in-flight job with no
// trace. This persists a recoverable snapshot to disk on every queue
// mutation so a restart can replay lost work.
//
// Only wire-driven job kinds with a raw JSON request body can be recovered
// this way — "generate" (POST /v1/generate and /v1/generate/async, which
// both funnel through WarmServerCoordinator.enqueueGenerate) and "lora_swap".
// ControlNet generate closes over resolved temp-file paths built before the
// queue is ever reached, and modelSwitch/localVideo close over live in-memory
// state (a Swift closure, not data) — none of those can be serialized, so
// they are simply never persisted. shutdown never needs recovery.

import Foundation

/// One recoverable queue entry: enough to replay the original HTTP request
/// through the exact same decode path a live route handler uses.
struct PersistedQueueJob: Codable, Sendable {
  let id: String
  let kind: String  // "generate" | "lora_swap"
  let source: String
  let enqueuedAt: Date
  let rawBody: Data
}

/// Full on-disk queue snapshot: the job that was mid-render (if any, so a
/// crash during a render doesn't lose it) plus everything still waiting
/// behind it, in original order.
struct PersistedQueueState: Codable, Sendable {
  var active: PersistedQueueJob?
  var pending: [PersistedQueueJob]
}

/// Reads/writes the queue snapshot at `~/.comfybox/queue-state.json`. Not
/// actor-isolated itself — the only writer is the WarmServerCoordinator
/// actor, which already serializes access to the queue it mirrors here.
enum QueueStateStore {
  /// The engine's state directory — `~/.comfybox`, or `COMFYBOX_STATE_DIR`
  /// when set.
  ///
  /// COMPUTED, not a cached `static let`: the override has to be readable
  /// after process start, because a test that drives a real coordinator would
  /// otherwise DELETE the live engine's queue snapshot (`save` of an empty
  /// state removes the file) and read its pause sentinel. Both are cheap and
  /// called only at queue transitions.
  static var stateDirectory: URL {
    if let override = ProcessInfo.processInfo.environment["COMFYBOX_STATE_DIR"], !override.isEmpty {
      let dir = URL(fileURLWithPath: (override as NSString).expandingTildeInPath, isDirectory: true)
      try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
      return dir
    }
    let dir = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".comfybox", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  static var path: URL { stateDirectory.appendingPathComponent("queue-state.json") }

  static func save(_ state: PersistedQueueState) {
    guard state.active != nil || !state.pending.isEmpty else {
      try? FileManager.default.removeItem(at: path)
      return
    }
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    guard let data = try? encoder.encode(state) else { return }
    try? data.write(to: path, options: .atomic)
  }

  static func load() -> PersistedQueueState? {
    guard let data = try? Data(contentsOf: path) else { return nil }
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return try? decoder.decode(PersistedQueueState.self, from: data)
  }
}
