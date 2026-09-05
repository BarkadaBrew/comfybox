// QueueLifecycleLedger.swift — comfybox#283 / comfybox#217: a pure,
// append-only INSTRUMENT for the render queue's lifecycle. TELEMETRY ONLY.
//
// #283's investigation found that a process restart re-enqueues the active
// job under its ORIGINAL id, re-renders it from step 1, and NOTHING in the
// system reports that accurately: the same job id survives with a fresh
// `enqueuedAt`, and nothing distinguishes "a job recovered after a restart"
// from "a brand-new job" in any log or endpoint. #217 separately found that
// `/health` and the Desktop queue/progress UI go stale during a render
// because they hop onto the busy `WarmServerCoordinator` actor. Both issues
// need an accurate, always-available record of what actually happened to a
// job — enqueued, admitted, started, ticked, checkpointed, resumed,
// interrupted, completed, failed, replayed after a restart, or dropped —
// before either issue's proposed *behavior* changes can be evaluated safely.
//
// This file changes NOTHING about queue behavior. Every call site that wires
// it (WarmServer.swift) is a read-only observation of a transition that
// already happens; recording an event never affects what the transition
// does or whether it succeeds.
//
// Storage is two-tier, mirroring `LiveHealthState` (in-memory, lock-free
// reads for HTTP routes) and `AuditLog` (durable JSONL under the state
// directory):
//   - an in-memory ring of the last `capacity` events, for
//     `GET /v1/queue/lifecycle` and the `last_event`/`lifecycle_tail`
//     fields, with no actor hop;
//   - `~/.comfybox/queue-lifecycle.jsonl` (or `$COMFYBOX_STATE_DIR`'s, same
//     override every other `.comfybox` path honors — see
//     `QueueStateStore.stateDirectory`), which survives a restart. The ring
//     does NOT survive a restart (a fresh process starts a fresh
//     `QueueLifecycleLedger`), so the ring and the sequence counter are
//     reseeded from the JSONL tail on init — a restart's own first events
//     continue the same monotonic sequence, and the ring is not empty the
//     instant the engine comes back up.
//
// Boot-id: `bootId` is a fresh UUID generated once per `QueueLifecycleLedger`
// instance (i.e. once per process start, since the engine constructs exactly
// one of these at startup). A restart is therefore visible directly in the
// event stream — two consecutive JSONL lines with different `bootId`s — with
// no cross-referencing of logs required, which is exactly the
// operator-visible signal #283 finding 1 asks for.

import Foundation

/// One event kind in a job's lifecycle. Raw values are the wire vocabulary
/// (`GET /v1/queue/lifecycle`, `last_event`, `lifecycle_tail`) — additive,
/// never renamed without a version bump (intent.md: the daemon contract is
/// production).
public enum QueueLifecycleEventKind: String, Codable, Sendable, CaseIterable {
  /// A job was appended to `pending` (an enqueue* call returned successfully;
  /// this fires before the caller's continuation is even created in some
  /// paths, so it is recorded from inside the enqueue methods themselves).
  case enqueued
  /// The job was dequeued from `pending` into the "active" slot — the FIFO
  /// process loop picked it up. This is what #283's `queue_remaining`
  /// finding (issue 2) is about: an "admitted" job that has not yet
  /// completed is exactly the state that endpoint reports as idle.
  case admitted
  /// The job's actual synchronous GPU section began (`activeRenderStartedAt`
  /// is set at every one of these sites) — distinct from `admitted` because
  /// a job can sit "active" briefly before its render method's own setup
  /// (model reload, LoRA application, recipe validation) reaches the GPU.
  case started
  /// A bounded-rate denoising/render tick. The ledger itself throttles these
  /// (see `progressMinInterval`) so a fast-ticking render cannot flood the
  /// ring or the JSONL file.
  case progress
  /// An LTX-2 video render yielded to a preempting image job, carrying a
  /// resumable checkpoint (`#1479`). Never fires for image generate/LoRA
  /// swap — those have no checkpoint mechanism (#283 finding 1: a restart
  /// always restarts them from step 1).
  case checkpointed
  /// The checkpointed video render resumed — either from its own checkpoint
  /// (normal preemption resolution) or, on the replay path, would resume
  /// from a checkpoint if one existed (never does today — see
  /// `ReplayClassifier`).
  case resumed
  /// An operator-visible interrupt (`/v1/queue/interrupt`, or a video
  /// render's own cancellation) stopped the job — not a pipeline failure.
  case interrupted
  /// The job finished successfully.
  case completed
  /// The job finished with an error (not an interrupt).
  case failed
  /// `recoverPersistedQueue()` replayed a job left over from before a
  /// restart, under the SAME id it had before (`originalJobId` is that same
  /// id, recorded for symmetry with a future world where a replay could run
  /// under a fresh id). `fromStep1` is `ReplayClassifier`'s verdict: true
  /// unless an unresolved `checkpointed` event exists for this job (never
  /// true in production today for `generate`/`lora_swap`, the only two kinds
  /// `queue-state.json` can persist).
  case replayedAfterRestart = "replayed_after_restart"
  /// A pending (not yet admitted) job was cancelled — `DELETE /v1/queue/{id}`
  /// or `/v1/queue/clear` — before it ever ran. Distinct from `interrupted`
  /// (which stops a job already running) and from `failed`.
  case dropped
}

/// One immutable lifecycle event. `sequence` is a process-wide monotonic
/// counter (not per-job), so the ring and the JSONL file preserve true
/// emission order across every job interleaved on the one render queue —
/// including across a restart, since `sequence` is reseeded from the JSONL
/// tail rather than reset to zero.
public struct QueueLifecycleEvent: Codable, Sendable, Equatable {
  public let sequence: UInt64
  public let bootId: String
  public let wallTime: Date
  public let jobId: String
  public let kind: QueueLifecycleEventKind
  /// The queue-job kind ("generate", "lora_swap", "video", …) at the time of
  /// this event — see `QueueJobKind` (QueueRecoveryGate.swift), the single
  /// source of truth for that vocabulary. Optional: a call site that does
  /// not have it cheaply to hand is not required to plumb it through.
  public var jobKind: String?
  /// Submitting client/app ("desktop", "comfyui/krita", "bree", "api", …) —
  /// see `PendingJob.source`. Carried here so a lifecycle query answers
  /// #283 finding 4 ("submissions are unattributable") for the events that
  /// have it.
  public var source: String?
  public var step: Int?
  public var totalSteps: Int?
  public var chunk: Int?
  public var totalChunks: Int?
  public var percent: Int?
  /// `.replayedAfterRestart` only: the id of the job being replayed (today
  /// always equal to `jobId` itself — AC-18 replays under the original id —
  /// carried as its own field so a future replay-under-a-fresh-id design
  /// does not have to overload `jobId`'s meaning).
  public var originalJobId: String?
  /// `.replayedAfterRestart` only: `ReplayClassifier`'s verdict.
  public var fromStep1: Bool?
  /// Free-text context: an error description or cancellation reason on
  /// `.interrupted`/`.failed`/`.dropped`, or the LTX-2 phase
  /// (`LTX2Phase.rawValue` — "baseDenoise"/"refineDenoise") on
  /// `.checkpointed`/`.resumed`. Never used for control flow.
  public var reason: String?
  /// `.completed` only, where known cheaply.
  public var durationMs: Int?

  public init(
    sequence: UInt64,
    bootId: String,
    wallTime: Date,
    jobId: String,
    kind: QueueLifecycleEventKind,
    jobKind: String? = nil,
    source: String? = nil,
    step: Int? = nil,
    totalSteps: Int? = nil,
    chunk: Int? = nil,
    totalChunks: Int? = nil,
    percent: Int? = nil,
    originalJobId: String? = nil,
    fromStep1: Bool? = nil,
    reason: String? = nil,
    durationMs: Int? = nil
  ) {
    self.sequence = sequence
    self.bootId = bootId
    self.wallTime = wallTime
    self.jobId = jobId
    self.kind = kind
    self.jobKind = jobKind
    self.source = source
    self.step = step
    self.totalSteps = totalSteps
    self.chunk = chunk
    self.totalChunks = totalChunks
    self.percent = percent
    self.originalJobId = originalJobId
    self.fromStep1 = fromStep1
    self.reason = reason
    self.durationMs = durationMs
  }
}

/// Pure classification: given a job's PRIOR recorded events (in emission
/// order), would replaying it now resume from a checkpoint or restart at
/// step 1? An "open" checkpoint is a `.checkpointed` event with no later
/// `.resumed` (or terminal: `.completed`/`.failed`/`.dropped`) event closing
/// it out — a checkpoint already resumed, or superseded by a terminal
/// outcome, has nothing left to resume FROM.
///
/// #283 finding 1: in production today, `generate`/`lora_swap` — the only
/// two queue-job kinds `queue-state.json` can persist and therefore the only
/// two `recoverPersistedQueue()` ever replays — never emit `.checkpointed`
/// (only LTX-2 video checkpoints, via `#1479`'s preemption mechanism, and
/// video is never persisted/replayed — see QueuePersistence.swift's file
/// doc comment). So `classify` always returns `fromStep1: true` for a real
/// restart replay today; kept general, and tested directly against
/// synthetic sequences, so the verdict is correct if that ever changes
/// rather than hard-coding today's answer.
public enum ReplayClassifier {
  public struct Classification: Equatable {
    public let fromStep1: Bool
    public let resumeStep: Int?
    public let resumeChunk: Int?

    public init(fromStep1: Bool, resumeStep: Int? = nil, resumeChunk: Int? = nil) {
      self.fromStep1 = fromStep1
      self.resumeStep = resumeStep
      self.resumeChunk = resumeChunk
    }
  }

  public static func classify(priorEvents: [QueueLifecycleEvent]) -> Classification {
    var openCheckpoint: QueueLifecycleEvent?
    for event in priorEvents {
      switch event.kind {
      case .checkpointed:
        openCheckpoint = event
      case .resumed, .completed, .failed, .dropped:
        openCheckpoint = nil
      default:
        break
      }
    }
    guard let checkpoint = openCheckpoint else {
      return Classification(fromStep1: true)
    }
    return Classification(fromStep1: false, resumeStep: checkpoint.step, resumeChunk: checkpoint.chunk)
  }
}

/// Append-only, thread-safe (single serial `DispatchQueue`, mirroring
/// `AuditLog`'s file-write discipline) lifecycle ledger. `record` is cheap
/// (an array append + a small JSON line) relative to how rarely a queue
/// transition actually happens compared to render duration — the same
/// tradeoff `persistQueueState()`'s doc comment makes for the queue snapshot.
public final class QueueLifecycleLedger: @unchecked Sendable {
  public static let defaultCapacity = 4000
  public static let defaultProgressMinInterval: TimeInterval = 1.0

  public let bootId: String
  private let capacity: Int
  private let jsonlURL: URL?
  private let progressMinInterval: TimeInterval
  private let clock: @Sendable () -> Date
  private let fileManager: FileManager
  private let queue = DispatchQueue(label: "com.comfybox.queue-lifecycle-ledger")

  private var ring: [QueueLifecycleEvent] = []
  private var nextSequence: UInt64
  private var lastProgressAt: [String: Date] = [:]

  /// `~/.comfybox/queue-lifecycle.jsonl`, or `$COMFYBOX_STATE_DIR`'s — the
  /// same override `QueueStateStore`/`AuditLog` honor, so a test pointing
  /// `COMFYBOX_STATE_DIR` at a temp directory never touches the LIVE file.
  public static var defaultJSONLPath: URL {
    QueueStateStore.stateDirectory.appendingPathComponent("queue-lifecycle.jsonl")
  }

  public init(
    capacity: Int = QueueLifecycleLedger.defaultCapacity,
    bootId: String = UUID().uuidString,
    jsonlURL: URL? = QueueLifecycleLedger.defaultJSONLPath,
    progressMinInterval: TimeInterval = QueueLifecycleLedger.defaultProgressMinInterval,
    fileManager: FileManager = .default,
    clock: @escaping @Sendable () -> Date = Date.init
  ) {
    self.capacity = max(1, capacity)
    self.bootId = bootId
    self.jsonlURL = jsonlURL
    self.progressMinInterval = progressMinInterval
    self.fileManager = fileManager
    self.clock = clock
    let seeded = jsonlURL.map { QueueLifecycleLedger.loadJSONL(from: $0, fileManager: fileManager) } ?? []
    self.ring = Array(seeded.suffix(self.capacity))
    self.nextSequence = (seeded.map { $0.sequence }.max()).map { $0 + 1 } ?? 0
  }

  // MARK: - Writing

  /// Record one event. Returns the recorded event, or `nil` when a
  /// `.progress` tick was throttled (see `progressMinInterval`) — throttled
  /// ticks consume no sequence number and are never written anywhere, so
  /// the bound is real, not just a display-side truncation.
  @discardableResult
  public func record(
    jobId: String,
    kind: QueueLifecycleEventKind,
    jobKind: String? = nil,
    source: String? = nil,
    step: Int? = nil,
    totalSteps: Int? = nil,
    chunk: Int? = nil,
    totalChunks: Int? = nil,
    percent: Int? = nil,
    originalJobId: String? = nil,
    fromStep1: Bool? = nil,
    reason: String? = nil,
    durationMs: Int? = nil
  ) -> QueueLifecycleEvent? {
    queue.sync {
      if kind == .progress {
        let now = clock()
        if let last = lastProgressAt[jobId], now.timeIntervalSince(last) < progressMinInterval {
          return nil
        }
        lastProgressAt[jobId] = now
      }
      let event = QueueLifecycleEvent(
        sequence: nextSequence, bootId: bootId, wallTime: clock(), jobId: jobId, kind: kind,
        jobKind: jobKind, source: source, step: step, totalSteps: totalSteps, chunk: chunk,
        totalChunks: totalChunks, percent: percent, originalJobId: originalJobId,
        fromStep1: fromStep1, reason: reason, durationMs: durationMs)
      nextSequence += 1
      ring.append(event)
      if ring.count > capacity {
        ring.removeFirst(ring.count - capacity)
      }
      switch kind {
      case .completed, .failed, .interrupted, .dropped:
        // A terminal event closes out this job id's progress-throttle
        // window — an id reused later (should never happen; ids are UUIDs)
        // or simply garbage in the dictionary otherwise grows unbounded
        // across a long-running engine's lifetime.
        lastProgressAt.removeValue(forKey: jobId)
      default:
        break
      }
      appendJSONL(event)
      return event
    }
  }

  /// Convenience: `ReplayClassifier.classify` over this job id's own prior
  /// history — the call `recoverPersistedQueue()` makes right before
  /// recording `.replayedAfterRestart`.
  public func classifyReplay(jobId: String) -> ReplayClassifier.Classification {
    ReplayClassifier.classify(priorEvents: events(jobId: jobId))
  }

  // MARK: - Reading

  /// Events in emission order, optionally filtered to one job id and/or
  /// capped to the last `limit`. No actor hop — reads the in-memory ring
  /// under the same serial queue writes use.
  public func events(jobId: String? = nil, limit: Int? = nil) -> [QueueLifecycleEvent] {
    queue.sync {
      let all = jobId == nil ? ring : ring.filter { $0.jobId == jobId }
      if let limit, limit < all.count {
        return Array(all.suffix(limit))
      }
      return all
    }
  }

  public func lastEvent(jobId: String) -> QueueLifecycleEvent? {
    events(jobId: jobId).last
  }

  public func tail(jobId: String, count: Int) -> [QueueLifecycleEvent] {
    events(jobId: jobId, limit: count)
  }

  // MARK: - JSONL persistence

  private func appendJSONL(_ event: QueueLifecycleEvent) {
    guard let url = jsonlURL else { return }
    guard let line = try? Self.makeEncoder().encode(event) else { return }
    var data = line
    data.append(0x0A)  // '\n'

    let dir = url.deletingLastPathComponent()
    try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)

    if let handle = try? FileHandle(forWritingTo: url) {
      defer { try? handle.close() }
      _ = try? handle.seekToEnd()
      try? handle.write(contentsOf: data)
    } else {
      // File doesn't exist yet — create it with this first line.
      try? data.write(to: url, options: .atomic)
    }
  }

  private static func makeEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.sortedKeys]  // single line — no .prettyPrinted
    return encoder
  }

  private static func makeDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return decoder
  }

  /// Read back every well-formed line of a lifecycle JSONL file, in file
  /// order (append-only, so file order is emission order across every boot
  /// that ever wrote to it). A malformed trailing line (a write that raced a
  /// crash) is skipped rather than failing the whole read — same discipline
  /// as `AuditLog.recent`.
  public static func loadJSONL(from url: URL, fileManager: FileManager = .default) -> [QueueLifecycleEvent] {
    guard fileManager.fileExists(atPath: url.path),
          let text = try? String(contentsOf: url, encoding: .utf8)
    else { return [] }
    let decoder = makeDecoder()
    var events: [QueueLifecycleEvent] = []
    for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
      guard let data = line.data(using: .utf8),
            let event = try? decoder.decode(QueueLifecycleEvent.self, from: data)
      else { continue }
      events.append(event)
    }
    return events
  }
}
