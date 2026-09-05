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
// abandoned, interrupted, completed, failed, replayed after a restart, or
// dropped — before either issue's proposed *behavior* changes can be
// evaluated safely.
//
// This file changes NOTHING about queue behavior. Every call site that wires
// it (WarmServer.swift) is a read-only observation of a transition that
// already happens; recording an event never affects what the transition
// does or whether it succeeds.
//
// PR #370 review round 1 (two Criticals) reshaped this file's internals
// twice over from the original version:
//
//   C1. `record` used to hold ONE lock across both the ring mutation AND the
//       synchronous JSONL disk append — and the sync control plane
//       (`GET /v1/queue`, answered with zero cooperative threads so it stays
//       responsive during a render, #217's whole point) read that same lock.
//       A slow disk write (a near-full volume, a network mount) would have
//       stalled `/v1/queue` behind a render exactly the way #217 describes.
//       Fixed: `lock` now guards ONLY in-memory state (the ring, the
//       sequence counter, the progress throttle, a bounded pending-write
//       buffer). Disk I/O runs exclusively on `writerQueue`, a dedicated
//       serial background queue, never while `lock` is held. If the writer
//       ever falls behind (a stalled disk), the pending-write buffer is
//       BOUNDED (`maxPendingWrites`) and drops the oldest queued writes
//       (counted, not silent) rather than growing without bound or blocking
//       a caller.
//   C2. The JSONL file was unbounded, and a fresh ledger's `init` read the
//       ENTIRE file synchronously to reseed the ring/sequence counter — a
//       stored-property initializer on `WarmServer.init`, i.e. on the
//       engine's own startup path. Fixed: (a) the writer rotates the file at
//       `rotateAtBytes` (default 20 MB), keeping `keepGenerations` (default
//       2) generations — bounded disk footprint; (b) reseeding is LAZY (on
//       first real use — the first `record`/`events` call — never in
//       `init`) and reads only the TAIL (`reseedTailBytes`, default 64 KB)
//       with a corrupt-or-truncated-last-line-tolerant parser, not the
//       whole file. Engine startup itself (`QueueLifecycleLedger()`'s
//       construction) never touches disk.

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
  /// The checkpointed video render resumed from its own checkpoint.
  case resumed
  /// PR #370 review I3: the checkpointed video was NOT resumed — an operator
  /// interrupt arrived during the preemption episode
  /// (`LTX2PreemptionEpisode.Disposition.abandonVideo`), so the checkpoint
  /// was dropped instead. Distinct from `.resumed` (the prior version of
  /// this ledger recorded `.resumed` here too, which was simply wrong — the
  /// render never continues after this). `ReplayClassifier` treats this the
  /// same as `.resumed`/a terminal outcome: it closes out the checkpoint.
  case abandoned
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
///
/// `Codable` is hand-written (not synthesized) for exactly one reason (PR
/// #370 review I4): `wallTime` must be ISO8601 on EVERY wire surface this
/// type crosses, and this file alone has three different encoders in play —
/// the ledger's own JSONL encoder, the generic snake_case HTTP encoder
/// `/v1/queue/lifecycle` and `/v1/generate/status/{id}` share with every
/// other route in WarmServer.swift, and `/v1/queue`'s hand-built dictionary
/// encoder. A shared ambient `dateEncodingStrategy` cannot be trusted to
/// agree across all three (the generic HTTP encoder does not set one, so
/// Foundation's default — a raw `Double` — would have leaked through), and
/// changing that SHARED encoder's default would be an unrelated, unaudited
/// change to every other `Date` field on every other route. This type
/// formats its own `wallTime` field instead, independent of whichever
/// encoder/decoder a caller uses.
public struct QueueLifecycleEvent: Sendable, Equatable {
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
  /// `.checkpointed`/`.resumed`/`.abandoned`. Never used for control flow.
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

extension QueueLifecycleEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case sequence, bootId, wallTime, jobId, kind, jobKind, source, step, totalSteps,
         chunk, totalChunks, percent, originalJobId, fromStep1, reason, durationMs
  }

  /// The ONE format `wallTime` is ever written or read in, regardless of the
  /// ambient encoder/decoder's own date strategy (see the type's doc
  /// comment). `ISO8601DateFormatter` is thread-safe for concurrent
  /// read-only use once constructed (Apple docs), and this instance is never
  /// mutated after creation.
  private static let wallTimeFormatter = ISO8601DateFormatter()

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    sequence = try c.decode(UInt64.self, forKey: .sequence)
    bootId = try c.decode(String.self, forKey: .bootId)
    let wallTimeString = try c.decode(String.self, forKey: .wallTime)
    guard let parsed = Self.wallTimeFormatter.date(from: wallTimeString) else {
      throw DecodingError.dataCorruptedError(
        forKey: .wallTime, in: c, debugDescription: "wallTime is not a valid ISO8601 string: \(wallTimeString)")
    }
    wallTime = parsed
    jobId = try c.decode(String.self, forKey: .jobId)
    kind = try c.decode(QueueLifecycleEventKind.self, forKey: .kind)
    jobKind = try c.decodeIfPresent(String.self, forKey: .jobKind)
    source = try c.decodeIfPresent(String.self, forKey: .source)
    step = try c.decodeIfPresent(Int.self, forKey: .step)
    totalSteps = try c.decodeIfPresent(Int.self, forKey: .totalSteps)
    chunk = try c.decodeIfPresent(Int.self, forKey: .chunk)
    totalChunks = try c.decodeIfPresent(Int.self, forKey: .totalChunks)
    percent = try c.decodeIfPresent(Int.self, forKey: .percent)
    originalJobId = try c.decodeIfPresent(String.self, forKey: .originalJobId)
    fromStep1 = try c.decodeIfPresent(Bool.self, forKey: .fromStep1)
    reason = try c.decodeIfPresent(String.self, forKey: .reason)
    durationMs = try c.decodeIfPresent(Int.self, forKey: .durationMs)
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(sequence, forKey: .sequence)
    try c.encode(bootId, forKey: .bootId)
    try c.encode(Self.wallTimeFormatter.string(from: wallTime), forKey: .wallTime)
    try c.encode(jobId, forKey: .jobId)
    try c.encode(kind, forKey: .kind)
    try c.encodeIfPresent(jobKind, forKey: .jobKind)
    try c.encodeIfPresent(source, forKey: .source)
    try c.encodeIfPresent(step, forKey: .step)
    try c.encodeIfPresent(totalSteps, forKey: .totalSteps)
    try c.encodeIfPresent(chunk, forKey: .chunk)
    try c.encodeIfPresent(totalChunks, forKey: .totalChunks)
    try c.encodeIfPresent(percent, forKey: .percent)
    try c.encodeIfPresent(originalJobId, forKey: .originalJobId)
    try c.encodeIfPresent(fromStep1, forKey: .fromStep1)
    try c.encodeIfPresent(reason, forKey: .reason)
    try c.encodeIfPresent(durationMs, forKey: .durationMs)
  }
}

/// Pure classification: given a job's PRIOR recorded events (in emission
/// order), would replaying it now resume from a checkpoint or restart at
/// step 1? An "open" checkpoint is a `.checkpointed` event with no later
/// `.resumed`/`.abandoned` (or terminal: `.completed`/`.failed`/`.dropped`)
/// event closing it out — a checkpoint already resumed, abandoned, or
/// superseded by a terminal outcome has nothing left to resume FROM.
///
/// PR #370 review I3: `.abandoned` closes a checkpoint out exactly like
/// `.resumed` does — the checkpoint is gone either way, just for a different
/// reason (an operator interrupt during the preemption episode, not a
/// completed resume).
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
      case .resumed, .abandoned, .completed, .failed, .dropped:
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

/// Append-only lifecycle ledger. See the file doc comment for the C1/C2
/// redesign this type went through in PR #370 review round 1.
///
/// Concurrency model: `lock` (a plain `NSLock`) guards ONLY in-memory,
/// O(1)-ish state — the circular ring buffer, `lastEventByJobId`, the
/// sequence counter, the progress throttle, the bounded pending-write
/// buffer, and the reseed-once flag. It is NEVER held across file I/O.
/// `writerQueue`, a dedicated serial `DispatchQueue`, owns every disk
/// operation (batched appends, rotation, the lazy tail reseed) so a slow or
/// stalled disk can never stall a caller — including the sync control plane
/// (`GET /v1/queue`, `GET /v1/queue/lifecycle`), which reads this ledger
/// with zero cooperative threads specifically so it answers during a render.
public final class QueueLifecycleLedger: @unchecked Sendable {
  public static let defaultCapacity = 4000
  public static let defaultProgressMinInterval: TimeInterval = 1.0
  /// C2(a): rotate the JSONL file before it would exceed this size.
  public static let defaultRotateAtBytes = 20 * 1024 * 1024
  /// C2(a): how many generations to keep on disk (the live file plus this
  /// many rotated backups: `queue-lifecycle.jsonl`, `.jsonl.1`, …). Worst
  /// case on-disk footprint is therefore `defaultRotateAtBytes *
  /// defaultKeepGenerations` (~40 MB with the defaults) — see
  /// docs/api-notes.md for the growth math.
  public static let defaultKeepGenerations = 2
  /// C2(b): the lazy reseed reads only this many trailing bytes of the
  /// file, never the whole thing.
  public static let defaultReseedTailBytes = 64 * 1024
  /// C1: the bounded in-memory buffer of events waiting to be written to
  /// disk. If the writer ever falls behind this far (a stalled disk), the
  /// OLDEST queued writes are dropped (counted via `droppedWriteCount`,
  /// exposed for tests) rather than growing without bound or blocking
  /// `record`'s caller. The ring/`lastEventByJobId` are unaffected — a
  /// dropped write only means that one event never reaches
  /// `queue-lifecycle.jsonl`; it is still visible in-memory via
  /// `events`/`lastEvent` for the life of this process.
  public static let defaultMaxPendingWrites = 2000

  /// C1 test seam: the actual disk-writing step, injectable so a test can
  /// substitute a slow/blocking stub and prove `record` never waits on it.
  /// Receives one BATCH (the writer drains everything queued in one pass)
  /// plus enough context to rotate. Runs exclusively on `writerQueue`.
  public typealias BatchWriter = @Sendable (
    _ events: [QueueLifecycleEvent], _ url: URL, _ fileManager: FileManager,
    _ rotateAtBytes: Int, _ keepGenerations: Int
  ) -> Void

  public let bootId: String
  private let capacity: Int
  private let jsonlURL: URL?
  private let progressMinInterval: TimeInterval
  private let rotateAtBytes: Int
  private let keepGenerations: Int
  private let reseedTailBytes: Int
  private let maxPendingWrites: Int
  private let clock: @Sendable () -> Date
  private let fileManager: FileManager
  private let writer: BatchWriter

  /// C1: dedicated serial queue for every disk operation this ledger does —
  /// appends, rotation, and the lazy tail reseed. Never the caller's thread,
  /// never while `lock` is held.
  private let writerQueue = DispatchQueue(label: "com.comfybox.queue-lifecycle-ledger.writer")

  /// C1/M: the ONE lock in this class — see the type's doc comment.
  private let lock = NSLock()
  private var ring: [QueueLifecycleEvent?]
  /// Index of the OLDEST valid element in `ring` (circular buffer — M: no
  /// `removeFirst`, which is O(n), on every insert once the ring is full).
  private var ringHead = 0
  private var ringCount = 0
  /// M: `lastEvent(jobId:)` used to scan the whole ring; this is the O(1)
  /// index instead, pruned as entries age out of the ring (see `record`) so
  /// it stays roughly bounded by `capacity` rather than growing for the
  /// life of a long-running engine.
  private var lastEventByJobId: [String: QueueLifecycleEvent] = [:]
  private var nextSequence: UInt64 = 0
  private var lastProgressAt: [String: Date] = [:]
  private var didReseed = false
  private var pendingWrites: [QueueLifecycleEvent] = []
  private var writerScheduled = false
  /// C1: how many queued writes have been dropped because the writer fell
  /// behind `maxPendingWrites` — exposed for tests/diagnostics (see
  /// `droppedWriteCountForTesting`), never for control flow.
  private var droppedWriteCount = 0

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
    rotateAtBytes: Int = QueueLifecycleLedger.defaultRotateAtBytes,
    keepGenerations: Int = QueueLifecycleLedger.defaultKeepGenerations,
    reseedTailBytes: Int = QueueLifecycleLedger.defaultReseedTailBytes,
    maxPendingWrites: Int = QueueLifecycleLedger.defaultMaxPendingWrites,
    fileManager: FileManager = .default,
    clock: @escaping @Sendable () -> Date = Date.init,
    writer: @escaping BatchWriter = QueueLifecycleLedger.defaultBatchWriter
  ) {
    self.capacity = max(1, capacity)
    self.bootId = bootId
    self.jsonlURL = jsonlURL
    self.progressMinInterval = progressMinInterval
    self.rotateAtBytes = rotateAtBytes
    self.keepGenerations = keepGenerations
    self.reseedTailBytes = reseedTailBytes
    self.maxPendingWrites = maxPendingWrites
    self.fileManager = fileManager
    self.clock = clock
    self.writer = writer
    self.ring = Array(repeating: nil, count: self.capacity)
    // C2(b): deliberately NO file I/O here — see `ensureReseededIfNeeded`.
    // This initializer is a stored property initializer on `WarmServer`
    // (i.e. on the engine's own startup path) and must never touch disk.
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
    ensureReseededIfNeeded()

    var shouldScheduleWriter = false
    let event: QueueLifecycleEvent? = {
      lock.lock()
      defer { lock.unlock() }

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

      if let evicted = ringAppend(event) {
        // M: prune the O(1) index only if the evicted event was still the
        // most recent one on record for its job — a newer event for that
        // same job (still in the ring) must not be clobbered by this.
        if lastEventByJobId[evicted.jobId]?.sequence == evicted.sequence {
          lastEventByJobId.removeValue(forKey: evicted.jobId)
        }
      }
      lastEventByJobId[event.jobId] = event

      switch kind {
      case .completed, .failed, .interrupted, .abandoned, .dropped:
        // A terminal event closes out this job id's progress-throttle
        // window — an id reused later (should never happen; ids are UUIDs)
        // or simply garbage in the dictionary otherwise grows unbounded
        // across a long-running engine's lifetime.
        lastProgressAt.removeValue(forKey: jobId)
      default:
        break
      }

      // C1: queue the write, but NEVER do the write itself under `lock`.
      pendingWrites.append(event)
      if pendingWrites.count > maxPendingWrites {
        let overflow = pendingWrites.count - maxPendingWrites
        pendingWrites.removeFirst(overflow)
        droppedWriteCount += overflow
      }
      if !writerScheduled {
        writerScheduled = true
        shouldScheduleWriter = true
      }
      return event
    }()

    if shouldScheduleWriter {
      writerQueue.async { [weak self] in self?.drainPendingWrites() }
    }
    return event
  }

  /// M: clear this job id's progress-throttle entry directly. The one
  /// caller today (`runPreemptionEpisode`) needs it for a preempting image
  /// job: that job bypasses the normal enqueue/admit path entirely (a
  /// mailbox handoff, not the FIFO — see WarmServer.swift's "Known
  /// limitations" note), so it never reaches a terminal event through
  /// `record` and its throttle entry would otherwise never be cleared —
  /// one leaked dictionary entry per preemption for the life of the
  /// process. Safe to call for any job id, including one never throttled.
  public func clearProgressThrottle(jobId: String) {
    lock.lock()
    lastProgressAt.removeValue(forKey: jobId)
    lock.unlock()
  }

  /// Convenience: `ReplayClassifier.classify` over this job id's own prior
  /// history — the call `recoverPersistedQueue()` makes right before
  /// recording `.replayedAfterRestart`.
  public func classifyReplay(jobId: String) -> ReplayClassifier.Classification {
    ReplayClassifier.classify(priorEvents: events(jobId: jobId))
  }

  // MARK: - Reading

  /// Events in emission order, optionally filtered to one job id and/or
  /// capped to the last `limit`. No actor hop, no disk I/O in the common
  /// case (only the FIRST call across the process's life touches disk, and
  /// only a bounded tail read — see `ensureReseededIfNeeded`).
  public func events(jobId: String? = nil, limit: Int? = nil) -> [QueueLifecycleEvent] {
    ensureReseededIfNeeded()
    lock.lock()
    let all = jobId == nil ? ringSnapshot() : ringSnapshot().filter { $0.jobId == jobId }
    lock.unlock()
    if let limit, limit < all.count {
      return Array(all.suffix(limit))
    }
    return all
  }

  /// M: O(1) — an indexed lookup, not a ring scan.
  public func lastEvent(jobId: String) -> QueueLifecycleEvent? {
    ensureReseededIfNeeded()
    lock.lock()
    defer { lock.unlock() }
    return lastEventByJobId[jobId]
  }

  public func tail(jobId: String, count: Int) -> [QueueLifecycleEvent] {
    events(jobId: jobId, limit: count)
  }

  /// Test/diagnostic seam: how many queued writes have been dropped because
  /// the background writer fell behind `maxPendingWrites`.
  public var droppedWriteCountForTesting: Int {
    lock.lock()
    defer { lock.unlock() }
    return droppedWriteCount
  }

  /// C1 test seam: block until the background writer has drained everything
  /// queued as of the moment this is called. `record` deliberately never
  /// waits on disk I/O (that is the whole point of C1's fix), so a test that
  /// asserts on `queue-lifecycle.jsonl`'s CONTENT must call this first.
  /// `writerQueue` is serial: submitting an empty synchronous block runs
  /// only after every `.async` write task already enqueued ahead of it has
  /// finished, including one still mid-drain (`drainPendingWrites`'s `while`
  /// loop keeps draining anything that arrived while it ran, so a single
  /// call here is sufficient even if `record` was still being called
  /// concurrently up to the moment this is invoked). Never used in
  /// production code.
  public func waitForPendingWritesForTesting() {
    writerQueue.sync {}
  }

  // MARK: - Circular ring buffer (M: no `removeFirst`)

  /// Appends `event`, evicting and returning the oldest element if the ring
  /// was already full. O(1) — no shifting. MUST be called with `lock` held.
  private func ringAppend(_ event: QueueLifecycleEvent) -> QueueLifecycleEvent? {
    if ringCount < capacity {
      let index = (ringHead + ringCount) % capacity
      ring[index] = event
      ringCount += 1
      return nil
    } else {
      let evicted = ring[ringHead]
      ring[ringHead] = event
      ringHead = (ringHead + 1) % capacity
      return evicted
    }
  }

  /// Oldest-to-newest snapshot of the ring's current contents. MUST be
  /// called with `lock` held.
  private func ringSnapshot() -> [QueueLifecycleEvent] {
    guard ringCount > 0 else { return [] }
    var result: [QueueLifecycleEvent] = []
    result.reserveCapacity(ringCount)
    for offset in 0..<ringCount {
      if let event = ring[(ringHead + offset) % capacity] {
        result.append(event)
      }
    }
    return result
  }

  /// Replaces the ring's entire contents (oldest-to-newest) and rebuilds
  /// `lastEventByJobId` from it — used only by the lazy reseed. MUST be
  /// called with `lock` held.
  private func rebuildRing(from events: [QueueLifecycleEvent]) {
    let trimmed = Array(events.suffix(capacity))
    ring = Array(repeating: nil, count: capacity)
    ringHead = 0
    ringCount = 0
    lastEventByJobId.removeAll()
    for event in trimmed {
      _ = ringAppend(event)
      lastEventByJobId[event.jobId] = event
    }
  }

  // MARK: - C1: background writer

  /// Drains `pendingWrites` to disk in one (or more, if more arrived while
  /// writing) batches. Runs exclusively on `writerQueue`; never touches
  /// `lock` while doing file I/O.
  private func drainPendingWrites() {
    while true {
      let batch: [QueueLifecycleEvent]
      lock.lock()
      if pendingWrites.isEmpty {
        writerScheduled = false
        lock.unlock()
        return
      }
      batch = pendingWrites
      pendingWrites = []
      lock.unlock()

      guard let url = jsonlURL else { continue }
      writer(batch, url, fileManager, rotateAtBytes, keepGenerations)
    }
  }

  /// The real disk-writing step (default `writer`). Batches every queued
  /// event into ONE write, rotates first if that write would push the file
  /// past `rotateAtBytes` (C2a), and — the bug this replaces — treats "the
  /// file could not be opened for writing" as ALWAYS meaning "create it"
  /// (M: on EMFILE, or any other transient open failure, the old code's
  /// `else` branch fell through to an ATOMIC OVERWRITE of a file that DID
  /// exist, destroying every prior line). Now: create only when the file is
  /// genuinely absent; otherwise, if it exists but cannot be opened, drop
  /// this batch rather than clobber history.
  public static let defaultBatchWriter: BatchWriter = { events, url, fileManager, rotateAtBytes, keepGenerations in
    guard !events.isEmpty else { return }
    let encoder = makeEncoder()
    var payload = Data()
    for event in events {
      guard let line = try? encoder.encode(event) else { continue }
      payload.append(line)
      payload.append(0x0A)  // '\n'
    }
    guard !payload.isEmpty else { return }

    let dir = url.deletingLastPathComponent()
    try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    rotateIfNeeded(
      url: url, incomingBytes: payload.count, rotateAtBytes: rotateAtBytes,
      keepGenerations: keepGenerations, fileManager: fileManager)

    if fileManager.fileExists(atPath: url.path) {
      if let handle = try? FileHandle(forWritingTo: url) {
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: payload)
      }
      // else: could not open an EXISTING file for writing (EMFILE,
      // permissions, …) — drop this batch. Falling back to
      // `Data.write(options: .atomic)` here would silently replace the
      // whole file with just this batch, which is strictly worse than
      // losing one batch.
    } else {
      try? payload.write(to: url, options: .atomic)
    }
  }

  /// C2(a): rotate `url` -> `url.1` -> `url.2` … (dropping the oldest once
  /// `keepGenerations` is reached) if appending `incomingBytes` would push
  /// the current file past `rotateAtBytes`. A brand-new (zero-byte or
  /// missing) file is never rotated — there is nothing to preserve.
  static func rotateIfNeeded(
    url: URL, incomingBytes: Int, rotateAtBytes: Int, keepGenerations: Int, fileManager: FileManager
  ) {
    guard keepGenerations >= 1, rotateAtBytes > 0 else { return }
    let currentSize = (try? fileManager.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
    guard currentSize > 0, currentSize + incomingBytes > rotateAtBytes else { return }

    if keepGenerations > 1 {
      // Shift older generations up one slot, oldest falls off the end.
      let oldest = url.appendingPathExtension("\(keepGenerations - 1)")
      try? fileManager.removeItem(at: oldest)
      var generation = keepGenerations - 1
      while generation > 1 {
        let from = url.appendingPathExtension("\(generation - 1)")
        let to = url.appendingPathExtension("\(generation)")
        if fileManager.fileExists(atPath: from.path) {
          try? fileManager.removeItem(at: to)
          try? fileManager.moveItem(at: from, to: to)
        }
        generation -= 1
      }
      let rotated = url.appendingPathExtension("1")
      try? fileManager.removeItem(at: rotated)
      try? fileManager.moveItem(at: url, to: rotated)
    } else {
      try? fileManager.removeItem(at: url)
    }
  }

  // MARK: - C2(b): lazy, bounded reseed

  /// Runs exactly once per ledger instance, on the FIRST real call to
  /// `record`/`events` (never in `init` — see the file doc comment). Reads
  /// only the file's TAIL (`reseedTailBytes`), never the whole file, so
  /// even the one-time cost is bounded regardless of how large the file has
  /// grown before rotation catches up.
  private func ensureReseededIfNeeded() {
    lock.lock()
    if didReseed {
      lock.unlock()
      return
    }
    didReseed = true
    lock.unlock()

    guard let url = jsonlURL else { return }
    let tail = Self.loadJSONLTail(from: url, maxBytes: reseedTailBytes, fileManager: fileManager)
    guard !tail.isEmpty else { return }

    lock.lock()
    let liveSoFar = ringSnapshot()  // rare: only non-empty if record() raced ahead of this reseed
    rebuildRing(from: tail + liveSoFar)
    let tailMax = tail.map { $0.sequence }.max() ?? 0
    nextSequence = max(nextSequence, tailMax + 1)
    lock.unlock()
  }

  // MARK: - JSONL persistence (shared encode/decode config)

  private static func makeEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]  // single line, deterministic — no .prettyPrinted
    return encoder
  }

  private static func makeDecoder() -> JSONDecoder {
    JSONDecoder()
  }

  /// Read back every well-formed line of a lifecycle JSONL file, in file
  /// order (append-only, so file order is emission order across every boot
  /// that ever wrote to it). A malformed trailing line (a write that raced
  /// a crash) is skipped rather than failing the whole read. Reads the
  /// WHOLE file — used by tests and any future offline tool, never by the
  /// ledger's own reseed (see `loadJSONLTail`).
  public static func loadJSONL(from url: URL, fileManager: FileManager = .default) -> [QueueLifecycleEvent] {
    guard let data = try? Data(contentsOf: url) else { return [] }
    return parseLines(data, fileManager: fileManager)
  }

  /// C2(b): read only the last `maxBytes` of `url` and parse whatever
  /// complete lines that window contains. Tolerant of BOTH a truncated
  /// trailing line (a write that raced a crash) and a truncated LEADING
  /// line (this window started mid-file, not at a line boundary) — the
  /// leading partial line is discarded rather than mis-parsed.
  public static func loadJSONLTail(from url: URL, maxBytes: Int, fileManager: FileManager = .default) -> [QueueLifecycleEvent] {
    guard fileManager.fileExists(atPath: url.path), let handle = try? FileHandle(forReadingFrom: url) else {
      return []
    }
    defer { try? handle.close() }
    let size = (try? handle.seekToEnd()) ?? 0
    let startedMidFile = size > UInt64(maxBytes)
    let start = startedMidFile ? size - UInt64(maxBytes) : 0
    guard (try? handle.seek(toOffset: start)) != nil else { return [] }
    guard let data = try? handle.readToEnd() else { return [] }
    guard startedMidFile else { return parseLines(data, fileManager: fileManager) }

    // Drop everything before the first newline — it is a partial line from
    // wherever this window happened to start.
    guard let newlineIndex = data.firstIndex(of: 0x0A) else { return [] }
    let remainder = data[data.index(after: newlineIndex)...]
    return parseLines(Data(remainder), fileManager: fileManager)
  }

  private static func parseLines(_ data: Data, fileManager: FileManager) -> [QueueLifecycleEvent] {
    guard let text = String(data: data, encoding: .utf8) else { return [] }
    let decoder = makeDecoder()
    var events: [QueueLifecycleEvent] = []
    for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
      guard let lineData = line.data(using: .utf8),
            let event = try? decoder.decode(QueueLifecycleEvent.self, from: lineData)
      else { continue }
      events.append(event)
    }
    return events
  }
}
