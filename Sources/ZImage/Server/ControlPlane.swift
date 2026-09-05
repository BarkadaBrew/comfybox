// ControlPlane.swift — 0.B-2 control-plane carve-out
// (FDD-ui-api-parity.md §3.1.4, §3.1.4a, §3.1.5; comfybox#300).
//
// 0.A (coordinator serial executor) + 0.B-1 (render task-executor preference)
// aim to keep the cooperative pool clear during a render. 0.B-2 is the
// GUARANTEE layer: the sync-servable control set is served on the connection's
// OWN DispatchQueue, synchronously, BEFORE any `Task {}` — so those routes need
// zero cooperative threads and answer even if some future dependency exhausts
// the pool again (it has been exhausted twice already, on two QoS tiers).
//
// Three pieces live here; the actor/lock-store wiring that consumes them lives
// in WarmServer.swift:
//   1. the env flag (rollback lever),
//   2. the queue-mutation DELTA type + its crash-survival sidecar,
//   3. the route classifier and the pure delta-application rule shared by the
//      read-compose path and the recovery replay.

import Foundation

// MARK: - Env flag (rollback lever)

/// `COMFYBOX_CONTROL_PLANE_SYNC`, default ON. Setting it to "0" reverts EXACTLY
/// to the pre-0.B-2 dispatch path: the sync classifier is never consulted and
/// every one of these routes flows through the full async `Task { await
/// respond(...) }` path byte-for-byte, hitting the same actor arms as before.
/// This is a real code path, not a flag that is read and ignored — see
/// `ConnectionHandler.handle()`.
enum ControlPlaneSyncFlag {
  static var isEnabled: Bool {
    ProcessInfo.processInfo.environment["COMFYBOX_CONTROL_PLANE_SYNC"] != "0"
  }
}

// MARK: - Queue-mutation deltas

/// A queue mutation recorded as a DELTA against whatever `pending` actually is
/// at drain time — never a snapshot/mirror of `pending` (which has other
/// writers: `enqueue`, `recoverPersistedQueue`; a wholesale mirror drops jobs,
/// FDD §0 row 6 / §3.1.4a). Off-actor/sync `cancel` and `move` cannot touch the
/// actor's `pending` array, so they record one of these; the actor applies them
/// at its scheduling points.
struct QueueControlCommand: Sendable, Equatable {
  enum Kind: Sendable, Equatable {
    case cancel(id: String)
    case move(id: String, direction: String)
  }

  let kind: Kind

  /// Structural guard against the F1 wedge (FDD §0 row 2, §3.1.4a): a command
  /// that PARKS the loop or changes pause state but needs the loop to WAKE
  /// cannot be a passive mailbox delta — it would 202 and wedge the queue
  /// forever, because the mailbox is drained BY the loop. Both delta kinds here
  /// mutate `pending` only and never need a wake, so this is always false. The
  /// memberwise initializer is PRIVATE and the only constructors are the two
  /// factories below, so a `requiresWake: true` command is impossible to build
  /// and put in the mailbox — that is the compile-time half of the guard. The
  /// drain additionally asserts `!requiresWake` (WarmServer.drainQueueDeltas),
  /// so if a future factory ever sets it true the drain fails in test rather
  /// than shipping the wedge. `resume` is the one command that requires a wake,
  /// and it is served fire-and-forget, never through this mailbox.
  let requiresWake: Bool

  private init(kind: Kind, requiresWake: Bool) {
    self.kind = kind
    self.requiresWake = requiresWake
  }

  static func cancel(_ id: String) -> QueueControlCommand {
    QueueControlCommand(kind: .cancel(id: id), requiresWake: false)
  }

  static func move(_ id: String, direction: String) -> QueueControlCommand {
    QueueControlCommand(kind: .move(id: id, direction: direction), requiresWake: false)
  }

  var targetId: String {
    switch kind {
    case .cancel(let id): return id
    case .move(let id, _): return id
    }
  }
}

// MARK: - Undrained-delta sidecar (queue-deltas.json)

/// On-disk form of an undrained delta. Separate from `PersistedQueueJob`
/// (`QueuePersistence.swift`) because the two are distinct concerns:
/// `persistQueueState()` is the actor's CANONICAL queue; this sidecar is
/// specifically the not-yet-drained deltas an off-actor/sync cancel or move
/// recorded, which the actor has not yet folded into that canonical queue
/// (FDD §3.1.4a point 4).
struct PersistedQueueDelta: Codable, Sendable {
  enum Op: String, Codable { case cancel, move }
  let op: Op
  let id: String
  let direction: String?
}

extension QueueControlCommand {
  var persisted: PersistedQueueDelta {
    switch kind {
    case .cancel(let id):
      return PersistedQueueDelta(op: .cancel, id: id, direction: nil)
    case .move(let id, let direction):
      return PersistedQueueDelta(op: .move, id: id, direction: direction)
    }
  }

  init(_ persisted: PersistedQueueDelta) {
    switch persisted.op {
    case .cancel:
      self = .cancel(persisted.id)
    case .move:
      self = .move(persisted.id, direction: persisted.direction ?? "up")
    }
  }
}

/// Reads/writes the undrained-delta sidecar at `<state>/queue-deltas.json`.
/// Same atomic temp+rename idiom and `COMFYBOX_STATE_DIR` honouring as
/// `QueueStateStore`, so a crash between recording a delta and the actor
/// draining it does not lose the delta: `recoverPersistedQueue` folds this file
/// into the recovered queue on the next boot, which is what makes
/// "cancel → bounce → stays cancelled" hold.
enum QueueDeltaStore {
  #if DEBUG
  /// F-2 test seam: the drain fires this between the canonical
  /// `persistQueueState()` write and the sidecar commit — the crash window the
  /// WAL ordering protects. Tests assert queue-deltas.json still exists here.
  nonisolated(unsafe) static var drainCrashWindowHook: (@Sendable () -> Void)?
  #endif

  static var path: URL {
    QueueStateStore.stateDirectory.appendingPathComponent("queue-deltas.json")
  }

  /// Persist the WHOLE undrained set (small JSON, atomic). An empty set deletes
  /// the file, exactly like `QueueStateStore.save` — no stale sidecar survives a
  /// clean drain.
  static func save(_ deltas: [QueueControlCommand]) {
    guard !deltas.isEmpty else {
      try? FileManager.default.removeItem(at: path)
      return
    }
    let encoder = JSONEncoder()
    guard let data = try? encoder.encode(deltas.map { $0.persisted }) else { return }
    try? data.write(to: path, options: .atomic)
  }

  static func load() -> [QueueControlCommand] {
    guard let data = try? Data(contentsOf: path) else { return [] }
    let decoder = JSONDecoder()
    guard let raw = try? decoder.decode([PersistedQueueDelta].self, from: data) else { return [] }
    return raw.map(QueueControlCommand.init)
  }

  static func clear() {
    try? FileManager.default.removeItem(at: path)
  }
}

// MARK: - Route classifier

/// The SYNC-SERVABLE control set (FDD §3.1.4). `ConnectionHandler.handle()`
/// consults this BEFORE entering its `Task {}`; a match is served synchronously
/// on the connection's own queue. The set EXCLUDES `/v1/characters*`
/// (actor-backed `CharacterStore`, cannot be read synchronously) and the
/// genuinely-async internals (civitai search/harvest, enhance, workflow runs —
/// 0.B-1's job; a synchronous classifier cannot serve network I/O and does not
/// try).
enum ControlPlaneClassifier {
  static func isSyncServable(method: String, path: String) -> Bool {
    switch (method, path) {
    case ("GET", "/v1/queue"),
         // comfybox#283/#217 review I5: must answer during a render exactly
         // like `/v1/queue` — the ledger it reads is lock-based with no
         // actor hop and no disk I/O under its lock (see
         // QueueLifecycleLedger.swift's C1 fix), so it is safe to serve here.
         ("GET", "/v1/queue/lifecycle"),
         ("GET", "/v1/models"),
         ("GET", "/v1/stats"),
         ("GET", "/v1/config"),
         ("GET", "/v1/controls"),
         ("POST", "/v1/queue/pause"),
         ("POST", "/v1/queue/resume"),
         ("POST", "/v1/queue/clear"),
         ("POST", "/v1/queue/interrupt"):
      return true
    case ("POST", _) where path.hasPrefix("/v1/queue/") && path.hasSuffix("/move"):
      return true
    case ("DELETE", _) where path.hasPrefix("/v1/queue/"):
      return true
    default:
      return false
    }
  }
}

// MARK: - Delta application (shared by read-compose and recovery replay)

/// The one rule for applying deltas to an ordered, id-bearing list — used by
/// BOTH the read-compose path (`GET /v1/queue`, the sync cancel/move
/// present-check) over `QueueJobInfo`, AND the recovery replay
/// (`recoverPersistedQueue`) over `PersistedQueueJob`. Keeping it a single pure
/// function guarantees the listing a caller sees and the queue that actually
/// replays after a bounce cannot disagree.
enum QueueDeltaApplier {
  /// Apply `deltas` to `items` IN ORDER: a `cancel` drops the matching item; a
  /// `move` repositions it (same top/bottom/up/down semantics as
  /// `movePending`). A delta whose id is not present is a no-op — a cancel of an
  /// id that is already running, finished, or already dropped simply matches
  /// nothing, exactly like `cancelPending` returning false.
  static func apply<T>(_ deltas: [QueueControlCommand], to items: [T], id: (T) -> String) -> [T] {
    guard !deltas.isEmpty else { return items }
    var list = items
    for delta in deltas {
      switch delta.kind {
      case .cancel(let cancelId):
        list.removeAll { id($0) == cancelId }
      case .move(let moveId, let direction):
        guard let index = list.firstIndex(where: { id($0) == moveId }) else { continue }
        let item = list.remove(at: index)
        let target: Int
        switch direction {
        case "top": target = 0
        case "bottom": target = list.count
        case "up": target = max(0, index - 1)
        case "down": target = min(list.count, index + 1)
        default:
          list.insert(item, at: index)
          continue
        }
        list.insert(item, at: target)
      }
    }
    return list
  }
}

// MARK: - F-3: sync-cancel ACK (adversarial review)

/// The sync `DELETE /v1/queue/{id}` ACK — 202, record-then-accept. It never
/// carries `deleted: true`: the sync path records a delta and cannot know
/// whether the loop dequeued the job in the race window, so it must not claim
/// a deletion it cannot guarantee. (The flag-off async arm still returns
/// `deleted: true` — it removes the job on the actor before responding.)
struct SyncCancelAccepted: Codable, Sendable {
  let accepted: Bool
  let id: String
  let note: String

  static func ack(id: String) -> SyncCancelAccepted {
    SyncCancelAccepted(
      accepted: true, id: id,
      note: "cancel recorded; job may start if dequeue raced — poll status")
  }
}

// MARK: - comfybox#362: /v1/queue/interrupt targeting
//
// During a #1479 preemption episode, `/health` and `/v1/queue` show the
// preempting IMAGE job as active (the identity swap in `runPreemptionEpisode`
// predates this ticket), but `/v1/queue/interrupt` had no `target` concept —
// it always cancelled whichever `Task` happened to be published as "the
// active render", which before this ticket's task-publishing fix was always
// the checkpointed VIDEO's own outer task, no matter what health showed.
// Interrupting during an episode therefore silently abandoned the video while
// the visibly-active image render kept going.
//
// The fix has two parts: (1) `runAsPublishedActiveRender` (WarmServer.swift)
// republishes the PREEMPTOR's own task as "active" for the episode's
// duration, so the default interrupt target agrees with health; (2) `target`
// becomes an explicit, additive request field so an operator (or the
// checkpoint-fallback tooling) can still reach the video specifically. This
// section is the PURE decision the two `/v1/queue/interrupt` implementations
// (`LiveHealthState.cancelActiveRender`, the sync no-actor-hop path, and
// `WarmServerCoordinator.cancelActiveRender`, the async fallback used when
// `ControlPlaneSyncFlag` is off) both consult, so they cannot disagree.

/// What `/v1/queue/interrupt` should stop, resolved from the request's
/// `target` against what is actually running. Pure — no `Task` in sight — so
/// it is unit-testable without a render anywhere near it.
enum InterruptTargetResolution: Equatable {
  /// Whatever `/health` currently reports as the active render — the
  /// preempting image job during a preemption episode, the video otherwise.
  case active
  /// The checkpointed/running video specifically, even while an episode has
  /// swapped `active` to the preempting image job.
  case video
  /// `target` named a job id that is neither the active job nor the
  /// checkpointed video. Distinct from "nothing to cancel": a client that
  /// named a real (but wrong, or no-longer-running) id gets a 404, not a
  /// silent `interrupted: false` — pending jobs are cancelled via
  /// `DELETE /v1/queue/{id}`, not this route, so a pending job's id lands
  /// here too.
  case unknownJobId
}

enum InterruptTarget {
  /// `target` is the request body's raw (optional) field, verbatim. `nil` and
  /// the literal `"active"` both mean "whatever health shows as active";
  /// `"video"` is reserved for the checkpointed video; any other string is
  /// treated as a job id and matched against what is actually running.
  static func resolve(
    target: String?, activeJobId: String?, checkpointedVideoJobId: String?
  ) -> InterruptTargetResolution {
    switch target {
    case nil, "active":
      return .active
    case "video":
      return .video
    case let jobId?:
      if let activeJobId, jobId == activeJobId { return .active }
      if let checkpointedVideoJobId, jobId == checkpointedVideoJobId { return .video }
      return .unknownJobId
    }
  }
}

/// The result of actually cancelling (or not) per an `InterruptTargetResolution`
/// — the shared shape both `/v1/queue/interrupt` implementations return, so
/// their JSON responses cannot drift apart.
enum InterruptCancelOutcome: Equatable {
  /// Something was cancelled. `kind` is a `QueueJobKind` raw value.
  case cancelled(jobId: String?, kind: String?)
  /// The resolved target (default "active", or an explicit "video") named a
  /// real category but nothing is currently running there — not an error,
  /// same as the pre-#362 `interrupted: false`.
  case nothingToCancel
  /// `target` was a job id that matched neither the active job nor the
  /// checkpointed video — a 404, not a silent false.
  case unknownTarget
}

/// `/v1/queue/interrupt` request body. Entirely optional — an absent body (the
/// pre-#362 shape) or an absent/`null` `target` both mean `target: "active"`.
struct InterruptRequestBody: Decodable {
  let target: String?
}

/// `/v1/queue/interrupt` response. `interrupted_job_id`/`interrupted_kind` are
/// additive: present only when something was actually cancelled, so a client
/// that only reads `success`/`interrupted` sees no change from before #362.
struct InterruptResponseBody: Encodable {
  let success: Bool
  let interrupted: Bool
  let interruptedJobId: String?
  let interruptedKind: String?
}

/// Maps an `InterruptCancelOutcome` to the HTTP status + body both
/// `/v1/queue/interrupt` implementations serve, so they cannot drift apart.
enum InterruptRouteResponse {
  static func build(from outcome: InterruptCancelOutcome) -> (status: Int, body: InterruptResponseBody) {
    switch outcome {
    case .cancelled(let jobId, let kind):
      return (200, InterruptResponseBody(success: true, interrupted: true, interruptedJobId: jobId, interruptedKind: kind))
    case .nothingToCancel:
      return (200, InterruptResponseBody(success: true, interrupted: false, interruptedJobId: nil, interruptedKind: nil))
    case .unknownTarget:
      return (404, InterruptResponseBody(success: false, interrupted: false, interruptedJobId: nil, interruptedKind: nil))
    }
  }
}
