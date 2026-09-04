// QueueRecoveryGate.swift — #339: never silently drop a submission that
// arrives while the post-restart persisted-queue replay is still running.
//
// Two distinct defects, fixed together (review r1 of PR #349):
//
// 1) DURABILITY (the deeper bug Codex found): `recoverPersistedQueue()`
//    (WarmServer.swift) replays a persisted backlog one job at a time,
//    `await`-ing each job's full completion before admitting the next into
//    the coordinator's `pending`. Each admission's own `persistQueueState()`
//    call only ever reflects what is CURRENTLY in `pending`/`active` — so
//    while job 1 is admitted, jobs 2..N exist ONLY in the replay loop's local
//    variable, invisible to `queue-state.json`. A SECOND restart during that
//    window (a flaky post-deploy health check, an OOM kill, another SIGTERM)
//    loses every job after the first one, permanently, with no trace.
//    `RecoverySnapshotMerger` fixes this: on every persist while a replay is
//    in flight, the not-yet-admitted tail is merged into the snapshot, so
//    the file always reflects the FULL truth.
//
// 2) NON-RECOVERABLE KINDS: `.localVideo`/`.controlGenerate`/`.modelSwitch`/
//    a detached `.modelOperation` load never carry a `rawBody`
//    (`WarmServerCoordinator.PendingJob`), so `persistQueueState()`
//    unconditionally excludes them from `queue-state.json` no matter how (1)
//    is fixed — they cannot be durably queued behind a recovered backlog at
//    all. A submission of one of these kinds while a replay is in flight is
//    refused up front with an explicit, retryable 503 instead of a 202 that
//    cannot be trusted.

import Foundation

// MARK: - Canonical kind vocabulary (review r1, item 2)

/// The single source of truth for every string `Self.kind(of:)`
/// (WarmServer.swift) and `ModelOperation.kind` (ModelPool.swift) hand back
/// on `/v1/queue`/`/health`'s `kind` field and in `queue-state.json`'s
/// `PersistedQueueJob.kind`. `QueueRecoveryGate` reads its allowlist from
/// THIS enum, not a hand-maintained string literal, so it cannot silently
/// drift from the real wire vocabulary — the original version of this gate
/// used `"local_video"`/`"control_generate"`, which never matched the actual
/// `"video"`/`"controlnet"` strings the queue emits, so the gate never
/// fired.
enum QueueJobKind: String, CaseIterable, Sendable {
  case generate
  case controlnet
  case loraSwap = "lora_swap"
  case modelSwitch = "model_switch"
  case modelLoad = "model_load"
  case modelActivate = "model_activate"
  case modelUnload = "model_unload"
  case video
  case shutdown
  #if DEBUG
  case synthetic
  #endif
}

// MARK: - 1) Recovery-window durability

/// Pure merge: what `persistQueueState()` should write to disk while a
/// persisted-queue replay is in flight. `admittedActive`/`admittedPending`
/// are exactly what the coordinator's own bookkeeping already produces
/// (unchanged); `unadmittedTail` is everything the replay loop has not yet
/// even attempted to re-enqueue.
///
/// The tail is placed AHEAD of `admittedPending` so a job that arrived live
/// during the replay window (already sitting in `admittedPending`) does not
/// jump the persisted ordering ahead of older, pre-restart work — if the
/// process restarts again, the tail is replayed to completion before that
/// live job's own turn, preserving the FIFO invariant recovery promises.
///
/// Deduplicated by id, first occurrence wins — `admittedActive` outranks
/// `unadmittedTail` outranks `admittedPending` (review r3, item 1a). A job
/// can legitimately be represented in more than one of these three inputs
/// for a brief instant around its own admission (`WarmServer.recoverPersistedQueue`
/// narrows the tail as SOON as a job is observably admitted, not when its
/// render finishes — but "as soon as" is still a real, if short, window);
/// without this dedup, the OUTPUT snapshot would carry the same id twice,
/// and a subsequent restart's `jobs = active + pending` concatenation
/// (also deduplicated, same rule, in `recoverPersistedQueue`) would replay
/// it twice — and every FURTHER crash during that replay would add another
/// copy, compounding without bound.
enum RecoverySnapshotMerger {
  static func merge(
    admittedActive: PersistedQueueJob?,
    admittedPending: [PersistedQueueJob],
    unadmittedTail: [PersistedQueueJob]
  ) -> PersistedQueueState {
    var seenIds: Set<String> = []
    if let activeId = admittedActive?.id { seenIds.insert(activeId) }
    var pending: [PersistedQueueJob] = []
    for candidate in unadmittedTail + admittedPending {
      guard !seenIds.contains(candidate.id) else { continue }
      seenIds.insert(candidate.id)
      pending.append(candidate)
    }
    return PersistedQueueState(active: admittedActive, pending: pending)
  }

  /// Dedupe a flat job list by id, first occurrence wins. Used by
  /// `recoverPersistedQueue` on the `active + pending` concatenation it
  /// loads from disk (review r3, item 1a) — the SAME rule `merge` above
  /// applies to its own output, so a snapshot that (however briefly) carried
  /// a duplicate never gets replayed twice on the NEXT restart.
  static func deduplicated(_ jobs: [PersistedQueueJob]) -> [PersistedQueueJob] {
    var seenIds: Set<String> = []
    var result: [PersistedQueueJob] = []
    for job in jobs {
      guard !seenIds.contains(job.id) else { continue }
      seenIds.insert(job.id)
      result.append(job)
    }
    return result
  }
}

// MARK: - 2) Cross-task recovery status

/// Cross-task recovery status: whether `recoverPersistedQueue()`'s replay is
/// in flight, and the KINDS of recovered jobs still waiting to be admitted —
/// gives a refused submission an actual, kind-weighted estimate
/// (`retry_after_seconds`, review r2 item 6) instead of a flat guess.
/// Deliberately not actor-isolated: set from the synchronous startup path,
/// read from arbitrary HTTP-handling Tasks, neither of which holds the
/// `WarmServerCoordinator` actor.
final class QueueRecoveryState: @unchecked Sendable {
  private let lock = NSLock()
  private var inProgress = false
  /// FIFO — `jobAdmitted()` pops the front, matching `recoverPersistedQueue`'s
  /// replay order. In current practice this only ever holds `.generate`/
  /// `.loraSwap` (the only kinds `queue-state.json` can persist), but is kept
  /// kind-general rather than a bare count so the estimate stays correct if
  /// that ever changes.
  private var remainingKinds: [QueueJobKind] = []

  /// Called once, right before the replay loop starts.
  func begin(jobKinds: [QueueJobKind]) {
    lock.lock()
    inProgress = true
    remainingKinds = jobKinds
    lock.unlock()
  }

  /// Called after each recovered job is admitted (successfully or not) —
  /// pops the job the replay loop just attempted. Safe to call more times
  /// than `begin` had kinds (a no-op past empty).
  func jobAdmitted() {
    lock.lock()
    if !remainingKinds.isEmpty { remainingKinds.removeFirst() }
    lock.unlock()
  }

  /// Called once, when the replay loop finishes (`defer`, so it always runs
  /// — including on cancellation, review r2 item 5).
  func finish() {
    lock.lock()
    inProgress = false
    remainingKinds = []
    lock.unlock()
  }

  func snapshot() -> (inProgress: Bool, remaining: Int, remainingKinds: [QueueJobKind]) {
    lock.lock()
    defer { lock.unlock() }
    return (inProgress, remainingKinds.count, remainingKinds)
  }
}

// MARK: - 3) The gate itself

/// Pure decision logic: which queue-job kinds cannot survive a second
/// restart at all (never persisted, regardless of the durability fix above —
/// see the file doc comment), and whether a submission of one should be
/// refused while a persisted-queue replay is in flight. Kept free of any
/// WarmServer/actor dependency so it is testable without a running server.
enum QueueRecoveryGate {
  /// Kinds whose `PendingJob.rawBody` is always `nil`, so `persistQueueState()`
  /// (WarmServer.swift) can never persist them — no fix to the durability
  /// merge above changes that, because there is nothing to merge in.
  /// `.generate`/`.loraSwap` are absent on purpose: those two ARE persisted
  /// and (with `RecoverySnapshotMerger`) survive any number of restarts.
  /// `.modelActivate`/`.modelUnload` are absent too: those routes are
  /// synchronous (`wait: true`-equivalent) — the caller holds the connection
  /// open, so a process death is a visible connection failure, not a silent
  /// lost job id. `.modelLoad` covers only `/v1/model/load`'s `wait: false`
  /// arm, which — like local video — hands back a 202 + job id nobody is
  /// waiting on.
  static let nonRecoverableKinds: Set<QueueJobKind> = [.video, .controlnet, .modelSwitch, .modelLoad]

  static let errorCode = "queue_recovery_in_progress"

  /// Names exactly why, so a client's retry log (or Todd's) isn't a guess.
  /// Deliberately makes NO promise about how long — review r1 flagged the
  /// original "retry in a few seconds" as a promise a real backlog (minutes
  /// to hours of video renders) cannot keep. `retryAfterSeconds` below is
  /// the actual, if rough, estimate; this string only names the cause.
  static let reason =
    "Engine is replaying the persisted queue after a restart; this job kind "
    + "cannot be safely queued until that finishes (it would be lost if the "
    + "engine restarts again before it completes). See retry_after_seconds."

  /// `true` when this submission must be refused rather than accepted.
  static func shouldReject(kind: QueueJobKind, recoveryInProgress: Bool) -> Bool {
    recoveryInProgress && nonRecoverableKinds.contains(kind)
  }

  /// A rough per-job ballpark (seconds), used only to give a retrying caller
  /// a SANE floor — never a promise. Review r2 item 6: a flat 20s badly
  /// underestimates a video render (routinely minutes), so this is now
  /// per-kind. Recovery itself only ever replays `.generate`/`.loraSwap`
  /// (the only two `queue-state.json` can persist — see the file doc
  /// comment), so `.video`'s 600s is dormant infrastructure today, kept for
  /// correctness if that ever changes rather than hard-coding an assumption
  /// that won't age well.
  static func perJobEstimateSeconds(for kind: QueueJobKind) -> Int {
    switch kind {
    case .video: return 600
    default: return 20
    }
  }

  /// Never advise a retry sooner than this — a restart's own settle time.
  static let minimumRetryAfterSeconds = 5

  /// The estimate a route actually uses: sum of each remaining recovered
  /// job's own per-kind estimate.
  static func retryAfterSeconds(remainingKinds: [QueueJobKind]) -> Int {
    max(minimumRetryAfterSeconds, remainingKinds.reduce(0) { $0 + perJobEstimateSeconds(for: $1) })
  }

  /// Count-only convenience for a call site that has no kind breakdown —
  /// uses the `.generate` (image) estimate, which is what recovery's
  /// remaining-job list is ALWAYS made of in current practice (see above).
  static func retryAfterSeconds(remainingJobs: Int) -> Int {
    max(minimumRetryAfterSeconds, remainingJobs * perJobEstimateSeconds(for: .generate))
  }
}

// MARK: - 4) MCP executor retry contract (review r1, item 4)

/// What an MCP tool call should do after getting `QueueRecoveryGate`'s 503:
/// retry with the server's own backoff hint, or give up and surface a clear
/// tool error, once total elapsed time crosses the budget. Pure so the
/// retry/give-up boundary is directly testable without a network client.
enum QueueRecoveryRetryDecision: Equatable {
  case retry(afterSeconds: Double)
  /// `errorCode`/`retryAfterSeconds` are carried structurally (not just
  /// baked into `reason`'s prose) so a caller that parses tool errors — the
  /// daemon rescheduling a retry of its own — can act on them without
  /// re-parsing English (review r2, item 3).
  case giveUp(errorCode: String, retryAfterSeconds: Int, reason: String)
}

enum QueueRecoveryRetryPolicy {
  /// Review r2, item 3: the original 15-minute budget could NEVER be
  /// reached — MCP tool calls run under a ~300s daemon/client timeout
  /// (`ImageJobTracker`'s doc comment: "the caller's own turn timeout
  /// (180s)... the daemon-side 300s MCP tool timeout"), so a retry loop
  /// waiting toward 15 minutes just dies mid-wait when the TOOL CALL itself
  /// times out, with no clean give-up message ever reached. Capped well
  /// under that ceiling so a give-up response can actually be returned.
  static let maxWait: TimeInterval = 240

  static func decide(elapsed: TimeInterval, retryAfterSeconds: Int?) -> QueueRecoveryRetryDecision {
    let hinted = retryAfterSeconds ?? QueueRecoveryGate.minimumRetryAfterSeconds
    guard elapsed < maxWait else {
      return .giveUp(
        errorCode: QueueRecoveryGate.errorCode,
        retryAfterSeconds: hinted,
        reason: "Engine is still replaying its persisted queue after \(Int(maxWait))s of retries "
          + "(error_code: \(QueueRecoveryGate.errorCode)). The server's own estimate is "
          + "retry_after_seconds=\(hinted)s from its last response — reschedule this call for "
          + "roughly that long from now, or check /health.")
    }
    let wait = max(Double(QueueRecoveryGate.minimumRetryAfterSeconds), Double(hinted))
    // Never schedule a wait that would overshoot the budget by a lot — clamp
    // so the LAST attempt lands close to the cap instead of well past it.
    return .retry(afterSeconds: min(wait, maxWait - elapsed))
  }
}

// MARK: - 5) The 503 wire shape (review r1, item 4)

/// `{success:false, error, error_code, retry_after_seconds}` — the
/// `error_code` lets a caller distinguish THIS specific, retryable refusal
/// from any other 503 (e.g. "LTX-2 not configured", which is not
/// retryable); `retry_after_seconds` backs both the JSON body and the
/// `Retry-After` header on the same response.
struct QueueRecoveringErrorPayload: Encodable {
  let success: Bool
  let error: String
  let errorCode: String
  let retryAfterSeconds: Int
}

extension HTTPResponse {
  /// The one 503 every non-recoverable-kind route returns while a
  /// persisted-queue replay is in flight (`QueueRecoveryGate`).
  /// `remainingKinds` is `QueueRecoveryState.snapshot().remainingKinds` at
  /// the moment of refusal — the kind-weighted estimate (review r2, item 6).
  static func queueRecovering(remainingKinds: [QueueJobKind]) -> HTTPResponse {
    .queueRecovering(retryAfterSeconds: QueueRecoveryGate.retryAfterSeconds(remainingKinds: remainingKinds))
  }

  /// Count-only convenience — see `QueueRecoveryGate.retryAfterSeconds(remainingJobs:)`.
  static func queueRecovering(remainingJobs: Int) -> HTTPResponse {
    .queueRecovering(retryAfterSeconds: QueueRecoveryGate.retryAfterSeconds(remainingJobs: remainingJobs))
  }

  /// Same 503, for a call site (`WarmServerError.queueRecoveryInProgress`)
  /// that already computed its own `retryAfterSeconds` at throw time.
  static func queueRecovering(retryAfterSeconds retryAfter: Int) -> HTTPResponse {
    var response = HTTPResponse.json(
      status: 503,
      payload: QueueRecoveringErrorPayload(
        success: false, error: QueueRecoveryGate.reason,
        errorCode: QueueRecoveryGate.errorCode, retryAfterSeconds: retryAfter))
    response.extraHeaders["Retry-After"] = String(retryAfter)
    return response
  }
}

// MARK: - 6) Krita model-switch failure decision (review r2, item 1)

/// What `ComfyBridge`'s enqueued auto-model-switch should do when the switch
/// throws. A queue-recovery refusal must FAIL the prompt outright —
/// continuing on the CURRENT (wrong) checkpoint renders output that looks
/// fine but silently isn't what was asked for, with no error the caller can
/// see. Any OTHER model-switch failure keeps the original, deliberate
/// resilience: log it and continue rendering on the current model rather
/// than fail a whole prompt over an auto-switch hiccup.
enum ModelSwitchFailureDecision: Equatable {
  case failPrompt(message: String)
  case continueOnCurrentModel
}

enum ModelSwitchFailurePolicy {
  static func decide(_ error: Error) -> ModelSwitchFailureDecision {
    guard let recoveryError = error as? WarmServerError,
          case .queueRecoveryInProgress(let retryAfterSeconds) = recoveryError
    else {
      return .continueOnCurrentModel
    }
    return .failPrompt(
      message: "\(QueueRecoveryGate.reason) (error_code: \(QueueRecoveryGate.errorCode), "
        + "retry_after_seconds: \(retryAfterSeconds))")
  }
}

// MARK: - 7) Krita model-switch GATE decision (review r3, item 2)

/// Whether the auto-model-switch handler should refuse an attempt. Only an
/// ACTUAL switch (the requested checkpoint differs from the active one) is
/// gated during recovery — a no-op switch (the workflow already names the
/// active checkpoint) must never be refused, because there is no pool
/// mutation to protect: r2's version gated BEFORE this check, so every
/// Krita prompt carrying a checkpoint node hard-failed during recovery even
/// when the active model already matched.
enum ModelSwitchGate {
  static func shouldReject(isNoOpSwitch: Bool, recoveryInProgress: Bool) -> Bool {
    !isNoOpSwitch && QueueRecoveryGate.shouldReject(kind: .modelSwitch, recoveryInProgress: recoveryInProgress)
  }
}

// MARK: - 8) Admission-narrowing decision (review r4)

/// Outcome of racing "the job became observably admitted" against "the
/// enqueue Task itself already finished" (success or failure — an
/// immediate `queueFull`/`shuttingDown` throw never appends the job at
/// all, so waiting for admission alone would burn the FULL timeout for
/// nothing every time recovery hits the capacity gate).
enum AdmissionRaceOutcome: Equatable {
  case admitted
  case renderFinishedFirst
  case timedOut
}

/// Whether `recoverPersistedQueue`'s loop should narrow the tail (drop this
/// job from it, since `pending`/`active` now durably represents it instead)
/// immediately after an admission race resolves. Only `.admitted` says yes.
///
/// r3's version narrowed unconditionally on ANY return from
/// `waitUntilAdmitted`, including a timeout — but a render can legitimately
/// block the coordinator actor's cooperative thread pool well past 5s (a
/// DOCUMENTED, known risk in this codebase: see the "#300" note on
/// `WarmServerCoordinator`), so a timeout does NOT mean the job failed to
/// admit — it means we don't yet KNOW. Narrowing anyway would drop the job
/// from the tail before the coordinator actually holds it durably,
/// reopening the exact loss window this PR exists to close. `.timedOut`
/// must leave the tail as-is; the caller's OWN next loop iteration narrows
/// past this job safely once its `renderTask.value` has been awaited
/// (proving it is truly done, success or failure, either way).
///
/// `.renderFinishedFirst` also must not narrow here: an immediate
/// `queueFull`/`shuttingDown` throw never appended the job at all, so there
/// is nothing to narrow past. (A render that both admits AND fully
/// completes before the poll ever notices is vanishingly unlikely given the
/// poll interval — but even then, not narrowing here is still SAFE: the job
/// is done either way, and the next iteration's own narrow drops it
/// correctly once `renderTask.value` has settled it.)
enum AdmissionNarrowingPolicy {
  static func shouldNarrowNow(_ outcome: AdmissionRaceOutcome) -> Bool {
    outcome == .admitted
  }
}
