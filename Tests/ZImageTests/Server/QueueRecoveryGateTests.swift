import XCTest

@testable import ZImage

/// #339: Codex's review chain on PR #349.
///
/// r1 (MERGE AFTER FIXES): durability, kind vocabulary, gate coverage, 503
/// shape, MCP retry, cleanup/logging minors — see `QueueRecoveryGateTests`
/// and `RecoverySnapshotMergerTests` below.
///
/// r2 (2 open + new breakage found on re-review):
/// 1. Krita model switch degraded to a silent wrong-checkpoint render on a
///    recovery refusal — `ModelSwitchFailurePolicyTests`.
/// 2. `executeLoadModel` wasn't wired to the MCP retry helper — wiring-only,
///    no new pure logic to test beyond the shared `postWithQueueRecoveryRetry`.
/// 3. The 15-minute retry budget could never be reached under the ~300s MCP
///    tool timeout — capped at 240s, `.giveUp` now carries `error_code`/
///    `retry_after_seconds` structurally — `testRetryPolicy*` below.
/// 4. A one-job admission window: the tail was published as `jobs[index+1...]`
///    BEFORE job[index] was admitted, so a crash during admission itself
///    (decode, nearline LoRA staging, or the enqueue call's own window) lost
///    exactly that job — `testUnionOfPersistedJobIdsNeverDropsAJobDuringItsOwnAdmission`.
/// 5. `setRecoveryUnadmittedTail([])` now runs in a `defer` alongside
///    `finish()` (WarmServer.swift) — not independently unit-testable
///    without driving the real Task; covered by code review + the existing
///    lifecycle tests proving `finish()`'s own defer semantics.
/// 6. `perJobEstimateSeconds` is now per-kind (video 600s, else 20s) —
///    `testPerKindEstimates*`.
/// 7. Full route-wiring test — see `RouteWiringFeasibilityTests` for why a
///    live `recoverPersistedQueue()` drive isn't feasible from this suite.
///
/// r3 (2 open — the "transient" duplicate was not transient, and new
/// breakage):
/// 1. `RecoverySnapshotMerger.merge`/`.deduplicated` now dedupe by id
///    (first occurrence wins), and `WarmServer.recoverPersistedQueue`
///    narrows the tail as soon as a job is OBSERVABLY admitted
///    (`waitUntilAdmitted`, polling `liveHealth`), not once its render
///    finishes — see the new dedup tests in `RecoverySnapshotMergerTests`.
/// 2. `ModelSwitchGate` — the Krita recovery gate now runs AFTER the
///    "already on this model" no-op check, not before — `ModelSwitchGateTests`.
/// 3. `RouteWiringFeasibilityTests` now drives the real actor (via
///    `WarmServerQueueProbe`, paused so no render ever starts) instead of a
///    placeholder assertion.
/// 4. The `execution_error`-without-`execution_start` Krita send-slot
///    concern is a live-verification note in the PR body, not a code/test
///    change — see the PR description.
final class QueueRecoveryGateTests: XCTestCase {

  // MARK: - QueueRecoveryGate (pure decision)

  func testRejectsVideoWhileRecoveryIsInFlight() {
    XCTAssertTrue(QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: true))
  }

  func testAllowsVideoOnceRecoveryHasFinished() {
    XCTAssertFalse(QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: false))
  }

  func testRejectsControlnetModelSwitchAndModelLoadWhileRecovering() {
    XCTAssertTrue(QueueRecoveryGate.shouldReject(kind: .controlnet, recoveryInProgress: true))
    XCTAssertTrue(QueueRecoveryGate.shouldReject(kind: .modelSwitch, recoveryInProgress: true))
    XCTAssertTrue(QueueRecoveryGate.shouldReject(kind: .modelLoad, recoveryInProgress: true))
  }

  /// "generate" and "lora_swap" ARE persisted (`QueuePersistence.swift`) —
  /// they queue durably behind the recovered backlog like any other pending
  /// job and must keep answering 200/202 exactly as before, even mid-replay.
  /// `modelActivate`/`modelUnload` are synchronous (the caller holds the
  /// connection), so a process death is a visible failure, not a silent
  /// lost job id — out of scope for this gate.
  func testNeverRejectsSynchronousOrPersistableKindsEvenWhileRecovering() {
    for kind: QueueJobKind in [.generate, .loraSwap, .modelActivate, .modelUnload, .shutdown] {
      XCTAssertFalse(
        QueueRecoveryGate.shouldReject(kind: kind, recoveryInProgress: true),
        "\(kind.rawValue) must never be gated")
    }
  }

  func testReasonMakesNoTimingPromiseButRetryAfterSecondsDoes() {
    // Review r1, item 4: "do not promise seconds" in the message — the
    // structured retry_after_seconds field carries the actual estimate.
    XCTAssertFalse(QueueRecoveryGate.reason.lowercased().contains("few seconds"))
    XCTAssertGreaterThanOrEqual(QueueRecoveryGate.retryAfterSeconds(remainingJobs: 0), 5)
  }

  func testRetryAfterSecondsFloorsAtFiveAndScalesWithRemainingJobs() {
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingJobs: 0), 5)
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingJobs: 1), 20)
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingJobs: 10), 200)
  }

  // MARK: - Per-kind estimates (review r2, item 6)

  func testPerKindEstimatesDistinguishVideoFromEverythingElse() {
    XCTAssertEqual(QueueRecoveryGate.perJobEstimateSeconds(for: .video), 600)
    for kind: QueueJobKind in [.generate, .loraSwap, .controlnet, .modelSwitch, .modelLoad] {
      XCTAssertEqual(QueueRecoveryGate.perJobEstimateSeconds(for: kind), 20, "\(kind.rawValue) should use the non-video estimate")
    }
  }

  func testRetryAfterSecondsWithRemainingKindsSumsPerKindEstimates() {
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingKinds: [.generate, .generate]), 40)
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingKinds: [.video]), 600)
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingKinds: [.video, .generate]), 620)
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingKinds: []), 5, "floors at the minimum even with nothing remaining")
  }

  // MARK: - QueueJobKind coverage (review r1, item 2: single source of truth)

  func testEveryQueueJobKindHasAnExplicitGateRuling() {
    let expectedGated: Set<QueueJobKind> = [.video, .controlnet, .modelSwitch, .modelLoad]
    for kind in QueueJobKind.allCases {
      let rejected = QueueRecoveryGate.shouldReject(kind: kind, recoveryInProgress: true)
      XCTAssertEqual(
        rejected, expectedGated.contains(kind),
        "kind '\(kind.rawValue)' gating changed unexpectedly — update expectedGated deliberately, not by accident")
    }
  }

  func testQueueJobKindRawValuesMatchTheWireVocabulary() {
    // These strings are the wire contract (/v1/queue, /health, queue-state.json)
    // — pinned here so a rename is a deliberate, visible diff.
    XCTAssertEqual(QueueJobKind.generate.rawValue, "generate")
    XCTAssertEqual(QueueJobKind.controlnet.rawValue, "controlnet")
    XCTAssertEqual(QueueJobKind.loraSwap.rawValue, "lora_swap")
    XCTAssertEqual(QueueJobKind.modelSwitch.rawValue, "model_switch")
    XCTAssertEqual(QueueJobKind.modelLoad.rawValue, "model_load")
    XCTAssertEqual(QueueJobKind.modelActivate.rawValue, "model_activate")
    XCTAssertEqual(QueueJobKind.modelUnload.rawValue, "model_unload")
    XCTAssertEqual(QueueJobKind.video.rawValue, "video")
    XCTAssertEqual(QueueJobKind.shutdown.rawValue, "shutdown")
  }

  // MARK: - QueueRecoveryState (cross-task signal)

  func testStateDefaultsNotInProgressWithZeroRemaining() {
    let state = QueueRecoveryState()
    let snap = state.snapshot()
    XCTAssertFalse(snap.inProgress)
    XCTAssertEqual(snap.remaining, 0)
    XCTAssertEqual(snap.remainingKinds, [])
  }

  func testStateLifecycleBeginAdmitFinish() {
    let state = QueueRecoveryState()
    state.begin(jobKinds: [.generate, .generate, .loraSwap])
    XCTAssertEqual(state.snapshot().inProgress, true)
    XCTAssertEqual(state.snapshot().remaining, 3)
    XCTAssertEqual(state.snapshot().remainingKinds, [.generate, .generate, .loraSwap])

    state.jobAdmitted()
    XCTAssertEqual(state.snapshot().remaining, 2)
    XCTAssertEqual(state.snapshot().remainingKinds, [.generate, .loraSwap], "pops FIFO, matching replay order")
    state.jobAdmitted()
    state.jobAdmitted()
    XCTAssertEqual(state.snapshot().remaining, 0)

    // jobAdmitted() past empty must not crash or go negative.
    state.jobAdmitted()
    XCTAssertEqual(state.snapshot().remaining, 0)

    state.finish()
    let snap = state.snapshot()
    XCTAssertFalse(snap.inProgress)
    XCTAssertEqual(snap.remaining, 0)
    XCTAssertEqual(snap.remainingKinds, [])
  }

  func testStateIsSafeUnderConcurrentAccess() {
    let state = QueueRecoveryState()
    state.begin(jobKinds: Array(repeating: QueueJobKind.generate, count: 500))
    DispatchQueue.concurrentPerform(iterations: 500) { _ in
      state.jobAdmitted()
      _ = state.snapshot()
    }
    XCTAssertEqual(state.snapshot().remaining, 0)
  }

  // MARK: - Route-level composition (mirrors what each gated route does)

  func testRouteLevelCompositionRefusesOnlyDuringTheRecoveryWindow() {
    let state = QueueRecoveryState()

    XCTAssertFalse(
      QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: state.snapshot().inProgress),
      "before recovery starts, local video must be accepted")

    state.begin(jobKinds: [.generate, .generate])
    XCTAssertTrue(
      QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: state.snapshot().inProgress),
      "while recovery is replaying the persisted backlog, local video must be refused, not silently lost")

    state.finish()
    XCTAssertFalse(
      QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: state.snapshot().inProgress),
      "once recovery's replay Task completes, local video must be accepted again")
  }

  /// Review r1, item 5: the recovery lifecycle + gate driven together —
  /// flag set for the span of a simulated replay, cleared after, with the
  /// remaining count trending to zero and the gate tracking it throughout.
  func testRecoveryLifecycleAndGateTogether() {
    let state = QueueRecoveryState()
    state.begin(jobKinds: [.generate, .generate, .generate])

    // Job 1 admitted: still recovering, gate still fires, remaining drops.
    state.jobAdmitted()
    var snap = state.snapshot()
    XCTAssertTrue(snap.inProgress)
    XCTAssertEqual(snap.remaining, 2)
    XCTAssertTrue(QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: snap.inProgress))
    XCTAssertEqual(QueueRecoveryGate.retryAfterSeconds(remainingKinds: snap.remainingKinds), 40)

    // Jobs 2 and 3 admitted.
    state.jobAdmitted()
    state.jobAdmitted()
    snap = state.snapshot()
    XCTAssertEqual(snap.remaining, 0)
    XCTAssertTrue(snap.inProgress, "still 'in flight' until finish() runs, even at remaining == 0")

    state.finish()
    snap = state.snapshot()
    XCTAssertFalse(snap.inProgress)
    XCTAssertFalse(QueueRecoveryGate.shouldReject(kind: .video, recoveryInProgress: snap.inProgress))
  }

  // MARK: - 503 body shape (review r1, item 5)

  func testQueueRecoveringResponseBodyAndHeaderShape() throws {
    let response = HTTPResponse.queueRecovering(remainingJobs: 2)
    XCTAssertEqual(response.status, 503)
    XCTAssertEqual(response.extraHeaders["Retry-After"], "40")

    let json = try JSONSerialization.jsonObject(with: response.body) as? [String: Any]
    let payload = try XCTUnwrap(json)
    XCTAssertEqual(payload["success"] as? Bool, false)
    XCTAssertEqual(payload["error_code"] as? String, "queue_recovery_in_progress")
    XCTAssertEqual(payload["retry_after_seconds"] as? Int, 40)
    let message = try XCTUnwrap(payload["error"] as? String)
    XCTAssertFalse(message.lowercased().contains("few seconds"), "no false timing promise in the message")
  }

  func testQueueRecoveringWithRemainingKindsUsesThePerKindEstimate() throws {
    let response = HTTPResponse.queueRecovering(remainingKinds: [.video])
    XCTAssertEqual(response.extraHeaders["Retry-After"], "600")
    let json = try JSONSerialization.jsonObject(with: response.body) as? [String: Any]
    XCTAssertEqual(json?["retry_after_seconds"] as? Int, 600)
  }

  func testQueueRecoveringFromPrecomputedRetryAfterSecondsMatchesTheHeader() throws {
    let response = HTTPResponse.queueRecovering(retryAfterSeconds: 123)
    XCTAssertEqual(response.extraHeaders["Retry-After"], "123")
    let json = try JSONSerialization.jsonObject(with: response.body) as? [String: Any]
    XCTAssertEqual(json?["retry_after_seconds"] as? Int, 123)
  }

  // MARK: - QueueRecoveryRetryPolicy (review r1 item 4 / r2 item 3: MCP executor retry)

  func testRetryPolicyRetriesUsingTheServersHint() {
    let decision = QueueRecoveryRetryPolicy.decide(elapsed: 0, retryAfterSeconds: 40)
    guard case .retry(let after) = decision else { return XCTFail("expected .retry, got \(decision)") }
    XCTAssertEqual(after, 40)
  }

  func testRetryPolicyFallsBackToTheFloorWithNoHint() {
    let decision = QueueRecoveryRetryPolicy.decide(elapsed: 0, retryAfterSeconds: nil)
    guard case .retry(let after) = decision else { return XCTFail("expected .retry, got \(decision)") }
    XCTAssertEqual(after, Double(QueueRecoveryGate.minimumRetryAfterSeconds))
  }

  /// Review r2, item 3: capped at 240s — MCP tool calls run under a ~300s
  /// daemon/client timeout, so the old 15-minute budget could never
  /// actually be reached; a give-up response has to land well before that.
  func testMaxWaitIsCappedWellUnderTheMCPToolTimeout() {
    XCTAssertEqual(QueueRecoveryRetryPolicy.maxWait, 240)
    XCTAssertLessThan(QueueRecoveryRetryPolicy.maxWait, 300, "must stay under the ~300s MCP/daemon tool timeout")
  }

  func testRetryPolicyGivesUpPastTheBudgetNamingErrorCodeAndRetryAfterSeconds() {
    let decision = QueueRecoveryRetryPolicy.decide(elapsed: 240, retryAfterSeconds: 42)
    guard case .giveUp(let errorCode, let retryAfterSeconds, let reason) = decision else {
      return XCTFail("expected .giveUp, got \(decision)")
    }
    XCTAssertEqual(errorCode, "queue_recovery_in_progress")
    XCTAssertEqual(retryAfterSeconds, 42)
    XCTAssertTrue(reason.contains("queue_recovery_in_progress"), "reason must literally name the error_code: \(reason)")
    XCTAssertTrue(reason.contains("42"), "reason must literally name retry_after_seconds: \(reason)")
  }

  func testRetryPolicyGiveUpFallsBackToTheFloorWithNoHint() {
    let decision = QueueRecoveryRetryPolicy.decide(elapsed: 240, retryAfterSeconds: nil)
    guard case .giveUp(_, let retryAfterSeconds, _) = decision else { return XCTFail("expected .giveUp, got \(decision)") }
    XCTAssertEqual(retryAfterSeconds, QueueRecoveryGate.minimumRetryAfterSeconds)
  }

  func testRetryPolicyNeverSchedulesPastTheBudget() {
    // 230s elapsed, a 60s hint — must clamp to the ~10s left, not 60s.
    let decision = QueueRecoveryRetryPolicy.decide(elapsed: 230, retryAfterSeconds: 60)
    guard case .retry(let after) = decision else { return XCTFail("expected .retry, got \(decision)") }
    XCTAssertLessThanOrEqual(after, 10)
  }

  func testRetryPolicyBoundaryAtExactlyTheBudgetGivesUp() {
    let decision = QueueRecoveryRetryPolicy.decide(elapsed: QueueRecoveryRetryPolicy.maxWait, retryAfterSeconds: 5)
    guard case .giveUp = decision else { return XCTFail("expected .giveUp at the exact boundary, got \(decision)") }
  }
}

// MARK: - RecoverySnapshotMerger (review r1 item 1 / r2 item 4: durability)

final class RecoverySnapshotMergerTests: XCTestCase {

  private func job(_ id: String) -> PersistedQueueJob {
    PersistedQueueJob(id: id, kind: "generate", source: "api", enqueuedAt: Date(), rawBody: Data("{}".utf8))
  }

  /// The exact scenario review r1 asked for: a persisted snapshot with 3
  /// jobs, "restart" simulated right after job 1 is admitted (active) — jobs
  /// 2 and 3 must still be present in what gets persisted.
  func testUnadmittedTailSurvivesASimulatedRestartAfterJobOneIsAdmitted() {
    let merged = RecoverySnapshotMerger.merge(
      admittedActive: job("job-1"),
      admittedPending: [],
      unadmittedTail: [job("job-2"), job("job-3")])

    XCTAssertEqual(merged.active?.id, "job-1")
    XCTAssertEqual(merged.pending.map(\.id), ["job-2", "job-3"], "jobs 2 and 3 must survive")
  }

  /// The OLD behavior this fixes: with no tail tracked at all (the bug),
  /// only job 1 would ever be persisted — asserted here as the negative case
  /// the merge must NOT reproduce.
  func testOldBehaviorWithNoTailWouldHaveLostJobsTwoAndThree() {
    let unmerged = PersistedQueueState(active: job("job-1"), pending: [])
    XCTAssertTrue(unmerged.pending.isEmpty, "pinning the bug this fix removes: no tail means nothing survives")

    let merged = RecoverySnapshotMerger.merge(
      admittedActive: job("job-1"), admittedPending: [], unadmittedTail: [job("job-2"), job("job-3")])
    XCTAssertFalse(merged.pending.isEmpty, "the fix: the tail is no longer silently dropped")
  }

  /// FIFO order preserved with a live job arriving mid-replay: a live
  /// submission (already admitted, so present in `admittedPending`) must not
  /// jump the persisted ordering ahead of older, pre-restart recovered work
  /// — the tail goes first, so a subsequent restart finishes it before the
  /// live job's own turn.
  func testFIFOOrderPreservedWithALiveJobArrivingMidReplay() {
    let liveJob = job("live-job")
    let merged = RecoverySnapshotMerger.merge(
      admittedActive: job("job-1"),
      admittedPending: [liveJob],
      unadmittedTail: [job("job-2"), job("job-3")])

    XCTAssertEqual(
      merged.pending.map(\.id), ["job-2", "job-3", "live-job"],
      "recovered tail must be replayed before a job that arrived live mid-replay")
  }

  func testEmptyTailIsANoOpMerge() {
    let merged = RecoverySnapshotMerger.merge(admittedActive: nil, admittedPending: [job("a")], unadmittedTail: [])
    XCTAssertNil(merged.active)
    XCTAssertEqual(merged.pending.map(\.id), ["a"])
  }

  func testNilActiveWithATailStillPersistsTheTail() {
    // The moment between "job 1 finished" and "job 2 admitted": no active
    // job, job 1 no longer pending, but jobs 2/3 must still show up.
    let merged = RecoverySnapshotMerger.merge(admittedActive: nil, admittedPending: [], unadmittedTail: [job("job-2"), job("job-3")])
    XCTAssertNil(merged.active)
    XCTAssertEqual(merged.pending.map(\.id), ["job-2", "job-3"])
  }

  // MARK: - Review r2, item 4: the one-job admission window

  /// Simulates exactly what `WarmServer.recoverPersistedQueue`'s loop does
  /// for a 3-job backlog, and asserts the invariant review r2 demanded:
  /// AT EVERY STEP, the union of ids across `merged.active` + `merged.pending`
  /// is a SUPERSET of every job from the current index onward — no job is
  /// EVER momentarily absent from what would be persisted. This is what
  /// "publish `jobs[index...]`, not `jobs[index+1...]`" buys: job[index]
  /// stays in the tail through its own decode/staging/enqueue window and
  /// only leaves once the NEXT iteration narrows the tail — a job may
  /// appear in both the tail and the admitted state for an instant (a
  /// harmless duplicate), but is never in NEITHER.
  func testUnionOfPersistedJobIdsNeverDropsAJobDuringItsOwnAdmission() {
    let jobs = [job("job-1"), job("job-2"), job("job-3")]

    for index in jobs.indices {
      let remainingFromHere = Set(jobs[index...].map(\.id))

      // Moment A: tail published as jobs[index...] (review r2's fix), BEFORE
      // this job's decode/staging/enqueue has even started — nothing is
      // admitted yet for this job.
      let beforeAdmission = RecoverySnapshotMerger.merge(
        admittedActive: index > 0 ? nil : nil,
        admittedPending: [],
        unadmittedTail: Array(jobs[index...]))
      let idsA = Set(([beforeAdmission.active].compactMap { $0 } + beforeAdmission.pending).map(\.id))
      XCTAssertTrue(
        remainingFromHere.isSubset(of: idsA),
        "index \(index), before admission: every remaining job must be represented — got \(idsA)")

      // Moment B: the enqueue call has appended job[index] to `pending`
      // (or dequeued it into `active`), but the tail hasn't been narrowed
      // to jobs[(index+1)...] yet — job[index] may appear in BOTH; that is
      // the accepted transient duplicate, not a loss.
      let duringAdmission = RecoverySnapshotMerger.merge(
        admittedActive: jobs[index],
        admittedPending: [],
        unadmittedTail: Array(jobs[index...]))
      let idsB = Set(([duringAdmission.active].compactMap { $0 } + duringAdmission.pending).map(\.id))
      XCTAssertTrue(
        remainingFromHere.isSubset(of: idsB),
        "index \(index), during admission: every remaining job must STILL be represented — got \(idsB)")

      // Moment C: job[index] has finished; the tail has NOT yet advanced to
      // index+1 (that happens at the top of the next loop iteration) — the
      // job is done (no longer "remaining"), but everything AFTER it must
      // still be fully represented via the (not-yet-narrowed) tail.
      if index + 1 < jobs.count {
        let afterAdmission = RecoverySnapshotMerger.merge(
          admittedActive: nil, admittedPending: [], unadmittedTail: Array(jobs[index...]))
        let idsC = Set(([afterAdmission.active].compactMap { $0 } + afterAdmission.pending).map(\.id))
        let remainingAfterThisJob = Set(jobs[(index + 1)...].map(\.id))
        XCTAssertTrue(
          remainingAfterThisJob.isSubset(of: idsC),
          "index \(index), after admission (tail not yet advanced): later jobs must still be represented — got \(idsC)")
      }
    }
  }

  /// The r1 regression this specifically fixes: publishing `jobs[index+1...]`
  /// (excluding job[index]) BEFORE its own admission left a real window
  /// where job[index] was in NEITHER the tail NOR `pending`/`active`.
  func testTheR1RegressionWouldHaveDroppedTheJobBeingAdmitted() {
    let jobs = [job("job-1"), job("job-2"), job("job-3")]
    let index = 1  // job-2, mid-admission — decoding/staging, not yet enqueued.

    // r1's (buggy) tail: everything AFTER job[index], excluding it.
    let r1Tail = Array(jobs[jobs.index(after: index)...])
    let r1Merged = RecoverySnapshotMerger.merge(admittedActive: nil, admittedPending: [], unadmittedTail: r1Tail)
    let r1Ids = Set(([r1Merged.active].compactMap { $0 } + r1Merged.pending).map(\.id))
    XCTAssertFalse(r1Ids.contains("job-2"), "pinning the r1 bug: job-2 is absent from BOTH the tail and admitted state here")

    // r2's fix: tail includes job[index] itself.
    let r2Tail = Array(jobs[index...])
    let r2Merged = RecoverySnapshotMerger.merge(admittedActive: nil, admittedPending: [], unadmittedTail: r2Tail)
    let r2Ids = Set(([r2Merged.active].compactMap { $0 } + r2Merged.pending).map(\.id))
    XCTAssertTrue(r2Ids.contains("job-2"), "the fix: job-2 stays represented through its own admission window")
  }

  // MARK: - Review r3, item 1a: deduplication (the compounding-duplicate bug)

  /// The exact failure mode r3 found: r2's tail only narrowed once the WHOLE
  /// RENDER finished (the enqueue call's `await` doesn't return until then),
  /// so for the render's entire duration the job was BOTH the tail's first
  /// entry AND `active`. Without dedup, that shows up as the same id twice
  /// in `merged.pending` — asserted here as the negative case `merge` must
  /// not reproduce now that it dedupes.
  func testMergeDedupesAJobThatIsBothActiveAndStillInTheTail() {
    let merged = RecoverySnapshotMerger.merge(
      admittedActive: job("job-1"),
      admittedPending: [],
      unadmittedTail: [job("job-1"), job("job-2")])  // job-1 still in the tail while it's active

    XCTAssertEqual(merged.active?.id, "job-1")
    XCTAssertEqual(merged.pending.map(\.id), ["job-2"], "job-1 must NOT also appear in pending — active wins")
  }

  /// `admittedPending` can carry the same overlap (job admitted into
  /// `pending`, not yet dequeued into `active`, tail not yet narrowed) —
  /// same dedup requirement, different input slot.
  func testMergeDedupesAJobThatIsBothPendingAndStillInTheTail() {
    let merged = RecoverySnapshotMerger.merge(
      admittedActive: nil,
      admittedPending: [job("job-1")],
      unadmittedTail: [job("job-1"), job("job-2")])

    XCTAssertEqual(merged.pending.map(\.id), ["job-1", "job-2"], "job-1 appears exactly once, tail position wins (first occurrence)")
  }

  /// The dedup must never cost the "no loss" invariant — every id present
  /// in ANY of the three inputs is still present exactly once in the output.
  func testDedupPreservesEveryIdWhileRemovingDuplicates() {
    let merged = RecoverySnapshotMerger.merge(
      admittedActive: job("job-1"),
      admittedPending: [job("job-1"), job("job-3")],
      unadmittedTail: [job("job-1"), job("job-2")])

    let allIds = ([merged.active].compactMap { $0 } + merged.pending).map(\.id)
    XCTAssertEqual(Set(allIds), ["job-1", "job-2", "job-3"], "no id lost")
    XCTAssertEqual(allIds.count, Set(allIds).count, "no id duplicated")
  }

  func testDeduplicatedKeepsFirstOccurrenceAndDropsLaterOnes() {
    let result = RecoverySnapshotMerger.deduplicated([job("a"), job("b"), job("a"), job("c"), job("b")])
    XCTAssertEqual(result.map(\.id), ["a", "b", "c"])
  }

  func testDeduplicatedIsANoOpWithNoDuplicates() {
    let result = RecoverySnapshotMerger.deduplicated([job("a"), job("b"), job("c")])
    XCTAssertEqual(result.map(\.id), ["a", "b", "c"])
  }

  /// A `queue-state.json` snapshot that (from a past bug, or a future one)
  /// carries the same id in both `active` and `pending` must collapse to
  /// ONE job when `recoverPersistedQueue` builds its replay list — this is
  /// the exact `active + pending` concatenation it dedupes (WarmServer.swift).
  func testASnapshotWithADuplicatedIdLoadsAsOneJobAfterDeduplication() {
    let combined = [job("job-1")] + [job("job-1"), job("job-2")]  // active=job-1, pending=[job-1 (dup), job-2]
    let deduped = RecoverySnapshotMerger.deduplicated(combined)
    XCTAssertEqual(deduped.map(\.id), ["job-1", "job-2"], "the duplicate collapses to a single replay, not two")
  }
}

// MARK: - ModelSwitchFailurePolicy (review r2, item 1)

final class ModelSwitchFailurePolicyTests: XCTestCase {

  func testQueueRecoveryRefusalFailsThePrompt() {
    let decision = ModelSwitchFailurePolicy.decide(WarmServerError.queueRecoveryInProgress(retryAfterSeconds: 42))
    guard case .failPrompt(let message) = decision else { return XCTFail("expected .failPrompt, got \(decision)") }
    XCTAssertTrue(message.contains("queue_recovery_in_progress"), "message must name the error_code: \(message)")
    XCTAssertTrue(message.contains("42"), "message must name retry_after_seconds: \(message)")
  }

  /// Any OTHER model-switch failure keeps the ORIGINAL, deliberate
  /// resilience: continue rendering on the current model rather than fail a
  /// whole prompt over an auto-switch hiccup (e.g. the model wasn't found,
  /// a transient pool error).
  func testOtherFailuresContinueOnCurrentModel() {
    XCTAssertEqual(
      ModelSwitchFailurePolicy.decide(WarmServerError.krea2NotLoaded), .continueOnCurrentModel)
    XCTAssertEqual(
      ModelSwitchFailurePolicy.decide(ModelPoolError.modelNotInPool("x")), .continueOnCurrentModel)
  }

  func testFailPromptDecisionIsNotJustAnyWarmServerError() {
    // A DIFFERENT WarmServerError case must not accidentally trip the
    // recovery-refusal branch.
    let decision = ModelSwitchFailurePolicy.decide(WarmServerError.invalidRequest(message: "bogus"))
    XCTAssertEqual(decision, .continueOnCurrentModel)
  }
}

// MARK: - ModelSwitchGate (review r3, item 2)

final class ModelSwitchGateTests: XCTestCase {

  /// The r3 regression: r2 gated BEFORE the no-op check, so a Krita prompt
  /// whose checkpoint already matches the active model hard-failed during
  /// recovery even though no switch — and therefore no pool mutation to
  /// protect — was ever going to happen.
  func testNoOpSwitchIsNeverRejectedEvenDuringRecovery() {
    XCTAssertFalse(ModelSwitchGate.shouldReject(isNoOpSwitch: true, recoveryInProgress: true))
    XCTAssertFalse(ModelSwitchGate.shouldReject(isNoOpSwitch: true, recoveryInProgress: false))
  }

  func testActualSwitchIsRejectedOnlyDuringRecovery() {
    XCTAssertTrue(ModelSwitchGate.shouldReject(isNoOpSwitch: false, recoveryInProgress: true))
    XCTAssertFalse(ModelSwitchGate.shouldReject(isNoOpSwitch: false, recoveryInProgress: false))
  }
}

// MARK: - Route-wiring probe test (review r3, item 3)

/// Review r2 item 7 asked for a route-wiring test "via the DEBUG coordinator
/// seam" — ruled infeasible in r2 because `queueRecoveryState` lives on
/// `WarmServer`, not the actor the probe wraps, and a full `recoverPersistedQueue`
/// drive needs `run()`, which binds the HTTP listener (forbidden here).
///
/// Review r3 identified a narrower, genuinely feasible slice of that: the
/// probe CAN drive `setRecoveryUnadmittedTail` + `enqueueGenerate` directly
/// (both added to `WarmServerQueueProbe` for this), under an isolated
/// `COMFYBOX_STATE_DIR`, and `queue-state.json` on disk is real, inspectable
/// output — this proves the ACTUAL persistence behavior
/// `recoverPersistedQueue` depends on (the merge, the dedup) through the
/// real actor, not just the pure `RecoverySnapshotMerger` function in
/// isolation. The queue stays PAUSED throughout, so `.generate` never
/// reaches a real render (`runsWhilePaused`) — no model weights are ever
/// touched, matching AGENT-RULES' unit-tests-only constraint.
final class RouteWiringFeasibilityTests: XCTestCase {

  private func tailJob(_ id: String) -> PersistedQueueJob {
    PersistedQueueJob(id: id, kind: "generate", source: "api", enqueuedAt: Date(), rawBody: Data("{}".utf8))
  }

  func testAdmittedJobAndUnadmittedTailPersistTogetherWithNoDuplicateIds() async throws {
    _ = try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()
    await probe.setPaused(true)  // `.generate` never runs while paused — no model weights touched.

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let rawBody = Data(#"{"prompt":"r3-probe"}"#.utf8)
    let payload = try decoder.decode(GeneratePayload.self, from: rawBody)

    // Mirrors `recoverPersistedQueue`'s "publish jobs[index...] BEFORE
    // attempting admission" (review r2 item 4): job-1 is about to be
    // admitted, job-2/job-3 are still purely in the tail.
    await probe.setRecoveryUnadmittedTail([tailJob("job-1"), tailJob("job-2"), tailJob("job-3")])

    let renderTask = Task { try? await probe.enqueueGenerate(payload, rawBody: rawBody, jobId: "job-1") }

    // Poll for admission exactly like `WarmServer.waitUntilAdmitted` does —
    // observable within milliseconds since the queue is otherwise idle.
    let deadline = Date().addingTimeInterval(5)
    while !probe.snapshotPendingIds.contains("job-1"), Date() < deadline {
      try await Task.sleep(nanoseconds: 2_000_000)
    }
    XCTAssertTrue(probe.snapshotPendingIds.contains("job-1"), "job-1 must be observably admitted (appended to pending)")

    // Narrow the tail NOW — review r3 item 1b: as soon as admission is
    // observed, not after the (paused, never-running) render "finishes".
    await probe.setRecoveryUnadmittedTail([tailJob("job-2"), tailJob("job-3")])

    let onDisk = try XCTUnwrap(QueueStateStore.load(), "queue-state.json must exist with job-1 admitted + the tail")
    let allIds = ([onDisk.active].compactMap { $0 } + onDisk.pending).map(\.id)
    XCTAssertEqual(Set(allIds), ["job-1", "job-2", "job-3"], "admitted job + unadmitted tail, all present")
    XCTAssertEqual(allIds.count, Set(allIds).count, "no id is duplicated on disk — the r3 compounding-duplicate bug")

    // Cleanup: job-1 never rendered (paused) — cancel it and clear the tail
    // so the probe drains for `makeQueueProbe`'s teardown guard.
    await probe.cancelPending(id: "job-1")
    await probe.setRecoveryUnadmittedTail([])
    renderTask.cancel()
    _ = await renderTask.value
    await probe.setPaused(false)
  }
}
