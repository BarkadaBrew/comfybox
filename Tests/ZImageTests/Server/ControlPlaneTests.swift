// ControlPlaneTests.swift — 0.B-2 control-plane carve-out
// (FDD-ui-api-parity.md §3.1.4, §3.1.4a, §3.1.5, §4.1; comfybox#300).
//
// The six named scenarios from the FDD's Phase-0 test budget, plus the
// classifier/flag unit coverage. They drive the REAL coordinator + lock store
// through `WarmServerQueueProbe` (the same seam ModelOperationQueueTests uses),
// with `COMFYBOX_STATE_DIR` isolated so the live engine's queue/pause/sidecar
// are never touched.

import Foundation
import XCTest

@testable import ZImage

final class ControlPlaneTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    try isolateComfyBoxStateDirectory()
  }

  /// Poll until `predicate` holds or the deadline passes; fail loudly on timeout.
  private func waitUntil(
    _ description: String, timeout: TimeInterval = 6,
    _ predicate: @escaping () -> Bool,
    file: StaticString = #filePath, line: UInt = #line
  ) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while !predicate() && Date() < deadline {
      try await Task.sleep(nanoseconds: 5_000_000)
    }
    XCTAssertTrue(predicate(), "waitUntil timed out: \(description)", file: file, line: line)
  }

  // MARK: - Scenario 1: lock-store authority

  /// Pause becomes visible IMMEDIATELY during a long-running operation — not
  /// after the actor catches up. `publishHealth()` no longer writes `isPaused`;
  /// the overlay in `LiveHealthState.read()` (the GET /v1/queue path) is the sole
  /// source, so this proves authority moved to the lock store (§3.1.5).
  func testPauseIsVisibleImmediatelyDuringALongOperation() async throws {
    let probe = makeQueueProbe()
    async let op: Bool = probe.enqueueSynthetic(durationMs: 1200, id: "long")
    try await waitUntil("synthetic op running") { probe.activeJobSummary != nil }

    XCTAssertFalse(probe.lockStorePaused)
    XCTAssertFalse(probe.isPaused)

    probe.controlPause()  // the sync /v1/queue/pause path — a lock-store write

    XCTAssertTrue(probe.lockStorePaused, "authoritative pause set immediately")
    XCTAssertTrue(probe.isPaused, "visible via read() overlay (the GET /v1/queue path) mid-op")

    let finished = try await op   // pause is a between-items gate: op still finishes
    XCTAssertTrue(finished)
  }

  // MARK: - Scenario 2: delta apply / no-drop under interleaved enqueue (F5)

  /// The FDD's F5 "lost jobs" regression: enqueue, record a cancel delta, enqueue
  /// more concurrently, drain — and assert no operation silently vanishes and the
  /// delta applies exactly once.
  func testDeltaAppliesExactlyOnceWithNoLostJobsUnderInterleavedEnqueue() async throws {
    let probe = makeQueueProbe()

    async let a: Bool = probe.enqueueSynthetic(durationMs: 700, id: "A")   // occupies the loop
    try await waitUntil("A running") { probe.activeJobSummary != nil }

    async let b: Bool = probe.enqueueSynthetic(durationMs: 300, id: "B")   // parks behind A
    try await waitUntil("B pending") { probe.snapshotPendingIds.contains("B") }

    // Cancel B via the sync delta path, then enqueue C concurrently (interleave).
    probe.controlCancel(id: "B")
    XCTAssertFalse(probe.composedPendingIds.contains("B"), "cancelled job gone from the composed queue at once")
    async let c: Bool = probe.enqueueSynthetic(durationMs: 300, id: "C")

    // No operation silently vanishes: A and C complete, B's continuation resolves
    // (as a cancellation) rather than hanging.
    let aResult = try await a
    let cResult = try await c
    XCTAssertTrue(aResult, "A completed")
    XCTAssertTrue(cResult, "C completed — not dropped by the interleaved cancel")

    var bWasCancelled = false
    do { _ = try await b; XCTFail("B should have been cancelled by the delta") }
    catch { bWasCancelled = true }
    XCTAssertTrue(bWasCancelled)

    // Applied exactly once: nothing left undrained.
    try await waitUntil("deltas drained") { probe.undrainedDeltaCount == 0 }
    XCTAssertEqual(probe.undrainedDeltaCount, 0)
  }

  // MARK: - Scenario 3: resume wakes idle loop (F1 wedge)

  /// The FDD's F1 "wedge": pause while idle, enqueue a job (it parks because the
  /// loop exits when paused with no `runsWhilePaused` work), then RESUME — and the
  /// loop must restart and run the parked job WITHOUT a new enqueue kicking it.
  /// v1's mailbox `resume` would 202 and wedge here forever; the fire-and-forget
  /// `setPaused(false) -> startProcessingIfNeeded()` path is what avoids it.
  func testResumeWakesAParkedLoopWithoutANewEnqueue() async throws {
    let probe = makeQueueProbe()

    probe.controlPause()
    XCTAssertTrue(probe.lockStorePaused)

    async let op: Bool = probe.enqueueSynthetic(durationMs: 250, id: "parked")
    // Give the loop a chance to (correctly) NOT run the job while paused.
    try await waitUntil("job parked in pending") { probe.snapshotPendingIds.contains("parked") }
    try? await Task.sleep(nanoseconds: 200_000_000)
    XCTAssertNil(probe.activeJobSummary, "a paused loop must not start the job")
    XCTAssertEqual(probe.pendingCount, 1)

    probe.controlResume()  // the sync fire-and-forget resume — the wake

    let finished = try await op   // hangs forever here if resume wedged the loop
    XCTAssertTrue(finished, "the parked job ran after resume, with no new enqueue")
    XCTAssertFalse(probe.lockStorePaused)
  }

  // MARK: - Scenario 4: sidecar persistence replay

  /// Undrained deltas written to `queue-deltas.json` survive a "restart" and
  /// replay correctly: this is exactly `QueueDeltaStore.load()` +
  /// `QueueDeltaApplier.apply()`, the two calls `recoverPersistedQueue` makes to
  /// fold the sidecar into the recovered queue before re-enqueue (§3.1.4a point 4).
  /// A persisted cancel keeps its job from resurrecting; a persisted move survives.
  func testUndrainedDeltasReplayFromTheSidecarOnRestart() throws {
    // Persist undrained deltas, as the sync route does before a crash.
    QueueDeltaStore.save([.cancel("B"), .move("C", direction: "top")])
    XCTAssertTrue(FileManager.default.fileExists(atPath: QueueDeltaStore.path.path))

    // "Restart": a fresh process reads the sidecar.
    let loaded = QueueDeltaStore.load()
    XCTAssertEqual(loaded.count, 2)

    // Fold against the recovered persisted job order [A, B, C, D].
    struct RecoveredJob { let id: String }
    let jobs = [RecoveredJob(id: "A"), RecoveredJob(id: "B"), RecoveredJob(id: "C"), RecoveredJob(id: "D")]
    let effective = QueueDeltaApplier.apply(loaded, to: jobs, id: { $0.id })

    XCTAssertEqual(effective.map { $0.id }, ["C", "A", "D"], "B cancelled, C moved to top, order otherwise preserved")
    XCTAssertFalse(effective.contains { $0.id == "B" }, "cancel -> bounce -> stays cancelled")

    // A clean drain deletes the sidecar (empty set removes the file).
    QueueDeltaStore.save([])
    XCTAssertFalse(FileManager.default.fileExists(atPath: QueueDeltaStore.path.path))
  }

  // MARK: - Scenario 5: classifier serves control routes with zero cooperative threads

  /// Saturate the cooperative pool with blocking tasks, then confirm a sync
  /// control operation (a pure lock-store call — exactly what
  /// `serveControlPlaneSync` does) still answers immediately, while a task
  /// scheduled onto the pool does NOT — proving the sync path needed no
  /// cooperative thread.
  ///
  /// Deliberately a SYNCHRONOUS test. It must never `await` while the pool is
  /// saturated: an async continuation would need one of the very cooperative
  /// threads this test blocks, deadlocking the whole run. XCTest drives sync
  /// tests off the pool, so `Thread.sleep`/semaphore waits here are safe and
  /// the saturators stay blocked until the defer releases them.
  func testSyncControlPathAnswersWithTheCooperativePoolSaturated() throws {
    let probe = makeQueueProbe()

    let cores = ProcessInfo.processInfo.activeProcessorCount
    let saturators = max(12, cores * 3)
    let release = DispatchSemaphore(value: 0)
    let started = DispatchSemaphore(value: 0)
    for _ in 0..<saturators {
      Task.detached {
        started.signal()
        release.wait()  // block this cooperative worker
      }
    }
    defer {
      for _ in 0..<saturators { release.signal() }
      probe.controlResume()
    }

    // Wait (on the XCTest thread, off the pool) until the pool is occupied:
    // the pool has ~activeProcessorCount workers, so once `cores` saturators
    // have started, every worker is parked in `release.wait()`.
    var occupied = 0
    let deadline = DispatchTime.now() + .seconds(10)
    while occupied < cores, started.wait(timeout: deadline) == .success { occupied += 1 }
    XCTAssertGreaterThanOrEqual(occupied, cores, "pool saturated before the probe")

    // Contrast: a task scheduled onto the (now saturated) pool cannot even start.
    let coopRan = LockedFlag()
    Task.detached { coopRan.trySet() }

    let start = Date()
    probe.controlPause()  // synchronous lock-store write — no Task, no actor hop
    let elapsed = Date().timeIntervalSince(start)

    XCTAssertTrue(probe.lockStorePaused, "sync control op took effect")
    XCTAssertLessThan(elapsed, 1.0, "sync control op must not wait on a cooperative thread")

    Thread.sleep(forTimeInterval: 0.3)  // XCTest thread — pool stays saturated
    XCTAssertFalse(coopRan.get(), "cooperative pool is genuinely saturated (contrast for the proof)")
  }

  /// The classifier's set: the sync-servable control routes, excluding the
  /// actor-backed CharacterStore and the genuinely-async internals.
  func testClassifierServesTheControlSetAndExcludesActorAndAsyncRoutes() {
    func servable(_ m: String, _ p: String) -> Bool { ControlPlaneClassifier.isSyncServable(method: m, path: p) }

    XCTAssertTrue(servable("POST", "/v1/queue/pause"))
    XCTAssertTrue(servable("POST", "/v1/queue/resume"))
    XCTAssertTrue(servable("POST", "/v1/queue/clear"))
    XCTAssertTrue(servable("POST", "/v1/queue/interrupt"))
    XCTAssertTrue(servable("POST", "/v1/queue/abc-123/move"))
    XCTAssertTrue(servable("DELETE", "/v1/queue/abc-123"))
    XCTAssertTrue(servable("GET", "/v1/queue"))
    XCTAssertTrue(servable("GET", "/v1/models"))
    XCTAssertTrue(servable("GET", "/v1/stats"))
    XCTAssertTrue(servable("GET", "/v1/config"))

    // Excluded: CharacterStore is an actor and cannot be read synchronously.
    XCTAssertFalse(servable("GET", "/v1/characters"))
    XCTAssertFalse(servable("POST", "/v1/characters"))
    XCTAssertFalse(servable("GET", "/v1/characters/abc"))
    XCTAssertFalse(servable("DELETE", "/v1/characters/abc"))
    // Excluded: genuinely-async internals (0.B-1's job).
    XCTAssertFalse(servable("GET", "/v1/civitai/search"))
    XCTAssertFalse(servable("POST", "/v1/civitai/harvest"))
    XCTAssertFalse(servable("POST", "/v1/enhance"))
    // Excluded: config WRITE stays async; render stays async.
    XCTAssertFalse(servable("PUT", "/v1/config"))
    XCTAssertFalse(servable("POST", "/v1/generate"))
  }

  // MARK: - comfybox#217: /health is sync-servable

  /// `/health` is what the Desktop queue/progress UI polls. It stopped hopping
  /// the coordinator actor when `LiveHealthState` landed, but it still entered
  /// `ConnectionHandler`'s `Task {}` — so the cooperative-pool saturation a
  /// blocking synchronous GPU render produces could still delay it. The whole
  /// payload is lock-based, so the WHOLE route is classified sync-servable
  /// (no `/v1/health/live` subset route was needed).
  func testHealthIsClassifiedSyncServable() {
    func servable(_ m: String, _ p: String) -> Bool { ControlPlaneClassifier.isSyncServable(method: m, path: p) }

    XCTAssertTrue(servable("GET", "/health"))
    // Only GET. A stray POST must still take the async path (and 404 there).
    XCTAssertFalse(servable("POST", "/health"))
    // The `/api` prefix is the BRIDGE's namespace and the bridge never claims
    // `/health`; the classifier matches the raw path, so this stays async.
    XCTAssertFalse(servable("GET", "/api/health"))
    XCTAssertFalse(servable("GET", "/healthz"))
  }

  /// The #217 claim itself: the progress-adjacent fields the Desktop polls are
  /// readable from the lock store WHILE the coordinator actor is occupied by a
  /// render. Drives the PRODUCTION payload assembly (`WarmServer.liveHealthPayload`,
  /// via the probe seam) against a real busy coordinator.
  func testHealthProgressFieldsAreReadableWhileTheCoordinatorIsBusy() async throws {
    let probe = makeQueueProbe()

    async let op: Bool = probe.enqueueSynthetic(durationMs: 1200, id: "busy")
    try await waitUntil("synthetic op running") { probe.activeJobSummary != nil }
    probe.publishProgress(42)

    // Synchronous call — no await, no actor hop — while the actor is busy.
    let json = probe.liveHealthJSON()
    XCTAssertEqual(json["is_rendering"] as? Bool, true, "render visible mid-render")
    XCTAssertEqual(json["progress_percent"] as? Int, 42)
    XCTAssertNotNil(json["pending_count"], "pending_count present")
    XCTAssertEqual(json["current_job_id"] as? String, "busy", "names the running job")

    _ = try await op
  }

  // MARK: - Scenario 6: flag-off byte-identical dispatch

  /// `COMFYBOX_CONTROL_PLANE_SYNC` defaults ON; "0" is the rollback lever that
  /// reverts to the pre-0.B-2 async dispatch path.
  func testSyncFlagDefaultsOnAndRevertsWithZero() {
    unsetenv("COMFYBOX_CONTROL_PLANE_SYNC")
    XCTAssertTrue(ControlPlaneSyncFlag.isEnabled, "default ON")
    setenv("COMFYBOX_CONTROL_PLANE_SYNC", "0", 1)
    XCTAssertFalse(ControlPlaneSyncFlag.isEnabled, "\"0\" reverts to the async path")
    setenv("COMFYBOX_CONTROL_PLANE_SYNC", "1", 1)
    XCTAssertTrue(ControlPlaneSyncFlag.isEnabled)
    unsetenv("COMFYBOX_CONTROL_PLANE_SYNC")
  }

  /// With the flag off the classifier is bypassed and pause flows through the
  /// async arm (`coordinator.setPaused`, the pre-0.B-2 path). It converges on the
  /// SAME authoritative state the sync path produces — so the wire response is
  /// byte-for-byte the same (and the reads go through the same shared payload
  /// builders regardless of path).
  func testFlagOffAsyncPauseConvergesOnTheSameAuthoritativeState() async throws {
    let probe = makeQueueProbe()
    setenv("COMFYBOX_CONTROL_PLANE_SYNC", "0", 1)
    defer { unsetenv("COMFYBOX_CONTROL_PLANE_SYNC") }
    XCTAssertFalse(ControlPlaneSyncFlag.isEnabled)

    await probe.setPaused(true)   // the async arm
    XCTAssertTrue(probe.lockStorePaused)
    XCTAssertTrue(probe.isPaused, "same is_paused the sync path would report")

    await probe.setPaused(false)
    XCTAssertFalse(probe.lockStorePaused)
  }

  // MARK: - Structural guard + store round-trip

  /// The compile-time half of the F1 wedge guard: the only constructors set
  /// `requiresWake = false`, so a wake-requiring command is unconstructable and
  /// cannot enter the mailbox (the drain asserts the same).
  func testQueueDeltasNeverRequireAWake() {
    XCTAssertFalse(QueueControlCommand.cancel("x").requiresWake)
    XCTAssertFalse(QueueControlCommand.move("x", direction: "up").requiresWake)
  }

  /// The sidecar round-trips a mix of cancel/move deltas through JSON.
  func testDeltaSidecarRoundTrips() {
    let deltas: [QueueControlCommand] = [.cancel("j1"), .move("j2", direction: "bottom"), .cancel("j3")]
    QueueDeltaStore.save(deltas)
    let loaded = QueueDeltaStore.load()
    XCTAssertEqual(loaded, deltas)
  }

  // MARK: - Adversarial review regressions (F-2, F-3)

  /// F-2: WAL ordering. The queue-deltas.json sidecar must survive until AFTER
  /// `persistQueueState()` writes canonical state — a kill in the window leaves
  /// the sidecar intact for replay (and replaying an applied cancel over the
  /// persisted state is a no-op), so a cancelled job can never resurrect. The
  /// crash-window hook fires between the canonical persist and the sidecar
  /// commit; the sidecar must still exist there.
  func testSidecarSurvivesUntilCanonicalStatePersists() async throws {
    let probe = makeQueueProbe()
    defer { QueueDeltaStore.drainCrashWindowHook = nil }

    probe.controlPause()  // between-items gate: enqueued jobs stay pending
    let jobA = Task { try await probe.enqueueSynthetic(durationMs: 30, id: "wal-a") }
    let jobB = Task { try await probe.enqueueSynthetic(durationMs: 30, id: "wal-b") }
    try await waitUntil("both jobs parked pending") {
      probe.snapshotPendingIds.contains("wal-a") && probe.snapshotPendingIds.contains("wal-b")
    }

    // Record WITHOUT the drain nudge so the delta deterministically sits
    // undrained — exactly the state a crash-before-drain leaves behind.
    probe.recordCancelDeltaOnly(id: "wal-a")
    XCTAssertTrue(FileManager.default.fileExists(atPath: QueueDeltaStore.path.path),
                  "recording a delta writes the sidecar")

    let sidecarAliveInWindow = LockedFlag()
    QueueDeltaStore.drainCrashWindowHook = {
      if FileManager.default.fileExists(atPath: QueueDeltaStore.path.path) {
        sidecarAliveInWindow.trySet()
      }
    }
    await probe.drainNow()

    XCTAssertTrue(sidecarAliveInWindow.get(),
      "sidecar intact AFTER persistQueueState, BEFORE the commit — a kill there replays instead of resurrecting")
    XCTAssertFalse(FileManager.default.fileExists(atPath: QueueDeltaStore.path.path),
                   "clean commit clears the sidecar")
    XCTAssertEqual(probe.undrainedDeltaCount, 0, "drained deltas dropped exactly once")
    XCTAssertFalse(probe.composedPendingIds.contains("wal-a"), "cancelled job stays cancelled")

    // The cancelled job resolves as a cancellation; the survivor completes.
    do { _ = try await jobA.value; XCTFail("wal-a should be cancelled") } catch {}
    probe.controlResume()
    let bFinished = try await jobB.value
    XCTAssertTrue(bFinished, "wal-b unaffected by the drained cancel")
  }

  /// F-3: the sync DELETE ACK is record-then-accept — 202, and it NEVER claims
  /// `deleted: true` (the loop may dequeue the job between the presence read
  /// and the drain). A cancel delta recorded for an already-STARTED job is a
  /// no-op: the render finishes and nothing pretended it was deleted.
  func testSyncCancelNeverClaimsDeletionAndSparesAStartedJob() async throws {
    // Response contract: accepted, honest note, no deletion claim.
    let ack = SyncCancelAccepted.ack(id: "j1")
    XCTAssertTrue(ack.accepted)
    XCTAssertEqual(ack.id, "j1")
    let json = String(decoding: try JSONEncoder().encode(ack), as: UTF8.self)
    XCTAssertFalse(json.contains("deleted"), "sync path must never claim deleted")
    XCTAssertTrue(json.contains("may start"), "ACK says the cancel may have raced a dequeue")

    // TOCTOU half: job starts before the delta drains -> it must run to completion.
    let probe = makeQueueProbe()
    let started = Task { try await probe.enqueueSynthetic(durationMs: 250, id: "raced") }
    try await waitUntil("synthetic op started") { probe.activeJobSummary != nil }

    probe.controlCancel(id: "raced")   // records the delta + fire-and-forget nudge
    await probe.drainNow()             // delta applies against pending — job is ACTIVE, not pending

    let finished = try await started.value
    XCTAssertTrue(finished, "started job completed — a cancel delta for an active id no-ops")
    try await waitUntil("delta consumed") { probe.undrainedDeltaCount == 0 }
  }

  // MARK: - comfybox#386: sidecar write must never block read()

  /// `LiveHealthState.recordDelta`/`commitDrainedDeltas` persist the
  /// undrained-delta sidecar (`queue-deltas.json`). Before the fix that write
  /// ran while holding the SAME `NSLock` `read()` needs — so a slow or
  /// nearly-full disk stalls the sync-servable `/health` and `/v1/queue`
  /// routes behind unrelated disk I/O. Block the sidecar writer mid-write
  /// (via `QueueDeltaStore.blockingWriteHook`, simulating that slow disk) and
  /// prove `read()` (driven here through `probe.isPaused`, which calls
  /// straight into `LiveHealthState.read()`) still returns within a tight
  /// bound instead of queuing behind the stuck writer.
  func testReadNeverBlocksBehindAStuckSidecarWrite() throws {
    let probe = makeQueueProbe()
    defer { QueueDeltaStore.blockingWriteHook = nil }

    let writerEntered = DispatchSemaphore(value: 0)
    let releaseWriter = DispatchSemaphore(value: 0)
    defer { releaseWriter.signal() }  // never leave the writer thread parked past this test
    QueueDeltaStore.blockingWriteHook = {
      writerEntered.signal()
      releaseWriter.wait()
    }

    // Off the test thread: record a delta. The in-memory mutation completes
    // immediately; the sidecar write it triggers blocks in the hook above,
    // simulating a disk stuck mid-write.
    Thread.detachNewThread {
      probe.recordCancelDeltaOnly(id: "comfybox-386-blocked-write")
    }
    XCTAssertEqual(writerEntered.wait(timeout: .now() + 5), .success,
                   "the sidecar writer must have entered the blocking hook")

    // read() must not queue behind the stuck writer — run it on its own
    // thread and bound the wait so a regression fails instead of hanging CI.
    let readDone = DispatchSemaphore(value: 0)
    let start = Date()
    Thread.detachNewThread {
      _ = probe.isPaused   // LiveHealthState.read() — the only thing /health and /v1/queue need
      readDone.signal()
    }

    XCTAssertEqual(readDone.wait(timeout: .now() + 0.5), .success,
                   "read() must return within a tight bound while the sidecar writer is stuck on disk")
    XCTAssertLessThan(Date().timeIntervalSince(start), 0.5,
                      "read() must not wait on the sidecar's disk write")
  }
}
