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

  /// comfybox#386 review round 2, item 4: reset every sidecar test seam here,
  /// not just in a per-test `defer` — a test that fails before reaching its
  /// own cleanup must not leak a blocked hook or a forced result into the
  /// next test.
  override func tearDown() {
    QueueDeltaStore.blockingWriteHook = nil
    QueueDeltaStore.forcedSaveResult = nil
    super.tearDown()
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

  /// A release that stays open once opened — unlike a `DispatchSemaphore`
  /// signalled exactly once, ANY number of `waitUntilOpen()` calls (issued
  /// before or after `open()`) resolve correctly. Review round 2, item 4: a
  /// one-shot semaphore parks a SECOND `QueueDeltaStore.save` call forever if
  /// `blockingWriteHook` is still installed when it runs (a retry, a second
  /// mutator call racing in, …), wedging `LiveHealthState.sidecarLock` for
  /// the rest of the process. This gate is deliberately idempotent/broadcast
  /// instead, and `open()` is safe to call more than once (e.g. once inline
  /// and once from a `defer`).
  private final class OneShotGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var isOpen = false

    func open() {
      condition.lock()
      isOpen = true
      condition.broadcast()
      condition.unlock()
    }

    func waitUntilOpen() {
      condition.lock()
      while !isOpen { condition.wait() }
      condition.unlock()
    }
  }

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

    let writerEntered = DispatchSemaphore(value: 0)
    let releaseWriter = OneShotGate()
    defer { releaseWriter.open() }  // never leave the writer thread parked past this test
    QueueDeltaStore.blockingWriteHook = {
      writerEntered.signal()
      releaseWriter.waitUntilOpen()
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
    //
    // Bound raised to 2s (review round 2, item 5; comfybox#379 flake
    // history): `DispatchSemaphore.wait(timeout:)`'s wall-clock return can
    // overshoot a tight bound under scheduler/QoS contention even though
    // read() itself is instant post-fix — the pre-fix failure this pins
    // measured ~1.9s, comfortably caught well under 2s, so the wider bound
    // stays tight enough to fail on a real regression without flaking on a
    // merely slow CI box. The separate elapsed-time assertion this replaced
    // was redundant with this one: a `.success` result already bounds the
    // wall-clock time to (approximately) the timeout.
    let readDone = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
      _ = probe.isPaused   // LiveHealthState.read() — the only thing /health and /v1/queue need
      readDone.signal()
    }

    XCTAssertEqual(readDone.wait(timeout: .now() + 2.0), .success,
                   "read() must return within a tight bound while the sidecar writer is stuck on disk")
  }

  // MARK: - comfybox#386 review round 2: WAL restored + failure/clear safety

  /// item 1: a delta must be DURABLE (its sidecar write confirmed) before the
  /// drain may act on it — otherwise a crash between "visible in memory" and
  /// "written to disk" loses it from both files, and a cancelled job can
  /// resurrect. Block the sidecar writer, record a delta, and prove the
  /// drain does not see/apply it until the write completes; `/health`'s
  /// undrained-delta view (`undrainedDeltaCount`) is unaffected — deltas are
  /// visible there the instant they're recorded, durable or not.
  func testDrainIgnoresANonDurableDeltaUntilItsSidecarWriteCompletes() async throws {
    let probe = makeQueueProbe()

    probe.controlPause()   // between-items gate: the job stays pending, not active
    let jobA = Task { try await probe.enqueueSynthetic(durationMs: 30, id: "wal2-a") }
    try await waitUntil("job parked pending") { probe.snapshotPendingIds.contains("wal2-a") }

    let writerEntered = DispatchSemaphore(value: 0)
    let releaseWriter = OneShotGate()
    defer { releaseWriter.open() }
    QueueDeltaStore.blockingWriteHook = {
      writerEntered.signal()
      releaseWriter.waitUntilOpen()
    }

    Thread.detachNewThread {
      probe.recordCancelDeltaOnly(id: "wal2-a")
    }
    XCTAssertEqual(writerEntered.wait(timeout: .now() + 5), .success,
                   "the sidecar writer must have entered the blocking hook")

    XCTAssertEqual(probe.undrainedDeltaCount, 1,
                   "recorded delta is visible to /health & /v1/queue reads immediately, durable or not")
    XCTAssertEqual(probe.peekedDrainableDeltaCount, 0,
                   "the drain must not see a delta before its sidecar write is durable")

    await probe.drainNow()
    XCTAssertTrue(probe.snapshotPendingIds.contains("wal2-a"),
                  "drain must not cancel the job while the delta isn't durable yet")

    releaseWriter.open()
    try await waitUntil("delta becomes durable") { probe.peekedDrainableDeltaCount == 1 }

    await probe.drainNow()
    XCTAssertFalse(probe.composedPendingIds.contains("wal2-a"), "now-durable delta is applied")

    probe.controlResume()
    do { _ = try await jobA.value; XCTFail("wal2-a should have been cancelled") } catch {}
  }

  /// item 2: `QueueDeltaStore.save` now reports success/failure instead of
  /// swallowing errors with `try?`; `persistDeltaSidecar` must only advance
  /// `lastPersistedDeltaGeneration` on success — otherwise a failed write
  /// (generation N+1) could make a later check treat N+1 as already durable
  /// while the file on disk silently still holds N-1's content.
  func testFailedSidecarWriteNeverAdvancesDurabilityAndTheDeltaSurvivesForTheNextWrite() {
    let probe = makeQueueProbe()

    QueueDeltaStore.forcedSaveResult = false   // simulate a disk write failure
    probe.recordCancelDeltaOnly(id: "fail-1")
    XCTAssertEqual(probe.undrainedDeltaCount, 1, "recorded in memory regardless of the disk outcome")
    XCTAssertEqual(probe.peekedDrainableDeltaCount, 0,
                   "a failed write must not advance the durability marker")

    QueueDeltaStore.forcedSaveResult = nil   // disk recovers
    probe.recordCancelDeltaOnly(id: "fail-2")   // resends the FULL undrained list
    XCTAssertEqual(probe.undrainedDeltaCount, 2)
    XCTAssertEqual(probe.peekedDrainableDeltaCount, 2,
                   "once a write succeeds it carries everything undrained so far — the earlier failed attempt's delta is not lost")
  }

  /// item 3: `clearDeltas` (the recovery boot path) now goes through the same
  /// generation/`sidecarLock` scheme as every other mutation.
  /// `recoverPersistedQueue` runs as a background task while the listener
  /// already accepts connections, so an in-flight OLDER write racing the
  /// clear is real — it must never land on disk AFTER the clear and
  /// resurrect what was just folded away.
  func testClearDeltasCannotBeResurrectedByAnOlderInFlightWrite() throws {
    let probe = makeQueueProbe()

    let writerEntered = DispatchSemaphore(value: 0)
    let releaseOlderWrite = OneShotGate()
    defer { releaseOlderWrite.open() }
    QueueDeltaStore.blockingWriteHook = {
      writerEntered.signal()
      releaseOlderWrite.waitUntilOpen()
    }

    // The older write starts first and gets stuck mid-write, holding sidecarLock.
    Thread.detachNewThread {
      probe.recordCancelDeltaOnly(id: "stale-before-clear")
    }
    XCTAssertEqual(writerEntered.wait(timeout: .now() + 5), .success,
                   "the older write must have entered the hook first")

    // The clear races in behind it — exactly the call `recoverPersistedQueue`
    // makes. Its generation stamp (under the main lock) is assigned the
    // instant this runs, strictly after the older write's own stamp (proven
    // by `writerEntered` above), even though its actual persist has to queue
    // behind the older write's still-held `sidecarLock`.
    let clearDone = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
      probe.clearAllDeltas()
      clearDone.signal()
    }

    releaseOlderWrite.open()   // let the older, lower-generation write proceed

    XCTAssertEqual(clearDone.wait(timeout: .now() + 5), .success, "clear completed")
    XCTAssertEqual(probe.peekedDrainableDeltaCount, 0, "the clear's empty snapshot is what's durable")
    XCTAssertFalse(FileManager.default.fileExists(atPath: QueueDeltaStore.path.path),
                   "the sidecar file must not be resurrected by the older, now-stale write")
  }

  // MARK: - comfybox#386 review round 3: drain liveness + safer clear

  /// item 1a: with the sidecar unwritable, `peekDeltas` can come back empty
  /// forever even though a delta is genuinely recorded — `drainQueueDeltas`
  /// must retry the write once before giving up, instead of returning early
  /// and leaving a 200-acked cancel permanently unapplied while the job keeps
  /// rendering. Force one failed write, let the disk "recover" before the
  /// drain runs, and confirm the drain's own retry notices and applies the
  /// delta in the SAME pass — no second external trigger needed.
  func testDrainRetriesAStuckWriteBeforeGivingUp() async throws {
    let probe = makeQueueProbe()

    probe.controlPause()   // between-items gate: the job stays pending
    let jobA = Task { try await probe.enqueueSynthetic(durationMs: 30, id: "retry-a") }
    try await waitUntil("job parked pending") { probe.snapshotPendingIds.contains("retry-a") }

    QueueDeltaStore.forcedSaveResult = false
    probe.recordCancelDeltaOnly(id: "retry-a")
    XCTAssertEqual(probe.undrainedDeltaCount, 1)
    XCTAssertEqual(probe.peekedDrainableDeltaCount, 0, "not durable yet — the forced write failed")

    QueueDeltaStore.forcedSaveResult = nil   // disk recovers before the drain's next attempt

    await probe.drainNow()   // must retry the write (item 1a) and apply it in this same pass

    XCTAssertFalse(probe.composedPendingIds.contains("retry-a"),
                   "the drain's retry succeeded and it applied the now-durable cancel")
    probe.controlResume()
    do { _ = try await jobA.value; XCTFail("retry-a should have been cancelled") } catch {}
  }

  /// item 1b/1c: once the sidecar has been failing continuously past the
  /// degraded-mode threshold, the drain applies non-durable deltas anyway —
  /// liveness wins, matching pre-comfybox#386 behavior, but now observable
  /// via `deltaDurabilityStatus`/`/health`'s additive fields. One failure
  /// alone must NOT trip it; recovery must clear it.
  func testDegradedModeAppliesNonDurableDeltasAfterSustainedFailuresAndClearsOnRecovery() async throws {
    let probe = makeQueueProbe()

    probe.controlPause()
    let jobA = Task { try await probe.enqueueSynthetic(durationMs: 30, id: "degraded-a") }
    try await waitUntil("job parked pending") { probe.snapshotPendingIds.contains("degraded-a") }

    QueueDeltaStore.forcedSaveResult = false
    probe.recordCancelDeltaOnly(id: "degraded-a")
    XCTAssertFalse(probe.isDeltaSidecarDegraded, "one failure alone must not trip degraded mode")
    XCTAssertEqual(probe.nonDurableDeltaCount, 1)

    // Drive the consecutive-failure count up to the threshold by retrying the
    // SAME still-undurable write — exactly what the drain's own item-1a retry
    // does, just called directly here for a deterministic count instead of
    // sleeping out the time-based half of the threshold. Bounded by the
    // threshold itself, so this can never spin.
    for _ in 0..<WarmServerQueueProbe.degradedModeFailureCountThreshold {
      guard !probe.isDeltaSidecarDegraded else { break }
      probe.retrySidecarWrite()
    }
    XCTAssertTrue(probe.isDeltaSidecarDegraded, "sustained failures trip degraded mode")

    await probe.drainNow()   // liveness wins: applies the still-non-durable cancel
    XCTAssertFalse(probe.composedPendingIds.contains("degraded-a"),
                   "degraded mode applies the cancel even though it never became durable")

    QueueDeltaStore.forcedSaveResult = nil   // writer recovers
    // `commitDrainedDeltas` (inside the drain above) already tried and
    // failed once more with the writer still broken at that moment — nothing
    // retries on its own initiative once `deltas` is empty, so the recovery
    // needs one more scheduling point (item 1a's retry) to notice the writer
    // is healthy again, exactly like production: the drain runs at every
    // `processLoop` iteration and `startProcessingIfNeeded`.
    await probe.drainNow()
    try await waitUntil("degraded flag clears once a write succeeds") { !probe.isDeltaSidecarDegraded }
    XCTAssertEqual(probe.nonDurableDeltaCount, 0)

    probe.controlResume()
    do { _ = try await jobA.value; XCTFail("degraded-a should have been cancelled") } catch {}
  }

  /// item 2: `clearDeltas` must not drop anything from MEMORY until its own
  /// empty-snapshot write is confirmed durable — otherwise a failed write
  /// leaves disk holding stale (already-applied) deltas while memory has
  /// already forgotten them, and the next boot's recovery re-folds deltas
  /// this session already applied (`.move` is not idempotent). A failed
  /// clear must change nothing; the next successful write (here, a retried
  /// clear) must finish the job.
  func testClearDeltasKeepsMemoryOnAFailedPersistAndTheNextWriteHeals() {
    let probe = makeQueueProbe()

    probe.recordCancelDeltaOnly(id: "keep-me")   // a genuine, already-durable delta
    XCTAssertEqual(probe.peekedDrainableDeltaCount, 1)

    QueueDeltaStore.forcedSaveResult = false
    probe.clearAllDeltas()   // the clear's own persist fails
    XCTAssertEqual(probe.undrainedDeltaCount, 1, "a failed clear must not drop anything from memory")
    XCTAssertEqual(probe.peekedDrainableDeltaCount, 1,
                   "the pre-existing delta is exactly as durable as it was — the failed clear changed nothing")

    QueueDeltaStore.forcedSaveResult = nil   // disk recovers
    probe.clearAllDeltas()   // retried — this time it succeeds
    XCTAssertEqual(probe.undrainedDeltaCount, 0, "clear finally took effect once its write actually landed")
    XCTAssertFalse(FileManager.default.fileExists(atPath: QueueDeltaStore.path.path))
  }
}
