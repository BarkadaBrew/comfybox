# Inference Slot (Engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ComfyBox grants leased, top-priority "inference slots" so a Glimmer call can hold the GPU: no render starts while a slot is held, and an in-flight LTX-2 video parks at its next step boundary until the slots release.

**Architecture:** A lock-based `InferenceSlotTable` (TTL leases, waiters) and an `InferenceHold` state machine live beside the existing #1479 preemption state in `WarmServer`. Slot routes are served on the sync control plane. The job loop treats a held slot like a pause. `runPreemptionEpisode` gains an inference-hold branch that parks the checkpointed video (weights stay resident) until the table is empty, then resumes it.

**Tech Stack:** Swift 5.9, mlx-swift, XCTest; `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/<Class>`.

**Spec:** `docs/FDD-glimmer-gpu-slot.md` (§3.1, §5)

## Global Constraints

- Agents run unit tests only (`-only-testing:ZImageTests/...`); integration/E2E need weights and are run by Todd (intent.md).
- The daemon contract is production: new routes are additive; no existing route changes shape (intent.md).
- One `xcodebuild`/`swift build` at a time on this Mac; macOS 27 needs the MetalToolchain component (already downloaded).
- TTL default 180 s, max 900 s. Waiter poll/timers never block the actor.
- Route set changes must keep `ControlSurfaceParityTests` green: dispatch-arm pins, MCP claim or `ParityExemptions` reason, regenerated `docs/api-reference.md`.

---

### Task E1: `InferenceSlotTable`

**Files:**
- Create: `Sources/ZImage/Server/InferenceSlots.swift`
- Test: `Tests/ZImageTests/Server/InferenceSlotTableTests.swift`

**Interfaces — Produces:**
```swift
public struct InferenceSlot: Sendable, Equatable { public let id: String; public let holder: String; public let acquiredAt: Date; public var expiresAt: Date }
public final class InferenceSlotTable: @unchecked Sendable {
  public static let defaultTTL: TimeInterval = 180
  public static let maxTTL: TimeInterval = 900
  public init(now: @escaping @Sendable () -> Date = Date.init)
  public var onEmpty: (@Sendable () -> Void)?            // fired once each time the table goes non-empty → empty
  public func acquire(holder: String, ttl: TimeInterval?) -> InferenceSlot
  public func renew(id: String, ttl: TimeInterval?) -> InferenceSlot?
  @discardableResult public func release(id: String) -> Bool
  public func get(id: String) -> InferenceSlot?          // nil when unknown or expired
  public func activeSlots() -> [InferenceSlot]            // sweeps expired first
  public func isHeld() -> Bool
  public func sweep()                                    // drops expired, fires onEmpty/waiters if it emptied
  public func waitUntilFree() async                      // returns immediately when empty; cancellation-aware
}
```

- [ ] **Step 1: Failing tests** — acquire/get/release; TTL clamp (nil→180, 5000→900, 0→1); expiry makes `get` nil and `isHeld` false via an injected clock; `renew` extends; `onEmpty` fires exactly once on last release and on sweep-expiry; `waitUntilFree` returns immediately when empty, resumes after last release, and returns on task cancellation.
- [ ] **Step 2: Run, expect compile failure** (`cannot find 'InferenceSlotTable'`).
- [ ] **Step 3: Implement** with `NSLock`, `[String: InferenceSlot]`, `[UUID: CheckedContinuation<Void, Never>]` waiters resumed outside the lock, `withTaskCancellationHandler` removing a waiter on cancel, and `DispatchQueue.global().asyncAfter(deadline: expiry)` scheduling `sweep()` on acquire/renew.
- [ ] **Step 4: Run tests, expect PASS.**
- [ ] **Step 5: Commit** `feat(engine): InferenceSlotTable — leased top-priority inference slots`.

### Task E2: `InferenceHold` + grant state

**Files:**
- Modify: `Sources/ZImage/Server/InferenceSlots.swift`
- Test: `Tests/ZImageTests/Server/InferenceHoldTests.swift`

**Interfaces — Produces:**
```swift
public final class InferenceHold: @unchecked Sendable {
  public init()
  public func request() -> Bool          // idle → requested; false if not idle
  public func takeRequest() -> Bool      // requested → parked (the episode claims it)
  public func cancelIfRequested() -> Bool// requested → idle (watchdog / table emptied)
  public func unpark()                   // parked → idle
  public var isRequested: Bool { get }
  public var isParked: Bool { get }
}
public enum InferenceSlotState: String, Sendable { case granted, preempting, waiting }
public enum InferenceSlotGrant {
  public static func state(isParked: Bool, isRequested: Bool, gpuBusy: Bool) -> InferenceSlotState
  public static func etaSec(progressPct: Int?, renderStartedAt: Date?, now: Date) -> Double?
}
```
Rules: parked → granted; !gpuBusy → granted; requested → preempting; else waiting. ETA = elapsed·(100−pct)/pct for 1 ≤ pct ≤ 99, else nil.

- [ ] Steps 1–5 as E1 (tests cover every transition incl. illegal ones returning false, and the ETA edge cases). Commit `feat(engine): InferenceHold state machine + slot grant state`.

### Task E3: Job loop honours held slots

**Files:**
- Modify: `Sources/ZImage/Server/WarmServer.swift` (`WarmServerCoordinator.init` params; the dequeue gate at `if liveHealth.isPausedAuthoritative()`; `wakeAfterAuthoritativeResume`; `WarmServer.init` wiring + `onEmpty` → `Task { await coordinator.wakeAfterInferenceRelease() }`).
- Test: `Tests/ZImageTests/Server/InferenceSlotAdmissionTests.swift` using the existing coordinator test surface (`enqueueFakeRender`, `activeJobId`, `pendingCount`).

- [ ] **Step 1: Failing test** — acquire a slot on the server's table, enqueue a fake render, assert it stays pending (not active) for 300 ms; release the slot, assert the fake render runs and completes. Second test: a manual pause + slot release leaves the render parked (manual pause wins).
- [ ] **Step 2: Run, expect FAIL** (render starts despite the slot).
- [ ] **Step 3: Implement** — gate becomes `if liveHealth.isPausedAuthoritative() || inferenceSlots.isHeld()`; add `func wakeAfterInferenceRelease()` = `guard !liveHealth.isPausedAuthoritative(), !inferenceSlots.isHeld() else { return }; startProcessingIfNeeded(); publishHealth()`.
- [ ] **Step 4: PASS**, plus rerun `ControlPlaneTests` and `InterruptTargetTests` (no regressions).
- [ ] **Step 5: Commit** `feat(engine): job loop parks renders while an inference slot is held`.

### Task E4: Slot routes, health fields, parity

**Files:**
- Modify: `Sources/ZImage/Server/ControlPlane.swift` (`isSyncServable`: `POST/GET /v1/queue/inference-slot`, `GET/DELETE /v1/queue/inference-slot/{id}`, `POST …/{id}/renew`), `WarmServer.swift` (sync + async arms calling shared `inferenceSlotResponse(request:)`; `/health` + `/v1/queue` gain `inference_slots`), `Sources/ZImage/MCP/ParityExemptions.swift` (reason: "daemon-internal GPU slot protocol; not an operator action"), `docs/api-reference.md` (regenerated), `Tests/ZImageTests/ControlSurfaceParityTests.swift` pins.
- Test: `Tests/ZImageTests/Server/InferenceSlotRouteTests.swift` (pure request/response builder: `InferenceSlotRoute.handle(method:path:body:table:hold:gpuBusy:progress:renderStartedAt:now:raise:) -> (status: Int, json: [String: Any])`).

Acquire flow inside `handle` for `POST /v1/queue/inference-slot`: decode `{holder, ttl_sec}` (400 without holder) → `table.acquire` → if `gpuBusy && videoRendering && !hold.isParked` call `raise()` (the server closure: `preemptionInFlight.trySet()` → `hold.request()` → `ltx2PreemptionSignal.raise()` → watchdog after `meanStepSec*2+30 ?? 120` s: `if hold.cancelIfRequested() { signal.clear(); preemptionInFlight.clear() }`) → respond `SlotStatus`. `onEmpty` additionally cancels a still-pending request the same way.

- [ ] **Step 1: Failing route tests** — 400 without holder; granted when idle; waiting + eta when an image render is at 50 %; preempting after raise with video rendering; renew/404; DELETE releases and reports active count; list route.
- [ ] **Step 2: FAIL.** **Step 3: Implement** builder + both dispatch arms + classifier + health fields + exemptions; regenerate docs with the repo's generator (`ControlSurfaceParityTests` names it); update pins.
- [ ] **Step 4: PASS** `InferenceSlotRouteTests`, `ControlSurfaceParityTests`, `ControlPlaneTests`.
- [ ] **Step 5: Commit** `feat(engine): /v1/queue/inference-slot routes on the sync control plane`.

### Task E5: Inference hold parks the in-flight video

**Files:**
- Modify: `WarmServer.swift` `runPreemptionEpisode` — before `guard let claimed = pendingPreemptorBox.claim()`:
```swift
if inferenceHold.takeRequest() {
  logger.info("inference hold: video checkpointed at chunk \(state.chunkIndex), step \(state.stepIndex) — GPU yielded to inference slot(s); weights stay resident")
  if let jobId = activeJobId {
    lifecycleLedger.record(jobId: jobId, kind: .checkpointed, jobKind: QueueJobKind.video.rawValue,
                           step: state.stepIndex, chunk: state.chunkIndex, reason: "inference-hold")
  }
  publishHealth()
  await inferenceSlots.waitUntilFree()
  inferenceHold.unpark()
  let disposition = LTX2PreemptionEpisode.disposition(videoInterrupted: Task.isCancelled)
  clearEpisodeState(abandoned: disposition == .abandonVideo)
  if disposition == .abandonVideo { throw CancellationError() }
  logger.info("inference hold released — resuming LTX-2 video")
  return try await resumeCheckpointedVideo(state: state, wantsAudio: wantsAudio, report: report)
}
```
- Test: extend `LTX2PreemptionTests`/`LocalVideoInterruptTests` only if their fakes can drive `runPreemptionEpisode`; otherwise a pure test of the ordering helper is not meaningful — record it as **integration-only** (Todd runs a real video + slot).

- [ ] **Step 1:** Inspect those test files for a fake video body that returns `.yielded`. If present, write the failing test (hold requested → episode parks → release → resume called once, generator not released). If absent, skip to Step 3 and mark integration-only in the PR.
- [ ] **Step 3: Implement** as above.
- [ ] **Step 4:** Run `LTX2PreemptionTests`, `LocalVideoInterruptTests`, `InterruptTargetTests` — all PASS.
- [ ] **Step 5: Commit** `feat(engine): inference hold parks an in-flight LTX-2 video until slots release`.

### Task E6 (after comfybox #464 merges): engine-local Glimmer callers

`MCPToolExecutor.diagnoseDefects` and desktop `VisionService.describe`/`AgentService` acquire a slot around their request when the resolved provider URL is a Glimmer endpoint (`VisionChat.isGlimmerEndpoint`: host:port in `COMFYBOX_GLIMMER_ENDPOINTS`, default `127.0.0.1:11234,localhost:11234,10.0.100.134:11234`). Separate PR, blocked on #464.
