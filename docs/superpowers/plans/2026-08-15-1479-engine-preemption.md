# #1479 Engine-Side LTX-2 Render Preemption — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let ComfyBox pause an in-flight LTX-2 render at a step boundary, run a preempting job, and resume bit-identically — so a T1 tap waits seconds, not minutes.

**Architecture:** Checkpoint-and-unwind, not suspend. A lock-protected `PreemptionSignal` (the #217 no-actor-hop pattern) is checked at the top of each denoise step; on signal the loop returns `.yielded(LTX2ResumeState)` carrying every non-weight tensor. The coordinator conditionally evicts weights, runs the preemptor, reloads, and re-enters the loop at step N. Seeded runs derive per-step RNG keys so resume is bit-identical. Phase telemetry lands first and feeds the refusal guard.

**Tech Stack:** Swift 5.9 / mlx-swift, XCTest via `xcodebuild`. Repo `/Users/toddwalderman/Projects/zimage.swift`, branch `claude/1479-engine-preemption`.

**Spec:** `docs/superpowers/specs/2026-08-15-1479-engine-preemption-design.md` — read it first. The spec is the authority; this plan argues from it.

## Global Constraints

- ComfyBox is self-standing Swift/MLX. No Python anywhere.
- **The running server on :7870 serves production.** Build and unit-test freely; NEVER restart the serving process, run `ComfyBox serve`, or deploy. Integration tests that load LTX-2 weights (~38GB) must not run while `curl -s localhost:7870/v1/queue` shows `is_rendering: true`.
- Checkpoint = **all non-weight tensors**; evict = **weights only** (spec, Fable review). The enumerated list in the spec's `LTX2ResumeState` section is normative.
- Per-step RNG keys apply to **seeded runs only**; base constant `0xD0D10` (video), distinct from audio's `0xA0D10/11/12`. Unseeded runs keep the global stream.
- The no-preemption path must be unchanged: no measurable per-step cost, no behavior change for unseeded noise, byte-identical seeded output *when the old global-stream path is selected* is NOT required (seeded noise sequence changes once, accepted by Todd 2026-08-15).
- `/v1/queue` and job-status JSON changes are ADDITIVE only.
- The refusal guard is **inert until telemetry has samples** — it never refuses on a guess.
- Nested preemption refused; checkpoint in-memory only (dies with the process).
- Unit tests: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/<Class>`
- Commit after every task; prefix `feat(ltx2): #1479 — <task>` (or `feat(server):` for WarmServer tasks).
- Verify `git show --stat HEAD` touches ONLY intended files before reporting.

## File Structure

- Create `Sources/ZImage/LTX2/LTX2PhaseTelemetry.swift` — per-phase timing recorder + published view. One responsibility: measure.
- Create `Sources/ZImage/LTX2/LTX2Preemption.swift` — `PreemptionSignal`, `LTX2ResumeState`, `LTX2DenoiseResult`, `ancestralVideoNoise(...)`. One responsibility: the preemption vocabulary.
- Modify `Sources/ZImage/LTX2/LTX2Pipeline.swift` — `denoisingLoop` gains signal check, `startStep`, keyed noise, yielding return type.
- Modify `Sources/ZImage/LTX2/LTX2VideoGenerator.swift` — threads the signal, produces/consumes `LTX2ResumeState`, reports phases.
- Modify `Sources/ZImage/Server/WarmServer.swift` — `preempt` request flag, coordinator orchestration, refusal guard, `/v1/queue` telemetry fields.
- Tests: `Tests/ZImageTests/LTX2PhaseTelemetryTests.swift`, `Tests/ZImageTests/LTX2PreemptionTests.swift` (unit, no weights); `Tests/ZImageIntegrationTests/LTX2PreemptionResumeTests.swift` (weights required).

---

### Task 1: `LTX2PhaseTelemetry` — measure before you build

**Files:**
- Create: `Sources/ZImage/LTX2/LTX2PhaseTelemetry.swift`
- Test: `Tests/ZImageTests/LTX2PhaseTelemetryTests.swift`

**Interfaces:**
- Consumes: nothing (pure, clock injected).
- Produces (Tasks 4/5 rely on these exact names):
  - `enum LTX2Phase: String, CaseIterable, Sendable { case modelLoad, textEncode, baseDenoise, refineDenoise, vaeDecode, vocoder, postProcess }`
  - `final class LTX2PhaseTelemetry: @unchecked Sendable` with
    `func begin(_ phase: LTX2Phase, nowMs: Double)`,
    `func end(_ phase: LTX2Phase, nowMs: Double)`,
    `func recordStep(seconds: Double)` (denoise per-step samples),
    `func view() -> LTX2PhaseTelemetryView`
  - `struct LTX2PhaseTelemetryView: Sendable { let phases: [String: (meanSec: Double, samples: Int)]; let meanStepSec: Double?; let maxUninterruptibleSec: Double?; let currentPhase: String? }`
  - `maxUninterruptibleSec` = the largest phase mean among phases with samples, EXCLUDING `baseDenoise`/`refineDenoise` (those yield per step; their uninterruptible unit is `meanStepSec`). Nil until any non-denoise phase has a sample.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
@testable import ZImage

final class LTX2PhaseTelemetryTests: XCTestCase {
  func testPhaseMeanAndSamples() {
    let t = LTX2PhaseTelemetry()
    t.begin(.vaeDecode, nowMs: 1_000); t.end(.vaeDecode, nowMs: 31_000)   // 30s
    t.begin(.vaeDecode, nowMs: 40_000); t.end(.vaeDecode, nowMs: 60_000)  // 20s
    let v = t.view()
    XCTAssertEqual(v.phases["vaeDecode"]?.samples, 2)
    XCTAssertEqual(v.phases["vaeDecode"]!.meanSec, 25.0, accuracy: 0.001)
  }

  func testMaxUninterruptibleExcludesDenoisePhases() {
    let t = LTX2PhaseTelemetry()
    t.begin(.baseDenoise, nowMs: 0); t.end(.baseDenoise, nowMs: 600_000)  // 600s, must NOT count
    t.begin(.vaeDecode, nowMs: 0); t.end(.vaeDecode, nowMs: 45_000)       // 45s
    t.begin(.vocoder, nowMs: 0); t.end(.vocoder, nowMs: 12_000)           // 12s
    XCTAssertEqual(t.view().maxUninterruptibleSec!, 45.0, accuracy: 0.001)
  }

  func testNilUntilSampled() {
    let t = LTX2PhaseTelemetry()
    XCTAssertNil(t.view().maxUninterruptibleSec)
    XCTAssertNil(t.view().meanStepSec)
    t.begin(.baseDenoise, nowMs: 0)   // begun but not ended
    XCTAssertEqual(t.view().currentPhase, "baseDenoise")
    XCTAssertNil(t.view().maxUninterruptibleSec)
  }

  func testStepSamples() {
    let t = LTX2PhaseTelemetry()
    t.recordStep(seconds: 2.0); t.recordStep(seconds: 4.0)
    XCTAssertEqual(t.view().meanStepSec!, 3.0, accuracy: 0.001)
  }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/LTX2PhaseTelemetryTests 2>&1 | tail -20`
Expected: FAIL — `LTX2PhaseTelemetry` not defined.

- [ ] **Step 3: Implement**

```swift
import Foundation

/// #1479: per-phase render timings. Feeds the preemption refusal guard and
/// /v1/queue observability. Lock-protected (NOT an actor) so the render loop
/// can record with no actor hop — same pattern as LiveHealthState (#217).
public enum LTX2Phase: String, CaseIterable, Sendable {
  case modelLoad, textEncode, baseDenoise, refineDenoise, vaeDecode, vocoder, postProcess
}

public struct LTX2PhaseTelemetryView: Sendable {
  public let phases: [String: (meanSec: Double, samples: Int)]
  public let meanStepSec: Double?
  public let maxUninterruptibleSec: Double?
  public let currentPhase: String?
}

public final class LTX2PhaseTelemetry: @unchecked Sendable {
  private let lock = NSLock()
  private var totals: [LTX2Phase: (sumSec: Double, samples: Int)] = [:]
  private var open: [LTX2Phase: Double] = [:]
  private var current: LTX2Phase?
  private var stepSumSec = 0.0
  private var stepSamples = 0

  public init() {}

  public func begin(_ phase: LTX2Phase, nowMs: Double = Date().timeIntervalSince1970 * 1000) {
    lock.lock(); defer { lock.unlock() }
    open[phase] = nowMs; current = phase
  }

  public func end(_ phase: LTX2Phase, nowMs: Double = Date().timeIntervalSince1970 * 1000) {
    lock.lock(); defer { lock.unlock() }
    guard let started = open.removeValue(forKey: phase) else { return }
    let prev = totals[phase] ?? (0, 0)
    totals[phase] = (prev.sumSec + (nowMs - started) / 1000.0, prev.samples + 1)
    if current == phase { current = nil }
  }

  public func recordStep(seconds: Double) {
    lock.lock(); defer { lock.unlock() }
    stepSumSec += seconds; stepSamples += 1
  }

  public func view() -> LTX2PhaseTelemetryView {
    lock.lock(); defer { lock.unlock() }
    var phases: [String: (meanSec: Double, samples: Int)] = [:]
    for (p, t) in totals where t.samples > 0 {
      phases[p.rawValue] = (t.sumSec / Double(t.samples), t.samples)
    }
    let denoise: Set<LTX2Phase> = [.baseDenoise, .refineDenoise]
    let maxUninterruptible = totals
      .filter { !denoise.contains($0.key) && $0.value.samples > 0 }
      .map { $0.value.sumSec / Double($0.value.samples) }
      .max()
    return LTX2PhaseTelemetryView(
      phases: phases,
      meanStepSec: stepSamples > 0 ? stepSumSec / Double(stepSamples) : nil,
      maxUninterruptibleSec: maxUninterruptible,
      currentPhase: current?.rawValue)
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Same command as Step 2. Expected: PASS, 4/4.

- [ ] **Step 5: Commit**

```bash
git add Sources/ZImage/LTX2/LTX2PhaseTelemetry.swift Tests/ZImageTests/LTX2PhaseTelemetryTests.swift
git commit -m "feat(ltx2): #1479 — phase telemetry: per-phase timings, meanStepSec, maxUninterruptibleSec"
```

---

### Task 2: Preemption vocabulary — signal, resume state, keyed noise

**Files:**
- Create: `Sources/ZImage/LTX2/LTX2Preemption.swift`
- Test: `Tests/ZImageTests/LTX2PreemptionTests.swift`

**Interfaces:**
- Consumes: `MLXRandom` (mlx-swift), `LTX2Phase` (Task 1).
- Produces (Tasks 3/4/5 rely on these exact names):
  - `final class PreemptionSignal: @unchecked Sendable` — `func raise()`, `func clear()`, `var isRaised: Bool`
  - `public let ltx2VideoNoiseKeyBase: UInt64 = 0xD0D10`
  - `func ancestralVideoNoise(shape: [Int], seed: UInt64?, step: Int) -> MLXArray`
  - `enum LTX2DenoiseResult { case completed(MLXArray); case yielded(LTX2ResumeState) }`
  - `struct LTX2ResumeState` with fields:
    `videoLatents: MLXArray`, `stepIndex: Int`, `sigmas: [Float]`,
    `phase: LTX2Phase` (`.baseDenoise` or `.refineDenoise`), `chunkIndex: Int`,
    `seed: UInt64?`, `audioLatents: MLXArray?`, `audioNoiseKey: MLXArray?`,
    `configFingerprint: String`

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
import MLX
import MLXRandom
@testable import ZImage

final class LTX2PreemptionTests: XCTestCase {
  func testSignalRaiseClear() {
    let s = PreemptionSignal()
    XCTAssertFalse(s.isRaised)
    s.raise(); XCTAssertTrue(s.isRaised)
    s.clear(); XCTAssertFalse(s.isRaised)
  }

  func testSeededNoiseIsDeterministicPerStep() {
    let a = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 7)
    let b = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 7)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self), "same seed+step must be identical")
    let c = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 8)
    XCTAssertFalse(MLX.allClose(a, c).item(Bool.self), "different step must differ")
    let d = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 43, step: 7)
    XCTAssertFalse(MLX.allClose(a, d).item(Bool.self), "different seed must differ")
  }

  func testSeededNoiseIgnoresGlobalStreamPosition() {
    MLXRandom.seed(1)
    let a = ancestralVideoNoise(shape: [2, 2], seed: 9, step: 3)
    MLXRandom.seed(999)
    _ = MLXRandom.normal([16])   // scramble the global stream
    let b = ancestralVideoNoise(shape: [2, 2], seed: 9, step: 3)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self),
      "seeded draw must be independent of global stream position — this IS the resume guarantee")
  }

  func testUnseededNoiseUsesGlobalStream() {
    MLXRandom.seed(7)
    let a = ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0)
    MLXRandom.seed(7)
    let b = ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self),
      "unseeded path must remain the plain global stream (unchanged behaviour)")
  }

  func testNoiseDtypeIsFloat32() {
    XCTAssertEqual(ancestralVideoNoise(shape: [2, 2], seed: 1, step: 0).dtype, .float32)
    XCTAssertEqual(ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0).dtype, .float32)
  }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/LTX2PreemptionTests 2>&1 | tail -20`
Expected: FAIL — symbols not defined.

- [ ] **Step 3: Implement**

```swift
import Foundation
import MLX
import MLXRandom

/// #1479 preemption vocabulary. See docs/superpowers/specs/2026-08-15-1479-…
///
/// PreemptionSignal is lock-protected, NOT an actor: the coordinator actor is
/// blocked for the whole synchronous GPU render (WarmServer.swift:4216), so
/// the render loop must read this with no actor hop — the #217 pattern.
public final class PreemptionSignal: @unchecked Sendable {
  private let lock = NSLock()
  private var raised = false
  public init() {}
  public func raise() { lock.lock(); raised = true; lock.unlock() }
  public func clear() { lock.lock(); raised = false; lock.unlock() }
  public var isRaised: Bool { lock.lock(); defer { lock.unlock() }; return raised }
}

/// Base constant for per-step video noise keys. Distinct from the audio
/// path's 0xA0D10/11/12 (LTX2Pipeline.swift:288 area) so streams never collide.
public let ltx2VideoNoiseKeyBase: UInt64 = 0xD0D10

/// Ancestral/SDE noise for the video stream.
/// Seeded: per-step derived key -> independent of global stream position,
/// which is exactly what makes bit-identical resume possible.
/// Unseeded: plain global stream, byte-for-byte the pre-#1479 behaviour.
public func ancestralVideoNoise(shape: [Int], seed: UInt64?, step: Int) -> MLXArray {
  if let seed {
    let key = MLXRandom.key(seed &+ ltx2VideoNoiseKeyBase &+ UInt64(step))
    return MLXRandom.normal(shape, key: key).asType(.float32)
  }
  return MLXRandom.normal(shape, dtype: .float32)
}

public enum LTX2DenoiseResult {
  case completed(MLXArray)
  case yielded(LTX2ResumeState)
}

/// Checkpoint = ALL non-weight tensors (spec rule). In-memory only —
/// deliberately dies with the process, unlike the isPaused sentinel.
public struct LTX2ResumeState {
  public var videoLatents: MLXArray
  public var stepIndex: Int
  public var sigmas: [Float]
  public var phase: LTX2Phase        // .baseDenoise or .refineDenoise
  public var chunkIndex: Int
  public var seed: UInt64?
  public var audioLatents: MLXArray?
  public var audioNoiseKey: MLXArray?
  public var configFingerprint: String
}
```

- [ ] **Step 4: Run tests to verify they pass**

Same command as Step 2. Expected: PASS, 5/5.

- [ ] **Step 5: Commit**

```bash
git add Sources/ZImage/LTX2/LTX2Preemption.swift Tests/ZImageTests/LTX2PreemptionTests.swift
git commit -m "feat(ltx2): #1479 — PreemptionSignal, LTX2ResumeState, per-step keyed ancestral noise"
```

---

### Task 3: `denoisingLoop` — yield at the boundary, resume at step N, keyed noise

**Files:**
- Modify: `Sources/ZImage/LTX2/LTX2Pipeline.swift` (the `private func denoisingLoop(...)` around line ~996; the step loop `for i in 0..<numSteps` at ~1057; the two `MLXRandom.normal(currentLatents.shape, dtype: .float32)` draws at ~1300 and ~1310). **Locate by symbol, not line number.**
- Test: covered by Task 2's unit tests (noise) + Task 6's integration tests (loop). This task's own gate is: compiles, all existing unit tests still green.

**Interfaces:**
- Consumes: `PreemptionSignal`, `ancestralVideoNoise`, `LTX2DenoiseResult`, `LTX2ResumeState`, `LTX2Phase` (Task 2), `LTX2PhaseTelemetry.recordStep` (Task 1).
- Produces (Task 4 relies on): `denoisingLoop` returns `LTX2DenoiseResult`, and accepts
  `startStep: Int = 0`, `seed: UInt64?`, `preemption: PreemptionSignal? = nil`,
  `telemetry: LTX2PhaseTelemetry? = nil`, `loopPhase: LTX2Phase`, `chunkIndex: Int = 0`.

- [ ] **Step 1: Change the signature and loop entry**

In `denoisingLoop`, add the new parameters and change the return type to `LTX2DenoiseResult`. Change the loop header from `for i in 0..<numSteps` to `for i in startStep..<numSteps`, and add the yield check at the very top of the loop body, BEFORE the input-side conditioning clamp:

```swift
    for i in startStep..<numSteps {
      // #1479: yield at the step boundary. Checked first so a raised signal
      // costs zero model passes. currentLatents at this point is the
      // end-of-step-(i-1) state — exactly what resume needs to re-enter at i.
      if let p = preemption, p.isRaised {
        return .yielded(LTX2ResumeState(
          videoLatents: currentLatents,
          stepIndex: i,
          sigmas: sigmas,
          phase: loopPhase,
          chunkIndex: chunkIndex,
          seed: seed,
          audioLatents: avState?.audioLatents,
          audioNoiseKey: avState?.audioNoiseKey,
          configFingerprint: resolvedConfig.fingerprint))
      }
      let stepStart = Date().timeIntervalSince1970
      let sigma = sigmas[i]
      ...
```

At the very bottom of the loop body (after the audio step), add:

```swift
      telemetry?.recordStep(seconds: Date().timeIntervalSince1970 - stepStart)
```

If `resolvedConfig` has no `fingerprint` property, add one: a computed `String` concatenating width/height/frames/steps/sampler/cfg — e.g. `"\(width)x\(height)f\(frames)s\(sigmas.count - 1)-\(samplerName)-cfg\(cfgScale)"`. Its only job is to make a mismatched resume detectable (Task 4 compares it).

- [ ] **Step 2: Replace the two global-stream noise draws**

Both SDE branches draw noise once per step. Replace exactly these two (CFG++ ancestral branch, then plain SDE branch):

```swift
// BEFORE (both sites):
let noise = MLXRandom.normal(currentLatents.shape, dtype: .float32)

// AFTER (both sites):
let noise = ancestralVideoNoise(shape: currentLatents.shape, seed: seed, step: i)
```

Do NOT touch the audio noise draw (`av.audioNoiseKey` chain) — it is already keyed and already resume-safe.

- [ ] **Step 3: Return `.completed` and fix the two call sites**

Change the final `return latents`-style return to `return .completed(currentLatents)`. The base pass (~line 313 region) and refine pass (~line 392 region) call sites unwrap:

```swift
let result = denoisingLoop(..., seed: seed, preemption: preemption,
                           telemetry: telemetry, loopPhase: .baseDenoise, chunkIndex: chunkIndex)
switch result {
case .yielded(let state): return .yielded(state)   // propagate up (Task 4 shapes the enclosing return)
case .completed(let latents): /* existing flow continues with `latents` */
}
```

For THIS task, the enclosing `generateT2V`/`generateI2V` may simply `fatalError("unreachable")` on `.yielded` when `preemption == nil` is passed everywhere — Task 4 threads the real parameter. The tree must compile with all existing callers passing no new arguments (all new params have defaults except `loopPhase`; give `loopPhase` a default of `.baseDenoise` and have the refine call site pass `.refineDenoise` explicitly).

- [ ] **Step 4: Build + full unit suite**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests 2>&1 | tail -12`
Expected: PASS — no existing test regresses. (The seeded-noise sequence change is invisible to unit tests; nothing hashes seeded video output at unit level.)

- [ ] **Step 5: Commit**

```bash
git add Sources/ZImage/LTX2/LTX2Pipeline.swift
git commit -m "feat(ltx2): #1479 — denoisingLoop yields at step boundaries, resumes at startStep, keyed video noise"
```

---

### Task 4: Generator-level checkpoint/resume

**Files:**
- Modify: `Sources/ZImage/LTX2/LTX2VideoGenerator.swift` (the chunk loop ~line 973 region and the base/refine orchestration), `Sources/ZImage/LTX2/LTX2Pipeline.swift` (only if `generateT2V`/`generateI2V` return shapes need the yielded variant threaded — keep the surface minimal).
- Test: `Tests/ZImageTests/LTX2PreemptionTests.swift` (extend — pure logic only), full behaviour in Task 6.

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces (Task 5 relies on these exact names):
  - `enum LTX2RenderOutcome { case completed(LTX2PipelineOutput); case yielded(LTX2ResumeState) }`
  - On `LTX2VideoGenerator`:
    `func setPreemptionSignal(_ s: PreemptionSignal?)`,
    `func setTelemetry(_ t: LTX2PhaseTelemetry?)`,
    generation entry points return `LTX2RenderOutcome` (or an equivalent
    thin wrapper if the existing public return type must stay — implementer's
    judgment, but Task 5 needs to receive the `LTX2ResumeState` and hand it back),
    `func resume(from state: LTX2ResumeState, /* original request context */) -> LTX2RenderOutcome`
  - `resume` MUST validate `state.configFingerprint` against the current
    resolved config and **throw/fail loudly on mismatch** — never silently
    restart from step 0 (spec, Error handling).

**Read first:** the generator's chunk loop and base→refine flow. The generator runs N chunks; each chunk runs base denoise (+ optional refine denoise) then decode. Preemption yields only inside base/refine denoise (Task 3). Chunk boundaries and phase boundaries are free unwind points — check `preemption?.isRaised` between chunks and between phases too, yielding a state with `stepIndex: 0` and the NEXT phase/chunk, which makes resume trivial there.

- [ ] **Step 1: Thread the signal and telemetry down**

Add stored `private var preemption: PreemptionSignal?` and `private var telemetry: LTX2PhaseTelemetry?` with the two setters. Pass both into every `denoisingLoop`-reaching call. Wrap the existing phases with telemetry:

```swift
telemetry?.begin(.vaeDecode); let frames = decode(...); telemetry?.end(.vaeDecode)
```

Apply to: model load (`.modelLoad`), text encode (`.textEncode`), base loop (`.baseDenoise` — begin/end around the loop call; per-step samples come from Task 3), refine loop (`.refineDenoise`), VAE decode (`.vaeDecode`), vocoder (`.vocoder`), post-process (`.postProcess`).

- [ ] **Step 2: Propagate `.yielded` up and shape `resume(from:)`**

When a chunk's base or refine loop yields, capture the generator-level context the resume needs that ISN'T in `LTX2ResumeState` (the i2v conditioning `state` object, text/negative/NAG embeddings, audio context, `positions`/`precomputedPE` inputs). Hold them in a `private var pendingResumeContext: ...?` on the generator — they are non-weight tensors and the generator instance survives eviction of the *pool models* (Task 5 evicts weights via the holder/pool, not by destroying the generator... **if the eviction path DOES destroy the generator instance, the context must move into a coordinator-held box instead — decide by reading `VideoGeneratorHolder.release()` first and write down which it is in the task report**).

`resume(from:)`:
1. Assert fingerprint match; on mismatch return a failed outcome with a message naming both fingerprints.
2. Re-derive `positions`/`precomputedPE` (deterministic from dims — spec).
3. Restore `avState.audioLatents`/`audioNoiseKey` from the state.
4. Re-enter `denoisingLoop(startStep: state.stepIndex, loopPhase: state.phase, chunkIndex: state.chunkIndex, ...)` for the yielded phase, then fall through to the normal remaining flow (refine if yielded in base, decode, vocoder, post) exactly as an uninterrupted render would.

- [ ] **Step 3: Unit-test the pure parts**

Extend `LTX2PreemptionTests`:

```swift
  func testFingerprintMismatchIsDetectable() {
    // Pure string comparison logic — the guard itself, not the render.
    let s = LTX2ResumeState(videoLatents: MLXArray.zeros([1]), stepIndex: 3,
      sigmas: [1.0, 0.5, 0.0], phase: .baseDenoise, chunkIndex: 0, seed: 1,
      audioLatents: nil, audioNoiseKey: nil, configFingerprint: "704x480f97s8-euler_a-cfg3.5")
    XCTAssertNotEqual(s.configFingerprint, "704x480f97s8-euler_a-cfg2.0")
  }
```

(The real resume behaviour is Task 6's integration test; do not fake it here with mocks.)

- [ ] **Step 4: Build + full unit suite**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests 2>&1 | tail -12`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/ZImage/LTX2/LTX2VideoGenerator.swift Sources/ZImage/LTX2/LTX2Pipeline.swift Tests/ZImageTests/LTX2PreemptionTests.swift
git commit -m "feat(ltx2): #1479 — generator checkpoint/resume: phase telemetry wiring, yielded propagation, fingerprint-guarded resume"
```

---

### Task 5: Coordinator orchestration — `preempt: true`, evict, run, restore

**Files:**
- Modify: `Sources/ZImage/Server/WarmServer.swift` — the `POST /v1/generate` request struct (locate by the route case string), the video render path that calls `videoHolder.beginRender()` (~line 5642 region), `queueListResponse()` (~line 2533), and the `WarmServerCoordinator` actor (~line 4638).
- Test: `Tests/ZImageTests/` — only what is pure; the orchestration is Task 6's integration surface.

**Interfaces:**
- Consumes: `LTX2RenderOutcome`, `resume(from:)`, `setPreemptionSignal`, `LTX2PhaseTelemetry` (Tasks 1–4), `VideoGeneratorHolder.isRendering()/release()`, `ModelPool.evictIfNeeded(neededMB:allowActiveEviction:)`.
- Produces: request field `preempt: Bool?` on the image-generate request (additive, default absent/false); `/v1/queue` JSON gains additive fields `"phase"`, `"max_uninterruptible_sec"`, `"phase_timings"`; job-status responses for a refused preemption carry `"preempt_refused": true, "eta_sec": <Double>`.

Behaviour to implement, in order:

1. **Request parse:** add `let preempt: Bool?` to the image-generate request struct.
2. **Refusal guard** (pure function — write it as a free function so it unit-tests):

```swift
/// #1479: refuse when finishing beats preempting. INERT until telemetry has
/// samples for both sides — never refuses on a guess (spec).
func preemptionRefusalETA(
  stepsRemaining: Int, meanStepSec: Double?,
  remainingPhaseMeansSec: [Double],           // phases still ahead, observed means only
  evictReloadRoundTripSec: Double?
) -> Double? {                                 // nil = allow; value = refuse, ETA seconds
  guard let stepSec = meanStepSec, let roundTrip = evictReloadRoundTripSec else { return nil }
  let projected = Double(stepsRemaining) * stepSec + remainingPhaseMeansSec.reduce(0, +)
  return projected < roundTrip ? projected : nil
}
```

3. **Orchestration** on the coordinator, when an image job arrives with `preempt == true` while `videoHolder.isRendering()`:
   - If a preemption is already in flight → treat as a normal (non-preempting) enqueue. Nested preemption refused (spec).
   - Evaluate the refusal guard; if it returns an ETA → respond with the normal queued response plus `preempt_refused: true, eta_sec: eta`.
   - Otherwise: `preemptionSignal.raise()`; the running video job's call stack returns `.yielded(state)` up to the coordinator's video-execution code, which stores `state` in `private var checkpointedVideo: LTX2ResumeState?` **on the coordinator actor** and marks the video job as `paused-for-preemption` in the tracker (NOT failed, NOT completed).
   - Decide eviction: ask `ModelPool` whether the image job's family fits alongside; if not, release video weights (`videoHolder.release()` and/or `pool.evictIfNeeded(neededMB:...)` — follow whichever path the video load used, read the `beginRender` region first).
   - Run the image job exactly as a normal render.
   - **Always resume** in a `defer`/completion path — preemptor success AND failure both resume the video (spec: a failed tap must never cost a video). Reload weights if evicted (same load path as a cold video start), call `generator.resume(from: state)`, clear the signal, clear `checkpointedVideo`.
   - Resume failure → the video job fails loudly with the fingerprint/reload error. No silent restart.
   - Checkpoint failure (yield never arrives within a generous window, e.g. 2× meanStepSec + 30s) → clear the signal, run the image job WITHOUT preemption (it just queues), log one loud line.
4. **Telemetry publication:** in `queueListResponse()`, merge `telemetry.view()` into the payload:

```swift
let tv = ltx2Telemetry.view()
if let phase = tv.currentPhase { payload["phase"] = phase }
if let m = tv.maxUninterruptibleSec { payload["max_uninterruptible_sec"] = m }
payload["phase_timings"] = tv.phases.mapValues { ["mean_sec": $0.meanSec, "samples": $0.samples] }
```

5. **Evict/reload timing:** record observed evict+reload durations into two `LTX2PhaseTelemetry`-style rolling means on the coordinator (a simple `(sumSec, samples)` pair each is fine); `evictReloadRoundTripSec` for the guard = sum of the two means, nil until both have a sample.

- [ ] **Step 1: Write the failing unit tests for the guard**

```swift
final class PreemptionRefusalTests: XCTestCase {
  func testInertWithoutTelemetry() {
    XCTAssertNil(preemptionRefusalETA(stepsRemaining: 1, meanStepSec: nil,
      remainingPhaseMeansSec: [], evictReloadRoundTripSec: 120))
    XCTAssertNil(preemptionRefusalETA(stepsRemaining: 1, meanStepSec: 30,
      remainingPhaseMeansSec: [], evictReloadRoundTripSec: nil))
  }
  func testRefusesNearlyFinishedRender() {
    // 2 steps * 10s + 40s decode = 60s remaining < 120s round trip -> refuse, ETA 60
    let eta = preemptionRefusalETA(stepsRemaining: 2, meanStepSec: 10,
      remainingPhaseMeansSec: [40], evictReloadRoundTripSec: 120)
    XCTAssertEqual(eta!, 60.0, accuracy: 0.001)
  }
  func testAllowsLongRemainingRender() {
    XCTAssertNil(preemptionRefusalETA(stepsRemaining: 40, meanStepSec: 12,
      remainingPhaseMeansSec: [40, 15], evictReloadRoundTripSec: 120))
  }
}
```

- [ ] **Step 2: Run to verify they fail**, then implement the guard, then PASS.

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/PreemptionRefusalTests 2>&1 | tail -8`

- [ ] **Step 3: Implement the orchestration** (items 1, 3, 4, 5 above). Read the `beginRender()` region and `VideoGeneratorHolder.release()` BEFORE writing, and record in your report which eviction path the video family actually uses.

- [ ] **Step 4: Build + full unit suite green.** Also verify `/v1/queue` additions compile as additive (no removed keys — diff the payload dictionary construction).

- [ ] **Step 5: Commit**

```bash
git add Sources/ZImage/Server/WarmServer.swift Tests/ZImageTests/PreemptionRefusalTests.swift
git commit -m "feat(server): #1479 — preempt flag orchestration: checkpoint, conditional evict, always-resume; refusal guard + /v1/queue phase telemetry"
```

---

### Task 6: Integration tests — bit-identity and always-resume

**Files:**
- Test: `Tests/ZImageIntegrationTests/LTX2PreemptionResumeTests.swift` (create)

**Interfaces:** consumes everything above. Requires LTX-2 weights on disk (present on this Mac — production serves them). **Do NOT run while the production server is rendering** (Global Constraints).

Use the smallest real config that exercises the loop: tiny dims (e.g. 256×256, 9 frames, 8 steps), seeded, audio ON (the A/V case is part of the test, not a variant — Fable review). Follow `Tests/ZImageIntegrationTests`' existing pattern for locating weights and for skip-if-absent behaviour (read a neighbouring test first and copy its guard).

- [ ] **Step 1: Write the bit-identity test**

```swift
import XCTest
import MLX
@testable import ZImage

final class LTX2PreemptionResumeTests: XCTestCase {
  // Small seeded A/V render. Uninterrupted vs preempt-at-step-3-then-resume
  // must produce IDENTICAL final latents — video AND audio. Asserts on
  // latents, not the MP4 (container encode is not bit-stable — spec).
  func testBitIdenticalResumeSeededAV() throws {
    let gen = try makeSmallGenerator()          // helper: tiny dims, audio on, per existing integration-test pattern
    // 1) Uninterrupted reference
    let ref = runToCompletion(gen, seed: 4242)
    // 2) Preempted run: raise at step 3 via a signal that trips after 3 recordStep calls
    let signal = PreemptionSignal()
    gen.setPreemptionSignal(signal)
    let outcome = runUntilYield(gen, seed: 4242, raiseAfterStep: 3, signal: signal)
    guard case .yielded(let state) = outcome else { return XCTFail("expected yield") }
    XCTAssertEqual(state.stepIndex, 3)
    signal.clear()
    let resumed = try gen.resume(from: state /* + original request context per Task 4 */)
    guard case .completed(let out) = resumed else { return XCTFail("expected completion") }
    // 3) Identity — video latents and audio latents
    XCTAssertTrue(MLX.allClose(ref.finalLatents, out.finalLatents, atol: 0, rtol: 0).item(Bool.self))
    XCTAssertTrue(MLX.allClose(ref.audioLatents!, out.audioLatents!, atol: 0, rtol: 0).item(Bool.self),
      "audio latents must match too, or the checkpoint dropped avState")
  }

  func testResumeRefusesMismatchedFingerprint() throws {
    let gen = try makeSmallGenerator()
    let outcome = runUntilYield(gen, seed: 7, raiseAfterStep: 1, signal: {
      let s = PreemptionSignal(); gen.setPreemptionSignal(s); return s }())
    guard case .yielded(var state) = outcome else { return XCTFail() }
    state.configFingerprint = "not-the-same-config"
    XCTAssertThrowsError(try gen.resume(from: state)) // loud failure, never step-0 restart
  }
}
```

(`makeSmallGenerator` / `runToCompletion` / `runUntilYield` are file-local helpers the implementer writes against the real generator API — no mocks; `raiseAfterStep` hooks the progress callback and raises the signal after N steps.)

- [ ] **Step 2: Check production is idle, then run**

```bash
curl -s localhost:7870/v1/queue | python3 -c 'import sys,json; d=json.load(sys.stdin); print("rendering:", d["is_rendering"])'
# Only if rendering: false —
xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageIntegrationTests/LTX2PreemptionResumeTests 2>&1 | tail -15
```

Expected: PASS. If `rendering: true`, wait — do not run heavyweight tests against a busy GPU; note it in the report and coordinate with the controller.

- [ ] **Step 3: Commit**

```bash
git add Tests/ZImageIntegrationTests/LTX2PreemptionResumeTests.swift
git commit -m "test(ltx2): #1479 — bit-identity resume (video+audio latents) + fingerprint-mismatch refusal"
```

---

### Task 7: Full gate

- [ ] **Step 1: Full unit suite + build**

```bash
xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests 2>&1 | tail -12
xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode 2>&1 | tail -5
```

Expected: tests PASS; release build succeeds. **Do NOT deploy the built binary** — `:7870` serves production; deploy is a separate, human-coordinated step (spec Deploy/rollback: v1 ships dark anyway, nothing sets `preempt: true` until the broker follow-up).

- [ ] **Step 2: No-regression spot-checks**

- `grep -n "MLXRandom.normal(currentLatents.shape" Sources/ZImage/LTX2/LTX2Pipeline.swift` → expected: zero hits (both draws now keyed-or-global via `ancestralVideoNoise`).
- Confirm `/v1/queue` payload only GAINED keys (read the `queueListResponse()` diff).

- [ ] **Step 3: Commit anything outstanding; verify branch state**

```bash
git status --porcelain   # must be clean
git log --oneline main..HEAD
```

---

## Self-Review (performed at write time)

- **Spec coverage:** telemetry-first sequencing → Task 1; signal/state/keyed-noise → Task 2; loop yield + resume entry + the two noise draws → Task 3; generator resume + fingerprint guard + phase wiring → Task 4; job flag, refusal guard (inert-until-sampled), always-resume, nested refusal, checkpoint-failure fallback, `/v1/queue` additive fields → Task 5; bit-identity incl. audio latents + loud-mismatch → Task 6. Deploy intentionally absent (ships dark; spec).
- **Known open point, made explicit rather than hidden:** whether the resume context lives on the generator or in a coordinator-held box depends on what `VideoGeneratorHolder.release()` actually destroys — Task 4 Step 2 instructs the implementer to read it first and record the answer. This is a deliberate decision-with-evidence, not a placeholder.
- **Type consistency check:** `LTX2DenoiseResult` (Task 2) consumed in Task 3; `LTX2RenderOutcome` (Task 4) consumed in Task 5; `LTX2PhaseTelemetryView` field names match between Tasks 1 and 5's payload merge; `ancestralVideoNoise(shape:seed:step:)` identical in Tasks 2 and 3.
