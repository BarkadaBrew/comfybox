# #1479 — Engine-Side Render Preemption (LTX-2)

Date: 2026-08-15 · Status: approved-pending-review · Repo: zimage.swift (ComfyBox)
Epic: #1469 · Complements: #1485 (owner lanes + credit governor, shipped 2026-08-15)

## Goal

#1485 gave the render queue owner lanes and a weighted-credit governor, so a
gallery tap is chosen next. It cannot preempt the job already on the GPU. With
LTX-2 clips running 13–60 minutes, "chosen next" can still mean a long wait.

This ticket lets the engine **pause an in-flight LTX-2 render at a step
boundary, run a preempting job, and resume from where it left off** — so
interactive work waits seconds rather than minutes.

## Two corrections to the issue text

The issue (#1479) describes work that does not match the code. Both were
verified before design:

1. **"non-LTX families lack step callbacks today" is stale.** Every family
   already has a per-step callback. They are two incompatible shapes:
   - LTX-2: `progressCallback: ((Int, Int) -> Void)?` — bare step/total
     (`LTX2Pipeline.swift:181` and throughout)
   - ZImage / Flux2 / Fibo: `progressHandler: ProgressHandler?` carrying
     `GenerationProgress` with a per-family `Stage` enum, duplicated three
     times (`ZImagePipeline.swift:538`, `Flux2Pipeline.swift:117`,
     `FiboPipeline.swift:111`)

   The work is not "add missing hooks" — it is "make one boundary yieldable."

2. **`POST /v1/gpu/lease` is the wrong shape for v1.** See Decision 3.

## Decisions (Todd, 2026-08-15)

1. **Pause-and-evict**, with pause-in-place as a fast path. The model pool
   budget blocks a 2×22GB swap, so a paused video render that keeps its
   weights would refuse most preemptions — failing in exactly the headline
   case. Latents are small; weights are fat. Evict weights, keep latents.
2. **LTX-2 video only in v1.** Video is where the pain is. Image renders
   finish in ~5 minutes, so preempting them buys little.
3. **Job-scoped `preempt: true` flag, not a standalone lease.** The broker
   holds a lease only to run one job; if that job is submitted *to* ComfyBox,
   the engine owns the whole lifecycle in-process. No TTL, no heartbeat, and
   no way for a dying broker to wedge the GPU — which matters because a deploy
   restarts the kira daemon every time. A standalone lease is the right shape
   later for #1520 and for consumers that bypass the queue (Bree's daemon
   calls ComfyBox directly; LM Studio and Ollama contend for the same memory).
   It becomes a thin wrapper over the same primitive.
4. **Bit-identical resume.** See "The RNG problem" below.
5. **The engine may refuse.** Only the engine knows remaining steps, current
   phase, and reload cost. If projected remaining time is less than the
   evict+reload round trip, decline and report an ETA; the broker waits.
   Without this guard a burst of taps against a nearly-finished render
   thrashes the model pool and costs more GPU than it saves.
6. **Per-step RNG keying for seeded runs only.** See below.

## Scope

**In:** the ComfyBox engine side only. The engine change is independently
testable and independently deployable.

**Out:** the broker setting `preempt: true` (a small `coffeeshop-server`
change, follow-up ticket). Splitting them means a bug in either half cannot
strand the other. Also out: image families, the standalone lease endpoint,
preemption of anything but LTX-2.

## The RNG problem (why resume is not just "restore latents")

`LTX2Pipeline.swift:251` calls `MLXRandom.seed(seed)` — the **global** MLX
stream — and the ancestral/SDE noise inside the step loop draws from it
(`useSDE = samplerIsAncestral && !forceDeterministic`, `:1055`). This is
reproducible today *only* because renders are single-flight and nothing else
draws in between. That invariant is undocumented, and preemption is the first
thing to break it: the preempting job calls `MLXRandom.seed(...)` itself and
destroys the stream position.

`MLXRandom` exposes `key(_ seed: UInt64) -> MLXArray`
(`mlx-swift/Source/MLXRandom/Random.swift:30`) but **no global state
save/restore**, so snapshotting the RNG is not available.

**Resolution:** derive per-step keys when a seed is supplied —
`MLXRandom.key(seed &+ 0xD0D10 &+ (UInt64(step) &* 0x9E37_79B9_7F4A_7C15))`,
using a base constant distinct from the audio path's `0xA0D10/11/12` so the
two streams cannot collide, and the step multiplied by a large odd constant
before folding in — following the convention this
file already uses for audio (`:288`, `:299`, `:364`:
`MLXRandom.key(seed &+ 0xA0D10 / 11 / 12)`, a pattern with a codex-review note
attached at `:286`). This is stronger than a naming convention: the audio step
at `:1322` already draws **keyed ancestral noise inside the loop**, commented
"so the global RNG (video noise sequence) is untouched," coexisting with the
global-stream video draw. The video-side change applies a mechanism the same
loop already proves out. Verified: there is exactly ONE global draw per step
in either SDE branch (`:1300`, `:1310`), so a per-step derived key maps 1:1;
and the sampler carries no cross-step momentum — latents plus RNG discipline
is the entire loop-carried state. The step multiplier is not cosmetic: the
production chunk scheduler derives each chunk's seed as
`request.seed + UInt64(chunk)` (`LTX2VideoGenerator.swift`), so a bare
`&+ UInt64(step)` term makes chunk `c` step `i` and chunk `c+1` step `i-1`
fold to the identical key — every multi-chunk ancestral render was reusing
bit-identical noise tensors across chunk/step pairs (codex review,
2026-08-15) — the large odd multiplier makes that collision unreachable.
Unseeded runs keep the global stream, which the code
keeps deliberately so unseeded noise varies; a render with no seed makes no
reproducibility promise, so a fresh draw on resume is fine.

**Consequence:** this changes the noise sequence for existing seeds — a
one-time visual shift on seeded LTX-2 renders. Todd's `ab-*` A/B baselines
would need re-basing if still compared against. Accepted: it also makes a seed
mean what people assume it means, independent of what else the process did.

## Architecture

Because we evict, the render must **unwind** to release weights — you cannot
drop 22GB while suspended mid-function. So this is
checkpoint-and-restart-from-checkpoint, not suspend/resume.

Simplification that follows: **always unwind, conditionally evict.** Unwinding
is cheap; eviction is the expensive part. One code path with eviction as a
decision inside it, rather than separate in-place and evicting paths.

Rejected: suspend-in-place (incompatible with eviction); subprocess +
`SIGSTOP` (a stopped process keeps its VRAM, and ComfyBox is in-process).

### Components

- **`PreemptionSignal`** — a lock-protected flag readable from inside the
  render loop with **no actor hop**. Reuses the pattern already built for #217
  (`WarmServer.swift:4273-4274`: "lock-protected so it can cross the actor
  boundary safely without an actor hop on every denoising step"). This is
  load-bearing: `WarmServerCoordinator` (`:4638`) is blocked for the full
  duration of a synchronous GPU render, so the signal cannot be an actor read.
- **`LTX2ResumeState`** — everything the loop references *except* model
  weights. The governing rule: **checkpoint = all non-weight tensors; evict =
  weights only.** Enumerated as of today (Fable review, from the loop body at
  `LTX2Pipeline.swift:1057-1330`):
  - `currentLatents`, step index, sigma position, seed, phase, config
    fingerprint (the original list);
  - **`avState.audioLatents`** — audio latents evolve every step alongside
    video, and Kira's production renders have audio ON; dropping them resumes
    video while silently restarting audio;
  - **`avState.audioNoiseKey`** — the audio ancestral chain head, split each
    step (`:1322`); loop-carried state;
  - the i2v conditioning tensors (`denoiseMask`, `cleanLatent`, `faceMask`,
    `faceRef`, `faceAnchorStrength`) — re-applied every step; keeping them
    avoids a VAE re-encode on resume;
  - `textEmbeddings`, `negativeEmbeddings`, `nagEmbeddings`,
    `av.audioContext`, `av.negativeAudioContext` — Gemma outputs referenced
    every step; megabytes, keep them so the text encoder can stay evicted.
  - `positions`/`precomputedPE`/`av.pe` are NOT checkpointed — they are
    deterministic functions of the dims and are recomputed at resume.

  In-memory only; it dies with the process, deliberately unlike the
  `isPaused` pause sentinel (`:4704`).
- **Resumable sampler entry** — `LTX2Pipeline`'s loop (`:1057`,
  `for i in 0..<numSteps`) gains the ability to start at step N with supplied
  latents. The loop already carries `currentLatents` and an explicit `sigmas`
  array, so re-entry is natural.
- **Eviction hook** — conditional weight release via the existing `ModelPool`.
- **Refusal guard** — declines when projected remaining < evict+reload round
  trip, returning an ETA. Both sides are computed from the phase telemetry,
  not guessed: *projected remaining* = `stepsRemaining × observedMeanStepSec`
  plus the mean observed duration of the phases that still follow (decode,
  vocoder, post); *round trip* = observed evict + reload time for the resident
  family. Until telemetry has samples for a family, the guard is inert (never
  refuses) rather than refusing on a guess.
- **Phase telemetry** — per-phase timings and `max_uninterruptible_sec`
  published on `/v1/queue`.

### Data flow

Broker submits a job with `preempt: true` → coordinator raises
`PreemptionSignal` → sampler observes it at its next step boundary and returns
`.yielded(LTX2ResumeState)` → coordinator evicts **only if** the preemptor
will not fit alongside → preemptor runs → weights reload if evicted → sampler
re-enters at step N → render completes as if uninterrupted.

## Error handling

- **Preemptor fails** → the checkpointed render resumes anyway. A failed tap
  must never cost a video.
- **Checkpoint fails** → refuse the preemption, keep rendering. Never lose
  work to a bookkeeping error.
- **Resume fails** (model reload, config drift) → surface as a render failure.
  Do **not** silently restart from step 0; that hides a real bug behind a
  15-minute cost.
- **Engine restarts holding a checkpoint** → the checkpoint is gone and the
  job requeues. In-memory is the intended behaviour.
- **Nested preemption** → refused. A preemptor cannot itself be preempted;
  this caps the stack at one and prevents unbounded thrash.

## Testing

- **Bit-identity (the decisive test):** render a seeded clip uninterrupted;
  render the same seed preempting at step 12; assert equality **on the final
  latent tensor (or decoded frames), not the MP4** — container encode is not
  bit-stable, so hashing the file would flake for reasons unrelated to the
  sampler. This test can genuinely fail — which is the point. (Contrast the
  #1485 governor-throw test that passed without ever entering its catch.)
  The A/V case is part of this test, not a variant: audio latents must match
  too, or the checkpoint dropped `avState`.
- Refusal guard fires when projected remaining is below threshold, and the
  reported ETA is sane.
- A failing preemptor still resumes the original render.
- Eviction path and fast path both resume correctly; the fast path performs no
  weight reload.
- Phase timings recorded; `max_uninterruptible_sec` published per family.
- **No-preemption path is unchanged** — no measurable per-step cost added to
  normal renders.
- Unseeded runs are unaffected by the RNG change.

## Sequencing

**Phase telemetry lands first.** `max_uninterruptible_sec` is currently an
assumption, not a measurement. The working assumption is that the denoise loop
yields cheaply per step while VAE decode and the vocoder are long single ops —
but there is a hint the decode may be softer than expected: it already chunks
internally (`CausalConv3d` chunks adaptively since the M·K > 2³² fix). If those
chunk boundaries are yield points, decode becomes interruptible too and the
feature gets materially better. Measure before building around assumed
boundaries.

## Deploy / rollback

ComfyBox `:7870` currently serves Kira's production renders, and the running
binary predates this branch. Rebuilds are production-affecting: see the
known `metallib`-after-clean-build and codesign-SIGKILL hazards.

Rollback: the feature is inert unless a job carries `preempt: true`. With the
broker wiring out of scope for v1, nothing in production sets it — so v1 ships
dark and is exercised by tests and manual calls until the follow-up lands.

## Branch note

Cut from `codex/kroma-v02` (12 commits ahead of `main`), not from `main`,
because that branch carries current LTX-2 work this design touches —
including `46217ef`, which changes the sigma schedule. If `codex/kroma-v02`
does not merge, this branch needs rebasing onto whatever supersedes it.
