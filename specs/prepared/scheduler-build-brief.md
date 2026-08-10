# Build Brief: Kira Reservation Scheduler (#23)

**For:** a fresh Claude Code session. Boot cost = this brief + the spec.
**Written:** 2026-08-04 by the session that authored the spec. Todd approved
the fresh-session build; Codex builds Ideogram (#22) in parallel — stay out
of each other's lanes.

## Mission

Build P1 of specs/kira-reservation-scheduler.md (rev 2 body + rev 4 ladder
addendum — read BOTH in full; rev 4 supersedes the greedy fit language).
Yield management over perishable GPU-seconds: typed MediaTypes, timeslot
reservations, interactive buffers, on-demand bumping, size-ladder rung
selection via GLOBAL packing (minima first), shadow-mode cutover (8 steps,
defined in the spec).

## Two gates BEFORE writing scheduler code

1. **GPU contention measurement** (spec requirement): ~1 hr instrumented
   renders quantifying admission latency under load — P50/P95 time from
   interactive request to GPU grant while a render runs. Method sketch in
   spec §GPUQuiescenceGate. Data feeds the interactive-window SLA.
2. **Todd's ad-server semantics** (he was CTO of Advertising.com 1999-2004
   — treat his framing as authoritative domain design): ask him for a
   15-minute dictation of priority classes, underdelivery/makegood
   behavior, and what "sold out" means for interactive windows. Capture
   verbatim into the spec before encoding policy.

## Where things live

- **Daemon (bulk of the build):** ~/Projects/coffeeshop-server —
  src/kira/kira-daemon.ts, src/kira/content-scheduler.ts (current 24/7
  cycle you are replacing incrementally — study its event shapes),
  src/kira/api/kira-api.ts, src/tools/video-tools.ts.
- **Engine touchpoints (Mac, this repo):** WarmServer admission
  (enqueueLocalVideo/wantsAudio pattern), RenderTraceStore (cost-model
  data source), /v1/video/generate contract.
- **Kira state root on server:** /home/todd/.kira (config.json holds
  clipSeconds, videoMode, videoAudio).

## Live state you must not regress

- clipSeconds = 5 (Todd set 2026-08-04); videoAudio default-ON (config
  kill switch { videoAudio: false }); audio-mode changes reload the
  engine transformer (~1-2 min) — the scheduler must BATCH same-audio-mode
  jobs, never alternate.
- Continuation chunks degenerate: ≤289f folds to single-pass server-side.
  Audio rejects chunked/multi-keyframe renders loudly (by design).
- Cost realities (measured): duration linear, resolution ~quadratic,
  refine large multiplier, speech ~2x steps (727s for 4s@16 steps),
  krea2 reloads ~65s after EVERY video eviction — admission signatures
  must include residency transitions (spec rev 4 finding #15).

## Operational rules (hard-won; violating these costs hours)

- Server deploy: push branch → ssh todd@10.0.100.232 (fish shell — always
  `bash -c "..."`) → `cd ~/coffeeshop-server && ./scripts/local-merge.sh
  <branch>` (732-test gate) → `git pull && node build.mjs && systemctl
  --user restart bree-daemon kira-daemon`. Webhook deploy is unreliable;
  do the manual steps.
- Multi-session coordination: post status to backroom :3777; check the
  server semaphore dir before gate/deploy (memory: multi-session-coordination).
- Engine deploys (if needed): wait for /health is_rendering=false; use
  the resign-request flow (memory: comfybox-desktop-deploy). Warn Todd
  BEFORE anything that can block on a permission dialog.
- TDD per repo convention; tests must pass the merge gate — check for
  stale payload assertions when adding fields (clipSeconds precedent).

## Memory (auto-loaded, but prioritize these)

kira-scheduler-pacing, kira-inner-loop-shipped, comfybox-queue-management,
ltx2-audio-wire1-progress (audio/warm-key facts), todd-adcom-cto,
kira-objectives-vision (the DEMAND side — do NOT build it in P1; campaigns
stay dumb; design the campaign interface so an objectives-driven generator
plugs in later), multi-session-coordination, dont-generalize-validate-at-
production-config.

## Build sequence (spec's fold-in order)

1. Contention measurement + Todd semantics (gates above).
2. Reservation core: calendar, MediaType registry + ladders (rev 4 rung
   model), typed + OPEN slots — SHADOW MODE (observes, logs decisions,
   controls nothing).
3. GPUQuiescenceGate (lock-backed, phase-aware, requested→granted).
4. Global packing planner + interactive buffers.
5. Shadow A/B vs live scheduler → begin the 8-step cutover.

Report progress in the task list (#23) and commit spec amendments as you
learn — the spec is the contract; keep it true.
