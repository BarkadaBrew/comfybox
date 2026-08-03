# SPEC: Kira reservation scheduler — typed media, timeslot packing, visible queue

**Status:** draft from Todd's 2026-08-03 direction · **Owner:** Todd
**Supersedes:** the count-based tick model ("N images + M videos per cycle")

## 1. Todd's direction (verbatim inputs, 2026-08-03)

1. "The Kira scheduler should be timeslot holders for stacked assets that
   can fit within a given period. Images of a given tier or Video be it i2v
   vs t2v should be sized to fit or be pushed. Each type should have a media
   Type and be scheduled like a reservation system."
2. "We have Animated portraits and Videos of action. Animated portraits are
   i2v and t2v are Dreams or vignettes of given activity. Some i2v are
   animations of action and others are more subtle. Each should be
   qualifiable and optimized for delivery."
3. "The Queue should show jobs that are stacked and pending. In the desktop
   Application there doesn't seem to be a manageable queue — just the
   current job."

## 2. Concept model

### MediaType — the qualifiable unit
Every scheduled asset is an instance of a declared MediaType:

| id | kind | engine path | recipe knobs (per-type optimization) |
|---|---|---|---|
| `image.<tier>` | still | krea2 preset per tier | preset id, dims, steps |
| `portrait.animated` | i2v, SUBTLE | i2v, low motion | strength high-hold, low img_compression, gentle template ("breathing, blinking, hair drift"), 4–6s |
| `animation.action` | i2v, ACTION | i2v, motion recipe | higher img_compression, action template w/ anatomy grounding, 8–10s |
| `dream.vignette` | t2v | t2v scene | buildT2VScene prompt, 720p, 8–12s, cfg/NAG per validated recipe |

- A MediaType is a **record, not code**: `{id, kind, engineParams (preset id +
  LTX2VideoTuning block + template id), durationSeconds | dims, estCostSec,
  deliveryRules, tierEligibility}` — stored in kira config, editable from
  the Kira tab. The engine-side knobs ride the machinery shipped today
  (presets, videoTuning, prompt templates, per-request tuning).
- "Qualifiable": every rendered asset carries its mediaType id into the
  trace + gallery metadata, so ratings/QA aggregate BY TYPE ("action i2v is
  drifting again") instead of by blob.
- `estCostSec` starts as a config number per type and is then LEARNED: the
  scheduler updates a rolling median from actual render durations (traces
  already record elapsed).

### Timeslots — the reservation ledger
- A **cycle** (the existing pacing interval, e.g. 30 min) is a timeslot with
  a capacity budget: `capacitySec = cycleLength − overheadReserve`.
- The planner holds a standing **slot template** per tier window (e.g.
  overnight banana: 1 dream.vignette + 2 image.banana + 1 portrait.animated),
  also config, editable in the tab.
- **Fit-or-push packing:** reservations are placed into the current slot in
  priority order until `Σ estCostSec` would exceed capacity; the remainder
  is PUSHED to the next slot's front (a persisted backlog, not dropped —
  today's overflow just doesn't happen and vanishes).
- Direct asks (Telegram/chat renders) preempt as today via the fast lane;
  the slot's plan re-fits around the actual time consumed.
- The ledger is persisted (survives daemon restarts; the orphan-reconciler
  pattern applies).

### Interactive windows — the responsiveness reserve (Todd, 2026-08-03)

> As a User, I want to ask the Muse a question and have it answer within a
> reasonable period of time.
> **AC:** Muse responds when GPU avails; the scheduler leaves windows of
> opportunity for responses.

The packer never fills a slot wall-to-wall. Rules:

- `capacitySec = cycleLength − overheadReserve − interactiveReserve`.
- The interactive reserve is distributed as **windows**, not one lump: a
  configurable cap on CONTIGUOUS GPU occupancy (default: no more than ~10
  consecutive minutes of scheduled rendering without a gap ≥ one typical
  chat-render, ~3 min). Long renders (dream.vignette at 12s ≈ 5–6 min) are
  scheduled so a window follows them.
- Direct chat/muse asks consume the interactive reserve via the existing
  fast lane — they do NOT debit the slot's content budget (Q2: resolved).
  Worst-case wait for a chat media reply = the remaining runtime of the
  current job, bounded by the contiguous-occupancy cap.
- Idle windows may be backfilled ONLY by a task that provably fits before
  the window's end (a short image), else the GPU rests — an empty window is
  the feature, not waste.
- Text-only muse replies are unaffected (LLM path, no render GPU); this
  reserve is about MEDIA responses.

### Shared-GPU contention — cognition vs generation (Todd, 2026-08-03)

> Problem: GPU is a shared resource for LLM and Gen models. Turn responses
> and cognition compete with image generations.

Evidence: 2026-08-03 09:35 — Kira's agent loop timed out (90s) calling the
LM Studio model mid-render. Windows between renders don't fix this: a
single dream.vignette holds the GPU 5–6 minutes, and tokens starve INSIDE
that span. Three layers, cheapest first:

1. **Cognition lease (engine governor) — the real fix.** Renders are step
   loops with existing per-step callbacks. Add a lease mechanism to the
   warm server: `POST /v1/gpu/lease` (holder, ttl ≤ 90s) makes the active
   render PAUSE AT THE NEXT STEP BOUNDARY and resume on release/expiry.
   Bounded token latency = one denoise step (2–30s at production configs) +
   LLM burst. The daemon takes a lease before agent-loop LLM calls during
   active renders; leases are metered so renders still finish (max N leases
   per render, else the render's own SLA dies).
2. **Scheduler windows (already spec'd)** handle the render-class
   interactive asks; they also lower the PROBABILITY of collision for
   cognition, but are not sufficient alone.
3. **Placement (structural, later):** cognition models contending on the
   render GPU is ultimately a placement smell. Candidates: a small
   cognition model on CPU for turn-critical paths (fallback when the lease
   is unavailable), or ANE via CoreML for the fixed cognition model —
   zero GPU contention. Measure before building: token latency during a
   render at production configs is the number that decides whether layer 1
   suffices.

Measurement task (pre-build): LLM tokens/sec and time-to-first-token
during (a) idle, (b) krea2 image render, (c) LTX 241f render — the
contention curve that sizes the lease TTL and decides layer 3.

### Queue visibility — the execution surface (Todd's #3)
- The engine ALREADY has queue management server-side (pause/resume/reorder,
  source attribution) and the desktop has a Queue tab — but stacked/pending
  work is invisible in practice because (verify during build): video jobs
  and daemon-side planned-but-not-yet-submitted reservations never appear;
  only the in-flight job does.
- Deliverable: the Queue tab shows THREE strata:
  1. **Running** — current job with progress (exists)
  2. **Queued in engine** — server FIFO with reorder/cancel (partially
     exists; make video jobs appear)
  3. **Reserved** — the daemon's slot plan for this + next cycle (new: GET
     /v1/kira/schedule/ledger), each row typed (mediaType badge), with
     push/pull/cancel controls
- One timeline view = the reservation system made visible.

## 3. Delivery optimization (Todd's "optimized for delivery")

Per-MediaType delivery rules: target surface (Telegram tier stream, gallery
only, suggestion-box reply), encode budget (bits/px — now request-tunable),
and caption/notification behavior. E.g. portrait.animated may deliver as a
looping short; dream.vignette as a full post.

## 4. Phasing

- **P1 (daemon):** MediaType records + slot planner + fit-or-push backlog +
  ledger endpoint. The existing tick becomes "execute this slot's plan".
  clipSeconds (shipped 2026-08-03) folds INTO per-type duration.
- **P2 (desktop):** Queue tab strata + Kira tab MediaType/slot-template
  editors (clipSeconds field arrives here, per-type).
- **P3:** learned estCostSec, per-type QA dashboards from trace aggregation.

## 5. Open questions for Todd

1. Slot templates per tier window — who authors the starting set? (Propose:
   I derive from current behavior — 4 images + 1–2 videos per 30min — and
   you edit in the tab.)
2. ~~Direct chat asks~~ RESOLVED 2026-08-03: they consume the interactive
   reserve (windows between reservations), never the content budget; the
   contiguous-occupancy cap bounds worst-case response latency.
3. Backlog aging: does a pushed reservation expire after N slots, or
   persist until rendered?
4. portrait.animated vs animation.action selection: planner-scheduled
   ratio, or content-driven (the muse/arc state decides)?

## 6. Non-goals

Multi-machine scheduling, external calendar integration, per-asset cost
billing. The engine's FIFO stays the execution primitive — reservations
plan ABOVE it, they don't replace it.
