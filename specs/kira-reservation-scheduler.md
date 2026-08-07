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

## 1.5 Programming model: yield management (Todd, 2026-08-03)

**The discipline this is:** yield management. GPU-seconds are PERISHABLE
inventory — an idle window unsold is gone, like an empty seat at takeoff.
Demand classes have different value (interactive > scheduled; types differ
by tier and delivery surface). The objective is YIELD — value delivered per
GPU-second — not raw utilization: an empty interactive window that catches
one chat reply out-yields a filler render nobody asked for. Learned
estCostSec + muse-state demand patterns are the forecasting side.


> "This is like programming a TV station or an ad delivery engine. The
> optimization is to keep it LOSSY — most of the media is generated just in
> time, or scheduled if pre-rendered, while still managing to be reasonably
> responsive to the User in chat. The Muse has several objective functions
> for delivery with competing ends."

Design consequences:

- **Lossy by design.** A slot that doesn't fill is dead air, not debt. The
  backlog is a SHORT grace window, not a work queue: pushed reservations
  expire after ~1–2 slots (this largely answers §5 Q3) because stale muse
  content isn't worth rendering later — JIT regeneration from CURRENT muse/
  arc/conversation state beats replaying an old plan.
- **JIT-first.** Content decisions (prompt, subject, type selection) happen
  as close to render time as possible — the slot plan reserves CAPACITY by
  type; the CONTENT is chosen at execution using live state. Pre-rendered
  assets (e.g. a film-stream backlog, suggestion-box picks) are the
  exception and get explicit scheduled slots.
- **Competing objective functions, satisficed not optimized:** freshness/
  relevance to muse state · tier stream fill rate · interactive latency
  (the contract above) · per-type quality (trace ratings) · GPU utilization.
  The planner uses fixed priority for the hard constraint (interactivity)
  and weights for the rest — weights in config, visible in the ledger so
  "why did she render this" is answerable.
- The Queue's Reserved stratum therefore shows TYPED capacity holds
  ("dream.vignette @ :15"), not finalized content — like a program guide.

## 2. Concept model

### MediaType — the qualifiable unit
Every scheduled asset is an instance of a declared MediaType:

| id | kind | engine path | recipe knobs (per-type optimization) |
|---|---|---|---|
| `image.<tier>` | still | krea2 preset per tier | preset id, dims, steps |
| `portrait.animated` | i2v, SUBTLE | i2v, low motion | strength high-hold, low img_compression, gentle template ("breathing, blinking, hair drift"), 4–6s |
| `animation.action` | i2v, ACTION | i2v, motion recipe | higher img_compression, action template w/ anatomy grounding, 8–10s |
| `dream.vignette` | t2v | t2v scene | buildT2VScene prompt, 720p, 8–12s, cfg/NAG per validated recipe |

**FPS is a per-type knob (Todd 2026-08-03):** playback fps rides the
MediaType (engine accepts per-request fps today; frames = duration x fps
snapped to the 1+8k grid, so fps directly scales render cost and must feed
estCostSec). Distinct from `cond_fps` — the temporal-RoPE MOTION dial in the
tuning block — which stays a recipe knob; the two are set independently
(e.g. portrait.animated might render 16fps playback for a dreamy cadence
while conditioning at model fps).

- A MediaType is a **record, not code**: `{id, kind, engineParams (preset id +
  LTX2VideoTuning block + template id), dims (width x height — stills AND
  video; video dims interact with the two-stage floor and /64 snapping),
  durationSeconds + fps (video kinds), estCostSec, deliveryRules,
  tierEligibility}`. Video cost scales with dims x frames — both feed
  estCostSec. — stored in kira config, editable from
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

### Priority: on-demand bumps scheduled (Todd, 2026-08-03)

> On-demand requests from User or Muse should bump queue.

Two priority classes, strictly ordered:

- **INTERACTIVE** — User asks (Telegram/chat/desktop) AND Muse-initiated
  on-demand work (her suggestion-box picks, in-conversation renders).
- **SCHEDULED** — the slot plan's reservations.

Bump semantics: an INTERACTIVE job enters at the FRONT of the engine queue
(the server's reorder capability, which already exists, becomes the
mechanism); queued SCHEDULED jobs slide back and the ledger re-fits — spilled
reservations go to the backlog per fit-or-push. The RUNNING job is never
killed (renders aren't resumable): worst-case interactive wait stays
bounded by the contiguous-occupancy cap. Combined with the cognition lease,
this gives: text answers ≤ one denoise step; media answers ≤ current job's
remainder; scheduled content never starves interactivity, only defers to it.

### Shared-GPU contention — cognition vs generation (Todd, 2026-08-03)

> Problem: GPU is a shared resource for LLM and Gen models. Turn responses
> and cognition compete with image generations.

Evidence: 2026-08-03 09:35 — Kira's agent loop timed out (90s) calling the
LM Studio model mid-render. Windows between renders don't fix this: a
single dream.vignette holds the GPU 5–6 minutes, and tokens starve INSIDE
that span. Three layers, cheapest first:

**Generalization (Todd): this holds for ANY model sharing the GPU** —
vision/captioning (the planned VLM provider), embeddings, face detection,
future audio models. The governor is therefore generic: every GPU consumer
declares a class — `render` (long-occupancy, pausable at step boundaries)
vs `inference` (short-burst, latency-sensitive: LLM turns, VLM captioning,
embeddings, PROMPT OPTIMIZATION). Leases are the inference class's admission ticket; nothing is
special-cased to "the chat LLM".

1. **Inference lease (engine governor) — the real fix.** Renders are step
   loops with existing per-step callbacks. Add a lease mechanism to the
   warm server: `POST /v1/gpu/lease` (holder, ttl ≤ 90s) makes the active
   render PAUSE AT THE NEXT STEP BOUNDARY and resume on release/expiry.
   Bounded token latency = one denoise step (2–30s at production configs) +
   LLM burst. The daemon takes a lease before any inference-class call during
   active renders (LLM turn, VLM caption pass, embedding batch, prompt
   optimization). Note the coupling: every scheduled render is PRECEDED by
   an optimizer call, so the planner front-loads a slot's optimizer passes
   into the opening window (batch-optimize, then render) instead of
   interleaving each optimize against the previous item's render; leases are metered so renders still finish (max N leases
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

## 2.5 Open slots, campaigns, and the unbounded horizon (Todd's Q4 answer)

- **Typed holds + OPEN holds.** A lineup mixes reserved MediaType capacity
  with **open slots** — capacity deliberately left untyped for the Muse to
  fill in-context (her arc/energy/conversation decides the type and
  content) or at random for stream variety. Open slots are the JIT
  principle given a first-class home.
- **Campaigns as reservation sources.** Muse campaigns (the existing
  "tonight's beat" system) reserve in two modes: **prepopulated** — the
  campaign books typed holds across future slots ahead of time (a program
  special); or **as-avail** — the campaign opportunistically fulfills into
  open slots and spare capacity as they arise. Campaign id rides the
  reservation into the trace.
- **Unbounded horizon.** The ledger is a CALENDAR, not a rolling cycle:
  reservations may be placed days or weeks out (a weekend arc special, a
  holiday program). The planner materializes near-term slots from
  (long-horizon reservations first, then slot template, then open space);
  far-horizon entries persist untouched until their window approaches.
  Lossiness still applies at execution: a materialized slot that can't fit
  its booking pushes/expires per the broadcast rules — a calendar entry is
  a hold, not a debt.

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

1. RESOLVED 2026-08-03: slot templates are authored by **User OR Agent** —
   Todd edits lineups in the tab; the Muse may also author/adjust her own
   lineups (an agent-writable policy surface, logged so authorship is
   visible in the ledger). Starting set derived from current behavior.
2. ~~Direct chat asks~~ RESOLVED 2026-08-03: they consume the interactive
   reserve (windows between reservations), never the content budget; the
   contiguous-occupancy cap bounds worst-case response latency.
3. Backlog aging: does a pushed reservation expire after N slots, or
   persist until rendered?
4. RESOLVED 2026-08-03 (Todd): "Defined Media types are the units
   scheduled. Open slots may be left for Agent (Muse) to use in context or
   random. Muse campaigns may either be prepopulated or as-avail fulfilled.
   Scope is wide, not limited to a day, week, hour etc." See §2.5.

## 6. Non-goals

Multi-machine scheduling, external calendar integration, per-asset cost
billing. The engine's FIFO stays the execution primitive — reservations
plan ABOVE it, they don't replace it.

---

## Rev 2 — Codex findings resolution (2026-08-03, 14 findings / 6 blockers in reviews/codex-reservation-review-2026-08-03.txt)

Verdict accepted: direction sound, draft guarantees corrected as follows.

1. **Lease mechanism (#1):** NOT a coordinator method (the actor blocks
   behind the render — the exact failure the health snapshots already dodge).
   A lock/condition `GPUQuiescenceGate` independent of the actor; lease has
   `requested → granted` states (granted only at an actual boundary); TTL
   from GRANT; holder token, monotonic expiry, idempotent release, shutdown
   wake-up; a DEDICATED boundary hook after each step's eval (not the
   telemetry callback); per-family hooks required (non-LTX families lack
   step callbacks today).
2. **Phase-aware latency contract (#2):** "≤ one denoise step" holds ONLY
   in the denoising phase. encode/load/refine-transition/decode DEFER or
   reject leases with retry_after (decode is the 25GB+ danger phase — never
   overlap it). Consumers declare expected resident+transient memory, and a
   lease grant requires memory admission, not just quiescence. Engine
   publishes current phase + max_uninterruptible_sec.
3. **Deadlock prevention (#3):** lease tokens propagate and are reentrant
   per holder; batch optimization happens BEFORE render submission and
   batched renders carry enhance:false (already the daemon's convention);
   no queued render may synchronously acquire a lease from its execution
   entry. Fairness = paused-seconds budget + minimum render-run quantum,
   not lease counts.
4. **Atomic priority (#4):** priority rides IN the enqueue call
   (`enqueue(priority, reservationId, jobId)`) — never enqueue-then-reorder
   (racy, and the tracker/queue IDs differ). ONE externally supplied job id
   flows reservation → daemon → engine queue → tracker → trace → gallery.
   Stable FIFO within priority. The existing reorder endpoint is a human
   tool, not the bump mechanism.
5. **Honest SLA (#5):** the 10-min cap was unsatisfiable (measured 923s for
   a 97f two-stage render). The cap becomes a conservative ADMISSION rule:
   never START a scheduled job whose P95 runtime + setup exceeds the
   remaining time before the next protected window. Long-form scheduled
   video runs in declared non-interactive windows; the interactive media
   SLA states the truth: bounded by the admitted job's P95, which the
   planner chooses.
6. **FPS contract (#6):** validate the (clipSeconds, fps) PAIR:
   `targetFrames = snap1p8k(clipSeconds × fps) ≤ 289`. Send explicit
   frames + fps + tuning.condFps. Engine work item: MCP generate_video
   lacks fps and the executor drops frame_rate — must be added before any
   per-type fps ships. The 12s ceiling is a 24fps fact, not a constant.
7. **Policy contract (#7): FIXED + SHIPPED** (ad2bd2e): clipSeconds now
   assigned + persisted + survives unrelated saves; absent-videoMode
   default unified to 'i2v'. New scheduler state (MediaTypes, templates,
   ledger) lives in a SEPARATE versioned document
   (`/v1/kira/schedule/config`, schemaVersion + ETag) — never in the block
   the legacy writer reconstructs.
8. **Cost model (#8):** Σ median is rejected. Costs are sequence-dependent
   (model eviction/reload, drain waits, i2v = seed image + up-to-3 VLM QA +
   render). Use a normalized execution signature (type, dims, frames,
   steps, two-stage, decode mode, resident-model state), P95 for admission
   (median only as UI estimate), explicit transition costs, minimum sample
   counts + configured floors, estimate-version reset on recipe change.
   Interactive work doesn't debit the content QUOTA but does consume
   wall-clock capacity → forces a re-fit.
9. **Unlimited + backlog (#9):** unlimited mode becomes a lowest-priority
   FILLER policy (`fillRemainingWith: image.<tier>`, start only when
   conservativeCost ≤ time-to-next-protected-window). Backlog: bounded
   size, max age (per §1.5 lossiness: 1–2 slots), max attempts, oversized-
   reservation quarantine, deficit fairness, overload telemetry.
10. **Plan-time typing (#10):** the mixed coin-flip moves from execution to
    PLANNING: a reservation persists its concrete type, immutable recipe
    revision, mode, duration/fps/frames/dims/tuning, seed/template/
    suggestion linkage, reservation id + idempotency key. Film/neutral
    stays stills-only. Suggestion-box picks consumed by scheduled cycles
    are SCHEDULED — the TRIGGER (who initiated now) defines priority, not
    the content's source.
11. **Orthogonal axes (#11):** the two-class model splits into
    `priority: interactive|userBackground|scheduled` ×
    `resource: gpuDiffusion|gpuNonPausable|gpuInference|cpuMedia|external` ×
    `preemption: stepBoundary|phaseBoundary|none` × `shape: atomic|composite`
    + memory/runtime estimates. Upscale (sync, non-pausable) must join the
    admission broker before "any model sharing the GPU" is true; desktop
    face swap stays out of scope until server-brokered; montage is cpuMedia
    and never consumes diffusion capacity.
12. **Durability (#12):** reservation state machine
    `reserved → dispatching → engineQueued → running → terminal|lost`, with
    the reservation id accepted idempotently by the engine, persisted
    atomically around submission, reconciled BY ID (not by discovering
    output files). Per-kind retry policy.
13. **Migration (#13):** the 8-step shadow cutover is adopted verbatim —
    contracts first, shadow planner (persists, submits nothing), compare
    against live ticks, fence the old emitter with a scheduler generation
    id, drain, activate one slot at a stable boundary, keep a rollback
    switch. Slot ids anchor to timezone+epoch, not process uptime.
14. **Four strata (#14):** the daemon's in-process priority queue is the
    missing stratum — most scheduled work waits THERE, not in the engine.
    Correction: engine video jobs already appear in /v1/queue (draft's
    premise was stale); the gaps are the daemon queue, cloud video,
    storyboard parents, montage/upscale, and reservation metadata. The
    desktop consumes a versioned aggregate DTO (reservationId/engineJobId/
    daemonJobId/parent, stratum, priority, resource, MediaType+revision,
    timestamps, estimates, phase/lease state, per-row allowed actions) with
    versioned mutation endpoints.

**Build order = the review's fold-in sequence:** lease/phase admission →
atomic priority + honest SLA → fps/policy contracts → conservative planner
→ taxonomy/durability → shadow cutover → desktop aggregation.

---

## Rev 4: Variable video size — ladders (Todd 2026-08-04; Codex findings 14-22 folded)

**Requirement:** "for the scheduler and for motion and Kira, we need variable
video size for delivery and production efficiencies." Size is a range WITHIN
a MediaType (Todd confirmed: "tied to media types" — yes).

### Rung model (Codex #16, #17)

Each MediaType carries an ORDERED ladder of rungs. A rung is a stable,
versioned, engine-complete signature — never an informal label:

- `rungId` (stable), `ladderRevision`, explicit **quality ordering** (the
  ladder's order IS the ordering; 1080p/5s vs 720p/12s comparisons are
  resolved by the MediaType author at spec time, not at runtime),
- `minAcceptable` marker per MediaType (the floor rung),
- eligibility predicates (`idleOnly`, `overnightOnly`),
- full engine-valid parameters: nominal dims budget + orientation, seconds,
  fps, steps, twoStage, audio, guidance profile. Speech and refine variants
  are DISTINCT rung signatures, not multipliers.
- **Resolved geometry** (Codex #16): i2v treats dims as an area budget
  (source aspect + /64 snap decide actuals; two-stage floors apply). A
  reservation persists nominal budget → resolved stage-1 geometry → final
  geometry + frames + fps once its source is known; cost, memory, and
  delivery checks re-run against RESOLVED values.

### Selection: global packing, not greedy (Codex #14 — BLOCKER)

Never per-reservation "largest that fits." Planning a window:
1. Place EVERY mandatory reservation at its `minAcceptable` rung first —
   if even minima don't fit, apply Rev 2 push/quarantine rules explicitly.
2. Distribute remaining capacity as UPGRADES by priority × quality-utility
   (highest-priority reservations climb rungs first).
This preserves "never dropped slots" as an invariant of step 1, not a hope.

### Admission cost contract (Codex #15 — BLOCKER; #20)

- Fit uses the **full conservative P95 execution signature** per rung:
  seed generation, VLM/optimizer attempts, model residency transitions
  (incl. the krea2 ~65s post-video reload), admission drains, denoise,
  refine, decode, mux. The scalar rung estimate is a UI baseline only.
- Each rung also carries peak-memory and uninterruptible-phase estimates
  for the GPU governor. A time-fitted rung failing memory/phase admission
  at dispatch: atomic downgrade-and-refit → else defer → else quarantine.
  Never a silent engine-queue entry, never burns a protected window on
  retries.

### Mutation rules (Codex #18)

A rung may change ONLY while the reservation is reserved/daemon-queued and
before any dimension-dependent work (seed render, prompt optimization, QA)
has begun — never after dispatching/engineQueued/running. Changes are
revision-checked amendments persisting {selectedRung, reason, alternatives}
atomically. Manual fixed-rung and campaign-minimum reservations are
non-shrinkable.

### Stability and fairness (Codex #19)

- Freeze horizon: rung selection locks N minutes before dispatch.
- Downgrade hysteresis + monotonic downgrade within a slot (no A→B→A).
- Upgrade cutoff: no upgrades inside the freeze horizon.
- **Quality debt**: reservations repeatedly forced to low rungs accrue
  age-weighted debt that raises their upgrade priority, so idle-only top
  rungs eventually win capacity instead of always losing to throughput.

### Delivery budget (Codex #21)

The bits/px formula is an ESTIMATE. Rungs carry a conservative byte budget
(cap × headroom factor); encoders run with max-rate bounds; POST-ENCODE
byte validation is mandatory, with a bounded fallback chain (re-encode at
lower bitrate → delivery-rung transcode) before delivery. The delivered
encoding is recorded separately from the render rung in the ledger.

### Shared ladder API (Codex #22)

Ladders live in the versioned schedule config, server-owned:
{ladderId, rungs[], revision, costEstimateVersion, confidence}. Ledger and
queue DTOs carry {selectedRung, selectionReason, alternatives, mutable}.
**Motion fetches the same server ladder** and submits `rungId + revision`
— the S/M/L/XL presets are rendered FROM the ladder, never hard-coded, so
manual and scheduled renders cannot drift.

### Pre-#23 quick wins (unchanged, now bounded)

- Kira cycle 5s clipSeconds: DONE 2026-08-04 (live).
- Motion size presets: DEFERRED until the ladder API exists (Codex #22 —
  hard-coding S/M/L first would create the drift the API prevents).
- Audio cost is rung-invariant (~negligible tokens).

---

## Gate 2 — Todd's ad-server semantics (verbatim, 2026-08-04)

Captured before any policy is encoded (build-brief requirement). Todd was CTO
of Advertising.com 1999–2004; this framing is the authoritative domain design.
Questions posed by the build session, answers verbatim.

**Q1 — Priority classes.** Beyond INTERACTIVE vs SCHEDULED, how is demand
ranked (user asks, Muse on-demand, campaign specials, filler)? Guaranteed vs
best-effort tiers?

> "User or Muse requests are highest priority"

**Q2 — Underdelivery / makegood.** When a slot can't fit its booking (or a
campaign underdelivers over a night), what's the behavior — drop-and-forget,
roll a makegood into future capacity, or in between? How long is a makegood
owed?

> "User or Muse requests are delivered are never dropped. Displaced units are
> made good when slot is available."

**Q3 — "Sold out".** What does a fully-booked interactive window mean to a
User asking in chat — hard reject with a wait estimate, queue-and-warn, or bump
scheduled content? Where is interactive "sold out"?

> "When sold out or high priority collision, Schedule is moved back to
> accommodate."

### Design consequences (encode these; they refine, not replace, rev 2/§1.5)

1. **Two-not-four priority, and the top class is guaranteed.** Todd collapses
   "user asks" and "Muse on-demand" into ONE top class — both are INTERACTIVE
   and both are **guaranteed-delivery: never dropped**. This confirms the
   §Priority two-class model (INTERACTIVE > SCHEDULED) and resolves the Q1
   sub-question: there is no separate "campaign special" or "filler" priority
   tier ABOVE scheduled — campaigns and filler are SCHEDULED-class demand that
   competes for slot capacity (campaign id / filler policy ride the reservation
   as metadata, per §2.5 and rev 2 #9, but do not outrank interactive).

2. **Makegood applies to DISPLACED SCHEDULED units, and the guarantee is
   asymmetric.** The lossiness of §1.5 ("a slot that doesn't fill is dead air,
   not debt") is now scoped precisely:
   - INTERACTIVE (user/Muse) requests are a HARD guarantee — never dropped,
     never expired. They are not subject to the 1–2-slot backlog aging.
   - A SCHEDULED unit that an interactive bump DISPLACES is not dead air — it
     is **owed a makegood** and re-placed "when slot is available" (the next
     slot with residual capacity that fits it). This is a rolling makegood
     queue, distinct from the pure-lossy overflow of the draft.
   - Reconciliation with §1.5 lossiness: the makegood still ages. A displaced
     scheduled unit rolls forward until a fitting slot opens OR it hits the
     backlog max age (rev 2 #9), whichever first — because stale JIT muse
     content still isn't worth rendering days later. So: interactive = owed
     forever (until rendered); scheduled makegood = owed until a slot avails or
     it ages out. The makegood is a distinct reservation flag
     (`makegoodFor: <displacedReservationId>`) so authorship/why is visible in
     the ledger.

3. **"Sold out" is never a hard reject — the schedule yields.** There is no
   user-facing "sold out" wait-estimate rejection. On a full interactive
   window OR a high-priority collision, **SCHEDULED content is moved back
   (deferred) to accommodate** the interactive request. This is exactly the
   rev 2 #4 bump (interactive enters at the FRONT; scheduled slides back and
   the ledger re-fits) bounded by rev 2 #5 (the RUNNING job is never killed —
   worst-case interactive wait = the running job's remainder, capped by the
   contiguous-occupancy admission rule). The only real "sold out" surface is
   the honest SLA: the wait is bounded by the currently-running job's P95
   remainder, not a refusal. The scheduler's job is to keep that remainder
   small (the interactive reserve + contiguous-occupancy cap), never to reject.

4. **What this pins for P1 policy encoding.** (a) Reservation priority is the
   two-class INTERACTIVE/SCHEDULED with interactive guaranteed. (b) A displaced
   scheduled reservation transitions to a makegood state (not `expired`) and
   re-enters packing at the next slot, carrying `makegoodFor` + inheriting the
   displaced unit's quality debt (rev 4 #19) so repeatedly-displaced units
   climb. (c) The packer's protected interactive windows are a HARD constraint
   the scheduled plan must yield to — encoded as the fixed-priority hard
   constraint already in §1.5, now confirmed by Q3.

---

## Gate 1 — GPU contention measurement (measured, 2026-08-04)

Full report: `reviews/gate1-contention-2026-08-04.md`. 67 min instrumented
against the live engine + local model while the content-scheduler ran normally.

**Result — the spec's layer-1-vs-layer-3 question is answered: layer 1 (the
inference lease) is REQUIRED.**

| Signal | Measured |
|---|---|
| Engine occupancy | 99% image-busy (GPU saturated by 24/7 content) |
| Interactive admission (submit→done, under load) | **p50 531s / p95 751s** (8.9–12.5 min FIFO wait) |
| LLM time-to-first-token under render load | **p50 2.5s / p95 47s / max 62s** |
| LLM TTFT > 10s | **21% of turns** (14/67) |
| LLM throughput under load | 3.8–5.4 tok/s (~4× the ~17 tok/s low-load ref) |
| VRAM, image renders | 33–37 GB (no video peaks this hour) |

The 62s max TTFT + 21%-over-10s reproduce the 2026-08-03 90s agent-loop timeout.
Windows alone (layer 2) cannot fix it — a render holds the GPU 5–6 min and tokens
starve INSIDE that span. The lease (pause at next step boundary, TTL ≤ 90s) caps
token latency at one denoise step vs the 12.5-min FIFO wait.

**Config seeds from this data:** interactiveReserveSec 300s (~17% of a 30-min
cycle); maxContiguousRenderSec 600s (a video + FIFO stacking is what produced the
12.5-min wait — the cap forces a following window); lease TTL ≤ 90s (validated —
62s waits fit); residency `videoToImage` ~65s (krea2 reload) as the dominant
sequence-dependent cost term.

**Cost-probe (Kira paused, 2026-08-04) — the "~20s image" reference was WRONG.**
Isolated krea2 image renders measured **200s and 345s** (10-step, ~1MP; plain
`/v1/generate/async`, warm). Effective image cost is **~200–350s (3–6 min)**,
an order of magnitude above the retired-mflux "~20s" — the plain-submit path
includes the polish pass (`polishStrength 0.6`) and/or per-job LoRA reload.
IMPLICATION: a 30-min slot (~1500s usable) fits only ~5–6 images, yet the live
config books avocado=6 img+1 vid, banana=3, apple=2 per cycle — and the LIVE
per-image adds optimizer + up-to-3 VLM QA. **The count-based scheduler is
already silently over-capacity and spilling** — exactly the lossy-by-accident
failure this design makes visible. Seed image rungs at median 250s / P95 350s.
Ask (zimage.swift): expose per-phase render timings on the status endpoint.

**Other limitations:** no clean idle TTFT baseline (system 99% busy); image-only
hour (no video memory/LLM peaks); video floors stay reference-seeded (~5–6
min/clip) pending a video cost-probe.

**Finding F-A (spec §2 correction):** §2 assumed "traces already record elapsed"
— they do NOT (`render-journal.ts` has no duration field, no kira path records
one). Cost learning (rev 2 #8) has no live data source today. Resolution: the
signature-keyed `scheduler/cost-store.ts` holds config-seeded floors now; the
elapsed-capture wiring (source: comfybox async status `durationMs`) lands with
the cutover (P3), not against live render paths in shadow P1.

---

## Yield policy + broker architecture (Todd, verbatim, 2026-08-05)

Direction, verbatim:
> "Kira's create tab has to route through the campaign mgr."
> "It uses resources and manages delivery. Kira can use it. Bree can use it. User can use it."
> "Its requests are treated as priority yield orders. Tier 2 subnet. Todd and Bree are tier 1. Network traffic tier 3."
> "Todd wins. Content stream is network traffic."

### Delivery IS the broker — the single front door
**Naming (Todd 2026-08-05):** the manager/tab is **Delivery** — it manages all GPU
traffic and its routing, so it is named for its job, not for one of its inputs. A
**campaign** is a first-class demand-source object *within* Delivery (a themed,
authored lineup — Kira's or the user's — that books reservations), one input
alongside create-tab asks, Bree/User asks, and the stream. Delivery is the
manager; a campaign is one of the things it delivers.

Delivery is the shared GPU **resource-and-delivery broker**. It is the
SOLE admission point to the engine — no render reaches the GPU without going
through it. Three principals are clients: **User (Todd), Bree, Kira**. It does two
jobs at once: allocate the scarce GPU (yield management) and manage delivery
(surface routing, encode budget, caption behavior).

### Priority tiers — a QoS stack (supersedes Gate 2 "user OR Muse = highest")
Strict ranking; higher bumps lower via the reservation fast lane; the RUNNING
render is never killed (bounded wait, sized by the interactive reserve):

- **Tier 1 — Todd (absolute) > Bree.** Human + PM-agent. Todd wins ties within
  the tier. Bumps everything.
- **Tier 2 — Kira's create-tab requests** ("priority yield orders"). Real orders,
  yield-managed in, but defer to Tier 1.
- **Tier 3 — the 24/7 content stream ("network traffic").** Background filler.
  Yields to Tier 1 and Tier 2. Starts a render only when its conservative cost
  fits before the next protected window (the existing filler rule).

This REVISES §Priority / Gate 2: Kira's create work is no longer co-top with the
user — it sits below Todd+Bree at Tier 2.

### Kira's create tab routes THROUGH the broker (mandatory)
The create surface (scene-creator, storyboard, gallery Motion/Repair/HQ, direct
generate) submits **Tier-2 reservations** to the broker; it never calls the engine
directly. The broker owns dispatch, ABOVE the existing render-queue (still the
execution primitive, spec §6). This makes the broker load-bearing for the create
surface — not shadow-only there. The broker is thus the one place where every
GPU consumer (all three principals + the stream) is admitted, prioritized,
leased, and delivered.

### Consequences for the build (additive to what's built)
1. Reservation `priority` becomes the three tiers; each reservation carries a
   **principal tag** (todd | bree | kira | stream) → tier. The built
   priority×resource×preemption axes already express this; it's a relabel + tag.
2. **Per-principal fairness** so Tier-1/2 traffic can't be starved; Tier-3 is pure
   filler (lowest, opportunistic).
3. The **content stream (the current count-based tick) formally becomes Tier 3** —
   the lowest traffic class, routed through the broker like everything else. This
   reframes the migration: the old tick's output is simply the bottom subnet, not
   a peer scheduler to "cut over" from.
4. **Bree is a client → the engine-side lease (#1479) is load-bearing.** Bree and
   Kira are separate processes; only an engine-side broker can arbitrate the GPU
   between them. Cross-process admission is required for true multi-tenancy.

---

## GPU-actions reserve + swap padding (Todd, verbatim, 2026-08-07)

> "we will need padding for swaps and reserved slots for GPU actions stored in a
> message queue. Not sure what they might be but there is always something."

Two capacity reserves BEYOND content, both conservative-by-design:

### 1. Swap padding
Residency/swap costs (model reload, LoRA swap, the audio-mode transformer reload)
are point estimates that vary with cold cache, disk, and contention. The planner
PADS them (a `swapPaddingFactor` / headroom) so a slot never packs to the line
and then blows its budget when a swap runs long. Same spirit as rev 4 #15's
"full conservative P95", applied specifically to swaps — the residency term is
the least predictable cost in the system.

### 2. GPU-actions reserve (message-queue intake)
A reserved capacity slice — like the interactive reserve — for ad-hoc/ops GPU
work that is NOT content (images/videos/campaigns):

`capacitySec = cycleLength − overheadReserve − interactiveReserve − gpuActionsReserve`

- **Intake:** a message queue the broker drains into the reserve. **Generic by
  design** — a GPU action = `{id, estCostSec, resource, priority, payload}`, so a
  new action type needs no schema change ("not sure what they might be").
- **Known members today** (validating the class — they currently steal GPU time
  ad-hoc): upscale (gpuNonPausable — already flagged rev 2 #11 to join the
  broker), model warmup, LoRA scan/quarantine, VLM caption/QA passes, embeddings,
  face detection. The reserve + queue gives them a GOVERNED lane instead of
  colliding with content.
- **Priority/lane:** ops get their own reserve so they neither starve content nor
  are starved. Prerequisite actions (a warmup before a render) schedule
  just-in-time; maintenance actions are opportunistic filler within the reserve.

**Design principle:** reserve capacity you can't yet name. A scheduler packed to
100% of KNOWN work chokes on the first unplanned GPU action; the reserve + the
generic queue are the slack that keeps it robust. "There is always something."
