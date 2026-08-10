# Gate 1 — GPU contention measurement (#23, 2026-08-04)

**Method.** 67 min instrumented against the live engine (ComfyBox :7870) + local
model (LM Studio :1234) while Kira's content-scheduler ran normally. Three probe
streams: 5s occupancy samples (770), streaming LLM latency probes every 45s (67),
and interactive image-admission probes every 5 min + two load-bursts (8). Raw:
`gate1.jsonl`; analyzer: `analyze.mjs`.

## Headline numbers

| Signal | Result | Meaning |
|---|---|---|
| Engine occupancy | 99% image-busy, 1% idle | the GPU is saturated by 24/7 content |
| **Interactive admission (submit→done)** | **p50 531s / p95 751s** (8.9–12.5 min) | a chat media request waits **up to 12.5 min** behind the FIFO |
| **LLM time-to-first-token under load** | **p50 2.5s / p95 47s / max 62s** | cognition stalls badly while a render holds the GPU |
| LLM TTFT > 10s | **21% of probes** (14/67) | ~1 in 5 turns suffers catastrophic first-token latency |
| LLM throughput under load | 3.8–5.4 tok/s (p50 4.5) | ~**4× slower** than the ~17 tok/s low-load reference |
| VRAM during image renders | 33–37 GB | well under the 65 GB OOM floor (no video peaks this hour) |

## What this decides (the spec's open question)

Rev 2 asked: *measure token latency during a render at production configs — that
number decides whether the inference-lease (layer 1) suffices or placement (layer
3) is also needed.* The answer:

- **Layer 1 (inference lease) is unambiguously required.** A 62s max TTFT and 21%
  of turns over 10s reproduce the Aug-3 90s agent-loop timeout. Scheduler windows
  alone (layer 2) cannot fix this — a single render holds the GPU 5–6 min and
  tokens starve *inside* that span. A lease that pauses the render at the next
  step boundary caps token latency at one denoise step (≤ the 90s TTL), vs the
  measured 12.5-min FIFO wait.
- **Layer 3 (CPU/ANE cognition placement) is worth prototyping but not blocking.**
  The lease should recover turn latency; layer 3 is the fallback for when a lease
  can't be granted (decode phase) and for turn-critical paths. Measure lease
  efficacy first, then decide.

## Sizing implications folded into config

- **`interactiveReserveSec` / windows** — the reserve exists so interactive MEDIA
  renders have GPU-time and so a protected window follows every long render. The
  lease (not the reserve) is what bounds *token* latency. Kept at 300s of the
  30-min cycle (~17% held), revisited once the lease lands and we can measure
  interactive media wait directly.
- **`maxContiguousRenderSec` = 600s** — a dream.vignette (~5–6 min) plus FIFO
  stacking is exactly what produced the 12.5-min wait. The cap forces a gap after
  long renders. Video single-renders inherently exceed interactive tolerance →
  confirms the lease is mandatory for video windows, not optional.
- **Lease TTL ≤ 90s** (GPUQuiescenceGate) — validated: observed cognition waits
  (max 62s) fit inside a 90s TTL, so one lease covers a stalled turn's burst.
- **Residency cost `videoToImage` ~65s** (krea2 reload) carried in the cost store
  as the dominant sequence-dependent term.

## Limitations (honest)

1. **No clean idle TTFT baseline.** The live scheduler kept the GPU 99% busy; the
   two brief idle gaps (~20s each) never coincided with a 45s-cadence LLM probe.
   The ~17 tok/s / ~1.3s TTFT low-load figure is from the smoke run (during video
   decode, near-idle), not a true idle. Not pausing live content to force one.
2. **Isolated per-rung render cost — measured 2026-08-04 (Kira paused).** See the
   cost-probe addendum below; the "~20s image" reference was WRONG for this
   pipeline.
3. **Image-only hour.** Occupancy was 99% image; no video-phase memory peaks or
   video-condition LLM probes landed. Video VRAM (~65 GB envelope) is from prior
   measurement, not this run.

## Cost-probe addendum — isolated image render cost (2026-08-04, Kira paused)

Todd authorized pausing Kira to measure against an idle engine. With no FIFO
wait, submit→done IS the render cost. Plain `/v1/generate/async` submits (krea2
warm from Kira's prior renders):

| Rung | submit→done |
|---|---|
| image 10-step 832×1216 | **200s** |
| image 10-step 1024×1024 | **345s** |
| image 25-step 832×1216 | inconclusive (poll deadline; Kira resumed mid-probe) |

**The load-bearing correction:** a krea2 image render here is **~200–350s (3–6
min)**, an order of magnitude above the "~20s" reference (which was the retired
mflux service). The plain-submit path almost certainly includes the default
polish/refine pass (config `polishStrength 0.6`, `polishCheckpoint krea2`)
and/or a per-job LoRA reload — the 200-vs-345s spread at near-equal pixel counts
points at a variable setup/reload term, not pure denoise. The engine returns no
per-phase breakdown, so it can't be decomposed client-side (`render_ms` was
null — the status never surfaced a `rendering` state to poll).

**Why this matters — the current count-based config is already over-capacity.**
A 30-min slot has ~1500s of usable capacity (30min − overhead − interactive
reserve). At ~270s/image that is ~5–6 images MAX. The live config schedules
avocado=6 images + 1 video, banana=3, apple=2 per 30-min cycle — and the LIVE
per-image cost is HIGHER still (it adds prompt optimization + up-to-3 VLM QA
passes on top of the raw render). So the count-based scheduler is already
silently spilling: it books more than a cycle can render, and the overflow just
vanishes. This is precisely the lossy-by-accident failure the reservation
scheduler makes visible and manages (fit-or-makegood, never silent drop).

**Actions:**
- Cost-store floors: seed image rungs at median 250s / P95 350s (conservative,
  from these anchors) when the default MediaType catalog is built. The unknown-
  signature defensive default (600/900s) already sits above these, so nothing
  under-books today.
- Video floors stay reference-seeded (~5–6 min/clip) pending a video cost-probe.
- **Engine instrumentation ask (zimage.swift):** expose per-phase render timings
  (load / denoise / refine / decode) on the status endpoint so cost learning
  (P3) can attribute cost and the governor can size max_uninterruptible per
  phase. Without it, floors stay whole-render blobs.
