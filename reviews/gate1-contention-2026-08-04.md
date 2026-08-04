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
2. **No isolated per-rung render cost.** Admission probes measured submit→done
   (queue wait + render), dominated by FIFO wait — the right number for the SLA,
   but it does not isolate a warm per-rung render time. Cost-store floors are
   therefore seeded from documented references (krea2 image ~20s, i2v ~2–3 min,
   dream.vignette ~5–6 min) and refined by learning (P3). A tighter seed needs a
   controlled solo-render timing (brief scheduler pause) — available on request.
3. **Image-only hour.** Occupancy was 99% image; no video-phase memory peaks or
   video-condition LLM probes landed. Video VRAM (~65 GB envelope) is from prior
   measurement, not this run.
