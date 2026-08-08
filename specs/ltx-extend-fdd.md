# FDD: LTX video extend — render short, extend the keepers

**Status:** design · **Owner:** Todd · **Author:** Fable (research 2026-08-08)
**Repos:** engine `zimage.swift` (primary), daemon `coffeeshop-server` (wiring)

## 1. Motivation

The workflow Todd wants: **render a fast short clip (5s), evaluate it, then extend
only the good ones** to 10s+ — instead of committing to a slow 10–12s render up
front, and instead of the late-clip drift a monolithic 12s single-pass shows. This
is "test short, extend the keepers." It also fits Delivery's yield model: a 5s
render is cheap; extending a proven keeper is cheaper than re-rolling long.

## 2. Current state (verified from source 2026-08-08)

### What EXISTS
- **`extend_to_seconds` request param** (`EngineService.swift:1056`,
  `LTX2VideoGenerator.extendToSeconds`) + a desktop **"Extend (s)" slider** (0–12s)
  in `MotionView.swift`. So an extend surface exists — in the desktop app only.
- **`LTX2VideoGenerator.chunkPlan(framesPerChunk, extendToSeconds, fps)`** splits a
  target duration into continuation chunks.
- **`LastFrameExtractor.extractLastFrame`** — pulls a clip's LAST FRAME as a PNG
  (built for storyboard chaining, comfybox#237).
- **`reanchor_interval` / `reanchor_strength`** (`LTX2_REANCHOR_*`) — identity
  re-anchor WITHIN a render; currently **0 / off** (full-frame re-anchoring
  collapsed the inter-anchor gaps — a prior finding).
- **`LTX2LatentState`** threads latent state through `LTX2Sampler` (the hook the
  real fix needs).

### What it actually DOES (the gap)
`extend_to_seconds` is **single-frame re-anchor chunked continuation**: render chunk
1 → `LastFrameExtractor` decodes its last frame to PNG → re-encode → i2v that PNG as
the frame-0 condition of chunk 2 → repeat. Per boundary it does a **full pixel
round-trip** (decode → PNG → VAE re-encode), which:
- **resets motion state** — the continuation restarts from a still, so velocity and
  direction are lost (a walking subject "re-poses" instead of continuing the stride);
- **accumulates VAE encode/decode error** — each boundary softens/shifts color.

This is precisely the "degenerates in the second chunk" failure the daemon avoids by
forcing single-pass ≤12s. So the existing extend is the wrong mechanism, and it is
**not wired into the daemon** (Kira's 24/7, gallery Motion, and Telegram paths never
send `extend_to_seconds`).

### What is NOT built
**True LTX extend: conditioning a continuation on a WINDOW of tail *latents*** (not a
single decoded frame). LTX-Video supports conditioning frames at arbitrary temporal
positions; feeding the last K latent frames of clip A as the leading conditioning of
clip B keeps the motion in latent space — no pixel round-trip, no motion reset.

## 3. Design

### 3.1 The core change — tail-latent-window conditioning (engine)
Instead of decode-last-frame → re-encode → i2v, keep the **trailing K latent frames**
of the prior chunk in-memory and seed the new chunk's **leading K latent positions**
with them, marked as conditioning (frozen / low-noise) during denoise. LTX's
positional conditioning makes the model continue the motion across the overlap rather
than restart it.

- **Overlap window K:** a few latent frames (the LTX temporal-compression factor means
  K latents ≈ several pixel frames of motion context). Config-tunable
  (`LTX2_EXTEND_OVERLAP`, tier A).
- **Conditioning strength ramp:** the overlap latents held firm at the seam, relaxing
  over the first steps so the model owns the new motion — reuse the existing
  head-ramp pattern from STG/re-anchor.
- **Color continuity:** reuse `LTX2_COLOR_ANCHOR` (already 1.0) so exposure carries.
- **Hook:** extend `LTX2LatentState` to carry the prior chunk's tail latents; the
  sampler seeds + freezes the leading positions. No new model, no pixel round-trip.

### 3.2 Daemon wiring (coffeeshop-server)
- `generate_video` gains an **extend mode**: `{ source_video, extend_seconds }`. The
  daemon hands the source's tail to the engine (engine keeps latents when the source
  was rendered in-session; else it re-encodes the tail frames as a fallback) and
  receives the continuation, then **splices** onto the source (`video-splice.ts`
  already exists) — or the engine returns the full extended clip.
- A `generate_video` `extend` action for the **gallery** (a "＋4s" button next to 🎬
  Motion) and a Telegram surface, so keepers extend on demand.
- The 24/7 scheduler stays single-pass; extend is on-demand + Delivery-driven, not a
  new autonomous cost sink.

### 3.3 Phasing
- **P1 — wire the EXISTING extend to the daemon (fast, honest).** Expose
  `extend_to_seconds` through `generate_video` so the test-short/extend-keepers loop
  works TODAY, with a clear label that it's the re-anchor path (motion may restart at
  the seam). Unblocks the workflow while P2 lands. Daemon-only; no engine change.
- **P2 — tail-latent-window conditioning (the real fix, engine).** §3.1. Seamless
  extend. This is the value.
- **P3 — Delivery integration.** Extend becomes a cheap **continuation rung** on the
  video MediaType: a keeper's extend costs ~one chunk, not a full re-roll — high yield.

## 4. Recipe / params
- Reuse the deployed i2v recipe (strength 0.5, CFG 1.0, PinkCherry v1.8, NAG). Extend
  conditions on latents, so `strength` (frame-0 hold) is replaced at the seam by the
  overlap-latent hold — a SEPARATE `LTX2_EXTEND_STRENGTH` (default ~0.6, firmer than
  i2v's 0.5 because we WANT the tail to carry) is the tuning lever.
- **Drift is bounded but compounds per extension** — each extend is a fresh generation
  conditioned on the tail. Less than single-frame re-anchor (motion carries), but
  cap chained extends (e.g. ≤3 hops → ~20s) and **extend from a CLEAN tail** (extend
  the good short clips, not already-drifting ones — which is the whole workflow).

## 5. Acceptance criteria
- Extend a 5s keeper to 10s; the seam shows **continuous motion** (no re-pose) and no
  visible color step at the boundary (measure: per-frame LapVar + boundary Δbrightness
  ≤ the single-pass baseline).
- P1: `generate_video { source_video, extend_seconds }` returns a spliced clip via the
  existing re-anchor path, labelled as such.
- P2: the tail-latent path beats the re-anchor path on the seam-motion metric at
  Kira's production config (241f base + 4s extend).

## 6. Risks / open questions
- **In-session latent retention:** the engine must keep the source's tail latents to
  avoid a pixel round-trip. Easy when the source was just rendered; extending an
  ARBITRARY gallery clip (latents gone) falls back to re-encoding the tail frames —
  better than single-frame but not zero-loss. Q: is the "extend only fresh keepers"
  constraint acceptable, or do we need the re-encode fallback to be good too?
- **Memory:** holding tail latents + a second chunk's activations — check against the
  65GB floor at 241f.
- **Two-stage refine across the seam:** the refine pass is shared with continuation
  chunks today; confirm it doesn't re-introduce a boundary.
- **Soak-hold:** all engine work here is gated on the LTX soak being clear (no plist /
  render-path changes during the soak).

## 7. Recommendation
The workflow is worth building and the engine is ~halfway there (surface + chunk
planner + latent-state hook already exist). **P1 (wire existing extend to the daemon)
is a small daemon-only change that unblocks test-short/extend-keepers immediately;
P2 (tail-latent conditioning) is the engine change that makes it seamless and is the
real deliverable.** Both are soak-gated on the engine side; P1's daemon wiring is not.
