# Roadmap: LTX video conditioning — one primitive, four capabilities

**Status:** design/research · **Owner:** Todd · **Author:** Fable (2026-08-08)
**Repos:** engine `zimage.swift` (primary), daemon `coffeeshop-server` (wiring)
**Related:** LTX extend FDD `specs/ltx-extend-fdd.md` (#1527)

## 1. The unifying insight

Every "condition the video on structured input" feature rides ONE engine
primitive, verified in source:

- **`LTX2VideoCondition`** (`LTX2Conditioning.swift`): inject a clean encoded
  latent at a **frame index** with a **strength**; a **denoise mask** controls
  which frames are conditioned vs generated.
- **IC-control** (`LTX2Pipeline.swift:683`): a **union-control IC-LoRA** appends a
  source-encoded latent as an extra conditioning stream.
- **`LTX2LatentState`** carries the noisy latent + clean conditioning latent +
  masks (incl. faceMask/faceRef) through the sampler.

So i2v, keyframes, extend, and structural control are not four separate systems —
they are four *inputs* to the same conditioning machinery. That's why they belong
on one roadmap: each is "supply a different structured latent to a primitive that
already exists."

## 2. Verified current state

| Capability | Primitive | State |
|---|---|---|
| **Identity / reference** | IC-control union IC-LoRA, fed the SOURCE IMAGE | ✅ live (defaults ON for i2v — without it the subject morphs). MVP: reference only, `reference_downscale_factor=1`. |
| **First-frame (i2v)** | `LTX2VideoCondition` @ frameIndex 0 | ✅ live (the deployed i2v recipe). |
| **Keyframes** | `generateMultiKeyframe` — N conditions at frame positions, temporal self-attention tweens between them (no separate interpolation step) | ⚠️ **built in the engine**, but the daemon's `long_video` tool SPLICES separate renders instead → motion resets at each seam. Wiring gap. |
| **Extend / v2v** | tail-latent window as leading condition | ❌ only single-frame re-anchor (decode→PNG→re-encode→i2v). Degenerate. FDD #1527. |
| **Structural (pose/depth/canny)** | control-map latent sequence → union IC-LoRA | ❌ for VIDEO. Control maps exist in the daemon but only shape the **seed IMAGE** (depth end-to-end on krea2; canny/pose via z-image Fun-ControlNet-Union). The motion is then uncontrolled. `generate_video` accepts no control param. |

## 3. The build family (three items, shared plumbing)

### A. Tail-latent extend / v2v  (FDD #1527)
Condition a continuation on a WINDOW of the source's tail latents (no pixel
round-trip) so motion momentum carries. Same `LTX2VideoCondition` hook, applied to
frames at the head of the new segment from the prior segment's tail.
- **P1 (daemon-only, soak-safe):** wire the existing `extend_to_seconds` re-anchor
  through `generate_video` — usable now, labelled honest.
- **P2 (engine, soak-gated):** tail-latent conditioning — seamless.

### B. Structural motion control  (the directed-action unlock — highest value)
The union-control IC-LoRA is ALREADY LOADED and used for reference. Feed it a
**control-map latent SEQUENCE** (pose/depth/canny per frame) instead of (or beside)
the single reference, so the **action itself is directed** — drive a thrust/stride/
turn from a pose sequence instead of prompt-roulette.
- The daemon already produces pose/depth/canny maps (`control-maps.ts`) for the
  seed frame — extend that to a per-frame control TRACK for the video.
- Engine: accept a control-latent sequence as an IC-control stream (the union LoRA
  is control-agnostic — this is what "union-control" means). `generate_video` gains
  a `control_video` / `control_track` param.
- **Why it's top value:** it converts motion from a hope into a spec. Pairs with the
  face/identity anchors already live, so directed action stays on-model.
- Engine work → soak-gated. Daemon control-track authoring → not gated.

### C. Native multi-keyframe wiring  (free quality win)
Switch Kira's long-video path from splice (`long_video` → concat) to the engine's
native `generateMultiKeyframe` (one render, tweened) so long clips stop resetting
motion at seams. Feature already built + tested (`docs/ltx2-multi-keyframe-fdd.md`);
this is pure daemon wiring to call the native path.
- **Mostly daemon-only** (call the existing engine entry) → largely soak-safe;
  confirm the engine entry needs no plist change.

## 4. Sequencing (value × cost)

1. **C — multi-keyframe wiring** (cheap, built, daemon-mostly): immediate long-clip
   quality with near-zero engine risk. Do first.
2. **A-P1 — extend wired to daemon** (cheap, daemon-only): unblocks test-short/
   extend-keepers now.
3. **B — structural motion control** (highest value, engine-gated): the directed-
   action capability. Spec the control-track contract now; build the engine stream
   when the soak clears.
4. **A-P2 — tail-latent extend** (engine-gated): seamless extend; shares B's
   latent-conditioning work.

## 5. Delivery integration
Each becomes a first-class MediaType/rung or mode in Delivery (#1469): `extend` (a
cheap continuation rung), `directed` (a control-tracked action MediaType), and
multi-keyframe as the default long-clip renderer. Conditioning inputs (control
track, source tail, keyframe set) ride the reservation as plan-time-typed params.

## 6. Not on this roadmap (separate)
Frame interpolation (`shouldInterpolate` off), camera-control v2 (latent viewpoint;
v1 is prompt-only), native 4K (memory-bound), prompt-travel. These are real but not
part of the shared conditioning primitive — list them, don't fold them in.

## 7. Open questions
- Structural control: does the loaded union IC-LoRA accept a MULTI-FRAME control
  latent, or only a single reference today? (Verify the IC-LoRA input rank before
  scoping B's engine work.)
- Multi-keyframe: does `generateMultiKeyframe` need any plist/env not currently set?
- Memory at 241f with an added control stream — check against the 65GB floor.
