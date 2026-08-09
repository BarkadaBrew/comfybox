# FDD: Continuous scene from keyframes — API + daemon + Motion Timeline UI

**Status:** design · **Owner:** Todd · **Author:** Fable (2026-08-08)
**Builds on:** `docs/ltx2-multi-keyframe-fdd.md` (primitive **built + spike-verified**)
**Repos:** engine `zimage.swift` (API), daemon `coffeeshop-server` (authoring/tool),
desktop (UI). Related: video-conditioning roadmap (#1527), Delivery (#1469).

## 1. What's already done (don't rebuild)
`LTX2Pipeline.generateMultiKeyframe(keyframes:…)` is built and verified: N
keyframes, each `{image, videoFrameIndex, strength}`, spliced into the dense latent
grid; the DiT's own temporal self-attention tweens between them in ONE render. A
2-keyframe spike landed both anchors correctly, no corruption. Missing: the server
API, a daemon authoring path, and a UI — this doc specs those.

## 2. The load-bearing constraint (from the spike): transition smoothness
The spike found the tween is **not** a smooth cross-fade by default — with only ~3
free latent frames between two `strength:1.0` anchors of *unrelated* images, the
model **jumps** to the destination in the first third of the gap. Two levers fix it,
and BOTH are things we control:

1. **Frame budget between keyframes** — more free frames = more room to ease. The
   design must enforce a **minimum gap** (in latent frames) between adjacent
   keyframes; the daemon spaces them accordingly and refuses to pack keyframes
   tighter than the floor.
2. **Keyframe continuity** — semantically close keyframes blend naturally (the
   tween is emergent, so small deltas ease; big jumps snap). Our
   `keyframe-sequence-gen` already produces on-model, img2img-chained keyframes
   (shared seed → wardrobe/identity/setting carry), so beats are continuous by
   construction. This is the difference between the spike's worst case (apple→
   portrait) and the real use case (Kira, pose A → pose B of the same scene).

**Design rule:** continuous scenes are authored from a *continuity-chained keyframe
set* with an enforced minimum inter-keyframe frame budget. A third lever —
anchoring interior keyframes at `strength < 1.0` (soft waypoints) so the model eases
through rather than snapping — is a tuning knob to validate.

## 3. Server API (engine — extends `/v1/video/generate`)
Backward-compatible: add an optional `keyframes` array. Absent/≤1 entry at position 0
→ today's i2v/t2v path unchanged. Present with positions > 0 → `generateMultiKeyframe`.

```jsonc
POST /v1/video/generate
{
  "prompt": "...", "width": 704, "height": 1280, "fps": 24, "steps": 8,
  "seed": 7, "duration_seconds": 10,          // → numFrames = snap1p8k(dur*fps), ≤289
  "keyframes": [
    { "image_path": "K0.png", "position_seconds": 0.0, "strength": 1.0 },
    { "image_path": "K1.png", "position_seconds": 5.0, "strength": 0.9 },
    { "image_path": "K2.png", "position_seconds": 10.0, "strength": 1.0 }
  ]
}
```
- `position_seconds → videoFrameIndex = round(pos*fps)`, then latent-frame convert
  (÷ temporalCompression) as the primitive already does.
- Validate: positions ascending, within `[0, numFrames)`, and each gap ≥
  `MIN_KEYFRAME_GAP_FRAMES` (reject with a clear error naming the offending pair).
- Wire through `LTX2VideoRequest`/`generate(request:)` (the accessors
  `loadedPipeline`/`loadedTokenizer` already exist for reaching the pipeline).

## 4. Daemon path (`build_scene` tool, coffeeshop-server)
A `build_scene` BuiltInTool (companion side, like `long_video`):
- **Input:** either explicit keyframe images, OR a **scene storyboard** (ordered
  beats as prompts). For the storyboard case the daemon renders each beat into an
  on-model keyframe via `keyframe-sequence-gen` (shared seed + img2img chain →
  continuity — §2 lever 2).
- **Placement:** distribute keyframes across `duration_seconds`, enforcing the
  min-gap floor; if beats × min-gap > the 289f window, cap the scene at the window
  and hand the overflow to **extend** (roadmap A) to chain a second window.
- **Render:** one call to the engine multi-keyframe API → one continuous mp4.
- **Kira-authorable** (AI-native paradigm): "describe the scene" → local model drafts
  the beat storyboard → keyframes → continuous scene. She composes the arc; the model
  fills the motion.
- Replaces the splice-based `long_video` as the default long/continuous path (splice
  stays only for deliberately-cut sequences).

## 5. Motion Timeline UI (desktop, Swift)
The FDD's proposed "Motion Timeline" tab, designed as a sibling of the Delivery tab
(same design session / language):
- A horizontal **keyframe strip** over a time axis: drop keyframe thumbnails at
  positions, drag to reposition, set per-keyframe **strength**, set total
  duration/fps.
- Visual **min-gap guides** (shaded "too tight" zones) so the smoothness floor is
  enforced in the UI, not just the API.
- Preview + Render via the API; result plays inline.
- Reuses the Delivery tab's dark mission-control language.

## 6. Delivery integration
A **`scene` MediaType** (#1469): a continuous multi-keyframe render as a plan-time-
typed reservation — keyframes, positions, duration, and the continuity-chain seed are
its resolved params. Costlier than one i2v (N keyframe encodes + one long render) →
its own rung. Kira campaigns can book `scene` beats.

## 7. Phasing
1. **Engine API** (`keyframes` on `/v1/video/generate`) — soak-gated. Small: the
   primitive is built; this is request plumbing + validation.
2. **Daemon `build_scene`** — needs the API; design now, build when it lands. Not
   itself soak-gated once the API exists.
3. **Motion Timeline UI** — with the Delivery UI session.
4. **Delivery `scene` MediaType.**

## 8. Acceptance criteria
- 3 on-model keyframes (Kira, continuity-chained) → ONE continuous ≤12s clip that
  hits each beat and transitions **visibly gradually** (not the spike's abrupt jump).
- Metric: no single-frame content discontinuity > threshold across a gap that meets
  the min-budget floor (vs the spike's tight-budget/dissimilar baseline).
- API rejects sub-min-gap keyframe spacing with a clear error.

## 9. Open questions
- **MIN_KEYFRAME_GAP_FRAMES value** — the spike flagged this needs a follow-up test
  (transition gradualness vs frame budget vs keyframe similarity). Set the floor from
  that data; until then, a conservative default (e.g. ≥ half the model window between
  distant keyframes).
- **Interior-keyframe strength schedule** — do soft (`<1.0`) interior anchors ease
  the tween? Validate.
- **>12s continuous scenes** — chain windows via extend (roadmap A); confirm the seam
  between a multi-keyframe window and its extend is clean.
- **Audio across a multi-keyframe scene** — A/V default-on: does the audio branch
  behave across mid-timeline conditioning? Verify.
