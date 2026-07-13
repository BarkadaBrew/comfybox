# LTX-2 Multi-Keyframe ("Tween") Video — Feature Design Document

**Goal:** let a user (via Desktop UI or the server API) define a sequence of
keyframe images placed at specific points along a video's timeline, and have
LTX-2 generate a video that transitions between them — like traditional
animation in-betweening, but driven by the video model's own denoising
rather than an explicit interpolation algorithm. Scope per the requester:
**both** a pipeline capability and a UI (a new "Motion Timeline" tab, video-editor-style keyframe strip) **and** the same capability exposed
directly through the server API so external applications can define
keyframes + durations programmatically, not just the Desktop app.

This document is a handoff: the pipeline primitive is now built and
empirically verified (see below); the server API and UI are still to be
designed in detail and built by whoever picks this up next.

## Status

| Piece | State |
|---|---|
| Pipeline primitive (`LTX2Pipeline.generateMultiKeyframe`) | **Built + spike-verified** |
| Server API (`/v1/video/generate` keyframes+durations) | Not started — see Proposed Design below |
| Desktop UI ("Motion Timeline" tab, keyframe strip) | Not started — see Proposed Design below |

## Background: what already existed

`LTX2Conditioning.applyConditioning(state:conditions:)`
(`Sources/ZImage/LTX2/LTX2Conditioning.swift:103`) already accepted a
**list** of `LTX2VideoCondition` (each: `latent`, `frameIndex`, `strength`).
It works by dense array-slice replacement: for each condition it splices the
encoded image latent into the noisy `(B,C,F,H,W)` latent tensor over
`[frameIndex, frameIndex+condFrames)`, and sets the per-frame denoise mask to
`1.0 - strength` there (0 = clean/fixed, 1 = fully denoised) — this mask
convention was already confirmed against the Python reference's
`LTXVImgToVideoInplace` earlier in this project.

Until now this was **only ever called with one item**, always at
`frameIndex: 0` (`LTX2Pipeline.generateI2V`, single-frame image-to-video).
The multi-condition capability existed at the primitive level but had no
caller and no verification that it behaved sanely for `frameIndex > 0`.

### Why that needed checking before building anything on top

The Lightricks Python reference (`~/Projects/LTX-Video/ltx_video/pipelines/pipeline_ltx_video.py`)
handles `frame == 0` and `frame > 0` conditioning through **two different
mechanisms**: frame-0 is a dense in-place `lerp` (same as this Swift port),
but frame>0 guides are patchified into a **separate token sequence** with
explicit position-coordinate offsets and concatenated onto the main token
list before the transformer runs (`prepare_conditioning`, line ~1488-1541).
That raised a real question: does this Swift port's simpler
"always dense-splice, regardless of frame index" approach actually produce a
coherent result at frame > 0, or would it require porting that separate
offset-token mechanism too?

Reading `LTX2Pipeline.denoisingLoop` answered this before any test ran:
positions come from `createPositionGrid` computed **uniformly over the
whole dense grid** (`LTX2Pipeline.swift:472-475`), and per-token timesteps
are individually derived from the denoise mask
(`LTX2Pipeline.swift:568-...`, "I2V: per-token timesteps based on denoise
mask"). There is no separate offset-token path in this port at all — every
frame, clean or noisy, is just a position in one uniform sequence. This
architecture should generalize to conditioning at any frame index without
rework, unlike the Python reference's design. The empirical spike (below)
confirms this.

## What was built

`LTX2Pipeline.generateMultiKeyframe(inputIds:attentionMask:keyframes:width:height:numFrames:steps:seed:guidance:negativeInputIds:negativeAttentionMask:progressCallback:)`
— `Sources/ZImage/LTX2/LTX2Pipeline.swift`, added after `generateI2V`.

```swift
public struct Keyframe {
  public let image: MLXArray       // normalized pixel image, same format as generateI2V's `image:`
  public let videoFrameIndex: Int  // position in VIDEO frames (0 = first) — converted to latent frames internally
  public let strength: Float       // 1.0 = fully replace with the image at this point
}
```

Mirrors `generateI2V` exactly, except it encodes **N** keyframe images via
the VAE (each converted from a video-frame index to a latent-frame index
via `videoFrameIndex / temporalCompression`, clamped to `latF - 1`) and
passes all N as a `[LTX2VideoCondition]` to the existing
`applyConditioning`. No new conditioning math was needed — this is
literally "call the existing primitive with more than one item," now that
it's been verified to be safe to do so.

`LTX2VideoGenerator` (the higher-level wrapper the server uses) got two new
accessors, `loadedPipeline`/`loadedTokenizer`, so the spike (and any future
caller) can reach the underlying `LTX2Pipeline` for shapes `generate(request:)`
doesn't cover yet. `LTX2VideoRequest`/`generate(request:)` itself was
**not** changed — that's the server API work below.

## Empirical verification (the spike)

`Tests/ZImageIntegrationTests/LTX2MultiKeyframeSpike.swift` — loads the real
local LTX-2 distilled weights + Gemma-3 12B text encoder
(`/Volumes/Bolt/Models/ltx2-distilled`,
`~/.cache/huggingface/hub/models/unsloth/gemma-3-12b-it`), then generates a
33-frame (`latF=5`), 512×320 clip with two keyframes at `strength: 1.0`:
a red apple at `videoFrameIndex: 0` and an unrelated portrait photo at
`videoFrameIndex: 32` (the last frame), 8 denoising steps, seed 7.

Ran in ~92s total (49s load, 41s generate) on an M3 Max. No NaN, no shape
errors. Visual result (frames saved to `/tmp/ltx2-multi-keyframe-spike/`,
also an `.mp4`):

| Frame | Content |
|---|---|
| 0 | Apple — matches keyframe A exactly |
| 8 | Apple, slightly softened |
| 16 | **Already** essentially the portrait (keyframe B) |
| 24 | Portrait, sharper |
| 32 | Portrait — matches keyframe B exactly |

**Both findings matter:**

1. **Feasibility confirmed.** Both keyframes are respected at their correct
   positions with no corruption — the dense-splice primitive genuinely does
   generalize to non-zero frame indices in this port's architecture. No new
   pipeline math is required for the core capability; `generateMultiKeyframe`
   as built is sufficient.
2. **"Tween" is not a smooth cross-fade by default.** The transition happens
   as a fairly abrupt jump somewhere in the first third of the gap between
   the two `strength: 1.0` anchors, not a gradual multi-frame blend across
   the whole span. With only 3 free (unconditioned) latent frames between
   two fully-clean anchors, the model has little room to ease between them,
   and its own attention seems to commit to the destination early rather
   than blend continuously.

   Two confounding variables worth separating in follow-up testing before
   concluding the transition quality is a hard limitation:
   - **Frame budget**: this spike used a short 33-frame gap. A real timeline
     UI would likely allow (and should probably *recommend*, maybe even
     enforce a minimum for) more frames between distant keyframes — more
     free latent frames may produce a visibly more gradual blend.
   - **Keyframe similarity**: the two test images were deliberately
     unrelated (an object vs. a person) to make the transition easy to
     judge. There is no explicit interpolation algorithm anywhere in the
     reference implementation either — "tweening" is entirely emergent from
     the DiT's own temporal self-attention during denoising, which almost
     certainly blends much more naturally between semantically continuous
     keyframes (e.g. two poses of the same subject, or a continuous camera
     move) than between two arbitrary, unrelated images. The intended real
     usage (animating a coherent shot) is the more favorable case; this
     spike deliberately tested the harder, more visually legible case.

## Proposed Design (for the next session to detail and build)

### 1. Server API

Extend the video-generation request shape to carry a keyframe list instead
of (or alongside) the current single `initImagePath`. Current single-image
shape for reference: `LTX2VideoRequest` (`Sources/ZImage/LTX2/LTX2VideoGenerator.swift:32`)
and the server-side `LocalVideoRequest` (`Sources/ZImage/Server/WarmServer.swift:1409`,
route `POST /v1/video/generate`, `WarmServer.swift:1046`) both currently
have exactly one `imagePath: String?` / `imageBase64: String?`.

Proposed shape (exact field names/JSON conventions TBD by whoever builds
this — match this codebase's existing snake_case-over-the-wire /
camelCase-in-Swift convention):

```jsonc
{
  "prompt": "...",
  "width": 704, "height": 448, "fps": 24,
  "keyframes": [
    { "image_base64": "...", "frame_index": 0,   "strength": 1.0 },
    { "image_base64": "...", "frame_index": 48,  "strength": 1.0 },
    { "image_base64": "...", "frame_index": 96,  "strength": 0.8 }
  ],
  // OR, if the API should think in durations rather than absolute frames:
  // "keyframes": [{ "image_base64": "...", "duration_seconds": 2.0, "strength": 1.0 }, ...]
  "steps": 8, "seed": 42
}
```

Open questions to resolve when designing this:
- **Frames vs. duration**: does the API take absolute `frame_index` per
  keyframe (simple, matches the pipeline primitive directly) or a duration
  per *segment* between keyframes (more natural for external callers "who
  don't want to think in frames," but requires converting duration→frame
  index server-side, and needs a decision on what "duration" means when fps
  is also configurable)? The user's request specifically named "duration"
  as a first-class concept — lean toward duration-based unless it proves
  awkward.
- **Chunking / long sequences**: `LTX2VideoGenerator` already has an
  `extendToSeconds`/chunk-continuation mechanism for videos longer than one
  97-frame chunk (`ChunkPlan`, `LTX2VideoGenerator.swift:179-204`). Whether
  multi-keyframe requests spanning multiple chunks are in scope for v1, and
  if so how a keyframe landing exactly on a chunk boundary is handled, is
  unresolved — not tested in the spike (single chunk only).
- **Validation**: should the server enforce a minimum frame gap between
  consecutive keyframes, given the jump-cut finding above? Or leave it to
  the caller/UI to guide toward sensible gaps?
- Whether to add a new route (`POST /v1/video/generate/keyframes`) or
  extend the existing `/v1/video/generate` — the existing route already has
  a lot of single-image-specific logic (`LocalVideoRequest`); a new route
  may be cleaner than overloading it, but duplicates chunking/queue-wiring
  logic. Recommend deciding this by reading `queueListResponse`/
  `enqueueLocalVideo`'s current wiring first (`WarmServer.swift`) — note
  `#217`/durable-queue work (`QueuePersistence.swift`) currently only
  recovers `generate`/`lora_swap` kinds after a crash; a new video route
  would need the same treatment (or an explicit documented gap) to match.

### 2. Desktop UI — "Motion Timeline" tab

A new tab (separate from the existing single-image `MotionView`), video-editor-style:

- Horizontal keyframe strip: thumbnails in sequence, each showing its image
  + a duration/frame-position field.
- Add a keyframe via file picker or **drag-and-drop a PNG onto the strip**
  — reuse `Sources/ComfyBoxDesktop/Views/ImageDropSupport.swift`
  (`handleImageDrop(_:apply:)`), already built for `MotionView`'s single
  reference image and `GenerationView`'s img2img reference; the drop
  handler's shape (resolve a file URL or write dropped image bytes to a
  temp PNG, then hand the path to a callback) applies directly to "append a
  new keyframe slot."
- Reorder by dragging strip items (no existing reorderable-thumbnail
  component in this codebase to reuse — this part is new SwiftUI work;
  `Sources/ComfyBoxDesktop/Views/QueueView.swift:96` has a reorderable list
  but for pending jobs, not images, so it's a reference for interaction
  pattern only, not a component to reuse directly).
- Per-keyframe duration/strength controls, a running total-duration
  display, and (given the jump-cut finding) probably a soft warning when a
  gap between two keyframes is very short.
- Wire the "Generate" action to the new server API above via
  `EngineService` (mirrors `EngineService.VideoRequest`,
  `Sources/ComfyBoxDesktop/EngineService.swift:968-1001`, which is
  currently single-image scalar and needs the equivalent array field).

### 3. Suggested phased build order for the next session

1. Decide the server API shape (frames vs. duration, chunking scope) —
   the open questions above.
2. Extend `LTX2VideoRequest` + `LocalVideoRequest` + the route handler to
   accept keyframes, calling `LTX2Pipeline.generateMultiKeyframe` (already
   built) instead of `generateI2V` when more than one keyframe is present.
3. Extend `EngineService.VideoRequest` + wire a minimal UI (even just a
   text-entry list of paths+frames) to confirm the server round-trip works
   before investing in the full timeline-strip UI.
4. Build the "Motion Timeline" tab UI proper (keyframe strip, drag-reorder,
   drag-and-drop-in via `ImageDropSupport.swift`, duration controls).
5. Re-run a spike-style real-weights test with the finished UI/API path,
   using keyframes closer to real intended usage (continuous
   subject/scene, more frames between keyframes) to judge whether the
   jump-cut behavior seen in the initial spike is actually a problem for
   real usage or was an artifact of the deliberately-extreme test case.

## Files touched so far

- `Sources/ZImage/LTX2/LTX2Pipeline.swift` — `Keyframe` struct +
  `generateMultiKeyframe(...)`.
- `Sources/ZImage/LTX2/LTX2VideoGenerator.swift` — `loadedPipeline`/
  `loadedTokenizer` accessors.
- `Tests/ZImageIntegrationTests/LTX2MultiKeyframeSpike.swift` — the
  verification spike (real-weights integration test, `XCTSkip`s if the
  local LTX-2/Gemma weights aren't present, matching the existing
  `LTX2IntegrationTest.swift` convention).
