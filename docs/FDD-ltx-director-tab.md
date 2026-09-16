# FDD: "Director" — a timeline editor tab for LTX-2.3 in CoffeeShop Desktop

Status: DRAFT for review (2026-09-16). Requested by Todd: "clone this product as a tab for
desktop app and make an FDD for the initiative" — the product being
[WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI), whose
flagship node **LTX Director 2.0** is "A Complete Timeline Editor For LTX 2.3".

This document is the design. Nothing here is built. §9 asks for the decisions.

## 1. Summary

Add a **Director** tab to CoffeeShop Desktop: a video-editor-style timeline that drives our
own LTX-2.3 engine. A user lays keyframe images, per-segment prompts, imported audio and
(later) a reference video on tracks, presses Generate, and gets one clip at the production
recipe. Timelines save and load as JSON. The engine gains one new route that compiles a
timeline into the LTX-2.3 conditioning we already have (multi-keyframe, chunk continuation,
audio) plus three new conditioning primitives (per-segment prompt scheduling, imported-audio
inpainting, temporal retake masks). Everything is native Swift/MLX; the upstream Python nodes
are the behavioural reference, not code we run or copy.

What the user gets that the Motion tab cannot do today:

| Capability | Motion tab today | Director |
|---|---|---|
| First / middle / last frame images | one reference image | any number, each at a time on the timeline, each with a strength, any one markable as the end frame |
| Prompt | one | a global prompt plus a prompt per segment, "prompt relay" |
| Longer than one chunk | Extend (s) slider, one prompt | timeline of any length; the compiler plans chunks and prompts per chunk |
| Audio | generated only | generated, or imported clips placed on an audio track with the gaps generated ("audio inpainting") |
| Fix one part of a clip | re-render the whole thing | Retake: select a time range of an existing clip and regenerate only that range |
| Reference motion | none | a reference-video track (Phase 2) |
| Project file | none | `.cbdirector` JSON with every asset and setting |

## 2. The upstream product, as a mechanism (read from its source, 2026-09-16)

LTX Director is a ComfyUI custom node (Python, GPL-3.0, ~2k stars) plus a browser-side
timeline widget. Its `ltx_director.py` (62 KB) owns the timeline JSON and outputs
conditioning; `ltx_director_guide.py` (35 KB) turns the timeline into LTX guide latents. What
it actually does:

**Timeline JSON.** Top level: `global_prompt`, `segments[]` (image/video/audio clips: `type`,
`start`, `length` in frames, `imageFile`/`videoFile`/`audioFile` or base64, `prompt`,
`trimStart`, `isEndFrame`, `strength`), `audioSegments[]`, `motionSegments[]`, and the retake
block (`retakeMode`, `retakeVideo`, `retakeStart`, `retakeLength`, `retakeStrength`,
`retake_global_prompt`). Settings: `frame_rate` (24), `start_frame`/`end_frame`, resize
method (`maintain aspect ratio | stretch to fit | pad | pad green | crop`), `divisible_by`
32, `img_compression` CRF applied to guide images, per-image `guide_strength`.

**Keyframes → guides.** Each image is VAE-encoded and appended at
`latent_idx = ceil(frame_idx / time_scale)` with its strength; a guide at frame 0 or a
single-frame guide gets the "causal fix" (first frame prepended). Latent length is
`ceil((frames − 1) / 8) × 8 + 1`. This is the same primitive as our
`LTX2Conditioning.applyConditioning` list of `(latent, frameIndex, strength)`.

**Prompt relay.** The global prompt conditions every frame; local prompts (pipe-delimited)
map to token ranges, and `build_segments` / `create_mask_fn` build attention masks so each
latent time range attends to its own segment's tokens. Single-segment timelines bypass the
masking "for maximum speed". `segment_lengths` are converted from pixel frames to latent
frames by largest-remainder rounding.

**Audio.** Imported clips are merged onto a 44.1 kHz stereo timeline
(`_build_combined_audio`), encoded with the audio VAE into the audio latent, and the gaps
carry a 3-D `noise_mask` so the model generates only where nothing was imported
("inpaint_audio"). `override_audio` takes the audio from the reference video instead.

**Reference video ("IC-LoRA track").** A dropped video is resampled to the timeline rate,
VAE-encoded, appended as a motion guide with `videoStrength` (1.0) and
`videoAttentionStrength` (0.65), and registered as an attention entry for the "LTX
Ingredients" IC-LoRA (weights loaded via `load_lora_for_models`, downscale factor read from
the LoRA metadata). A mid-timeline motion guide gets a two-frame noise ramp
(`1 + 0.75·s`, `1 + 0.35·s`) so the splice does not pop.

**Retake.** The base video is VAE-encoded and pasted into the latent; a temporal noise mask
is 0 (frozen) outside `[retakeStart, retakeStart + retakeLength)` and 1 inside; if the
range covers the whole clip the base is skipped. The author's own note: "Retake mode is not
potent enough", overhaul planned.

**Other nodes** (Multi Image Loader with resize + LTXVPreprocess, LTX Sequencer/Keyframer,
Speech Length Calculator — words in quotes → seconds, Load Video/Audio UI with trim and
aspect presets) are conveniences around the same primitives.

**Performance note.** v2.0.4 claims 1.5–3× from caching the patched model between runs and
bypassing prompt relay for single segments.

## 3. What we already have

Engine (`Sources/ZImage/LTX2`, `Sources/ZImage/Server`):

- `LTX2Pipeline.generateMultiKeyframe` — built and spike-verified
  (`docs/ltx2-multi-keyframe-fdd.md`): any number of `Keyframe(image, videoFrameIndex,
  strength)` spliced into the latent by `LTX2Conditioning.applyConditioning`. Not yet on the
  wire. Audio is refused together with mid-pass identity re-anchoring.
- i2v with `extendToSeconds` chunk continuation (`ChunkPlan`, last-frame carry-over), the
  289-frame single-pass window, 24 fps, production recipe via `LTX2ConfigResolver`
  (euler, 10-step schedule, NAG 5/0.25/2.5, STG 0.3 flat, color anchor 1, audio on).
- `POST /v1/video/extend` and `/v1/video/rerender` (winner actions), `/v1/video/generate/async`
  job model, `/v1/video/status/{id}`, traces with rating/promote.
- Storyboard engine (`Sources/ZImage/Video/StoryboardTypes.swift`, `POST /v1/storyboard/render`):
  shots with `prompt`, `durationS`, `anchorImage`, an optional `Insert` (spatial inpaint with
  mask region/grow/feather), transitions, output; seamless last-frame chaining.
- Montage compose (`POST /v1/montage/compose`: cut / fade / dissolve / ken-burns).
- Temporal upscaler ×2 (opt-in), native audio generation and mux, vocoder.
- ComfyUI workflow import (`/v1/workflows/import`, async run) — an oracle, not a serving path.

Desktop (`Sources/ComfyBoxDesktop`):

- `MotionView`: one reference image (drop via `ImageDropSupport`), resolution, duration,
  motion fps, Audio toggle (2026-09-16), steps, strength, Extend (s), advanced tuning,
  LoRA picker, submit-and-poll progress, result preview.
- `EngineService.VideoRequest` (single `initImagePath`, `extendToSeconds`, `audio`, `tuning`).
- Tabs are an `AppTab` enum in `ComfyBoxDesktopApp.swift` (Motion sits in the Create section).

Not present anywhere: per-segment prompt conditioning, imported-audio conditioning, temporal
(time-range) masks, reference-video guides, IC-LoRA weights, a timeline document.

## 4. Design

### 4.1 The timeline document (`DirectorTimeline`, engine-agnostic, versioned)

`Sources/ZImage/Video/DirectorTypes.swift`, Codable, snake_case on the wire, `.cbdirector`
on disk (a JSON file; assets referenced by path, with an optional embedded base64 fallback
exactly like upstream so a project file stays portable).

```jsonc
{
  "version": 1,
  "settings": { "fps": 24, "width": 576, "height": 896, "seconds": 12.0, "seed": 771144,
                "preset": "kira-video-avocado", "loras": [{"path": "...", "scale": 1.0}],
                "resize": "crop" },              // maintain | crop | pad | stretch
  "global_prompt": "...",
  "keyframes": [ { "id": "k1", "image_path": "...", "image_base64": null,
                   "frame": 0, "strength": 1.0, "is_end_frame": false, "compression": 0 } ],
  "prompt_segments": [ { "id": "p1", "start_frame": 0, "length_frames": 96, "prompt": "..." } ],
  "audio_clips": [ { "id": "a1", "audio_path": "...", "start_frame": 0, "length_frames": 120,
                     "trim_start_frames": 0, "gain": 1.0 } ],
  "audio": { "mode": "generated" },             // generated | imported | inpaint
  "reference_clips": [ { "id": "r1", "video_path": "...", "start_frame": 0, "length_frames": 96,
                         "trim_start_frames": 0, "strength": 1.0, "attention": 0.65 } ],
  "retake": { "enabled": false, "video_path": null, "start_frame": 0, "length_frames": 0,
              "strength": 1.0, "prompt": null }
}
```

Frame arithmetic is one pure module (`DirectorTimeline.Math`): frames ↔ seconds at the
timeline fps; total length snapped up to `1 + 8k`; keyframe frame → latent index by ceiling
division; chunk planning against the 289-frame window with the existing continuation rule.
Pure, tested, shared by the engine, the CLI and the desktop tab (which needs the same
snapping for its ruler).

### 4.2 The compiler (`DirectorCompiler`, engine)

`compile(timeline) -> [LTX2VideoRequest]` (one request per chunk) plus a `DirectorPlan`
record (which keyframes, prompt and audio span land in which chunk) that is stored in the
trace and returned to the client for the timeline scrubber. Rules:

1. **Keyframes** → `keyframes[]` on `LTX2VideoRequest` (new field) → `generateMultiKeyframe`
   when more than one, else the existing i2v path. `is_end_frame` places the image at the
   chunk's last frame. A keyframe exactly on a chunk boundary belongs to both chunks (last
   frame of one, first of the next), which is what continuation already does.
2. **Prompts, Phase 1 = per-chunk scheduling.** Each chunk's text = global prompt + the
   prompt segments overlapping that chunk, weighted by overlap, joined in time order. This is
   an honest subset of prompt relay that needs no model patch: LTX-2.3 reads one text
   conditioning per denoise pass, and our chunks are already separate passes.
   **Phase 2 = true prompt relay:** per-segment token ranges with a temporal attention mask
   in `LTX2Transformer` cross-attention (text tokens visible only to latent frames in their
   segment, global prompt visible everywhere). Requires a mask hook in the attention kernel
   and a validation ladder of its own.
3. **Audio.** `generated`: as today. `imported`: mux the combined 44.1 kHz timeline over the
   video and drop generated audio (Phase 1, pure post-processing — `LTX2PostProcess` already
   muxes). `inpaint` (Phase 2): encode imported spans with the audio VAE encoder into the
   audio latent and denoise only the gaps under a per-latent-frame audio noise mask — the
   audio-side twin of the retake mask.
4. **Reference clips (Phase 2).** Decode the video with AVFoundation, resample to timeline
   fps, encode frames through the video VAE, append as conditions at their frame indices with
   `strength`; a mid-timeline guide gets the two-frame noise ramp. IC-LoRA attention entries
   and the Ingredients LoRA weights are Phase 2b, gated on obtaining the weights and on a
   ladder showing the guide-only path is not enough.
5. **Retake (Phase 2).** New pipeline entry: base video decoded and VAE-encoded into the
   initial latent; a temporal noise mask (0 outside the range, 1 inside, `strength` scaling)
   applied at every step so frozen frames are restored after each update; if the range covers
   the clip, plain generate. `retake.prompt` overrides the global prompt for the pass.

Everything the compiler emits goes through the same `LTX2ConfigResolver` (request > preset
> config > env > builtin), so Director renders at the production recipe by default and
honours the preset's negative and tuning like every other clip.

### 4.3 Server API

- `POST /v1/video/director` — body = the timeline document (assets by path or base64), plus
  `wait: false` for the job model. Validates (frame alignment, asset existence, ≤ N chunks),
  compiles, enqueues one video job per chunk in order (continuation carries the last frame),
  returns `job_id` and the `plan`. Status via `/v1/video/status/{id}` (stages = chunks).
- `POST /v1/video/director/validate` — pure: returns the plan, snapped length, warnings
  (keyframes closer than 24 frames → the spike's jump-cut finding; a segment shorter than
  one latent frame; audio clip past the end) without rendering.
- `GET/PUT /v1/director/timelines/{id}` — server-side store beside presets so a timeline can
  be opened from any client; the file is the same JSON as `.cbdirector`.
- Persisted-queue treatment: director jobs are `kind: video_director` and recovered after a
  crash like `generate` (the gap the multi-keyframe FDD flagged).
- MCP: `generate_director_video(timeline)` + `validate_director_timeline`; CLI
  `ComfyBox director-render <file.cbdirector>`; `docs/api-notes.md` section.

### 4.4 Desktop "Director" tab

New `AppTab.director` in the Create section, `Views/Director/`:

- **Timeline canvas**: a ruler in seconds and frames at the timeline fps; tracks for
  Keyframes, Prompts, Audio, Reference; playhead; zoom; snapping to latent frames (every 8th
  frame) and to clip edges; in/out points; multi-select; split at playhead for prompt and
  audio clips; keyboard: space (play preview), S (split), I/O (in/out), ⌘Z.
- **Keyframe track**: drop PNGs (reuse `ImageDropSupport.handleImageDrop`), reorder by drag,
  per-keyframe strength and "end frame" toggle, thumbnails via `NSImage`.
- **Prompt track**: resizable prompt boxes per segment; a global prompt field in the sidebar;
  the Enhance button routes each box through the existing optimizer.
- **Audio track**: drop WAV/MP3/M4A; waveform via `AVAssetReader` peaks; trim handles; the
  audio mode picker (generated / imported / inpaint — the last disabled until Phase 2).
- **Reference track** (Phase 2): drop a clip; a frame strip via `AVAssetImageGenerator`.
- **Sidebar**: preset picker (video presets only), resolution, seconds (snapped readout),
  seed, LoRAs, audio mode, Speech length readout (quoted words ÷ 2.5 words/s, live), and
  Generate / Validate / Save / Load.
- **Retake** (Phase 2): load a rendered clip as the base, drag a range, Retake.
- **Result**: the clip in the existing player with the plan overlaid on the ruler (chunk
  boundaries, keyframe ticks) so the user sees where each prompt applied.
- Persistence: `.cbdirector` via `NSSavePanel`/`NSOpenPanel`; autosave of the open timeline
  in Application Support; Save-to-server through the timelines route.

The Motion tab stays as the quick single-shot path; Director is the editor.

### 4.5 What this deliberately does not do

- Run or vendor any upstream Python (repo rule: ComfyBox is self-standing Swift/MLX).
- Copy upstream code. GPL-3.0 governs their source; this is a clean-room port of documented
  behaviour. The timeline JSON is ours (the field names above), with an import shim for
  upstream's `timeline_data` as a convenience.
- Replace the Krita storyboard docker (#237/#247). Storyboards stay in Krita; Director is the
  desktop's timeline for a single LTX clip, which is a different job.
- Touch Kira's scheduler. The MCP tool exists so it can adopt Director later.
- Change the production recipe. Director renders through the same resolver.

## 5. Work packages and phasing

| WP | Scope | Depends on | Size |
|---|---|---|---|
| WP1 | `DirectorTypes` + `Math` + Codable + tests; `.cbdirector` read/write; upstream import shim | — | S |
| WP2 | `LTX2VideoRequest.keyframes[]` + `LocalVideoRequest` wire + `generateMultiKeyframe` dispatch; `DirectorCompiler` (chunks, per-chunk prompts, keyframes, generated/imported audio); `/v1/video/director` + `/validate`; queue kind + recovery; trace plan | WP1 | L |
| WP3 | Desktop Director tab, Phase 1 tracks (keyframes, prompts, audio), sidebar, save/load, generate + progress + plan overlay | WP1 (types), WP2 (route) | L |
| WP4 | MCP tools, CLI command, api-notes, timelines store route | WP2 | S |
| WP5 | Validation ladder at the production recipe (see §6) | WP2, WP3 | M |
| WP6 (Phase 2) | Retake temporal mask; audio latent inpainting (needs audio VAE encoder path); reference-video guides + noise ramp | WP2 | L |
| WP7 (Phase 2) | True prompt relay (temporal cross-attention mask in the DiT) | WP2 | L |
| WP8 (Phase 2b) | IC-LoRA weights + attention entries | WP6 | M |

Phase 1 = WP1–WP5. It ships a usable Director: FFLF and middle keyframes, per-chunk prompts,
long timelines, imported audio, project files, on the production recipe.

## 6. Validation ladder (each rung a real render, Todd's read, same seed where possible)

1. FFLF at 576×896, 10 s: two keyframes 0 and end, one prompt — the spike repeated on the
   wire. Pass = both frames honoured, no jump cut with ≥ 48 frames between.
2. Three keyframes across two chunks (a keyframe on the boundary). Pass = continuity across
   the boundary at least as good as Extend today.
3. Prompt schedule: two segments with contradictory actions on one 12 s timeline. Pass = the
   action changes at the chunk boundary, not before.
4. Imported audio: a 10 s voice clip over a talking-head keyframe pair, `imported` mode.
   Pass = lip motion plausible, no generated audio bleeding through.
5. Retake (Phase 2): freeze 0–4 s and 8–12 s of a rendered clip, regenerate 4–8 s with a new
   prompt. Pass = frozen frames byte-identical after decode tolerance, seam invisible.
6. Reference guide (Phase 2): a hand-wave clip on the reference track, strength 1.0. Pass =
   the wave lands on the timeline where placed.

## 7. Risks

- **Model ceiling.** The spike found jump cuts when keyframes sit too close; the motion
  envelope rule (no full turns) still applies. Validate warns, it does not forbid.
- **Prompt relay Phase 1 is coarse.** Segment changes land on chunk boundaries (≤ 12 s
  granularity at 24 fps). Phase 2 is the real thing and is transformer work.
- **Memory.** Encoding a reference clip through the video VAE on top of the DiT is
  40 GB-class territory; the guide must be encoded first and the pipeline released, like the
  image-pool vacate before video today.
- **Audio inpainting needs an encoder** we have not shipped (we decode; we do not encode
  audio into latents). Confirm the audio VAE has an encoder before WP6 is scheduled.
- **SwiftUI timeline UI is the largest desktop view yet.** Keep the model pure and tested;
  keep the canvas a thin renderer.
- **IC-LoRA weights and licence.** Obtain and check before WP8.
- **GPL-3.0 upstream.** Behaviour only; no code, no assets.

## 8. Codex review

(§8 is filled from `codex exec` against this document and the engine source; see
`docs/FDD-ltx-director-tab.codex-review.md`.)

## 9. Decisions requested

1. Phase 1 scope as in §5 (Director with keyframes, per-chunk prompts, long timelines,
   imported audio, project files) — build now; Phase 2 after a ladder read.
2. Route shape: new `/v1/video/director` compiling to the existing job model, not an
   overload of `/v1/video/generate`.
3. The desktop gets its own timeline tab (this reverses "no custom desktop UI" for
   storyboards only insofar as Director is not a storyboard tool).
4. Upstream import shim (their `timeline_data` → ours) in Phase 1, or later.
