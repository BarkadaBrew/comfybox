# FDD: "Director" — a timeline editor tab for LTX-2.3 in CoffeeShop Desktop

Status: v1.3 (2026-09-17) — codex-reviewed (v1.1); v1.2 adds Phase 2 regional prompting (§4.6, WP10) adapted from ComfyUI-LTX-BBox-Animator; v1.3 adds audio-driven chunks for long dialogue (§4.7, WP11) animated GIF export (§4.8, WP12), and Sequences with presets and AI-assisted drafting (§4.9, WP13–15), and agent access through API, MCP and a skill (§4.9.5, WP16); v1.4 adds Phase 3 Programs, long-form video on request with no human in the loop (§4.10, WP17–WP22); v1.5 adds GPU priority lanes, hold-until and the overnight window (§4.10.9, WP23). Requested by Todd: "clone this product as a tab for
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
- i2v with `extendToSeconds` chunk continuation (`ChunkPlan`, last-frame carry-over,
  `LTX2VideoGenerator.swift:487-516`, chunk loop `:1593`), the 289-frame single-pass window
  (`WarmServer.swift:3217-3239`), 24 fps default. Recipe resolution is request > preset >
  `~/.comfybox/config.json` `video` block > env > builtin (`LTX2ConfigResolver`); the
  production recipe (euler, 10-step schedule, NAG 5/0.25/2.5, STG 0.3 flat, color anchor 1)
  lives in the config tier, not in resolver builtins. Audio is a per-request flag
  (engine default off; the desktop and the scheduler send it on).
- **Temporal prompt scheduling already exists**: `beat_schedule` on `/v1/video/generate`
  (`BeatSegment`, `LTX2BeatSchedule.swift`) compiles per-beat token ranges into an additive
  temporal text-attention bias (`buildVideoBias`, `(1,1,videoTokens,textLen)`) that the
  denoise loop threads as `beatBias`. This IS prompt relay within a chunk; the scheduler's
  authored beats use it today.
- **A per-latent-frame denoise mask already exists** (`LTX2LatentState`,
  `LTX2Conditioning.swift:27-29`), restored every step (`LTX2Pipeline.swift:1869-1902`,
  `:2079-2085`, `:2178-2183`). Retake needs an API and a compiler on top of it, not a new
  primitive.
- **The audio VAE has an encoder** (`LTX2AudioVAE.encode(mel)`, `LTX2AudioVAE.swift:304-339`,
  weights remapped `:371-389`). What is missing is the ingest path: imported waveform →
  mel → latent, and an audio-side noise mask; generation today starts audio from noise
  (`LTX2Pipeline.swift:487-504`).
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

Not present: a timeline document, keyframes on the wire, a retake (base-video time-range)
API, imported-audio ingest (decode/resample/mix, mel encode, audio noise mask),
reference-video guides, IC-LoRA weights. Also: local video jobs are non-durable —
`QueuePersistence` recovers only `generate` and `lora_swap` (`QueuePersistence.swift:9-27`)
and recovery explicitly refuses local video (`WarmServer.swift:3717-3728`).

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
2. **Prompts = the existing beat schedule.** Per chunk: the text conditioning is the global
   prompt followed by the prompt segments that overlap the chunk, in time order (one text
   embedding per denoise pass — confirmed `LTX2Pipeline.swift:395-404`, `:1068-1077`); and the
   segments become a `beat_schedule` for that chunk (`BeatSegment` start/end fractions and
   token ranges), so within the chunk each latent span attends to its own segment's tokens
   through the existing temporal bias. This is prompt relay, built on what ships. A segment
   that spans a chunk boundary is split at the boundary with the same text on both sides.
   **Phase 2** is the residue: authoring-time preview of the bias, bias strength as a dial,
   and a ladder that measures how sharply an action change lands inside a chunk.
3. **Audio.** `generated`: as today. `imported` (Phase 1): a new ingest step decodes the
   clips with `AVAssetReader`, resamples and mixes them onto a 44.1 kHz stereo timeline
   (trim, gain, pad/trim to the clip length), suppresses generated audio for the chunk, and
   hands the PCM to `LTX2PostProcess.writeMP4`, which already muxes an in-memory
   `AudioTrack` (`LTX2PostProcess.swift:271-306`) but takes no file paths itself.
   `inpaint` (Phase 2): the same ingest feeds waveform → mel → `LTX2AudioVAE.encode` with
   the pipeline's normalization/patchify parity, into the audio latent under an audio-side
   per-latent-frame noise mask so only the gaps are generated.
4. **Reference clips (Phase 2).** Decode the video with AVFoundation, resample to timeline
   fps, encode frames through the video VAE, append as conditions at their frame indices with
   `strength`; a mid-timeline guide gets the two-frame noise ramp. IC-LoRA attention entries
   and the Ingredients LoRA weights are Phase 2b, gated on obtaining the weights and on a
   ladder showing the guide-only path is not enough.
5. **Retake (Phase 2).** New pipeline entry on top of the existing per-frame denoise mask:
   the base video is decoded, VAE-encoded, and written into the clean latent for the frozen
   spans; the time range compiles to a mask (0 frozen, 1 regenerate, `strength` scaling) that
   the loop already restores every step; if the range covers the clip, plain generate.
   `retake.prompt` overrides the global prompt for the pass. Seam validation is the work.

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
- Durability: local video jobs are non-persisted today and recovery refuses them. Phase 1
  Director jobs inherit that (documented, same as Motion and the scheduler's clips); a
  serializable, recoverable video/director job state is its own Phase 2 package (WP9).
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

### 4.6 Regional prompting — animated boxes per prompt (Phase 2, WP10)

**Source of the idea.** [ComfyUI-LTX-BBox-Animator](https://github.com/yuvraj108c/ComfyUI-LTX-BBox-Animator)
(GPL-3.0, reviewed 2026-09-16): the user draws boxes around subjects, keyframes their position
and size across the clip, and gives each box its own prompt. Two mechanisms work together
there: (a) each object prompt is spatially restricted in text attention ("LTX Apply Regional
Conditioning"), and (b) an IC-LoRA (`LTX-2.5-22b-IC-LoRA-Bbox`) is fed a control video of
white boxes on black so the model tracks where each subject goes. Their stated limits: small
boxes and unusual shapes lose adherence; memory grows with each object.

**What we take and what we do not.**

- **Not the IC-LoRA.** It is trained on LTX-2.5 22B; our base is LTX-2.3 (PinkCherry v1.8),
  and the Hugging Face repo returned 401 to an unauthenticated request on 2026-09-16 (private,
  gated, or pulled). Mechanism (b) is out unless a 2.3-compatible, obtainable adapter appears.
- **Not the code.** GPL-3.0; behaviour only, as with Director itself.
- **The idea (a), on what we already ship.** Our beat schedule is regional prompting in *time*:
  `LTX2BeatScheduleBuilder.buildVideoBias` already walks every video token `q`, derives its
  frame as `q / tokensPerFrame`, and writes an additive penalty into the text-attention cost
  for tokens outside a beat's time window (`LTX2BeatSchedule.swift:258-300`). The token's
  position *within* the frame is `q % tokensPerFrame`, i.e. a row and column on the latent
  grid. A region is the same penalty applied when the token's (row, col) at that frame lies
  outside the box interpolated for that frame. No transformer change, no LoRA, 2.3 weights.

**Why it matters to us.** It targets failures we have measured, not a nice-to-have:

- Two-body scenes where one person's description bleeds into the other (the 2026-09-15
  extra-limbs / wrong-action renders; Kira's traits landing on the partner).
- Multi-character apple and banana clips where one subject should hold still while the other
  moves, and cast renders where "her friend" must stay a different body from Kira.

**Design.**

- **Timeline document:** a `regions` track — `{ id, prompt, strength, soft_edge,
  keys: [{ frame, x, y, w, h }] }` with normalized box coordinates (0–1, resolution
  independent), linear interpolation between keys, and the box held before the first key and
  after the last. A region's prompt text is appended to the chunk's text like a prompt
  segment, and its token range is recorded for the bias.
- **Compiler:** each region becomes a spatiotemporal beat — token range + per-frame box. The
  bias builder gains an optional per-frame box: tokens outside the box get the penalty with a
  soft edge (distance to the box in latent cells, same Gaussian falloff as the time window) so
  edges do not hard-cut. The global prompt stays unbiased and visible everywhere.
- **Memory:** unchanged order — the bias is already a dense `videoTokens × textLen` cost
  (576×896 × 37 latent frames ≈ 18.6k tokens × ~1k text tokens ≈ 75 MB float32). More regions
  add token columns, not rows.
- **Desktop:** a Regions track in the Director canvas; boxes drawn and resized on the preview
  frame (a keyframe image or the last render as the backdrop, editor-only like upstream),
  keyed at the playhead, colored per region, with the region prompt in the sidebar.
- **Chunks:** a region spanning a chunk boundary is split with its interpolated box at the
  boundary, same as prompt segments.

**Validation first, UI second.** Before any canvas work, one wire-level ladder (WP10 rung 1):
a two-person 10 s clip at the production recipe, same seed, (A) global prompt only, (B) two
region prompts on static left and right boxes, (C) the same boxes with one subject crossing.
Pass for (B) = attributes stay with their box; pass for (C) = the moving subject keeps its
attributes across the crossing. If (B) fails at usable bias strengths, the bias alone is not
enough and WP10 stops there (an adapter becomes the prerequisite).

### 4.7 Audio-driven chunks — a monologue that stays in sync across splices (Phase 2, WP11)

**Problem.** A spoken line that runs longer than one chunk (a 40 s monologue is five
chunks) cannot stay in sync with Phase 1:

- **Imported audio** lands frame-exact across every splice. Ladder rung 4 measured an envelope
  correlation of 1.00. But chunks render with `audio: false`, so the video never sees the
  voice. The mouth moves generically and does not match the words in any chunk.
- **Generated audio** is written per chunk. The speech cannot continue from one chunk into the
  next, and the stitcher butts the chunk tracks together. The two-chunk rung 2 clip shows a
  sample jump of 7.7× the local median 8 ms after the splice and a 6 dB level step over
  300 ms (measured 2026-09-17).

**Design.** The voice drives the picture, and the voice is never re-rendered.

1. **One master voice track.** The monologue exists as audio before any video renders:
   recorded, generated by the voice lane, or rendered once as a single audio-only pass. It
   is an `audio_clips[]` entry on the timeline with a new `drives_video: true` flag.
2. **Boundaries snap to pauses.** Before chunking, the compiler runs a silence detector
   (RMS below −40 dBFS for ≥ 150 ms) over the driving clip. It moves each balanced chunk
   boundary to the nearest pause inside ±36 frames, subject to the existing constraints:
   ≤ 289 frames per chunk, and a `1 + 8k` length. A boundary that cannot reach a pause keeps
   its balanced position and raises `audio_boundary_mid_speech` (warning). Cutting
   mid-word is where lips are most likely to break, because chunk k+1 starts from a single
   carry-over frame and has no memory of the mouth shape.
3. **Each chunk conditions on its own slice.** Chunk k takes the master PCM for exactly its
   frame span, using the same arithmetic as `DirectorAudioMixer.chunkPlacement`: for k > 0 the
   first frame's worth of samples is dropped and placement starts at `start_k + 1`. The
   slice goes through waveform → mel → `LTX2AudioVAE.encode` (the WP6 encode-parity path) into
   the chunk's audio latent under an audio noise mask of 0 (keep) across the whole span. The
   video denoises normally. The a2v cross-attention (`LTX2Pipeline`) then carries lip and jaw
   motion from the fixed audio. A small pre-roll is optional: a guard band of the previous
   chunk's last 0.25 s is encoded and masked, then discarded. Whether it helps is a ladder
   question.
4. **Assembly muxes the master, not the chunks.** The stitcher ignores every chunk's decoded
   audio and muxes the untouched master PCM once through the Phase 1 imported-audio mux.
   Clicks and drift at splices are then impossible by construction. Sync depends only on the
   frame-exact placement that already measures 1.00.
5. **Sync is measured on every render.** After stitching, a sync probe compares
   mouth-region motion energy with the voice envelope per chunk and records the best-lag
   offset. The mouth region comes from the existing face detector (face-anchor path), or the
   lower-center third of the frame if no face is found. The lag search is ±200 ms. Offsets
   are written to the aggregate sidecar as `av_sync[]` (ms per chunk, with correlation). A
   chunk beyond ±80 ms, or with correlation below a ladder-set floor, raises
   `av_sync_drift` on the job result. It is not retried automatically.

**Wire.** `audio_clips[].drives_video` (bool, default false; at most one driving clip).
`audio.mode` stays `imported`. The compiler emits a per-chunk `audio_condition`
`{ audio_path, trim_start_frames, length_frames }`, and the generator honours it only on the
multi-keyframe and i2v arms. A driving clip in `generated` mode is a validation error
(`driving_clip_requires_imported`).

**Validation first, as with WP10.** WP11 rung 1 is a single chunk. It gates everything
after it.

- **(A)** A 6 s spoken line on a talking-head keyframe, rendered with `audio_condition`.
- **(B)** The same seed with the Phase 1 mux only.
- **Pass:** (A) beats (B) on the sync probe and on Todd's read.
- **Fail:** if (A) shows no lip-following at the production recipe, the model does not follow
  fixed audio well enough, and WP11 stops before the multi-chunk work.

### 4.8 Animated GIF export (Phase 2, WP12)

**Goal.** Any Director output, or any range of it, exports as a looping animated GIF for
messages, previews and posts. This is an export of a finished render, not a render mode.
GIF carries no audio, and the export dialog says so.

**Encoder.** Native ImageIO (`CGImageDestination`, `UTType.gif`) inside the engine. There is
no ffmpeg or Python dependency (repo rule). Frames are decoded with the same
`AVAssetReader` BGRA path the stitcher uses, which is frame-exact by construction.

**Options** (`POST /v1/video/export/gif`, also a desktop sheet):

| Option | Default | Notes |
|---|---|---|
| `video_path` | required | Must be inside the allowed output directory, like every video route. |
| `start_frame`, `end_frame` | whole clip | Timeline frames, inclusive. The desktop sheet pre-fills the in/out selection. |
| `fps` | 12 | 6–30. Frames are decimated by source index (`round(i · src_fps / fps)`), never blended. A non-divisor rate is accepted, and each frame's delay carries the exact timing. |
| `width` | 480 | Height follows the aspect ratio, rounded to even. Downscaling uses a Lanczos-quality CIImage scale. |
| `loop` | 0 (forever) | `kCGImagePropertyGIFLoopCount`. |
| `dither` | true | ImageIO quantizes each frame to a 256-colour palette. Dither on avoids banding on skin and sky, at some size cost. |
| `max_bytes` | 15 MB | When the first encode exceeds the cap, the export steps width down (480 → 360 → 270), then fps (12 → 10 → 8), until it fits. It reports the settings it used. If it still does not fit, it fails with `gif_too_large` and the measured size. |
| `output_path` | `<source>-<start>-<end>.gif` | Same containment rules. |

**Response.** `{ output_path, frames, fps, width, height, bytes, steps_down[] }`. The
export is synchronous up to 20 s of source, and becomes a job (`/v1/video/status`) beyond
that. Encoding is CPU-only and does not take the GPU lease, so it runs beside a render.

**Desktop.**
- **Entry points:** the Director result view and the projects render strip get "Export GIF…".
- **Sheet:** range from the timeline selection, fps and width presets (Messages 360/10, Post
  480/12, Preview 270/8), a live size estimate from a 12-frame sample encode, and Reveal in
  Finder when done.
- **Project storage:** exports land in the render's folder in a Director project
  (`renders/<id>/export-<start>-<end>.gif`).

**MCP.** `export_gif(video_path, start_frame?, end_frame?, fps?, width?)` so Kira and Bree
can send a GIF instead of an mp4 where a channel prefers it.

### 4.9 Sequences, Sequence presets and AI-assisted drafting (Phase 2, WP13–WP15)

**Todd's direction (2026-09-17).** A user can create, read, update and delete presets for
video length and any other parameter. The AI-assisted system generates a video from those
parameters. The Sequence concept already exists but is unused. The product of Director is a
Sequence, and a Sequence has its own sidecar, the way ComfyUI has JSON workflows.

#### 4.9.1 What exists today

- **Kira's image Sequence.** `Sequence` in coffeeshop-server `src/kira/media-pool/sequence.ts`
  is an ordered set of stills. It has a shared seed, img2img chaining, consistent elements
  and a stored record. Nothing in the engine or Director uses it.
- **Presets.** `PresetStore` (`Sources/ZImage/Server/PresetStore.swift`) already does CRUD
  over `/v1/presets`. It stores `mediaKind` `"image" | "video"` and a `videoTuning` Tier A
  block. The desktop edits presets in `PresetView`.
- **Per-render records.** Every render writes a mandatory JSON sidecar and embeds the same
  generation record in the mp4's metadata atom (`VideoGenerationRecord`, comfybox#401).
  `RenderTraceStore` keeps append-only lifecycle traces.
- **Director's aggregate sidecar.** Director writes one sidecar for the stitched output
  (`kind: "director"`, `chunk_count`, `recipe_hash`, `stitch_path`).
- **Workflows.** `WorkflowStore` imports ComfyUI workflow JSON (`/v1/workflows`).
- **The desktop assistant.** `AgentService` runs on the `assistant` provider (Glimmer). It
  emits `AgentAction` JSON that drives the Generate view's fields.

#### 4.9.2 The Sequence document (WP13)

A **Sequence** is the saved, replayable product of Director: everything needed to reopen,
re-render, audit or share a clip. It generalises the Kira image Sequence. `kind: "director"`
is the new video form, and `kind: "frames"` is reserved for the existing image form.

```jsonc
{
  "schema": "comfybox.sequence", "version": 1,
  "id": "seq_…", "kind": "director", "name": "…",
  "created_at": "…", "updated_at": "…",
  "preset": { "id": "…", "name": "…", "snapshot": { /* §4.9.3 fields as resolved */ } },
  "brief": { "text": "…", "assistant_model": "…", "template_version": "director-author/1" },  // absent when hand-built
  "timeline": { /* DirectorTimeline v1, unchanged */ },
  "plan": { "chunks": [ { "index": 0, "start_frame": 0, "frames": 289, "seed": 42,
                          "recipe_hash": "…", "resolved_config": { } } ] },
  "assets": [ { "role": "keyframe|audio|reference", "path": "…", "sha256": "…" } ],
  "audio": { "mode": "generated|imported", "master_path": "…", "master_sha256": "…", "av_sync": [] },
  "outputs": [ { "kind": "mp4|gif|lastframe|thumb", "path": "…", "sha256": "…", "frames": 385 } ],
  "engine": { "build_sha": "…", "stitch_path": "reencode", "tone_match": true },
  "status": [ { "at": "…", "state": "drafted|validated|rendering|succeeded|failed|interrupted" } ]
}
```

- **Sidecar.** `<output>.sequence.json` sits next to the stitched mp4, beside the existing
  generation sidecar. It is written atomically after the mp4, following the comfybox#401 rule.
- **Embedded like a ComfyUI workflow.** A compact copy (timeline, plan, preset snapshot, no
  outputs) rides in the mp4 metadata atom under `com.barkadabrew.comfybox.sequence`. Dropping
  any Director mp4 onto the Director tab, or choosing "Open as Sequence" in the Gallery,
  reopens the exact timeline, the way dragging a ComfyUI PNG restores its graph.
- **Replay.** `POST /v1/sequences/render { sequence, reseed?: false }` renders the plan with the
  recorded seeds and recipe. When `recipe_hash` differs from the current resolver, the response
  warns `recipe_drift`, so a replay never silently looks different.
- **Store and catalog.** `GET /v1/sequences`, `GET/PUT/DELETE /v1/sequences/{id}`. The
  gallery catalog indexes `kind: sequence` so pickers and search find them.
- **Projects (FDD-director-projects-pickers).** A project render record *is* a Sequence
  document. `renders/<id>/sequence.json` replaces the separate `timeline.cbdirector` +
  `plan.json` pair, so one schema serves projects, the gallery and replay.
- **Kira adoption is out of scope here.** The image Sequence keeps its own store. A later
  package can write `kind: "frames"` documents so both appear in one catalog.

#### 4.9.3 Sequence presets (WP14)

A **Sequence preset** is a named, user-owned bundle of Director parameters. It extends
`PresetStore` rather than adding a second store: `mediaKind: "sequence"` with a `sequence`
block. Every `mediaKind` switch today treats "not video" as image
(`PresetStore.swift` ~1126, ~1220, ~1343). Those three sites must learn the third kind, and
tests pin each one.

| Field | Example | Meaning |
|---|---|---|
| `length_seconds` or `length_frames` | 10 | Snapped to `1 + 8k` at the preset fps. |
| `fps` | 24 | |
| `aspect`, `width`, `height` | 9:16, 576×896 | |
| `video_preset_id` | a video preset | The render recipe: LoRAs and `videoTuning` (including `color_anchor` and `beat_window_margin`). |
| `keyframe_policy` | `fflf` \| `every_n_seconds:4` \| `none` | Where the drafter places keyframes. |
| `keyframe_image_preset_id` | an image preset | Used when the drafter generates keyframe stills. |
| `segment_cadence_seconds` | 3 | Target prompt-segment length for the drafter. |
| `audio` | `generated` \| `imported` \| `driven` | `driven` requires a driving clip (§4.7). |
| `character` | null | As `settings.character`. |
| `negative_prompt`, `steps`, `seed_policy` | `fixed:42` \| `random` | |
| `export` | `{ gif: { fps: 12, width: 480 } }` | Defaults for §4.8. |
| `author_notes` | text | Style guidance handed to the drafter, such as "static camera, slow push-in". |

- **CRUD.** The existing `/v1/presets` routes with `mediaKind: "sequence"` handle it.
  Validation rejects an unknown `video_preset_id`, a length outside 97–4609 frames after the
  snap, and `audio: driven` with `keyframe_policy: none` on an i2v-only recipe.
- **Desktop.** `PresetView` gains a Sequence section with the same list, detail and duplicate
  flow, and a length slider that shows seconds and snapped frames. Director's toolbar gets a
  preset picker that applies a preset to the open timeline's settings. Applying it never
  deletes placed clips. It warns when a clip falls past the new end.
- **MCP.** The preset CRUD tools in §4.9.5.

#### 4.9.4 AI-assisted drafting (WP15)

The user gives a **brief** (text, plus optional stills or a voice clip) and picks a Sequence
preset. The assistant drafts a complete Director timeline that respects every preset
parameter. The user reviews it on the canvas, then renders.

1. **Template-driven author.** The assistant follows an engine-owned authoring template,
   `director-author/1`. The template carries the LTX-2.3 prompt practice: one camera line,
   present-tense action per segment, sound woven into the action, no meta language, and
   segment prompts that name only what changes. It also carries the preset's constraints as
   hard rules: exact segment count and boundaries, keyframe slots, length. The template is
   versioned, and the version is recorded in the Sequence `brief` so drafts are auditable.
2. **Structured output.** The model returns a `DirectorTimeline` JSON object only. The engine
   fills everything deterministic itself: frame math, snapping, ids, seeds from the seed
   policy, and settings from the preset. The model authors prose and keyframe descriptions,
   never numbers the preset already fixed.
3. **Validate and repair once.** The draft goes through `/v1/video/director/validate`. Errors
   go back to the model once with the issue list. A second failure returns the draft with
   the issues attached for the user to fix, never a silent fallback.
4. **Keyframes.**
   - A keyframe slot the brief did not fill gets a still description.
   - With `keyframe_image_preset_id` set, the drafter renders those stills through the image
     lane before the video, as separate queued jobs, and places them.
   - User-supplied stills always win their slot.
5. **Surfaces.**
   - Route: `POST /v1/sequences/draft { preset_id, brief, assets? }` returns a Sequence in
     state `drafted`.
   - Desktop: a "Draft from brief" sheet on the Director tab, and an `AgentAction`
     `{ kind: "director_draft" }` so the chat assistant can hand a draft to the tab.
   - MCP: `draft_sequence`.
   - Rendering is always a separate, explicit step (`/v1/sequences/render`).

**Validation first.** WP15 rung 1 runs 10 briefs × 2 presets (a 10 s FFLF preset and a 30 s
every-4 s preset), with no rendering.
- **Pass:** 20/20 drafts validate within one repair, every draft honours its preset's length,
  segment count and keyframe slots exactly, and Todd judges ≥ 16/20 prompts usable as
  written.
- **Then:** two of those drafts render at the production recipe.

#### 4.9.5 Agent access: API, MCP and a skill (WP16)

Kira, Bree and Claude can all create Sequences. Access has three layers with one source of
truth, so no layer carries logic another layer lacks.

1. **API: the only place work happens.** The `/v1/sequences/*`, `/v1/video/director/*`,
   `/v1/presets` (`mediaKind: "sequence"`) and `/v1/video/export/gif` routes do all drafting,
   validation, rendering and export.
   - **Kira's scheduler is code, not an LLM tool loop.** It calls the API through a typed
     client in coffeeshop-server (`src/video/director-client.ts`), with the same
     queue-admission and GPU-lease rules as its other video renders.
   - **Contract.** Every route is documented in `docs/api-notes.md` → "Sequences". The
     wire schema is the Sequence document (§4.9.2), versioned by `schema` and `version`.
2. **MCP: the API for LLM agents.** The MCP tools are thin wrappers with no logic of their own.
   A tool's result is the route's JSON.

   | Tool | Route |
   |---|---|
   | `list_sequence_presets`, `upsert_sequence_preset`, `delete_sequence_preset` | `/v1/presets` (sequence kind) |
   | `draft_sequence(preset_id, brief, assets?)` | `POST /v1/sequences/draft` |
   | `validate_sequence(sequence)` | `POST /v1/video/director/validate` |
   | `render_sequence(sequence_id, reseed?)` | `POST /v1/sequences/render` (returns a job id) |
   | `sequence_status(job_id)` | `GET /v1/video/status/{id}` |
   | `get_sequence(id)`, `list_sequences(query?)` | `/v1/sequences` |
   | `export_gif(video_path, …)` | `POST /v1/video/export/gif` |

   - **Where the tools live.** They join ComfyBox's MCP server (`Sources/ZImage/MCP/MCPToolRegistry.swift`),
     beside the existing video tools. Bree reaches them through her `mcp_comfybox__*` proxy
     with no daemon change. Claude reaches them through the same MCP server.
   - **Render is always explicit.** `draft_sequence` never renders. An agent must call
     `render_sequence`, which makes the GPU cost a deliberate step.
3. **Skill: judgment, not execution.** A `comfybox-director` skill in the repo
   (`skills/comfybox-director/SKILL.md`) teaches an agent to use the tools well.
   - **Workflow:** pick or create a preset, draft, read the validation issues, fix or accept,
     render, poll, then check `av_sync` and `recipe_drift` in the result.
   - **Cost:** about 25 min of GPU per chunk at the production recipe. Before rendering,
     state the chunk count to the user or the owning persona. Never render during a
     declared soak or while admission is `local` for someone else's work.
   - **Prompt practice:** generated from the drafter's `director-author/1` template at build
     time (`scripts/gen-director-skill.sh`), so the skill and the drafter cannot drift apart.
     A test fails when the skill's embedded template version differs from the engine's.
   - **Consumers:** Claude loads it as a skill. Bree reads it with `read_skill` (her skill
     loader). Kira's chat side gets it through the same loader. Kira's scheduler does not
     need it, because code follows the API contract.

**Acceptance (WP16).**
- Each MCP tool round-trips against a stubbed route, and the tool result equals the route JSON.
- A Claude session with only the skill and MCP drafts, validates and renders a 10 s Sequence
  from a one-line brief, and reports the chunk count and GPU estimate before rendering.
- Bree lists and drafts through the proxy with no daemon code change.
- The skill-template version test is green.

### 4.10 Programs — long-form video on request (Phase 3, WP17–WP22)

**Todd's direction (2026-09-17).** "I ask Claude or Bree for a 12 minute video, then I get a
12 minute video after it is created and verified." An AI cannot operate Resolve or any
manual editor. "This is the most powerful element. No meat in the middle."

**Principle: no human between the ask and the delivery.** Every step that a person does in an
editing suite is either automated with a measurable pass/fail check or removed. The requester
is asked a question only before work starts (§4.10.2), never during it. A Program either
delivers a verified video or delivers a report saying exactly what failed. It never waits
silently for a human.

#### 4.10.1 Scale

| Quantity | 12 min at 24 fps |
|---|---|
| Frames | 17,280 |
| Director ceiling per Sequence | 4,609 frames, about 3.2 min |
| Chunks at ≤ 289 frames | about 60 |
| GPU at the production recipe (about 25 min per chunk) | about 25 h, before retries |

A Program is therefore a multi-day-capable job, not a render. Durability (WP9), scheduling and
honest time estimates are requirements, not polish.

#### 4.10.2 Flow

1. **Ask.** The agent (Bree, Claude, or Kira's chat side) calls `request_program` with the
   brief, the target duration and an optional Sequence preset. The engine returns an **estimate**
   before any GPU work: shot count, chunk count, GPU hours, earliest finish time given the
   current queue and admission state, and disk needed. The agent relays the estimate. A
   request under the requester's standing budget (§4.10.6) starts without a question.
   Anything over budget asks once, then runs.
2. **Plan.** The planner, an LLM on the `assistant` provider following the versioned template
   `program-planner/1`, writes the **Program document** (§4.10.3):
   - A logline and a scene list.
   - A shot list whose durations sum exactly to the target. The engine enforces the sum and
     rebalances durations, never the model.
   - A **bible** of the elements held constant across shots: characters, wardrobe, settings,
     look, and the lens and lighting vocabulary.
   - The narration script, timed per scene.
   - The music plan.
   
   The plan is validated structurally before any render. That covers durations, the chunk
   count per shot, bible references, and a script whose reading time fits its scenes.
3. **Reference stills.** Before any video, the bible's characters and settings are rendered as
   reference stills through the preset's image lane. The best of N is chosen by a vision check
   against the bible text. Every shot's keyframes derive from these stills, so continuity is
   anchored in pixels, not only in words.
4. **Narration.** The script is voiced once through the voice lane into a single master track
   per scene. Shots that carry narration become audio-driven Sequences (§4.7): boundaries snap to
   pauses, each chunk conditions on its slice, and assembly muxes the untouched master.
5. **Shots.** Each shot is drafted as a Sequence (§4.9.4) from the shot list, the bible and its
   reference stills. Each draft is validated, then rendered through the normal queue. Shots
   render in dependency order: a shot that continues the previous shot's last frame waits for
   it, and independent shots can queue in any order.
6. **Verify each shot** (§4.10.4). A failing shot takes the retry ladder. A passing shot is
   frozen, and its Sequence version is pinned in the Program.
7. **Assemble** (§4.10.5). This is automatic, rule-driven and engine-side.
8. **Verify the whole program** (§4.10.4). It either passes or goes back to the retry ladder for
   the offending shots.
9. **Deliver.** The agent sends the video (or a link when the file is too large for the channel)
   plus a delivery report: runtime, shots, retries, GPU hours used, anything flagged, and
   the Program id for replay.

#### 4.10.3 The Program document

A Program is to Sequences what a Sequence is to chunks: a replayable sidecar.
`<output>.program.json` sits next to the final mp4, and a compact copy rides in the mp4
metadata atom (`com.barkadabrew.comfybox.program`).

```jsonc
{
  "schema": "comfybox.program", "version": 1,
  "id": "prg_…", "requested_by": "bree|claude|kira|desktop", "brief": "…",
  "target_seconds": 720, "preset_id": "…",
  "estimate": { "shots": 64, "chunks": 60, "gpu_hours": 25.1, "finish_by": "…" },
  "planner": { "model": "…", "template_version": "program-planner/1" },
  "bible": { "characters": [ { "id": "c1", "text": "…", "reference_stills": ["…"] } ],
             "settings": [ … ], "look": "…" },
  "scenes": [ { "id": "s1", "summary": "…", "narration": { "text": "…", "master_path": "…" },
                "shots": [ { "id": "s1.1", "seconds": 9.0, "intent": "…", "bible_refs": ["c1"],
                             "continues_from": null, "sequence_id": "seq_…", "sequence_version": 2,
                             "verdicts": [ … ], "attempts": 2 } ] } ],
  "music": { "source": "library|none", "path": "…", "gain_db": -18, "duck_under_narration": true },
  "edit": { "transitions": [ { "after": "s1.3", "kind": "cut|crossfade|dip", "frames": 12 } ] },
  "outputs": [ { "kind": "mp4|gif|report", "path": "…", "sha256": "…" } ],
  "state": "estimating|planning|references|narration|shots|assembling|verifying|delivered|failed|cancelled",
  "ledger": [ { "at": "…", "event": "…", "detail": { } } ]
}
```

Every state change is appended to `ledger`, and the ledger is what resume reads (§4.10.6).

#### 4.10.4 Verification gates

Every gate is automated and records a verdict with its evidence. A verdict with no evidence
fails.

**Per shot**

| Gate | Check | Evidence |
|---|---|---|
| Structural | Frame count equals the plan exactly, fps, dims, an audio track present and as long as the video | ffprobe-equivalent AVAsset read |
| Render health | No NaN or black frames, no frozen run longer than 1 s unless the shot intent says static, exposure step between chunks under the tone-match threshold | per-frame luma and diff stats (the seam metrics used on ladder rung 2) |
| Prompt adherence | A vision model scores sampled frames (1 per second plus every segment boundary) against the shot intent and its segment prompts | per-sample score, threshold set by ladder |
| Continuity | Bible characters match their reference stills, and a shot that continues another matches its predecessor's last frame | face/appearance similarity against the reference stills, last-to-first frame diff |
| Audio | `av_sync[]` within ±80 ms for driven shots, no clipping, no click at internal splices | §4.7 sync probe, sample-jump detector |

**Whole program**

| Gate | Check |
|---|---|
| Runtime | Within ±0.5 s of the target |
| Joins | No click or level step at any shot boundary. Transition frames exactly as planned. |
| Loudness | Integrated loudness at the delivery target (−16 LUFS stereo), true peak ≤ −1 dBTP, narration intelligible over music (ducking verified) |
| Coverage | Every scene and every narration line present, in order |
| Final read | A vision-model pass over a contact sheet (one frame every 5 s) against the logline and scene list. A flag here names the shots involved. |

**Retry ladder** (per failing shot, stops at the first rung that passes)
1. **Reseed.** The same draft with a new seed, up to 2 attempts.
2. **Re-draft.** The drafter rewrites the shot with the verdict evidence in context, then up to
   2 renders.
3. **Simplify.** The shot intent is reduced (static camera, fewer segment changes, shorter
   duration borrowed from a neighbouring shot so the total holds), then 1 render.
4. **Flag.** The best-scoring attempt is kept and marked `flagged` in the report. The program
   still assembles. A Program never blocks delivery on one shot unless the requester set
   `strict: true`.

Retries count against the GPU budget (§4.10.6). When the budget is exhausted, remaining failing
shots go straight to Flag.

#### 4.10.5 Automatic assembly

- **Edit decision list.** The edit is data in `edit`, derived by rule from the plan, never hand-built:
  - A cut is the default.
  - A crossfade (12 frames) is used at scene changes.
  - A dip to black (24 frames) is used at act breaks, when the planner marks one.
  - `continues_from` shots are always butt-joined. Their shared frame is dropped, as Director
    chunks do.
- **Picture.** Shots are referenced by their pinned Sequence versions. Plain cuts use an
  `AVMutableComposition` passthrough. Only transition regions re-encode, with tone matching
  across every join (`DirectorToneMatch`).
- **Audio.**
  - **Beds.** Native LTX audio stays linked to its shot as the ambience bed.
  - **Narration** masters are placed at their scene offsets and override the beds of driven shots.
  - **Music** comes from a library file or is absent. Music generation is out of scope until a
    local music model exists. It ducks under narration.
  - **Joins.** Every audio join gets a 10 ms equal-power crossfade, so a click is impossible by
    construction.
- **No external tools.** Encoding uses AVFoundation only (repo rule). Output is H.264 or HEVC
  with AAC. The GIF and thumbnail exports (§4.8) are produced from the final program.

#### 4.10.6 Execution: durable, budgeted, schedulable

- **Durable (WP9 is a hard prerequisite).**
  - The Program is a persisted state machine. The ledger is written before and after every step.
  - After an engine or daemon restart, the Program resumes at the first unfinished step. A
    finished shot is never re-rendered, and an interrupted shot restarts from its last
    finished chunk.
- **Budget.**
  - Each requester has a standing budget in config: `max_gpu_hours` per Program and `finish_by`
    tolerance.
  - The estimate is checked against it before work starts, and live spend is checked
    after every shot.
- **Schedule.**
  - Program renders run in the **batch** tier with a hold-until disposition (§4.10.9), so
    interactive work and Glimmer inference always go first.
  - A declared soak or `local` admission pauses Program rendering. Planning, reference
    selection and verification can continue.
  - The agent is told the new finish estimate whenever it moves by more than 1 h.
- **Cancel.** `cancel_program` stops at the next step boundary and keeps every finished artifact.
- **Progress.** `program_status` returns the state, shots done and total, GPU hours spent and
  the estimate, the current step, and flags so far. Bree can answer "how's my video going" from
  it.

#### 4.10.9 GPU priority lanes, hold-until and the overnight window (WP23)

**Problem.** A Program or any long Sequence render occupies the GPU for hours (about 25 min per
chunk, about 60 chunks for 12 min). Under pure FIFO it blocks daytime interactive work, and
nothing keeps overnight work overnight. Written up by Bree from Todd's request "write up
priority lanes" (2026-09-17). Corrected here against what is already deployed.

**What exists (do not rebuild).**

| Mechanism | Where | What it does |
|---|---|---|
| Single-flight render loop | engine job queue | One render at a time. Pause, resume, reorder and source attribution. |
| Step-boundary preemption | engine, #1479 / comfybox#322 | An LTX-2 render checks for interrupt and preempt signals at every denoise step, checkpoints, and resumes. |
| Inference slots | engine #465 + #468, daemon #1878 | A Glimmer call takes a TTL-leased top-priority slot. The job loop starts no render while one is held, and an in-flight video parks at its next step and resumes after. |
| Owner lanes + credit governor | daemon, #1485 | Fairness between daemon owners (chat, content stream, and so on) before work reaches the engine. |
| Active-hours windows | daemon, Kira scheduler tiers | Per-tier hours for Kira's own content. |

**Tiers.** Strict priority, highest first. A higher tier always goes next.

1. **Inference.** Glimmer chat, tools, vision and authoring through the inference slot. This
   already exists (#465). It parks renders at the next denoise step.
2. **Interactive.** Image renders, single clips, desktop Generate and Motion, and Director
   renders of one chunk. These start as soon as the GPU is free of inference. An in-flight
   **batch** render is preempted at its next denoise step, not at the next segment. Waiting
   for a segment boundary would make an image wait up to a whole chunk (25–40 min). The
   batch render checkpoints and resumes from its step (#1479). Nothing is lost except the
   step in progress.
3. **Batch.** Sequence and Program renders, multi-chunk Director timelines, bulk re-renders,
   and Kira's scheduled clips. These fill idle GPU time.
   - **Order:** batch jobs run in submission order among themselves.
   - **Segment boundaries:** between chunks the runner re-checks the queue, so a newly
     eligible batch job with an earlier `hold_until` does not leapfrog a running Sequence.
     Sequences finish chunk by chunk in order.

**Disposition at submit (batch only).** Every batch job carries `hold_until`.

| Disposition | `hold_until` | Behaviour |
|---|---|---|
| `now` | null | Eligible immediately. Runs whenever no inference slot or interactive job is pending. |
| `tonight` | the next window opening (default 23:00 America/New_York) | Never touches the GPU before that time, however idle the GPU is. Daytime GPU time stays free. |
| `at` | an explicit timestamp | As `tonight`, at a chosen time. |

- **Carried by requests:** the Sequence, Program and multi-chunk Director requests.
  Interactive requests reject a `hold_until` field (400), so a lane cannot be misused.
- **Persisted (WP9 — NOT TRUE TODAY).** A held job does **not** survive a restart as of
  2026-09-18, and the two reasons interlock, so neither fix works alone:
  `PersistedQueueJob` carries no schedule field, and `hold_until` is armed by an in-memory
  `Task.sleep` that dies with the process; *and* a held job is always a video/Director job,
  whose `rawBody` is nil, so `persistQueueState` drops it before the schedule would even
  matter. Until WP9a+WP9b land together, a bounce releases every hold. There is no test
  covering this — the missing test is the one WP9 should lead with.

**Overnight window.**
- **Config:** engine-side `batch_window` `{ start: "23:00", end: "07:00", tz: "America/New_York" }`,
  in `~/.comfybox/config.json`, readable through `/v1/queue`.
- **Inside the window:** all eligible batch work drains in submission order. Interactive and
  inference still preempt.
- **Spill policy per job** (`spill`):
  - `wait`: the default for `tonight`. Work left at 07:00 pauses at the next segment boundary
    and resumes the next night.
  - `idle`: after the window, keep running batch as a `now` job, still below interactive.
  - `strict`: hold at 07:00 regardless.
- **The capacity is small, so the estimate must say so.** An 8 h window holds about 19 chunks
  at the production recipe, roughly 4 min of video. A 12-min Program at `tonight` + `wait` is
  about three nights. The Program estimate (§4.10.2) states `nights` and `finish_by` from the
  window, the spill policy and the batch jobs already queued ahead, before work starts.

**Delivery.** A batch job that completes posts through the requester's channel as soon as it
finishes, even at night. Bree through Telegram, Claude through its session. Each Program also
gets a morning digest at window end (07:00 by default): what finished overnight, what is still
held, and new finish estimates.

**Relationship to the daemon lanes.** #1485's owner lanes stay the daemon's fairness layer
between owners. The engine tier is the GPU layer. A daemon submission states its tier
(`interactive` or `batch`) and its disposition. The daemon never implements its own overnight
hold for engine work, so there is one clock and one queue. Kira's active-hours tiers keep
deciding *what* Kira authors and when. The render itself is submitted as batch with a
disposition.

**API.**
- `POST` bodies gain `lane: "interactive" | "batch"`, plus `hold_until`, `disposition` and
  `spill` for batch. The default is `interactive` for single renders and `batch` for Sequence,
  Program and multi-chunk Director.
- `/v1/queue` reports each job's tier, `hold_until`, disposition and eligibility, plus the
  window and whether it is open.
- `PUT /v1/queue/batch-window` changes the window.
- `POST /v1/queue/jobs/{id}/disposition` moves a job between `now`, `tonight` and `at`.
- MCP `queue_status` shows the same fields. The `comfybox-director` skill teaches choosing
  `tonight` for anything over about 1 GPU hour unless the requester asks for now.

**Acceptance (WP23).**
- A `tonight` job submitted at 14:00 does not start before 23:00 with the GPU idle all
  afternoon. The ledger shows `held` until 23:00.
- During a running batch chunk, an image request starts within one denoise step, not after the
  chunk. Measured 2026-09-17 on engine df4fb24: a slot requested mid-render was granted 52 s
  later, at the next step boundary (576×896, 10 steps), and the video resumed immediately on
  release. The batch render resumes and produces
  output identical to an uninterrupted run within the #1479 tolerance.
- A Glimmer slot request during an interactive image parks nothing that is not a video, and
  queues ahead of the next job.
- A `wait` job still running at 07:00 stops at the next segment boundary and resumes at
  23:00. An engine restart at 03:00 resumes it without re-rendering finished chunks.
- A 12-min Program estimate at `tonight` + `wait` reports `nights ≥ 3` and a `finish_by` that
  the ladder run lands within +20%.

#### 4.10.7 Agent access

The same three layers as §4.9.5, with the engine as the only place work happens.
- **API.**
  - Routes: `POST /v1/programs` (estimate, then start), `GET /v1/programs/{id}`,
    `POST /v1/programs/{id}/cancel`, `GET /v1/programs`.
  - `POST /v1/programs/{id}/resume` is for an operator after a hard failure. Normal resume is
    automatic.
- **MCP:** `request_program(brief, target_seconds, preset_id?, strict?)` returns the estimate and
  the Program id. Also `program_status(id)`, `cancel_program(id)` and `list_programs()`.
- **Skill:** the `comfybox-director` skill gains a Programs section.
  - Relay the estimate honestly before starting.
  - Never promise a finish time the estimate does not support.
  - Poll sparingly: on state changes, not on a timer.
  - Deliver with the report, and surface flagged shots plainly instead of calling the video
    perfect.
- **Delivery channels:** Bree through Telegram (file or link), Claude through the session.
  The report is always attached.

#### 4.10.8 Validation ladder (Phase 3)

Each rung must pass before the next starts, at the production recipe.
1. **Program dry run (no GPU).** 5 briefs at 1, 3 and 12 min.
   - **Pass:** every plan validates, durations sum exactly, and estimates are within 15% of
     the arithmetic.
   - **Also:** Todd reads 2 plans and judges them coherent.
2. **1-minute Program** end to end from a Bree request, including narration.
   - **Pass:** delivered without human input, all gates recorded with evidence, runtime
     ±0.5 s, no join click, and Todd's read.
3. **Kill test.** Restart the engine and the daemon mid-shot during a 3-minute Program.
   - **Pass:** it resumes, no finished shot re-renders, the final output matches the ledger,
     and the delivery report lists the interruption.
4. **Verification honesty.** Inject a known-bad shot (wrong prompt).
   - **Pass:** the adherence gate fails it, the retry ladder runs, and the report names it.
     A gate that passes a known-bad shot fails the rung.
5. **12-minute Program** requested by Claude.
   - **Pass:** delivered verified within the estimate +20%, with flagged shots ≤ 5% of shots
     and each one named in the report.
   - **Also:** Todd watches it end to end.

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
| WP2a | `LTX2VideoRequest.keyframes[]` + `LocalVideoRequest` wire + `generateMultiKeyframe` dispatch (audio allowed when no mid-pass re-anchoring) | — | M |
| WP2b | `DirectorCompiler`: chunk plan, per-chunk text + `beat_schedule` from prompt segments, keyframe placement (incl. boundary keyframes), audio plan; `DirectorPlan` record | WP1, WP2a | M |
| WP2c | `POST /v1/video/director` + `/validate`, job orchestration over the existing async video job model (one job per chunk, continuation), status stages, trace plan | WP2b | M |
| WP2d | Imported-audio ingest + mux: AVAssetReader decode/resample/mix/trim/gain → PCM `AudioTrack`, generated-audio suppression | WP1 | M |
| WP3 | Desktop Director tab, Phase 1 tracks (keyframes, prompts, audio), sidebar, save/load, generate + progress + plan overlay | WP1 (types), WP2 (route) | L |
| WP4 | MCP tools, CLI command, api-notes, timelines store route | WP2 | S |
| WP5 | Validation ladder at the production recipe (see §6) | WP2, WP3 | M |
| WP6 (Phase 2) | Retake: base-video decode → VAE encode → clean latent for frozen spans, time-range mask compiler, seam ladder; audio latent inpainting (mel encode parity + audio noise mask); reference-video guides + noise ramp | WP2c, WP2d | L |
| WP7 (Phase 2) | Prompt relay residue: bias-strength dial, authoring preview, intra-chunk action-change ladder | WP2b | S |
| WP8 (Phase 2b) | IC-LoRA weights + attention entries | WP6 | M |
| WP9a (Phase 2) | Local video durability: thread the request `Data` into `enqueueLocalVideo` as `rawBody`, add a `video` arm to the boot replay (re-prepare via `prepareLocalVideo`), drop `.video` from `QueueRecoveryGate.nonRecoverableKinds`. Contract: at-least-once, replayed from chunk 0 — the same contract `generate` already has | WP2c | S |
| WP9b (Phase 2) | Schedule durability: `schedule` on `PersistedQueueJob`, re-arm `scheduleHeldJobWake()` after replay. Useless without WP9a (every held job today IS a video job), so they ship together. `QueueSpill.strict` persists as `distantFuture` and needs an explicit release route or it becomes an immortal ledger entry | WP9a | S |
| WP9c (Phase 2) | Director sequence resume: `kind: "director"`, a `director-<session>-state.json` written after each chunk (chunk cursor, `toneTransforms`, `carryTarget`, chunk metadata), a `runDirector` entry point that skips completed chunks, `VideoJobTracker` re-registration so the client's polled job id survives, and a boot sweep for orphaned intermediates — the `defer` cleanup cannot run on a process kill, so a crash leaks every chunk mp4 and carry-over PNG today | WP9a | M |
| WP9 note | Step/latent-level resume across restarts is explicitly OUT of scope. `LTX2ResumeState` is in-memory by design ("deliberately dies with the process") and persisting it would mean writing multi-GB `MLXArray` latents plus the decoded frame bank to disk. The durable granularity is the CHUNK boundary, which needs no tensors — only `(chunkIndex, seed, request JSON, previous chunk's last frame)`, all of which Director already puts on disk incidentally | — | — |
| WP10 (Phase 2) | Regional prompting (§4.6): `regions` track in the timeline schema; spatiotemporal bias (per-frame interpolated box + soft edge) in `LTX2BeatSchedule`; compiler support incl. chunk-boundary split; wire-level ladder FIRST (two-person static + crossing), then the desktop Regions track | WP2b, WP7 | M (engine) + M (desktop) |
| WP11 (Phase 2) | Audio-driven chunks (§4.7): `drives_video` flag, pause-snapped chunk boundaries, per-chunk `audio_condition` (mel → audio VAE encode under a keep mask), master-track-only assembly mux, `av_sync[]` probe + `av_sync_drift`; single-chunk ladder FIRST | WP6 (encode parity), WP2d | M (engine) + S (desktop flag + sync readout) |
| WP12 (Phase 2) | Animated GIF export (§4.8): ImageIO encoder with frame-exact decimation + size-cap step-down, `POST /v1/video/export/gif` (sync ≤ 20 s, job beyond), desktop Export GIF sheet in result view + render strip, `export_gif` MCP tool | WP2d (decode path) | S (engine) + S (desktop) |
| WP13 (Phase 2) | Sequence document (§4.9.2): schema + `.sequence.json` sidecar, mp4 metadata embed, open-from-mp4, `/v1/sequences` store + replay with `recipe_drift`, catalog `kind: sequence`; project render records adopt it | WP2c | M |
| WP14 (Phase 2) | Sequence presets (§4.9.3): `mediaKind: "sequence"` in `PresetStore` (+ the three image/video switch sites), validation, PresetView section, Director preset picker, MCP CRUD | WP13 | M |
| WP15 (Phase 2) | AI-assisted drafting (§4.9.4): `director-author/1` template, structured timeline output, validate-and-repair-once, keyframe still generation via image preset, `/v1/sequences/draft`, desktop "Draft from brief" + `director_draft` AgentAction; 20-draft ladder FIRST | WP13, WP14 | M (engine) + M (desktop) |
| WP16 (Phase 2) | Agent access (§4.9.5): api-notes "Sequences" contract, coffeeshop-server `director-client.ts`, full MCP tool set as thin wrappers, `comfybox-director` skill generated from `director-author/1` with a version-lock test | WP13–WP15 | S (engine MCP) + S (client) + S (skill) |
| WP17 (Phase 3) | Program document + planner (§4.10.2–3): `program-planner/1` template, bible, exact-duration shot list with engine-side rebalancing, estimate (shots/chunks/GPU h/finish-by), structural plan validation; dry-run ladder rung 1 FIRST | WP13–WP16 | M |
| WP18 (Phase 3) | Reference stills + narration (§4.10.2 steps 3–4): best-of-N reference stills with vision check against the bible, per-scene narration masters via the voice lane, audio-driven shots | WP17, WP11 | M |
| WP19 (Phase 3) | Verification gates + retry ladder (§4.10.4): per-shot and whole-program gates with recorded evidence, reseed → re-draft → simplify → flag, budget-aware | WP17 | L |
| WP20 (Phase 3) | Automatic assembly (§4.10.5): rule-derived EDL, composition passthrough + transition-only re-encode with tone match, beds/narration/music mix with ducking and equal-power joins, loudness normalisation, final exports | WP17, WP12 | M |
| WP21 (Phase 3) | Durable execution (§4.10.6): persisted Program state machine + ledger resume, `program` QoS lane, budgets, soak/admission awareness, cancel, progress | WP9, WP17 | M |
| WP22 (Phase 3) | Agent access + delivery (§4.10.7): `/v1/programs` routes, MCP tools, skill Programs section, Bree Telegram delivery with report; ladder rungs 2–5 | WP17–WP21 | M |
| WP23 (Phase 3, can ship before WP17) | GPU priority lanes (§4.10.9): inference > interactive > batch tiers in the engine queue, step-boundary preemption of batch by interactive, `hold_until` with `now`/`tonight`/`at` dispositions and `wait`/`idle`/`strict` spill in the durable ledger, engine `batch_window` config, window-aware estimates, completion posts + morning digest, queue/MCP readout | WP9, #465 | M |

Phase 1 = WP1, WP2a–d, WP3, WP4, WP5. It ships a usable Director: FFLF and middle keyframes,
prompt relay via the beat schedule, long timelines, imported audio, project files, at
whatever recipe the resolver stack yields (the production config today).

Phase 3 = WP17–WP22, **Programs**: long-form video on request, planned, rendered,
verified, assembled and delivered by the engine with no human in the loop (§4.10). It
depends on Phase 2's Sequences, presets, drafting, agent access (WP13–WP16), audio-driven
chunks (WP11), GIF export (WP12) and durable jobs (WP9). Its ladder (§4.10.8) runs 1-minute
before 3-minute before 12-minute. WP23 (priority lanes, hold-until, overnight window) has no
Program dependency and ships first, so Director and Kira batch work benefit immediately.

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

7. Regions (Phase 2, WP10 gate): two people, one global prompt vs two region prompts on
   static left/right boxes vs the same with a crossing. Pass = attributes stay in their box,
   including through the crossing. Fail at usable strengths = stop WP10.
8. Audio-driven single chunk (Phase 2, WP11 gate): a 6 s spoken line on a talking-head
   keyframe, same seed, with `audio_condition` vs Phase 1 mux only. Pass = the conditioned
   render wins on the sync probe and on Todd's read. Fail = stop WP11.
9. Monologue (Phase 2, WP11): a 40 s voice track over five chunks, boundaries pause-snapped.
   Pass = every chunk within ±80 ms on `av_sync[]`, no audible click or level step at any
   splice (the master track is untouched), lips plausible through each boundary.

10. GIF export (Phase 2, WP12, no GPU): export the rung 2 clip at 480 px, 12 fps, full
    length, and a 2 s range at 24 fps. Pass = frame count matches the requested range and fps,
    loops in Safari and Messages, under the size cap, no visible banding on skin at the default
    dither.
11. Sequence round trip (Phase 2, WP13, no GPU): open a rendered Director mp4 by dropping it
    on the tab. Pass = the timeline, plan seeds and preset snapshot match the sidecar exactly,
    and a replay reports no `recipe_drift` on an unchanged engine.
12. AI drafting (Phase 2, WP15 gate): 10 briefs × 2 Sequence presets, no render. Pass = 20/20
    validate within one repair, presets honoured exactly, ≥ 16/20 prompts usable per Todd;
    then two drafts rendered at the production recipe.

## 7. Risks

- **Model ceiling.** The spike found jump cuts when keyframes sit too close; the motion
  envelope rule (no full turns) still applies. Validate warns, it does not forbid.
- **Prompt relay rides the beat bias.** It is an additive attention bias, not a hard mask;
  how sharply an action change lands inside a chunk is an empirical question (WP7 ladder).
- **Memory.** Encoding a reference clip through the video VAE on top of the DiT is
  40 GB-class territory; the guide must be encoded first and the pipeline released, like the
  image-pool vacate before video today.
- **Audio ingest parity.** The encoder exists; the risk is matching the pipeline's mel
  normalization and patchify so an encoded clip lands where a generated one would. WP2d's
  mux path does not depend on it; WP6's inpaint path does.
- **Audio-driven lip sync is model-bound.** LTX-2.3 is a joint audio-video model, but how
  strongly its video follows a *fixed* audio latent is unmeasured. WP11's single-chunk rung
  decides it. The encode-parity risk above applies doubly: a wrong mel normalization makes
  the video follow noise.
- **Pause snapping can fail on continuous speech.** A breathless 12 s sentence has no pause
  inside a chunk's reach. The boundary then falls mid-word with a warning, and the pre-roll
  guard band is the only mitigation.
- **Programs amplify every weakness.** A defect that shows once in a 10 s clip shows sixty
  times in a 12-minute Program. Phase 3 does not start until the Phase 2 ladders pass, and
  §4.10.8 climbs 1 → 3 → 12 minutes.
- **Automated judges can be wrong both ways.** A vision gate that passes bad shots delivers a
  bad film unseen. One that fails good shots burns GPU on retries. Rung 4 (a known-bad shot)
  is mandatory, and every verdict carries its evidence so the report can be audited.
- **Throughput.** About 25 GPU hours per 12 minutes means one long Program a day at most,
  competing with the soak and Kira. Budgets, the `program` lane and honest estimates keep the
  request from silently starving other work.
- **Non-durable jobs in Phase 1.** A daemon or engine restart mid-render loses the Director
  job like it loses a Motion job today. WP9 closes it.
- **SwiftUI timeline UI is the largest desktop view yet.** Keep the model pure and tested;
  keep the canvas a thin renderer.
- **IC-LoRA weights and licence.** Obtain and check before WP8.
- **GPL-3.0 upstream.** Behaviour only; no code, no assets. Applies to both Director and the
  BBox Animator idea in §4.6.
- **Regions without a tracking adapter may be weak.** An additive bias shapes attention; it
  does not force a subject into a box. The upstream pairs the bias with an IC-LoRA for a
  reason. WP10's first rung is the go/no-go, and small boxes are expected to underperform.

## 8. Codex review (2026-09-16, `codex exec`, read-only against the engine source)

Full text: `docs/FDD-ltx-director-tab.codex-review.md`. Every §3 claim was checked with
file:line evidence. Five Majors and three Minors, all folded into v1.1 above:

1. Per-segment prompt conditioning already exists as `beat_schedule` (temporal attention
   bias) — WP7 is a residue, not a new attention hook; Phase 1 compiles segments to beats.
2. Local video jobs are non-durable; recovery refuses them — durability is its own package
   (WP9), Phase 1 documents the gap.
3. The audio VAE encoder exists; the missing piece is the ingest path — WP2d (mux) now,
   inpaint parity in WP6.
4. The per-frame denoise mask exists; retake's work is base-video encode + time-range
   compile + seam validation.
5. The production recipe lives in the config tier, not resolver builtins; audio is a request
   flag — §3 corrected.
6. `writeMP4` muxes PCM, not files — imported audio needs decode/mix/resample (WP2d).
7. WP2 split into WP2a–d.
8. "Temporal masks absent" reworded to "no retake API".

## 9. Decisions requested

1. Phase 1 scope as in §5 (Director with keyframes, per-chunk prompts, long timelines,
   imported audio, project files) — build now; Phase 2 after a ladder read.
2. Route shape: new `/v1/video/director` compiling to the existing job model, not an
   overload of `/v1/video/generate`.
3. The desktop gets its own timeline tab (this reverses "no custom desktop UI" for
   storyboards only insofar as Director is not a storyboard tool).
4. Upstream import shim (their `timeline_data` → ours) in Phase 1, or later.

## 10. Phase 1 implementation deltas (2026-09-16)

Phase 1 was built on `claude/director-phase1`: WP1, WP2a–d and WP4 are done; WP3 (the desktop tab) is in progress. Where the build
differs from §4–§6 above, the build is authoritative and the difference is recorded here.
The API contract as built is in `docs/api-notes.md` → "Director timeline (LTX-2)".

**Timeline document (§4.1)**
- `settings.seconds` → `settings.length_frames`. Frames are the source of truth and seconds
  (`frames / fps`) are only a UI readout. The length is snapped UP to `1 + 8k` and must be
  ≥ 97 after the snap (the production floor) and ≤ 4609 (16 chunks).
- `settings.resize` dropped. Every keyframe goes through the engine's existing i2v loader
  (center-crop + resize to width × height + H.264 conditioning round-trip).
- Per-keyframe `compression` dropped. The resolved `img_compression` applies.
- In/out points dropped. The desktop has no I/O keys in Phase 1.
- Added `settings.negative_prompt`, `settings.steps` and `settings.character`. With
  `character: null` (the default) every chunk sends `skip_character_injection`, so a
  keyframe-less chunk 0 (T2V) never silently becomes a character render.
- Decoding is explicit and lenient: unknown keys are ignored, a missing `version` reads as 1,
  and `version > 1` is rejected (`DirectorError.unsupportedVersion`, HTTP 400 on both routes).
- Keyframe frames must be multiples of 8 (`keyframe_off_grid` otherwise). The latent index is
  **floor**(frame / 8), matching `LTX2Pipeline.generateMultiKeyframeResumable`, not the
  ceiling division §4.1 describes. Because keyframes are grid-snapped, floor and ceil agree.
  The desktop ruler snaps to the same grid, rounding ties down (100 → 96).
- Added the `invalid_gain` error code. `duplicate_id` is checked across all tracks.
  `no_keyframes` fires whenever there is no keyframe at frame 0.

**Compiler (§4.2)**
- One single-pass `/v1/video/generate` body per chunk: `extend_to_seconds: 0` and
  `identity_anchor_strength: 0` are sent explicitly, plus `enhance: false`, seed + k, and a
  generic `keyframes[]`.
- Chunks are balanced: `ceil(S/36)` chunks of near-equal latent steps, each ≤ 289 frames. They
  are not the fixed-289 continuation layout.
- **Rule 1 wording changed:** a keyframe exactly on a chunk boundary conditions **chunk k's
  last frame only**. Chunk k+1 starts from the rendered carry-over frame, not the user image.
  The validator warns with `keyframe_on_chunk_boundary`, and adds
  `keyframe_partial_strength_on_boundary` when the keyframe's strength is < 1.
- Rule 1's "else the existing i2v path" holds per chunk: a body whose only keyframe is at
  frame 0 takes the untouched i2v arm. Any keyframe at frame > 0 takes the multi-keyframe
  arm, which now also generates audio (WP2a).
- **Chunk-0 asymmetry.** Every director chunk is chunk 0 to the generator, so continuation
  chunks get face/refine anchors and audio on the carry-over frame. A continuation chunk that
  carries an end keyframe gets no face/refine anchoring. With the double-lossy carry-over
  (mp4 → PNG → H.264 conditioning round-trip), this is the first suspect for a chunk-to-chunk
  quality step.
- **Imported audio is mixed at 48 kHz stereo, not 44.1 kHz.** 48 kHz matches the generated
  lane and the tested AAC mux. Imported clips are not mastered: overlapping clips sum, then
  hard-clip at ±1.
- **Generated audio across chunks.** Each chunk generates its own audio. Chunk k > 0's audio
  is placed at `start_k + 1` with its first frame's worth trimmed, matching the dropped video
  boundary frame. With more than one chunk there are seams (`generated_audio_seams`).
- The imported-audio track is not handed to `writeMP4` per chunk. Chunks render with
  `audio: false`, and the timeline mix is muxed once by the stitcher.
- **The stitcher always decodes and re-encodes** (`stitch_path: "reencode"`). There is no
  passthrough concat. Before writing, it verifies each chunk's fps and frame count.

**Server API (§4.3)**
- `wait: true` dropped; the route uses the 202 job model only.
- `GET/PUT /v1/director/timelines/{id}` (the server-side timeline store) deferred. The desktop
  (WP3) is planned to save `.cbdirector` files locally and autosave to
  `~/.comfybox/director-autosave.cbdirector`, not Application Support.
- **Submit-time dry run.** Every chunk body is prepared and validated before the 202. A preset
  that forces multi-chunk (`chunk_not_single_pass`), enables temporal upscale
  (`temporal_upscale_unsupported`) or changes frames/fps (`chunk_frames_mismatch`) is a 400.
  Temporal upscale is rejected in Phase 1 because the stitcher runs at `settings.fps`.
- The status route is unchanged, with additive `plan`, `stage_index` and `stage_count`.
  Failures read `chunk k/n (stage): message`.
- Durability: non-durable, as §4.3 already stated (WP9).
- Intermediates are deleted on success and on failure. `director-*` names follow the
  storyboard's non-`ltx2-` prefix, which the orchestration assumes the daemon's orphan
  reconciler ignores; that was not re-verified against coffeeshop-server.
- MCP: `generate_director_video` (additive) and `validate_director_timeline` (read-only),
  bringing the registry to 61 tools.
- CLI: `comfybox director-render <file.cbdirector> [--server URL] [--output path]
  [--source name] [--validate-only] [--wait] [--poll-seconds n]`. It is an HTTP client of the
  running server and never renders in-process.

**WP5 validation ladder — handed to Todd.** The Phase 1 house rules forbid touching the
production engine from the build sessions: no second engine, no `/v1/video*` calls against
:7870, no pausing the soak. The WP5 ladder is therefore a Todd-run follow-up. Each rung is one
`.cbdirector` file plus one command. Validate first; it costs nothing and runs while the soak
renders. Then submit when the queue allows. Use the production preset, the same seed across
re-runs, and 576×896 @ 24 fps unless noted.

1. **FFLF, 10 s.** `length_frames: 241`; keyframes `k1` at 0 and `k2` with
   `is_end_frame: true` (strength 1.0); one global prompt, no segments.
   ```
   comfybox director-render ~/Director/rung1-fflf.cbdirector --validate-only
   comfybox director-render ~/Director/rung1-fflf.cbdirector --output rung1-fflf.mp4 --wait
   ```
   Pass = both frames honoured and no jump cut (they are 240 frames apart, well over 48).
2. **Three keyframes across two chunks, one on the boundary.** `length_frames: 577`, which
   gives chunks [0..288] and [288..576]; keyframes at 0, 288 and end. Expect the warning
   `keyframe_on_chunk_boundary` for the keyframe at 288.
   ```
   comfybox director-render ~/Director/rung2-boundary.cbdirector --validate-only
   comfybox director-render ~/Director/rung2-boundary.cbdirector --output rung2-boundary.mp4 --wait
   ```
   Pass = continuity across frame 288 at least as good as Extend today. Watch for the
   chunk-0-asymmetry and double-lossy carry-over effects noted above.
3. **Prompt schedule.** `length_frames: 289` (a single chunk, 12 s); keyframe at 0;
   `prompt_segments` p1 [0, 144) and p2 [144, 289) with contradictory actions.
   ```
   comfybox director-render ~/Director/rung3-beats.cbdirector --validate-only
   comfybox director-render ~/Director/rung3-beats.cbdirector --output rung3-beats.mp4 --wait
   ```
   Pass = the action changes near frame 144, not before. Phase 1 compiles segments to the beat
   schedule inside one chunk, so this rung measures the change within a chunk. To test the §6
   wording (change at a chunk boundary), repeat with `length_frames: 577` and the segments
   split at 288.
4. **Imported audio.** `length_frames: 241`; a talking-head keyframe pair (0 and end);
   `audio: {mode: "imported"}`; `audio_clips: [{id: a1, audio_path: <10 s voice clip>,
   start_frame: 0, length_frames: 240}]`.
   ```
   comfybox director-render ~/Director/rung4-voice.cbdirector --validate-only
   comfybox director-render ~/Director/rung4-voice.cbdirector --output rung4-voice.mp4 --wait
   ```
   Pass = lip motion plausible and no generated audio (the sidecar reads
   `audio_source: "imported"`).
5. **Retake (Phase 2, not runnable in Phase 1).** A timeline with `retake.enabled: true`
   validates to the error `retake_unsupported`, and the render route returns 400:
   `comfybox director-render ~/Director/rung5-retake.cbdirector --validate-only`. Run it only
   to confirm the refusal. The real rung waits for WP6.
6. **Reference guide (Phase 2, not runnable in Phase 1).** Non-empty `reference_clips`
   validates to the error `reference_clips_unsupported`:
   `comfybox director-render ~/Director/rung6-reference.cbdirector --validate-only`. The real
   rung waits for WP6/WP8.

The file paths above are placeholders: author the files in the desktop Director tab (Save)
or by hand. `--server` defaults to `http://127.0.0.1:7870`.
