# HANDOFF — LTX video quality: ComfyBox vs the ComfyUI reference

**Date:** 2026-08-01 · **Session:** ComfyUI validation of PinkCherry 1.6-dev/1.7-alpha,
then porting those findings into ComfyBox.

---

## 1. Goal

**ComfyBox's LTX video output must equal or beat the ComfyUI reference**, at Kira's
production shape: 8-second clips (193 frames @ 24fps), rendered as **one chunk**, no
chained continuation. Todd, 2026-08-01: *"If you can do it in UI you can make box do it."*

Secondary goals, in order: 12s single-chunk (289f, done), audio, then author-scale
resolution (he ships 1472x1152 x 313f).

---

## 2. Status: what is DONE and proven

| Item | Evidence |
|---|---|
| 1.6-dev validated in ComfyUI | t2v NSFW/SFW, i2v NSFW/SFW, v2v — all clean 121f, no drift |
| 1.7-alpha validated in ComfyUI | t2v NSFW/SFW, i2v NSFW — sharp, **no haze** |
| Single-chunk 8s in ComfyBox | 193f, plain decode vol 7200, frames 0/96/100/192 clean |
| Single-chunk 12s in ComfyBox | 289f, plain decode vol 10656, final frame clean |
| Daemon single-pass deployed | `computeI2VMotionRecipe` was ALREADY merged; #1462's dead webhook meant the running daemon never got it. Restarted 15:52 on merged code |
| `~/Pictures/ComfyBox` symlink | was a Finder **alias** (opaque to POSIX + Swift) — would have failed every save |
| Reference recipe documented | `docs/methods/ltx-pinkcherry-comfyui-reference.md` |

**The chunk-seam corruption and the streamed-decode mosaic are both structurally
eliminated at 8s and 12s** — no seam to corrupt, no streaming to mosaic.

---

## 3. ~~THE OPEN BLOCKER~~ — RESOLVED 2026-08-01 evening (see 3.1)

> **RESOLVED.** Bug 2 was never the upsampler or the refine denoise: it was the
> **plain VAE decode** — MLX's Metal conv kernel silently corrupts when the
> implicit-GEMM virtual matrix `M·K = (T_pad·H·W)·(C_in·k³)` crosses ~2^32
> (measured: 4.16e9 exact, 4.51e9 corrupt, same boundary across slabs, both
> dtypes). `CausalConv3d`'s fixed 64-frame temporal chunk (calibrated on
> SeedVR2 slabs) stayed over that boundary at the LTX-2 decoder's conv_out
> slab (H128·W224·C128·k27 ⇒ safe chunk is 21 frames). Fixed in
> `CausalConv3d`: the chunk size is now adaptive (`M·K ≤ 2^31` per launch),
> plus a latent chunk-walk bug (extended final chunk followed by a duplicate
> tail chunk) that the adaptive size exposed. Verified: unit tests green
> (`LTX2VAERowBandTests`, `LTX2ConvOutOverflowProbeTests`,
> `LTX2StreamedDecodeParityTests` still exact) AND an end-to-end 97f 448x256
> i2v with `LTX2_TWO_STAGE=1` refining at 896x512 — all frames coherent, no
> band, no flat frames. The old "row 296/304" boundary was where the corrupt
> conv_out output region began; the "bottom band on every frame" and the
> "whole frames flat" are the same bug at different shapes.
>
> Instrumentation added along the way (env-gated, in `applyTwoStageRefine`):
> `LTX2_REFINE_ROWSTATS=1` logs per-row latent energy at base/upsampled/mixed/
> refined; `LTX2_REFINE_DUMP_DIR=<dir>` dumps those latents as .npy.
> Note: `/private/tmp` fixtures (up_in/up_out.npy) were lost to the reboots;
> `LTX2UpsamplerFixtureTests` skips until regenerated (§3 fixture recipe).
> Likely also explains the "streamed decode mosaics frame 0 above volume
> 4500" note in ltx-t2v-cfg-and-decode — streamed chunks still ran each chunk
> through the same over-large launches at big slabs. Worth re-testing streamed
> at those configs before trusting that memory.

### Original investigation notes (kept for the eliminated-list)

Without the refine, ComfyBox is capped at single-pass quality: clean but **soft/waxy**
compared to the reference. Todd can see the difference. This is the whole remaining gap.

### Bug 1 — upsampler weights never bound (FIXED)
`LTX2_UPSAMPLER_PATH` pointed at `/Volumes/Bolt/Models/ltx2-distilled/spatial_upscaler_x2_v1_1.safetensors`,
a locally converted file whose keys ALL carry a `spatial_upscaler_x2_v1_1.` prefix and
whose tensors are already MLX-layout. The loader matches bare names and permutes
PyTorch->MLX, so **zero parameters bound** → module ran on random init → full-frame
periodic mesh. Now uses the official Lightricks file, symlinked at
`~/LocalModels/ltx2-upsampler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`.

### Bug 2 — vertical truncation (OPEN)
With weights correctly bound, the TOP of each frame is coherent anatomy and the
**bottom ~41% is a flat magenta wash**.

Row-variance measurement (this is the key diagnostic):

| Run | Upsampler weights | Detail ends at |
|---|---|---|
| converted file | nothing bound (mesh) | row **296** / 512 (58%) |
| official file | all bound (coherent) | row **304** / 512 (59%) |

**Identical truncation regardless of weights** ⇒ it is NOT the upsampler math.

### Eliminated, with evidence — do not re-investigate
- Upsampler module key names — all 6 groups match the file (`initial_conv`, `initial_norm`,
  `res_blocks`x4, `upsampler`, `post_upsample_res_blocks`x4, `final_conv`)
- Conv padding — `padding: 1` on every conv, dims preserved
- `LTX2PixelShuffle2D` — channel split `(outC, rH, rW)` + transpose `(0,1,4,2,5,3)` matches
  the reference `PixelShuffleND(dims=2)` rearrange `"b (c p1 p2) h w -> b c (h p1) (w p2)"` exactly
- `centerCropped` re-anchor — correct (source 1.857 > target 1.5 ⇒ crops width, full height kept)
- Plain vs streamed decode — plain decode verified clean at volumes 7200 AND 10656; this
  failure was at 9600, between them
- **IC-control position grid — NOT the bug.** I hypothesised the refine's
  `createPositionGrid(... latH: rLatH, latW: rLatW)` at `LTX2Pipeline.swift:1318` omits
  `refFrames:` that the base pass passes at line 549. It does omit it, but line 578 *drops*
  the IC reference frames before the refine, so shapes match. **Retracted.**

### Remaining suspects (in order)
1. `LTX2Pipeline.swift:1272-1274` — `stats.unNormalize(...)` → `ups(...)` → `stats.normalize(...)`.
   Per-channel statistics applied at the wrong scale/axis could damage a contiguous band.
2. `LTX2Pipeline.swift:1309-1317` — refine re-noise + `refineInit`/`refClean`/`refMask`
   construction and the frame-0 anchor concat.

### THE NEXT TEST (do this before any more 10-minute renders)
Reference fixtures are already saved:
- `/private/tmp/up_in.npy` — input latent `(1,128,3,8,12)`, `torch.manual_seed(0)`
- `/private/tmp/up_out.npy` — reference output `(1,128,3,16,24)`

Reference per-row energy is **uniform** across all 16 rows (0.427–0.477) — that's what a
correct upsample looks like. Run `LTX2LatentUpsampler` on the same input in a Swift test
and diff. If our per-row energy collapses after ~row 9, the fault is in the upsampler
module; if uniform, it's downstream in the refine. Regenerate fixtures with:

```python
# in /Volumes/Bolt/ComfyUI-validate/ComfyUI, ./.venv/bin/python
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
import comfy.ops
m = LatentUpsampler.from_config(cfg, operations=comfy.ops.disable_weight_init)
# cfg = json.loads(safetensors __metadata__['config'])
```

---

## 4. Findings from the ComfyUI reference work

### The author's REAL settings are in his mp4 metadata, not the model card
Every `v1.6/*.mp4` he published embeds the full API graph in the mp4 `comment` tag
(`{"prompt": <api graph>, "workflow": <ui graph>}`). Extract with
`ffprobe -show_entries format_tags=comment`. This is the source of truth — the shipped
workflow JSON and the README differ from what he actually rendered.

Validated recipe (full detail in `docs/methods/ltx-pinkcherry-comfyui-reference.md`):
- sampler pass1 `euler_ancestral_cfg_pp`, pass2 `euler_cfg_pp`, **CFG 1.0**
- stage-1 sigmas 20-step; stage-2 `0.85, 0.7250, 0.4219, 0.0` (= 3 denoise steps, not 4)
- NAG 11.0 / 0.25 / 2.5
- distil LoRA **384** @ 0.6 (v1.6); **per-pass 0.6 → 0.45** for v1.7 (HF discussion #8)
- **`LTXVPreprocess img_compression: 22`** — we had 2; ComfyBox default is 35
- stage 1 renders at HALF the final size, upsampler x2 brings it back

### Convention trap: dims mean different things
- **ComfyUI**: the numbers are the FINAL size; stage 1 is internally halved (node 164, x0.5)
- **ComfyBox**: `width`/`height` are the STAGE-1 size; two-stage doubles them
⇒ With two-stage on, the daemon must send HALF dims or every clip is double-size with the
refine silently skipped by `LTX2_REFINE_MAX_VOL` (12000).

### Divisibility: 32 vs 64
Model requires **/32** (`areValidDimensions`, VAE downsamples by 32). The author's own
`ImageResizeKJv2` uses `divisible_by: 32`. **/64 is only needed because stage 1 runs at
half size and must itself be /32.** All his published dims are /64 (1472x1152, 1408x1024,
960x1280). Frames are always 1+8k (313, 409, 193, 289).

### LTX-2.3 A/V is broken on ComfyUI 0.29.0 + MPS — but the audio codec is FINE
Bisection (this is why all early renders were black):

| NAG | Audio | Result |
|---|---|---|
| on | on | black |
| off | on | black |
| off | off | **content** |
| on | off | **content** |

Audio is the sole discriminator; NAG is innocent. Crash signature:
`RuntimeError: cannot reshape tensor of 0 elements into shape [2, 0, 32, -1]` in
`apply_rotary_emb_qk`, from `comfy/ldm/lightricks/av_model.py`.

**BUT the reference audio VAE + vocoder work perfectly on MPS** — round-trip of real audio
in 2.84s, level-matched (source mean −28.9 dB → output −28.5 dB; a silent render measures
−91 dB). The vocoder's BWE upsamples 16k→48k. So the fault is specifically the
**transformer's joint A/V attention**, not the audio codec path. Our MLX transformer is an
independent implementation and may not share the bug.

### ComfyBox audio: mostly built, four wires missing
Already ported + unit-tested: `LTX2AudioVAE` (399 lines), `LTX2Vocoder` (378, BigVGAN v2),
`LTX2CrossModalAttention`, full audio branch in `LTX2Transformer`/`TransformerBlock`
(gated on `hasAudio`). **The weights are already on disk** — our monolith carries
`audio_vae` (102) + `vocoder` (1227). Config in the checkpoint metadata matches
`LTX2AudioVAEConfig` defaults exactly (mel_bins 64, z_channels 8, in_ch 2, ch 128,
ch_mult [1,2,4]).

Missing: (1) `LTX2VideoGenerator.swift:298` builds the transformer without `hasAudio:`
(defaults false; audio tensors deliberately left lazy); (2) no joint A/V latent handling
in the sampler; (3) nothing calls the decode chain; (4) `LTX2PostProcess.swift:223` writes
a video-only `AVAssetWriterInput`.
Caveats: the existing audio tests only assert weight-shape coverage, never numerics; the
dual-stream path has never run end-to-end; **audio must bypass the refine pass** (HF #8:
pass 2 "wiped out the sound from pass 1"; PRD story 3.2 says the same).

### AUDIO — work IN FLIGHT (task #8, paused mid-Phase A)

Todd said "Try" on 2026-08-01; work started and is **partially complete**. Plan is four
phases; Phase A is half done.

**Phase A — prove the codec, get ground truth.**
- ✅ **DONE: reference round-trip.** Real audio (extracted from the author's
  `back_in_black.mp4`, 16kHz stereo 5s) through the REFERENCE audio VAE + vocoder via the
  ComfyUI API: `LoadAudio → LTXVAudioVAEEncode → LTXVAudioVAEDecode → SaveAudio`
  (script: `/Volumes/Bolt/ComfyUI-validate/audio_roundtrip.py`). **2.84 seconds**, output
  level-matched to source (−28.9 → −28.5 dB mean; a silent render measures −91 dB).
  Artifacts: input `ComfyUI/input/audio-src.wav`, output
  `ComfyUI/output/audio/ref-roundtrip_00001.flac` (48kHz — the vocoder's BWE upsamples).
  ⇒ The audio VAE + vocoder work fine on Metal; only the transformer's joint A/V attention
  is broken upstream.
- ❌ **NOT DONE: the Swift side.** Need to export a mel fixture and run
  `LTX2AudioVAE.decode` + `LTX2Vocoder.synthesize` against it, comparing numerically to the
  reference output above (same approach as the upsampler fixtures in §3).

**Mel parameters** (from the audio VAE safetensors `__metadata__.config` — use these exactly):
```
sampling_rate 16000 · n_fft 1024 · win_length 1024 · hop_length 160 · n_mels 64
f_min 0.0 · f_max sr/2 · hann · center=True · pad_mode reflect · power 1.0
mel_scale "slaney" · norm "slaney" · then log(clamp(mel, min=1e-5))
final permute (0,1,3,2) → (B, C, T, F)
preprocessing: duration 5.12s · stereo · causal_padding 3 · max_wav_value 32768
autoencoder: mel_bins 64 · z_channels 8 · in/out ch 2 · ch 128 · ch_mult [1,2,4] · num_res_blocks 2
```
Reference code path: `comfy/ldm/lightricks/vae/audio_vae.py` (`AudioPreprocessor.waveform_to_mel`).
Note the standalone `LTX23_audio_vae_bf16.safetensors` (Kijai/LTX2.3_comfy, `vae/`) bundles
**audio_vae + vocoder** = 1329 tensors = 102 + 1227, which is why the checkpoint's
`audio_vae.*` alone (102) is not a complete audio VAE.

**Phases B–D — not started.**
- B: set `hasAudio: true` at `LTX2VideoGenerator.swift:298`, stop prefix-filtering the audio
  tensors out, verify weight coverage, measure RAM + per-step cost
- C: joint A/V latent — allocate the audio latent from frames+fps, concat, run the
  dual-stream blocks, split after (equivalents of `LTXVEmptyLatentAudio` /
  `LTXVConcatAVLatent` / `LTXVSeparateAVLatent`); **audio must bypass the refine pass**
- D: second `AVAssetWriterInput(mediaType: .audio)` in `LTX2PostProcess.swift:223`

**Risks recorded before starting:** no local reference to A/B against (ComfyUI can't run the
A/V path on MPS, so validation is against the author's published clips or a CUDA box); the
dual-stream path has never run end-to-end; existing audio tests assert weight shapes only,
never numerics.

### Other ComfyUI traps that cost time
- The latent upsampler MUST be the official Lightricks file (see Bug 1). ComfyUI's loader
  fails it with a bare `UnboundLocalError: model` in `nodes_hunyuan.py`.
- Core `SaveVideo` cannot mux this audio: it hands the whole waveform to AAC as ONE frame →
  `avcodec_send_frame() returned 22`. The author uses VHS_VideoCombine. Workaround: save
  audio separately (`SaveAudio`) and mux with ffmpeg — also stops a late failure discarding
  a 20-minute render.
- v2v: the author's graph has NO v2v path. Ours grafts `LoadVideo` →
  `EncodeVideoComponents` → truncated sigmas. **Encode at HALF the target** or it renders at
  2x size for 5x the runtime. `EncodeVideoComponents` requires `max_frames` and
  `upscale_method`. At denoise 0.73 the source dominates identity — good for changing
  action/scene, not who's in frame.
- 1.7-alpha vs 1.6: measured sharpness slightly favoured 1.6, but the comparison was
  confounded (per-pass distil on 1.7 only) and content differs across checkpoints even at
  fixed seed. Todd: "moving forward regardless." The #1447 rejection of 1.7 for haze was
  reached in ComfyBox/MLX with a single distil strength on both passes — **not settled**.

---

## 5. Current machine state

**ComfyBox plist** (`~/Library/LaunchAgents/com.barkadabrew.comfybox.plist`; backups in
`/private/tmp/comfybox.plist.bak-*`):
- `--ltx2-weights /Volumes/Bolt/Models/pinkcherry-v17` (**swapped from
  `~/LocalModels/sexgod-distill06-bf16`** — revert if you want the old baked-distil model)
  - ⚠️ **v17 has NO distil baked and its model_details.txt requires the official
    distil LoRA @0.6.** Nothing in the plist or daemon passes it, so every prod
    render since the swap smears moving subjects into haze (A/B verified
    2026-08-01 evening, same seed: distil-less = dissolved subject, distil @0.6 =
    clean). Fix: daemon sends
    `loras:[{path:/Volumes/Bolt/Models/ltx2-distilled/ltx-2.3-22b-distilled-lora-384.safetensors, scale:0.6}]`
    per request, or bake a v17+distil monolith (two-step bake like sexgod), or
    revert the plist to the sexgod monolith.
- `LTX2_TWO_STAGE=0` (refine broken — see §3)
- `LTX2_PLAIN_DECODE_MAX_VOL=12000` (keeps 193f/289f single-chunk off the streamed decoder)
- `LTX2_UPSAMPLER_PATH` → official file
- All six env values had **trailing whitespace** (`'1  '`, `'24  '`) which made
  `== "1"` comparisons false and paths not exist. Fixed. `Float("24  ")` is nil too.

**Other:** Kira PAUSED (`/home/todd/.kira/scheduler-paused`); kira-daemon active on merged
single-pass code; no coordination locks held; `~/Pictures/ComfyBox` → `/Volumes/Bolt/ComfyBox`
(temporary, for disk space).

**Uncommitted code changes** (written, NOT built):
1. `WarmServer.swift` `deriveVideoDims` — searches the /64 neighbourhood for minimum aspect
   error instead of rounding each axis independently. Fixes 7.7% → 0.0% (full budget),
   19-24% → ~6% (halved two-stage budgets).
2. `LTX2VideoGenerator.swift` upsampler load — counts **bound parameters** and refuses
   two-stage on mismatch. The old `loaded (N tensors)` line counted FILE tensors and
   printed "72" while binding zero; that false signal cost a full render.

---

## 6. Process lessons (these cost hours today)

- **`launchctl kickstart -k` does NOT re-read the plist.** Use `bootout` + `bootstrap`.
  A whole "parity" render ran on the old model because of this.
- **`ps eww | tr ' ' '\n'` cannot detect trailing whitespace in env values** — it splits on
  spaces, so `=1` and `=1  ` look identical. Verify via ProgramArguments (unambiguous) or
  read the plist.
- **Beware silent fallbacks.** Three today: padded env disabling two-stage, a stale restart,
  and an upsampler that "loaded" while binding nothing. Each produced plausible-but-wrong
  output. Prefer failing loud.
- **Verify the output path from the job status**, not `ls -t` — the newest file on disk was
  a previous render and nearly got reported as a result.
- A matched-seed regression harness (fixed seeds/prompts, sharpness + temporal metrics,
  stored frames) would have caught all of the above cheaply. Highest-value infrastructure item.

---

## 7. Roadmap, in order

1. ~~**Fix the refine truncation** (§3)~~ — DONE (adaptive conv chunking in
   CausalConv3d; refine verified clean end-to-end at 896x512 x 97f). NOT yet
   deployed: prod plist still has LTX2_TWO_STAGE=0 and enabling it needs the
   daemon to send HALF dims (item 3) or every request doubles.
2. ~~Build + deploy the two written code fixes (§5)~~ — DONE, deployed to the
   daemon 2026-08-01 ~21:20 with `--ltx2-lora <distil>@0.6` in the plist
   (pinkcherry-v17 alpha requires distil; bare/Kira requests were smearing).
2a. **Bake distil into pinkcherry-v17 + quantize int8** (Todd approved
   2026-08-01 night): the v17 swap silently reverted the DiT to 46GB bf16 —
   a Kira 12s 896x512 render (16.5k tokens) took ~1h denoise with constant
   memory-pressure warnings. Bake @0.6 → quantize-ltx2 int8 → repoint plist,
   drop --ltx2-lora. Then re-check render times.
3. Daemon: send HALF dims when two-stage is on, plus `img_compression: 22`
4. **NAG** — not implemented in ComfyBox; the author runs CFG 1.0 + NAG, we run CFG 3.0.
   The last un-portable reference ingredient. `LTX2Guidance.swift`
5. Per-pass split: stage-2 sampler `euler_cfg_pp`, distil 0.45 on the refine
6. Text-encoder A/B: ours `gemma-3-12b-heretic-q8` (8-bit) vs reference bf16
   `gemma-3-12b-it-heretic-v2` (already downloaded, 23GB)
7. **Audio — resume mid-Phase A** (§4 "AUDIO — work IN FLIGHT"): reference round-trip is
   done and proves the codec works on Metal; next is the Swift numeric comparison, then
   phases B–D (four wires)
8. Author-scale resolution — needs the MLX attention optimisation Todd flagged
9. Regression harness (§6)

---

## 8. Where things live

- ComfyUI validation rig: `/Volumes/Bolt/ComfyUI-validate/` — `run_validation.py`
  (`--mode t2v|i2v|v2v`), `isolate.py` (short-schedule bisection), `audio_roundtrip.py`,
  `base_prompt_video.json` (video-only), `base_prompt_parity.json` (keeps the audio path
  for retesting if the MPS bug is fixed), `author-workflows/`, `author-samples/`,
  `frames/`, `comparison/`
- Reference recipe: `docs/methods/ltx-pinkcherry-comfyui-reference.md`
- Models staged: `~/LocalModels/pinkcherry-v16b/`, `/Volumes/Bolt/Models/pinkcherry-v17/`,
  `~/LocalModels/ltx2-upsampler/`
- Memories: `ltx-single-chunk-8s`, `ltx2-two-stage-refine-broken`, `comfybox-media-on-bolt`,
  `ltx-t2v-cfg-and-decode`, `comfybox-video-checkpoint`
