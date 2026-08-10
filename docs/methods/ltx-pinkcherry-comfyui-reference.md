# PinkCherry LTX-2.3 — validated ComfyUI reference recipe

**Validated 2026-08-01 on the M3 Max (137 GB) against ComfyUI 0.29.0 + MPS.**
This is the reference our MLX/ComfyBox engine is measured against: settings taken
from the author's own renders, confirmed to produce clean output here.

Source of truth for the settings is not the model card — it is the workflow JSON
embedded in the author's published sample clips (`v1.6/*.mp4`, metadata `comment`
field, which holds `{"prompt": <api graph>, "workflow": <ui graph>}`). Those are
what he actually rendered with, and they differ from the shipped workflow file.

## Environment

| Piece | Value |
|---|---|
| ComfyUI | 0.29.0 at `/Volumes/Bolt/ComfyUI-validate/ComfyUI`, `--listen 127.0.0.1 --port 8188 --cache-none` |
| Custom nodes | `ComfyUI-LTXVideo`, `ComfyUI-KJNodes` |
| Checkpoint (1.6) | `SexGod_PinkCherry_dev_bf16_LTX23_v16b.safetensors` (46 GB) |
| Checkpoint (1.7a) | `PinkCherry_FineTune_bf16_v1_7-alpha.safetensors` (46 GB) |
| Text encoder | `gemma-3-12b-it-heretic-v2.safetensors` + `ltx-2.3_text_projection_bf16.safetensors`, `DualCLIPLoader` type `ltxv` |
| Video VAE | `LTX23_video_vae_bf16.safetensors` (`Kijai/LTX2.3_comfy`, `vae/`) |
| Audio VAE | `LTX23_audio_vae_bf16.safetensors` (same repo) — unused, see Deviations |
| Latent upscaler | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` from **`Lightricks/LTX-2.3`** |
| Distil LoRA | `ltx-2.3-22b-distilled-lora-384.safetensors` |

## Author-exact settings

| Setting | Value |
|---|---|
| Sampler pass 1 | `euler_ancestral_cfg_pp` |
| Sampler pass 2 | `euler_cfg_pp` |
| CFG | 1.0 (both passes) |
| Stage-1 sigmas | `1.0, 0.998, 0.995, 0.99, 0.982, 0.97, 0.94, 0.89, 0.82, 0.73, 0.62, 0.50, 0.38, 0.27, 0.18, 0.11, 0.06, 0.03, 0.01, 0.0` |
| Stage-2 sigmas | `0.85, 0.7250, 0.4219, 0.0` |
| NAG | `LTX2_NAG` scale 11.0, alpha 0.25, tau 2.5, inplace |
| Distil strength | 0.6 (see per-pass note below) |
| `LTXVPreprocess img_compression` | **22** |
| Conditioning frame rate | 24 |
| Structure | stage 1 at half target resolution → `LTXVLatentUpsampler` x2 → stage 2 refine |
| Duration math | `1 + 8*round(seconds*fps/8)` frames — 5 s @ 24 = 121f |

**Per-pass distil strength.** HF discussion #8 (tarn1's 1.7-alpha field report) states
the distil LoRA runs at **0.6 on pass 1 and 0.45 on pass 2**. These are per-pass
values, not competing alternatives — an ambiguity that previously led to A/B arms
testing one strength across both passes. Implemented here by giving stage 2 its own
`LoraLoaderModelOnly` + `LTX2_NAG` chain into the stage-2 guider.

Also from that thread, untested here: the 384 distil reportedly dampens motion and
makes eyes "jumpy"; Kijai's `ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16`
is lighter on RAM and reportedly improves movement. An alternative 10-step stage-1
schedule: `1.000, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0`.

## Deviations from the author, and why

1. **`VAEDecode` instead of `VAEDecodeTiled`.** The author decodes tiled
   (512/64/4096/8). Todd's instruction is to avoid tiling. Non-tiled decode
   completed without memory trouble at 1280x768 x 121f and at 2560x1536 x 121f.
2. **Audio path removed — forced, not chosen.** LTX-2.3's joint A/V latent path
   produces **black video and silent audio** on ComfyUI 0.29.0 + MPS. Established by
   bisection, not inference:

   | NAG | Audio | Result |
   |---|---|---|
   | on | on | black |
   | off | on | black |
   | off | off | content |
   | on | off | content |

   Audio is the sole discriminator; NAG is innocent and is retained. The same fault
   shows up earlier in the chain as `RuntimeError: cannot reshape tensor of 0 elements
   into shape [2, 0, 32, -1]` in `apply_rotary_emb_qk`, reached from
   `comfy/ldm/lightricks/av_model.py`. Consequence: **audio quality cannot be
   validated on Apple silicon** with this stack.

## Traps that cost time

- **The latent upscaler must be the official Lightricks file.** ComfyBox's converted
  `spatial_upscaler_x2_v1_1.safetensors` has keys prefixed `spatial_upscaler_x2_v1_1.*`
  and no `__metadata__.config`, so ComfyUI's `LatentUpscaleModelLoader` matches none of
  its branches and fails with `UnboundLocalError: model`.
- **`img_compression` is 22, not 2.** The shipped workflow and any hand-rebuilt graph
  may carry a different value.
- **Core `SaveVideo` cannot mux this audio.** It hands the whole waveform to AAC as one
  frame → `avcodec_send_frame() returned 22`. The author uses VHS_VideoCombine. Save
  audio separately (`SaveAudio`) and mux with ffmpeg, so a late failure cannot discard
  a 20-minute render.
- **v2v: encode at half the target resolution.** Stage 1 runs at half; encoding the
  source at full size overshoots to 2x the intended output (and ~5x the runtime).

## Validation results

All at 1280x768, 5 s (121 frames), seed 4242, ~18 min per render.

| Model | Mode | Content | Result |
|---|---|---|---|
| 1.6-dev | t2v | NSFW | clean start to finish, no drift |
| 1.6-dev | t2v | SFW | clean, genuine per-frame motion blur |
| 1.6-dev | i2v | NSFW | identity and scene held from source still |
| 1.6-dev | i2v | SFW | scene held, motion advanced |
| 1.6-dev | v2v | NSFW | source scene preserved, subject re-rendered (denoise 0.73) |
| 1.7-alpha | t2v | NSFW | sharp, **no haze**, better prompt adherence than 1.6 |
| 1.7-alpha | i2v | NSFW | sharp, identity held |

v2v note: at denoise 0.73 the source dominates identity — good for changing
action/scene, not for changing who is in frame.

**1.7-alpha and the earlier rejection.** coffeeshop-server#1447 closed 1.7-alpha as
REJECTED for haze/soft-focus. That verdict came from ComfyBox/MLX renders, with a
single distil strength across both passes. In ComfyUI with per-pass distil, 1.7-alpha
is sharp. That does not by itself prove the haze was configuration rather than engine —
the two runs differ in more than one variable — but the rejection should not be
treated as settled.

## What this says about ComfyBox

The reference renders **121 frames clean in a single pass**. Our engine's overnight
output (2026-07-31/08-01) corrupts precisely where it chains a continuation chunk:
frames 0-96 clean, chroma blotching from ~frame 100, compounding to the end, on 2 of 2
multi-chunk clips sampled. Both chunks logged `plain single-pass` decode under the
safety gate, so the decoder is not implicated there. Separately, the 241f clip
(`volume 11284 > 4500`, streamed decode) shows a 4-tile mosaic at an interior chunk
boundary while neighbouring frames are fine.

So the tuning target is the **chaining** — chunk-seed conditioning compression 35,
`strength=1.0` continuation, and the temporal color anchor defaulted OFF since
2026-07-27 — not the model. ComfyBox also emits no audio at all, which is moot for
parity while the reference stack cannot do audio on Metal either.

## Reproducing

```bash
cd /Volumes/Bolt/ComfyUI-validate
./ComfyUI/.venv/bin/python run_validation.py --name my-run --mode t2v \
  --seed 4242 --prompt "..." \
  --checkpoint SexGod_PinkCherry_dev_bf16_LTX23_v16b.safetensors \
  --lora-strength 0.6 --lora-strength-pass2 0.45
```

`--mode i2v --image <file in ComfyUI/input>`; `--mode v2v --video <clip> --denoise 0.73`.
Graph: `base_prompt_video.json` (video-only). `base_prompt_parity.json` keeps the audio
path for retesting if the upstream MPS bug is ever fixed. `isolate.py` runs the
short-schedule bisection variants.
