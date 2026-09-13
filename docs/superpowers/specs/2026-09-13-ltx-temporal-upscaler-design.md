# LTX 2.3 temporal upscaler (×2 fps in latent space) — design

Date: 2026-09-13. Requested by Todd ("Do the proper fix") after the 24 fps
judder analysis of scheduler clips: the model moves bodies too far per frame
at 24 fps, and pixel-space interpolation (ffmpeg minterpolate) warps at exactly
those instants.

## Goal

Render the clip the engine already renders (289 frames, 24 fps, single pass),
then double its frame rate in latent space with Lightricks'
`ltx-2.3-temporal-upscaler-x2-1.0` before decode, delivering 577 frames at
48 fps with unchanged duration and audio. Opt-in per request; the scheduler
adopts it only after Todd judges a same-latent A/B.

## Model

Same trunk as the ported spatial `LTX2LatentUpsampler` (initial Conv3d →
GroupNorm → SiLU → 4 ResBlock3D → upsample → 4 ResBlock3D → final Conv3d)
with three differences, all from the checkpoint's embedded config
(`spatial_upsample: false, temporal_upsample: true, mid_channels: 512`):

1. `mid_channels` 512 (spatial: 1024).
2. Upsample stage: `Conv3d(512 → 1024, k3, p1)` then a **temporal** pixel
   shuffle: `b (c p) f h w -> b c (f p) h w`, p = 2 — channel `2c` feeds output
   frame `2f`, channel `2c+1` feeds `2f+1`.
3. The first output frame is dropped (`x[:, :, 1:]`): the first latent frame
   encodes exactly one pixel frame. F latent frames → 2F − 1.

Applied on un-normalized latents (`unNormalize` → upsampler → `normalize`),
identical to the spatial path. 72 tensors, bf16, 262 MB, at
`~/LocalModels/ltx2-upsampler/ltx-2.3-temporal-upscaler-x2-1.0.safetensors`.
Loader: the existing PyTorch→MLX permute and `upsampler.0.*` → `upsampler.conv.*`
rename; must bind all 72 or refuse (a partial bind renders a mesh — 2026-08-01).

## Pipeline

After the base denoise (and after the spatial refine if enabled), when the
request carries `temporal_upscale: 2`:

1. `latents (1,128,37,H,W)` → temporal upsampler → `(1,128,73,H,W)`.
2. Decode through the existing tiled decoder → 577 frames.
3. Mux at `fps × 2` (48). Audio latents untouched (same duration).

Decode-only first. No second denoise pass: the A/B decodes the SAME base
latents at 24 and at 48 so the upscaler is judged alone. If the 48 fps decode
looks soft, phase 2 adds a short refine denoise at 48 fps (existing refine
machinery, position grid at latF 73, cond fps 48).

Cost: decode ≈ 2× (~50 s → ~100 s at 448×704); denoise unchanged; upsampler
< 10 s. Memory: 73×56×88×512 activations ≈ 0.7 GB per layer — fine.

## Surface

- Engine: `LTX2ResolvedConfig.temporalUpsamplerPath` (`temporal_upsampler_path`,
  env `LTX2_TEMPORAL_UPSAMPLER_PATH`), lazy-loaded once like the spatial one.
- Request: `temporal_upscale` (int, 1 or 2) on `/v1/video/generate[/async]`,
  MCP `generate_video`, and the winner-action replay. Sidecar records
  `temporal_upscale` and `output_fps`.
- Daemon (coffeeshop-server, later): `generate_video` passes it through;
  scheduler config `videoTemporalUpscale` — after the A/B.

## Tests

1. Temporal pixel-shuffle ordering vs an explicit index construction.
2. Loader binds 72/72 tensors from the real checkpoint (skipped if absent).
3. Shape: 5 latent frames → 9.
4. Parity vs the PyTorch reference (ComfyUI venv, `comfy.ldm.lightricks.
   latent_upsampler.LatentUpsampler` with the temporal config and the real
   weights) on a seeded (1,128,5,8,8) latent — fixture pinned under
   `Tests/ZImageTests/Fixtures/`, tolerance bf16-appropriate.
5. Pipeline: frame count 289 → 577 and mux fps 48 on the request path
   (integration test gated on weights).

## Out of scope

Native 30/48 fps generation, changes to the 289-frame window, the spatial
refine path, scheduler defaults.
