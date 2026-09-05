# ComfyUI Protocol Bridge Specification

**Issue**: #21
**Date**: 2026-04-30
**Status**: Scoping complete, ready for implementation

## Overview

The Krita AI Diffusion plugin communicates with ComfyUI via HTTP + WebSocket.
This document specifies the exact protocol surface we must implement so the
plugin can connect to ZImageCLI's warm server as if it were ComfyUI.

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/system_stats` | GET | Device info |
| `/object_info` | GET | Node definitions + model lists |
| `/prompt` | POST | Submit workflow |
| `/ws?clientId=X` | WS | Progress + results |
| `/api/etn/image/<id>` | PUT | Upload input image |
| `/api/etn/image/<id>` | GET | Download result image |
| `/api/etn/model_info/<folder>` | GET | Model metadata |
| `/interrupt` | POST | Cancel current job |
| `/queue` | POST | Delete queued jobs |

## 1. GET /system_stats

Called first during `ComfyClient.connect()`. `DeviceInfo.parse` reads:

```json
{
  "devices": [{
    "name": "Apple M3 Max",
    "type": "mps",
    "vram_total": 137438953472
  }]
}
```

Fields consumed:
- `devices[0].name` — split on ":" if present
- `devices[0].type` — string: "cuda", "mps", "cpu", "privateuseone"
- `devices[0].vram_total` — bytes, divided by 1024^3 and rounded to GB

## 2. GET /object_info

Returns node type definitions. The plugin checks `_check_for_missing_nodes`
against `required_custom_nodes` in `resources.py`.

### Required nodes that MUST appear

- **comfyui_controlnet_aux**: `InpaintPreprocessor`, `DepthAnythingV2Preprocessor`
- **ComfyUI_IPAdapter_plus**: `IPAdapterModelLoader`, `IPAdapter`
- **comfyui-tooling-nodes**: `ETN_LoadImageCache`, `ETN_SaveImageCache`, `ETN_Translate`
- **comfyui-inpaint-nodes**: `INPAINT_LoadFooocusInpaint`, `INPAINT_ShrinkMask`, `INPAINT_StabilizeMask`, `INPAINT_ColorMatch`

### Z-Image model discovery queries

The plugin queries options from these nodes to discover available models:
- `DualCLIPLoader` / `DualCLIPLoaderGGUF` → `clip_name1` (text encoders)
- `VAELoader` → `vae_name`
- `ControlNetLoader` → `control_net_name`
- `UpscaleModelLoader` → `model_name`
- `ModelPatchLoader` → `name` (Z-Image fun controlnet patches)
- `UNETLoader` / `NunchakuZImageDiTLoader` (diffusion model)
- `CLIPLoader` with `type: "lumina2"` (Z-Image text encoder)

### Schedule-shift node (comfybox#154)

`ModelSamplingAuraFlow` is advertised and its `shift` input is read:

```json
{"ModelSamplingAuraFlow": {
  "input": {"required": {"model": ["MODEL", {}], "shift": ["FLOAT", {"default": 1.73}]}},
  "output": ["MODEL"], "category": "model/patch"}}
```

`ModelSamplingSD3` is advertised and read identically (upstream default `3.0`):
it is the same class with a different timestep `multiplier`, and the multiplier
cancels out of the sigma grid, so the same `shift` gives the same schedule.

Both map to the engine's `shift` request field (the flow schedule's linear
shift), and are honoured **only when the resident model family is Z-Image** —
on any other family `/prompt` returns a 400 naming the node and the family.

`ModelSamplingFlux` is accepted in a graph but NOT read: its shift is a
log-shift, a different curve.

### Response format per node

```json
{
  "NodeName": {
    "input": {
      "required": {
        "param": ["type_or_options", {"opts": "..."}]
      },
      "optional": {}
    },
    "output_name": ["..."]
  }
}
```

## 3. POST /prompt

### Request

```json
{
  "prompt": {
    "1": { "class_type": "...", "inputs": {...} },
    "2": { "class_type": "...", "inputs": {...} }
  },
  "client_id": "<uuid>",
  "prompt_id": "<uuid>"
}
```

### Response

Must include `{"prompt_id": "<same uuid>"}`. A mismatch raises an error.

## 4. WebSocket Protocol

### Connection

`ws://<host>/ws?clientId=<client_id>`
- Max message size: 2^30 bytes
- Ping timeout: 60s

### Text messages (JSON)

| type | data fields | Purpose |
|------|------------|---------|
| `status` | — | Triggers `connected` event |
| `execution_start` | `prompt_id` | Job started |
| `executing` | `prompt_id`, `node` (null = done) | Node executing; node=null = workflow complete |
| `execution_cached` | `prompt_id`, `nodes` (list) | Cached nodes skipped |
| `progress` | `prompt_id` | Sampling step progress |
| `executed` | `prompt_id`, `output.images[].source`, `output.images[].id` | Node output with images |
| `execution_interrupted` | `prompt_id` | Job cancelled |
| `execution_error` | `prompt_id`, `exception_message`, `traceback` | Job failed |

### Progress formula

```
0.2 * (nodes_done / (total+1)) + 0.8 * (samples_done / sample_count)
```

### Binary messages (preview images)

Header: `>II` (big-endian, 2x uint32)
- Event type 1 = `PREVIEW_IMAGE`
- Format 2 = PNG
- Image data follows immediately after the 8-byte header

## 5. Image Transfer

### Input images (inpainting)

Uploaded via `PUT /api/etn/image/<id>` as raw bytes.
The `<id>` is a CRC32 hash of the PNG data (`_add_image_hashed`).
In the workflow, `ETN_LoadImageCache` nodes reference images by this hash ID.

### Output images

Retrieved via `GET /api/etn/image/<id>` when the `executed` message
contains `output.images` with `source: "http"` and an `id` field.

The `ETN_SaveImageCache` node is the output node (replaces `SaveImage`),
with `format: "PNG"`.

## 6. Z-Image txt2img Workflow

The plugin builds this node graph for `Arch.zimage`:

1. **Load model**: `NunchakuZImageDiTLoader` (svdq quantized) or `UNETLoader` (diffusion format)
2. **Load CLIP**: `CLIPLoader` with `type: "lumina2"`, loading `qwen_3_4b` text encoder
3. **Load VAE**: `VAELoader` (z-image or flux VAE)
4. **Encode prompts**: `CLIPTextEncode` for positive and negative
5. **Empty latent**: `EmptySD3LatentImage` (Z-Image uses SD3 latent format)
6. **Sampling**: `SamplerCustomAdvanced` with `BasicGuider` (cfg=1) or `CFGGuider`, `BasicScheduler`, `RandomNoise`, `KSamplerSelect`
7. **Decode**: `VAEDecode`
8. **Output**: `ETN_SaveImageCache` with format "PNG"

## 7. Z-Image Inpaint Workflow

Differences from txt2img:

1. Adds `DifferentialDiffusion` node wrapping the model
2. Input image loaded via `ETN_LoadImageCache`, mask loaded separately
3. Mask processed through `INPAINT_ExpandMask` (grow/feather) and `INPAINT_StabilizeMask`
4. Fill masked region via `INPAINT_MaskedBlur` (blur fill mode)
5. If inpaint controlnet available: `ZImageFunControlnet` node applies the model patch from `ModelPatchLoader` with `inpaint_image`, `mask`, `vae`, `model_patch`, and `strength: 0.5` at range `(0.0, 1.0)`
6. Otherwise: `VAEEncode` + `SetLatentNoiseMask` for standard latent-space inpainting
7. Uses `INPAINT_VAEEncodeInpaintConditioning` when `use_inpaint_model` is true and no controlnet
8. Post-process: `INPAINT_ColorMatch`, `ETN_ApplyMaskToImage` for compositing

### Z-Image controlnet patches

- `z-image-turbo-fun-controlnet-union-2.1` (universal)
- `z-image-turbo-fun-controlnet-tile-2.1` (blur/tile)

## Implementation Strategy

### Phase 1: Discovery (static responses)
- `/system_stats` — hardcoded device info from MLX Metal query
- `/object_info` — static JSON declaring all required nodes with our model paths
- `/ws` — basic WebSocket accept, send `status` on connect

### Phase 2: txt2img
- `/prompt` — parse workflow JSON, match Z-Image txt2img pattern, extract: prompt, negative prompt, width, height, steps, seed, sampler, scheduler, cfg
- Route to ZImageCLI pipeline
- Send progress events over WebSocket during denoising
- Save output to `/api/etn/image/<id>`, send `executed` event

### Phase 3: Inpaint
- `/api/etn/image/<id>` PUT — accept and store input images
- Parse inpaint workflow, extract: image, mask, fill mode, grow, feather, strength
- Route to inpaint pipeline (requires inpaint support in ZImageCLI)
- Compositing and color matching post-process

### Phase 4: ControlNet
- Parse controlnet nodes from workflow
- Route control images and mode to ControlNet pipeline
