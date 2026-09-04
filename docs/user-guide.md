# ComfyBox User Guide

## Quick Start

```bash
# Generate an image
ComfyBox -p "A mountain landscape at sunset" -o landscape.png

# Generate with a specific seed (reproducible)
ComfyBox -p "A portrait of a woman" --seed 42 -o portrait.png

# Run as a warm server (keeps model loaded between requests)
ComfyBox serve -m Tongyi-MAI/Z-Image-Turbo --port 7862
```

## Installation

### Pre-built Binary

```bash
curl -LO https://github.com/BarkadaBrew/comfybox/releases/download/latest/ComfyBox-macos-arm64.tar.gz
tar -xzf ComfyBox-macos-arm64.tar.gz
cd ComfyBox
sudo ./install.sh
```

Installs to `/usr/local/lib/comfybox/ComfyBox` with a wrapper at `/usr/local/bin/ComfyBox`.

### Build from Source

```bash
git clone https://github.com/BarkadaBrew/comfybox.git
cd comfybox
xcodebuild -scheme ComfyBox -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/xcode
```

Binary and Metal library appear in `.build/xcode/Build/Products/Release/`.

### Requirements

- macOS 14.0+ (Sonoma or later)
- Apple Silicon (M1, M2, M3, M4)
- ~6 GB disk for base model (auto-downloaded on first run)
- Swift 5.9+ (build from source only)

## Text-to-Image

### Basic Generation

```bash
ComfyBox -p "A beautiful mountain landscape at sunset" -o output.png
```

### Resolution

Width and height must be divisible by 16:

```bash
# Landscape (16:9)
ComfyBox -p "panoramic desert" -W 1920 -H 1080 -o desert.png

# Portrait (2:3)
ComfyBox -p "portrait of a woman" -W 768 -H 1152 -o portrait.png

# Square (1:1)
ComfyBox -p "abstract art" -W 1024 -H 1024 -o art.png
```

For resolutions above 1024px, DyPE (Dynamic Positional Encoding) activates automatically to maintain quality. Disable with `--no-dype` if needed.

### Prompt Enhancement

Use the built-in LLM to expand a short prompt into a detailed one:

```bash
ComfyBox -p "a cat" --enhance -o cat.png
```

This loads the Qwen text encoder in generation mode (~5 GB extra VRAM) to write a richer prompt. Control output length with `--enhance-max-tokens`.

### Seeds

Seeds make generation reproducible — same seed + same prompt = same image:

```bash
# Fixed seed
ComfyBox -p "a rose" --seed 42 -o rose.png

# Multiple seeds (batch)
ComfyBox -p "a rose" --seed 42 --seed 99 --seed 7 -o roses/rose.png
```

### Samplers and Schedules

Different sampler/schedule combinations produce different aesthetics:

```bash
# Default (euler + flow) — fastest, good quality
ComfyBox -p "a forest" -o forest.png

# Heun (higher quality, 2x slower)
ComfyBox -p "a forest" --scheduler heun -s 9 -o forest-heun.png

# DPM++ 2M with Karras schedule
ComfyBox -p "a forest" --scheduler dpmpp-2m --sigma-schedule karras -s 20 -o forest-dpm.png

# DDIM (deterministic, good for animation)
ComfyBox -p "a forest" --scheduler ddim -s 20 -o forest-ddim.png
```

Available samplers: `euler`, `heun`, `res_2s`, `dpmpp-2m`, `dpmpp-2s-a`, `deis`, `ddim`
Available schedules: `flow`, `karras`, `exponential`, `beta`, `beta57`

### Post-Processing

Adjust output contrast with levels:

```bash
# Increase contrast (crush blacks, boost whites)
ComfyBox -p "a portrait" --levels-min 0.05 --levels-max 0.95 -o portrait.png
```

## Image-to-Image

Transform an existing image using a text prompt:

```bash
# Heavy rework (strength 0.3 — most change from original)
ComfyBox -p "oil painting style" --image photo.jpg --image-strength 0.3 -o painting.png

# Light touch (strength 0.7 — preserves most of the original)
ComfyBox -p "enhance lighting" --image photo.jpg --image-strength 0.7 -o enhanced.png

# Using "creativity" (inverse of strength — more intuitive)
ComfyBox -p "watercolor style" --image photo.jpg --creativity 0.7 -o watercolor.png
```

> **Note:** ComfyBox strength is inverted from the typical convention. `--image-strength 0.3` = heavy rework (skips fewer steps, runs more denoising). `--image-strength 0.7` = light touch. The `--creativity` flag provides the intuitive direction: higher = more creative change.

## LoRA

LoRA weights customize the model's style without full fine-tuning:

```bash
# Single LoRA
ComfyBox -p "a lion" --lora ostris/z_image_turbo_childrens_drawings -o lion.png

# Single LoRA with scale
ComfyBox -p "a portrait" --lora style.safetensors=0.8 -o portrait.png

# Multiple LoRAs stacked
ComfyBox -p "a fantasy portrait" \
  --lora style.safetensors=0.8 \
  --lora detail.safetensors=0.6 \
  -o combined.png
```

LoRA sources:
- **HuggingFace ID:** `ostris/z_image_turbo_childrens_drawings` (auto-downloaded)
- **Absolute path:** `/Users/me/loras/style.safetensors`
- **Relative path:** `./loras/style.safetensors`

### Krea-2 Raw accelerator presets

Krea-2 Raw can use a distillation LoRA as an accelerator. In the CoffeeShop
Desktop Preset editor, keep the adapter visible in the ordinary LoRA list so
its scale remains adjustable, then set its role menu to **Accelerator**.

For the tested r256 stack:

- File: `krea2_turbo_distill_r256.safetensors`
- Role: **Accelerator** (`accel` on the wire)
- Current Krea-Kira scale: `0.6`
- Normal guidance: `1`

Kroma is separate: adjust it through the structured Kroma row, not by adding
the Kroma file a second time to the ordinary LoRA list. A duplicate Kroma
entry makes the preset contradictory and is rejected.

Guidance around `2` enables the positive/negative CFG pair and makes negative
prompts active, at roughly twice the model evaluations. It is optional and
is not required by the r256 accelerator.

See [Krea-2 Raw + r256 preset stack](methods/krea2-r256-preset-stack.md) for
the exact preset JSON and non-rendering verification commands.

### LoRA Library Management

```bash
# List available LoRAs
ComfyBox lora list

# Filter by model compatibility
ComfyBox lora list --model z-image

# Show detailed info
ComfyBox lora info my-style-lora

# Scan for new LoRA files
ComfyBox lora scan

# Check compatibility with a model
ComfyBox lora check KLEIN-Unchained-V2 --model z-image

# Quarantine an incompatible LoRA
ComfyBox lora quarantine bad-lora --reason "Wrong architecture"

# Search by text
ComfyBox lora search portrait
```

## ControlNet

Generate images conditioned on a control image (edges, depth, pose):

```bash
# Canny edge control
ComfyBox control \
  -p "A leopard face in jungle" \
  --control-image canny_edges.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --control-scale 0.75 \
  -o leopard.png

# Inpainting
ComfyBox control \
  -p "a cat sitting on the chair" \
  --inpaint-image room.jpg \
  --mask mask.png \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  -o inpainted.png
```

**Control types:** Canny, HED, Depth, Pose (OpenPose/DWPose), MLSD

**Tips:**
- `--control-scale 0.65-0.90` — higher = stricter adherence to control image
- Increase `--steps` for higher control scale values
- Inpainting mask: white = fill, black = preserve

## Upscaling

### SeedVR2 (AI Upscale)

AI-powered 2x upscaling with intelligent detail synthesis:

```bash
# Basic 2x upscale to 2048px
ComfyBox upscale -i photo.jpg -w ~/Models/seedvr2-3b -o photo-2x.png

# Target 4096px with fixed seed
ComfyBox upscale -i photo.jpg -w ~/Models/seedvr2-3b -r 4096 --seed 42 -o photo-4k.png

# With preprocessing softness (smooths input before upscale)
ComfyBox upscale -i photo.jpg -w ~/Models/seedvr2-3b --softness 0.3 -o photo-soft.png
```

SeedVR2 models:
- **3B** — ~7 GB VRAM, faster, good for most images
- **7B** — ~16 GB VRAM, higher quality detail synthesis

### ESRGAN (Traditional Upscale)

Deterministic upscaling with tiling for large images:

```bash
ComfyBox upscale -i photo.jpg --esrgan-weights ./models/4x-ultrasharp --tile-size 512 -o photo-4x.png
```

## Batch Mode

Generate multiple images from the same prompt with different seeds:

```bash
# Auto-generate 10 random seeds
ComfyBox -p "a portrait" --auto-seeds 10 -o batch/portrait.png

# Specific seeds
ComfyBox -p "a portrait" --seed 42 --seed 99 --seed 7 -o batch/portrait.png

# Resume an interrupted batch
ComfyBox -p "a portrait" --resume-batch batch/portrait-checkpoint.json

# Re-read prompt from file each iteration (for external prompt mutation)
ComfyBox --prompt-file prompt.txt --auto-seeds 5 -o batch/output.png
```

Output files are named with seed suffixes: `portrait-seed42.png`, `portrait-seed99.png`, etc.

Batch runs are sequential (one GPU render at a time) with checkpoint files for resume. If the process is interrupted, `--resume-batch` picks up from the last completed seed.

## Models

### List Installed Models

```bash
ComfyBox models           # Show model families and status
ComfyBox models --paths   # Show filesystem paths
```

### Model Families

| Family | Default ID | Use Case |
|--------|-----------|----------|
| Z-Image Turbo | `Tongyi-MAI/Z-Image-Turbo` | Fast generation (4-9 steps) |
| Z-Image Base | `Tongyi-MAI/Z-Image` | High quality with CFG guidance |
| Flux 2 Klein | varies | High quality 9B model |
| FIBO | `briaai/FIBO` | JSON-structured prompts |
| Chroma | varies | Guidance-free architecture |

### Using Different Models

```bash
# Z-Image Turbo (default)
ComfyBox -p "a cat" -o cat.png

# Quantized model (less VRAM)
ComfyBox -p "a cat" -m mzbac/Z-Image-Turbo-8bit -o cat.png

# Local model directory
ComfyBox -p "a cat" -m ~/Models/z-image-turbo-bf16 -o cat.png

# AIO checkpoint (single .safetensors with all components)
ComfyBox -p "a cat" -m path/to/aio-checkpoint.safetensors -o cat.png
```

### Quantization

Reduce memory usage by quantizing weights:

```bash
ComfyBox quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --bits 8
```

| Precision | Memory | Quality | Speed |
|-----------|--------|---------|-------|
| BF16 | ~21 GB | Best | Baseline |
| 8-bit | ~7.5 GB | Near-identical | ~Same |
| 4-bit | ~4 GB | Slight degradation | Slightly faster |

## SVG Export

Convert generated images to scalable vector graphics:

```bash
# Basic SVG
ComfyBox -p "geometric pattern" --svg -o pattern.png
# Creates: pattern.png AND pattern.svg

# Logo preset (clean, minimal)
ComfyBox -p "coffee cup logo, flat design" --svg --svg-preset logo -o logo.png
```

Requires [vtracer](https://github.com/visioncortex/vtracer):
```bash
cargo install vtracer
```

**Presets:** `default`, `logo`, `detailed`, `simplified`, `bw`

**Tips for best SVG results:**
- Use high contrast prompts: "bold colors, white background"
- Add "flat design, no gradients" for cleaner conversion
- `logo` preset produces the smallest, cleanest files

## Metadata

### Generation Metadata

Save generation parameters alongside the output image:

```bash
ComfyBox -p "a cat" --metadata -o cat.png
# Creates: cat.png AND cat.json
```

The JSON sidecar contains all parameters needed to reproduce the image: prompt, seed, steps, guidance, model, LoRAs, scheduler, resolution.

### Reproduce from Metadata

Load parameters from a previous generation's metadata:

```bash
# Exact reproduction
ComfyBox --config-from-metadata cat.json -o cat-copy.png

# Override specific parameters
ComfyBox --config-from-metadata cat.json --steps 20 -o cat-hq.png
```

## Scripting

For automation and pipelines:

```bash
# Disable progress bars
ComfyBox -p "batch image" --no-progress -o output.png

# Audit model weight coverage (diagnostic)
ComfyBox -p "" --audit-weights -m ~/Models/my-model
```

## GPU Memory Management

```bash
# Set cache limit (prevents OOM on constrained devices)
ComfyBox -p "a scene" --cache-limit 8192 -o scene.png
```

**Rules of thumb:**
- M1/M2 (16 GB): Use 8-bit models, avoid SeedVR2 7B
- M1/M2 Pro (32 GB): BF16 models work fine, one at a time
- M3 Max (128 GB): Multiple BF16 models simultaneously
- Never run multiple GPU renders concurrently — sequential only

### DyPE / high-resolution pre-flight (#22)

A high-resolution request (DyPE-territory, i.e. above 1024px on the longer
edge) is checked BEFORE the server loads or runs anything. Two independent
gates run, in order:

1. **Resolution cap** — a hard ceiling on the request itself, checked with no
   memory probing at all: the longer edge must be at or under
   `imageMemoryCaps.maxLongEdge` (default **4096px**), and total pixels
   (`width * height`) must be at or under `imageMemoryCaps.maxPixels`
   (default **16,777,216**, i.e. 4096²). A non-square request can still be
   refused here even under the long-edge cap (e.g. 4096×4097). **Always
   enforced** — refuses with HTTP 413 (REST) / 400 (the ComfyUI bridge's
   `POST /prompt`) regardless of any other setting below.
2. **Live memory budget** — the render's estimated peak activation memory is
   compared against how much system memory is actually free right now.

**The memory-budget estimate is ADVISORY by default** —
`imageMemoryCaps.enforceMemoryEstimate` is **false** out of the box. The
estimate formula is not yet calibrated against any live memory trace (see
"Calibrating the estimate" below), so an unenforced estimate that runs high
does NOT refuse the request: the server logs a warning and the response
carries two additive fields, `memory_estimate_bytes` and
`memory_available_bytes`, so you can watch how the estimate tracks reality
before deciding whether to trust it. Only the resolution cap is a hard limit
until you turn enforcement on.

```json
{
  "success": true,
  "output_path": "/Users/todd/Pictures/ComfyBox/img-1234.png",
  "duration_ms": 84213,
  "memory_estimate_bytes": 3854761984,
  "memory_available_bytes": 106300440576
}
```

If a request is refused (resolution cap always; the memory budget only when
`enforceMemoryEstimate` is true), the response is instead:

```json
{
  "success": false,
  "error": "[insufficient_memory] estimated 3676MB exceeds the 921MB memory cap (1024MB available right now)",
  "error_code": "insufficient_memory",
  "estimate_bytes": 3854761984,
  "available_bytes": 1073741824,
  "cap_bytes": 966367641
}
```

`error_code` is `"resolution_cap"` (no `estimate_bytes`/`available_bytes` —
refused before any probing) or `"insufficient_memory"` (only reachable when
enforcement is on). Existing error responses are unaffected — every field
above is new and additive (`Optional`, omitted rather than `null` when
absent).

**Config keys** (`~/.comfybox/config.json`, writable via `PATCH /v1/config`
— see `engine.imageMemoryCaps.*` in [api-reference.md](api-reference.md)):

| Key | Default | Meaning |
|---|---|---|
| `imageMemoryCaps.maxLongEdge` | 4096 | Hard ceiling (px) on the longer of width/height. Always enforced. Bounded to ≤16384. |
| `imageMemoryCaps.maxPixels` | 16777216 | Hard ceiling on width×height. Always enforced. Bounded to ≤2²⁸ (268,435,456). |
| `imageMemoryCaps.minAvailableHeadroomFraction` | 0.10 | Fraction of live free memory a render's estimated footprint must leave clear. Only acted on when `enforceMemoryEstimate` is true. |
| `imageMemoryCaps.enforceMemoryEstimate` | **false** | Whether the memory-budget estimate above actually refuses a request. See "Calibrating the estimate" below before flipping this on. |

Neither cap is a hardcoded per-device table — when enforcement IS on, the
live memory-budget gate adapts to whatever headroom your particular Mac has
free right now (shared with LM Studio, a resident video model, etc. — see
`intent.md`, "Memory is a shared resource"), which is stricter than a static
table on a loaded machine and more permissive on an idle one.

`--cache-limit` (the MLX buffer-cache ceiling) is orthogonal to these caps: it
bounds how much *idle* buffer cache MLX is allowed to retain between calls, not
what a single render is projected to need. Lowering it can reduce steady-state
RSS between renders, but does not change what the pre-flight estimates for the
render you are about to submit.

#### Calibrating the estimate before enforcing it

`imageMemoryCaps.enforceMemoryEstimate` should stay `false` until you have
some evidence the estimate formula tracks real memory pressure on your
machine. A simple recipe:

1. Leave enforcement off. Submit renders across a spread of resolutions —
   1024², 1536², 2048², with and without DyPE — under whatever else is
   normally resident (LM Studio, an idle/warm video model, etc.).
2. For each render, note `memory_estimate_bytes` from the response
   alongside a `GET /health` sample taken just before and just after (its
   `memory.system_free_mb`/`memory.process_rss_mb` fields — see
   `StatsProvider.swift`). The estimate and the observed drop in free memory
   should move together, even if the absolute numbers don't match exactly —
   `ImageMemoryPreflight.swift`'s header lists which constants are cited
   engine values and which are explicitly-flagged, uncalibrated assumptions
   (a safety multiplier for concurrent-layer activations; the VAE decoder's
   channel-width modeling).
3. Once you trust the estimate's SHAPE (it refuses roughly where real
   pressure shows up, not wildly early or wildly late), `PATCH /v1/config`
   `imageMemoryCaps.enforceMemoryEstimate: true`. From then on, an
   over-budget request gets the `insufficient_memory` refusal shown above
   instead of just a logged warning.
4. If it turns out to be miscalibrated once live, flip it back to `false`
   and retune the constants in `ImageMemoryPreflight.swift` (each is
   individually documented with what it approximates and why) rather than
   just raising `minAvailableHeadroomFraction` to paper over it.

**The estimate is activations-only.** It does not count the resident model's
own weights — only the extra memory the RENDER's activations need on top of
whatever is already loaded. That is deliberate (the model may already be
warm, or may load as part of admission handled elsewhere — see
`HeavyModelAdmission.swift`), but it means enforcement can UNDER-refuse right
after a model swap: the first render on a newly-loaded family pays for that
family's weights landing in memory (tens of GB) in addition to the
activations this estimate covers, and `available_bytes` at that moment may
not yet reflect it. If you calibrate with `enforceMemoryEstimate` on, include
at least one "first render right after a model swap" sample in your test
spread — don't only calibrate against a machine that's been rendering the
same family for a while.

Aggressive mid-render cache clearing and cross-step RoPE-frequency-table
caching (the other two items from issue #22) are **not** part of this
pre-flight — they remain open follow-up work. `POST /v1/upscale` is also not
gated by this pre-flight yet — a separate follow-up.

**Deploy note:** the resolution cap is wired into the Krita/ComfyUI bridge
too (`POST /prompt`), including its inpaint/refine paths — a workflow whose
canvas exceeds `imageMemoryCaps.maxLongEdge`/`maxPixels` (4096px /
16,777,216px by default) now gets an HTTP 400 instead of running. This is
operator-tunable: raise `imageMemoryCaps.maxLongEdge`/`maxPixels` via
`PATCH /v1/config` if a real workflow needs a larger canvas than the
defaults allow (bounded to `maxLongEdge <= 16384`, `maxPixels <= 2²⁸`).

## Edit tab

Non-destructive tone, color, crop, local brush adjustments, and background removal for any PNG/JPEG/TIFF. Open from the sidebar (⌘U), from an asset's **Edit** button, or from the gallery context menu.

- Sliders preview live; one drag is one undo step (⌘Z / ⇧⌘Z).
- **Local** paints a mask; its sliders apply inside the mask only. **Feather** softens the edge. Paint the local mask *after* settling the crop — the mask is normalized to the post-crop frame, so changing the crop afterward rescales it and the painted region shifts relative to the picture.
- **Subject → Find Subject** runs Vision; **Remove Background** saves a transparent PNG. **Save** refuses (with an error) if Remove Background is on and Find Subject hasn't found a mask yet — run Find Subject first, or turn Remove Background off.
- **Save** writes `edit-<time>.png` into your output folder as a new asset. The recipe is stored in the adjacent `.json`, so opening the saved asset in Edit reopens the original pixels with your settings.
- **Save & Inpaint** saves, then opens the result in Inpaint with your local mask pre-painted. Note the shortcuts don't follow the tab order: Edit is ⌘U, Inpaint is ⌘E.
