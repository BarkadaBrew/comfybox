# Krita AI Diffusion UI -> ComfyBox Mapping

**Date:** 2026-05-08
**Plugin version:** Krita AI Diffusion 1.50.0
**Source:** 5 screenshots of the Krita AI Diffusion settings dialog (Todd's Mac, M3 Max 128GB)
**Bridge:** ComfyBox (ZImageCLI ComfyBridge, port 7861)

---

## Screenshot Inventory

| # | Panel Title | What It Shows |
|---|-------------|---------------|
| 1 | Plugin Information and Updates | Version info, update controls, diagnostics, support links |
| 2 | Performance Settings | History size, GPU preset, batch size, resolution, caching, threading |
| 3 | Interface Settings | Prompt UI, tag completion, layer behavior, image format, workflow dump |
| 4 | Diffusion Settings | Selection feather/blend/padding, color match, NSFW filter |
| 5 | Style Presets | Checkpoint, LoRA, style/negative prompts, sampler presets, linked edit style |

---

## Panel 1: Plugin Information and Updates

This panel is client-side only -- version management, diagnostics export, and links to docs/support. No server interaction except the "Check for Updates" button (hits Acly's GitHub, not our server).

| Krita UI Field | Type | Values/Range | ComfyBox Equivalent | Status |
|---|---|---|---|---|
| Current version | Label | 1.50.0 | N/A (client-side) | N/A |
| Latest version | Label | 1.50.0 | N/A (client-side) | N/A |
| Check for updates on startup | Checkbox | on/off (checked) | N/A (client-side) | N/A |
| Check for Updates | Button | -- | N/A (client-side) | N/A |
| Download and Install | Button | -- | N/A (client-side) | N/A |
| Collect Diagnostics | Button | -- | N/A (client-side) | N/A |
| View log files | Link | -- | N/A (client-side) | N/A |
| Restore Defaults | Button | -- | N/A (client-side) | N/A |
| Open Settings folder | Button | -- | N/A (client-side) | N/A |

**ComfyBox impact: None.** This panel is entirely client-side. No server API calls.

---

## Panel 2: Performance Settings

Client-side performance tuning. The **Performance Preset** and **Maximum Batch Size** are the only fields with indirect server relevance -- they control how many images Krita sends per prompt and what resolution scaling is applied before submission.

| Krita UI Field | Type | Values/Range | ComfyBox Equivalent | Status |
|---|---|---|---|---|
| Active History Size | Spinner (MB) | 1000 MB (currently 0.0 MB in use) | N/A (client-side RAM cache) | N/A |
| Stored History Size | Spinner (MB) | 20 MB (currently 0.0 MB on disk) | N/A (client-side .kra storage) | N/A |
| Performance Preset | Dropdown | GPU high (more than 12GB), GPU medium, GPU low, CPU | `/system_stats` device detection | ✅ Supported -- bridge reports MPS device + VRAM via `/system_stats`, Krita auto-selects preset |
| Device display | Label | [MPS] Apple M3 Max (128 GB) | `/system_stats` response `device.name`, `device.type`, `device.vram_total` | ✅ Supported |
| Maximum Batch Size | Slider | 1-? (shown: 6) | `RepeatLatentBatch` node in workflow; `batch_size` in `EmptySD3LatentImage` | ⚠️ Partial -- `EmptySD3LatentImage` batch_size=1 works. `RepeatLatentBatch` node not parsed. Batch >1 silently generates only 1 image |
| Resolution Multiplier | Slider | shown: 1.0x | Client-side -- scales canvas before sending to server | N/A (client-side scaling) |
| Maximum Pixel Count | Spinner (MP) | 8 MP (FullHD~2MP, 4K~8MP) | Client-side -- caps resolution before sending | N/A (client-side) |
| Tiled VAE | Toggle | Automatic / on / off | `VAEEncodeTiled` / `VAEDecodeTiled` nodes | ❌ Missing -- bridge does not declare or handle tiled VAE nodes. Not critical for MLX (Metal manages memory differently) but Krita may send these nodes when Tiled VAE is On |
| Dynamic Caching | Toggle | Off / On (shown: Off) | First Block Cache (FBC) on server | ❌ Missing -- no FBC implementation in bridge. This is a server-side optimization that Krita expects to toggle |
| Multi-Threading | Toggle | Off / On (shown: On) | N/A (client-side) | N/A |

### Key gaps

1. **Batch size >1**: When `Maximum Batch Size` is set above 1, Krita sends `RepeatLatentBatch` or sets `batch_size` >1 on `EmptySD3LatentImage`. The bridge ignores `RepeatLatentBatch` and only reads batch_size=1 from the empty latent node. Users setting batch size in performance settings will not get multiple images per generation.

2. **Tiled VAE**: If Krita is configured with Tiled VAE = On (vs Automatic), it may emit `VAEEncodeTiled` / `VAEDecodeTiled` instead of `VAEEncode` / `VAEDecode`. These nodes are not declared or handled. Practical impact is low on MLX but could cause workflow parse failures.

3. **Dynamic Caching (First Block Cache)**: This is a genuine performance feature in ComfyUI backends. ComfyBox does not implement it. Krita shows the toggle and may query the server about it. Low priority since MLX has its own memory management, but it is a visible gap.

---

## Panel 3: Interface Settings

Mostly client-side UI preferences. The notable server-relevant fields are **Tag Auto-Completion** (implies the server could provide tag lists), **Dump Workflow** (exports the ComfyUI JSON Krita sends to the server), and **Save Image Metadata** (PNG metadata embedding).

| Krita UI Field | Type | Values/Range | ComfyBox Equivalent | Status |
|---|---|---|---|---|
| Language | Dropdown | English (shown) | `/api/etn/languages` endpoint | ✅ Supported -- bridge returns empty `[]` (no translation engine), Krita defaults to English |
| Prompt Translation | Dropdown | Disabled (shown) | `/api/etn/translate/<lang>/<text>` endpoint | ⚠️ Partial -- endpoint not implemented. Returns would 404. Krita shows "Disabled" which means no server calls are made |
| Prompt Line Count | Spinner | 10 (shown) | N/A (client-side text editor size) | N/A |
| Negative Prompt | Toggle | Show / Hide (shown: Hide) | `CFGGuider` node with negative conditioning | ✅ Supported -- bridge parses `CFGGuider` nodes with positive + negative + cfg values |
| Show Steps | Toggle | Off / On (shown: Off) | N/A (client-side UI) | N/A |
| Recent Styles | Spinner | 4 (shown) | N/A (client-side MRU list) | N/A |
| Tag Auto-Completion | Multi-checkbox + toggle | Disabled / enabled per source: Danbooru, Danbooru NSFW, e621, e621 NSFW | No server-side tag database | ❌ Missing -- Krita expects tag CSV files for auto-completion. These are client-side files but could be served. Low priority since Z-Image prompts are natural language, not tag-based |
| Finished Generation | Dropdown | Preview (shown), likely also: Apply, Discard | N/A (client-side behavior) | N/A |
| Apply Behavior | Dropdown (x2) | "New layer on top" / "Layer group" (shown) | N/A (client-side Krita layer handling) | N/A |
| Apply Behavior (Live) | Dropdown (x2) | "Modify active layer" / "Modify region layers" (shown) | N/A (client-side) | N/A |
| Live: New Seed after Apply | Toggle | Off / On (shown: Off) | N/A (client-side seed behavior) | N/A |
| Save Image Format | Dropdown | PNG (shown), likely also JPEG, WebP | N/A (client-side save format) | N/A |
| Save Image Metadata | Toggle | Off / On (shown: Off) | Metadata embedded in PNG output | ⚠️ Partial -- bridge sends raw image bytes. Metadata embedding (prompt, seed, model, steps) in PNG chunks is not implemented. Krita can request it via this toggle |
| Dump Workflow | Toggle | Off / On (shown: On) | N/A (client-side debug logging) | N/A -- but extremely useful for debugging. When On, Krita writes the exact workflow JSON it sends to `POST /prompt` to a log file |

### Key gaps

1. **Tag auto-completion sources**: Krita supports Danbooru, Danbooru NSFW, e621, and e621 NSFW tag databases for prompt auto-completion. These are CSV files loaded client-side. Not server-relevant but could be served from ComfyBox's model/resource directory in the future. Low priority for Z-Image (natural language prompts).

2. **PNG metadata embedding**: When "Save Image Metadata" is enabled, Krita expects the returned PNG to contain generation metadata (prompt, seed, steps, model, etc.) in PNG text chunks (tEXt/iTXt). The bridge currently returns raw image bytes without metadata. This is a polish item.

---

## Panel 4: Diffusion Settings

Server-relevant settings that control how inpainting/selection-based generation works. These map to specific node parameters that Krita injects into workflows.

| Krita UI Field | Type | Values/Range | ComfyBox Equivalent | Status |
|---|---|---|---|---|
| Selection Feather | Slider (%) | 5% (shown), range likely 0-100% | `INPAINT_ExpandMask` node `feather` parameter | ✅ Supported -- bridge parses `grow` and `feather` from `INPAINT_ExpandMask` |
| Selection Blend | Slider (px) | 25 px (shown) | `INPAINT_MaskedBlur` and compositing blend area | ✅ Supported -- handled by inpaint pipeline |
| Selection Padding | Slider (%) | 7% (shown) | Crop padding around selection area | ✅ Supported -- `ImageCrop` parser extracts selection bounds with padding |
| Color Match | Toggle | On / Off (shown: On) | `INPAINT_ColorMatch` node | ✅ Supported -- bridge declares and handles `INPAINT_ColorMatch` |
| NSFW Filter | Dropdown | Disabled (shown), likely also: Soft, Strong | `ETN_NSFWFilter` node in workflow | ❌ Missing -- node not declared in `/object_info`, not parsed. If user enables NSFW filter, Krita would inject `ETN_NSFWFilter` node which would cause a parse error or be silently ignored |

### Key gaps

1. **NSFW Filter node**: If a user enables any NSFW filter level, Krita injects an `ETN_NSFWFilter` node into the workflow. The bridge does not declare this node in `/object_info` and does not handle it in the workflow parser. For our use case (disabled NSFW filter), this is a non-issue. But for completeness, the node should be declared as a pass-through in `/object_info`.

**Good news**: Inpainting settings (feather, blend, padding, color match) are fully supported. This panel is mostly green.

---

## Panel 5: Style Presets (CRITICAL -- highest integration surface)

This is the most important panel for ComfyBox integration. It defines how Krita presents models, LoRAs, samplers, and style configurations to the user, and directly maps to what the server must expose via `/object_info`.

| Krita UI Field | Type | Values/Range | ComfyBox Equivalent | Status |
|---|---|---|---|---|
| Style selector | Dropdown + toolbar | "New Style (default-8)" shown. Buttons: +, ..., delete, refresh, folder. Checkbox: "Show pre-installed styles" | Krita's local style database. Styles are CLIENT-SIDE presets that bundle checkpoint + LoRAs + prompts + sampler settings | ⚠️ Partial -- Krita manages styles locally. The server provides the ingredient lists (models, LoRAs, samplers) via `/object_info` but does not push style definitions. ComfyBox's `StylePreset` type exists in spec but is not served to Krita |
| Name | Text input | "New Style" (shown) | N/A (client-side style name) | N/A |
| Model Checkpoint | Dropdown + refresh | "z-image-turbo-bf16" (shown, with icon) | `/object_info` node `NunchakuZImageDiTLoader` or `CheckpointLoaderSimple` input `ckpt_name` options | ✅ Supported -- bridge populates model options from `/api/etn/model_info/diffusion_models`. Krita shows z-image-turbo-bf16 correctly |
| Checkpoint configuration (advanced) | Expandable section | collapsed | Likely exposes quantization, architecture details | ⚠️ Unknown -- need to expand to see fields. Bridge provides `quant` field in model_info |
| LoRA | Add/Upload buttons + filter | "Add" button, "Upload" button, filter dropdown "All", refresh button | `/object_info` node `LoraLoader` input `lora_name` options; `PUT /api/etn/upload/loras/<id>` for upload | ⚠️ Partial -- LoRA listing works (bridge scans ~/bin/zimage/loras/). LoRA ADD works (dropdown populated from server). LoRA UPLOAD (`PUT /api/etn/upload/loras/<id>`) is NOT implemented -- user cannot upload LoRAs from Krita to the server. Filter dropdown ("All") queries LoRA categories |
| Style Prompt | Text area | "best quality, highres" (shown). Note: "{prompt} placeholder can be used to wrap prompts" | `CLIPTextEncode` -- Krita prepends/appends this to the user's prompt before sending | ✅ Supported -- this is client-side prompt wrapping. The final combined prompt reaches the server via `CLIPTextEncode` node |
| Negative Prompt | Text area | "bad quality, low resolution, blurry" (shown) | `CFGGuider` negative conditioning path | ✅ Supported -- bridge parses `CFGGuider` with positive + negative text |
| Linked Edit Style | Dropdown | "None" (shown) | N/A (client-side -- selects alternate style for instruction-based editing mode) | N/A |
| Quality Preset (generate and upscale) | Dropdown | "Default - DPM++ 2M" (shown) | `KSamplerSelect` sampler_name + `BasicScheduler` scheduler + steps | ✅ Supported -- bridge exposes samplers and schedulers via `/object_info`. Available samplers: euler, heun, dpmpp_2m, dpmpp_2s_ancestral, deis, ddim, uni_pc, res_2s. Schedulers: normal, karras, exponential, beta, sgm_uniform |
| Performance Preset (live mode) | Dropdown | "Realtime - Hyper" (shown) | Same as above but optimized for speed (fewer steps, faster sampler) | ✅ Supported -- Krita uses same sampler/scheduler nodes with different step counts |

### How Krita Styles Work (Critical Architecture Understanding)

Krita's "Style Presets" are **client-side bundles** that combine:
1. A **model checkpoint** (selected from server's model list)
2. Zero or more **LoRAs** (selected from server's LoRA list, with per-LoRA strength)
3. A **style prompt** (text appended/wrapped around user prompt -- client-side)
4. A **negative prompt** (text for negative conditioning -- client-side)
5. A **quality preset** for generation (sampler name + scheduler + steps + CFG)
6. A **performance preset** for live mode (same structure, tuned for speed)
7. A **linked edit style** (optional alternate style for editing operations)

The **server's job** is to provide accurate ingredient lists via `/object_info`:
- Model checkpoint names (from `diffusion_models` folder scan)
- LoRA names (from `loras` folder scan)
- Sampler names (KSamplerSelect options)
- Scheduler names (BasicScheduler options)
- ControlNet model names

Krita then bundles these into styles locally. The server never sees "style" as a concept -- it only sees the resulting workflow JSON with specific models, LoRAs, and sampler settings already selected.

### Sampler Presets (Quality and Performance)

Krita's quality/performance preset dropdowns are **named bundles** of sampler + scheduler + steps + CFG. The visible ones:

**Quality Presets** (for generate and upscale):
| Preset Name | Sampler | Scheduler | Steps (est.) | CFG (est.) |
|---|---|---|---|---|
| Default - DPM++ 2M | dpmpp_2m | normal/karras | 20-30 | model-dependent |
| (others likely: Euler, DPM++ SDE, UniPC, DDIM) | varies | varies | varies | varies |

**Performance Presets** (for live mode):
| Preset Name | Sampler | Scheduler | Steps (est.) | CFG (est.) |
|---|---|---|---|---|
| Realtime - Hyper | euler/hyper | normal | 4-8 | 1-4 |
| (others likely: Realtime - LCM, Realtime - Lightning) | varies | varies | varies | varies |

These presets are defined in Krita's plugin code, not served by the server. The server must support all the samplers and schedulers these presets reference.

### Key gaps

1. **LoRA upload endpoint**: The "Upload" button in Krita's LoRA section calls `PUT /api/etn/upload/loras/<id>` with streaming file upload. Bridge does not implement this. Users must manually copy LoRA files to `~/bin/zimage/loras/` on the server.

2. **Pre-installed styles**: The "Show pre-installed styles" checkbox reveals Krita's built-in style definitions. These reference specific checkpoints by name (e.g., "sd_xl_base_1.0.safetensors", "flux1-schnell-Q4_K_S.gguf"). If any pre-installed style references a Z-Image model name that doesn't match what the bridge reports, the style will show as unavailable. Bridge must ensure model names in `/object_info` match what Krita pre-installed styles expect for Z-Image architecture.

3. **"Hyper" sampler/scheduler**: The "Realtime - Hyper" performance preset may reference a "hyper" scheduler or sampler variant not currently in the bridge's sampler list. Need to verify the bridge exposes all sampler/scheduler names that Krita's built-in presets expect.

---

## Cross-Panel Gap Summary

### Fully Supported (no work needed)

| Feature | Panel | Notes |
|---|---|---|
| Model checkpoint selection | Style Presets | z-image-turbo-bf16 visible, model_info pagination works |
| LoRA listing and loading | Style Presets | Dynamic scan of loras directory, multi-LoRA chaining |
| Sampler selection | Style Presets | 8 samplers exposed via `/object_info` |
| Scheduler selection | Style Presets | 5 sigma schedules exposed |
| Negative prompt (CFG guidance) | Style Presets + Interface | `CFGGuider` node fully parsed |
| Style prompt wrapping | Style Presets | Client-side -- server sees final prompt |
| Inpainting feather/blend/padding | Diffusion Settings | All `INPAINT_*` nodes handled |
| Color match | Diffusion Settings | `INPAINT_ColorMatch` node handled |
| Device detection | Performance Settings | `/system_stats` reports MPS + memory |
| Plugin version / updates | Plugin Info | Client-side, no server dependency |
| Language / prompt UI | Interface Settings | Client-side preferences |

### Partially Supported (functional but incomplete)

| Feature | Panel | Gap | Priority |
|---|---|---|---|
| Batch generation (batch_size >1) | Performance Settings | `RepeatLatentBatch` not parsed; only single-image output | **MEDIUM** -- users setting batch_size >1 get silent single-image fallback |
| LoRA upload from Krita | Style Presets | `PUT /api/etn/upload/loras/<id>` not implemented | **MEDIUM** -- blocks uploading user LoRAs from Krita UI |
| Prompt translation | Interface Settings | `/api/etn/translate` not implemented | **LOW** -- Krita shows "Disabled" by default |
| PNG metadata embedding | Interface Settings | Bridge returns raw image bytes without PNG metadata chunks | **LOW** -- cosmetic, not functional |
| Checkpoint advanced config | Style Presets | Unknown what fields are in the expandable section | **LOW** -- likely quant info we already provide |

### Missing (not implemented)

| Feature | Panel | Gap | Priority |
|---|---|---|---|
| Tiled VAE decode/encode | Performance Settings | `VAEEncodeTiled` / `VAEDecodeTiled` nodes not declared or handled | **LOW** -- MLX manages memory differently, but Krita may send these |
| Dynamic Caching (FBC) | Performance Settings | No First Block Cache implementation | **LOW** -- server-side optimization, not currently relevant for MLX |
| NSFW Filter node | Diffusion Settings | `ETN_NSFWFilter` not declared in `/object_info` | **LOW** -- our use case has it disabled, but should be pass-through for completeness |
| Tag auto-completion databases | Interface Settings | No Danbooru/e621 tag CSV serving | **VERY LOW** -- Z-Image uses natural language, not tags |
| "Hyper" sampler variant | Style Presets | May need to verify bridge exposes all sampler names Krita's live-mode presets expect | **MEDIUM** -- could break live painting mode |

---

## Implementation Priorities for ComfyBox

### P0 -- Required for feature completeness

1. **Batch generation support**: Parse `RepeatLatentBatch` node from workflow to support `batch_size > 1`. This is visible in the Performance Settings panel and users will expect it to work. Alternatively, the bridge could parse `batch_size` from `EmptySD3LatentImage` when >1.

2. **LoRA upload endpoint**: Implement `PUT /api/etn/upload/loras/<id>` with streaming upload. Users see the Upload button in the Style Presets LoRA section and will try to use it. Save to `~/bin/zimage/loras/` and invalidate `/object_info` cache.

### P1 -- Important for polish

3. **Verify "Hyper" and all live-mode sampler/scheduler names**: The "Realtime - Hyper" performance preset shown in the Style Presets panel may reference sampler/scheduler names not in the bridge's current list. Audit Krita's built-in preset definitions against our exposed sampler/scheduler lists.

4. **Declare `ETN_NSFWFilter` as pass-through**: Add to `/object_info` so workflows with NSFW filter don't fail to parse. Execute as no-op.

5. **Declare `VAEEncodeTiled` / `VAEDecodeTiled`**: Add to `/object_info` and alias to standard VAE encode/decode. Prevents parse failures when users enable Tiled VAE.

### P2 -- Nice to have

6. **PNG metadata embedding**: Embed generation parameters (prompt, seed, model, steps, sampler, scheduler, LoRA stack) in PNG tEXt chunks so Krita's "Save Image Metadata" toggle works.

7. **`/object_info` cache invalidation**: Currently cached forever after first build. Adding a LoRA or model at runtime requires server restart. Add file watcher or expose refresh endpoint.

---

## How Styles Map to ComfyBox's Architecture

Krita's style system maps cleanly to ComfyBox's `StylePreset` type from the product design spec:

```
Krita "Style Preset"          →  ComfyBox StylePreset
─────────────────────────────────────────────────────
Model Checkpoint              →  model family + variant
LoRAs (name + strength)       →  loras: [{ path, scale }]
Style Prompt                  →  promptSuffix
Negative Prompt               →  negativeSuffix
Quality Preset sampler        →  samplerOverride
Quality Preset CFG            →  guidanceOverride
```

The key insight: **Krita styles are client-side.** The server provides ingredients (models, LoRAs, samplers, schedulers via `/object_info`), and Krita bundles them locally. ComfyBox does NOT need to push style definitions to Krita -- Krita manages its own style library. ComfyBox's server-side `StylePreset` type is for the Orchestrator layer and Telegram/API users, not for Krita.

The server's contract with Krita is:
1. `/object_info` -- expose all available models, LoRAs, samplers, schedulers, nodes
2. `/api/etn/model_info/<folder>` -- expose model metadata (base_model, quant, is_inpaint)
3. `/system_stats` -- expose device capabilities
4. `POST /prompt` -- accept and execute workflow JSON with whatever style ingredients Krita bundled
5. WebSocket -- report progress and deliver results

As long as ComfyBox accurately reports its capabilities via these endpoints, Krita's style system works automatically.

---

## Questions to Resolve

1. **What's inside "Checkpoint configuration (advanced)"?** The expandable section in Style Presets is collapsed in the screenshot. It likely contains quantization selection (q4, q8, bf16), architecture type, and possibly inference mode. Need to expand this section to map those fields.

2. **Full list of Krita's built-in quality/performance presets?** We see "Default - DPM++ 2M" and "Realtime - Hyper" but need the full list to audit sampler/scheduler coverage. These are defined in Krita's plugin source code (`ai_diffusion/style.py` or similar).

3. **What sampler name does "Hyper" use?** The live-mode preset "Realtime - Hyper" likely uses HyperSD/Hyper scheduler -- need to check if this maps to an existing sampler name or requires a new one in the bridge's `/object_info`.

4. **Does Krita send tiled VAE nodes when "Automatic" is selected?** The Tiled VAE toggle is set to "Automatic". Need to verify whether Automatic mode checks server capabilities or just decides based on resolution threshold.

5. **Filter dropdown for LoRAs**: The LoRA section shows a filter dropdown set to "All". What categories does Krita expect? Does it read these from the server or from local metadata? Likely reads `base_model` from `/api/etn/model_info/loras`.
