# Updated Krita Bridge Gap Analysis (2026-05-08)

Previous analysis: 2026-05-01. This update incorporates all changes merged since then.

## Completed Since Last Analysis

### New Model Pipelines
- **Chroma 8.9B integration** (PR #53, #55) — Full pipeline: T5-XXL text encoder, Approximator, FLUX.1 VAE, CFG support, unpadded tokenization, flash-heun scheduler. WarmServer auto-detects Chroma models and routes accordingly.
- **Klein 9B LoRA support** (PR #52) — LoRA adapters for Flux 2 Klein 9B models.
- **FIBO pixel-perfect parity** (PR #49) — Dynamic-length tokenization fixes, FlowMatchEulerDiscrete sigma schedule, full pipeline integration with WarmServer and CLI.

### Upscaling
- **ESRGAN/RRDBNet upscaler** (PR #72) — Native MLX Swift port of Real-ESRGAN. Full pipeline: `ESRGANConfig`, `ESRGANModel`, `ESRGANPipeline`, `ESRGANWeightLoader`. Supports 4x upscaling with standard ESRGAN `.pth` and `.safetensors` weights.
- **SeedVR2 bridge executor** (PR #51) — SeedVR2 upscale wired through the ComfyUI bridge. Krita can submit upscale workflows and receive results via WebSocket events.

### Sampling
- **RES 2s sampler** (PR #76) — Refined Exponential Solver (2-stage) with beta57 sigma schedule. Registered in `SchedulerFactory`, accessible via WarmServer `/v1/generate` API.

### Post-Processing
- **Image levels** (PR #70) — `ImageLevels.apply(image:min:max:)` for contrast adjustment. Integrated into ZImagePipeline, Img2ImgPipeline, ZImageControlPipeline, Flux2Pipeline, FiboPipeline. Exposed via WarmServer `levelsMin`/`levelsMax` parameters.

### Bridge Infrastructure
- **Model registry** (PR #47) — `ComfyBoxModelRegistry` with 10 models across 4 families (Z-Image, Flux 2 Klein, FIBO, SeedVR2), 15 style presets, WarmServer endpoints for model/style discovery.
- **Upscale workflow parser** (PR #50) — `ComfyBridgeWorkflowParser.parseWorkflow()` detects and routes upscale vs. generate workflows. SeedVR2 upscale fully wired through executor.
- **LoRA dynamic swap** — Bridge parses `LoraLoader` nodes from Krita workflows, resolves paths from `~/bin/zimage/loras/`, and swaps LoRAs before generation.
- **ControlNet routing** — Bridge parses `ZImageFunControlnet`/`ModelPatchLoader`/`ControlNetLoader` nodes, routes to `ZImageControlPipeline` with strength/start/end parameters.
- **Dynamic LoRA/ControlNet model discovery** — `ComfyBridgeObjectInfo` scans `~/bin/zimage/loras/` and `~/bin/zimage/controlnet/` at runtime for available models.
- **Phase 4 hardening** (PR #64) — README defaults, CLI parser hardening, SafeTensors reader validation, PNG validation.
- **CI and testing** (PR #63) — Warm server bound to loopback, CI test suite, test resource fixes.

---

## Remaining Gaps

### P0 — Blocks basic Krita connection or txt2img

| # | Gap | What Krita Expects | Current ComfyBox State | Effort |
|---|-----|--------------------|-----------------------|--------|
| 1 | **Missing `ImageCrop` in ObjectInfo** | Krita sends `ImageCrop` nodes in inpaint/img2img workflows. The parser already handles it, but the node is not declared in `/object_info` — Krita may warn about missing nodes. | Parser handles `ImageCrop` for dimension extraction. ObjectInfo lacks the declaration. | Small |
| 2 | **Missing `SplitSigmas` in ObjectInfo** | Krita uses `SplitSigmas` to control effective denoise strength. Parser reads `step` from it. Not declared in ObjectInfo. | Parser extracts split step and computes denoise = 1.0 - (splitStep/totalSteps). Node not declared. | Small |
| 3 | **Missing `PreviewImage` in ObjectInfo** | Krita may use `PreviewImage` as output node (fallback in parser). Not declared. | Parser falls back to PreviewImage if no ETN_SaveImageCache. Not declared in ObjectInfo. | Small |
| 4 | **`res_2s` not in KSamplerSelect options** | If Krita sends `res_2s` as sampler_name it will fail validation. The scheduler exists internally but is not exposed in the KSamplerSelect option list. | `SchedulerFactory` supports `res_2s`. `KSamplerSelect` in ObjectInfo lists: euler, heun, dpmpp_2m, dpmpp_2s_ancestral, deis, ddim, uni_pc. | Small |
| 5 | **No sampler mapping in bridge generate path** | The bridge `bridgeGenerate()` does not extract or map the `sampler_name` from the KSamplerSelect node. All bridge generations use the pipeline default (euler). | `ComfyBridgeGenerateRequest` has no sampler field. `bridgeGenerate()` constructs `GeneratePayload` without specifying scheduler. | Medium |

### P1 — Required for Krita inpainting/img2img workflow

| # | Gap | What Krita Expects | Current ComfyBox State | Effort |
|---|-----|--------------------|-----------------------|--------|
| 6 | **No binary WebSocket preview images** | Krita expects `PREVIEW_IMAGE` binary frames (8-byte header: event_type=1, format=2, then PNG data) during denoising for live preview. | `ComfyBridgeExecutor` sends text progress events but never sends binary preview frames. `ComfyWebSocketManager.sendBinary()` exists but is unused. | Medium |
| 7 | **ESRGAN not wired through bridge** | Krita can submit `ImageUpscaleWithModel` workflows with ESRGAN models (e.g. `RealESRGAN_x4`, `4x-UltraSharp`). Bridge should route these to ESRGAN pipeline. | ESRGAN pipeline exists (`ESRGANPipeline`). ObjectInfo lists ESRGAN models in `UpscaleModelLoader`. But the upscale handler in WarmServer only routes to SeedVR2. | Medium |
| 8 | **Image levels not exposed through bridge** | No way for Krita to request levels adjustment. The `levelsMin`/`levelsMax` parameters exist in the direct WarmServer API but not in the ComfyUI workflow parser. | `ImageLevels` integrated in all pipelines. `GeneratePayload` supports `levelsMin`/`levelsMax`. Bridge `ComfyBridgeGenerateRequest` has no levels fields. | Small |
| 9 | **Sigma schedule not extracted from workflow** | Krita's BasicScheduler has a `scheduler` field (normal, karras, exponential, etc.). The workflow parser ignores this field — all bridge generations use the default sigma schedule. | `BasicScheduler` is declared in ObjectInfo with schedule options. Parser reads `steps` and `denoise` from it but ignores the `scheduler` field. | Small |
| 10 | **ETN_Translate not implemented** | Krita sends translate requests for non-English prompts. Node is declared in ObjectInfo but no implementation exists. | ObjectInfo declares ETN_Translate with text/source_language/target_language inputs. No actual translation handler — requests would silently fail or error. | Medium |
| 11 | **Chroma not in model registry** | Krita cannot discover or select Chroma models via the bridge. | Chroma pipeline is fully integrated in WarmServer with auto-detection. `ComfyBoxModelRegistry` does not include Chroma models. `/api/etn/model_info/diffusion_models` won't list Chroma. | Small |
| 12 | **Missing `KSampler`/`KSamplerAdvanced` in ObjectInfo** | Some Krita workflows use `KSampler` or `KSamplerAdvanced` instead of `SamplerCustomAdvanced`. Parser checks for these but ObjectInfo doesn't declare them. | Parser reads `denoise` from `KSampler`/`KSamplerAdvanced` as fallback. Not declared in ObjectInfo. | Small |

### P2 — Nice-to-have for full Krita feature parity

| # | Gap | What Krita Expects | Current ComfyBox State | Effort |
|---|-----|--------------------|-----------------------|--------|
| 13 | **No live painting mode** | Krita AI Diffusion has a "Live" mode that sends rapid img2img requests at low resolution/steps. Bridge lacks optimized path for this. | Style preset `live-paint` exists (flux2-klein-4b, 4 steps, 512x512, strength 0.5) but is not auto-applied. No fast-path for rapid requests. | Medium |
| 14 | **No model hot-swap via bridge** | Krita should be able to switch between models (e.g. Z-Image Turbo vs Klein 4B) without restarting the server. | WarmServer loads one model at startup. No endpoint to switch models at runtime via the bridge. Model registry knows about all models but loading is static. | Large |
| 15 | **No ControlNet preprocessors** | `DepthAnythingV2Preprocessor` and `InpaintPreprocessor` are declared in ObjectInfo but have no implementation. Krita expects these to generate depth maps / preprocessed images server-side. | Nodes declared to satisfy connection validation. No actual preprocessing — Krita would need to send pre-processed control images. | Large |
| 16 | **IPAdapter not implemented** | `IPAdapterModelLoader` and `IPAdapter` nodes are declared for connection validation but not functional. | Declared in ObjectInfo to prevent "missing nodes" errors. No IPAdapter pipeline exists. | Large |
| 17 | **No queue management** | Krita sends `POST /queue` to cancel/clear jobs. `GET /queue` returns running status. Both are stubbed. | `GET /queue` returns basic running/pending status. `POST /queue` returns 200 with empty body. No actual queue with multiple pending jobs. | Small |
| 18 | **Style presets not accessible from Krita** | 15 style presets exist but Krita has no native way to select them. | `ComfyBoxStylePresets` with full preset registry. WarmServer has `/v1/styles` endpoint. No mechanism to inject preset into Krita workflow. | Medium |
| 19 | **No `/api/etn/model_info` enrichment for ESRGAN** | The model info endpoint returns metadata for diffusion models and SeedVR2, but ESRGAN models listed in `UpscaleModelLoader` options don't appear in the `upscale_models` folder response. | `ComfyBoxModelRegistry.bridgeModelInfo(for: "upscale_models")` only returns SeedVR2. ESRGAN models are listed in ObjectInfo but have no model_info metadata. | Small |

---

## New Capabilities to Expose

### ESRGAN Upscaling Through Bridge
The `ESRGANPipeline` is fully implemented (PR #72) but not connected to the bridge upscale handler. Currently the bridge's `upscaleHandler` closure in WarmServer only routes to SeedVR2. The fix requires:
1. Model name routing in `WarmServer.bridgeUpscale()` — if model name matches ESRGAN patterns (`RealESRGAN_x4`, `4x-UltraSharp`, `OmniSR_*`, `4x_NMKD-*`), create/use an `ESRGANPipeline` instead of `SeedVR2Pipeline`.
2. Add ESRGAN entries to `ComfyBoxModelRegistry.bridgeModelInfo(for: "upscale_models")`.
3. Lazy-load ESRGAN pipeline similar to how SeedVR2 is lazy-loaded.

**Effort:** Medium. Pipeline exists, needs routing + lazy loading.

### RES 2s Sampler Option
The scheduler is implemented and registered in `SchedulerFactory`. To expose via bridge:
1. Add `"res_2s"` to `KSamplerSelect` sampler_name options in ObjectInfo.
2. Add a `sampler` field to `ComfyBridgeGenerateRequest`.
3. Parse `KSamplerSelect.sampler_name` in `ComfyBridgeWorkflowParser`.
4. Map sampler name to `SchedulerKind` in `bridgeGenerate()` and pass to `GeneratePayload`.

**Effort:** Small-Medium. All internal machinery exists.

### Image Levels Post-Processing
Integrated in all pipelines via `levelsMin`/`levelsMax`. To expose via bridge:
1. Add `levelsMin`/`levelsMax` fields to `ComfyBridgeGenerateRequest`.
2. This is tricky because there is no standard ComfyUI node for levels. Options:
   - Custom node declaration in ObjectInfo (e.g. `ComfyBoxImageLevels`)
   - Parse from a generic post-processing node
   - Apply automatically based on a style preset flag

**Effort:** Small for the plumbing. Medium if a custom Krita plugin extension is needed.

### LoRA Browsing/Selection from Krita
LoRA discovery already works: `ComfyBridgeObjectInfo.zimageLoraModels()` dynamically scans `~/bin/zimage/loras/` and returns filenames in the `LoraLoader.lora_name` options. Krita sees these during connection and can include `LoraLoader` nodes in workflows. The bridge parses and applies them. This is **already functional** for Krita users who know how to add LoraLoader nodes.

What's missing:
1. LoRA metadata in `/api/etn/model_info/loras` — Krita may query this for display names, base model compatibility, preview images.
2. Scale/strength UI — Krita passes `strength_model` from the LoraLoader node, which the bridge already extracts.

**Effort:** Small for metadata endpoint.

### Chroma Model Exposure
Chroma 8.9B is integrated in WarmServer with auto-detection but invisible to Krita:
1. Add Chroma model entry to `ComfyBoxModelRegistry.allModels`.
2. Add Chroma text encoder models to `zimageClipModels()` or create a new loader type.
3. Ensure bridge generate path handles Chroma-specific defaults (28 steps, guidance 0.0).

**Effort:** Small. The bridge generate path already routes through the family-aware switch that handles `.chroma`.

---

## Recommended Next Steps

### Phase 1: ObjectInfo completeness (P0, ~1 day)
Fix gaps #1-5 to ensure Krita connects cleanly and bridge respects sampler selection.

1. **Add missing nodes to ObjectInfo:** `ImageCrop`, `SplitSigmas`, `PreviewImage`, `KSampler`, `KSamplerAdvanced`, `EmptyLatentImage` (parser uses both `EmptySD3LatentImage` and `EmptyLatentImage`).
2. **Add `res_2s` to KSamplerSelect options.**
3. **Add sampler field to `ComfyBridgeGenerateRequest`** — parse from `KSamplerSelect.sampler_name`, map to `SchedulerKind`, pass through `GeneratePayload.scheduler`.
4. **Add sigma schedule extraction** — parse `BasicScheduler.scheduler` field, map to `SigmaScheduleKind`.

### Phase 2: Upscale model routing (P1, ~2 days)
Fix gap #7 and expose ESRGAN through the bridge.

1. **Refactor upscale handler** to accept a model discriminator: if model name matches ESRGAN patterns, route to `ESRGANPipeline`; if `seedvr2-*`, route to `SeedVR2Pipeline`.
2. **Add ESRGAN to model info endpoint** — register ESRGAN models in `ComfyBoxModelRegistry.bridgeModelInfo(for: "upscale_models")`.
3. **Lazy-load ESRGAN pipeline** in WarmServer, mirroring the SeedVR2 pattern.

### Phase 3: Preview images + levels (P1, ~2 days)
Fix gaps #6 and #8.

1. **Implement binary preview frames** — during denoising, periodically decode the current latent, downscale to ~256px, encode as PNG, and send as a binary WebSocket frame with the 8-byte ComfyUI header (type=1, format=2).
2. **Wire image levels through bridge** — add `levelsMin`/`levelsMax` to `ComfyBridgeGenerateRequest`, pass through to `GeneratePayload`.

### Phase 4: Model registry + discovery (P1-P2, ~1 day)
Fix gaps #11, #19.

1. **Add Chroma to `ComfyBoxModelRegistry`** — one model entry with appropriate metadata.
2. **Add ESRGAN entries to model info** — proper metadata for each ESRGAN weight file.
3. **Add LoRA metadata endpoint** — `/api/etn/model_info/loras` returning available LoRAs with base model compatibility flags.

### Phase 5: Advanced features (P2, ongoing)
Gaps #13-16.

1. **Live painting fast-path** — detect rapid sequential requests and auto-apply live-paint preset (lower resolution, fewer steps, skip LoRA swap).
2. **Runtime model swap** — new endpoint or workflow node to switch loaded model without server restart.
3. **ControlNet preprocessors** — DepthAnythingV2 and InpaintPreprocessor are large scope items requiring CoreML or MLX vision model ports.
4. **IPAdapter** — requires porting the IPAdapter architecture to MLX. Largest remaining gap.

---

## Summary

Since the May 1 analysis, the bridge has gained significant capabilities: SeedVR2 upscale execution, ESRGAN pipeline (not yet wired), RES 2s sampler, image levels, Chroma integration, LoRA/ControlNet routing, and a full model registry with style presets. The critical P0 gaps are all small fixes in ObjectInfo declarations and adding sampler/schedule fields to the workflow parser. The most impactful P1 work is wiring ESRGAN through the bridge and implementing binary WebSocket preview images for Krita's live denoising preview.
