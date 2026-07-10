# PRD: ComfyBox Studio Packs

**Date:** 2026-07-10  
**Status:** Draft  
**Repo:** BarkadaBrew/comfybox  
**Reference:** Local review of `/Users/toddwalderman/Projects/AIbstraction-Studio`

## 1. Problem Statement

ComfyBox already has a stronger native engine than AIbstraction-Studio: multi-model generation, LoRA hot swap, ControlNet, inpainting, SVG export, DAM/gallery, presets, MCP, and a Swift desktop app. What AIbstraction demonstrates well is an opinionated studio workflow: a user selects a domain-specific creative mode and the app automatically applies the right prompt grammar, negative prompt, LoRA stack, export settings, templates, and production flow.

Today ComfyBox exposes powerful primitives, but the user still has to compose many of those choices manually. For repeated production domains such as healthcare training illustrations, logo/icon design, character identity shoots, or Kira media campaigns, ComfyBox should provide packaged workflows that make the correct defaults obvious and reproducible.

## 2. Product Goal

Add a Studio Packs layer to ComfyBox: domain-specific creative packs that bundle generation settings, prompt templates, style rules, export defaults, QA/lint checks, and assistant/MCP actions into coherent production workflows.

The initial pack should be a **Life Design / Healthcare Training** pack inspired by AIbstraction's Life Design concept: faceless figures, flat shading, clean vector-friendly output, healthcare training scenarios, and SVG-first export.

## 3. Non-Goals

- Do not port AIbstraction's Python/FastAPI implementation.
- Do not duplicate existing ComfyBox primitives such as presets, prompt library, DAM, canvas, queue, or SVG export.
- Do not introduce a separate history database. Use existing DAM and sidecar metadata.
- Do not require cloud services.
- Do not block normal freeform generation behind Studio Packs.

## 4. Existing ComfyBox Assets To Reuse

- Saved prompts and prompt history: `PromptLibraryStore`, `PromptLibraryView`
- Server-backed presets: `PresetView`, `ServerPreset`
- Camera and lighting directives: `CameraDirective`, `LightingDirective`
- Canvas projects: `CanvasProject`, `CanvasView`
- DAM/gallery and sidecar ingestion: `DAMStore`, `AssetIngestor`, `SidecarService`
- Assistant surfaces: `AgentView`, `GenerateAssistantPanel`
- SVG export via vtracer: CLI `--svg --svg-preset`
- Queue, MCP, WarmServer, ControlNet, inpaint, img2img

## 5. User Stories

- As a training-content creator, I can choose a Life Design pack and produce consistent healthcare training illustrations without manually remembering style keywords, negative prompts, LoRAs, sizes, and SVG settings.
- As a designer, I can start from a structured template, fill in a subject or scenario, generate several variations, compare them, and export the selected PNG/SVG assets.
- As an AI assistant or MCP client, I can ask ComfyBox for a named studio workflow and get a complete generation recipe instead of assembling raw parameters manually.
- As Todd, I can keep ComfyBox's full-power freeform mode while also having repeatable production workflows for recurring domains.

## 6. Feature Requirements

### FR-1: StudioPack Schema And Built-In Life Design Pack

Create a first-class `StudioPack` model. A pack defines:

- id, name, description, domain, version
- default prompt suffix/prefix and negative prompt
- recommended model, steps, guidance, resolution presets
- recommended LoRA stack and scales
- SVG preset and export defaults
- camera and lighting defaults
- template categories
- QA/lint rules
- optional MCP/tool metadata

The initial bundled pack is `life-design-healthcare`.

Acceptance:

- ComfyBox Desktop can list available Studio Packs.
- Life Design pack ships with healthcare-oriented defaults and vector-friendly generation defaults.
- Applying a pack does not mutate user presets until explicitly saved.
- Pack data is serializable and can be loaded from bundled JSON plus user pack directories.

### FR-2: Pack-Aware Prompt And Template Composer

Upgrade prompt composition from raw text insertion to pack-aware templates with slots.

Example template:

```text
{procedure} training scene, {clinician_role} assisting {patient_role}, faceless figures, flat vector style, clean healthcare training illustration
```

Acceptance:

- User can select a pack, choose a template, fill slots, and send the composed recipe to Generate.
- Composer outputs prompt, negative prompt, size, SVG preset, camera/lighting, model, and LoRA hints where defined.
- Existing saved prompts still work unchanged.
- Prompt history can be promoted into a pack template or saved prompt.

### FR-3: Vector-First Generation Mode

Add a mode optimized for vector/logo/icon/flat illustration output.

Acceptance:

- Vector-first mode turns SVG export on by default.
- Mode applies pack negative prompt and vector-safe prompt rules.
- Generated result shows PNG and SVG links/previews together where SVG exists.
- Metadata records SVG preset, pack id, template id, and vector mode.
- Failure to generate SVG is surfaced as an export failure without hiding the successful PNG.

### FR-4: Batch Variation Board

Add a production surface for generating and comparing multiple seed/prompt variations from one recipe.

Acceptance:

- User can generate N variations from the same recipe.
- Fixed seed sweeps seed+1, seed+2, etc.; random seed produces N random seeds.
- Results display in a comparison grid with prompt, seed, duration, pack, and export status.
- Actions: promote winner, send to Generate, send to Canvas, save as preset, export selected assets.
- Uses existing batch runner, queue, DAM, and sidecar metadata.

### FR-5: Control Template Library

Add reusable control/composition templates that can feed Canvas, ControlNet, and Inpaint workflows.

Acceptance:

- Templates can be stored as prompt-only, control-image-only, or prompt+control-image entries.
- Life Design pack includes starter healthcare templates such as CPR, hospital bed, seated patient, clinical handoff, and medical equipment tutorial.
- User can apply a template to Generate or Canvas.
- Control template metadata records compatible control modes such as canny, depth, pose, or inpaint.

### FR-6: Assistant Task Cards

Upgrade assistant output from a plain prompt suggestion into structured generation cards.

Task card fields:

- prompt
- negative prompt
- Studio Pack
- template id
- model
- LoRA stack
- size
- steps/guidance/scheduler
- camera/lighting
- SVG/export options
- generate/apply action

Acceptance:

- Assistant can return a structured card and the Generate tab can apply it.
- Card preview clearly shows what will change before applying.
- Invalid or unavailable pack/model/LoRA values are flagged and not silently ignored.
- Existing `PROMPT:` text extraction remains as fallback.

### FR-7: Project Workspaces

Add project-level organization for production runs.

Acceptance:

- A project can hold a brief, active Studio Pack, prompts/templates used, generated assets, selected finals, canvas boards, and exported files.
- DAM assets can be assigned to a project without copying pixel files.
- Project view supports filtering assets by pack/template/status.
- Exports can include selected PNG/SVG plus a recipe manifest.

### FR-8: Pack QA And Style Lint

Add pack-specific quality checks that validate prompt/metadata initially and can later consume local vision tags.

Life Design checks:

- faceless/no facial features requested
- flat/vector-friendly terms present
- gradients/photorealistic terms discouraged
- SVG export requested for vector templates
- output has pack metadata

Acceptance:

- QA runs before generation as prompt lint and after generation as metadata/export lint.
- QA produces warnings, not hard failures, unless a pack marks a rule required.
- Rule output is visible in Desktop and available through API/MCP.
- Tests cover Life Design lint rules.

### FR-9: Studio Pack API And MCP Surface

Expose Studio Packs through WarmServer and MCP so assistants can use the same workflow as the desktop app.

Proposed endpoints/tools:

- `list_studio_packs`
- `get_studio_pack`
- `apply_studio_pack`
- `compose_from_template`
- `generate_variations`
- `run_pack_quality_check`

Acceptance:

- WarmServer can list packs and compose a recipe from pack/template/slot input.
- MCP tools expose the same pack list and template composition.
- API output is structured and can be fed directly into `generate_image`.
- Contract tests cover the JSON shape.

## 7. Build Sequence

1. FR-1 StudioPack schema + Life Design pack
2. FR-2 pack-aware prompt/template composer
3. FR-3 vector-first generation mode
4. FR-4 batch variation board
5. FR-6 assistant task cards
6. FR-5 control template library
7. FR-7 project workspaces
8. FR-8 QA/style lint
9. FR-9 API/MCP surface

The MVP is FR-1 through FR-4.

## 8. Open Questions

- Should Studio Packs live only in `~/.comfybox/studio-packs`, or should built-in packs be compiled resources with user overrides?
- Should packs be allowed to reference remote/HuggingFace LoRAs, or only local scanned LoRA library entries?
- Should Life Design require a dedicated LoRA before shipping, or ship with prompt/SVG defaults first?
- Should vector-first SVG preview render inside Desktop or only expose file links initially?
- Should project workspaces be stored in the existing DAM database or as separate JSON manifests?

## 9. Issue Mapping

- FR-1: StudioPack schema + built-in Life Design pack — https://github.com/BarkadaBrew/comfybox/issues/195
- FR-2: Pack-aware prompt/template composer — https://github.com/BarkadaBrew/comfybox/issues/198
- FR-3: Vector-first generation mode with SVG review — https://github.com/BarkadaBrew/comfybox/issues/196
- FR-4: Batch variation board — https://github.com/BarkadaBrew/comfybox/issues/202
- FR-5: Control template library — https://github.com/BarkadaBrew/comfybox/issues/200
- FR-6: Assistant task cards — https://github.com/BarkadaBrew/comfybox/issues/199
- FR-7: Project workspaces — https://github.com/BarkadaBrew/comfybox/issues/197
- FR-8: Studio Pack QA/style lint — https://github.com/BarkadaBrew/comfybox/issues/201
- FR-9: Studio Pack API/MCP surface — https://github.com/BarkadaBrew/comfybox/issues/203
