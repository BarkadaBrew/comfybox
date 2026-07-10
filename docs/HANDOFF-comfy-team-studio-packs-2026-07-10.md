# ComfyBox Team Handoff: Studio Packs

**Date:** 2026-07-10  
**Source review:** `/Users/toddwalderman/Projects/AIbstraction-Studio`  
**Target repo:** `BarkadaBrew/comfybox`  
**PRD:** `docs/prd-comfybox-studio-packs.md`  
**Ticket index:** `docs/TICKETS-comfybox-studio-packs-2026-07-10.md`

## Summary

Codex reviewed the local `AIbstraction-Studio` repo and translated the useful product ideas into a ComfyBox-native feature set.

The conclusion: do not port AIbstraction's Python/FastAPI app. ComfyBox already has the stronger engine and most primitives: native Swift/MLX generation, Desktop app, DAM/gallery, presets, prompt history, canvas, queue, MCP, SVG export, assistant, ControlNet, and inpaint.

The missing product layer is **Studio Packs**: opinionated, domain-specific workflows that bundle prompt grammar, negative prompts, LoRA stack, model/settings defaults, SVG/export settings, camera/lighting defaults, templates, QA rules, and assistant/MCP actions into one repeatable production surface.

Initial pack: **Life Design / Healthcare Training**, inspired by AIbstraction's Life Design style: faceless figures, flat shading, clean vector-friendly output, healthcare training scenarios, and SVG-first export.

## Created Artifacts

- PRD: `docs/prd-comfybox-studio-packs.md`
- Ticket index: `docs/TICKETS-comfybox-studio-packs-2026-07-10.md`
- GitHub issues: #195-#203 in `BarkadaBrew/comfybox`

## GitHub Issues

MVP / foundation:

1. **#195** — Studio Packs: schema + built-in Life Design pack  
   https://github.com/BarkadaBrew/comfybox/issues/195
2. **#198** — Studio Packs: pack-aware prompt/template composer  
   https://github.com/BarkadaBrew/comfybox/issues/198
3. **#196** — Studio Packs: vector-first generation mode with SVG review  
   https://github.com/BarkadaBrew/comfybox/issues/196
4. **#202** — Studio Packs: batch variation board  
   https://github.com/BarkadaBrew/comfybox/issues/202

Follow-on:

5. **#199** — Studio Packs: assistant task cards  
   https://github.com/BarkadaBrew/comfybox/issues/199
6. **#200** — Studio Packs: control template library  
   https://github.com/BarkadaBrew/comfybox/issues/200
7. **#197** — Studio Packs: project workspaces  
   https://github.com/BarkadaBrew/comfybox/issues/197
8. **#201** — Studio Packs: pack QA and style lint  
   https://github.com/BarkadaBrew/comfybox/issues/201
9. **#203** — Studio Packs: API and MCP surface  
   https://github.com/BarkadaBrew/comfybox/issues/203

## Recommended Build Order

1. Build #195 first. Everything else depends on the `StudioPack` model and pack loading.
2. Build #198 next. The composer makes packs useful in Desktop.
3. Build #196 next. Vector-first mode makes the Life Design pack visibly valuable.
4. Build #202 next. Variation boards turn the workflow into a production tool.
5. Then #199 assistant task cards.
6. Then #200 control template library.
7. Then #197 project workspaces.
8. Then #201 QA/style lint.
9. Then #203 API/MCP once the local model and Desktop workflow are stable.

MVP is #195, #198, #196, and #202.

## Existing ComfyBox Pieces To Reuse

Do not rebuild these:

- Prompt library/history: `PromptLibraryStore`, `PromptLibraryView`
- Presets: `PresetView`, `ServerPreset`, `PresetManager`
- Camera/lighting: `CameraDirective`, `LightingDirective`, `ShotTemplateStore`
- Canvas: `CanvasProject`, `CanvasView`
- DAM/gallery: `DAMStore`, `DAMAsset`, `AssetIngestor`, `SidecarService`
- Assistant: `AgentView`, `GenerateAssistantPanel`
- SVG export: CLI `--svg --svg-preset`, vtracer integration
- Queue/server/MCP: WarmServer, MCP tools, queue endpoints
- Control/editing: ControlNet, inpaint, img2img

Studio Packs should orchestrate these primitives, not duplicate them.

## Product Shape

Studio Pack fields should cover:

- id, name, description, domain, version
- prompt prefix/suffix and negative prompt
- recommended model, steps, guidance, scheduler, resolution presets
- recommended LoRA stack and scales
- SVG preset and export defaults
- camera and lighting defaults
- template categories and slot schemas
- QA/lint rules
- optional API/MCP metadata

Life Design pack should start with prompt/SVG defaults even if no dedicated Life Design LoRA is available yet. If a LoRA exists locally, the pack can reference it as optional and surface missing-LoRA warnings instead of failing hard.

## Acceptance Themes

The feature set is successful when:

- A user can choose a pack and get coherent defaults without manually assembling settings.
- A user can fill a structured template and generate repeatable outputs.
- Vector-first workflows produce PNG plus SVG, and SVG failure does not hide a successful PNG.
- Batch variation board makes it easy to compare seeds and promote a winner.
- Pack metadata is written into sidecars/DAM so assets can be reconstructed later.
- Freeform generation remains available and unaffected.

## Open Decisions

Resolve these before or during #195:

- Built-in pack storage: compiled resources, bundled JSON files, or runtime-loaded defaults?
- User pack storage: likely `~/.comfybox/studio-packs`.
- Pack LoRA references: local scanned LoRA ids only, or allow remote/HuggingFace references?
- Life Design first release: prompt/SVG defaults only, or require a dedicated LoRA?
- Pack metadata: exact sidecar keys for `studio_pack_id`, `template_id`, `vector_mode`, and QA result.

Resolve before #197:

- Project workspaces: DAM database tables or separate JSON manifests?

Resolve before #203:

- API/MCP naming and contract shape, after Desktop workflow stabilizes.

## Implementation Guidance

Keep the first PR narrow:

- Define pack data structures.
- Load built-in + user packs.
- Add a minimal Desktop list/apply path.
- Add tests for decoding, defaults, missing optional LoRA behavior, and applying a pack without mutating saved presets.

Avoid making #195 also build the template composer or vector UI. Those are separate issues.

Use existing ComfyBox conventions:

- Prefer Swift value models and pure helpers where possible.
- Store user-visible structured data as JSON under `~/.comfybox`.
- Preserve sidecar metadata compatibility.
- Keep Desktop work tied to reusable model/services rather than view-only logic.

## Local Worktree Note

At handoff time, this repo already had unrelated dirty files in source code. Codex only added docs for this Studio Packs handoff/PRD/ticket work and did not modify the unrelated code changes.
