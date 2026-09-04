# ComfyBox Server — API Reference

> **GENERATED FILE — do not edit by hand.** Regenerate with `comfybox docs generate`
> (run from the repo root). CI byte-compares this file against a fresh generation
> (`ControlSurfaceParityTests`), so a stale copy fails the build.
>
> Sources of truth (FDD-ui-api-parity §3.4): the dispatch switches in
> `Sources/ZImage/Server/WarmServer.swift` and
> `Sources/ZImage/Server/ComfyBridge/ComfyBridge.swift` (parsed), the compile-time
> `ControlRegistry`, `MCPToolRegistry`, and `ParityExemptions`.

The warm server (`comfybox serve`, default port **7870**) exposes an HTTP/JSON API.
`{id}` marks a path parameter. Mutating `v1` routes are each claimed by an MCP tool
or carry a reasoned exemption (§3.5 assertion 3).

> **See also: [`api-notes.md`](api-notes.md)** — the HAND-MAINTAINED companion with
> body schemas and operational guidance (LTX-2 video bodies, `POST /v1/enhance`,
> LoRA `role` semantics, Krea-2 preset kroma rules, startup imports). This file is
> regenerated wholesale; durable prose belongs there, never here.

## Warm-server routes (`surface: v1`)

| Method | Path | MCP tools | Exemption |
|---|---|---|---|
| GET | `/health` |  |  |
| GET | `/v1/audit-log` |  |  |
| GET | `/v1/characters` |  |  |
| POST | `/v1/characters` | create_character |  |
| PUT | `/v1/characters` | create_character |  |
| DELETE | `/v1/characters/{id}` | delete_character |  |
| GET | `/v1/characters/{id}` |  |  |
| POST | `/v1/civitai/harvest` | civitai_prompts |  |
| GET | `/v1/civitai/repo` |  |  |
| GET | `/v1/civitai/search` |  |  |
| GET | `/v1/config` | get_config, set_warm_preset |  |
| PATCH | `/v1/config` | patch_config |  |
| PUT | `/v1/config` | set_warm_preset, update_config |  |
| GET | `/v1/content-modes` |  |  |
| DELETE | `/v1/content-modes/{id}` |  | Reverts a mode to its built-in definition; same posture as the PUT — no agent caller yet, discoverable via GET /v1/controls. |
| PUT | `/v1/content-modes/{id}` |  | Phase 3 creative-layer write (Class E); no agent caller edits content modes yet. Discoverable via GET /v1/controls (creative.contentMode.* descriptors). |
| GET | `/v1/controls` |  |  |
| POST | `/v1/enhance` | enhance_prompt |  |
| GET | `/v1/gallery/file` |  |  |
| GET | `/v1/gallery/list` |  |  |
| POST | `/v1/generate` | generate_image, repair_image |  |
| POST | `/v1/generate/async` |  | Async job variant of /v1/generate; MCP agents call generate_image (a synchronous MCP call wrapping the same render). A dedicated async tool was deferred from the Phase 1 worklist. |
| GET | `/v1/generate/preview` |  |  |
| GET | `/v1/generate/status/{id}` |  |  |
| POST | `/v1/lora/swap` | swap_loras |  |
| GET | `/v1/loras` |  |  |
| POST | `/v1/loras/import` |  | Desktop drag-and-drop import (local file paths on the server host); agent flows discover LoRAs via lora_scan / nearline_stage instead. |
| POST | `/v1/loras/scan` | lora_scan |  |
| GET | `/v1/loras/{id}` |  |  |
| DELETE | `/v1/loras/{id}/quarantine` | lora_quarantine |  |
| POST | `/v1/loras/{id}/quarantine` | lora_quarantine |  |
| POST | `/v1/loras/{id}/update` | update_lora_triggerwords |  |
| GET | `/v1/memory` |  |  |
| POST | `/v1/model/activate` | set_warm_preset, switch_model |  |
| POST | `/v1/model/load` | load_model, set_warm_preset |  |
| GET | `/v1/model/pool` |  |  |
| POST | `/v1/model/unload` | unload_model |  |
| GET | `/v1/models` |  |  |
| POST | `/v1/montage/compose` | compose_montage |  |
| GET | `/v1/nearline` |  |  |
| POST | `/v1/nearline/evict` | nearline_evict |  |
| POST | `/v1/nearline/scan` | nearline_scan |  |
| POST | `/v1/nearline/stage` | nearline_stage |  |
| GET | `/v1/presets` |  |  |
| POST | `/v1/presets` | create_preset |  |
| PUT | `/v1/presets` | create_preset |  |
| POST | `/v1/presets/import-legacy` | import_legacy_presets |  |
| POST | `/v1/presets/resolve` |  | POST-for-body READ: resolves a preset against the loaded model without changing state. Not a mutation; list_presets covers agent reads. |
| DELETE | `/v1/presets/{id}` | delete_preset |  |
| GET | `/v1/presets/{id}` |  |  |
| GET | `/v1/providers/status` |  |  |
| GET | `/v1/queue` |  |  |
| POST | `/v1/queue/clear` |  | The clear_queue tool targets the ComfyUI-bridge queue path (POST /queue {"clear": true}, executeClearQueue) -- a pre-parity contract the old api-reference documented; the native /v1/queue/clear route currently has no agent caller. Declared reality (G1); re-point the tool in a behavior phase. |
| POST | `/v1/queue/interrupt` | interrupt_render |  |
| POST | `/v1/queue/pause` | pause_queue |  |
| POST | `/v1/queue/resume` | resume_queue |  |
| DELETE | `/v1/queue/{id}` | cancel_job |  |
| POST | `/v1/queue/{id}/move` | move_queue_job |  |
| POST | `/v1/shutdown` | shutdown_server |  |
| GET | `/v1/stats` |  |  |
| POST | `/v1/storyboard/render` | render_storyboard |  |
| GET | `/v1/styles` |  |  |
| POST | `/v1/upscale` | upscale |  |
| GET | `/v1/video/config/effective` |  |  |
| POST | `/v1/video/config/effective` |  | POST-for-body READ: echoes the effective video config for a hypothetical request without changing state (GET variant also exists). |
| POST | `/v1/video/extend` | extend_video |  |
| POST | `/v1/video/generate` |  | Synchronous variant; the generate_video tool proxies POST /v1/video/generate/async (job-based) so an agent is never blocked for a whole video render. |
| POST | `/v1/video/generate/async` | generate_video |  |
| GET | `/v1/video/output` |  |  |
| POST | `/v1/video/rerender` | rerender_video |  |
| GET | `/v1/video/status/{id}` |  |  |
| GET | `/v1/video/traces` |  |  |
| POST | `/v1/video/traces/{id}/promote` |  | Video-trace curation is Desktop/gallery-driven today; promote_video_trace tool deferred from the Phase 1 worklist. |
| POST | `/v1/video/traces/{id}/rating` |  | Video-trace curation is Desktop/gallery-driven today; rate_video_trace tool deferred from the Phase 1 worklist. |
| GET | `/v1/workflows` |  |  |
| POST | `/v1/workflows/import` | import_workflow |  |
| GET | `/v1/workflows/runs/{id}` |  |  |
| DELETE | `/v1/workflows/{id}` |  | Desktop workflow management; delete_workflow tool deferred from the Phase 1 worklist (no agent caller deletes workflows today). |
| GET | `/v1/workflows/{id}` |  |  |
| POST | `/v1/workflows/{id}/run` | run_workflow |  |

## ComfyUI compatibility bridge routes (`surface: comfyUICompat`)

Served by `ComfyBridge.route()` for ComfyUI/Krita clients, BEFORE the main switch.
Declared policy (§3.5): these need no MCP tool, but every one must be enumerated here
so adding one is visible in review. A tool listed here proxies the bridge path by
declared contract (e.g. `clear_queue`).

| Method | Path | MCP tools |
|---|---|---|
| GET | `/embeddings` |  |
| GET | `/experiment/models` |  |
| GET | `/extensions` |  |
| GET | `/history` |  |
| POST | `/interrupt` |  |
| GET | `/object_info` |  |
| GET | `/prompt` |  |
| POST | `/prompt` |  |
| GET | `/queue` |  |
| POST | `/queue` | clear_queue |
| GET | `/settings` |  |
| GET | `/system_stats` |  |
| POST | `/upload/image` |  |
| GET | `/userdata*` |  |
| GET | `/users` |  |
| GET | `/view` |  |
| GET | `/ws` |  |

## Controls (`GET /v1/controls`)

One call answers "what can I change and how" (§3.4). Each row is a
`ControlDescriptor`; `GET /v1/controls` returns these plus per-request resolved
`value`s. Writes with a pointer are JSON-pointer targets for the write route's
document (config writes: RFC 7386 merge patch via `PATCH /v1/config`).

| Id | Scope | Type | Range | Default | Write | MCP tool | Restart |
|---|---|---|---|---|---|---|---|
| `creative.contentMode.avocado.guidanceBoost` | creative | double | 0–10 | 2.5 | PUT `/v1/content-modes/avocado` @ `/guidanceBoost` |  |  |
| `creative.contentMode.avocado.negativePromptAdditions` | creative | object |  | [] | PUT `/v1/content-modes/avocado` @ `/negativePromptAdditions` |  |  |
| `creative.contentMode.avocado.promptHint` | creative | string |  | `explicit, uncensored, anatomically detailed` | PUT `/v1/content-modes/avocado` @ `/promptHint` |  |  |
| `creative.contentMode.avocado.styleVariant` | creative | enum |  | `nsfw` | PUT `/v1/content-modes/avocado` @ `/styleVariant` |  |  |
| `creative.contentMode.banana.guidanceBoost` | creative | double | 0–10 | 1.5 | PUT `/v1/content-modes/banana` @ `/guidanceBoost` |  |  |
| `creative.contentMode.banana.negativePromptAdditions` | creative | object |  | [] | PUT `/v1/content-modes/banana` @ `/negativePromptAdditions` |  |  |
| `creative.contentMode.banana.promptHint` | creative | string |  | `sensual, intimate, suggestive` | PUT `/v1/content-modes/banana` @ `/promptHint` |  |  |
| `creative.contentMode.banana.styleVariant` | creative | enum |  | `sensual` | PUT `/v1/content-modes/banana` @ `/styleVariant` |  |  |
| `creative.contentMode.neutral.guidanceBoost` | creative | double | 0–10 | 0 | PUT `/v1/content-modes/neutral` @ `/guidanceBoost` |  |  |
| `creative.contentMode.neutral.negativePromptAdditions` | creative | object |  | [] | PUT `/v1/content-modes/neutral` @ `/negativePromptAdditions` |  |  |
| `creative.contentMode.neutral.promptHint` | creative | string |  |  | PUT `/v1/content-modes/neutral` @ `/promptHint` |  |  |
| `creative.contentMode.neutral.styleVariant` | creative | enum |  | `neutral` | PUT `/v1/content-modes/neutral` @ `/styleVariant` |  |  |
| `creative.contentModeDefaultPresets` | creative | object |  |  | PATCH `/v1/config` @ `/contentModeDefaultPresets` | patch_config |  |
| `engine.allowedOutputDirectory` | engine | string |  |  | PATCH `/v1/config` @ `/allowedOutputDirectory` | patch_config |  |
| `engine.imageMemoryCaps.enforceMemoryEstimate` | engine | bool |  | false | PATCH `/v1/config` @ `/imageMemoryCaps/enforceMemoryEstimate` | patch_config |  |
| `engine.imageMemoryCaps.maxLongEdge` | engine | int |  | 4096 | PATCH `/v1/config` @ `/imageMemoryCaps/maxLongEdge` | patch_config |  |
| `engine.imageMemoryCaps.maxPixels` | engine | int |  | 16777216 | PATCH `/v1/config` @ `/imageMemoryCaps/maxPixels` | patch_config |  |
| `engine.imageMemoryCaps.minAvailableHeadroomFraction` | engine | double | 0–1 | 0.1 | PATCH `/v1/config` @ `/imageMemoryCaps/minAvailableHeadroomFraction` | patch_config |  |
| `engine.seedvr2WeightsPath` | engine | string |  |  | PATCH `/v1/config` @ `/seedvr2WeightsPath` | patch_config | yes |
| `model.krea2Models` | model | object |  |  | PATCH `/v1/config` @ `/krea2Models` | patch_config | yes |
| `model.spec` | model | string |  |  | PATCH `/v1/config` @ `/modelSpec` | patch_config | yes |
| `provider.captioning.apiKey` | provider | string |  |  | PATCH `/v1/config` @ `/providers/captioning/apiKey` | patch_config |  |
| `provider.captioning.baseUrl` | provider | string |  |  | PATCH `/v1/config` @ `/providers/captioning/baseUrl` | patch_config |  |
| `provider.captioning.model` | provider | string |  |  | PATCH `/v1/config` @ `/providers/captioning/model` | patch_config |  |
| `provider.promptOptimization.apiKey` | provider | string |  |  | PATCH `/v1/config` @ `/providers/promptOptimization/apiKey` | patch_config |  |
| `provider.promptOptimization.baseUrl` | provider | string |  |  | PATCH `/v1/config` @ `/providers/promptOptimization/baseUrl` | patch_config |  |
| `provider.promptOptimization.model` | provider | string |  |  | PATCH `/v1/config` @ `/providers/promptOptimization/model` | patch_config |  |
| `provider.replicate.apiKey` | provider | string |  |  | PATCH `/v1/config` @ `/replicate/apiKey` | patch_config |  |
| `provider.replicate.baseUrl` | provider | string |  |  | PATCH `/v1/config` @ `/replicate/baseUrl` | patch_config |  |
| `provider.replicate.imageModel` | provider | string |  |  | PATCH `/v1/config` @ `/replicate/imageModel` | patch_config |  |
| `provider.replicate.model` | provider | string |  |  | PATCH `/v1/config` @ `/replicate/model` | patch_config |  |
| `provider.replicate.videoModel` | provider | string |  |  | PATCH `/v1/config` @ `/replicate/videoModel` | patch_config |  |
| `provider.vision.apiKey` | provider | string |  |  | PATCH `/v1/config` @ `/providers/vision/apiKey` | patch_config |  |
| `provider.vision.baseUrl` | provider | string |  |  | PATCH `/v1/config` @ `/providers/vision/baseUrl` | patch_config |  |
| `provider.vision.model` | provider | string |  |  | PATCH `/v1/config` @ `/providers/vision/model` | patch_config |  |
| `queue.clear` | queue | action |  |  | POST `/v1/queue/clear` |  |  |
| `queue.interrupt` | queue | action |  |  | POST `/v1/queue/interrupt` | interrupt_render |  |
| `queue.pause` | queue | action |  |  | POST `/v1/queue/pause` | pause_queue |  |
| `queue.resume` | queue | action |  |  | POST `/v1/queue/resume` | resume_queue |  |
| `render.defaults.chroma.guidance` | engine | double |  | 0 | PATCH `/v1/config` @ `/renderDefaults/byFamily/chroma/guidance` | patch_config |  |
| `render.defaults.chroma.height` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/chroma/height` | patch_config |  |
| `render.defaults.chroma.steps` | engine | int |  | 28 | PATCH `/v1/config` @ `/renderDefaults/byFamily/chroma/steps` | patch_config |  |
| `render.defaults.chroma.width` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/chroma/width` | patch_config |  |
| `render.defaults.fibo.guidance` | engine | double |  | 4 | PATCH `/v1/config` @ `/renderDefaults/byFamily/fibo/guidance` | patch_config |  |
| `render.defaults.fibo.height` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/fibo/height` | patch_config |  |
| `render.defaults.fibo.steps` | engine | int |  | 30 | PATCH `/v1/config` @ `/renderDefaults/byFamily/fibo/steps` | patch_config |  |
| `render.defaults.fibo.width` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/fibo/width` | patch_config |  |
| `render.defaults.flux1.guidance` | engine | double |  | 0 | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux1/guidance` | patch_config |  |
| `render.defaults.flux1.height` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux1/height` | patch_config |  |
| `render.defaults.flux1.steps` | engine | int |  | 9 | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux1/steps` | patch_config |  |
| `render.defaults.flux1.width` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux1/width` | patch_config |  |
| `render.defaults.flux2.guidance` | engine | double |  |  | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux2/guidance` | patch_config |  |
| `render.defaults.flux2.height` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux2/height` | patch_config |  |
| `render.defaults.flux2.steps` | engine | int |  |  | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux2/steps` | patch_config |  |
| `render.defaults.flux2.width` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/flux2/width` | patch_config |  |
| `render.defaults.guidance` | engine | double |  |  | PATCH `/v1/config` @ `/renderDefaults/default/guidance` | patch_config |  |
| `render.defaults.height` | engine | int |  |  | PATCH `/v1/config` @ `/renderDefaults/default/height` | patch_config |  |
| `render.defaults.krea2.guidance` | engine | double |  |  | PATCH `/v1/config` @ `/renderDefaults/byFamily/krea2/guidance` | patch_config |  |
| `render.defaults.krea2.height` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/krea2/height` | patch_config |  |
| `render.defaults.krea2.steps` | engine | int |  |  | PATCH `/v1/config` @ `/renderDefaults/byFamily/krea2/steps` | patch_config |  |
| `render.defaults.krea2.width` | engine | int |  | 1024 | PATCH `/v1/config` @ `/renderDefaults/byFamily/krea2/width` | patch_config |  |
| `render.defaults.steps` | engine | int |  |  | PATCH `/v1/config` @ `/renderDefaults/default/steps` | patch_config |  |
| `render.defaults.width` | engine | int |  |  | PATCH `/v1/config` @ `/renderDefaults/default/width` | patch_config |  |
| `server.host` | engine | string |  | `127.0.0.1` | PATCH `/v1/config` @ `/host` | patch_config | yes |
| `server.port` | engine | int | 1–65535 | 7870 | PATCH `/v1/config` @ `/port` | patch_config | yes |
| `video.defaults.frames` | engine | int |  |  | PATCH `/v1/config` @ `/videoDefaults/default/frames` | patch_config |  |
| `video.defaults.height` | engine | int |  |  | PATCH `/v1/config` @ `/videoDefaults/default/height` | patch_config |  |
| `video.defaults.ltx2.frames` | engine | int |  | 97 | PATCH `/v1/config` @ `/videoDefaults/byFamily/ltx2/frames` | patch_config |  |
| `video.defaults.ltx2.height` | engine | int |  | 448 | PATCH `/v1/config` @ `/videoDefaults/byFamily/ltx2/height` | patch_config |  |
| `video.defaults.ltx2.width` | engine | int |  | 704 | PATCH `/v1/config` @ `/videoDefaults/byFamily/ltx2/width` | patch_config |  |
| `video.defaults.width` | engine | int |  |  | PATCH `/v1/config` @ `/videoDefaults/default/width` | patch_config |  |
