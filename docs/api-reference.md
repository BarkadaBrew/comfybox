# ComfyBox Server — API Reference

The warm server (`ComfyBox serve`, default port **7870**) exposes an HTTP/JSON
API. New-route responses are **snake_case**. This reference is generated from
the route switch in `Sources/ZImage/Server/WarmServer.swift`; keep it in sync
when adding endpoints. Endpoints marked **(MCP)** have a matching tool in the
ComfyBox MCP server (`Sources/ZImage/MCP/`).

## Health & status

| Method | Path | Purpose | MCP tool |
|---|---|---|---|
| GET | `/health` | Liveness + model/queue summary | `server_health` |
| GET | `/v1/stats` | Server + memory stats | `system_stats` |
| GET | `/v1/memory` | Memory-pressure detail | |
| GET | `/v1/config`, PUT `/v1/config` | Read / replace the server config document | |
| GET | `/v1/providers/status` | Configured AI-provider status | |

## Generation

| Method | Path | Purpose | MCP tool |
|---|---|---|---|
| POST | `/v1/generate` | Text-to-image (also ControlNet / img2img via fields) | `generate_image` |
| POST | `/v1/enhance` | Optimize a prompt via the configured provider (Dan's model); optional character injection | `enhance_prompt` |
| POST | `/v1/upscale` | Creative upscale (SeedVR2 / ESRGAN) | `upscale` |
| POST | `/v1/video/generate` | Image-to-video (Wan 2.2) | `generate_video` |
| GET | `/v1/video/status/{id}` | Video job status | `video_status` |

`POST /v1/enhance` body: `{prompt, character?, character_description?, content_mode?}`
→ `{success, prompt, enhanced, note?}`.

## Queue

| Method | Path | Purpose | MCP tool |
|---|---|---|---|
| GET | `/v1/queue` | Active job + pending jobs (each with an id) | `queue_list` |
| POST | `/v1/queue/interrupt` | Cancel the in-flight render | `interrupt_render` |
| POST | `/v1/queue/clear` | Cancel all pending jobs | `clear_queue` |
| DELETE | `/v1/queue/{id}` | Cancel one pending job | `cancel_job` |

## Models & LoRAs

| Method | Path | Purpose | MCP tool |
|---|---|---|---|
| GET | `/v1/models` | Known model specs | `list_models` |
| GET | `/v1/model/pool` | Loaded model pool | `model_pool` |
| POST | `/v1/model/load` | Load a model into the pool | `load_model` |
| POST | `/v1/model/activate` | Switch the active model | `switch_model` |
| POST | `/v1/model/unload` | Unload a pooled model | `unload_model` |
| GET | `/v1/loras` | LoRA library index | `lora_library` |
| POST | `/v1/loras/scan` | Rescan the LoRA library | `lora_scan` |
| POST/DELETE | `/v1/loras/{id}/quarantine` | Quarantine / unquarantine a LoRA | `lora_quarantine` |
| POST | `/v1/lora/swap` | Swap active LoRAs (auto-stages nearline items) | `swap_loras` |

## Nearline storage (attached disk, staged on demand)

| Method | Path | Purpose | MCP tool |
|---|---|---|---|
| GET | `/v1/nearline` | Catalog + staging state + budget | `nearline_list` |
| POST | `/v1/nearline/scan` | Rescan attached-storage roots | `nearline_scan` |
| POST | `/v1/nearline/stage` | Stage an item locally (`{name}`; LRU eviction) | `nearline_stage` |
| POST | `/v1/nearline/evict` | Remove a staged copy (`{name}`) | `nearline_evict` |

## Creative layer

| Method | Path | Purpose | MCP tool |
|---|---|---|---|
| GET | `/v1/characters` | Characters + scenes (bare array; `kind`, tiered descriptions, default LoRAs) | `list_characters` |
| GET | `/v1/characters/{id}`, POST `/v1/characters`, DELETE `/v1/characters/{id}` | Character CRUD | |
| GET | `/v1/presets` | Generation presets | `list_presets` |
| GET `/v1/presets/{id}`, POST `/v1/presets`, DELETE `/v1/presets/{id}` | | Preset CRUD | |
| POST | `/v1/presets/resolve` | Resolve a preset onto defaults | |
| POST | `/v1/presets/import-legacy` | Import presets from the old image service (idempotent) | `import_legacy_presets` |
| GET | `/v1/content-modes` | Content-mode definitions | `list_styles` (styles) |
| GET | `/v1/styles` | Style presets | `list_styles` |
| GET | `/v1/audit-log?limit=N` | Recent audit events | |

## Lifecycle

| Method | Path | Purpose | MCP tool |
|---|---|---|---|
| POST | `/v1/shutdown` | Graceful shutdown | `shutdown_server` |

## Notes

- Character + preset legacy imports also run **once at server startup**
  (idempotent), merging from `~/.coffeeshop/image-service/`.
- The MCP server (`MCPToolRegistry` / `MCPToolExecutor`) currently exposes 32
  tools; the `queue_status` / `clear_queue` tools target the ComfyUI-bridge
  queue path, while `queue_list` / `interrupt_render` / `cancel_job` target the
  native `/v1/queue`.
