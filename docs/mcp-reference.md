# ComfyBox MCP Tool Reference

ComfyBox exposes 18 tools via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). These tools let AI assistants generate images, manage models, swap LoRAs, and monitor server health — all through a standardized JSON-RPC 2.0 interface.

## Setup

### Local

```bash
ComfyBox mcp --port 7862
```

Reads JSON-RPC requests from stdin, writes responses to stdout. All logging goes to stderr. Runs until stdin closes.

### Claude Code Registration

```bash
claude mcp add comfybox -- ComfyBox mcp --port 7862
```

### Remote (SSH Bridge)

For servers managed by a remote daemon:

```bash
ssh user@mac-host "cd /path/to/comfybox && .build/release/ComfyBox mcp --port 7862"
```

The daemon spawns this command as a child process. Tools are registered as `mcp_comfybox__<tool_name>`.

## Tools

### Generation

#### `generate_image`

Generate an image from a text prompt. Supports text-to-image and img2img modes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | **Yes** | Text prompt describing the desired image |
| `negative_prompt` | string | No | Concepts to avoid (non-distilled models only) |
| `width` | integer | No | Width in pixels, divisible by 16 (default: 1024) |
| `height` | integer | No | Height in pixels, divisible by 16 (default: 1024) |
| `steps` | integer | No | Denoising steps. Distilled: 4-9, Base: 20-50 |
| `guidance` | number | No | CFG scale. 0.0 for distilled, 3.5-7.0 for base |
| `seed` | integer | No | Random seed for reproducibility |
| `output_path` | string | No | Output file path (must be in allowed directory) |
| `scheduler` | string | No | Sampler: euler, heun, res_2s, ddim, dpmpp_2m |
| `sigma_schedule` | string | No | Schedule: flow, beta, beta57, linear |
| `image_path` | string | No | Source image for img2img mode |
| `image_strength` | number | No | Img2img strength 0.0-1.0 (default: 0.7) |

**Returns:** Output file path and render duration.

**Example:**
```json
{
  "prompt": "A red rose in a field of daisies, golden hour lighting",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "seed": 42
}
```

**Img2img example:**
```json
{
  "prompt": "Oil painting style, vibrant colors",
  "image_path": "/path/to/source.jpg",
  "image_strength": 0.5,
  "steps": 9
}
```

---

### LoRA Management

#### `swap_loras`

Hot-swap active LoRA weights on the loaded model. Replaces all currently active LoRAs. Pass an empty array to remove all.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `loras` | array | **Yes** | Array of LoRA configurations |
| `loras[].path` | string | **Yes** | Path to .safetensors, or HuggingFace ID. Bare filenames resolve to `~/bin/zimage/loras/` |
| `loras[].scale` | number | No | Weight scale 0.0-2.0 (default: 1.0) |

**Example:**
```json
{
  "loras": [
    {"path": "/Users/me/loras/style.safetensors", "scale": 0.8},
    {"path": "ostris/z_image_turbo_childrens_drawings", "scale": 1.0}
  ]
}
```

> **Important:** Use full absolute paths for local files. Bare filenames (e.g., `"nudeart6-e10.safetensors"`) resolve to `~/bin/zimage/loras/`.

#### `list_loras`

List LoRA files in the LoRA directory (`~/bin/zimage/loras/`). Returns filenames that can be passed to `swap_loras`.

*No parameters.*

#### `lora_library`

List all LoRAs in the library with compatibility info.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | No | Filter by model family (e.g., "z-image", "klein-9b") |
| `include_quarantined` | boolean | No | Include quarantined LoRAs (default: false) |

#### `lora_scan`

Scan the LoRA directory for new or changed files and update the library index.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `force` | boolean | No | Force full rescan of all files (default: false) |

#### `lora_quarantine`

Quarantine or un-quarantine a LoRA. Quarantined LoRAs are hidden from model loading.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | **Yes** | LoRA identifier from the library |
| `quarantine` | boolean | **Yes** | true to quarantine, false to restore |
| `reason` | string | No | Reason for quarantine action |

---

### Model Management

#### `load_model`

Load a model into the pool. Supports all families: Z-Image, Klein, FIBO, Chroma, SeedVR2.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | **Yes** | Model spec (e.g., "z-image-turbo-bf16", "klein-9b-q8", "briaai/FIBO") |
| `quantization` | string | No | Quantization level ("4bit", "8bit"). Omit for bf16 |
| `activate` | boolean | No | Activate after loading (default: true) |
| `wait` | boolean | No | Wait for load to complete (default: true) |

#### `switch_model`

Switch the active model to one already loaded in the pool. Instant — no loading required.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | **Yes** | Pool model ID to activate (from `model_pool` results) |

#### `model_pool`

List all models currently loaded in the pool with VRAM usage, active status, and last-used timestamps.

*No parameters.*

#### `unload_model`

Unload a model from the pool to free VRAM. Cannot unload the active model — switch first.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | **Yes** | Pool model ID to unload |

#### `list_models`

List all supported model families with capabilities, recommended parameters, and HuggingFace IDs.

*No parameters.*

---

### Styles

#### `list_styles`

List available style presets with prompt templates, parameters, and model pairings.

*No parameters.*

#### `apply_style`

Apply a style preset to a prompt. Returns enhanced prompt, negative prompt, and recommended generation parameters. Does not generate — use `generate_image` with the returned values.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `style_id` | string | **Yes** | Style preset ID (from `list_styles`) |
| `prompt` | string | **Yes** | Base prompt to enhance |
| `negative_prompt` | string | No | Negative prompt to merge with style's negative |

---

### Server Control

#### `server_health`

Get server health: loaded model, LoRA state, memory usage, render stats, queue depth.

*No parameters.*

#### `queue_status`

Get current generation queue: pending jobs, running job, and history.

*No parameters.*

#### `clear_queue`

Cancel all pending generation jobs. Does not affect the currently running job.

*No parameters.*

#### `system_stats`

Get hardware info: Apple Silicon device name, total memory, available VRAM.

*No parameters.*

#### `shutdown_server`

Gracefully shut down the WarmServer. In-flight renders complete before exit.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `confirm` | boolean | **Yes** | Must be true (safety guard) |

> **Warning:** Requires manual restart. Use sparingly.

## Timeout Configuration

GPU renders can take 30-300+ seconds depending on model and resolution. When bridging through a daemon, configure timeouts at both layers:

1. **MCP client timeout** — The inner `tools/call` JSON-RPC timeout (default: 60s)
2. **Tool executor timeout** — The daemon's outer tool timeout guard (default: 60s)

For GPU-heavy servers, set `toolTimeoutMs: 300000` (5 minutes) in the daemon's MCP server config. This propagates to both timeout layers.

## Error Handling

Tool calls return standard MCP results:

- **Success:** `{"content": [{"type": "text", "text": "..."}]}`
- **Error:** `{"content": [{"type": "text", "text": "error message"}], "isError": true}`

Common errors:
- `Model not found` — LoRA path doesn't exist (use absolute paths)
- `Output path not allowed` — File outside the server's allowed directory
- `No model loaded` — WarmServer started without a model, or model failed to load
- `VRAM exhausted` — Reduce resolution, use quantized model, or unload unused models
