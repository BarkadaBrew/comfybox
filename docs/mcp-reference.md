# ComfyBox MCP Tool Reference

ComfyBox exposes 56 tools via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). These tools let AI assistants generate images and video, manage models, swap LoRAs, poll jobs, and monitor server health — all through a standardized JSON-RPC 2.0 interface.

This file documents the tools by hand. `docs/api-reference.md` is GENERATED and lists every HTTP route with the tool(s) claiming it — read that one for the route-level map.

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

### The bridge never starts a server (comfybox#153)

`ComfyBox mcp` is connect-only. At startup it checks `--port`: if anything is listening — healthy or not — it connects to that and reports what it found to stderr (worst case ~2.25s: a 250ms port probe plus a 2s `/health` fetch before it gives up and reports "no response"). It never spawns a `ComfyBox serve` process itself, under any flag or environment variable.

If nothing is listening, the bridge does **not** exit. launchd's `RunAtLoad` and the MCP host commonly race at login, so a bridge that starts before the engine must keep serving — it prints one warning to stderr (naming the port and the command below) and starts anyway. Every tool call made before the engine comes up fails with that same "nothing is listening" message (surfaced by `MCPToolExecutor`, not a generic connection error), so a client sees one consistent, actionable error until the engine answers — and the MCP host never sees the bridge itself as dead.

Lifecycle ownership of the engine belongs to launchd (`com.barkadabrew.comfybox`) alone. This is what fixes comfybox#153 (the bridge racing a manual server restart and colliding with it, or masking a freshly-rebuilt manual server behind one it started itself) — the bridge has no path that can start a second engine, so there is nothing left to race. If the port is free, start the managed engine with:

```bash
launchctl kickstart -k gui/$(id -u)/com.barkadabrew.comfybox
```

The port check is IPv4-only by design: `WarmServer` and the bridge's own `--host` both default to the literal `127.0.0.1`, never `localhost`, so there is no dual-stack case to handle.

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
| `async` | boolean | No | Submit and return a `job_id` immediately; poll with `get_job` (default: **false** — synchronous, unchanged) |
| `return_image` | boolean | No | Also return the PNG as MCP image content (default: **false** — see [Image results](#image-results)) |

**Returns (default, `async: false`):** Output file path and render duration — unchanged.

**Returns (`async: true`):**
```json
{"job_id": "3F2A…", "kind": "image", "state": "queued", "progress": 0, "poll_with": "get_job"}
```

Use `async: true` for anything slow or behind a busy queue. A blocking call
can outlive the MCP client's tool timeout (commonly 300 s) and lose a render
the engine actually produced — the reason this parameter exists (comfybox#288).

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

### Jobs

#### `get_job`

One polling tool for every render job — image, video, LoRA swap, storyboard
(comfybox#289). Before this there were three ways to ask "is it done yet?"
(`video_status`, `workflow_run_status`, and nothing at all for images).

`video_status` and `workflow_run_status` are **unchanged and still supported**;
`get_job` is additive. Workflow runs keep their own `run_id` namespace and
their own tool — `get_job` covers the `job_id` namespace the generation routes
share.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | string | **Yes** | Job id from an async submit |
| `kind` | string | No | `image` \| `video` \| `storyboard`. Omit to probe image then video |
| `return_image` | boolean | No | On a completed image job, also return the PNG as MCP image content (default: false) |

**Returns:**
```json
{"job_id": "3F2A…", "kind": "image", "state": "running", "progress": 42}
```
```json
{"job_id": "3F2A…", "kind": "image", "state": "completed", "progress": 100,
 "result": {"output_path": "/Users/…/render.png", "duration_ms": 41200}}
```

`result`, `error` and `retry_after_seconds` are **omitted** when they do not
apply — they are never present as `null`.

| Field | Notes |
|-------|-------|
| `state` | `queued` \| `running` \| `completed` \| `failed` \| `unknown`. Everything except `queued` and `running` is **terminal** — stop polling |
| `unknown` | The engine reported a state this build does not know (or none at all). Terminal on purpose: the alternative is a client polling forever. `error` names the raw value as `unmapped_state:<raw>` |
| `progress` | 0-100. Video reports its own percent; an image job's live percent comes from the engine's queue snapshot, and is 0 until the job is the active render |
| `result` | Present only when `state` is `completed`. Image: `output_path`, `duration_ms`. Video: plus `file_size_bytes`, `video_duration_seconds`, `frame_count` |
| `error` | Present when the job failed. An operator interrupt reports `failed` with a plain-English reason — never a state your client does not know |
| `retry_after_seconds` | The engine's own reschedule hint while it replays its persisted queue after a restart. The job is queued, not lost |

**Which id goes where**

| Submitted by | `kind` |
|---|---|
| `generate_image` with `async: true` | `image` |
| `generate_video`, `rerender_video`, `extend_video` | `video` |
| `render_storyboard` | `storyboard` |

Storyboards share the video tracker — pass `kind: "storyboard"` explicitly or
they are reported as `video`.

There is no `swap` kind. LoRA swaps are synchronous (`swap_loras` returns the
result), so no caller ever holds a swap job id to poll; the only swap ids that
exist come from queue replay after a restart, they live in the image tracker,
and they resolve as `image`.

**Polling:** every 2-5 s is plenty. A queued or finished job costs one HTTP
call; a *running image* job costs one extra (the queue snapshot that carries
the live percent).

---

### Progress notifications

If your MCP client sends a `progressToken` in the request `_meta` (MCP
utilities/progress), a synchronous `generate_image` call emits
`notifications/progress` roughly every 2 seconds while the render runs
(comfybox#292):

```json
{"jsonrpc":"2.0","method":"notifications/progress",
 "params":{"progressToken":"tok-1","progress":42,"total":100,
           "message":"rendering — 42% (1 queued behind it)"}}
```

- Sent **only** when a token was supplied. No token, no polling — the engine
  is never touched on behalf of a client that isn't listening.
- `progress` increases monotonically; a percent that hasn't moved emits
  nothing rather than repeating.
- Progress comes from `GET /v1/queue`, which is served from a lock-based
  snapshot and keeps answering while a render holds the coordinator actor
  (comfybox#217) — unlike `/health`.
- The poller is cancelled and awaited before the tool call returns, on success
  and on error. Nothing outlives the request.
- Narration stops after 30 minutes; the render itself still completes and
  still returns its result.

Clients that ignore progress notifications lose nothing.

---

### Image results

`generate_image` and `get_job` accept `return_image: true` to return the
finished PNG as an MCP `image` content block (base64, `image/png`) **next to**
the existing text/path result (comfybox#294). The text block stays first and
unchanged, so a client that only reads `content[0]` sees exactly what it saw
before.

**Payload sizes — the reason this is opt-in:**

| Render | PNG on disk | base64 in the JSON-RPC line |
|---|---|---|
| 1024x1024 turbo | ~1.3-2.0 MB | ~1.7-2.7 MB |
| 1024x1536 Krea 2 | ~2.0-3.0 MB | ~2.7-4.0 MB |
| 2048px upscale | ~6-12 MB | ~8-16 MB (over the cap) |

Every byte lands in the client's context. Rules:

- Default is **false**. Leave it off for batch work and for anything a human
  will open from the path.
- Cap is 8 MB **encoded**. It is checked against the file's size on disk
  before the bytes are read, so an oversized render costs a `stat`, not a
  12 MB read. Over the cap the image is omitted and the text result says so —
  `output_path` is always returned, and it is the durable artifact.
- An ignored `return_image` **always** explains itself in the text result:
  not finished yet, no output path, not an image, or over the cap. It is never
  a silent omission.
- At most **one** image per result: the engine renders one image per request
  (`POST /v1/generate` has no batch `count`).
- Video results are never inlined; use `output_path`.

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

Better still, do not race the timeout at all: submit with `generate_image
{"async": true}` and poll `get_job`. The render then survives a client timeout,
a client restart, and a queue backlog, because the engine owns the job and the
job id is durable. Add a `progressToken` if you want the wait narrated.

## Error Handling

Tool calls return standard MCP results:

- **Success:** `{"content": [{"type": "text", "text": "..."}]}`
- **Success with `return_image`:** `{"content": [{"type": "text", …}, {"type": "image", "data": "<base64>", "mimeType": "image/png"}]}`
- **Error:** `{"content": [{"type": "text", "text": "error message"}], "isError": true}`

Common errors:
- `Model not found` — LoRA path doesn't exist (use absolute paths)
- `Output path not allowed` — File outside the server's allowed directory
- `No model loaded` — WarmServer started without a model, or model failed to load
- `VRAM exhausted` — Reduce resolution, use quantized model, or unload unused models
