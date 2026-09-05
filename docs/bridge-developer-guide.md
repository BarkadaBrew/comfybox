# ComfyBox Bridge Developer Guide

Build integrations that connect any application to ComfyBox's image generation engine.

## Overview

ComfyBox exposes a **ComfyUI-compatible HTTP API** (the "bridge") that any application can connect to. If your app can speak HTTP, it can generate images through ComfyBox.

Three integration paths, from easiest to most powerful:

| Path | Effort | Best For |
|------|--------|----------|
| **REST API** | Hours | Scripts, bots, simple tools |
| **ComfyUI Protocol** | Days | Apps with existing ComfyUI support (Krita, etc.) |
| **MCP (Model Context Protocol)** | Days | AI assistants, LLM-powered tools |

## Architecture

```
┌─────────────────────────────┐
│  Your Application           │
│  (Krita, Inkscape, custom)  │
└──────────┬──────────────────┘
           │ HTTP / WebSocket
┌──────────▼──────────────────┐
│  ComfyBox Bridge (:7870)    │  ← ComfyUI-compatible protocol
│  ComfyBox REST API (:7862)  │  ← Native REST API
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│  MLX Engine (Metal GPU)     │
│  Models, LoRAs, VAE         │
└─────────────────────────────┘
```

## Path 1: REST API (Simplest)

The WarmServer exposes a clean REST API on port 7862.

### Generate an Image

```bash
curl -X POST http://localhost:7862/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A mountain landscape at sunset",
    "width": 1024,
    "height": 1024,
    "steps": 9,
    "seed": 42,
    "output_path": "/tmp/landscape.png"
  }'
```

**Response:**
```json
{
  "output_path": "/tmp/landscape.png",
  "duration_seconds": 12.3,
  "seed": 42
}
```

### Img2img

```bash
curl -X POST http://localhost:7862/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Oil painting style",
    "image_path": "/path/to/source.jpg",
    "image_strength": 0.5,
    "steps": 9
  }'
```

### Hot-Swap LoRAs

```bash
curl -X POST http://localhost:7862/v1/lora/swap \
  -H "Content-Type: application/json" \
  -d '{
    "loras": [
      {"path": "/path/to/style.safetensors", "scale": 0.8}
    ]
  }'
```

### Health Check

```bash
curl http://localhost:7862/health
```

Returns: loaded model, VRAM usage, active LoRAs, render queue depth.

### Shutdown

```bash
curl -X POST http://localhost:7862/v1/shutdown \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```

## Path 2: ComfyUI Protocol (For Existing ComfyUI Clients)

ComfyBox emulates the ComfyUI API on port 7870. Applications that already support ComfyUI (like Krita AI Diffusion) connect without modification.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/object_info` | GET | Node type registry (emulated) |
| `/prompt` | POST | Submit a generation workflow |
| `/queue` | GET | Queue status |
| `/interrupt` | POST | Abort the in-flight render (see below) |
| `/view` | GET | Retrieve generated images |
| `/ws` | WebSocket | Real-time progress notifications |

### `/interrupt` now aborts LTX-2 VIDEO renders too (comfybox#322)

`POST /interrupt` (and its `/v1/queue/interrupt` twin) cancels whatever render
is in flight, then broadcasts `execution_interrupted` for the active prompt(s).

**This changed in comfybox#322, and the change is intended.** Before it, the
interrupt reached image renders only: an in-flight LTX-2 video render published
no cancellable handle, so `/interrupt` returned `interrupted: false` and the
clip — 5 to 60 minutes of it — ran to completion. It now stops within one unit
of work (one sampler step, one chunk, one decode volume).

What a bridge client should know:

- **An interrupt from Krita stops a video render started by anything else.**
  The engine runs ONE render at a time across every surface (bridge, `/v1`,
  MCP, Desktop), so the interrupt is not scoped to the caller's prompt. A
  client that offers a Cancel button is offering it for whatever the box is
  doing. This was already true for image renders; it now covers video.
- **A video render started as a `/v1/video/generate/async` job reports
  `status: failed` with an additive `interrupted: true` field**, and a
  plain-English `error`, rather than a new status value — polling clients that
  switch on `status` keep working unchanged. The synchronous route returns
  HTTP 500 with `LTX-2 video interrupted by /v1/queue/interrupt`.
- **A preempting image job is never collateral, and the interrupt now targets
  what `/health` actually shows as active (comfybox#362).** During an
  engine-preemption episode (#1479) — video checkpointed, an image job
  running in its place — a plain interrupt (no `target`) cancels the VISIBLE
  image render and the checkpointed video resumes once it finishes. Before
  this fix, a plain interrupt silently abandoned the invisible video instead,
  even though `/health` and `/v1/queue` showed the image job as active the
  whole time. See `/v1/queue/interrupt`'s `target` field below to reach the
  video specifically — the bridge's own `/interrupt` has no `target` and
  always means "the active render".
- **The queue proceeds immediately** to the next job, and the interrupted
  render leaves no output file behind.

### `/v1/queue/interrupt` gains an optional `target` (comfybox#362)

`POST /v1/queue/interrupt` now accepts an optional JSON body:

```json
{"target": "active"}
```

> **The bridge's own `POST /interrupt` takes no body.** It stays exactly
> ComfyUI-shaped — Krita and every other ComfyUI client send an empty POST and
> would not know what to put in one — so it always acts on the DEFAULT target
> (whatever `/health` shows as active), which is what a Cancel button wants
> anyway. `target` is a `/v1/queue/interrupt` feature only; a body sent to the
> bridge's `/interrupt` is ignored.

- **`target` omitted, `null`, `""`, or `"active"` (default)** — cancel
  whatever `/health`/`/v1/queue` currently report as the active render.
  Outside a preemption episode this is the render itself, exactly as before.
  DURING an episode it is the preempting IMAGE job, not the checkpointed video
  underneath it — the two now always agree. With nothing rendering, the
  response is the pre-#362 body unchanged (`200`, `interrupted: false`).
- **`target: "video"`** — cancel the checkpointed/running video specifically,
  even while an episode has swapped `active` to a preempting image job. This
  is the pre-#362 (accidental) behaviour, now explicit and opt-in: the video
  is abandoned (its checkpoint dropped, not resumed) and the active image job
  is left alone. If there is no video anywhere — no episode and no video
  render — this returns **HTTP 404**: an *explicit* target that names nothing
  is a client error, and only the default target keeps the legacy
  `interrupted: false` body.
- **`target: "<job id>"`** — a job id resolves to whichever of the two it
  actually names (the active job's id behaves like `"active"`; the
  checkpointed video's ids behave like `"video"`). A job id that matches
  neither — including a merely-*pending* job's id, which this route never
  touches (`DELETE /v1/queue/{id}` cancels those) — returns HTTP 404 instead
  of a silent `interrupted: false`.

  **A video answers to BOTH of its ids.** comfybox#283 documents that a video
  job's queue id (the one `/v1/queue` and `/health.active_job_id` show) is not
  the same as its `/v1/video/status/{id}` id (the one the client that
  submitted it holds). Either one targets the video. `interrupted_job_id` in
  the response always reports the **queue** id, whichever id you targeted with.

**Target vocabulary.** `"active"` and `"video"` are reserved words, matched
case-insensitively and with surrounding whitespace trimmed (`" VIDEO "`
works). Everything else is a job id, matched **exactly** after trimming — ids
are opaque and their case is significant (a `UUID` renders upper-case), so
they are never case-folded.

The response gains two additive fields, present only when something was
actually cancelled:

```json
{"success": true, "interrupted": true, "interrupted_job_id": "…", "interrupted_kind": "video"}
```

`interrupted_kind` is a `QueueJobKind` value (`"generate"`, `"video"`, …). A
client that only reads `success`/`interrupted` sees no change from before
#362: with no body and nothing running, the response is byte-identical to what
it has always been.

### Workflow Format

ComfyUI clients send workflow JSON — a graph of nodes. ComfyBox parses the graph and extracts generation parameters:

```json
{
  "prompt": {
    "1": {
      "class_type": "KSampler",
      "inputs": {
        "seed": 42,
        "steps": 9,
        "cfg": 0.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "model": ["2", 0],
        "positive": ["3", 0],
        "negative": ["4", 0],
        "latent_image": ["5", 0]
      }
    },
    "2": {
      "class_type": "CheckpointLoaderSimple",
      "inputs": {
        "ckpt_name": "z-image-turbo-bf16"
      }
    },
    "3": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "A mountain landscape at sunset",
        "clip": ["2", 1]
      }
    }
  }
}
```

The bridge walks the node graph, resolves references, and maps to a native `/v1/generate` call.

### WebSocket Progress

Connect to `ws://localhost:7870/ws` for real-time updates:

```json
{"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}}
{"type": "progress", "data": {"value": 3, "max": 9}}
{"type": "progress", "data": {"value": 9, "max": 9}}
{"type": "executed", "data": {"node": "1", "output": {"images": [{"filename": "output.png"}]}}}
```

### Supported Node Types

The bridge recognizes these ComfyUI node types:

| Node | Maps To |
|------|---------|
| `KSampler` | Sampler, steps, CFG, scheduler |
| `CheckpointLoaderSimple` | Model selection |
| `CLIPTextEncode` | Prompt text |
| `EmptyLatentImage` | Width, height |
| `VAEDecode` | (implicit — always runs) |
| `SaveImage` | Output path |
| `LoadImage` | Img2img source |
| `LoraLoader` | LoRA path and scale |
| `ImageUpscaleWithModel` | SeedVR2/ESRGAN upscale |
| `ModelSamplingAuraFlow` | Flow-matching schedule `shift` (comfybox#154) |

Unsupported nodes are silently ignored — the bridge extracts what it can and generates with those parameters.

#### `ModelSamplingAuraFlow` — schedule shift (comfybox#154)

A workflow carrying a `ModelSamplingAuraFlow` node has its `shift` input read
and applied as the flow schedule's linear shift,
`σ' = shift·σ / (1 + (shift − 1)·σ)` — ComfyUI's `time_snr_shift`
(`comfy/model_sampling.py`). It is what Zeta Chroma needs (its author publishes
**shift 3.00**, Euler, `simple`/`normal`, CFG 4.5–5.5); `1.0` is the exact
identity, and a workflow with no such node renders exactly as it did before.

- The node is advertised in `GET /object_info` under category `model/patch`
  with upstream's own default (`1.73`), so a client can place it and set it.
- On a multi-pass graph the **lowest node id wins**, the same deterministic rule
  the bridge uses for schedulers and controlnets; the others are logged and
  ignored.
- A `shift` that is not a positive finite number is logged and DROPPED — the
  graph renders on the model's own schedule rather than failing with a 400 a
  Krita user would never see.
- The shift is refused (400) if the resident model family does not read it —
  see the `shift` table in [`api-notes.md`](api-notes.md).
- **`ModelSamplingSD3` and `ModelSamplingFlux` are deliberately NOT mapped.**
  Same sigma warp, different parameterisations (a 1000× timestep `multiplier`,
  and a log-shift) that the engine has no seam for; reading either as AuraFlow
  would be a silent substitution. They remain structural glue.

## Path 3: MCP (For AI Assistants)

The Model Context Protocol exposes 18 tools via JSON-RPC 2.0 over stdio. See [MCP Tool Reference](mcp-reference.md) for the complete tool catalog.

```bash
# Register with Claude Code
claude mcp add comfybox -- ComfyBox mcp --port 7862

# Register with any MCP-compatible client
ComfyBox mcp --port 7862
# Reads JSON-RPC from stdin, writes to stdout
```

### Example JSON-RPC Call

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
  "name": "generate_image",
  "arguments": {
    "prompt": "A portrait in renaissance style",
    "width": 768,
    "height": 1152,
    "seed": 42
  }
}}
```

## Writing a New Conduit

A "conduit" is an integration that bridges your application to ComfyBox. Here's the pattern:

### Step 1: Choose Your API

- **REST API** if you're starting from scratch
- **ComfyUI Protocol** if your app already has ComfyUI support
- **MCP** if your app is an AI assistant

### Step 2: Discover Capabilities

```bash
# What model is loaded?
curl http://localhost:7862/health

# What LoRAs are available?
# (via MCP: list_loras tool)

# What models are supported?
# (via MCP: list_models tool)
```

### Step 3: Generate

```python
import requests
import json

COMFYBOX_URL = "http://localhost:7862"

def generate(prompt, width=1024, height=1024, steps=9, seed=None):
    """Generate an image via ComfyBox REST API."""
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
    }
    if seed is not None:
        payload["seed"] = seed
    
    response = requests.post(f"{COMFYBOX_URL}/v1/generate", json=payload)
    response.raise_for_status()
    return response.json()

# Text-to-image
result = generate("A cat sitting on a windowsill", seed=42)
print(f"Image saved to: {result['output_path']}")

# Img2img
result = requests.post(f"{COMFYBOX_URL}/v1/generate", json={
    "prompt": "Oil painting style",
    "image_path": "/path/to/photo.jpg",
    "image_strength": 0.5,
}).json()
```

### Step 4: Handle Progress (Optional)

For long-running renders, connect to the WebSocket for progress:

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data["type"] == "progress":
        step = data["data"]["value"]
        total = data["data"]["max"]
        print(f"Step {step}/{total}")
    elif data["type"] == "executed":
        print("Done!")
        ws.close()

ws = websocket.WebSocketApp("ws://localhost:7870/ws",
                            on_message=on_message)
ws.run_forever()
```

### Step 5: Manage LoRAs (Optional)

```python
def swap_loras(loras):
    """Hot-swap LoRA weights."""
    requests.post(f"{COMFYBOX_URL}/v1/lora/swap", json={
        "loras": [{"path": path, "scale": scale} for path, scale in loras]
    })

# Apply a style LoRA
swap_loras([("/path/to/watercolor.safetensors", 0.8)])

# Remove all LoRAs
swap_loras([])
```

## Swift Conduit Example

For macOS/iOS apps using Swift:

```swift
import Foundation

struct ComfyBoxClient {
    let baseURL: URL
    
    init(host: String = "localhost", port: Int = 7862) {
        self.baseURL = URL(string: "http://\(host):\(port)")!
    }
    
    func generate(prompt: String, width: Int = 1024, height: Int = 1024,
                  steps: Int = 9, seed: Int? = nil) async throws -> GenerateResult {
        var body: [String: Any] = [
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
        ]
        if let seed { body["seed"] = seed }
        
        var request = URLRequest(url: baseURL.appendingPathComponent("/v1/generate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(GenerateResult.self, from: data)
    }
    
    func health() async throws -> HealthStatus {
        let (data, _) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("/health"))
        return try JSONDecoder().decode(HealthStatus.self, from: data)
    }
}

struct GenerateResult: Codable {
    let outputPath: String
    let durationSeconds: Double
    let seed: Int
}

struct HealthStatus: Codable {
    let model: String
    let status: String
}
```

## Network Configuration

### Local (Same Machine)

Default setup — app and ComfyBox on the same Mac:

```
App → http://localhost:7862 (REST)
App → ws://localhost:7870/ws (ComfyUI WebSocket)
```

### Remote (Different Machine)

ComfyBox on a Mac, your app elsewhere on the network:

```bash
# Start WarmServer bound to all interfaces
ComfyBox serve -m Tongyi-MAI/Z-Image-Turbo --port 7862 --host 0.0.0.0
```

```
App → http://mac-ip:7862 (REST)
App → ws://mac-ip:7870/ws (ComfyUI WebSocket)
```

### SSH Tunnel (Secure Remote)

For remote access without exposing ports:

```bash
ssh -L 7862:localhost:7862 -L 7870:localhost:7870 user@mac-host
```

Then connect to `localhost:7862` / `localhost:7870` as if local.

## Error Handling

All endpoints return standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Endpoint not found |
| 500 | Server error (model crash, OOM) |
| 503 | Server busy (queue full) |

Error response body:
```json
{
  "error": "Output path not within allowed directory",
  "code": "INVALID_OUTPUT_PATH"
}
```

## Performance Tips

- **Keep the server warm** — first render loads the model (~30s), subsequent renders are fast (~10-15s for 1024x1024)
- **Reuse connections** — HTTP keep-alive reduces overhead
- **Batch with seeds** — generate variations by changing only the seed
- **Hot-swap LoRAs** instead of restarting (~2s vs ~30s)
- **Never run concurrent GPU renders** — queue them sequentially
- **Use quantized models** on memory-constrained devices (8-bit: ~4GB vs ~7GB for BF16)

## Existing Conduits

| Conduit | Status | Integration Path |
|---------|--------|-----------------|
| Krita AI Diffusion | Working | ComfyUI Protocol |
| coffeeshop.app | Planned | REST API + WebSocket |
| Inkscape | Planned | REST API |
| Telegram ImageBot | Working | MCP via daemon |
| SDK (Python) | This guide | REST API |
| SDK (Swift) | This guide | REST API |
