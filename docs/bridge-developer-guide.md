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
- **A preempting image job is never collateral.** If the interrupt lands while
  an engine-preemption episode (#1479) is running someone else's image render,
  that image render finishes normally; only the video is abandoned.
- **The queue proceeds immediately** to the next job, and the interrupted
  render leaves no output file behind.

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

Unsupported nodes are silently ignored — the bridge extracts what it can and generates with those parameters.

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
