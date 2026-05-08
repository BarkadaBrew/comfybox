# ComfyBox MCP Server Design

**Issue**: BarkadaBrew/comfybox#86
**Date**: 2026-05-08
**Status**: Draft

## Overview

Add a Model Context Protocol (MCP) server to ComfyBox, enabling Claude Code (and other MCP clients) to interact with the WarmServer image generation API over stdio JSON-RPC 2.0. The MCP server wraps the existing WarmServer REST endpoints as discrete tools, following the same architecture as the coffeeshop-server MCP bridge (`src/mcp/mcp-server.ts`).

## Motivation

Today, the Bree daemon on the Linux server (10.0.100.232) talks to ComfyBox on the Mac (10.0.100.134) via raw HTTP calls to the WarmServer REST API. This works, but:

1. **Claude Code on the Mac has no direct access** to ComfyBox's capabilities. Adding MCP means `claude mcp add comfybox -- comfybox mcp` instantly gives Claude Code generate, swap, list, and health tools.
2. **The Bree daemon can spawn ComfyBox MCP as a child process**, using the existing `McpClient`/`McpManager` infrastructure. This replaces the custom HTTP client code with the standard MCP tool bridge pattern.
3. **Codex and other MCP-aware agents** get image generation for free via the tool catalog.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ MCP Client (Claude Code, Bree daemon, Codex)         │
│   stdin/stdout ←→ JSON-RPC 2.0                       │
└──────────────┬───────────────────────────────────────┘
               │ stdio (one JSON object per line)
┌──────────────▼───────────────────────────────────────┐
│ ComfyBox MCP Server                                   │
│   Process: `comfybox mcp [--port 7862]`               │
│   Protocol: MCP 2024-11-05                            │
│   Transport: stdin/stdout line-delimited JSON-RPC 2.0 │
│   Logging: stderr                                     │
│                                                       │
│   ┌─────────────────────────────────────────┐         │
│   │ Tool Router                             │         │
│   │   initialize → handshake                │         │
│   │   tools/list  → static tool catalog     │         │
│   │   tools/call  → HTTP to WarmServer      │         │
│   └──────────────┬──────────────────────────┘         │
└──────────────────┼────────────────────────────────────┘
                   │ HTTP (localhost only)
┌──────────────────▼────────────────────────────────────┐
│ WarmServer (already running)                           │
│   localhost:7862                                       │
│   POST /v1/generate, POST /v1/lora/swap, etc.         │
└───────────────────────────────────────────────────────┘
```

The MCP server is a thin stdio-to-HTTP bridge. It does not load models or run inference. It translates MCP `tools/call` requests into HTTP requests to the local WarmServer, then returns the response as MCP tool results.

### Key design decisions

- **Static tool catalog.** Unlike the coffeeshop-server MCP server (which fetches tools from the daemon's `/v1/tools/catalog` at runtime), ComfyBox defines its tool list at compile time. The WarmServer API is stable and known; there is no dynamic tool registration.
- **No heartbeat.** The coffeeshop MCP server sends heartbeats because the Bree daemon tracks Channels liveness. ComfyBox has no such requirement. The MCP server is stateless beyond its HTTP connection to WarmServer.
- **Swift implementation.** The MCP server is implemented in Swift and compiled into the existing `ZImageCLI` binary as a new subcommand (`comfybox mcp`). No additional binary or npm dependency.

## Protocol

MCP 2024-11-05 over stdio. One JSON object per line on stdin (requests) and stdout (responses). Stderr for logging.

### Handshake

```json
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"claude-code","version":"1.0.0"}}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"comfybox","version":"1.0.0"}}}
→ {"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
```

Note: `listChanged: false` because ComfyBox tools are static. The coffeeshop-server uses `listChanged: true` because daemon tools can change at runtime.

## Tool Definitions

### 1. `generate_image`

Text-to-image generation. Wraps `POST /v1/generate`.

```json
{
  "name": "generate_image",
  "description": "Generate an image from a text prompt using the loaded model. Supports text-to-image and img2img. Returns the output file path and render duration.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "Text prompt describing the desired image."
      },
      "negative_prompt": {
        "type": "string",
        "description": "Negative prompt — concepts to avoid. Only effective on non-distilled models (Z-Image Base, Flux 2 Klein Base, FIBO)."
      },
      "width": {
        "type": "integer",
        "description": "Image width in pixels. Must be divisible by 16. Default: model-dependent (typically 1024)."
      },
      "height": {
        "type": "integer",
        "description": "Image height in pixels. Must be divisible by 16. Default: model-dependent (typically 1024)."
      },
      "steps": {
        "type": "integer",
        "description": "Number of denoising steps. More steps = higher quality but slower. Distilled models (Turbo): 4-9. Base models: 20-50."
      },
      "guidance": {
        "type": "number",
        "description": "CFG guidance scale. Higher values = stronger prompt adherence. 0.0 for distilled models; 3.5-7.0 for base models."
      },
      "seed": {
        "type": "integer",
        "description": "Random seed for reproducibility. Omit for random."
      },
      "output_path": {
        "type": "string",
        "description": "Output file path. Must be within the server's allowed output directory. Omit to write to a temp file."
      },
      "scheduler": {
        "type": "string",
        "description": "Sampler/scheduler algorithm. Options: euler (default), heun, res_2s, ddim, dpmpp_2m."
      },
      "sigma_schedule": {
        "type": "string",
        "description": "Sigma noise schedule. Options: flow (default), beta, beta57, linear."
      },
      "image_path": {
        "type": "string",
        "description": "Source image path for img2img. The loaded model must support img2img."
      },
      "image_strength": {
        "type": "number",
        "description": "Img2img denoise strength (0.0-1.0). 1.0 = full txt2img, 0.5 = preserve composition. Default: 0.7."
      }
    },
    "required": ["prompt"]
  }
}
```

**WarmServer mapping**: POST `/v1/generate` with JSON body. Field names convert from snake_case to camelCase (the WarmServer uses `JSONDecoder.keyDecodingStrategy = .convertFromSnakeCase`).

**Response**: `{ "success": true, "output_path": "/path/to/image.png", "duration_ms": 14500 }`

### 2. `swap_loras`

Hot-swap LoRA weights without restarting the server. Wraps `POST /v1/lora/swap`.

```json
{
  "name": "swap_loras",
  "description": "Hot-swap active LoRA weights on the loaded model. Replaces all currently active LoRAs with the provided set. Pass an empty array to remove all LoRAs.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "loras": {
        "type": "array",
        "description": "LoRA configurations to activate.",
        "items": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to .safetensors file, or HuggingFace model ID (e.g. 'owner/repo'). Bare filenames resolve to ~/bin/zimage/loras/."
            },
            "scale": {
              "type": "number",
              "description": "LoRA weight scale (0.0-2.0). Default: 1.0."
            }
          },
          "required": ["path"]
        }
      }
    },
    "required": ["loras"]
  }
}
```

**WarmServer mapping**: POST `/v1/lora/swap` with `{ "loras": [...] }`.

**Response**: `{ "success": true, "lora_count": 2, "loras": [{"source": "/path/to/lora.safetensors", "scale": 1.0}] }`

### 3. `list_models`

List all registered model families and their capabilities. Wraps `GET /v1/models`.

```json
{
  "name": "list_models",
  "description": "List all supported ComfyBox model families with capabilities, recommended parameters, and HuggingFace IDs.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**WarmServer mapping**: GET `/v1/models`.

**Response**: `{ "models": [...], "count": N }` where each model includes `id`, `family`, `variant`, `quantization`, `display_name`, `description`, `default_steps`, `default_guidance`, `supports_lora`, `supports_img2img`, `default_resolution`, `estimated_vram_gb`, `huggingface_id`.

### 4. `list_styles`

List available style presets. Wraps `GET /v1/styles`.

```json
{
  "name": "list_styles",
  "description": "List available style presets with prompt engineering templates, recommended parameters, and model pairings.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**WarmServer mapping**: GET `/v1/styles`.

**Response**: `{ "styles": [...], "count": N }` where each style includes `id`, `name`, `category`, `description`, `prompt_prefix`, `prompt_suffix`, `negative_prompt`, `recommended_model`, `steps`, `guidance`, `width`, `height`.

### 5. `server_health`

Check WarmServer status, memory, queue depth, and loaded model info. Wraps `GET /health`.

```json
{
  "name": "server_health",
  "description": "Get ComfyBox server health: loaded model, LoRA state, memory usage, render stats, and queue depth.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**WarmServer mapping**: GET `/health`.

**Response**: `{ "status": "ok", "model": "...", "model_family": "flux2-klein", "loaded": true, "loras": [...], "uptime_seconds": 3600, "render_count": 42, "memory_usage_mb": 8192, "pending_count": 0, "max_pending": 10, ... }`

### 6. `queue_status`

Check the generation queue. Wraps `GET /queue`.

```json
{
  "name": "queue_status",
  "description": "Get the current generation queue status: pending jobs, running job, and history.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**WarmServer mapping**: GET `/queue`.

**Response**: ComfyUI-compatible queue object with `queue_running` and `queue_pending` arrays.

### 7. `clear_queue`

Cancel all pending generation jobs. Wraps `POST /queue`.

```json
{
  "name": "clear_queue",
  "description": "Cancel all pending generation jobs in the queue. Does not affect the currently running job.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**WarmServer mapping**: POST `/queue` with `{ "clear": true }`.

**Response**: `{ "success": true, "cleared": N }`

### 8. `list_loras`

List available LoRA files discovered on disk. Wraps the object_info endpoint's LoraLoader node input.

```json
{
  "name": "list_loras",
  "description": "List LoRA files available in the LoRA directory (~/bin/zimage/loras/). Returns filenames that can be passed to swap_loras.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**Implementation note**: The WarmServer does not have a dedicated `/v1/loras` endpoint. This tool scans `/object_info` response and extracts the LoraLoader node's `lora_name` input values. Alternatively, it can scan the LoRA directory directly. The MCP server should prefer the `/object_info` approach since ComfyBridgeObjectInfo already enumerates LoRA files.

**Response**: `{ "loras": ["style-kira-v3.safetensors", "lighting-golden-hour.safetensors", ...], "count": N, "directory": "~/bin/zimage/loras" }`

### 9. `shutdown_server`

Gracefully shut down the WarmServer. Wraps `POST /v1/shutdown`.

```json
{
  "name": "shutdown_server",
  "description": "Gracefully shut down the ComfyBox WarmServer. In-flight renders complete before exit. Use sparingly — requires manual restart.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "confirm": {
        "type": "boolean",
        "description": "Must be true to confirm shutdown. Safety guard against accidental invocation."
      }
    },
    "required": ["confirm"]
  }
}
```

**WarmServer mapping**: POST `/v1/shutdown` (only if `confirm` is `true`).

**Response**: `{ "success": true, "message": "Shutting down" }`

### 10. `system_stats`

Get hardware and system information. Wraps the ComfyUI bridge `GET /system_stats`.

```json
{
  "name": "system_stats",
  "description": "Get system hardware information: Apple Silicon device name, total memory, and VRAM available for generation.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**WarmServer mapping**: GET `/system_stats`.

**Response**: ComfyUI-compatible system stats with `system.os`, `system.ram`, `devices[].name`, `devices[].vram_total`.

### 11. `apply_style`

Apply a style preset to a prompt (client-side transform, no HTTP call). This is a convenience tool that looks up a style preset and returns the enhanced prompt without generating an image.

```json
{
  "name": "apply_style",
  "description": "Apply a style preset to a prompt. Returns the enhanced prompt with prefix/suffix, negative prompt, and recommended generation parameters. Does not generate an image — use generate_image with the returned parameters.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "style_id": {
        "type": "string",
        "description": "Style preset ID (from list_styles)."
      },
      "prompt": {
        "type": "string",
        "description": "Base prompt to enhance with the style."
      },
      "negative_prompt": {
        "type": "string",
        "description": "Optional negative prompt to merge with the style's negative prompt."
      }
    },
    "required": ["style_id", "prompt"]
  }
}
```

**Implementation**: Fetches styles from `/v1/styles`, finds the matching preset, applies prefix/suffix/negative prompt transforms, and returns the composed result with recommended parameters.

**Response**: `{ "enhanced_prompt": "cinematic film still, a woman in a red dress, dramatic lighting, 35mm film grain", "negative_prompt": "cartoon, illustration, painting", "recommended": { "steps": 25, "guidance": 4.0, "width": 1344, "height": 768 } }`

## WarmServer API Wrapping

### HTTP Client

The MCP server includes a minimal HTTP client that connects to `127.0.0.1:<port>` (default 7862). This is structurally identical to the coffeeshop-server pattern:

```swift
struct WarmServerClient {
    let host: String  // "127.0.0.1"
    let port: UInt16  // 7862

    func get(_ path: String) async throws -> (Int, Data)
    func post(_ path: String, body: Encodable) async throws -> (Int, Data)
}
```

The client uses Foundation's `URLSession` (or raw `NWConnection` for zero-dependency purity matching the project's style). All requests are localhost-only. Timeout: 300 seconds (matching the coffeeshop MCP server, since renders can take minutes).

### Error mapping

| WarmServer HTTP Status | MCP Response |
|----------------------|--------------|
| 200 | `{ "content": [{ "type": "text", "text": "<JSON result>" }] }` |
| 400 | `{ "content": [{ "type": "text", "text": "Error: <message>" }], "isError": true }` |
| 429 | `{ "content": [{ "type": "text", "text": "Error: Queue full" }], "isError": true }` |
| 500 | `{ "content": [{ "type": "text", "text": "Error: <message>" }], "isError": true }` |
| Connection refused | `{ "content": [{ "type": "text", "text": "Error: WarmServer not running at 127.0.0.1:7862" }], "isError": true }` |

### Field name convention

The WarmServer uses `JSONDecoder.keyDecodingStrategy = .convertFromSnakeCase`, so the MCP server can send snake_case JSON keys and they will be decoded correctly. The MCP tool schemas use snake_case for consistency with the coffeeshop-server convention.

## Swift Implementation Plan

### New Files

| File | Purpose |
|------|---------|
| `Sources/ZImage/MCP/MCPServer.swift` | Main entry point: stdio read loop, JSON-RPC dispatch, tool routing |
| `Sources/ZImage/MCP/MCPTypes.swift` | Protocol types: JsonRpcRequest, JsonRpcResponse, JsonRpcNotification, MCPToolDefinition |
| `Sources/ZImage/MCP/MCPToolCatalog.swift` | Static tool definitions (the 11 tools above) with inputSchema |
| `Sources/ZImage/MCP/MCPToolExecutor.swift` | Tool execution: maps tool name + args to WarmServer HTTP calls |
| `Sources/ZImage/MCP/WarmServerClient.swift` | Minimal HTTP client for localhost WarmServer requests |

### Existing Code to Reuse

| Existing File | What to Reuse |
|---------------|---------------|
| `Sources/ZImageCLI/main.swift` | Add `case "mcp"` to the subcommand switch alongside `serve`, `upscale`, `control` |
| `Sources/ZImage/Server/WarmServer.swift` | Reference for API contracts (GeneratePayload, LoRASwapPayload, response types) |
| `Sources/ZImage/Server/ComfyBridge/ComfyBridgeStylePresets.swift` | `ComfyBoxStylePresets.toJSON()` for style listing; `ComfyBoxStylePreset.apply()` for `apply_style` |
| `Sources/ZImage/Server/ComfyBridge/ComfyBridgeModelRegistry.swift` | `ComfyBoxModelRegistry.allModels` for model listing |
| `Sources/ZImage/Server/ComfyBridge/ComfyBridgeObjectInfo.swift` | LoRA file enumeration for `list_loras` |

### CLI Integration

Add `mcp` as a new subcommand to `ZImageCLI`:

```
comfybox mcp [--port 7862] [--host 127.0.0.1]
```

The subcommand starts the MCP server, which reads JSON-RPC from stdin, dispatches to tools, and writes responses to stdout. It runs until stdin closes or SIGTERM/SIGINT is received.

```swift
// In main.swift subcommand dispatch:
case "mcp":
    try runMCP(args: Array(args.dropFirst()))
    return
```

### MCPServer Core Loop

The core loop mirrors the coffeeshop-server pattern exactly:

```swift
public final class MCPServer {
    private let client: WarmServerClient
    private let catalog: MCPToolCatalog
    private let executor: MCPToolExecutor

    func run() {
        log("Starting — bridging to WarmServer at \(client.host):\(client.port)")

        while let line = readLine(strippingNewline: true) {
            guard !line.isEmpty else { continue }

            guard let data = line.data(using: .utf8),
                  let msg = try? JSONDecoder().decode(JsonRpcMessage.self, from: data) else {
                log("Invalid JSON: \(line.prefix(200))")
                continue
            }

            if let id = msg.id {
                // Request — dispatch and respond
                Task {
                    let response = await handleRequest(id: id, method: msg.method, params: msg.params)
                    send(response)
                }
            } else {
                // Notification
                handleNotification(method: msg.method)
            }
        }

        log("stdin closed — shutting down")
    }

    private func send(_ response: JsonRpcResponse) {
        guard let data = try? JSONEncoder().encode(response),
              let line = String(data: data, encoding: .utf8) else { return }
        print(line)  // stdout, includes newline
        fflush(stdout)
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("[mcp-server:comfybox] \(message)\n".utf8))
    }
}
```

### Binary naming

The `ZImageCLI` executable will be installed as `comfybox` (or symlinked) so the MCP registration command is clean:

```bash
claude mcp add comfybox -- comfybox mcp
```

This follows the pattern from CLAUDE.md: the executable name is the MCP server name.

## Claude Code Integration

### Registration

```bash
# Add ComfyBox MCP server to Claude Code
claude mcp add comfybox -- comfybox mcp --port 7862

# Or with explicit path if not in $PATH
claude mcp add comfybox -- /path/to/comfybox mcp --port 7862

# Verify
claude mcp list
```

This registers ComfyBox as an MCP server in `~/.claude.json`. Claude Code will spawn `comfybox mcp --port 7862` as a child process on startup, perform the initialize handshake, and make all 11 tools available.

### Tool usage in Claude Code

Once registered, Claude Code can use the tools directly:

```
User: Generate a portrait of a woman in golden hour lighting

Claude: [calls generate_image with prompt="portrait of a woman, golden hour lighting, warm tones"]
→ { "success": true, "output_path": "/tmp/zimage-abc123.png", "duration_ms": 14200 }

The image has been generated at /tmp/zimage-abc123.png (14.2 seconds).
```

### Workflow example

```
User: What models and LoRAs are available?

Claude: [calls list_models] → 6 models
        [calls list_loras] → 4 LoRA files
        [calls server_health] → Flux 2 Klein 9B loaded, 2 LoRAs active

You have 6 models registered. Currently loaded: Flux 2 Klein 9B (distilled).
Available LoRAs: style-kira-v3.safetensors, lighting-golden-hour.safetensors, ...
Active LoRAs: style-kira-v3 (scale 1.0), lighting-golden-hour (scale 0.8)
```

## Bree Daemon Integration

### Spawn as MCP child process

The Bree daemon already has `McpManager` and `McpClient` for spawning and managing MCP servers. ComfyBox MCP is configured as an MCP server in `~/.bree/config.json`:

```json
{
  "mcp": {
    "servers": [
      {
        "id": "comfybox",
        "name": "ComfyBox Image Engine",
        "command": "ssh toddwalderman@10.0.100.134 comfybox mcp --port 7862",
        "enabled": true
      }
    ]
  }
}
```

**Note on cross-machine access**: The Bree daemon runs on the Linux server (10.0.100.232) while ComfyBox runs on the Mac (10.0.100.134). The MCP stdio transport works over SSH: the daemon spawns `ssh user@mac comfybox mcp` as a child process, and JSON-RPC flows over the SSH tunnel's stdin/stdout. The WarmServer HTTP calls remain localhost-only on the Mac.

### Tool namespacing

Per `McpManager` convention, ComfyBox tools are registered with the prefix `mcp_comfybox__`:

| MCP Tool | Daemon Tool Name |
|----------|-----------------|
| `generate_image` | `mcp_comfybox__generate_image` |
| `swap_loras` | `mcp_comfybox__swap_loras` |
| `list_models` | `mcp_comfybox__list_models` |
| `server_health` | `mcp_comfybox__server_health` |
| ... | ... |

### Migration path

The existing `image-gen-tools.ts` image service integration (`ImageServiceClient`) can gradually migrate to use the MCP tools instead. Phase 1 keeps both paths; Phase 2 removes the custom HTTP client once MCP is stable.

## Security Considerations

### Localhost-only API access

The WarmServer binds to `127.0.0.1` by default. The MCP server connects only to localhost. No network exposure.

When accessed from the Bree daemon over SSH, the SSH tunnel terminates on the Mac and the MCP server's HTTP calls stay on localhost. The WarmServer is never exposed to the network.

### Output path validation

The WarmServer validates that `output_path` is within the configured `allowedOutputDirectory` (see `WarmServerOutputPathValidator`). The MCP server passes through the user-provided path without modification. Symlink traversal and path component resolution are handled server-side.

The MCP server should NOT add its own path validation — that would duplicate logic and risk divergence from the WarmServer's canonical validation.

### No authentication on MCP transport

The MCP protocol over stdio does not include authentication. This is acceptable because:

1. The MCP server is a local child process spawned by a trusted parent (Claude Code or Bree daemon).
2. The WarmServer HTTP API has no authentication (localhost-only design).
3. SSH handles authentication for cross-machine access.

### Shutdown guard

The `shutdown_server` tool requires `confirm: true` as a safety guard. This prevents the LLM from accidentally shutting down the server in a tool-use loop.

### Resource limits

The WarmServer already enforces `maxPendingRequests` (default: 10). The MCP server does not add additional rate limiting. HTTP 429 responses from a full queue are surfaced as MCP tool errors.

## Testing Plan

### Unit tests

- JSON-RPC parsing: valid requests, notifications, malformed input
- Tool catalog: verify all 11 tools have valid inputSchema
- Tool executor: mock WarmServerClient, verify HTTP method/path/body mapping
- Error mapping: verify all HTTP status codes map to correct MCP responses
- Field name conversion: verify snake_case MCP params map to camelCase WarmServer fields

### Integration tests

- Start WarmServer in test mode, start MCP server, exercise full tool lifecycle:
  1. `initialize` handshake
  2. `tools/list` returns all 11 tools
  3. `server_health` returns valid status
  4. `list_models` returns model registry
  5. `generate_image` with minimal prompt returns output path
  6. `swap_loras` with empty array clears LoRAs
  7. `shutdown_server` with `confirm: false` is rejected
  8. `shutdown_server` with `confirm: true` triggers shutdown

### Manual verification

- `claude mcp add comfybox -- comfybox mcp` registers successfully
- `claude mcp list` shows comfybox with 11 tools
- In a Claude Code session, ask to generate an image and verify the tool is called

## Open Questions

1. **Image data in responses**: Should `generate_image` return the image as a base64 `image` content block, or just the file path? File path is simpler and matches the WarmServer response. Base64 would let Claude Code display the image inline but adds significant payload size. **Recommendation**: File path only in v1. Add an optional `include_image` parameter in v2 if needed.

2. **Progress reporting**: MCP supports `notifications/progress` for long-running operations. Image generation can take 3-90 seconds. Should the MCP server stream denoising progress? **Recommendation**: Not in v1. The WarmServer doesn't expose progress on the HTTP API (only via WebSocket). Add progress notifications in v2 when the WarmServer gains SSE or chunked progress on `/v1/generate`.

3. **Binary name**: The current executable is `ZImageCLI`. Should we rename to `comfybox` for the MCP registration? **Recommendation**: Yes. Add `comfybox` as an alias (symlink or rename the product in Package.swift). `ZImageCLI` is an internal name; `comfybox` is the user-facing brand.
