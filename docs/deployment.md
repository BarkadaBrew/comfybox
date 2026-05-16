# ComfyBox Deployment Guide

How to run ComfyBox as a persistent service — warm server, keepalive, remote access, and integration with the CoffeeShop daemon.

## Serve Mode

ComfyBox can run as a warm HTTP server that keeps the model loaded in GPU memory:

```bash
ComfyBox serve \
  -m Tongyi-MAI/Z-Image-Turbo \
  --port 7862 \
  --host 0.0.0.0 \
  --allowed-output-directory ~/renders
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model, -m` | Z-Image-Turbo | Model to load on startup |
| `--port` | 7862 | HTTP port to bind |
| `--host` | 127.0.0.1 | Interface to bind (0.0.0.0 for network access) |
| `--allowed-output-directory` | current dir | Sandboxed output path |
| `--cache-limit` | unlimited | GPU memory cache limit in MB |
| `--text-encoder-path` | auto-detect | Override text encoder directory |
| `--lora` | none | Pre-load LoRA weights (repeatable) |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/generate` | POST | Submit a render (txt2img or img2img) |
| `/v1/lora/swap` | POST | Hot-swap LoRA weights |
| `/health` | GET | Server health and loaded model info |
| `/v1/shutdown` | POST | Graceful shutdown |

### Health Check

```bash
curl http://localhost:7862/health
```

Returns loaded model, VRAM usage, active LoRAs, render statistics, and queue depth.

## Keepalive with Screen

For production use, wrap the server in a keepalive script that auto-restarts on crashes:

### Keepalive Script

Save as `~/bin/comfybox-keepalive.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

COMFYBOX="$HOME/Projects/zimage.swift/.build/release/ComfyBox"
MODEL="Tongyi-MAI/Z-Image-Turbo"
ENCODER="$HOME/Models/z-image-turbo-bf16/text_encoder QWen Large"
PORT=7862
OUTPUT_DIR="$HOME/renders"

CRASH_COUNT=0
MAX_CRASHES=5
WINDOW_START=$(date +%s)
WINDOW_SECS=600  # 10 minute sliding window

# Reset crash counter on SIGHUP
trap 'CRASH_COUNT=0; WINDOW_START=$(date +%s); echo "[keepalive] Reset crash counter"' HUP

while true; do
    echo "[keepalive] Starting ComfyBox serve on :${PORT}..."
    
    "$COMFYBOX" serve \
        -m "$MODEL" \
        --text-encoder-path "$ENCODER" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --allowed-output-directory "$OUTPUT_DIR" \
    || EXIT_CODE=$?
    
    # Clean exit — don't restart
    if [ "${EXIT_CODE:-0}" -eq 0 ]; then
        echo "[keepalive] Clean exit. Done."
        break
    fi
    
    # Crash — check circuit breaker
    NOW=$(date +%s)
    ELAPSED=$(( NOW - WINDOW_START ))
    
    if [ "$ELAPSED" -gt "$WINDOW_SECS" ]; then
        # Reset window
        CRASH_COUNT=1
        WINDOW_START=$NOW
    else
        CRASH_COUNT=$(( CRASH_COUNT + 1 ))
    fi
    
    if [ "$CRASH_COUNT" -ge "$MAX_CRASHES" ]; then
        echo "[keepalive] CIRCUIT BREAKER: $MAX_CRASHES crashes in ${WINDOW_SECS}s. Giving up."
        exit 1
    fi
    
    echo "[keepalive] Crash #${CRASH_COUNT}/${MAX_CRASHES}. Restarting in 3s..."
    sleep 3
done
```

### Running in Screen

```bash
# Start keepalive in a detached screen session
screen -dmS warmserver ~/bin/comfybox-keepalive.sh

# Attach to see logs
screen -r warmserver

# Detach: Ctrl+A, then D

# Check if running
screen -ls | grep warmserver

# Reset crash counter (without restarting)
screen -S warmserver -X stuff $'\x01'  # Send SIGHUP
```

### Why Not launchd?

launchd is macOS's native service manager, but ComfyBox's GPU initialization can deadlock under launchd's sandbox. Symptoms: binary spawns at 15MB RSS, 0% CPU, never progresses past Metal device init.

**Workaround:** Use screen-based keepalive instead. The screen approach:
- Auto-restarts on crashes (up to 5 in 10 minutes)
- Survives terminal disconnect
- Logs are visible via `screen -r`
- Clean exit (code 0) does not restart
- SIGHUP resets crash counter

## ComfyUI Bridge

Run the ComfyUI-compatible bridge alongside the WarmServer:

```bash
# WarmServer on 7862, Bridge on 7870
ComfyBox serve -m Tongyi-MAI/Z-Image-Turbo --port 7862 &
# Bridge connects to WarmServer and exposes ComfyUI protocol
# (Bridge is built into the serve command on port 7870)
```

The bridge translates ComfyUI workflow JSON into native WarmServer API calls. Krita AI Diffusion, Lingdong, or any ComfyUI client connects to port 7870.

## MCP Server

Run the MCP server alongside a WarmServer for AI assistant integration:

```bash
# Start WarmServer first
ComfyBox serve -m Tongyi-MAI/Z-Image-Turbo --port 7862

# Then run MCP server pointing to it
ComfyBox mcp --port 7862
```

The MCP server reads JSON-RPC from stdin and writes to stdout. It bridges to the WarmServer via HTTP. See [MCP Tool Reference](mcp-reference.md) for the 18 available tools.

### Remote MCP via SSH

For a daemon on a remote server that needs to control a Mac's GPU:

```bash
# The daemon spawns this as a child process:
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
  user@mac-host \
  "cd /path/to/comfybox && .build/release/ComfyBox mcp --port 7862"
```

**Daemon configuration** (`~/.bree/config.json`):
```json
{
  "mcp": {
    "servers": [
      {
        "id": "comfybox",
        "name": "ComfyBox",
        "command": "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 user@mac \"cd /path/to/comfybox && .build/release/ComfyBox mcp --port 7862\"",
        "enabled": true,
        "toolTimeoutMs": 300000
      }
    ]
  }
}
```

**Important:** Set `toolTimeoutMs` to at least 300000 (5 minutes) for GPU render servers. The default 60s timeout will kill renders mid-generation.

### Auto-Reconnect

The daemon's MCP manager automatically reconnects when the SSH connection drops:
- Exponential backoff: 5s → 10s → 20s → ... → 5min cap
- Maximum 10 reconnect attempts before giving up
- Tools are automatically re-registered on reconnect
- All pending requests are rejected on disconnect

## Network Topology

```
┌─────────────────────────────────────────────────────┐
│  Linux Server (10.0.100.232)                        │
│                                                     │
│  ┌──────────────────────────────────┐               │
│  │ Bree Daemon (:3777)             │               │
│  │  ├─ MCP Manager                 │               │
│  │  │   └─ McpClient (SSH child)  ─┼── SSH ──┐    │
│  │  ├─ Tool Executor               │          │    │
│  │  └─ Image Service (:7861)       │          │    │
│  └──────────────────────────────────┘          │    │
└────────────────────────────────────────────────┼────┘
                                                 │
┌────────────────────────────────────────────────┼────┐
│  Mac (10.0.100.134, M3 Max 128GB)              │    │
│                                                │    │
│  ┌──────────────────┐    ┌─────────────────┐   │    │
│  │ ComfyBox MCP     │◀───│ SSH connection  │◀──┘    │
│  │ (stdio JSON-RPC) │    └─────────────────┘        │
│  └────────┬─────────┘                               │
│           │ HTTP                                     │
│  ┌────────▼─────────┐    ┌─────────────────┐        │
│  │ WarmServer       │    │ ComfyUI Bridge  │        │
│  │ (:7862)          │    │ (:7870)         │        │
│  └────────┬─────────┘    └────────┬────────┘        │
│           │                       │                  │
│  ┌────────▼───────────────────────▼────────┐        │
│  │ MLX Engine (Metal GPU)                   │        │
│  │  ├─ Z-Image Turbo (~7 GB)               │        │
│  │  ├─ SeedVR2 3B (~7 GB)                  │        │
│  │  └─ LoRA weights                         │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

## Monitoring

### Server Health

```bash
# Quick check
curl -s http://mac:7862/health | python3 -m json.tool

# Via MCP (from daemon)
# Tool: mcp_comfybox__server_health

# System stats
# Tool: mcp_comfybox__system_stats
```

### Logs

- **WarmServer:** stdout of the screen session (`screen -r warmserver`)
- **MCP Server:** stderr of the SSH child process (daemon captures this)
- **Daemon MCP:** `journalctl --user -u bree-channels -f | grep mcp`

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Render timeout at 60s | Default tool timeout too low | Set `toolTimeoutMs: 300000` in MCP config |
| SSH connection drops | Network instability | Auto-reconnect handles this; check `screen -ls` on Mac |
| Model fails to load | Insufficient VRAM | Use quantized model or unload other models |
| WarmServer crashes repeatedly | Circuit breaker trips | Check `screen -r warmserver` logs, fix root cause |
| Port 7862 in use | Previous instance didn't clean up | `lsof -ti:7862 | xargs kill` |
| Metal library not found | Build artifact missing | Rebuild with xcodebuild, copy `mlx.metallib` next to binary |
