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

Provenance fields (WP-E10, `docs/FDD-krea2-raw-recipe.md` §3.10):

| key | meaning |
|---|---|
| `build_sha` | git short sha stamped into the binary at build time (`-dirty` when the worktree had uncommitted changes); `"unknown"` for a build that did not run `scripts/gen-build-info.sh`. The deploy smoke (`scripts/deploy-serve.sh`, step e) fails unless it matches the sha being deployed — a clobbered or wrong-branch binary is detectable from outside. |
| `model_alias` | the declared alias of the active Krea 2 model (`krea2-raw`, `kroma-v0.2-turbo`) beside the resolved directory in `model`; null for other families |
| `model_variant` | `raw` / `turbo` for the Krea 2 family — the physical variant read off the loaded checkpoint |
| `last_recipe` | the `applied` record of the last successful Krea 2 render (same value the `/v1/generate` response, the async job status and the PNG's EXIF `UserComment` carry); null until one has run |

**Stamping `build_sha`.** `Sources/ZImage/Support/BuildInfo.swift` is committed with the placeholder `"unknown"`. `scripts/gen-build-info.sh` rewrites its one marked line with `git rev-parse --short HEAD` (plus `-dirty`); `scripts/gen-build-info.sh --reset` restores the placeholder. `deploy-serve.sh` stamps before `swift build -c release` and resets from its EXIT trap, so the tree is never left dirty and a real sha is never committed. For a hand build: `scripts/gen-build-info.sh && swift build -c release --product ComfyBox; scripts/gen-build-info.sh --reset`.

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

## CI: Nightly Integration Runner

Two test tiers exist (#332, #137):

- **Unit tier** (`.github/workflows/ci.yml`) — `ZImageTests`, `ComfyBoxDesktopTests`, `ComfyBoxCatalogTests`. Runs on every push/PR on GitHub-hosted `macos-latest`. No model weights needed.
- **Integration tier** (`.github/workflows/nightly-integration.yml`) — `ZImageIntegrationTests`, `ZImageE2ETests`. Needs real model weights, a GPU, and (for a few tests) network access to Hugging Face. Runs nightly on a cron plus `workflow_dispatch`, and **only** on a self-hosted runner Todd registers and opts in explicitly. Until that setup is done, the `integration` job shows as **skipped** on every run — this is expected, not a failure.

### Why this can't run on GitHub-hosted runners

GitHub-hosted macOS runners have no Apple-silicon GPU and no local copy of the model weights (`mzbac/z-image-turbo-8bit`, the LTX-2 `pinkcherry-v18-distill06-int8` checkpoint, the Gemma-3 text encoder, LoRA fixtures). `intent.md` is explicit that agents run the unit tier only; integration/E2E need Todd's Mac.

### Registering the self-hosted runner

**This is Todd's call — nothing here registers or configures a runner automatically.** Steps, when you're ready:

1. Pick a Mac with the weights already present (or budget the disk/network to fetch them — see "Required weights" below). **Caution:** if this is the production Mac (10.0.100.134) that also runs the live ComfyBox engine, LM Studio, and the gallery service (see `.superpowers/burndown/AGENT-RULES.md`), a nightly integration run will contend for the same GPU/model memory as production traffic. Consider a dedicated/secondary Mac, or schedule the cron for a window (edit the `schedule:` cron in `nightly-integration.yml`) when production load is lowest, and be ready to cancel a run manually if it collides with live traffic.
2. In the repo (Settings → Actions → Runners → New self-hosted runner), follow GitHub's generated `./config.sh` instructions for macOS/ARM64.
3. When prompted for labels, add (in addition to the defaults GitHub adds): `macOS`, `comfybox-weights`. The workflow targets `runs-on: [self-hosted, macOS, comfybox-weights]` — the runner must carry all three labels (`self-hosted` is implicit).
4. Install the runner as a persistent service (`./svc.sh install && ./svc.sh start`) so it survives reboots, matching this repo's launchd conventions (see "Keepalive with Screen" above).
5. Once the runner is online (Settings → Actions → Runners shows it "Idle"), set the repository variable that arms the nightly job: Settings → Secrets and variables → Actions → Variables → New repository variable → `COMFYBOX_WEIGHTS_RUNNER_AVAILABLE` = `true`. (The job is gated on this variable rather than probing for the runner via the API, since listing self-hosted runners needs the "administration" token permission, which the workflow's `GITHUB_TOKEN` cannot hold.)

### Required weights and env on the runner

| Env var | Default | Used by |
|---|---|---|
| `LTX2_TEST_WEIGHTS_DIR` | `/Volumes/Bolt/Models/pinkcherry-v18-distill06-int8` (falls back to `/tmp/ltx2-local-weights` if present) | `LTX2PreemptionResumeTests` |
| `ZIMAGE_TEST_LORA_PATH` | `ostris/z_image_turbo_childrens_drawings` (downloaded from Hugging Face) | `LoRAIntegrationTests` |

Not overridable via env — must exist at these fixed paths on the runner:

- Gemma-3 text encoder: `/Users/toddwalderman/LocalModels/gemma-3-12b-heretic-q8` (`LTX2PreemptionResumeTests`)
- LTX-2 distilled checkpoint: `/Volumes/Bolt/Models/ltx2-distilled` (`LTX2IntegrationTest`, `LTX2MultiKeyframeSpike`)

`PipelineIntegrationTests`, `ControlNetIntegrationTests`, `LoRAIntegrationTests`, `PerformanceTests` pull `mzbac/z-image-turbo-8bit` (~7.5GB) through the Hugging Face Hub client on first run and cache it locally — the runner needs network egress to Hugging Face the first time (or a pre-warmed cache) and ~10GB of free disk for it.

`ZImageE2ETests` builds and runs the CLI binary directly (`swift build -c release --product ComfyBox` under the hood) — it does not need a running warm server or any port.

**The `CI` environment variable must NOT be set when the tests run.** Most integration/E2E tests call `skipIfNoGPU()`, which does `throw XCTSkip(...)` whenever `ProcessInfo.processInfo.environment["CI"] != nil` — true for any value, including an empty string. GitHub Actions sets `CI=true` by default on every runner, hosted and self-hosted, so `nightly-integration.yml` strips it with `env -u CI` immediately before invoking `xcodebuild`. If you ever run this suite by hand on the runner outside the workflow, do the same (`env -u CI xcodebuild test ...`) or every GPU test will silently report as skipped rather than actually running.

### Known state of the integration/E2E suites (as of #332/#137)

- `LoRAIntegrationTests.testLoRAConfigurationScaleClamped` is quarantined (`XCTSkip`, not deleted) — it asserts a clamping contract the engine deliberately abandoned. Someone still needs to decide whether to delete it or rewrite it against the current no-clamp behavior.
- `PipelineIntegrationTests.testDeterministicSeed` and `CLIEndToEndTests.testControlNetWithCanny`/`testControlNetWithHed` were red during the 2026-08-31 gate run but **not** quarantined here — the root cause (real nondeterminism vs. orphaned-process GPU contention) was never established (see #332). Quarantining them without evidence would hide a possible real bug; if the first real nightly run reproduces the failure, `#332`'s comment thread is the place to classify it before deciding whether to quarantine or fix.

### What happens on failure

The `nightly-integration.yml` job uploads the `.xcresult` bundle and full test log as a workflow artifact (14-day retention) and appends a comment to **#332** listing the failing test names, using `actions/upload-artifact` and `actions/github-script` — both need only the default `GITHUB_TOKEN`, no extra secrets.

### Verifying the setup

After registering the runner and setting the repo variable: Actions tab → "Nightly Integration" → "Run workflow" (`workflow_dispatch`). Confirm the `integration` job actually executes (not skipped) on the self-hosted runner, and — if you want to see the failure path exercised — check that a deliberately broken run produces both the artifact and the comment on #332.
