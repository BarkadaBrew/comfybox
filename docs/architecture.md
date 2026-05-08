# ComfyBox Architecture

> ComfyUI performance. Apple silicon native. Zero Python.

ComfyBox is an LLM-enabled image generation engine for Apple Silicon. It runs Z-Image, Flux, Klein, FIBO, Chroma, and SeedVR2 models natively in Swift via MLX — no Python, no ComfyUI runtime, no node graphs. An LLM orchestrator translates user intent into generation parameters, manages models, and self-heals on failure.

## Three Layers

### 1. Engine (Swift/MLX)

The core model runner. Loads safetensors weights, runs inference on Metal, outputs images.

```
Text Prompt → TextEncoder (Qwen) → Embeddings
                                      ↓
Random Noise Latents ──→ DiT Transformer (denoising loop) ──→ Refined Latents
                              ↑                                     ↓
                         FlowMatch Scheduler                  VAE Decoder → RGB Image
```

**Supported pipelines:**
- **txt2img** — Text-to-image generation
- **img2img** — Image-to-image with controllable strength
- **ControlNet** — Canny, HED, Depth, Pose, MLSD conditioning
- **Inpainting** — Mask-guided region replacement
- **Upscale** — SeedVR2 (AI upscale) and ESRGAN (traditional upscale)
- **Quantization** — 4-bit and 8-bit weight compression

**Key components:**

| Module | Purpose |
|--------|---------|
| `ZImagePipeline` | Text-to-image orchestration, LoRA application, denoising loop |
| `ZImageControlPipeline` | ControlNet conditioning and inpainting |
| `FlowMatchScheduler` | Flow Matching Euler scheduler with dynamic shifting |
| `QwenTextEncoder` | Qwen-based transformer (encoder + prompt enhancement LLM) |
| `ZImageTransformer2DModel` | Diffusion Transformer with refiner streams |
| `AutoencoderKL` | VAE encode/decode (latent ↔ pixel space) |
| `LoRAApplicator` | Runtime LoRA weight injection (standard, LoKr, quantized) |
| `SeedVR2Pipeline` | 2x AI upscaling with tiled VAE for large images |
| `BatchRunner` | Sequential multi-seed/multi-prompt batch generation |

### 2. Orchestrator (LLM)

The product layer. An LLM translates natural language into generation parameters, selects models and LoRAs, diagnoses failures, and retries with corrected settings.

**Key concepts:**
- **Creative Spectrum** — Deterministic (locked params) ↔ Fully Creative (AI explores). Fields marked "locked" or "AI-discretion" in the prompt JSON.
- **JSON Prompt** — Universal structured format for any model. Contains prompt, parameters, style, LoRA config — the single source of truth for a render.
- **Self-healing** — On failure (OOM, bad LoRA, timeout), the orchestrator diagnoses the error, adjusts parameters, and retries without user intervention.

### 3. Protocol (ComfyUI Bridge)

A ComfyUI-compatible API on port 7870. Any ComfyUI client (Krita AI Diffusion, custom UIs) connects without modification.

```
Krita AI Diffusion → HTTP :7870 → ComfyUI Bridge → WarmServer → Engine
```

**Emulated endpoints:**
- `GET /object_info` — Node registry (emulated)
- `POST /prompt` — Submit generation workflow
- `GET /ws` — WebSocket progress notifications
- `GET /view` — Retrieve generated images

> The bridge emulates the ComfyUI protocol only — zero actual ComfyUI runtime.

## Runtime Architecture

### WarmServer (HTTP API)

The WarmServer keeps models loaded in GPU memory between requests. Cold start loads the model once; subsequent renders skip loading entirely.

```bash
ComfyBox serve -m Tongyi-MAI/Z-Image-Turbo --port 7862 --host 0.0.0.0
```

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/generate` | POST | Submit render (txt2img or img2img) |
| `/v1/lora/swap` | POST | Hot-swap LoRA weights without restart |
| `/health` | GET | Server status, loaded model, memory |
| `/v1/shutdown` | POST | Graceful shutdown |

**Key features:**
- Warm model pool — multiple models loaded simultaneously
- Hot LoRA swap — change styles without model reload (~2s vs ~30s)
- Request queue — sequential GPU access prevents OOM
- Allowed output directory — sandboxed file writes

### Model Pool

Multiple models can be loaded simultaneously and switched instantly:

```
load_model("z-image-turbo-bf16")     # Load Z-Image (~7GB)
load_model("klein-9b-q8")            # Load Klein (~12GB)
switch_model("klein-9b-q8")          # Instant switch, no reload
unload_model("z-image-turbo-bf16")   # Free VRAM
```

The pool tracks VRAM usage per model and last-used timestamps for eviction decisions.

### MCP Server (Model Context Protocol)

Bridges the WarmServer to AI assistants via JSON-RPC 2.0 over stdio:

```bash
ComfyBox mcp --port 7862
```

18 tools exposed — see [MCP Tool Reference](mcp-reference.md) for full details.

**Remote access via SSH:**
```bash
ssh user@mac-host "cd /path/to/comfybox && .build/release/ComfyBox mcp --port 7862"
```

The daemon spawns this as a child process, reads JSON-RPC from stdout, writes to stdin. All 18 tools appear as `mcp_comfybox__<tool_name>` in the daemon's tool registry.

### ComfyUI Bridge

Translates ComfyUI workflow JSON into native WarmServer API calls:

```
Krita sends ComfyUI workflow → Bridge parses nodes → Maps to /v1/generate params → Returns result
```

Runs alongside the WarmServer on port 7870. Shares the same warm model — no duplicate loading.

## Supported Models

| Family | Models | Steps | Guidance | Notes |
|--------|--------|-------|----------|-------|
| **Z-Image** | Turbo (distilled), Base (CFG) | 4-9 / 20-50 | 0.0 / 3.5-7.0 | Default. Fastest distilled model |
| **Flux 2 Klein** | 9B, 9B-q4, 9B-q8 | 20-50 | 3.5+ | High quality, CFG-guided |
| **FIBO** | 8B, 8B-q4 | 20+ | 3.5+ | JSON-structured prompts, CC-BY-NC-4.0 |
| **Chroma** | 8.9B | 20+ | N/A | Guidance-free architecture |
| **SeedVR2** | 3B, 7B | 1 | N/A | AI upscaling only (2x) |

## Memory Footprint

| Model | Precision | Approx. VRAM |
|-------|-----------|-------------|
| Z-Image Turbo | BF16 | ~7 GB |
| Z-Image Turbo | 8-bit | ~4 GB |
| Klein 9B | BF16 | ~21 GB |
| Klein 9B | 8-bit | ~12 GB |
| FIBO 8B | 4-bit | ~8 GB |
| SeedVR2 3B | BF16 | ~7 GB |
| SeedVR2 7B | BF16 | ~16 GB |

Multiple models can coexist. Example: Z-Image Turbo 8-bit (~4GB) + SeedVR2 3B (~7GB) = ~11GB total.

## File Layout

```
Sources/
├── ComfyBox/              # CLI entry point
│   └── main.swift         # Argument parsing, subcommand dispatch
├── ZImage/
│   ├── Pipeline/          # Generation pipelines
│   │   ├── ZImagePipeline.swift
│   │   ├── ZImageControlPipeline.swift
│   │   ├── SeedVR2Pipeline.swift
│   │   ├── BatchRunner.swift
│   │   └── FlowMatchScheduler.swift
│   ├── Model/             # Neural network layers
│   │   ├── TextEncoder/
│   │   ├── Transformer/
│   │   └── VAE/
│   ├── Server/            # HTTP server mode
│   │   ├── WarmServer.swift
│   │   └── ModelPool.swift
│   ├── MCP/               # Model Context Protocol
│   │   ├── MCPServer.swift
│   │   ├── MCPToolRegistry.swift
│   │   ├── MCPToolExecutor.swift
│   │   └── WarmServerClient.swift
│   ├── Bridge/            # ComfyUI protocol bridge
│   ├── LoRA/              # LoRA weight management
│   ├── Weights/           # Model downloading and loading
│   └── Quantization/      # 4-bit and 8-bit compression
Tests/
├── ZImageTests/           # Unit tests
├── ZImageIntegrationTests/# Tests requiring model weights
└── ZImageE2ETests/        # End-to-end CLI tests
```

## Build

```bash
# Release build (recommended)
xcodebuild -scheme ComfyBox -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/xcode

# Tests
xcodebuild test -scheme comfybox-Package \
  -destination 'platform=macOS' -enableCodeCoverage NO

# The binary + Metal library end up in:
# .build/xcode/Build/Products/Release/ComfyBox
# .build/xcode/Build/Products/Release/mlx.metallib
```

## Requirements

- macOS 14.0+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Swift 5.9+
- ~6-21 GB VRAM depending on model
