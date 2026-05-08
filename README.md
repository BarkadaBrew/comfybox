# ComfyBox

> ComfyUI performance. Apple silicon native. Zero Python.

ComfyBox is a native Swift image generation engine for Apple Silicon. It runs diffusion models via MLX — no Python runtime, no node graphs, no ComfyUI dependency. Text-to-image, img2img, ControlNet, inpainting, LoRA, batch generation, and AI upscaling in a single binary.

An LLM orchestrator manages models, LoRAs, and generation parameters via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), enabling AI assistants to generate images without manual configuration.

## Documentation

| Doc | Description |
|-----|-------------|
| [User Guide](docs/user-guide.md) | CLI usage, generation modes, LoRA, batch mode |
| [Architecture](docs/architecture.md) | Engine, orchestrator, and protocol layers |
| [MCP Tool Reference](docs/mcp-reference.md) | 18 MCP tools for AI assistant integration |
| [Deployment](docs/deployment.md) | Serve mode, keepalive, remote SSH bridge |
| [ComfyUI Bridge Spec](docs/comfyui-bridge-protocol-spec.md) | ComfyUI protocol emulation details |
| [SeedVR2 Porting Guide](docs/seedvr2-porting-guide.md) | SeedVR2 upscaler implementation notes |

## Quick Start

```bash
# Generate an image
ComfyBox -p "A mountain landscape at sunset" -o landscape.png

# Img2img transformation
ComfyBox -p "oil painting style" --image photo.jpg --creativity 0.7 -o painting.png

# Batch generation (10 random seeds)
ComfyBox -p "portrait" --auto-seeds 10 -o batch/portrait.png

# Run as warm server
ComfyBox serve -m Tongyi-MAI/Z-Image-Turbo --port 7862

# Start MCP server for AI assistants
ComfyBox mcp --port 7862
```

## System Requirements

| Requirement | Details |
|-------------|---------|
| **macOS** | 14.0+ (Sonoma or later) |
| **Chip** | Apple Silicon (M1, M2, M3, M4) |
| **VRAM** | 4-21 GB depending on model and precision |
| **Disk** | ~6 GB for base model files |
| **Swift** | 5.9+ (building from source only) |

## Installation

### Pre-built Binary

```bash
curl -LO https://github.com/BarkadaBrew/comfybox/releases/download/0.2.3/ComfyBox-0.2.3-macos-arm64.tar.gz
tar -xzf ComfyBox-0.2.3-macos-arm64.tar.gz
cd ComfyBox-0.2.3
sudo ./install.sh
```

### Building from Source

```bash
git clone https://github.com/BarkadaBrew/comfybox.git
cd comfybox
xcodebuild -scheme ComfyBox -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/xcode
```

## Supported Models

| Family | Steps | VRAM (BF16) | Notes |
|--------|-------|-------------|-------|
| Z-Image Turbo | 4-9 | ~7 GB | Default, fastest |
| Z-Image Base | 20-50 | ~7 GB | CFG-guided, higher quality |
| Flux 2 Klein 9B | 20-50 | ~21 GB | High quality, 8-bit available (~12 GB) |
| FIBO 8B | 20+ | ~21 GB | JSON prompts, CC-BY-NC-4.0 |
| Chroma 8.9B | 20+ | ~21 GB | Guidance-free |
| SeedVR2 3B/7B | 1 | 7/16 GB | AI upscaling only |

## Features

### Generation
- **Text-to-image** — prompt to pixels via Diffusion Transformer
- **Image-to-image** — transform existing images with controllable strength
- **ControlNet** — Canny, HED, Depth, Pose, MLSD conditioning
- **Inpainting** — mask-guided region replacement
- **Batch mode** — multi-seed/multi-prompt with checkpoint resume

### Models & LoRA
- **Multi-model** — Z-Image, Flux 2 Klein, FIBO, Chroma families
- **Quantization** — 4-bit and 8-bit weight compression
- **Multi-LoRA** — stack multiple style adapters with per-LoRA scaling
- **Hot swap** — change LoRAs without model reload (~2s)
- **LoRA library** — scan, search, quarantine, compatibility checking

### Upscaling
- **SeedVR2** — AI-powered 2x upscale with tiled VAE for large images
- **ESRGAN** — traditional deterministic upscale with tile support

### Server
- **WarmServer** — HTTP API with warm model pool, hot LoRA swap
- **Model Pool** — multiple models loaded simultaneously, instant switching
- **MCP Server** — 18 tools for AI assistant integration via JSON-RPC 2.0
- **ComfyUI Bridge** — protocol-compatible API for Krita AI Diffusion

### Post-Processing
- **SVG export** — vector conversion via vtracer with style presets
- **Metadata sidecars** — reproducible generation with JSON parameter files
- **Levels adjustment** — post-decode contrast correction
- **Prompt enhancement** — built-in LLM prompt expansion

## Samplers

| Sampler | Speed | Quality | Best For |
|---------|-------|---------|----------|
| `euler` | Fastest | Good | Default, distilled models |
| `heun` | 2x slower | Higher | Fine detail |
| `dpmpp-2m` | Medium | High | Base models with many steps |
| `dpmpp-2s-a` | Medium | High | Stochastic variation |
| `res_2s` | Medium | High | Balanced |
| `deis` | Medium | High | Smooth outputs |
| `ddim` | Medium | Good | Animation, deterministic |

**Sigma schedules:** `flow` (default), `karras`, `exponential`, `beta`, `beta57`

## Examples

```bash
# Basic generation with seed
ComfyBox -p "a cute cat sitting on a windowsill" --seed 42 -o cat.png

# Portrait with specific resolution
ComfyBox -p "portrait of a woman in renaissance style" -W 768 -H 1152 -o portrait.png

# Quantized model (low VRAM)
ComfyBox -p "a futuristic city at night" -m mzbac/Z-Image-Turbo-8bit -o city.png

# Multiple LoRAs stacked
ComfyBox -p "a fantasy portrait" \
  --lora style.safetensors=0.8 \
  --lora detail.safetensors=0.6 \
  -o combined.png

# SVG logo
ComfyBox -p "minimalist coffee cup logo, flat design" --svg --svg-preset logo -o logo.png

# ControlNet with pose
ComfyBox control -p "a woman on a beach" -c pose.jpg \
  --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  -o beach.png

# SeedVR2 upscale
ComfyBox upscale -i photo.jpg -w ~/Models/seedvr2-3b -r 2048 -o photo-2x.png

# Heun sampler with Karras schedule
ComfyBox -p "detailed landscape" --scheduler heun --sigma-schedule karras -s 20 -o landscape.png
```

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple's ML framework for Apple Silicon
- [swift-transformers](https://github.com/huggingface/swift-transformers) — Tokenizer support
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) — CLI argument parsing

## License

MIT License
