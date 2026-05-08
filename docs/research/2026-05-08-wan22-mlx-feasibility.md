# Wan 2.2 Research: ComfyBox (zimage.swift) Port Feasibility

**Date:** 2026-05-08
**Issue:** BarkadaBrew/zimage.swift #69
**Researcher:** Claude (Opus 4.6)

---

## 1. Architecture Overview

### What is Wan 2.2?

Wan 2.2 is Alibaba's open-source video generation model family, released in mid-2025 with MoE (Mixture of Experts) enhancements over the dense Wan 2.1 architecture. It is currently the primary video generation model used by coffeeshop-server via Replicate's API.

### Core Components

| Component | Model | Parameters | Size (FP16) | Purpose |
|-----------|-------|------------|-------------|---------|
| Text Encoder | UMT5-XXL | ~11B | ~22 GB | Text prompt encoding, 512 token context, 4096-dim embeddings |
| Diffusion Transformer (Expert 1) | High-noise expert | ~14B | ~28 GB | Early denoising: layout, composition, motion patterns |
| Diffusion Transformer (Expert 2) | Low-noise expert | ~14B | ~28 GB | Late denoising: detail refinement, lighting, color |
| VAE | Wan-VAE (3D Causal) | ~150M (est.) | ~300 MB | Spatio-temporal encoding/decoding, 4x16x16 compression |
| Scheduler | Flow-matching | N/A | N/A | Diffusion sampling |

**Total model size:** ~126 GB on disk (FP16, full repository including both experts)
**Active parameters per step:** ~14B (only one expert active at a time)
**Active memory footprint (FP16):** ~50 GB (one expert + text encoder + VAE + activations)

### Model Variants

| Variant | Type | Parameters | Resolution | Notes |
|---------|------|------------|------------|-------|
| T2V-A14B | Text-to-Video | 27B total / 14B active | 720p/24fps | MoE, two experts |
| I2V-A14B | Image-to-Video | 27B total / 14B active | 480p-720p/16fps | MoE, what coffeeshop-server uses |
| TI2V-5B | Text+Image-to-Video | 5B dense | 480p-720p | Smaller, single dense model |
| T2V-1.3B | Text-to-Video | 1.3B dense | 480p | Lightweight variant |

### How It Differs from Flux/Z-Image

| Aspect | Flux / Z-Image (image) | Wan 2.2 (video) |
|--------|------------------------|-----------------|
| Output | Single 2D image | Multi-frame 3D tensor (81+ frames) |
| VAE | 2D spatial only | 3D causal VAE (spatial + temporal) |
| Attention | 2D self-attention | 3D spatiotemporal attention |
| Text encoder | CLIP + T5 | UMT5-XXL (multilingual) |
| Architecture | Dense DiT | MoE DiT (2 experts, timestep-routed) |
| Memory | ~8-12 GB active | ~50+ GB active (FP16) |
| Compute | Seconds per image | Minutes per video clip |

---

## 2. MLX Availability Status

### 2.1 mflux (Reference Port Source)

**Status: VAE only -- no full Wan 2.2 video pipeline.**

mflux (filipstrand/mflux, installed on the Mac at 10.0.100.134) has a complete MLX implementation of the **Wan 2.2 VAE** -- but only as a sub-component of the FIBO image model. FIBO uses the Wan 2.2 3D Causal VAE for its encoder/decoder while using its own DiT transformer and SmolLM3-3B text encoder.

The Wan 2.2 VAE implementation in mflux includes:
- `Wan2_2_VAE` -- top-level encoder/decoder with patchify/unpatchify
- `Wan2_2_Encoder3d` -- 3D convolutional encoder with down blocks
- `Wan2_2_Decoder3d` -- 3D convolutional decoder with up blocks
- Full set of common blocks: `CausalConv3d`, `RMSNorm`, `AttentionBlock`, `ResidualBlock`, `MidBlock`, `Resample`

**What's missing from mflux for full Wan 2.2:**
- The Wan 2.2 diffusion transformer (DiT with MoE routing)
- UMT5-XXL text encoder
- The flow-matching scheduler
- I2V conditioning (CLIP image encoder for reference frame)
- No CLI commands for Wan video generation (only image models listed)

mflux CLI commands on the Mac: `mflux-generate`, `mflux-generate-fibo`, `mflux-generate-z-image-turbo`, `mflux-upscale-seedvr2`, etc. -- **no Wan video commands exist**.

### 2.2 Wan2.2-mlx (osama-ata)

**Status: Full MLX port exists, community project.**

Repository: [github.com/osama-ata/Wan2.2-mlx](https://github.com/osama-ata/Wan2.2-mlx)

This is a pure MLX-native port of Wan 2.2 that has fully migrated all model code from PyTorch to Apple MLX. Key details:
- All PyTorch code and dependencies removed
- Uses `uv` and `pyproject.toml` for dependency management
- Targets macOS 14+ on M1/M2/M3/M4 Apple Silicon
- No multi-GPU / distributed inference support (runs on unified memory)
- Model weights from HuggingFace/ModelScope

This is the most complete MLX reference for a ComfyBox port.

### 2.3 mlx-video (Blaizzy)

**Status: Actively maintained MLX video generation package.**

Repository: [github.com/Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video)

A higher-level package for video generation on Apple Silicon via MLX. Supports:
- Wan 2.2 T2V-14B, TI2V-5B, I2V-14B
- Wan 2.1 models
- LTX-2 (Lightricks)
- LoRA support (including Wan2.2-Lightning for 4-step generation)
- Unified inference and finetuning interface
- Install via: `uv pip install git+https://github.com/Blaizzy/mlx-video.git`

This is likely the **most production-ready** MLX video generation package.

### 2.4 Wan2GP (deepbeepmeep)

Repository: [github.com/deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP)

A "GPU Poor" optimized implementation supporting Wan 2.1/2.2, HunyuanVideo, LTX Video. Focuses on memory optimization for lower-VRAM GPUs. Primarily PyTorch/CUDA, not MLX-native.

---

## 3. Memory Requirements

### FP16 (Full Precision)

| Component | Memory |
|-----------|--------|
| UMT5-XXL text encoder | ~22 GB |
| Transformer (one active expert) | ~28 GB |
| Wan-VAE | ~0.3 GB |
| Activations + intermediates (81 frames @ 480p) | ~8-15 GB |
| **Total (FP16)** | **~50-65 GB** |

### Quantized (INT8 / INT4)

| Quantization | Estimated Memory | Quality Impact |
|--------------|------------------|----------------|
| FP8 (transformer only) | ~35-40 GB | Minimal |
| INT8 (all) | ~25-35 GB | Minor |
| INT4 (transformer) + FP16 (VAE/encoder) | ~18-25 GB | Moderate |
| GGUF Q4_K_S | ~15-20 GB | Noticeable but usable |

### M3 Max 128GB Assessment

**Verdict: Easily fits, even at FP16.**

The M3 Max with 128 GB unified memory has ~4x the memory needed for the full FP16 model. This means:
- Can run the full 27B model (both experts loaded) simultaneously
- Room for multiple resolution outputs without memory pressure
- No need for aggressive quantization -- FP16 quality with full memory headroom
- Could potentially keep the model warm in memory alongside image generation models

Community reports confirm Wan 2.2 running on M3 Max 36GB (tight) and M2 Max 64GB (comfortable). At 128GB, memory is a non-issue.

---

## 4. Current coffeeshop-server Video Pipeline

The daemon currently routes video generation through two paths:

### Primary: Image Service (Mac LAN)
```
Telegram request -> daemon -> image service (10.0.100.134:7861) -> Replicate API -> .mp4
```
The image service on the Mac acts as a proxy to Replicate, handling queue management, asset registration, and gallery storage.

### Fallback: Direct Replicate
```
Telegram request -> daemon -> Replicate API directly -> download .mp4 -> vault gallery
```

### Models Used
- **I2V**: `wan-video/wan-2.2-i2v-a14b` (81 frames, 16fps, ~5 sec clips)
- **T2V primary**: `lightricks/ltx-2.3-fast` (up to 20 sec, up to 1080p)
- **T2V fallback**: `wan-video/wan-2.2-t2v-fast` (when LTX content-filters)

### Cost
Each Replicate I2V generation costs ~$0.05-0.15 depending on resolution. At 5-10 videos/day, that's $0.25-1.50/day in API costs that a local port would eliminate.

---

## 5. Feasibility Assessment for ComfyBox Port

### What Would Need to Be Built

| Component | Effort | Source Available |
|-----------|--------|-----------------|
| Wan 2.2 VAE (encode + decode) | Low | **Already in mflux** (MLX native, battle-tested via FIBO) |
| UMT5-XXL text encoder | Medium | Wan2.2-mlx and mlx-video have implementations |
| Diffusion Transformer (DiT) | High | Wan2.2-mlx has full MLX port |
| MoE routing (2 experts) | Medium | Wan2.2-mlx has implementation |
| Flow-matching scheduler | Low-Medium | Standard, well-documented |
| I2V conditioning (CLIP image encoder) | Medium | Wan2.2-mlx has implementation |
| Integration with ComfyBox pipeline | Medium | Needs ModelBackend/GenerationPipeline wiring |
| Swift/MLX bridge (Python -> Swift) | High | Novel work -- no existing reference |

### Key Challenge: Python MLX to Swift MLX

All existing MLX Wan 2.2 implementations are in **Python** using the `mlx` Python package. ComfyBox (zimage.swift) is **Swift** using the Swift MLX bindings. The port requires translating Python MLX model definitions to Swift MLX -- the same translation pattern used for Flux, Z-Image Turbo, FIBO, and SeedVR2 in ComfyBox.

This is the dominant effort item. The math operations are identical (MLX ops are the same in both languages), but the model definitions, weight loading, and tensor manipulation code must all be rewritten in Swift.

### Estimated Effort

| Phase | Scope | Estimate |
|-------|-------|----------|
| Phase 1: VAE port | Port Wan2_2_VAE from mflux Python to Swift (or reuse if FIBO VAE is already in ComfyBox) | 1-2 days |
| Phase 2: Text encoder | Port UMT5-XXL to Swift MLX, weight loading | 3-5 days |
| Phase 3: DiT + MoE | Port the 14B transformer with MoE routing | 5-8 days |
| Phase 4: Scheduler + I2V | Flow-matching sampler, I2V conditioning | 2-3 days |
| Phase 5: Integration | ComfyBox pipeline, CLI, warm server, progress reporting | 3-5 days |
| Phase 6: Optimization | Memory management, quantization support, tiling | 3-5 days |
| **Total** | | **17-28 days** |

### Value Assessment

| Factor | Score | Notes |
|--------|-------|-------|
| Eliminates Replicate cost | High | $10-45/month in API fees |
| Latency improvement | Medium | Local may be slower than cloud GPU but no network round-trip or queue wait |
| Privacy | High | No frames leaving the local network |
| Quality control | High | Can run at FP16, adjust steps, use local LoRAs |
| LoRA ecosystem | High | Wan 2.2 has active LoRA community (Lightning, character LoRAs) |
| Portfolio value | High | First native Swift/MLX video generation engine |

---

## 6. Recommended Approach

### Strategy: Port from Wan2.2-mlx + reuse mflux VAE

1. **Start with the mflux Wan 2.2 VAE** -- it's already MLX-native, proven via FIBO, and likely already partially ported to Swift in ComfyBox if FIBO support exists. This is your fastest win: get video encoding/decoding working immediately.

2. **Use Wan2.2-mlx (osama-ata) as the reference** for the diffusion transformer and MoE routing. It's a clean, dependency-free MLX implementation that maps directly to Swift MLX.

3. **Use mlx-video (Blaizzy) as a second reference** for architectural decisions, LoRA loading, and optimization patterns. It's more actively maintained and has LoRA support.

4. **Start with the 5B dense model (TI2V-5B)**, not the 14B MoE. Rationale:
   - 5B dense is architecturally simpler (no MoE routing)
   - Fits easily in memory even at FP16 (~15 GB)
   - Validates the full pipeline (text encoder -> transformer -> VAE -> video)
   - Can upgrade to 14B MoE after the pipeline is proven
   - The 5B model does both T2V and I2V (TI2V = Text+Image-to-Video)

5. **Wire into the image service protocol** so it integrates seamlessly with the existing daemon pipeline -- the daemon already tries image service first, falls back to Replicate. Making ComfyBox respond to the same `/v1/generate` calls with `mediaKind: 'video'` means zero daemon changes.

### Do NOT:
- Port from scratch -- too much work, too many bugs
- Port the full 27B (both experts) first -- get single-expert working, then add MoE
- Try to use PyTorch/MPS -- the MLX path is better optimized for Apple Silicon

---

## 7. Priority Recommendation

### Verdict: **Wait -- queue for Phase 3 or later**

**Rationale:**

1. **The Replicate pipeline works today.** I2V quality is good, costs are low ($0.25-1.50/day), and latency (~2-5 minutes per video including queue) is acceptable for the Telegram use case.

2. **ComfyBox has higher-priority image generation work.** The PRD reviews identify Phase 1-2 items (model management, batch rendering, LoRA support, Krita integration) that benefit the daily-driver image pipeline. Video is a secondary use case (~5-10 videos/day vs. 20-30 images/day).

3. **The MLX ecosystem is maturing fast.** mlx-video is actively adding models and optimizations. Waiting 2-3 months means a more stable reference implementation to port from, possibly with quantization support and memory optimizations already solved.

4. **The mflux VAE is a free head start.** If ComfyBox already has the FIBO pipeline (which uses Wan 2.2 VAE), you already have ~20% of the work done. This head start doesn't expire.

**Recommended timeline:**
- **Now:** Track mlx-video and Wan2.2-mlx development. File the issue with this research.
- **Phase 2-3 of ComfyBox:** Start with TI2V-5B port as a proof of concept.
- **Phase 4+:** Full 14B MoE support with LoRA loading and warm server.

**Trigger to move earlier:** If Replicate increases pricing, adds content restrictions that block the I2V use case, or if mlx-video publishes Swift bindings.

---

## Sources

- [Wan-Video/Wan2.2 (official repo)](https://github.com/Wan-Video/Wan2.2)
- [Wan-AI/Wan2.2-I2V-A14B (HuggingFace)](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [Wan-AI/Wan2.2-T2V-A14B (HuggingFace)](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
- [osama-ata/Wan2.2-mlx (full MLX port)](https://github.com/osama-ata/Wan2.2-mlx)
- [Blaizzy/mlx-video (MLX video package)](https://github.com/Blaizzy/mlx-video)
- [filipstrand/mflux (MLX image models, has Wan 2.2 VAE)](https://github.com/filipstrand/mflux)
- [deepbeepmeep/Wan2GP (GPU-poor optimizer)](https://github.com/deepbeepmeep/Wan2GP)
- [wan22.io (official Wan 2.2 site)](https://wan22.io/)
- [Local AI Video Generation Guide (2026)](https://localaimaster.com/blog/local-ai-video-generation)
- [Vast.ai Wan 2.2 Explained](https://vast.ai/article/wan-2-2-explained-new-approach-ai-video-generation)
