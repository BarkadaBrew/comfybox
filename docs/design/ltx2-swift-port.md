# LTX-2 Swift/MLX Port — Design Plan

**Date:** 2026-05-09
**Author:** Bree (BaristaBree)
**Status:** Approved for implementation
**Target:** ComfyBox (zimage.swift)
**Reference:** [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) (Python MLX reference)

---

## 1. Overview

Port LTX-Video 2.3 (Lightricks' 22B DiT video generation model) to native Swift/MLX for ComfyBox. This adds text-to-video (T2V) and image-to-video (I2V) generation to ComfyBox's existing image generation and upscaling capabilities, running entirely on Apple Silicon with zero Python dependencies.

**Why LTX-2:**
- First open model with synchronized audio+video generation
- Up to 4K native resolution, 50fps, 10-second clips
- 22B params fits in M3 Max 128GB (~30-35GB Q4)
- Runs local — no cloud API costs (replaces Replicate Wan 2.2 for many use cases)

**Why Swift (not Python wrapper):**
- ComfyBox tagline: "Zero Python"
- Single binary, warm server, no conda/venv
- Direct Metal GPU access via MLX-Swift
- ComfyUI bridge integration — video gen through existing protocol

## 2. Architecture Summary

### LTX-2.3 Components

| Component | Params | Memory (BF16) | Memory (Q4) |
|-----------|--------|---------------|-------------|
| DiT Transformer | ~22B | ~44GB | ~11GB |
| Gemma 3 12B Text Encoder | ~12B | ~24GB | ~6GB |
| 1D Connector (8 layers) | ~100M | ~200MB | ~50MB |
| Video VAE (encoder+decoder) | ~300M | ~600MB | ~600MB |
| Audio VAE + Vocoder | ~70M | ~140MB | ~140MB |
| Latent Upsampler | ~200M | ~400MB | ~400MB |
| **Total** | **~35B** | **~69GB** | **~18GB** |

### Key Dimensions

- **Inner dim:** 4096 (video), 2048 (audio)
- **Attention heads:** 32 (video), 32 (audio), head_dim 128/64
- **Transformer blocks:** 48 (uniform)
- **Latent channels:** 128 (video), 8→128 (audio, after patchify)
- **VAE compression:** 32x spatial, 8x temporal
- **RoPE:** 3D (time, height, width), SPLIT mode for v2.3

### Pipeline Flow (T2V)

```
Prompt → Gemma 3 12B → Feature Extractor → 1D Connector
                                               ↓
                              video_embeddings (B, seq, 4096)
                              audio_embeddings (B, seq, 2048)
                                               ↓
Random noise (B, 128, F/8, H/32, W/32) → Patchify → 48x DiT Blocks → Unpatchify
                                          ↕ AdaLN timestep conditioning
                                          ↕ Cross-attn (text)
                                          ↕ Cross-modal (A2V / V2A)
                                               ↓
                              Video latents → VAE Decoder → RGB frames
                              Audio latents → Audio VAE → Vocoder → WAV
```

### Two-Stage Pipeline (Distilled)

```
Stage 1: Generate at half resolution (8 fixed steps)
    ↓
LatentUpsampler: 2x spatial upscale in latent space
    ↓
Stage 2: Refine at full resolution (4 fixed steps)
    ↓
VAE Decode → frames
```

## 3. Reuse from Existing Codebase

### Direct Reuse (no changes)

| Component | Source File | LTX-2 Usage |
|-----------|-----------|-------------|
| `SeedVR2RMSNorm` | `Upscale/SeedVR2/Common/SeedVR2RMSNorm.swift` | All normalization layers |
| `FlowMatchEulerScheduler` | `Pipeline/Scheduler/FlowMatchEulerScheduler.swift` | Primary scheduler |
| `SafeTensorsReader` | `Weights/SafeTensorsReader.swift` | Weight file loading |
| `WeightsLoader` | `Weights/WeightsLoader.swift` | Weight loading orchestrator |

### Adapt (minor changes)

| Component | Source File | Changes Needed |
|-----------|-----------|----------------|
| `CausalConv3d` | `Upscale/SeedVR2/Common/CausalConv3d.swift` | Verify padding mode matches LTX-2 (frame replication vs zero-pad) |
| `SeedVR2RoPE` | `Upscale/SeedVR2/Transformer/SeedVR2RoPE.swift` | Different freq computation, SPLIT vs INTERLEAVED mode |
| `SeedVR2VAE` (encoder) | `Upscale/SeedVR2/VAE/SeedVR2Encoder3D.swift` | 128 latent channels (vs 16), different block config |
| `SeedVR2VAE` (decoder) | `Upscale/SeedVR2/VAE/SeedVR2Decoder3D.swift` | Same — channel count and block config |
| `SeedVR2AdaModulation` | `Upscale/SeedVR2/Transformer/SeedVR2AdaModulation.swift` | 6 or 9 params (vs SeedVR2's pattern) |
| `SeedVR2PatchIn/Out` | `Upscale/SeedVR2/Transformer/` | 128→4096 projection, different layout |

### New Components Required

| Component | Complexity | Notes |
|-----------|-----------|-------|
| GEGLU FFN | Low | tanh-GELU gating (vs SwiGLU in SeedVR2) |
| QK-Norm Attention | Medium | RMSNorm on Q,K before attention, per-head gating (v2.3) |
| AdaLayerNormSingle | Medium | Sinusoidal timestep → MLP → 6/9 scale-shift-gate params |
| Cross-Modal Attention | Medium | Bidirectional A2V/V2A with separate gating |
| Gemma 3 Text Encoder | High | 12B model, tokenizer, feature extraction, connector |
| LatentUpsampler | Medium | Conv3d + ResBlocks + PixelShuffle for 2-stage pipeline |
| res_2s Sampler | Medium | Rosenbrock-Runge-Kutta integrator + SDE noise injection |
| I2V Conditioning | Medium | Denoise mask, per-token timesteps, frame conditioning |
| Audio VAE + Vocoder | High | Full audio pipeline (can defer to Phase 5) |
| Weight Converter | Medium | HF monolithic → modular layout, key sanitization |

## 4. Implementation Phases

### Phase 1: Video VAE (encoder + decoder)

**Goal:** Encode/decode video frames to/from 128-channel latents.

**Files to create:**
```
Sources/ComfyBox/LTX2/
├── VAE/
│   ├── LTX2VAE.swift              # Top-level encode/decode, tiled support
│   ├── LTX2Encoder3D.swift        # 3D encoder (CausalConv3d ResNet)
│   ├── LTX2Decoder3D.swift        # 3D decoder (CausalConv3d ResNet)
│   ├── LTX2ResnetBlock3D.swift    # ResNet block with GroupNorm + CausalConv3d
│   ├── LTX2DownBlock3D.swift      # Encoder downsample block
│   ├── LTX2UpBlock3D.swift        # Decoder upsample block
│   ├── LTX2MidBlock3D.swift       # Middle block with attention
│   └── LTX2Patchify.swift         # Patchify/unpatchify ops (latent → token sequence)
├── Common/
│   ├── LTX2ModelConfig.swift      # Model config (block counts, channels, dimensions)
│   └── LTX2WeightLoader.swift     # Weight loading + key remapping
```

**Key specs:**
- Input: RGB frames `(B, F, H, W, 3)` → Latent `(B, 128, F/8, H/32, W/32)`
- 32x spatial compression (patch_size=4 + 3 stages of 2x downsample)
- 8x temporal compression
- CausalConv3d throughout (reuse SeedVR2 implementation)
- Tiled decode for memory efficiency (reuse SeedVR2 tiling)

**Validation:** Encode a test frame, decode back, compare with Python reference output.

**Python reference files:**
- `mlx_video/models/video_vae/video_vae.py`
- `mlx_video/models/video_vae/encoder.py`
- `mlx_video/models/video_vae/decoder.py`
- `mlx_video/models/video_vae/convolution.py`
- `mlx_video/models/video_vae/resnet.py`
- `mlx_video/models/video_vae/ops.py` (patchify/unpatchify)
- `mlx_video/models/video_vae/tiling.py`

---

### Phase 2: Text Encoder (Gemma 3 + Connector)

**Goal:** Encode text prompts into video/audio conditioning embeddings.

**Files to create:**
```
Sources/ComfyBox/LTX2/
├── TextEncoder/
│   ├── LTX2TextEncoder.swift        # Top-level: tokenize → Gemma 3 → feature extract → connect
│   ├── LTX2GemmaWrapper.swift       # Gemma 3 12B model (MLX weights, Q4/Q8 quantized)
│   ├── LTX2FeatureExtractor.swift   # Extract+concat hidden states → linear projection
│   ├── LTX2Connector1D.swift        # 1D transformer (8 layers) with RoPE + registers
│   └── LTX2TextProjection.swift     # PixArtAlpha 2-layer MLP (caption → inner_dim)
```

**Key specs:**
- Gemma 3 12B-IT as text encoder (all 49 hidden states extracted)
- Feature extractor: norm + concat selected hidden states + linear → 3840-dim
- 1D connector: 8 transformer layers with RoPE, learnable register tokens
- Output: `video_embeddings (B, seq, 4096)` + `audio_embeddings (B, seq, 2048)`
- Q4 quantization for Gemma 3 (~6GB vs 24GB FP16)

**Dependencies:** swift-transformers for Gemma tokenizer. May need to add Gemma 3 model support if not already in mlx-swift ecosystem.

**Validation:** Compare connector output embeddings with Python reference for same prompt.

**Python reference files:**
- `mlx_video/models/ltx_2/text_encoder.py`
- `mlx_video/models/ltx_2/text_projection.py`

---

### Phase 3: DiT Transformer (48 blocks)

**Goal:** Implement the core denoising transformer.

**Files to create:**
```
Sources/ComfyBox/LTX2/
├── Transformer/
│   ├── LTX2Transformer.swift          # Top-level: patchify → blocks → unpatchify → x0 prediction
│   ├── LTX2TransformerBlock.swift     # BasicAVTransformerBlock: self-attn, cross-attn, cross-modal, FFN
│   ├── LTX2Attention.swift            # Multi-head attention: QK-RMSNorm, RoPE, per-head gating
│   ├── LTX2GEGLU.swift                # GEGLU feed-forward (tanh-GELU gating, 4x expansion)
│   ├── LTX2AdaLayerNorm.swift         # AdaLayerNormSingle: sinusoidal timestep → MLP → 6/9 params
│   ├── LTX2CrossModalAttention.swift  # Bidirectional A2V/V2A attention with separate gating
│   ├── LTX2RoPE.swift                 # 3D RoPE: SPLIT mode, pixel-space coords
│   ├── LTX2TimeEmbedding.swift        # Sinusoidal timestep embedding + MLP
│   └── LTX2PatchEmbed.swift           # Linear 128→4096 patchify projection + positional info
```

**Key specs:**
- 48 uniform blocks (no dual/single-stream split like Flux/SeedVR2)
- Each block: RMSNorm → AdaLN modulate → self-attn(RoPE) → cross-attn(text) → cross-modal(A2V/V2A) → GEGLU FFN
- Attention: QK-RMSNorm before softmax, 32 heads × 128 dim = 4096
- Per-head sigmoid gating (LTX-2.3 feature)
- Velocity prediction: `x0 = latent - sigma * velocity`
- Input: `(B, 128, F/8, H/32, W/32)` → flatten → patchify → `(B, F*H*W, 4096)`

**Validation:** Single forward pass through transformer, compare output tensor with Python reference (byte-level match).

**Python reference files:**
- `mlx_video/models/ltx_2/ltx_2.py` (LTXModel, X0Model)
- `mlx_video/models/ltx_2/transformer.py` (BasicAVTransformerBlock)
- `mlx_video/models/ltx_2/attention.py`
- `mlx_video/models/ltx_2/feed_forward.py`
- `mlx_video/models/ltx_2/adaln.py`
- `mlx_video/models/ltx_2/rope.py`

---

### Phase 4: Pipeline Integration (T2V + I2V)

**Goal:** End-to-end video generation from text or image input.

**Files to create:**
```
Sources/ComfyBox/LTX2/
├── LTX2Pipeline.swift              # Main pipeline: T2V, I2V orchestration
├── LTX2Sampler.swift               # res_2s sampler (Rosenbrock-RK + SDE noise)
├── LTX2LatentUpsampler.swift       # Conv3d + ResBlock + PixelShuffle for 2-stage
├── LTX2Conditioning.swift          # I2V conditioning: denoise mask, frame injection
├── LTX2Guidance.swift              # CFG / APG / STG guidance strategies
└── LTX2PostProcess.swift           # Frame extraction, video encoding (ffmpeg or AVFoundation)
```

**Key specs:**
- **T2V:** noise → 48-block denoise (8 steps distilled) → VAE decode → frames
- **I2V:** image → VAE encode → inject at frame 0 with denoise mask → denoise → decode
- **Two-stage:** Stage 1 at half-res (8 steps) → LatentUpsampler 2x → Stage 2 (4 steps) → decode
- **Guidance:** CFG (classifier-free), APG (attention-perturbation), STG (spatio-temporal)
- **Output:** MP4 via AVFoundation or raw frames to ffmpeg
- **CLI integration:** `ComfyBox video --prompt "..." --width 768 --height 512 --frames 97 --output video.mp4`

**Validation:** Generate a 5-second 512×384 clip from text, compare visual quality with Python reference.

**Python reference files:**
- `mlx_video/models/ltx_2/generate.py`
- `mlx_video/models/ltx_2/samplers.py`
- `mlx_video/models/ltx_2/upsampler.py`
- `mlx_video/models/ltx_2/conditioning/latent.py`

---

### Phase 5: Audio Pipeline (Optional — can defer)

**Goal:** Synchronized audio generation within the same denoising pass.

**Files to create:**
```
Sources/ComfyBox/LTX2/
├── Audio/
│   ├── LTX2AudioVAE.swift          # Audio VAE encoder/decoder (mel-spectrogram)
│   ├── LTX2Vocoder.swift           # HiFi-GAN variant (mel → waveform)
│   ├── LTX2AudioProcessor.swift    # Audio preprocessing (mel-spec extraction)
│   └── LTX2AudioPostProcess.swift  # WAV encoding, A/V muxing
```

**Key specs:**
- Audio latent shape: `(B, 8, T_audio, 16)` where `T_audio = round(duration * 25)`
- Audio patchified to 128-dim tokens, processed alongside video in same transformer
- Cross-modal attention already in Phase 3 blocks (A2V/V2A)
- Audio VAE: Conv2d encoder/decoder on mel spectrograms
- Vocoder: HiFi-GAN variant → 24kHz stereo waveform
- Output: mux audio+video into MP4

**Python reference files:**
- `mlx_video/models/audio_vae/`
- `mlx_video/models/ltx_2/config.py` (AudioDecoderModelConfig, VocoderModelConfig)

---

## 5. Model Storage

Models stored on Bolt drive (`/Volumes/Bolt/Models/`) via existing `~/Models` symlink.

```
~/Models/ltx2/
├── transformer/           # 22B DiT weights (Q4: ~11GB)
├── text_encoder/          # Gemma 3 12B (Q4: ~6GB)
├── vae/
│   ├── encoder/           # ~150MB
│   └── decoder/           # ~150MB
├── audio_vae/             # ~20MB
├── vocoder/               # ~50MB
├── latent_upsampler/      # ~200MB
└── config.json            # Model config
```

**Download source:** `Lightricks/LTX-2.3` on HuggingFace, pre-converted MLX weights from `prince-canuma/LTX-2.3-distilled`.

**Weight conversion:** Port `mlx_video/models/ltx_2/convert.py` logic into `LTX2WeightLoader.swift` — handle monolithic → modular split, key sanitization, Conv3d transpose.

## 6. ComfyUI Bridge Integration

After Phase 4, wire LTX-2 into the ComfyUI bridge:

- Register new node types in `/object_info`: `LTX2Sampler`, `LTX2TextEncode`, `LTX2VAEDecode`
- Map incoming ComfyUI workflows to `LTX2Pipeline` calls
- WebSocket progress: report step/total during denoising
- Support model switching: load/unload LTX-2 alongside Z-Image/Flux

## 7. Memory Management

M3 Max 128GB budget:

| Configuration | Memory | Fits? |
|--------------|--------|-------|
| LTX-2 Q4 + Gemma Q4 | ~18GB | Yes |
| LTX-2 BF16 + Gemma Q4 | ~50GB | Yes |
| LTX-2 Q4 + Z-Image BF16 (dual) | ~25GB | Yes |
| LTX-2 BF16 + Z-Image BF16 (dual) | ~51GB | Yes (tight) |

Strategy: Load text encoder for encoding phase, unload before denoising. VAE loaded only for encode/decode phases. Transformer stays resident during generation.

## 8. Testing Strategy

Each phase includes:
1. **Unit tests:** Per-module forward pass verification against Python reference
2. **Integration test:** End-to-end generation with fixed seed, compare output
3. **Benchmark:** Time per step, total generation time, peak memory

Reference comparison method: Save Python intermediate tensors as `.npy`, load in Swift, compare with tolerance `< 1e-4` (BF16).

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| Gemma 3 not in mlx-swift | Use mlx-lm's Gemma support or port from Python MLX |
| 22B transformer OOM during backward (if needed) | Q4 quantization, gradient checkpointing not needed (inference only) |
| Video output quality gap | Byte-level validation at each phase boundary |
| Audio sync complexity | Phase 5 is optional — ship video-only first |
| Conv3d transpose differences | Already solved in SeedVR2 — same pattern |

## 10. Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| Phase 1: Video VAE | 2-3 days | None (heavy SeedVR2 reuse) |
| Phase 2: Text Encoder | 3-4 days | Gemma 3 MLX-Swift availability |
| Phase 3: Transformer | 4-5 days | Phase 2 (needs text embeddings for testing) |
| Phase 4: Pipeline | 3-4 days | Phase 1 + 2 + 3 |
| Phase 5: Audio | 3-4 days | Phase 3 + 4 |
| **Total** | **15-20 days** | |

Phases 1 and 2 can run in parallel. Phase 3 can start once Phase 2 produces test embeddings.
