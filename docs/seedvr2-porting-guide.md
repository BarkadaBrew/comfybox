# SeedVR2 Native Swift/MLX Porting Guide

## Overview

SeedVR2 is a 3B-parameter diffusion transformer (NaDiT) for image super-resolution. This guide covers porting the mflux Python/MLX implementation to native Swift/MLX within ZImageCLI.

**Source**: mflux Python at `~/Projects/mflux/src/mflux/models/seedvr2/`
**Target**: ZImageCLI Swift at `Sources/ZImage/Upscale/`
**Weights**: HuggingFace `numz/SeedVR2_comfyUI` — `seedvr2_ema_3b_fp16.safetensors` (~6GB) + `ema_vae_fp16.safetensors` (~150MB)

## Architecture Summary

- 32 transformer blocks (10 dual-stream + 22 single-stream)
- 3D Video VAE (CausalConv3d, 8x spatial downscale, 16 latent channels)
- Windowed attention with shifted windows (Swin-style)
- 3D axial RoPE (temporal + height + width)
- Pre-computed text embeddings (no runtime T5/CLIP)
- 1-step default inference (flow-matching Euler scheduler)
- Wavelet + LAB color correction post-processing

## Key Dimensions

- vid_dim = 2560, txt_dim = 2560, txt_in_dim = 5120
- heads = 20, head_dim = 128, inner_dim = 2560
- MLP expand: 2560 → 6912 → 2560 (SwiGLU)
- emb_dim = 15360 (6 × vid_dim)
- patch_size = (1, 2, 2)
- window = (4, 3, 3)
- VAE: 3→128→256→512→512→32(encode) / 16→512→512→256→128→3(decode)
- scaling_factor = 0.9152

## Implementation Phases

### Phase 1: Foundation (independently testable)
1. RMSNorm — `mx.fast.rms_norm` wrapper
2. SwiGLUMLP — 3 Linears + SiLU gate
3. CausalConv3d — causal temporal padding + `conv_general`
4. SeedVR2EulerScheduler — flow-matching Euler step
5. Text embeddings loader — load `pos_emb.safetensors` (58, 5120)

### Phase 2: VAE (test with weight loading)
6. ResnetBlock3D — 2× CausalConv3d + GroupNorm + optional shortcut
7. Attention3D — per-frame spatial self-attention
8. Downsample3D / Upsample3D — spatial/temporal scale
9. DownBlock3D / UpBlock3D / MidBlock3D — composition
10. Encoder3D + Decoder3D
11. SeedVR2VAE — thin wrapper with scaling

### Phase 3: Transformer (most complex)
12. TimeEmbedding — sinusoidal(256) → MLP → (15360)
13. AdaModulation — learned per-layer shift/scale/gate + time embedding
14. PatchIn / PatchOut — patchify/unpatchify 5D → tokens
15. RoPEModule — 3D axial frequencies
16. WindowPartitioner — adaptive windows, shifted windows (HARDEST MODULE)
17. MMAttention — windowed + text broadcast + QK-norm + RoPE
18. MMSwiGLU — multi-modal SwiGLU wrapper
19. TransformerBlock — full block with AdaLN
20. SeedVR2Transformer — 32 blocks + embeddings

### Phase 4: Pipeline Integration
21. SeedVR2LatentCreator — noise + conditioning
22. SeedVR2Util — preprocessing, wavelet, LAB color transfer
23. Full pipeline — load, encode, denoise, decode, color correct

## Inference Pipeline (tensor shapes)

```
Input image → preprocess → (1, 3, H, W) [-1, 1]
  → add temporal dim → (1, 3, 1, H, W)
  → VAE encode → (1, 16, 1, H/8, W/8) × 0.9152
  → concat ones mask → condition (1, 17, 1, H/8, W/8)
  → create noise → latents (1, 16, 1, H/8, W/8)
  → concat [latents, condition] → model_input (1, 33, 1, H/8, W/8)
  → load text embeddings → (1, 58, 5120)
  → transformer(model_input, text, timestep) → noise (1, 16, 1, H/8, W/8)
  → Euler step → denoised latents
  → VAE decode → (1, 3, 1, H, W)
  → crop padding + wavelet + LAB color correction → output image
```

## Weight Mapping

### Transformer
- `vid_in.proj.{w,b}` → direct
- `txt_in.{w,b}` → direct
- `emb_in.proj_{in,hid,out}.{w,b}` → direct
- `vid_out_ada.out_{shift,scale}` → rename to `out_{shift,scale}`
- Blocks 0-9: `.vid.` and `.txt.` separate weights
- Blocks 10-31: `.all.` weights shared → load into BOTH vid and txt attributes
- Block 31: is_last_layer (no txt MLP, txt frozen in attention)
- `blocks.{i}.ada.vid.` → `blocks.{i}.ada.params_vid.`
- `blocks.{i}.attn.rope.rope.freqs` → `blocks.{i}.attn.rope.freqs`

### VAE
- All conv3d weights: transpose `(out, in, kt, kh, kw)` → `(out, kt, kh, kw, in)`
- GroupNorm, Linear, bias: direct

## Risk Areas

1. **WindowPartitioner** — most complex module, index computation is subtle
2. **CausalConv3d** — causal padding + weight layout must exactly match
3. **3D RoPE** — text offset and axial frequency concatenation, off-by-one = garbage
4. **Shared weights (blocks 10-31)** — same safetensors tensor → both vid and txt attrs

## 7B Variant

Same architecture, wider dimensions. Once 3B works, 7B is a config change:
- Larger vid_dim, more heads, wider MLP
- ~14GB weights (fp16), ~7GB (fp8)

## Workflow

- Claude 4.7 (Opus) for architecture and planning
- Codex 5.5 for code review after each module
- Test each module against Python reference output before proceeding
