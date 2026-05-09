# LTX-2 Swift/MLX Port — Codex Handoff

**Date:** 2026-05-09
**Author:** Bree
**Branch:** `bree/ltx2-phase4-pipeline`
**Tracking Issue:** #117
**Status:** Phases 1–4 code complete, builds clean, 13 smoke tests pass. Three blockers prevent end-to-end video generation with real weights.

---

## What's Done

### Code (30 files, 7,727 lines)

| Phase | Files | Lines | Status |
|-------|-------|-------|--------|
| Phase 1: Video VAE | 8 files in `Sources/ZImage/LTX2/VAE/` + `Common/` | ~1,758 | ✅ Complete |
| Phase 2: Text Encoder | 6 files in `Sources/ZImage/LTX2/TextEncoder/` | ~2,091 | ✅ Complete |
| Phase 3: DiT Transformer | 9 files in `Sources/ZImage/LTX2/Transformer/` | ~1,545 | ✅ Complete |
| Phase 4: Pipeline Integration | 7 files in `Sources/ZImage/LTX2/` | ~2,090 | ✅ Complete |
| Tests | `Tests/ZImageTests/LTX2SmokeTest.swift` | 385 | ✅ 13/13 pass |

### Commits on branch

```
1d9c7f2 feat(ltx2): Phase 4 — T2V + I2V pipeline integration
911b7f3 feat(ltx2): Phase 3 — 48-block DiT transformer
c6e0278 test(ltx2): Phase 1+2 smoke tests (VAE, Patchify, TextEncoder, Connector)
d1825a6 feat(ltx2): Phase 2 — Text Encoder (Gemma 3 + 1D Connector)
be2730b feat(ltx2): Phase 1 — Video VAE encoder/decoder
```

### What's been validated with real weights

- **Transformer forward pass**: 1457 video keys loaded from pre-extracted weights, zero shape mismatches, 8-step distilled denoising produces output
- **VAE decode**: Works with random weights — produces frames, writes MP4 via AVFoundation
- **MP4 encoding**: AVAssetWriter + H.264 works correctly
- **Metal JIT**: First denoising step ~4.1s (shader compilation), subsequent steps ~0.5s each
- **Memory**: Transformer fits in memory on M3 Max 128GB

### Model weights on disk

All weights are on the Bolt Thunderbolt drive at `/Volumes/Bolt/Models/ltx2-distilled/`:

| File | Size | Keys | Status |
|------|------|------|--------|
| `transformer-distilled.safetensors` | 38 GB | 4186 | ⚠️ MLX hangs loading this |
| `transformer-dev.safetensors` | 38 GB | ? | Not tested |
| `transformer-distilled-1.1.safetensors` | 38 GB | ? | Not tested |
| `connector.safetensors` | 5.9 GB | ? | Connector weights for text encoder |
| `vae_decoder.safetensors` | 814 MB | ? | ⚠️ Config mismatch |
| `vae_encoder.safetensors` | 638 MB | ? | ⚠️ Config mismatch |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.6 GB | ? | Distilled LoRA adapter |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.6 GB | ? | Distilled LoRA v1.1 |
| `spatial_upscaler_x2_v1_1.safetensors` | 996 MB | ? | Phase 5 upsampler |
| `spatial_upscaler_x1_5_v1_0.safetensors` | 1.1 GB | ? | Phase 5 upsampler |
| `temporal_upscaler_x2_v1_0.safetensors` | 262 MB | ? | Phase 5 upsampler |
| `audio_vae.safetensors` | 107 MB | ? | Phase 5 audio |
| `vocoder.safetensors` | 258 MB | ? | Phase 5 audio |

Pre-extracted transformer weights (created during testing):
- `/tmp/transformer-video-only.safetensors` (24 GB, 1457 video-only keys) — loads instantly

NSFW merge model:
- `/Volumes/Bolt/Models/ltx2-loras/nsfw/ltx-2-19b-phr00tmerge-nsfw-v6.safetensors` (18 GB)

---

## Three Blockers

### Blocker 1: MLX hangs on 38 GB safetensors file

**Symptom:** `MLX.loadArrays(url:)` hangs indefinitely (0% CPU) when loading `transformer-distilled.safetensors` (38 GB, 4186 keys). Process never returns.

**Workaround found:** Pre-extract video-only keys via Python:
```python
import safetensors.torch as st
tensors = st.load_file("/Volumes/Bolt/Models/ltx2-distilled/transformer-distilled.safetensors")
video_keys = {k: v for k, v in tensors.items() if "audio" not in k.lower()}
# 1457 keys, ~24 GB
st.save_file(video_keys, "/tmp/transformer-video-only.safetensors")
```
The 24 GB file loads instantly via `MLX.loadArrays`.

**Proper fix needed:** Either:
1. **Streaming loader** — load keys on demand from the 38 GB file without reading everything into memory at once
2. **Pre-split at download time** — split the 38 GB file into video-only and audio-only shards as part of model setup
3. **Fix in MLX** — report upstream if it's a bug in the MLX safetensors parser (possibly a 32-bit offset issue for files > ~4 GB? or memory mapping failure?)

**File:** `Sources/ZImage/LTX2/Common/LTX2WeightLoader.swift` — add a `loadTransformerWeights` method that handles the split/streaming

### Blocker 2: VAE config mismatch (v2.3 vs hardcoded default)

**Symptom:** `LTX2WeightLoader.loadVAEWeights()` fails with shape mismatches because the hardcoded `.default` config doesn't match the v2.3 model architecture.

**Current `.default` encoder blocks:**
```swift
.resX(numLayers: 4),
.compressSpaceRes(multiplier: 2),
.resX(numLayers: 6),
.compressTimeRes(multiplier: 2),
.resX(numLayers: 6),          // ← should be 4
.compressAllRes(multiplier: 2),
.resX(numLayers: 2),
.compressAllRes(multiplier: 2), // ← should be multiplier: 1
.resX(numLayers: 2),
```

**Correct v2.3 encoder blocks** (from `embedded_config.json`):
```json
["res_x", {"num_layers": 4}],
["compress_space_res", {"multiplier": 2}],
["res_x", {"num_layers": 6}],
["compress_time_res", {"multiplier": 2}],
["res_x", {"num_layers": 4}],
["compress_all_res", {"multiplier": 2}],
["res_x", {"num_layers": 2}],
["compress_all_res", {"multiplier": 1}],
["res_x", {"num_layers": 2}]
```

**Current `.default` decoder blocks:**
```swift
.resX(numLayers: 5),
.compressAll(multiplier: 2, residual: true),
.resX(numLayers: 5),
.compressAll(multiplier: 2, residual: true),
.resX(numLayers: 5),
.compressAll(multiplier: 2, residual: true),
.resX(numLayers: 5),
```

**Correct v2.3 decoder blocks** (from `embedded_config.json`):
```json
["res_x", {"num_layers": 4}],
["compress_space", {"multiplier": 2}],
["res_x", {"num_layers": 6}],
["compress_time", {"multiplier": 2}],
["res_x", {"num_layers": 4}],
["compress_all", {"multiplier": 1}],
["res_x", {"num_layers": 2}],
["compress_all", {"multiplier": 2}],
["res_x", {"num_layers": 2}]
```

**Fix needed (3 files):**

1. **`Sources/ZImage/LTX2/Common/LTX2ModelConfig.swift`**:
   - Add `DecoderBlockDef.compressSpace(multiplier:)` and `DecoderBlockDef.compressTime(multiplier:)` enum cases
   - Add a `.v23` static config that matches the `embedded_config.json` block layouts above
   - Update `spatialCompression` and `temporalCompression` computed properties to handle the new cases
   - Consider: add `static func fromEmbeddedConfig(_:)` that parses `embedded_config.json` at runtime

2. **`Sources/ZImage/LTX2/VAE/LTX2Decoder3D.swift`**:
   - Handle `.compressSpace` and `.compressTime` in the decoder block builder (currently only handles `.resX` and `.compressAll`)
   - These are spatial-only and temporal-only upsample blocks respectively

3. **`Sources/ZImage/LTX2/VAE/LTX2Sampling.swift`**:
   - Ensure `LTX2DepthToSpaceUpsample` can handle spatial-only and temporal-only modes
   - May need separate upsample modules for each dimension

**Reference:** The `embedded_config.json` at `/Volumes/Bolt/Models/ltx2-distilled/embedded_config.json` is the ground truth for the v2.3 architecture.

### Blocker 3: Missing Gemma 3 12B text encoder weights

**Symptom:** `LTX2TextEncoder` requires Gemma 3 12B backbone weights (~24 GB at fp16, ~6 GB at Q4). These are NOT included in the Lightricks/LTX-2.3 download. Without them, the pipeline can only use dummy random embeddings (produces noise, not meaningful video).

**What we have:** `connector.safetensors` (5.9 GB) — this is the 1D connector that sits between Gemma and the transformer. It is NOT the Gemma backbone itself.

**Fix needed:**
1. Download Gemma 3 12B-IT weights from HuggingFace: `google/gemma-3-12b-it`
   - Full fp16: ~24 GB
   - Q4 quantized: ~6 GB (use ComfyBox's existing quantization pipeline)
   - Destination: `/Volumes/Bolt/Models/gemma-3-12b/`
   - **Note:** Requires HuggingFace token + Google license acceptance

2. Add weight loading to `Sources/ZImage/LTX2/TextEncoder/LTX2TextEncoder.swift`:
   - `loadGemmaWeights(from:dtype:)` — loads Gemma backbone
   - `loadConnectorWeights(from:)` — loads connector from `connector.safetensors`
   - Handle HuggingFace key format → Swift module key remapping for Gemma

3. Add tokenizer support:
   - Gemma 3 uses SentencePiece tokenization (262144 vocab)
   - Either: port a minimal SentencePiece decoder to Swift
   - Or: use a pre-tokenized approach (tokenize on Python side, pass IDs)
   - Or: use the MLX Swift tokenizer library if available

**Alternative (faster for demo):** Pre-compute text embeddings via Python for a few test prompts and save as `.safetensors`. Load those directly in Swift and pass to `generateT2VWithEmbeddings()` — this proves the full pipeline without porting the tokenizer.

---

## Additional Issues Found

### xcodebuild test runner limitations
`swift test` fails on this repo because the MLX Metal shader library (`mlx.metallib`) is not found by SwiftPM's test runner. Must use `xcodebuild test` instead, but xcodebuild struggles with long-running weight loads. For integration tests, prefer standalone CLI commands over xctest.

### Render constraints
- NEVER run multiple GPU-heavy renders concurrently — M3 Max crashes
- One render at a time, wait for completion
- Check Mac responsiveness before starting GPU-heavy work

---

## File Map

```
Sources/ZImage/LTX2/
├── Common/
│   ├── LTX2ModelConfig.swift         ← NEEDS FIX (Blocker 2: add .v23 config)
│   └── LTX2WeightLoader.swift       ← NEEDS FIX (Blocker 1: add transformer weight loading)
├── VAE/
│   ├── LTX2VAE.swift                 — Top-level encode/decode/decodeTiled
│   ├── LTX2Encoder3D.swift           — 9-block encoder
│   ├── LTX2Decoder3D.swift           ← NEEDS FIX (Blocker 2: add compressSpace/Time)
│   ├── LTX2ResnetBlock3D.swift       — ResNet + MidBlock
│   ├── LTX2Sampling.swift            ← MAY NEED FIX (Blocker 2: per-dim upsample)
│   └── LTX2Patchify.swift            — Static patchify/unpatchify
├── TextEncoder/
│   ├── LTX2TextEncoder.swift         ← NEEDS FIX (Blocker 3: weight loading)
│   ├── LTX2TextEncoderConfig.swift   — Config structs (correct for v2.3)
│   ├── LTX2GemmaModel.swift          — Full Gemma 3 12B (code done, needs weights)
│   ├── LTX2FeatureExtractor.swift    — V1/V2 normalization
│   ├── LTX2Connector1D.swift         — 8-layer 1D transformer
│   └── LTX2TextProjection.swift      — 2-layer MLP
├── Transformer/
│   ├── LTX2Transformer.swift         — Top-level with sanitizeWeights()
│   ├── LTX2TransformerBlock.swift    — BasicAVTransformerBlock
│   ├── LTX2Attention.swift           — QK-RMSNorm + gated attention
│   ├── LTX2RoPE.swift                — 3D RoPE (SPLIT mode)
│   ├── LTX2TimeEmbedding.swift       — Timestep MLP
│   ├── LTX2GEGLU.swift               — GELU-gated FFN
│   ├── LTX2AdaLayerNorm.swift        — 6/9-param modulation
│   ├── LTX2PatchEmbed.swift          — Position grid computation
│   └── LTX2CrossModalAttention.swift — Phase 5 stub (pass-through)
├── LTX2Pipeline.swift                — T2V + I2V orchestrator
├── LTX2PipelineConfig.swift          — Distilled/dev sigma schedules
├── LTX2Sampler.swift                 — Euler + res_2s
├── LTX2Conditioning.swift            — I2V frame conditioning
├── LTX2Guidance.swift                — CFG + APG
├── LTX2LatentUpsampler.swift         — Two-stage upsampler
└── LTX2PostProcess.swift             — Frames → CGImage → MP4 (AVFoundation)
```

---

## Suggested Codex Task Order

1. **Fix Blocker 2 (VAE config)** — smallest scope, unblocks VAE weight loading
   - Add decoder block types, create `.v23` config, test with real VAE weights
   - Test: `LTX2WeightLoader.loadVAEWeights()` succeeds with zero shape mismatches

2. **Fix Blocker 1 (transformer weight loading)** — add streaming/split loading
   - Add `loadTransformerWeights(from:dtype:)` to `LTX2WeightLoader`
   - Either stream keys or filter to video-only keys before loading
   - Test: loads from the 38 GB file without hanging

3. **Add CLI demo subcommand** — add `ltx2-demo` to `Sources/ComfyBox/main.swift`
   - Add `generateT2VWithEmbeddings()` public method to `LTX2Pipeline` (takes pre-computed embeddings, bypasses text encoder)
   - Demo: loads transformer + VAE, uses dummy embeddings, runs 4-step distilled, saves MP4
   - Test: `ComfyBox ltx2-demo --output /tmp/test.mp4` produces a valid MP4

4. **Fix Blocker 3 (text encoder)** — largest scope, download + tokenizer
   - Download Gemma 3 12B-IT to Bolt
   - Add weight loading for Gemma + connector
   - Either port SentencePiece tokenizer or pre-compute embeddings
   - Test: full T2V pipeline with real text prompt produces meaningful video

---

## Build & Test Commands

```bash
# Build
cd ~/Projects/zimage.swift
swift build -c release

# Sign (required for Metal)
codesign --force --sign - .build/release/ComfyBox
xattr -cr .build/release/ComfyBox

# Run smoke tests (use xcodebuild, NOT swift test)
xcodebuild test -scheme comfybox-Package \
  -destination "platform=macOS" \
  -only-testing:ZImageTests/LTX2SmokeTest 2>&1 | tail -20

# Run demo (once CLI subcommand exists)
.build/release/ComfyBox ltx2-demo \
  --model-dir /Volumes/Bolt/Models/ltx2-distilled \
  --width 512 --height 320 --frames 9 --steps 4 \
  --output /tmp/ltx2-demo.mp4
```

---

## Reference Configs

### embedded_config.json location
`/Volumes/Bolt/Models/ltx2-distilled/embedded_config.json`

### config.json location
`/Volumes/Bolt/Models/ltx2-distilled/config.json`

### Python mflux reference
`~/Projects/mflux/src/mflux/` — NOT used for LTX-2 (LTX-2 is a new architecture, not in mflux)

### LTX-Video Python reference
The original Python implementation is at `github.com/Lightricks/LTX-Video`. Key files:
- `ltx_video/pipelines/pipeline_ltx_video.py` — pipeline logic
- `ltx_video/models/autoencoders/causal_video_autoencoder.py` — VAE
- `ltx_video/models/transformers/symmetric_patchifier.py` — patchify
- `ltx_video/models/transformers/transformer3d.py` — transformer
