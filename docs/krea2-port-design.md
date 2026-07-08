# Krea-2 → ComfyBox (Swift/MLX) Port — Design

**Decision:** Port Krea-2-Turbo into ComfyBox as an art model, usable from Krita via the ComfyUI bridge. Chroma dropped. Krea-2 validated strong on realism, documentary, and stylized art (Frida-Kahlo pass). ComfyBox is Python-free → this is a native Swift/mlx-swift port (no mflux at runtime).

## Status of groundwork (DONE)
- **All weights cached + validated** in `~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<hash>/`:
  - `turbo.safetensors` — transformer, 430 tensors, native MLX names (bf16). 24.4 GB.
  - `text_encoder/model.safetensors` — Qwen3-VL-4B, 713 tensors (keys prefixed `language_model.`).
  - `vae/diffusion_pytorch_model.safetensors` — Qwen-Image VAE, 194 tensors. `vae/config.json` present.
  - `tokenizer/` (tokenizer.json, tokenizer_config.json, chat_template.jinja), `scheduler/`, `model_index.json`, `transformer/config.json`.
  - (Diffusers `transformer/*.safetensors` shards NOT needed — `turbo.safetensors` is the native MLX transformer.)
- **Reference MLX implementation** downloaded to `docs/krea2-reference/` (from `avlp12/Krea-2-Turbo-Alis-MLX-8bit`): `krea2/{transformer,text_encoder,sampling,quant_recipes,pipeline}.py`. This is a faithful, working blueprint — module names match `turbo.safetensors` exactly (direct weight load, no remapping). Optional `transformer_8bit.safetensors` NOT downloaded (we quantize in-Swift from bf16).

## Architecture (three models + a scheduler)

### 1. Transformer — `SingleStreamDiT` (Krea2Transformer2DModel), ~12.9B
Config: features 6144, 28 layers, heads 48 / kv 12 (GQA 4:1), head_dim 128, patch 2, channels 16, in_channels 64, intermediate 16384, rope_theta 1000, `axes_dims_rope=[32,48,48]` (3-axis t/h/w). Text side: txtdim 2560, txtheads 20, 12 selected encoder layers, 2 layerwise + 2 refiner text-fusion blocks.
- **Reuse** existing primitives: RMSNorm (weight = scale+1, f32, eps 1e-5), 3-axis RoPE (interleaved), SwiGLU, GQA attention.
- **New**: single-stream block (DoubleSharedModulation: 6-way split, pre/post gate), **sigmoid-gated attention output** (extra `gate` proj), `TextFusionTransformer` (layerwise blocks over the 12 stacked encoder layers → `projector` Linear(12→1) → refiner blocks), timestep MLP + tproj (6·features), LastLayer with SimpleModulation. See `docs/krea2-reference/krea2/transformer.py` (line-by-line portable).
- Flow: `first` patch-embed → fuse text (TextFusion) → concat [text; img] → 28 joint blocks → LastLayer → slice img tokens. Predicts flow velocity `v`.

### 2. Text encoder — Qwen3-VL-4B (text-only) `Qwen3TextModel`
Standard Qwen3 decoder: vocab 151936, hidden 2560, 36 layers, heads 32 / kv 8 (GQA), head_dim 128 (decoupled from hidden), inter 9728, RMSNorm eps 1e-6, per-head QK-norm, RoPE θ=5e6, causal+padding mask.
- Outputs **all 36 hidden states**; select layers `(2,5,8,11,14,17,20,23,26,29,32,35)` → stack `(B,L,12,2560)`.
- Special conditioning template: PREFIX = system "Describe the image by detailing color, shape, size, texture…" + user turn; SUFFIX = assistant turn. Hardcoded slice `prefix_idx=34`. Guard against tokenizer drift (prefix must tokenize to 34 tokens, suffix to 5).
- Weight load strips `language_model.` prefix. `~/…/text_encoder/model.safetensors`.
- **New** encoder (ComfyBox's existing Qwen encoder is a different config/model). Tokenizer: Qwen BPE — verify existing `QwenTokenizer` can load `tokenizer/tokenizer.json`; adapt if needed.

### 3. VAE — `AutoencoderKLQwenImage` (⚠ biggest net-new piece)
**3D causal (Wan-style) VAE, NOT the existing 2D `AutoencoderKL`.** Config: z_dim 16, base_dim 96, dim_mult [1,2,4,4], temporal_downsample [F,T,T], spatial_scale 8, per-channel `latents_mean`/`latents_std` (16 vals) for normalization. Decode returns `(n,3,1,H,W)` (temporal dim 1 for images). 194 tensors.
- Port mflux's `QwenVAE` (Wan 2.x VAE family) to Swift, or implement a decoder-only variant (inference only). Apply mean/std denorm before decode. This is the largest new module.

### 4. Scheduler — flow-matching Euler with resolution shift
`img = img + (t_next - t_cur) * v`, no CFG (turbo, guidance 0). Timesteps: exp/sigmoid warp with `mu` derived from token count (`y1=0.5, y2=1.15`, `x1=(minres/align)², x2=(maxres/align)²`, minres 256, maxres 1280, align = 8·patch = 16). Same family as ComfyBox `FlowMatchScheduler` (dynamic shift) → **reuse/extend** with Krea-2 mu params. patchify/unpatchify: patch 2, 16ch → 64-dim tokens; 3-axis positions (text=0, img h/w).

## Port plan (suggested phases, each independently testable)
1. **Weights/loader + config** — Krea2 config structs; load `turbo.safetensors` (verify 430 keys map 1:1), text_encoder (713, strip prefix), vae (194).
2. **Text encoder** — Qwen3TextModel + 12-layer tap + template; unit-test hidden-state shapes vs reference on a fixed prompt.
3. **VAE decoder** — QwenImage 3D VAE decode; test decode of a known latent vs reference output.
4. **Transformer** — SingleStreamDiT; test a single forward (velocity) against reference for a fixed (img, ctx, t, pos, mask).
5. **Scheduler + pipeline** — wire encode→sample→decode; end-to-end image vs reference (same seed/prompt) — golden-image parity.
6. **Quantization** — ZImageQuantizer 4/8-bit on transformer (+ encoder) for memory/speed.
7. **Integration** — register model family (mirror flux1/flux2/fibo/chroma); expose in ComfyBridge `/object_info` + generate path so Krita and the desktop see "krea2".

## Notable risks / findings
- **3D Qwen-Image VAE** is the main new-arch risk — budget the most time here.
- **Render cost**: ~3 min/image at 8-bit/9 steps in mflux on M3 Max — the Swift/quantized path should aim to beat that; still far slower than Z-Image Turbo. Acceptable for hero/art shots, painful for live iteration.
- **Parity testing**: the reference impl gives golden outputs (same seed → same image) — port each component to match numerically before moving on.
- **Tokenizer drift guard** from the reference must be preserved (misaligned slice = garbage conditioning).

## References
- Blueprint: `docs/krea2-reference/krea2/*.py`
- Weights: `~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/…`
- Eval findings + verdict: memory `krea2-eval-findings`. Constraint: memory `comfybox-no-python`.
