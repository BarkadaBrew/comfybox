# FDD: Krea2 Depth-ControlNet (Control-LoRA) port to ComfyBox

**Repo:** `BarkadaBrew/comfybox` (`~/Projects/zimage.swift`, Swift/MLX)
**Author:** Opus (technical architect)  **Date:** 2026-07-20
**Status:** v2 — Fable review resolved (BLOCK→addressed); ready to implement

## v2 changelog (resolves Fable review)
- **Checkpoint verified** (`depth-control-lora.safetensors`, 450 tensors, all F32):
  `first.weight [6144,128]`, `first.bias [6144]`; per block i∈0..27 × 8 targets, `A [64,in]`,
  `B [out,64]` (rank 64). **Metadata: `base: krea/Krea-2-Raw`, step 6000, rank 64.**
- **§3.2 input projection → GATED OPTION A (revised).** Do NOT use the parallel-sum (option B):
  Fable showed it silently drops the file's trained `first.bias` and isn't equivalent under q8
  (base `first` is quantized: predicate `dim(1)%64==0 && !path.contains("projector")`, and
  `first.weight` is `(6144,64)`). Instead add a **separate `controlFirst: Linear(128→6144)`**
  loaded directly from the file's `first.weight`+`first.bias` (correct trained bias). Path name
  contains `"projector"` so it is **excluded from q8 quantization** (load fp/bf16). Control ON:
  `img = controlFirst(concat([imgPatched, ctrlPatched], axis:-1))`. Control OFF: unchanged
  `first(imgIn)` → **byte-identical** to today (crit #1 trivially holds). Load-time asserts:
  `allclose(file first.weight[:, :64], base first.weight)` (catches checkpoint/base drift).
- **Blocker 2 (LoRA lifecycle) RESOLVED.** `Krea2Pipeline.loadLoRAs` clears-all then applies each
  config via `LoRAApplicator.applyDynamically` in a loop (production already stacks KNP+Pinay this
  way). Control-LoRA is added as **one more config in the `loadLoRAs` list**, tracked in
  `appliedLoRAs`, so hot-swap (`swap_loras`) re-applies it too. No separate slot needed. The
  `controlFirst` module + cached control latent are pipeline state, applied per control render.
- **Blocker 4 (VAE.encode) RESOLVED.** `Krea2VAE.encode` exists and is **deterministic** —
  returns normalized mean `(mean − datasetMean)/datasetStd`, no sampling/randn (verified
  Krea2VAE.swift:355-362). Encode the depth map through the SAME `vae.encode` → 16-ch latent in
  the model's normalized space; patchify (16×2×2=64) then concat → 128. Caching + fixed-seed safe.
- **Blocker 3 (turbo) = EMPIRICAL GO/NO-GO.** Trained on Krea-2-Raw (metadata) though the card
  claims Turbo-8step support. Must validate on our **q8-turbo**; if it degrades, fallback is a
  bf16 Krea-2-Raw base for control renders (needs download) — decision gated on the test, not the build.
- **Blocker 5 (tests) RESOLVED** — added crit #8–#11 below (reference-parity, strength-0,
  config-matrix pin, VLM rubric). Also confirmed (Fable): RoPE positions, `additiveMask`, timestep
  `mu`/`seqLen` shift are **unaffected** (control is a feature-dim concat, not extra tokens).
- **Strength knob:** map `controlnet_strength` to the **LoRA α only** (conventional); the control
  latent enters `controlFirst` at gain 1.0. (Avoids the compounding double-attenuation Fable noted.)
- **Control-image resolution:** resize the depth map to the `align=16`-rounded target BEFORE
  `vae.encode` (mirror the z-image ControlNet path); do not assert-fail on nominal mismatch.

_original v1 below (superseded where it conflicts with v2)_

**Status(v1):** Draft — for Codex review
**Reference:** `Tanmaypatil123/Krea-2-controlnet` · model `Patil/Krea-2-depth-controlnet` (`depth-control-lora.safetensors`, 862MB, downloaded to `/Volumes/Bolt/Models/krea2-controlnet/`)

## 1. Summary & motivation

Give ComfyBox's **krea2** pipeline depth-conditioned generation via the Patil "Control-LoRA"
mechanism. Today, on-model Kira + exact body position requires a **two-stage** pipeline:
z-image (ControlNet, position) → krea2 img2img (Krea-Kira identity). krea2 has **no native
ControlNet** (confirmed: no HF adapter). This feature collapses that to **one stage** —
krea2 renders identity (Krea-Kira: KNPV4.1 + Filipina_Pinay, "Pinay" trigger) **and** follows
a depth map in a single pass. Wins: better identity (krea2-native, no z-image handoff/quality
loss), ControlNet-grade pose lock on the good model, fewer renders, simpler scene pipeline.

Scope: **depth only** (the released checkpoint). Canny/pose/tile are trainable later with the
same surgery (control-agnostic) but out of scope here.

## 2. Mechanism (reference) and ComfyBox mapping

The Control-LoRA is **not** a separate ControlNet network. It is three pieces applied to the
frozen krea2 DiT:

| Reference (`k2_lora.py`, PyTorch) | ComfyBox (`Sources/ZImage/Krea2/`, Swift/MLX) |
|---|---|
| `model.first: nn.Linear` (input proj, C→hidden) | `Krea2Transformer` `@ModuleInfo(key:"first") var first: Linear`; forward `let img = first(imgIn)` |
| `ControlInputLayer` replaces `first`: weight `(out, 2C)`, `[:, :C]=pretrained`, `[:, C:]` trained (in file) | New `Krea2ControlFirst` (or expand `first`) that projects concat `[noisy‖control]` of width 2C |
| rank-64 LoRA on 28 blocks, targets `attn.{wq,wk,wv,wo,gate}`, `mlp.{gate,up,down}` | Same names exist: `Krea2Attention` keys wq/wk/wv/wo/gate; `Krea2SwiGLU` keys gate/up/down; `Krea2Config.layers=28`. Apply via existing `Sources/ZImage/LoRA/LoRALinear.swift` |
| depth → Depth-Anything-V2 → Qwen VAE encode → depth latent | `Krea2VAE` (Qwen-Image VAE, already present) encodes a **pre-supplied** depth image |
| each step: concat depth latent channel-wise to noisy latent (64→128/token) | Concat once-encoded control latent to `imgIn` before `first(...)`, every denoise step |

Safetensors keys in `depth-control-lora.safetensors`:
- `first.weight` `(out, 2C)`, `first.bias` — the **full expanded** projection (load directly; do not re-zero the control half).
- `blocks.{i}.{target}.A` `(rank, in)`, `blocks.{i}.{target}.B` `(out, rank)` for i∈0..27, 8 targets.

## 3. Design

### 3.1 Model loading — `Krea2ControlLoRA` loader
New file `Sources/ZImage/Krea2/Krea2ControlLoRA.swift`:
- Parse the safetensors; split into (a) `first.weight`/`first.bias`, (b) per-block A/B tensors.
- Validate: 28 blocks × 8 targets present; `first.weight.shape[1] == 2 * baseFirstInFeatures`.
- Key adapter: map reference names → ComfyBox `@ModuleInfo` paths (identical leaf names; only
  the container path differs — `blocks[i].attn.wq` etc.). No diffusers-name translation needed.

### 3.2 Expanded input projection
- When control is active, `first` must accept width `2C`. Two options:
  - **(A, preferred)** Swap in a `Krea2ControlFirst` module holding the file's `first.weight`
    `(out, 2C)` + `first.bias`; forward = `linear(concat([img, ctrl], axis=-1))`.
  - (B) Keep base `first`, add a parallel `firstControl: Linear(C→out, bias:false)` initialized
    from `first.weight[:, C:]`, and sum: `first(img) + firstControl(ctrl)`. Mathematically
    identical to the concat (block-wise matmul), avoids reshaping the base module; easier to
    gate on/off. **Recommend (B)** for cleaner enable/disable + smaller blast radius.
- Base weights untouched when control inactive (zero regression risk to normal krea2).

### 3.3 LoRA injection
- Reuse `LoRALinear` to wrap the 8×28 target `Linear`s at load, scale=1.0 (Control-LoRA is not
  a style LoRA — fixed strength; expose `controlnet_strength` mapping to LoRA α + control-latent
  gain, default 1.0).
- Must **compose** with the Krea-Kira identity LoRAs (KNP + Pinay) already applied. Verify
  additive stacking (two LoRA sets on the same Linear) — `LoRALinear` chaining or summed deltas.

### 3.4 Control image path
- Input: a **pre-computed depth map** image (client supplies it; ComfyBox does NOT run
  Depth-Anything — keep the engine preprocessor-free, same contract as z-image ControlNet which
  takes `control_image_data`). Optionally accept a raw image + `control_preprocess:"depth"` later.
- Encode via `Krea2VAE.encode` → control latent; patchify to match token grid; cache for all steps.
- Resolution/patch alignment must equal the noisy latent grid (assert equal H/W in latent space).

### 3.5 Forward pass
- `Krea2Transformer.callAsFunction` gains optional `control: MLXArray?`. When non-nil:
  `let img = first(imgIn) + firstControl(control)` (option B). Threaded from
  `Krea2Pipeline` denoise loop, which holds the cached control latent.

### 3.6 API surface (`WarmServer` krea2 generate path)
New optional request fields (snake_case → camelCase), mirroring z-image ControlNet:
- `krea2_control: bool` / inferred when `control_image_data` + model family krea2.
- `control_image_data` (base64 depth map) or `control_image_path`.
- `controlnet_strength` (default 1.0).
- `controlnet_lora_path` (default the downloaded `depth-control-lora.safetensors`).
Returns normally (image to gallery/output_path). No change to non-control krea2 calls.

## 4. Technical acceptance criteria
1. **No regression**: krea2 generate/img2img with no control fields produces byte-comparable
   output to pre-change (base `first`, no LoRA) for a fixed seed.
2. **Load**: loader parses `depth-control-lora.safetensors`, asserts 28×8 A/B + expanded `first`;
   clear error on shape/key mismatch (no silent no-op, per the deedee-symlink lesson).
3. **Depth fidelity**: given a depth map + Krea-Kira stack + Kira prompt, output follows the
   depth structure (subject pose matches) — qualitative match ≥ z-image ControlNet at strength 0.7.
4. **Identity preserved**: Kira identity (Krea-Kira) holds with control active (Pinay trigger).
5. **Compose**: control-LoRA + KNP + Pinay all applied simultaneously without NaN/melt.
6. **Perf/memory**: single krea2-turbo 8-step control render fits current VRAM headroom; no OOM;
   render time within ~2× a plain krea2 render.
7. **Enable/disable**: control path fully inert when fields absent (gated `controlFirst`).
8. **Reference-parity (golden)**: same depth latent + seed through the reference `k2_lora.py`
   vs ComfyBox → first-step velocity `allclose` (strongest port-correctness test).
9. **Strength-0 == baseline**: control fields present, `controlnet_strength`=0 → output ≈ no-control.
10. **Config matrix pinned**: crit #3-#5 must pass on the **deployed q8-turbo**; if not, bf16-Raw
    fallback becomes ship config (accept crit #6 VRAM implication).
11. **VLM rubric**: crit #3 "follows depth" scored by the qwen-VLM QA loop against a fixed
    pose-keypoint question set (not vibes); include one non-square aspect case (grid-alignment).

## 5. Risks & mitigations
- **MLX channel concat / patch layout** differs from PyTorch token layout → assert latent grid
  equality; unit-test the concat vs summed-proj equivalence numerically.
- **Dtype**: file weights fp32; ComfyBox krea2 runs bf16/q8 → cast on load; watch precision on
  the expanded proj.
- **LoRA stacking** (control + identity) may exceed a single-LoRA assumption in `LoRALinear` →
  verify multi-adapter summation; add a test.
- **VAE mismatch**: confirm ComfyBox `Krea2VAE` == Qwen-Image VAE the LoRA was trained against
  (scale/shift, channel count). If different, depth latent won't align.
- **q8 base**: control-LoRA trained on raw/bf16; verify it works on the q8 krea2-turbo we run,
  else pin bf16 for control renders.

## 6. Test plan
- Unit: loader key/shape validation; concat-vs-parallel-proj numeric equivalence; multi-LoRA sum.
- Golden: fixed-seed no-control regression (crit #1).
- Integration: depth map (reuse `depth_reverse.png`) + Krea-Kira + Kira prompt → assert pose
  follows depth + identity holds (visual + the qwen-VLM QA loop).
- Compare head-to-head vs current z-image→krea2 two-stage on the reverse-cowgirl keyframe.

## 7. Rollout
- Branch in comfybox; build `.build/release/ComfyBox`; **Developer-ID re-sign** (GUI or unlocked
  keychain) per single-session deploy policy; restart WarmServer (`launchctl kickstart`).
- Feature is opt-in (fields absent = today's behavior), so low-risk incremental deploy.
- Ship depth first; canny/pose = follow-up (train with the reference `trainer/`).

## 8. Out of scope
Depth-Anything preprocessing in-engine; canny/pose/tile checkpoints (trainable later); ComfyUI
node parity; changing the z-image ControlNet path (kept as fallback).
