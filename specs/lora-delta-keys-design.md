# LoRA applicator: `.diff` / `.diff_b` / `.set_weight` support

**Status:** approved 2026-08-03 · **Task:** #16 · **Driver:** Kroma v0.1 (Krea 2)

## Problem

`LoRAWeightLoader` recognises three key shapes: `lora_down/lora_up`, `lora_A/lora_B`,
`lokr_w1/lokr_w2`. LoRAs that ship bare parameter deltas are **silently partially
applied** — no error, no warning, success reported.

Concrete case, verified from the safetensors header: Kroma v0.1 = 687 tensors —
264 `lora_A` + 264 `lora_B` + **159 `.diff`**. The `.diff` tensors are all norm and
modulation parameters (`prenorm.scale`, `postnorm.scale`, `qknorm.{k,q}norm.scale`,
`mod.lin`, `txtfusion.*`, `last.modulation.lin`) — the global look controls. ComfyBox
would apply 62% of the file and drop exactly the part that shifts tone/contrast.
ComfyUI applies all 687 (verified live: zero "lora key not loaded" warnings).

This is the fourth silent-partial-application defect found on 2026-08-01/02, after
the LTX transformer remap, the runtime LoRA key rename, and the upsampler that bound
zero parameters. The fix therefore includes a guard, not just the new key support.

Reference implementation: `~/Projects/ComfyUI/comfy/lora.py` (~lines 68–90):
`.diff` → `W += strength × diff`; `.diff_b` → same for bias; `.set_weight` →
replacement (ignores strength).

## Design decision: patch-and-snapshot (Approach A)

Deltas have no rank structure, so adapter wrappers buy nothing. Apply
`param += scale × diff` **directly to the parameter**, snapshotting the original
first; `clearDynamicLoRA` restores snapshots.

- Zero per-step compute cost; identical mechanics for norm scales, biases, full weights.
- `QuantizedLinear` targets reuse the existing dequantize→add→requantize path from
  `applyToTransformer`.
- Snapshot memory worst case for Kroma ≈ 430 MB bf16 — acceptable.

Rejected: **B** — wrap norm modules in dynamic adapters (new wrapper type per module
kind, per-step cost, no benefit). **C** — bake-only support (Kira's presets and the
desktop apply LoRAs at runtime; bake-only serves neither caller).

Snapshot-restore over reload-on-clear because `Krea2Pipeline.setControlLoRA` clears
and re-applies the dynamic stack **every control render**, and its contract requires
control-OFF to be byte-identical to base.

## Components

1. **Loader** — parse the three suffixes into `deltas: [String: DeltaPatch]` on
   `LoRAWeights`; `DeltaPatch = enum { diff(MLXArray), diffBias(MLXArray),
   setWeight(MLXArray) }`. Loader records a `declaredTensorCount`; a tensor key
   matching **no** known suffix is a **load error**, not a skip.
2. **Applicator** — `applyDeltas(to:weights:scale:)`: walk module tree by key path
   (same traversal as `applyDynamically`), snapshot target param into an undo store
   keyed by path, patch. `set_weight` replaces and ignores scale (ComfyUI parity).
   Shape guard: auto-transpose a transposed 2-D delta (mlx-chroma's defensive
   pattern); any other mismatch throws.
3. **Bind-count guard** — both apply paths return applied counts; caller compares
   with loader counts and **throws `LoRAError.partialApplication(missing:)`** on any
   shortfall. Never warn-and-continue.
4. **Restore** — `clearDynamicLoRA` also restores all snapshots (and clears the
   store), keeping the `setControlLoRA` clear/re-apply cycle idempotent.

## Tests (TDD — written first, production-config first)

Fixture: test-written safetensors, 2-layer toy model, LoRA containing all five key
forms **plus one junk key**.

- deltas change output; scale honored; `set_weight` replaces (ignores scale)
- apply → clear → apply is byte-identical (the `setControlLoRA` cycle)
- junk key throws at load; missing-target key throws `partialApplication`
- transposed 2-D delta auto-corrects; higher-rank mismatch throws
- delta on `QuantizedLinear` survives dequant→requant within tolerance
- integration: real Kroma binds 687/687 or throws (guarded by file-exists skip)

## Scope

Krea2 path first (Kroma is the driver). LTX merge path already has its own guard.
No UI. Rank variants (-rl r512, -rl-mild r384) are structurally identical to base
r256 — same 687 keys — so no extra work.
