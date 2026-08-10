# LoRA applicator: `.diff` / `.diff_b` / `.set_weight` support

**Status:** rev 2, post-Codex review 2026-08-03 · **Task:** #16 · **Driver:** Kroma v0.1 (Krea 2)
**Review:** codex-fdd-review (9 findings, 3 critical) — all folded in below.

## Problem

`LoRAWeightLoader` recognises three key shapes: `lora_down/lora_up`, `lora_A/lora_B`,
`lokr_w1/lokr_w2`. LoRAs shipping bare parameter deltas are **silently partially
applied** — no error, success reported.

Kroma v0.1 (verified from the safetensors header): 687 tensors = 264 `lora_A` +
264 `lora_B` + **159 `.diff`**. The `.diff` tensors are all norm/modulation
parameters (`prenorm.scale`, `postnorm.scale`, `qknorm.{k,q}norm.scale`, `mod.lin`,
`txtfusion.*`, `last.modulation.lin`) — the global look controls. ComfyBox today
applies 62% and drops exactly the tensors that shift tone/contrast. ComfyUI applies
all 687 (verified live, zero unloaded-key warnings; visual A/B confirmed the deltas
carry the film-look).

Reference semantics: `~/Projects/ComfyUI/comfy/lora.py` — `.diff` → `W += strength ×
diff`; `.diff_b` → same on the **real `.bias` parameter**; `.set_weight` → replacement
that overwrites all prior work on that weight (ordering matters, lora.py:413/463).

## Architecture (rev 2): typed patch plan → preflight → transactional commit

The rev-1 "patch-and-snapshot with a count guard" was rejected in review for three
critical holes: scalar bind counts can't detect partial fan-out binds; throwing
after mutation strands a partial stack; and snapshotting an `MLXArray` parameter
stores an alias, not an undo value. Rev 2 replaces it with:

### 1. Loader → `DeltaPatch` set + key classification

`LoRAWeights` gains `deltas: [String: DeltaPatch]`:

```swift
enum DeltaPatch { case diff(MLXArray), diffBias(MLXArray), setWeight(MLXArray) }
```

`.diff_b` keys map to the target's real `bias` parameter path (no invented keys).

**Key classification, not blanket strictness** (finding 6). Every tensor key falls
into exactly one class:

| class | examples | behaviour |
|---|---|---|
| bindable | pair/lokr/delta suffixes | must bind or the apply fails |
| metadata | `.alpha`, safetensors `__metadata__` | consumed, never counted as targets |
| known-unsupported | `dora_scale`, `w_norm`, `b_norm`, `lora_mid`, `reshape_weight` | **load error naming the feature** (explicit unsupported ≠ silent skip) |
| out-of-scope prefix | text-encoder keys on a transformer-only apply | logged + reported in the result, not fatal (composite adapters stay loadable for their transformer half) |
| unknown | anything else | load error naming the key |

### 2. Patch plan with source→target identity tracking (finding 1)

Loading produces a **plan**: `[PatchOp]` where each op records its *source keys*
(from the file) and its *resolved target parameter path* in the model. One QKV pair
fanning out to three targets becomes three ops sharing source keys. The guard then
compares **sets** — every bindable source key must appear in ≥1 committed op, every
op must resolve — and a failure lists the exact missing keys. No scalar counts
anywhere.

Ordering (finding 4): ops apply in file order per target; a `set_weight` op on a
target **rejects** coexistence with pair-adapters on the same target (ComfyUI's
sequential overwrite semantics can't be reproduced across our hybrid
wrapper/base-patch split — reject loudly rather than differ silently). Kroma has no
`set_weight`, so this is a guard rail, not a feature.

### 3. Parameter-path index, not module traversal (finding 7)

Kroma's key targets (`prenorm.scale`, `qknorm.*.scale`, raw modulation arrays) are
**leaf parameters, not Linear modules** — `applyDynamically`'s traversal cannot
reach them. Build a flattened `[String: MLXArray]` index of the transformer's
parameter tree (`module.parameters().flattened()`) and resolve delta targets
against it directly.

### 4. Transactional apply (finding 2)

```
plan = try makePlan(weights, paramIndex)     // resolves ALL targets — throws
                                             // BEFORE any mutation
try txn.begin(plan)                          // snapshots every target (see 5)
txn.commit()                                 // mutates; any internal failure →
                                             // txn.rollback() restores all
```

`Krea2Pipeline.loadLoRAs` keeps its current shape but a failure at config N rolls
back N's partial work AND the earlier configs remain intact and tracked — the
pipeline never ends in a state where applied weights and `appliedLoRAs` disagree.

### 5. Snapshot store: detached copies, per-transformer, first-write-wins (finding 3)

- Snapshot = **detached evaluated copy** (`MLXArray(copying:)` + `eval`), never the
  parameter object (MLX params are references mutated in place by `update`).
- Store is **owned by the transformer's applicator state** (instance-scoped object
  handed to LoRAApplicator calls), not a global keyed by path — two pipelines in one
  process must not collide.
- **First-write-wins**: when stacked LoRAs patch the same path, only the first apply
  snapshots; clear restores the true base, not an intermediate.
- Quantized targets (finding 5): snapshot the **exact packed tuple**
  (weight, scales, biases) and restore it directly on clear — never requantize a
  dequantized snapshot. Stacked deltas on one quantized target accumulate against a
  single dequantization and quantize **once**.

### 6. Scaling (finding 8) — explicit, delta ≠ pair

- pair: `userScale × layerAlpha / layerRank` (existing behaviour)
- `.diff` / `.diff_b`: `userScale × delta` — **never** alpha/rank scaled
- `.set_weight`: replacement; ignores userScale entirely (even 0)

### 7. Restore

`clearDynamicLoRA` (and the txn rollback) restores all snapshots and empties the
store. `setControlLoRA`'s clear → re-apply cycle must be byte-identical for
control-OFF; with exact packed-tuple restore this holds for quantized targets too.

## Tests (finding 9 — production path, real lifecycle)

Fixtures: test-written safetensors + a **toy Module with the real shapes**: a biased
`Linear`, a `QuantizedLinear`, and a bare `scale` leaf parameter (the Kroma target
kinds). Krea2-path tests go through `loadForKrea2`, not the generic loader.

1. deltas change module **output**; scale honored; delta unaffected by alpha≠rank
   (fixture rank 4 / alpha 2: pair gets 0.5×, delta gets 1.0×)
2. `.diff_b` patches the real bias of a biased Linear
3. delta on a bare (non-module) scale param applies via the param index
4. apply → clear → apply byte-identical; **two stacked LoRAs on the SAME param**,
   clear restores true base (first-write-wins)
5. failure at config N rolls back N, leaves earlier stack intact and tracked
6. missing-target key → error **before any mutation** (preflight), names the key
7. QKV-style fan-out: two of three targets binding = failure listing the third
8. quantized: packed weight/scales/biases restored **exactly** over repeated
   apply/clear cycles (not "within tolerance")
9. `set_weight` + pair on same target → rejected loudly
10. key classification: `.alpha` consumed silently; `dora_scale` errors naming
    DoRA; junk key errors naming the key; text-encoder keys reported not fatal
11. transposed 2-D delta auto-corrects (mlx-chroma defensive pattern); other
    mismatches throw
12. integration (Bolt file present, else skipped locally — CI relies on 1–11):
    real Kroma via `loadForKrea2` + Krea2 transformer: all 423 ops commit, zero
    unresolved, output actually changes

## Scope

Krea2 path first (Kroma is the driver). LTX merge path keeps its own guard.
Estimate: ~2 days including the transaction layer and tests. Rank variants
(-rl r512 / -rl-mild r384) are key-identical to base r256 — no extra work.
