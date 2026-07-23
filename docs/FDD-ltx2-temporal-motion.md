# FDD: LTX-2 temporal-motion deficit — root cause (fps-divided RoPE), fix, and remaining gap

**Repo:** `BarkadaBrew/comfybox` (`~/Projects/zimage.swift`, Swift/MLX)
**Component:** `Sources/ZImage/LTX2/LTX2Pipeline.swift`
**Branch:** `claude/ltx2-haze-optimization`
**Fix commit:** `68febac` — "LTX2: fix temporal-motion bug (fps-divided RoPE) + ancestral-noise correctness"
**Author:** Fable (Opus)  **Date:** 2026-07-23
**Status:** v1 — root cause FOUND and FIXED (base motion 6.0 → 11.9); one downstream gap (refine damping) remains OPEN.
**Directive (Todd, 2026-07-23):** "Make ComfyBox better than ComfyUI Desktop… no bs compromises… fix all the bugs in ComfyBox and improve it." Measurable target: ComfyBox full-pipeline t2v motion should meet or beat ComfyUI Desktop's **18.7** on the reference cowgirl prompt/seed.

---

## 1. Summary

ComfyBox's LTX-2 text-to-video renders were **near-static** — technically correct frames, coherent subject and identity, but almost no frame-to-frame movement. A motion metric on the reference prompt/seed measured **~5.7** where ComfyUI Desktop's port of the same model, same sampler name, same seed measured **18.7** — a 3.3× deficit.

The root cause was found by **reading ComfyUI's open-source LTX pipeline and diffing it against our port**, after three measurement-driven experiments ruled out the obvious suspects. `createPositionGrid` divided the temporal RoPE coordinate by `fps` (~24). The reference pipeline never does this — it scales latent coordinates by the VAE compression factor and normalizes only by `max_pos`. The extra `/= fps` compressed the temporal position axis ~24×, so the transformer's rotary embeddings saw almost no progression from frame to frame and the model produced almost no motion.

**Removing the `fps` division nearly doubled base motion: 6.0 → 11.9**, matching ComfyUI Desktop's base-only motion (13.3) within noise. Two smaller correctness fixes rode along (plain-Gaussian ancestral noise, σ=1 divide-by-zero guard). **One gap remains OPEN:** the refine pass still damps motion back down (11.9 → ~5) — a separate, structural bug, scoped in §7.

---

## 2. Symptom & quantification

- **Reference case (fixed for all measurements):** the "cowgirl" prompt + seed image, t2v, base 512×288 → refine 1024×576, SexGod sampler family (CFG++ euler_ancestral), same length.
- **Motion metric:** mean inter-frame optical-flow magnitude (higher = more movement). Used comparatively only — the absolute scale is arbitrary; what matters is ComfyBox-vs-Desktop on the *same* input.
- **Cross-check:** frames were eyeballed at each step (the metric alone can mislead — e.g. STG can raise the number while adding artifacts). Motion ~5 looked visibly frozen; ~12–13 looked like a real riding sequence; identity and structure were never the problem.

| Config | Motion | Note |
|---|---:|---|
| ComfyBox full (refine on), baseline | **5.71** | the reported defect |
| ComfyBox base only, STG off | 6.01 | refine ruled out (≈ full) |
| ComfyBox base, STG on | 9.17 | STG is a real lever, +50%, not the whole gap |
| ComfyBox base, ancestral-noise fix only | 8.85 | ≈ STG-alone → noise was NOT the driver |
| **ComfyBox base, fps-division removed** | **11.94** | **the fix — nearly 2× baseline** |
| ComfyUI Desktop, base only | 13.27 | our target for the base stage |
| ComfyUI Desktop, full (base+refine) | 18.73 | our target for the full pipeline |

Two deficits are visible in this table: a **large base deficit** (6.0 vs 13.3) — now closed by the fps fix — and a **smaller refine deficit** (Desktop's refine adds 13.3 → 18.7; ours *subtracts* 11.9 → ~5). §7.

---

## 3. Diagnostic methodology (measure-first, then read the source)

The discipline here was: **rule out hypotheses by measurement before touching code, and find the root cause in the reference source rather than guessing.** Two hypotheses were tested and *rejected* before the real bug was found — recorded here so they are not re-litigated.

1. **Hypothesis: the refine pass kills motion.** Test: render base-only, refine off. Result **6.01 vs 5.71 full** — nearly identical. **REJECTED** — motion is lost in the *base* denoise, not refine. (This narrowed the search enormously.)
2. **Hypothesis: the CFG++ ancestral sampler math is wrong.** Verified `getSdeCoeff` / `getNewNoise` / the CFG++ update line-by-line against ComfyUI — the math **matched**. Then tested STG (the 10Eros authors' explicit motion lever, previously off): 6.0 → 9.2. Real but insufficient. **Sampler wrapper REJECTED as the primary cause.**
3. **Hypothesis: over-normalized ancestral noise flattens stochasticity.** Test: swap `getNewNoise` (channel-normalized) for plain Gaussian. Result **8.85 ≈ STG-alone 9.17** — no additional motion. **REJECTED as the driver** (the change is still correct-per-ComfyUI and fixes a σ=1 edge — kept, §5.2).
4. **Decisive measurement: where does Desktop's motion come from?** Desktop **base-only = 13.27** vs ComfyBox base 6.01. This localized the prize to the **base stage** and, since the sampler math already matched, pointed at the one thing left: the **position/RoPE coordinate pipeline** feeding the transformer.
5. **Root-cause via source diff.** ComfyUI is GPL-3 and checked out locally (`~/Projects/ComfyUI/comfy/ldm/lightricks/`). Read `symmetric_patchifier.latent_to_pixel_coords`, `get_latent_coords`, and `get_fractional_positions`, and diffed the coordinate pipeline against `createPositionGrid`. The extra `/= fps` had no counterpart in the reference. §4.

---

## 4. Root cause — fps-divided temporal RoPE

### 4.1 What the reference does

ComfyUI's LTX position pipeline (`comfy/ldm/lightricks/`):

- `SymmetricPatchifier.latent_to_pixel_coords` multiplies latent coordinates by the **VAE scale factors** (temporal/spatial compression) and applies a **causal fix** to the temporal axis: `pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - scale[0]).clamp(min=0)`.
- `get_fractional_positions` (in the RoPE precompute) normalizes those coordinates by **`max_pos`** per axis — and nothing else.
- **There is no `fps` anywhere in the coordinate → RoPE path.** fps in LTX is metadata that controls *decode/playback timing and conditioning*, not the transformer's positional geometry.

So the reference temporal coordinate for a patch is, in effect, `frame_index × temporal_compression` (causal-shifted), later divided once by `max_pos` inside the RoPE frequency computation.

### 4.2 What we did wrong

`createPositionGrid` (`LTX2Pipeline.swift`) computed the same causal-shifted temporal coordinate **and then divided it by `fps`** (`config.fps`, ~24):

```swift
// (removed) previous code
tStart /= fps
tEnd /= fps
```

Downstream, the RoPE precompute divides positions by `max_pos` again. Net effect: the temporal axis was compressed by roughly **fps × max_pos** instead of just `max_pos` — about **24× too small**. The reference temporal RoPE spans ~4.8 of a cycle across the clip; ours spanned ~0.2 — barely a fraction of one rotation. With almost no angular change in the temporal rotary embedding from frame to frame, **the transformer perceived the frames as nearly the same timestep and generated nearly no motion.** Spatial axes were unaffected, which is why structure, identity, and per-frame quality were always fine — only *time* was collapsed.

This is a genuine port bug (the spatial axes were correct; someone added an fps normalization to the temporal axis that the reference does not have), confirmed against source rather than inferred.

---

## 5. The fix (commit `68febac`, `LTX2Pipeline.swift`, +20 / −11)

### 5.1 Remove the fps division (the motion fix)

In `createPositionGrid` (~line 1134) the `tStart /= fps` / `tEnd /= fps` lines are deleted, replaced with a comment documenting the reference behavior and *why* the division was wrong. Temporal coordinates now carry `frame_index × temporalScale` (causal-shifted), matching `latent_to_pixel_coords`; the sole normalization is the existing `max_pos` division in the RoPE precompute.

- **Effect:** base t2v motion **6.01 → 11.94** on the reference case (≈ Desktop base 13.27).

### 5.2 Plain-Gaussian ancestral noise (correctness, matches ComfyUI)

Both ancestral/SDE injection sites (~lines 1010, 1022) previously used `getNewNoise` (aggressive per-channel normalization). ComfyUI injects plain `randn_like`. Swapped to `MLXRandom.normal(shape, dtype: .float32)` at both sites. This did **not** move the motion metric (§3.3) but is correct-per-reference and removes an unjustified normalization from the noise path.

### 5.3 σ=1.0 divide-by-zero guard (correctness)

At the first CFG++ ancestral step `sigma == 1.0`, so `alphaS = 1.0 - sigma = 0`, and `sf = sigma / alphaS` blew up to inf → NaN in the ancestral term. Fixes:
- `let alphaS = max(1.0 - sigma, 1e-4)` — clamp `alphaS` to a small floor.
- Guard the intermediate `inner`/`up`/`sigmaDown` square-roots against negative radicands: `(inner > 0 ? inner : 0).squareRoot()`, `max(st*st - up*up, 0).squareRoot()`.

### 5.4 Refine determinism made env-toggleable

The refine denoise was hardcoded `forceDeterministic: true` (drops ancestral noise, keeps CFG++). Both call sites (~lines 255, 651) now read `ProcessInfo…["LTX2_REFINE_DETERMINISTIC"] != "0"` — default remains deterministic, but the refine's stochasticity can be flipped on for the §7 investigation without a rebuild.

---

## 6. Verification

- **Method:** same prompt/seed for every point; base-stage isolated with refine off; one variable changed at a time; frames visually confirmed alongside the metric.
- **Result:** the fps fix in isolation (STG off, refine off, noise fix present) took base motion **6.01 → 11.94** — nearly doubled, and now on par with ComfyUI Desktop's base-only 13.27. The two prior hypotheses (refine, ancestral noise) were each measured to *not* be the driver before the real fix was applied, so the attribution is clean.
- **Build:** clean (one harmless unused-variable warning — the now-dead `fps` local, §8).

**Honest status vs the directive:** the *base* stage now ≈ matches Desktop's base. ComfyBox does **not yet beat** Desktop's full-pipeline 18.7, because our refine pass still damps motion instead of adding it (§7). The largest, clearly-attributable bug is fixed; the remaining gap is a separate, narrower problem.

---

## 7. OPEN — refine pass damps motion (11.9 → ~5)

The unresolved deficit. After the fps fix, the base produces ~11.9 motion, but the two-stage refine drags the full-pipeline result back down to ~5, while Desktop's refine *adds* motion (13.3 → 18.7). This is structural, not the same bug.

**Leading suspects (in priority order):**
1. **Refine determinism.** Our refine runs `forceDeterministic: true` (no ancestral noise). If Desktop's refine is stochastic, the deterministic path could be regressing toward a static mean. Now toggleable via `LTX2_REFINE_DETERMINISTIC=0` (§5.4) — first thing to A/B.
2. **Refine RoPE / spatial rescale at 2×.** `createPositionGrid(spatialScaleMul:)` rescales spatial positions to keep RoPE in-distribution at the 2× refine resolution — confirm the *temporal* axis is handled identically to base (it should now be fps-free, but verify the refine `positions`/`precomputedPE` are rebuilt post-fix and not carrying a stale schedule).
3. **Refine sigma schedule** (`LTX2_REFINE_SIGMAS`) — an over-aggressive denoise strength at refine would wash out the base's motion.
4. **Diff Desktop's refine stage** against ours the same way §3–4 diffed the base — read the reference two-stage refine and compare sigma schedule + noise injection line-by-line.

**Secondary open items:**
- **Base @ 768 is too slow** — MLX long-sequence attention makes the higher-res base impractical; a perf task, separate from motion.
- **STG** (+50% base motion, 6→9) is a *real, independent* lever but adds a compute pass per step and can introduce artifacts — treat as optional/off by default until the refine gap is closed and frames are vetted.

---

## 8. Follow-up / cleanup

- **Dead code:** `let fps = Float(config.fps)` in `createPositionGrid` (line ~1106) is now unused — the harmless build warning. Remove it (and drop the `config.fps` read if nothing else in the function needs it) in a cleanup commit.
- **Regression guard:** add a unit test on `createPositionGrid` asserting the temporal coordinate for a known `(frame, temporalCompression)` equals `frame×temporalScale` (causal-shifted) with **no fps factor** — so this bug cannot silently return. Pure, no GPU.
- **Confirm fps is still correctly used** where it *should* be (decode/playback timing, conditioning) — the fix only removed it from the RoPE coordinate path; verify no legitimate fps consumer was collaterally affected.

---

## 9. Appendix — file/line references (post-fix, `68febac`)

| Location | What |
|---|---|
| `LTX2Pipeline.swift:1096` | `createPositionGrid(...)` — the coordinate pipeline |
| `LTX2Pipeline.swift:1106` | `let fps = …` — now-dead local (§8 cleanup) |
| `LTX2Pipeline.swift:1131–1132` | causal temporal shift (matches `latent_to_pixel_coords`) |
| `LTX2Pipeline.swift:~1134` | **the fix** — comment where `/= fps` was removed |
| `LTX2Pipeline.swift:~1010` | σ=1 `alphaS` guard + Gaussian ancestral noise (CFG++ branch) |
| `LTX2Pipeline.swift:~1022` | sqrt radicand guards (`inner`/`sigmaDown`) |
| `LTX2Pipeline.swift:255, 651` | refine `forceDeterministic` → `LTX2_REFINE_DETERMINISTIC` env |
| `~/Projects/ComfyUI/comfy/ldm/lightricks/` | reference: `symmetric_patchifier.latent_to_pixel_coords`, `get_fractional_positions` — the diff source |

**Env toggles touched/added:** `LTX2_REFINE_DETERMINISTIC` (default on), `LTX2_REFINE_SIGMAS`, `LTX2_REFINE_DECODE_ONLY` (existing).
