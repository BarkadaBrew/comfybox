# HANDOFF — LTX video quality, 2026-08-02

Continues `HANDOFF-ltx-quality-2026-08-01.md` (band bug, single-chunk, two-stage).
**Objective context:** ComfyBox must be a viable alternative to ComfyUI for LTX video
and image-gen work, and the desktop suite is a product with features — not a renderer
with a UI bolted on.

---

## 1. What shipped to production today

| Change | Where | Status |
|---|---|---|
| Adaptive conv chunking (M·K ≤ 2^31) — fixes the refine band | `CausalConv3d` | LIVE, committed `2245aab` |
| Distil baked + int8 quantized (v1.7) | `/Volumes/Bolt/Models/pinkcherry-v17-distill06-int8` | LIVE, 2.2x faster (417s → 192s) |
| Two-stage refine ON; request dims = FINAL (server halves for stage 1) | `WarmServer` | LIVE |
| Single-pass fold: any ≤12s request renders one chunk (t2v included) | `WarmServer` | LIVE |
| Runtime LoRA merge key remap — was applying 388/584 layers | `LTX2VideoGenerator` | LIVE |
| `guidance` forwarded to `generateT2V` — was silently dropped | `LTX2VideoGenerator` | LIVE |
| T2V guidance 3.5 → 1.0 | daemon `video-tools.ts` `91950b07` | LIVE |
| `LTX2_GUIDANCE_RESCALE` → 0 | plist | LIVE |

## 2. The t2v chroma bug — CLOSED

**Cause: cfg 3.5, which today's `guidance` fix delivered to t2v for the first time.**
Before it, t2v silently ran cfg 1.0 no matter what the daemon sent, so every
"previously good" t2v render was already at the author's setting and 3.5 had never
been exercised.

Matched seed 4242, identical prompt/dims:

| config | sat | bri | pixels sat>200 |
|---|---|---|---|
| cfg 3.5 + rescale 0.6 | 72.9 | 105.6 | **9.32%** |
| cfg 1.0, rescale OFF | 69.1 | 108.1 | **0.84%** |

**Guidance rescale was never a fix.** It moved only the MEAN (φ 0.5 → 87.8,
0.6 → 72.9, 0.75 → 41.9) while the cyan blotches persisted underneath. Chasing that
mean cost three renders.

> **Metric lesson:** mean saturation hides localized blowout. Use
> `frac(sat > 200)` plus the median hue of those pixels. Add it to every render check.

**i2v is NOT affected** at its current settings (blotch 0.17–1.70%), which is why the
daemon's i2v guidance stays at 3.0. Caveat: those samples used the config default —
**i2v at the daemon's actual cfg 3.0 is still unmeasured.**

## 3. NAG — implemented, validated, ON

`Sources/ZImage/LTX2/Transformer/LTX2NAG.swift` + wiring through
`LTX2TransformerBlock` → `LTX2Transformer` → `LTX2Pipeline` (t2v, i2v,
multi-keyframe, refine). 9 tests. Inert unless `LTX2_NAG_SCALE` is set.

```
z_guid  = z_pos*scale − z_neg*(scale−1)
ratio   = ‖z_guid‖ / ‖z_pos‖          ← PER attention vector, not global
z_clamp = ratio > tau ? z_guid*(tau/ratio) : z_guid
z_out   = alpha*z_clamp + (1−alpha)*z_pos
```

Why it matters: CFG extrapolates the final prediction with **no magnitude bound**, so
buying adherence with cfg drives channels into saturation. NAG extrapolates inside
cross-attention and renormalises, so adherence rises while magnitude stays capped at
`tau × ‖positive‖`. It rides the POSITIVE pass only — applying it to the
unconditional pass would double-count against the baseline CFG++ steps along.

**Reference recipe, verified from the author's own artifacts** (API graph embedded in
his published mp4s: `ffprobe -v error -show_entries format_tags=comment`; plus the
v1.8 workflow JSON). **Unchanged across v1.6 / v1.7 / v1.8:**

- CFG **1.0** both passes · NAG **scale 11.0, alpha 0.25, tau 2.5**
- samplers `euler_ancestral_cfg_pp` → `euler_cfg_pp` · `img_compression` 22
- distil LoRA 384-1.1 @ **0.6** (community: 0.6 pass 1 → **0.45** pass 2)
- stage 1 at **0.5×** (confirms our two-stage dims convention)
- NAG is a MODEL PATCH fed by dedicated negative video+audio conditionings:
  `logos, voice over, narration, off camera speech, watermarks, poor anatomy,
   low detail, slow motion, slow, boring`

**A/B at seed 4242, cfg 1.0:**

| | sat | bri | blotch | drift sat q1→q4 |
|---|---|---|---|---|
| NAG off | 69.1 | 108.1 | 0.84% | −20.2% |
| NAG on | 71.0 | 101.5 | **0.16%** | **−42.8%** |

Adherence visibly improved — the prompt's third beat ("brushes hair from her face,
looks toward the window") actually executes, which cfg 3.5 was meant to buy and
couldn't. Chroma improved. **But NAG more than doubles the late-clip drift** (§4).

## 4. OPEN — late-clip exposure/desaturation drift

Todd observed it; confirmed by measurement. On a 97f t2v: **brightness +8.0%,
saturation −20.2%** from first quarter to last, inflecting at 60–70%.

**Refine EXONERATED.** Single-stage (`LTX2_TWO_STAGE=0`, same seed and stage-1 dims)
drifts identically: +7.2% / −18.1%. **It is the BASE denoise pass.**

**t2v-specific.** i2v saturation is flat (−1.3% to −0.0%; one clip even +17.7%).
Mechanism hypothesis: i2v re-injects the clean frame-0 latent every step, which
anchors color for the whole clip; t2v has nothing holding it. The mode without an
anchor drifts; the mode with one doesn't.

**NAG amplifies it** to −42.8%, presumably because as positive conditioning weakens
across frames, NAG's extrapolation grows relatively stronger and magnifies the
existing direction. So the drift fix is now load-bearing, not cosmetic.

**Next tests, in order:**
1. `LTX2_COLOR_ANCHOR` (`LTX2PostProcess.stabilizeColor`, currently 0) — already
   built for this exact symptom. Cheapest possible mitigation; A/B it with NAG on.
2. Root cause discriminator: swap base-pass `euler_ancestral_cfg_pp` →
   `euler_cfg_pp`. If drift collapses, ancestral noise accumulation is implicated;
   if not, suspect temporal conditioning decay (same family as the known
   intra-chunk motion decay).
3. Matched-seed i2v pair to confirm the anchor hypothesis properly.

## 5. Model upgrade — v1.8 (staged, NOT switched)

Author released v1.8 while we worked. **Do the pipeline fixes first**, then upgrade as
a single clean variable against the seed-4242 baseline.

**Bake from bf16 — do NOT use his int8.** His int8 is `I8` + `F32 .weight_scale`
(torch-style, 1344 quantized tensors, 4603 left bf16). Ours is MLX affine group-wise
with `.scales`/`.biases` at group 64 — `LTX2Quantizer.applyQuantizedLayout` would not
recognise his. Also his has no distil baked, and merging a LoRA into an
already-quantized checkpoint costs dequantize→merge→requantize (two roundtrips).

Procedure (~15 min after download):
```bash
python scripts/bake_ltx2_lora.py --base <v1.8 bf16> \
  --lora .../ltx-2.3-22b-distilled-lora-384-1.1.safetensors --scale 0.6 --out <bf16 baked>
ComfyBox quantize-ltx2 -i <bf16 baked> -o <int8 dir> --bits 8 --group-size 64
```

## 6. Ops rules learned the hard way (cost ~2h today)

- **TCC grants are keyed to the CODE SIGNATURE.** The daemon reads models from
  `/Volumes/Bolt` (removable volume). `swift build` re-signs ad-hoc → the grant no
  longer applies → the daemon **hangs forever** inside `open()` waiting on a
  permission prompt. Symptom: 25MB RSS, 0% CPU, last log line
  `ComfyImageCache: using …`. It does not error; it looks like a slow model load.
  **Always** `touch ~/.comfybox/resign-request` and verify
  `codesign -dv` shows `TeamIdentifier=STHPB624H2` immediately before `bootstrap`.
- **Never bounce while a build is running** — a build finishing after the bounce
  swaps an ad-hoc binary underneath and the *next* restart hangs.
- **Warn the user before any action that can block on a permission dialog.**
- **Don't run the unit suite and a video render concurrently** — the VAE tests take a
  large bite and the render admission gate needs 40GB; the render fails to admit.
- `launchctl kickstart -k` does NOT re-read the plist. Use `bootout` + `bootstrap`.

## 7. Roadmap

1. **Drift fix** (§4) — now gating, because NAG doubles it
2. v1.8 upgrade (§5) against the seed-4242 baseline
3. Per-pass distil split (0.6 → 0.45 on the refine)
4. Measure i2v at the daemon's real cfg 3.0
5. Text-encoder A/B: ours `gemma-3-12b-heretic-q8` vs reference bf16
6. Audio — resume mid-Phase A (see the 08-01 handoff)
7. Specs queued: `motion-tab-prompt-lab.md`, `ltx2-parameter-externalization.md`,
   `prd-prompt-optimizer-finetuning.md` — all three have an adversarial Codex review
   with 14 ranked findings still to be folded in (stable artifact IDs instead of
   `output_path`, `schema_version` + `task_kind` now to avoid a migration, retention
   vs Gallery-rating contradiction).
