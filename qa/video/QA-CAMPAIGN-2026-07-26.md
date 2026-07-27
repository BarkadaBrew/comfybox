# LTX-2 SexGod QA Campaign — 2026-07-26

Directive: "Build the fixes and generate videos to identify issues for improvement. I2v and then t2v."

## Fixes under test this run
- **Male-only face-anchor** (`LTX2_FACE_ANCHOR_MALE_ONLY=1`): skip the largest detected face (Kira, held by seed+LoRA); anchor only peripheral (male) faces. Kills the pointless anchoring that seamed/damped the primary subject.
- **Feathered anchor mask** (`LTX2_FACE_ANCHOR_FEATHER=2`): box-blurred falloff replaces the hard 0→1 binary edge (the chest-seam cause).
- Baseline env: `LTX2_REFINE_MAX_VOL=20000`, `LTX2_GUIDANCE_RESCALE=0.5`, `LTX2_FACE_ANCHOR_STRENGTH=0.35`.
- Recipe: i2v strength 0.6, img_compression 30, guidance 2.0, seed 43 (KIRA_APPROVED_SEED), character=kira.

## Metrics legend
`action_ratio` = hip-band motion / head-band motion (portrait framing). >1.5 = real driven action; ~1.0 = idle sway. `flicker` = std of frame-to-frame brightness delta (8mm flashing). `sat_range` wide = color instability.

---

## I2V results

### i2v_01_faceanchor_verify (cowgirl, 97f, fix-verification)
Metrics: action_ratio **3.67** (hip 5.43 / head 1.48), motion_mean 3.91, flicker 0.246 (low), sat_mean 21.9, sat_range [18.6, 43.6], bright 42.7. Output 896×1664 (refine ran).

**FIXES CONFIRMED WORKING:**
- ✅ Male-only anchor: male partner's face held stable across the whole clip (no drift). Kira no longer anchored.
- ✅ Feathered mask: **no visible hard chest seam** on Kira's torso — the box-blur removed the boundary line.
- ✅ Motion strong and hip-driven (action_ratio 3.67), no idle-sway regime.
- ✅ Kira identity/build correct (morena, slender, small natural bust, correct face) and consistent.
- ✅ Low flicker (0.246) — no 8mm flashing this pass.

**DEFECTS FOUND:**
1. **[HIGH] Progressive color/tone drift within the clip.** Contact sheet: frame 0 (seed) is a warm, correctly-lit natural morena tone; skin darkens and *muddies* monotonically toward the clip end, losing golden undertone (sat_range swing 18.6→43.6). This is a SINGLE 97f pass — no chunk chaining — so it's in-clip temporal drift (denoise or causal-VAE accumulation), not a join artifact. #1 visible issue.
2. **[MED] Mottled/blotchy skin** on Kira's torso, worsening in later frames (correlates with the darkening). The persistent reticulated-patch mottle.
3. **[MED] Male anatomy at contact point** — glans not clearly formed on out-stroke (#41, under-trained male anatomy in the LoRA).

### Candidate root cause for color drift (code lead)
`LTX2Decoder3D.swift`: plain decode (`callAsFunction`, L200-203) applies `x = noise·0.025 + 0.975·x` (decode_noise). The **streamed** path (`decodeStreamed`, L296) **omits this step** → level mismatch between paths. Separately, the monotonic frame-axis darkening most likely comes from causal-conv streaming cache warmup (later frames carry more accumulated state). Plan: (a) confirm drift universal across batch; (b) plain-vs-streamed decode A/B on one 97f clip; (c) if streamed is the culprit, fix cache warmup / add the missing noise step for parity.

## T2V results

_(populated as each render lands)_

## Confirmed issues → fixes
_(rolled up at end)_
