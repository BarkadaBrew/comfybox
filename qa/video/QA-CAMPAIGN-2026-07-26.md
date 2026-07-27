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

### i2v_02_slow_grind (97f)
Metrics: action_ratio **2.57** (hip 4.00 / head 1.56), flicker 0.151 (very low), sat_mean 22.0, sat_range [19.4, 43.6]. Motion appropriately lower than the fast bounce → recipe responds to prompt. Male identity held, no chest seam (fixes robust). **Same monotonic color darkening** — sat_range swing identical to case 1 → drift is UNIVERSAL, prompt-independent.

### i2v_03_fast (97f, vigorous)
Metrics: action_ratio **5.11** (hip 8.47 / head 1.66) — highest, vigorous. motion_mean 6.19, flicker 0.435. Consec-frame check: body coherence HOLDS frame-to-frame (hand sweeps, body shifts, anatomy stays coherent) → the elevated flicker is **genuine fast motion, not coherence breakdown**. Motion system scales cleanly to high action. Male held. Minor: small blurry artifact bottom-right; glans issue persists. Same baseline color drift.

### ✅ FIX BUILT: temporal color anchor (commit 2d7c212)
Offline PoC on case 1 output (zero GPU): per-frame per-channel mean/std → frame 0, strength 1.0 → **sat spread 25.0→2.1, brightness spread 6.7→0.2**, warm morena tone held across the entire clip, no banding (visually verified before/after). Implemented in `LTX2PostProcess.stabilizeColor`, env `LTX2_COLOR_ANCHOR` (default 0.9), applied once in `extractFrames`. Built clean; deploy after baseline batch completes, then validate live.

### Candidate root cause for color drift (code lead)
`LTX2Decoder3D.swift`: plain decode (`callAsFunction`, L200-203) applies `x = noise·0.025 + 0.975·x` (decode_noise). The **streamed** path (`decodeStreamed`, L296) **omits this step** → level mismatch between paths. Separately, the monotonic frame-axis darkening most likely comes from causal-conv streaming cache warmup (later frames carry more accumulated state). Plan: (a) confirm drift universal across batch; (b) plain-vs-streamed decode A/B on one 97f clip; (c) if streamed is the culprit, fix cache warmup / add the missing noise step for parity.

## T2V results

_(populated as each render lands)_

### #43 Dan's PE recipe — DRAFT ready (not deployed)
Corrected MOTION_SYSTEM drafted at `scratchpad/qa6h/dans_pe_motion_system_v2.txt`, grounded in the validated high-action prompt. Fixes: caption-driven (describe anatomy+action, NOT motion-only), female-driven framing, mandatory "Pinay"+"deep morena skin", tempo word as motion lever, explicit anti-mottle ban (no glistening/wet/oiled/sheen on dark skin), 80-120w prose. Needs render-validation via the enhance path, then deploy to coffeeshop-server (gated push→ssh→webhook flow).

## CODEX REVIEW 2026-07-27 + baseline reframe (MAJOR redirect)
Codex review (with rendered frames attached) + the CRF test + the t2v baseline converge on a new understanding:

- **The color anchor was a band-aid.** Defaulted OFF (commit pending). Root cause is i2v appearance drift, not a decode bug.
- **CRF-30 conditioning was self-inflicted damage.** My QA batch forced `img_compression=30` as a motion lever; the CRF test proves comp30 SMOOTHS/plasticizes the seed (removes skin texture → the "8mm/soft" look). Production default is already comp2 (sharp). Wrong lever — motion should come from CFG/STG, not by degrading the anchor.
- **t2v baseline (t2v_01) is SHARP, drift-free, mottle-free** (sat_range [21.9,25.1] tight vs i2v [19,45]). Proves the model produces the target quality natively; mottle+softness+drift are **i2v-path artifacts**, NOT inherent to dark morena skin. i2v was degrading native quality.
- **Root-side fix already existed, disabled:** `LTX2_REANCHOR_INTERVAL=0`. Codex recommended exactly this (periodic low-strength appearance re-anchor). Enabled at 24/0.15.
- **Refine-skip cliff:** 289f join skipped refine (>20k gate) → soft. Needs a real policy (chunk/lower-dims), not silent decode-only.
- **Decode-noise lead was dead** — LTX-2.3 disables VAE timestep conditioning (`LTX2ModelConfig.swift:115`), so that path is inert. Dropped.

### Baseline metrics (old recipe, for before/after)
- i2v_04_12s_join (289f single-pass): action_ratio 1.63, refine SKIPPED (soft), same drift.
- t2v_01_solo: action_ratio 1.02 (weak motion), fidelity EXCELLENT, no drift/mottle. Identity≠Kira (no seed). = the fidelity target for i2v.

### EXPERIMENT B1 (running): Codex correct-levers recipe
Same cowgirl scene / seed 43, but: img_compression **8** (sharp), STG **1.0**, re-anchor **24/0.15**, color-anchor **off**, cfg 2.0, strength 0.6. Latent dumped (`/tmp/kira_exp.safetensors`) for the decode A/B. Question: does sharp-seed + STG hold motion while recovering t2v-level fidelity and killing drift/mottle at the root?

### B1 RESULT (crf8 + STG1.0 + reanchor24/0.15 + anchor-off)
**WINS (the big three, all fixed):**
- Fidelity: sharp, fine skin texture restored (vs plasticky comp30). Approaching t2v level.
- Mottle: GONE. Confirms mottle was a comp30 compression artifact, not skin/LoRA.
- Color drift: FIXED at root — sat_range spread 7.4 (vs 25 baseline), richer sat 39. Re-anchor works; confirms drift is denoise-side (so the offline decode A/B is moot).
**REGRESSIONS:**
- Motion weak (action_ratio 1.02, near-static mid-clip). Sharp comp8 seed re-froze motion; STG didn't recover it.
- Late-clip catastrophic breakdown: frames 0–77 calm, **77–96 explode** (f87→88 Δ73.8). = the "last-third flashing". NEW to B1 (baseline flicker 0.246 → 0.966). Re-anchor boundaries (f24/48/72) are NOT the spikes.
**Ablation running — B2: STG OFF, else identical.** If B2 stabilizes → STG=1.0 caused the chaos. If still chaotic → sharp-seed divergence / reanchor / sampler. Then tackle weak motion via strength↓/cfg↑ or a moderate comp15-18 middle ground (motion without full comp30 fidelity loss).

### B2 RESULT (crf8, STG OFF, reanchor) — fidelity target ACHIEVED, motion missing
- STG=1.0 confirmed as the chaos cause: late breakdown 20 frames (B1) → 1 residual event (B2); flicker 0.966→0.629.
- Color PERFECTLY consistent across the whole clip (sheet), sharp texture, no mottle, Kira+male held. This is the quality bar.
- Motion near-zero (action_ratio 1.14, frames 1–7 near-identical). The f88 event is an abrupt-but-coherent late pose shift (same index in B1+B2 — likely tail/reanchor-96 related, not corruption).
- **Problem now cleanly isolated to MOTION on top of a great static base.** Levers to test (clean→costly): strength↓ (B3, running), CFG↑ (2.5), frame_rate↓ (LTX temporal-RoPE motion lever, untried), moderate comp15-18 (partial seed freedom, small fidelity cost).

## Confirmed issues → fixes
_(rolled up at end)_
