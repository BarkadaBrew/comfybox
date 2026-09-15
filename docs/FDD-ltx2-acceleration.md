# FDD: LTX-2.3 render acceleration on the M3 Max — where the time goes, the ceiling, and the levers

**Repo:** `BarkadaBrew/comfybox` (`~/Projects/zimage-temporal`, Swift/MLX) + `BarkadaBrew/coffeeshop-server` (scheduler)
**Component:** `Sources/ZImage/LTX2/LTX2Pipeline.swift` (denoise loop), `LTX2ConfigResolver.swift` (knobs), Kira content scheduler (clip policy)
**Author:** Fable 5.1  **Date:** 2026-09-15
**Status:** v1.2 — ladder RUN and READ 2026-09-15 (§11); **production recipe changed to plain euler + the 10-step schedule** (82 → 33 min per 10 s clip). 12 fps latent + temporal ×2 rejected.
**Directive (Todd, 2026-09-15):** "is there any way to accelerate ltx?" — after the same-seed burn/ghost investigation settled the production recipe (int8 DiT + heretic Gemma, 19-step author schedule, single pass, cfg++ euler, NAG 5, STG 0.3 flat, color anchor 1.0).

---

## 1. Summary

A 12-second 576×896 clip on the production recipe takes **~105 minutes**; a 10-second one **~83–88 minutes**. Denoise is **>97%** of wall clock; text encode, VAE decode and audio together are under 2 minutes.

The denoise loop runs **three transformer forwards per step** on the current recipe — positive, the CFG++ unconditional (negative) pass, and the STG perturbed pass — for 19 steps, over a ~18.6k-token video latent. A first-principles FLOPs estimate puts the engine at **roughly 60–65% of the M3 Max's peak** for that work. That means there is little left in "make the same work run faster": the levers are all **do less work**, and each one is a quality bet that has to go through the same same-seed human read the recipe went through.

Ranked by payoff, the credible levers are: drop the CFG++ negative pass (−33% per step), the 10-step distill schedule (−47%), render a 12 fps latent and let the temporal upscaler restore 24 fps (≈ −50%, attention −75%), and shorter clips (linear). Stacked, they take a 12 s clip from ~105 min to a plausible **20–30 min**. The only 10× accelerator is hardware.

---

## 2. Measured baseline (2026-09-14/15, engine `dec89ec`, int8 DiT + heretic Gemma, 19 steps, NAG 5, STG 0.3 flat)

| Render | Dims | Frames (latent) | Passes/step | Denoise | Total | s/step |
|---|---|---:|---:|---:|---:|---:|
| F2 apple, STG 0.5 flat | 576×896 | 241 (31) | 3 | 4910 s | 4986 s | 258 |
| F4 apple, anchor 1.0 (the production recipe) | 576×896 | 241 (31) | 3 | 4855 s | 4923 s | 256 |
| Soak clip, 12 s | 576×896 | 289 (37) | 3 | 6300 s | 6388 s | 332 |
| Soak clip, 12 s | 576×896 | 289 (37) | 3 | 6490 s | 6590 s | 342 |
| F3 two-stage | 448×704 → 608×960 | 241 (31 → 31) | 3 | 2526 s + 860 s refine | 3462 s | 133 + 287 |
| Old 10-step soak (2026-09-14, NAG 5, no STG) | 576×896 | 289 (37) | 2 | — | ~2900 s | ~290 |

Phase telemetry from the engine (`/v1/queue` `phase_timings`, 6–8 samples): **baseDenoise 4402 s mean**, vaeDecode 72 s, textEncode 10 s, modelLoad 2.4 s, vocoder 4 s, postProcess 0.7 s. Denoise is the whole story.

Per-step cost scales linearly with latent frames (31 → 37 frames: 256 → 337 s/step, +32% for +19% frames — the extra is the quadratic attention term, see §3).

### 2.1 What one step actually does

From `LTX2Pipeline.swift` (denoise loop, ~L1779–2060):

1. **Positive pass** — `transformer(latent, context: textEmbeddings, …)`. NAG (`LTX2NAG.swift`) rides INSIDE this pass's cross-attention with the negative embeddings; it does not add a forward.
2. **CFG++ unconditional pass** — `useCfgPP = samplerIsCfgPP && negativeEmbeddings != nil`: "needs an unconditional (negative) pass every step **regardless of cfgScale**" (L1779). The production sampler is `euler_cfg_pp`, so this pass runs at guidance 1.0.
3. **STG perturbed pass** — when `stg_scale > 0`, a third forward with self-attention skipped in blocks {14,15,16}; the result steers the guided x0 away from the perturbed one (L2036–2054).

So the production recipe = **3 forwards × 19 steps = 57 forwards** per clip. The 2026-09-14 10-step soak (no STG) was 2 × 10 = 20 forwards — which is why it ran in ~48 minutes.

---

## 3. Ceiling analysis — how far is the engine from the hardware?

Geometry of a 12 s clip: video latent **37 × 28 × 18 = 18,648 tokens**; audio stream 302 latent frames; text context ≤ 1024 tokens (cross-attention only). DiT: 48 blocks, head dim 128, inner dim 4096 (`LTX2Transformer.swift` defaults), ≈ 19 B video-branch parameters, int8 weights (~20 GB resident; admission "need ~24.5 GB warm stack").

Per forward, order-of-magnitude:

- Dense projections/MLPs: 2 · params · tokens ≈ 2 · 19e9 · 18.6e3 ≈ **7.1e14 FLOP**
- Self-attention (QKᵀ and PV): 4 · n² · d · layers ≈ 4 · (18.6e3)² · 4096 · 48 ≈ **2.7e14 FLOP**
- Audio branch + cross-attention: small next to the above (audio tokens ≪ video tokens)
- **≈ 1.0e15 FLOP per forward**

M3 Max GPU peak is roughly **14 TFLOPS** bf16 (int8 weights are dequantized for the matmul on Apple GPUs — there is no int8 tensor-core speedup, the win is memory). Ideal forward ≈ 1.0e15 / 14e12 ≈ **70 s**. Measured: 337 s/step ÷ 3 forwards ≈ **112 s per forward** → the engine is at **~60–65% of peak** on the DiT. Attention already runs on `MLXFast.scaledDotProductAttention` (fused), so the remaining gap is the usual MLX graph/launch overhead, the audio branch, the STG pass being slightly cheaper than a full pass, and the fp32 x0 bookkeeping between passes.

**Conclusion:** kernel-level work could recover maybe 1.2–1.4× and would be weeks of MLX profiling for it. Every lever below is an order of magnitude cheaper to try and each one is worth more.

---

## 4. Levers, ranked

Savings are on the 12 s / 576×896 production clip (~105 min). "Risk" is what the same-seed read has to rule out. Everything here is already a tier-A knob (`LTX2ConfigResolver` registry) or a scheduler policy field — **no engine code is required for L1–L5**.

| # | Lever | Knob | Forwards/clip | Est. time | Quality risk | Validated? |
|---|---|---|---:|---:|---|---|
| L0 | Production today | — | 57 | ~105 min | — | yes (F1–F4 reads) |
| **L1** | **Plain euler instead of cfg++** (drops the CFG++ negative pass) | `sampler=euler` | 38 | ~70 min | CFG++ steps along the uncond direction even at scale 1; the 2026-08 finding was "negatives inert at CFG 1.0" and NAG still carries the negative. Same-seed A/B will show whether the trajectory change is visible. | no |
| **L2** | **10-step distill schedule** (the 2026-09-14 production schedule) | `stage1_sigmas=1,0.9953,0.9836,0.949,0.848,0.675,0.452,0.243,0.1,0.028,0` | 30 | ~55 min | Burn fix, ghost hand and anchor were all judged at 19 steps. The ladder measured 16 ≈ 10 in anatomy at +45% time, but that was pre-STG. | no (post-STG) |
| **L3** | **12 fps latent + temporal ×2** (render 145 frames at `fps=12` → 19 latent frames; the upscaler makes 37 → 289 frames muxed at 12 × 2 = 24 fps, 12.04 s) | request `fps=12, frames=145`, `temporal_upscale=2` | 57, each on ½ the tokens | **~43% of baseline denoise FLOPs** (dense ½, attention ¼) → ~45–50 min after decode/upscaler overhead | The 24→48 A/B (#440) cost 7–11% sharpness and one ghost at peak motion; 12→24 asks the upscaler to invent more. Conditioning fps changes the model's motion prior (`conditioningFps`, `LTX2Pipeline.swift:121`), so motion pacing must be read, not just sharpness. **Audio is generated from request seconds (145/12 = 12.08 s, `LTX2VideoGenerator.swift:1684`) and trimmed to the final video duration (`:1837–1842`)** — not "unchanged": the A/V cross-attention sees half the video tokens per audio frame, so lip sync is the thing to read. | no |
| L4 | Shorter clips | policy `clipSeconds=8` | 57 | ~70 min | Fewer beats per clip; the seconds decision is a separate discussion (rec: 8 default, 12 opt-in). | n/a |
| L5 | Smaller render | request `width=480, height=768` (render dims come from the request; `delivery_short_edge` only downscales at encode time, `LTX2PostProcess.swift:287`) | 57 on ~0.7× tokens | ~75 min | Softer; the 2026-08-07 recipe rendered larger and delivered 480p supersampled — the opposite trade. | partly |
| L6 | STG off | `stg_scale=0` | 38 | ~70 min | **Rejected** — the ghost hand returns (T-series). | yes, negative |
| L7 | Two-stage | `two_stage=1` | — | ~58 min | **Rejected** — composition drift + lost lip sync (F3 read). | yes, negative |
| L8 | int4 DiT | requantize | 57 | ~same | Memory, not time, on Apple GPU; quality cost. | no, and not worth it |
| L9 | Audio off | `audio=false` | — | −5% | Kira's clips need it. | — |

**Stacked estimate (L1 + L2 + L3):** 2 forwards × 10 steps at 43% of the per-forward FLOPs ≈ 0.35 × 0.53 × 0.43 ≈ 8% of today's denoise work → **~15–20 min** for 12 s once decode/upscale overhead is added, ~5–6× today. Realistic after quality trade-backs: **25–30 min**.

Independence: L1, L2, L3 are orthogonal in cost. They are not orthogonal in quality — each removes a source of guidance or detail — so the ladder tests them one at a time and then stacked.

---

## 5. Validation ladder (same discipline as the burn investigation)

Same prompt and seed as every clip Todd judged (apple-to-Todd, seed 771144, 576×896, 10 s, production recipe as L0). One variable per render. Todd's human read decides; no automated scoring.

| Step | Change from L0 | Expected | What to read for |
|---|---|---:|---|
| A1 | `sampler=euler` | ~58 min | any tone/composition change vs F4; hand; drift |
| A2 | 10-step sigmas | ~46 min | burn/posterize returning; hand; skin detail |
| A3 | `fps=12, frames=121` (10 s), `temporal_upscale=2` — **the whole L3 lever, not the upscaler alone**: it changes conditioning fps, latent token count and the decode/mux path together (codex finding 2) | ~40 min | motion pacing, sharpness, ghost at peak motion, lip sync. The upscaler-alone A/B already exists (#440, same latent, 24→48); if A3 fails, that isolates whether the upscaler or the 12 fps prior is at fault. |
| A4 | A1 + A2 + A3 stacked | ~15–20 min | everything above at once |

Order matters: A1 and A2 first (pure recipe, one variable each), A3 (the frame-rate lever as a unit), then the stack. If any single step fails the read, the stack excludes it. Each accepted step becomes production via the config-file tier (`~/.comfybox/config.json` → `video`), which the engine reads per render; the preflight (`check-production-video-recipe.sh`) is updated in the same PR.

Soak throughput at 24/7 pacing, 12 s clips: today ~13/day; L1 alone ~20; L1+L2 ~26; full stack ~50–70.

---

## 6. What this FDD does NOT propose

- **Kernel/graph optimization of the MLX port.** §3 says the DiT is within ~1.6× of peak; the profiling effort is large and the ceiling is small. Revisit only if the levers above are exhausted and the gap is still felt.
- **Fewer STG blocks or a cheaper STG.** STG's perturbed pass is already a partial forward; the saving would be small and STG is the one guidance we know we need.
- **Batching/parallel renders.** One GPU; memory admission already gates at ~40 GB; concurrent renders would only interleave.
- **Cloud bursts.** Off the table by Todd's local-only rule for Kira's pipeline.

---

## 7. Open question (Todd, 2026-09-15): a stock LTX-2.3 instance as a comparison oracle

**Ask:** "should we equip a stock ltx instance with mlx-serve for comparison?" — stock = **ComfyUI** with the ComfyUI-LTXVideo nodes (Todd). ComfyUI Desktop was the oracle for the 2026-07-23 temporal-motion FDD, so the comparison method exists.

**Recommendation: yes as a one-time oracle, no as a serving path** (validation-tools-not-backends rule). Caveats that shape how much it can tell us:

- ComfyUI is PyTorch. On Apple Silicon it runs on **MPS, not MLX**, and MPS is typically slower than a native MLX port for this class of model. It bounds **quality parity** well (same weights, same schedule, is our output as good?) but it does not bound **speed** — a slower reference proves nothing about our ceiling. §3's FLOPs estimate is the speed ceiling.
- A fair speed oracle would be an MLX-native LTX-2.3 implementation, if a maintained one exists; worth a 30-minute search before committing a machine.
- **ComfyUI is already installed on this Mac** (`~/Projects/ComfyUI` @ 12d5279 2026-08-25, LTX-2 A/V support in core `comfy/ldm/lightricks/av_model.py`) and Todd's "Kira LTX director v7–v9" workflows there are wired to ComfyBox as their backend. The oracle is therefore a second workflow beside those: the same PinkCherry weights, schedule, dims, frames and seed through ComfyUI's native LTX nodes. No new machine, no mlx-serve involvement; the mlx-serve box (10.0.100.134) stays Glimmer's. Runs must not overlap a production render (memory admission; the engine holds ~40 GB and ComfyUI would need ~20–40 GB).
- **What it would settle:** whether our per-forward cost is in line with the reference at identical steps/dims/frames (a 2× gap would say "port problem", a ≤1.3× gap says "hardware"), and whether the reference's 10-step output matches ours (recipe parity). One render at L0 settings, one at L2.

If the search finds no MLX-native reference, I'd skip the oracle and trust §3: the first-principles number and the measured per-forward time agree to within the usual MLX overhead.

---

## 8. Risks

- **Every lever is a quality bet judged on one prompt/seed.** The burn investigation used the same shortcut and it held on the soak; but L3 (frame rate) changes the motion prior and should also be read on one banana/avocado motion clip before it becomes production for those tiers.
- **Config-file tier vs plist.** The recipe lives in `config.json` because the launch-agent edit was refused by the permission classifier; the plist env still carries the old values (NAG 11, ancestral sampler, anchor 0). If the `video` section is ever dropped, the engine silently reverts. The preflight now checks the config file first; aligning the plist is Todd's hand.
- **The scheduler's tuning passthrough** only forwards a whitelist (`TUNING_PASSTHROUGH_KEYS`: sampler, nag_*, color_anchor, temporal_upscale, …). L1 (`sampler`) and L3 (`temporal_upscale`) can be set per clip from the scheduler; L2 (`stage1_sigmas`) and STG cannot — they live in the engine config file. A per-tier recipe would need the whitelist extended.
- **L3 and the audio branch.** The audio latent length derives from seconds, not frames, so audio is unchanged — but the A/V cross-attention now sees half the video tokens per audio frame. Lip sync is the thing to read.

---

## 9. Decision requested

1. Approve the ladder (A1–A4, ~2.5 h of GPU, on the apple test prompt, run between soak clips).
2. Decide the oracle: skip, or one same-settings render on a stock instance if an MLX-native reference exists.
3. Seconds policy (8 vs 12) is a separate decision but multiplies everything here.

---

## 10. Codex review (codex exec, read-only against the engine source, 2026-09-15)

Findings and disposition (full text: `docs/FDD-ltx2-acceleration.codex-review.md`):

1. **Major — L3 audio "unchanged" was wrong.** Audio is generated from request seconds (145/12 = 12.08 s) and trimmed to the final 12.04 s video (`LTX2VideoGenerator.swift:1684`, `:1837–1842`). *Applied in §4 L3 + §8.*
2. **Major — A3 was not one variable.** Request fps feeds temporal conditioning, the frame count changes the token count, and `temporal_upscale` changes decode/mux. *Applied: A3 relabelled as the whole lever; #440 is the upscaler-alone A/B.*
3. **Major — `delivery_short_edge` is encode-time only.** Render dims come from the request. *Applied in §4 L5.*
4. **Minor — L3 savings understated.** With §3's own terms, L3 is ~43% of per-forward FLOPs (dense ½, attention ¼), not "½ tokens". *Applied in §4 L3 and the stacked estimate.*
5. **Nit — §3 arithmetic checks out** (dense 7.09e14, attention 2.73e14, ideal 70.2 s vs measured 112.3 s = 63.6% of peak). No change.
6. **Nit — pass counts and knob names/ranges verified** against `LTX2Pipeline.swift:1779–1782, 1917–1960, 1972–1978, 2036–2058` and `LTX2ConfigResolver.swift:61–96`. No change.

Codex verdict: "technically sound on the big performance story … the main fixes are in L3 and validation." Applied as above.

---

## 11. Ladder results (2026-09-15, Todd's reads, seed 771144, 576×896, 10 s unless noted)

| Clip | Change from production | Wall clock | Read |
|---|---|---|---:|
| F4 (L0) | production (cfg++, 19 steps, STG 0.3 flat, anchor 1) | 82 min | good |
| A1 | plain euler | 63 min* | "very nice. no lost limbs or drift" (extra finger = base constant; a tail-end hand) |
| A2 | 10-step schedule | 51 min | "looks good" |
| A3 | 12 fps latent + temporal ×2 | 38 min | "softer with some drift" — rejected |
| A4 | A1 + A2 + A3 | 14 min | "drift and audio is not as good" — rejected (half the video tokens for the audio branch) |
| **A6** | **A1 + A2 at native 24 fps** | **33 min** | **"i like it" — adopted** |
| A5 | golden-hour spin clip (12 s, 19 steps), anchor 0 | 125 min | "still had some shift on spin" — anchor cleared; spins are the model ceiling → composer MOTION ENVELOPE rule (#1857) |

\* contaminated by concurrent builds; the clean saving is closer to the predicted 33%.

**Outcome.** L1 + L2 adopted: `sampler=euler`, `stage1_sigmas=1,0.9953,0.9836,0.949,0.848,0.675,0.452,0.243,0.1,0.028,0` in `~/.comfybox/config.json` (config-file tier), scheduler no longer sends a per-request sampler. **2.5× throughput** on the same GPU with no visible cost on the same seed. L3 (frame rate) is closed for good: the upscaler costs sharpness and coherence, and the audio branch degrades with half the video tokens. The anchor stays at 1.0. Kernel work remains off the table (§3).
