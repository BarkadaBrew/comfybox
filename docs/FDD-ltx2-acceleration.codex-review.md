# codex exec review of FDD-ltx2-acceleration.md (2026-09-15, read-only, codex-cli 0.154.0)

1. **Major, §4 L3**: Claim: `fps=12, frames=145`, `temporal_upscale=2` gives “same duration, same audio.”
   Code/arithmetic: base latents are `(145 - 1) / 8 + 1 = 19` (`LTX2Pipeline.swift:389`, `:449`); temporal upscale makes `19 -> 37` latent frames (`LTX2Pipeline.swift:2319-2327`), which decodes to 289 frames and muxes at `12 * 2 = 24 fps` (`LTX2VideoGenerator.swift:1794-1797`, `:1966`). Final video duration is `289/24 = 12.0417s`, matching the 289@24 baseline, but audio is initially sized from pre-upscale seconds: `145/12 = 12.0833s` (`LTX2VideoGenerator.swift:1684-1685`; `LTX2Pipeline.swift:487-504`) and then trimmed to final video duration (`LTX2VideoGenerator.swift:1837-1842`).
   Correction: Say final mux duration/timing matches the 24 fps 289-frame baseline, but audio is overgenerated from request seconds and trimmed, not intrinsically unchanged.

2. **Major, §5 A3**: Claim: validation ladder uses “One variable per render” and A3 is “upscaler alone.”
   Code/arithmetic: A3 changes at least three coupled variables: request `fps` feeds temporal conditioning (`LTX2VideoGenerator.swift:1277`; `LTX2Pipeline.swift:121-123`, `:2717`), `framesPerChunk` changes latent token count (`LTX2Pipeline.swift:449`), and `temporal_upscale=2` changes decode/mux behavior (`LTX2Pipeline.swift:2319-2338`; `LTX2VideoGenerator.swift:1794-1797`).
   Correction: Label A3 as the full 12 fps latent lever, not an isolated upscaler test; add a separate same-latent temporal-upscale-only A/B if isolation is required.

3. **Major, §4 L5**: Claim: `delivery_short_edge=480` is the knob for “480p delivery at 480×768 render.”
   Code/arithmetic: `delivery_short_edge` is Tier A and valid `0...4320` (`LTX2ConfigResolver.swift:78`), but it only downscales at encode time (`LTX2PostProcess.swift:287-296`, `LTX2VideoGenerator.swift:1945-1950`). Render dimensions come from request width/height, not this knob.
   Correction: Split this into “render at 480×768 via request dimensions” plus optional `delivery_short_edge=480`; the listed knob alone is not a 480×768 render lever.

4. **Minor, §4 L3 savings**: Claim: “≈ −50%, attention −75%,” estimated `~50 min`.
   Code/arithmetic: Using §3’s own terms, dense work halves and attention quarters: `(7.086e14/2 + 2.735e14/4) / (9.821e14) ≈ 43%` of per-forward FLOPs, before decode/upscaler overhead. The `~50 min` estimate is conservative from a 105 min baseline, but the text “½ tokens” undersells the arithmetic.
   Correction: State expected denoise FLOPs are about 43% of baseline for L3, with wall time rounded to ~45-50 min after overhead.

5. **Nit, §3 arithmetic**: Claim: FLOP and ceiling math.
   Code/arithmetic: Dims are supported: 48 layers, 32 heads, head dim 128, inner dim 4096 (`LTX2Transformer.swift:96-116`, `:135-138`); 576×896×289 gives latent tokens `37×18×28 = 18,648` (`LTX2Pipeline.swift:449`). Recompute: dense `7.09e14`, attention `2.73e14`, total `9.82e14`; ideal at 14 TFLOP/s is `70.2s`, measured `337/3 = 112.3s`, or `~63.6%` of peak.
   Correction: No substantive correction; the section’s rounded numbers are consistent.

6. **Nit, §2/§4 forwards and knobs**: Claim: 3 forwards/step, NAG does not add a forward, and knob names/ranges.
   Code/arithmetic: Positive pass includes NAG only there (`LTX2Pipeline.swift:1917-1960`); CFG++ negative pass runs when `samplerIsCfgPP && negativeEmbeddings != nil` regardless of cfg scale (`LTX2Pipeline.swift:1779-1782`, `:1972-1978`); STG adds a perturbed pass when `stg_scale > 0` (`LTX2Pipeline.swift:2036-2058`). Registry names/ranges match for `sampler`, `stage1_sigmas`, `temporal_upscale 1...2`, `cond_fps 1...120`, `stg_scale 0...20`, `nag_scale 0...50`, `color_anchor 0...1` (`LTX2ConfigResolver.swift:61-96`).
   Correction: No correction needed for these claims.

**Verdict:** The FDD is technically sound on the big performance story: 3 forwards/step, NAG cost, CFG++ negative-pass cost, STG cost, transformer dimensions, and §3 math all line up. The main fixes are in L3 and validation: the fps/frame/upscale lever is real and likely valuable, but it is not an isolated upscaler experiment, and its audio path is “generated from request seconds, then trimmed,” not simply unchanged. The 480p lever also needs wording tightened because `delivery_short_edge` is encode-time delivery, not render resolution.