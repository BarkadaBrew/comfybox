---
name: kira-ltx-tuning
description: Use when tuning Kira's LTX video quality — presets, LoRA stacks, tuning params, motion/i2v recipes, or audio recipes. Encodes the proven A/B discipline, render/deploy mechanics, and the regressions that taught us the rules.
---

# Kira LTX Tuning

Fine-tune what Kira's video renders look and sound like: preset recipes
(LoRA stacks + strengths), Tier A/B tuning params, motion levers, and the
audio recipe. This skill is the distillation of the 2026-07/08 tuning
campaigns (Kroma integration, i2v strength A/B, NAG, stage floors, audio).

## Iron rules (each one paid for in regressions)

1. **Validate at production config.** Test at the real dims/cfg/mode/steps
   Kira actually renders (480p-720p, her presets, her content mode) — two
   same-day regressions came from testing at convenient settings. Small
   probe formats are for plumbing checks only, never quality verdicts.
2. **Same seed, one variable.** Every comparison is a matched-seed pair
   differing in exactly one lever. No seed = no conclusion.
3. **Judge chains visually, not by RMSE.** Metrics screen; eyes (and for
   audio, ears) decide. Todd is the final judge — deliver clips to
   ~/Pictures/ComfyBox with self-describing names (`EXP_<lever>_<value>.mp4`).
4. **Back up before preset edits.** Convention: copy presets.json to
   `presets.json.bak-<slug>` first (see the existing .bak-* trail).
5. **Don't fight the scheduler.** Kira renders 24/7. Either pause her from
   the Kira tab for a tuning block, or ride idle windows (poll /health
   `is_rendering`; use a Monitor, never a sleep-loop). Never deploy the
   engine mid-render.
6. **One greppable truth.** Every render logs `[LTX2] effective-config:`
   with all params + provenance. ALWAYS verify your override actually won
   (request > preset > configFile > env > builtin) before judging results.

7. **Iterate at 4–6s; reference at 12s.** (Todd 2026-08-05.) Audio/recipe
   iteration uses 97–145-frame clips (~5–12 min each) — duration is the
   linear cost axis, and a 4s clip exposes the same artifact classes as a
   12s one. The 12s/289f format (~45–55 min) is reserved for FINAL
   reference renders after the lever has already won at short length.
   Also remember: `steps` is a NO-OP on the distilled pipeline (pinned
   sigma schedule governs; engine warns since 2026-08-05) — the density
   lever is `tuning.stage1_sigmas`.

## The levers

| lever | where | notes |
|---|---|---|
| LoRA stack + strengths | presets.json (kira-video-* / krea-kira presets) | Kira canon: kroma 1.0 + sea 0.6 + content 0.4 (apple drops content). Max 2-3 LoRAs; act LoRAs selected per-position by the daemon |
| Tier A tuning | request `tuning:{}` or preset `videoTuning` | cfg (t2v needs ~3.5; i2v 2.0), stg_scale, sampler, guidance_rescale, img_compression (motion lever), cond_fps |
| i2v strength | request | HIGHER = closer to source (denoise = 1-strength); 0.5 is the validated default |
| motion | imgCompression (higher = more motion), cfg, prompt action verbs | pristine stills freeze i2v — compression preprocess is load-bearing |
| duration/size | Kira config clipSeconds (live-editable, currently 5), resolution gate | duration linear cost, dims quadratic |
| audio | request `audio` (Kira default ON), steps (speech wants ~16 vs foley 8) | audio-mode flips reload the transformer (~1-2 min) — batch same-mode experiments |
| identity | seed image + kroma preset; face-anchor env (partnered only — solo anchoring ghosts) | char LoRA future work |

## Render commands

Manual A/B (bypasses Kira, uses her preset):
```bash
curl -s -X POST http://127.0.0.1:7870/v1/video/generate -H 'Content-Type: application/json' -d '{
  "prompt": "...", "image_path": "<seed>", "preset": "<kira preset id>",
  "width": 480, "height": 832, "frames": 121, "steps": 8, "seed": 4242,
  "fps": 24, "audio": true, "enhance": false, "source": "tuning",
  "tuning": {"<param>": <value>},
  "output_path": "/Users/toddwalderman/Pictures/ComfyBox/EXP_<lever>_<value>.mp4"
}'
```
- Long renders: run curl with `run_in_background`; the route is synchronous.
- Effective config check: `grep "effective-config" ~/.comfybox/serve.err.log | tail -1`
- Audio waveform sanity: `afconvert -f WAVE -d LEI16 <mp4> /tmp/x.wav` + RMS/peak
  (silence check); no-clipping check via ffmpeg astats.

## The loop

1. State the hypothesis and the ONE lever. Pick production-config dims/preset.
2. Render matched-seed pair(s) in an idle window.
3. Verify effective-config lines show the intended values.
4. Deliver clips to Todd with a one-line "what to look for".
5. On his verdict: fold the winner into the preset (backup first), redeploy
   preset (hot: presets.json is read per-render; engine restart NOT needed
   for preset changes — only for code).
6. If it changes Kira's defaults: confirm the change appears in her next
   scheduled render's trace (`~/.comfybox/traces/`, has_audio / config line).
7. Record the outcome in the relevant spec or qa/ notes — recipes without
   provenance get re-litigated.

## Current canon (2026-08-04 — verify against presets.json before trusting)

- Presets: krea-kira identity = kroma 1.0 + sea 0.6 + content 0.4 by fruit
  tier; apple = kroma + sea only. Distill-1.1 LoRA in the video stack.
- t2v cfg 3.5 / i2v cfg 2.0; sampler euler_ancestral_cfg_pp base +
  euler_cfg_pp refine; two-stage ON with refine-volume gate.
- clipSeconds 5, videoAudio ON, videoMode i2v-default, ≤289f single-pass.
- Known open items: t2v oversaturation (#25), speech-step recipe (#26),
  audio enhance chain not yet in-engine (#26).

## Escalation

- Quality bug that survives lever sweeps → suspect the pipeline, not the
  recipe: same-seed vs ComfyUI oracle (validation tool ONLY, never a
  backend), then golden-tensor bisect (see scripts/export_*_goldens.py
  pattern).
- Anything touching engine code: full TDD + Codex review before deploy.
