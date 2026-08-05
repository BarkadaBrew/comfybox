# Handoff: LTX Audio Quality (Kira) — fresh session

**Written:** 2026-08-05 by the session that built the audio stack.
**Why handoff:** two days deep, context saturated, and the last few
diagnoses were wrong. A clean session with the evidence below and no
sunk-cost theories will move faster. Invoke `/kira-ltx-tuning` first.

## The one-line problem

Todd's verdict: **`~/Pictures/ComfyBox/audio-REFERENCE-v2-cleanlimiter.mp4`
audio is PERFECT. Kira's generated clips are "mostly filled with
distortion, and not English."** Same engine, same day, same monolith.
Find and fix the difference.

## Verified facts (do not re-derive)

- Engine audio stack SHIPPED and validated on the reference clip:
  guided audio via negatives (CFG/CFG++ dual-stream negative pass),
  audio guidance capped at cfg 1.0 (`LTX2_AUDIO_CFG`), audio bypasses
  two-stage refine (Todd's ear verdict), in-engine mastering chain
  (hp50 + 7.5kHz dip + loudness + knee limiter), gate-skip native decode.
- **LoRAs are NOT the cause** — checked: her `kira-video-*` presets have
  EMPTY lora arrays; no LTX LoRA merges since Aug 2. Kroma/SEAsian/
  content are KREA2 IMAGE LoRAs (seed frames only, never enter LTX).
- `steps` is a NO-OP on the distilled pipeline (pinned sigma schedule
  governs; engine warns). The density lever is `tuning.stage1_sigmas`.
- Container timing on her clips is exact (video dur == audio dur,
  48kHz) — this is NOT a clock/mux bug.

## Open leads, most promising first

1. **Prompt-side (strongest).** The reference had an explicit quoted
   ENGLISH line ('she says "the next chapter is my favorite"'). Her
   avocado t2v scenes have EMPTY `audio` fields (fallback: "natural
   ambient sounds of the scene"); i2v voice lines come from the VLM and
   may be absent/garbled. Hypothesis: no quoted line → LTX emits
   non-verbal vocalization → reads as "not English", and the vocoder
   renders those poorly → reads as distortion.
   **DISCRIMINATOR (run this first, ~5 min):** her exact request shape
   (i2v, 512x832, her cfg, character-length prompt) with a NEUTRAL
   subject + one English quoted line. Clean ⇒ prompt-side. Distorted ⇒
   request-path bug still unfound.
2. **Duration plumbing BUG (real, unfixed).** Daemon logs "(16:9, 5s)"
   / config `clipSeconds: 4`, engine queues 241–289 frames (10–12s).
   Duration is lost between `content-scheduler.ts` → `video-tools.ts`
   (suspect the #1440 default-duration logic) → `/v1/video/generate`.
   Fix + prove with a log pair: daemon "4s" → engine "97f". This also
   poisons sound: one beat of action stretched over 10s produces the
   measured wash (voice/floor 1.9x vs 5.9x on the good reference).
3. **NAG is silently DROPPED on the audio path.** `callAV` takes no NAG
   params, so when audio is on, the video stream loses NAG too
   (`nag_scale=11` in her effective config). Quality regression for
   video, and a real difference vs pre-audio renders.
4. Sigma-density A/B (`stage1_sigmas`): community (PinkCherry HF #8,
   tarn1) reports a SHORTER first-pass schedule gives "better modulated,
   less tinny" sound. Never tested here.

## Codex review — 6 findings, 3 unfixed Mediums

`reviews/codex-audio-fixes-review-2026-08-05.txt` (verdict: conditional
GO; production config is safe). Unfixed: (1) audio ignores ancestral
noise outside the CFG++ branch; (2) unseeded ancestral CFG++ applies
sigmaDown without compensating noise; (3) `refineAVState` drops
negativeAudioContext + audioNoiseKey (breaks any LTX2_AUDIO_REFINE=1
A/B); (4) T2V early gate regresses `LTX2_REFINE_DECODE_ONLY`.

## Discipline (learned the hard way today)

- **Iterate at 4s, never 12s** (12s ≈ 55 min; 4s ≈ 5 min).
- Matched seed, ONE variable, production config.
- **Content boundary:** Kira's current explicit prompts state a minor
  subject age; do NOT evaluate/tune those clips. Diagnose with neutral
  prompts at her exact request shape — that is sufficient and has
  worked all day.
- Verify overrides landed: `grep "effective-config" ~/.comfybox/serve.err.log | tail -1`.
- Claim only what a file/test/Todd's ears prove (see memory
  `calibrate-claims`).

## Deploy mechanics

Engine: build Release → wait `/health is_rendering=false` → bootout →
fresh-inode copy → `touch ~/.comfybox/resign-request` → bootstrap →
health. Daemon: branch → push → `ssh todd@10.0.100.232` (fish! use
`bash -c`) → `./scripts/local-merge.sh <branch>` (that suite flakes;
retry once) → pull + `node build.mjs` + `systemctl --user restart
bree-daemon kira-daemon`.

## Reference material

specs/ltx2-audio.md (rev 2 + wire-4 shipped notes), the good clip
(`audio-REFERENCE-v2-cleanlimiter.mp4`), A/B pairs `audio-ab-*.mp4`,
memory: ltx2-audio-wire1-progress, ltx-quality-over-speed,
dont-generalize-validate-at-production-config.
