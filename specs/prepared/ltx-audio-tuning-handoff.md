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

## ROOT CAUSE — CONFIRMED by Codex 2026-08-05 (read this first)

**The engine deletes the audio clause from every Kira prompt.**

- `LTX2VideoGenerator` hardcodes `maxLength: 128` for both positive and
  negative encoding ([LTX2VideoGenerator.swift:552, :663]); the tokenizer
  truncates with `tokens.prefix(128)` — it keeps the HEAD and drops the
  TAIL ([LTX2GemmaTokenizer.swift:152]).
- Kira's prompts are character-injected and long, and the audio clause is
  appended LAST. Codex measured her actual t2v scene pool:
  **audio clauses surviving the 128-token cap: 0 of 16.**
- So the model never sees the requested sound or dialogue. It invents
  non-verbal vocalization ⇒ "not English"; the vocoder renders that badly
  ⇒ "distortion". The clean reference clip was SHORT, so its quoted line
  survived. One mechanism explains both symptoms and the difference
  between the two clip families.

**Fix direction (validate before committing):**
1. Make the audio clause truncation-proof: reorder it ahead of the
   character description, and/or raise `maxLength` (the connector tiles
   128 registers across any divisible length — [LTX2Connector1D.swift:442]
   — so >128 is architecturally allowed but MUST be validated against the
   trained recipe before shipping).
2. Add an invariant: when `audio:true`, reject or reorder any request
   whose audio clause would fall beyond the cap.
3. Log structured truncation facts per render: pre-truncation token
   count, audio-marker token index, whether a quoted line survived,
   effective-prompt hash. (Codex #8: the current log line cannot prove
   dialogue reached the engine.)

## Corrections to my original analysis (all from the same review)

- **My P0 discriminator was NOT discriminating** — it changed modality,
  audio semantics, prompt length, negative conditioning, duration policy
  and dims at once. Replaced by the token probe below.
- **i2v is not a valid proxy** for the known-good t2v path; settle t2v
  first.
- **"Empty LoRA arrays" does NOT make her presets inert** — they still
  carry negative prompts, tuning, and dims policy. My "LoRAs ruled out"
  claim was too broad: LoRA *weights* are ruled out, preset *effects*
  are not.
- **The daemon's logged request shape is not the engine's** — verify at
  the engine, never from daemon logs.
- **"Duration plumbing bug" was wrong**: the 10-12s renders are an
  INTENTIONAL server-side policy (the ≤289f single-pass fold in
  `prepareLocalVideo`), not lost plumbing. Decide whether that policy
  should still apply to short audio clips — it is a policy question, not
  a bug hunt. (Ticket 1491 updated accordingly.)

## Work plan (revised — Codex ordering)

**P0 · Token probe. NO GPU, minutes.** Take her real t2v scene strings
(and a real i2v enriched intent), tokenize WITHOUT truncation, and report:
total tokens, index of the `audio:` marker, whether any quoted line
survives tokens 0-127, and the decoded first 128 tokens. This proves or
kills the root cause on the desk, not on the GPU.

**P1 · Fix + prove.** Reorder/raise per the fix direction, add the
`audio:true` invariant + structured truncation logging, then ONE matched
4s t2v pair: identical prompt with the audio clause INSIDE vs BEYOND the
128-token boundary. A-good/B-bad isolates truncation.

**P2 · Only after t2v is clean:** re-test i2v, then swap in the Kira
preset negative as a single variable (Codex flagged the preset negative
as the next most likely contributor once truncation is fixed).

**P3 · Then the known regressions:** NAG dropped on the audio path
(ticket 1497), Codex Mediums 1-3 from the earlier review, sigma-density
A/B (tarn1's shorter schedule).

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

## Work plan — spec'd, in order

**P0 · Discriminator (~10 min, do FIRST).** Two 4s i2v renders, same
seed, her exact request shape (512x832, her cfg, character-length
prompt, audio:true), NEUTRAL subject:
  A) prompt WITH one quoted English line in the audio clause
  B) prompt with the current fallback ("natural ambient sounds")
Judge by ear + `voice/floor` and zero-crossing-rate stats (script
pattern in this session's history; good reference = 5.9x / ~2500 zcr,
bad Kira clip = 1.9x / ~1073 zcr). Outcome decides everything below:
A-clean/B-bad ⇒ prompt-side, go to P1. Both bad ⇒ request-path bug,
go to P3 first (bisect her request field-by-field against the known-
good reference request until the fault appears).

**P1 · Prompt-side fix (if P0 says prompt).** Two parts:
  a) t2v: authored `audio` fields per scene in `content-scheduler.ts`
     (apple/banana already authored as examples; avocado fields are
     TODD'S to write — ask him, don't invent explicit copy).
  b) i2v: verify the VLM voice line actually reaches the prompt —
     grep daemon logs for `enriched intent`; if the VOICE line is
     missing/`none` on person-visible frames, fix the vision question
     or parse (`src/kira/vision/i2v-voice-enrichment.ts`).
  Acceptance: a Kira-shaped neutral clip with a quoted line renders
  intelligible English; Todd confirms on a real cycle clip.

**P2 · Duration plumbing (do regardless — it poisons every sample).**
Trace `duration` from `content-scheduler.ts` → `video-tools.ts` →
`/v1/video/generate`; suspect the #1440 default-duration logic.
Acceptance: daemon logs "(4s)" and the engine logs "queued (WxH, 97f)"
for the same job; add a unit test asserting the submitted `duration`
survives to the request body.

**P3 · Request-path bisect (only if P0 says both arms bad).** Start
from the known-good reference request (t2v, 480x832, no character
injection, quoted line, no preset) and add ONE Kira attribute per
render until sound breaks: preset id → character injection → cfg 3.5 →
i2v seed path → her dims. The first render that breaks names the cause.

**P4 · Known regressions to fold once sound is good.**
  - NAG dropped on the audio path (lead 3 above) — restore or document.
  - Codex Mediums 1-3 (ancestral noise, unseeded sigmaDown, refineAVState
    context loss).
  - Sigma-density A/B (lead 4).

## External references (community + upstream)

- **PinkCherry checkpoint discussions** (the checkpoint Kira renders on —
  SexGod1979/PinkCherry_NSFW_LTX23). Discussion #8 "feedback on version
  1.7" is the load-bearing one: user tarn1 reports (a) a SHORTER first-
  pass sigma schedule — `1.000, 0.955, 0.893, 0.812, 0.715, 0.603,
  0.482, 0.241, 0.121, 0.0` — gave "better modulated sound... less
  impactful/tinny than the longer sigma"; (b) "pass 2 was wiping out the
  sound version from pass 1... made it much more sedate, wiped out much
  of the background noise" (independent corroboration of our
  audio-bypasses-refine finding); (c) their fix routes pass-1 audio
  directly into the combine node, bypassing the second sampler.
  https://huggingface.co/SexGod1979/PinkCherry_NSFW_LTX23/discussions/8
  Check for NEWER discussions — this community reports audio issues
  first and in production terms.
- **Author workflows on disk** (ground truth for the recipe, incl. the
  audio negative terms we adopted):
  /Volumes/Bolt/ComfyUI-validate/author-samples/*.workflow.json —
  `back_in_black.workflow.json`, `taking_it_from_the_side.workflow.json`.
  Their standard negative ends "...distorted sound, saturated sound,
  loud". They run CFG 1.0 + NAG, never high cfg on audio.
- **Reference implementation** (parity oracle, VALIDATION ONLY — never a
  serving path): /Volumes/Bolt/ComfyUI-validate/ComfyUI/comfy/ldm/
  lightricks/{av_model.py, model.py, symmetric_patchifier.py}. The
  golden-tensor exporters we used live in scripts/export_*_goldens.py;
  that bisect method (chain -> stage -> op) found two single-op bugs and
  is the escalation path if a lever sweep fails.
- **IC-LoRA "Ingredients" character consistency** (task #27, t2v identity
  without training): https://aistudynow.com/how-to-keep-characters-
  consistent-in-ltx-2-3-ic-lora-ingredients/ — stack distilled 0.6 +
  Ingredients 1.4 + VBVR 0.7, CFG 1 + NAG, dual-section prompt
  ("reference:" sheet description + "Generated video:" action).
- **Skin/hair quality LoRA** (task #27, downloaded, unevaluated):
  https://huggingface.co/TheBurgstall/LTX-2.3-skin-hair — local copy at
  /Volumes/Bolt/Models/loras/ltx-quality/skin-LTXLora-step00004000.comfy.safetensors
  (strength 0.75-0.9; note the author's repo also ships LTX-2 AUDIO demo
  clips — same stack, worth listening to as a quality bar).
- **LTX Director / timeline audio** (task #24): ~/Projects/WhatDreamsCost-
  ComfyUI — `ltx_director.py` (PromptRelay temporal attention masking),
  `speech_length_calculator.py` (text -> frames at fps — the sizing math
  for spoken lines), `example_workflows/`. Relevant here because
  prose cannot place a sound in time (the 12s reference's rain never
  arrived); segment-scoped audio is the structural fix.

## Guidance for the next session

- **Ask Todd for the avocado `audio` clauses** — the t2v scene list has
  the field wired with apple/banana authored as examples. Do not invent
  explicit copy; it is his to write (persona-content division).
- **Listen to the author's own audio demos** (skin-hair repo, PinkCherry
  discussion attachments) before deciding what "good" sounds like on
  this checkpoint — they set the achievable bar for this model.
- **Community first, code second.** Both audio insights that mattered
  (refine wiping ambience; the shorter sigma schedule) came from the
  checkpoint's discussion page, not from our code. Check it before
  starting a deep bisect.
- **The engine is not obviously broken.** A neutral t2v request produces
  perfect audio on the same binary. Bias the search toward what the
  DAEMON sends (prompt assembly, truncation, duration, enhance path)
  before suspecting the pipeline.

## Reference material

specs/ltx2-audio.md (rev 2 + wire-4 shipped notes), the good clip
(`audio-REFERENCE-v2-cleanlimiter.mp4`), A/B pairs `audio-ab-*.mp4`,
memory: ltx2-audio-wire1-progress, ltx-quality-over-speed,
dont-generalize-validate-at-production-config.
