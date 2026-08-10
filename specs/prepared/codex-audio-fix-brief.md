# Codex brief: fix the LTX audio truncation bug

**Paste this as your task prompt.** You are BUILDING, not reviewing.

---

You confirmed this root cause yourself earlier today. Start by re-reading
your own findings, then execute:

1. `reviews/codex-handoff-review-2026-08-05.md` (your review — the
   measurements and file:line citations)
2. `specs/prepared/ltx-audio-tuning-handoff.md` (the work order: root
   cause, corrections, P0-P3 plan, external references)
3. `.claude/skills/kira-ltx-tuning/SKILL.md` (A/B discipline, deploy
   mechanics, iron rules)

**Problem in one line:** `LTX2VideoGenerator` hardcodes `maxLength: 128`
and the tokenizer truncates with `tokens.prefix(128)` (keeps head, drops
tail). Kira's prompts are character-injected and long with the audio
clause appended LAST, so **0 of 16** of her scene audio clauses reach the
model. Result: no sound/dialogue direction → non-verbal vocalization
("not English") that the vocoder renders badly ("distortion"). Short
neutral prompts are unaffected, which is why
`~/Pictures/ComfyBox/audio-REFERENCE-v2-cleanlimiter.mp4` sounds perfect.

**Execute P0 then P1 from the handoff.**

**Design constraint (important):** PREFER REORDERING the audio clause
ahead of the character description over raising `maxLength`. The
connector tiles 128 registers across any divisible length
([LTX2Connector1D.swift:442]) so >128 is architecturally allowed, but the
trained recipe may assume 128 and a wrong choice degrades every render
silently. If you propose raising the cap, gate it behind a matched-seed
A/B and present the evidence — do not ship it as part of the fix.

**Also required:**
- An invariant: when `audio:true`, reject or reorder any request whose
  audio clause would fall beyond the cap (fail loud, never silently).
- Structured truncation logging per render: pre-truncation token count,
  audio-marker token index, whether a quoted line survived, effective-
  prompt hash. (Your own finding #8: the current log line cannot prove
  dialogue reached the engine.)
- TDD; full `ZImageTests` green before any deploy.

**Verification bar:** one matched 4s t2v pair — identical prompt with the
audio clause INSIDE vs BEYOND the 128-token boundary. Render at 4s, never
12s (12s ≈ 55 min). Deliver both clips to `~/Pictures/ComfyBox/` with
self-describing names for Todd's ears; do not declare success yourself.

**Content boundary:** Kira's current explicit prompts state a minor
subject age. Do NOT evaluate, tune against, or render that content.
Diagnose and verify with NEUTRAL prompts at her exact request shape —
this has been sufficient for every finding so far.

**Deploy mechanics** (engine bootout → fresh-inode copy → resign-request
→ bootstrap → health; daemon gate via `bash -c` over ssh, retry the
flaky suite once) are in the skill file. Wait for `/health
is_rendering=false` before touching the engine; Kira renders 24/7.

**Tracking:** ticket
https://github.com/BarkadaBrew/coffeeshop-server/issues/1490 — comment
your findings and link the fix commit there. Related: 1491 (duration
policy, re-scoped), 1497 (NAG dropped on the audio path).

**Repo:** `~/Projects/zimage.swift` (engine, Swift/MLX) and
`~/Projects/coffeeshop-server` (daemon, TypeScript). Commit in small
steps; do not push to main — branch and use the gate.
