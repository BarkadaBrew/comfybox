# SPEC: Kira character LoRA (face/features) for Krea 2

**Task:** #17 · **Status:** draft for Todd's review, 2026-08-03
**Depends on:** delta-key engine (f6d4a7f, shipped) · Kroma stack in `krea-kira` (live)

## 1. Problem

Kira has no identity anchor. Her face today is `prompt + SEAsian_Women 0.6` —
a *family* of similar faces, not one woman. The three-seed baseline
(`~/Pictures/ComfyBox/kroma-eval/kira-newface-seed{1234,5678,9012}.png`,
recipe kroma 1.0 + sea 0.6 + content 0.4) shows three sisters. KNP never
pinned identity either — it was a realism LoRA, now replaced by Kroma.

Every downstream surface inherits this: chat selfies vary face-to-face, i2v
seeds start from inconsistent identity, and continuity across a session is
luck. A character LoRA pins face/features to one chosen reference, composing
UNDER the style stack (Kroma) rather than fighting it.

## 2. Hard requirement (non-negotiable, gates everything)

**The character this LoRA encodes must render as an unambiguously adult
woman — in the training set, in validation, and in production output.**

- Dataset admission: every image is reviewed; anything ambiguous is excluded.
  The 2026-08-03 balcony-portrait arms are the reference standard for
  "unambiguously adult"; the same-day explicit-tier output is the
  counter-example that triggered this requirement.
- The character sheet gets rewritten alongside this work: age-emphasis and
  youth-coded descriptors ("18-year-old", "youthful proportions") are removed
  in favor of describing the ADULT woman in the reference set (e.g. "a
  Filipina woman in her mid-to-late twenties").
- Acceptance includes adversarial checks (§7): explicit-tier prompts with the
  content LoRA stacked must still render an adult-reading subject. If the
  identity LoRA cannot hold that under production conditions, it does not
  ship.

This is also the engineering fix, not just a policy gate: a strong identity
anchor is precisely the mechanism that stops prompts and content LoRAs from
dragging apparent age around.

## 3. Reference identity (Todd decides first — blocks everything else)

Todd picks THE face from candidate renders (kroma-eval arms + fresh
generations). One canonical image becomes the identity reference; the
dataset is built to agree with it. Until this pick exists, nothing else in
this spec can start.

## 4. Dataset

- **Size:** 25–40 images. Below ~20 underfits identity; above ~50 with one
  synthetic source mostly memorizes artifacts.
- **Generation:** render candidates with the CURRENT stack at production
  dims, then curate hard. Same woman (visual agreement with the reference),
  varied everything else:
  - angles: frontal / three-quarter / profile / high / low
  - expression: neutral, smile, laugh, serious
  - lighting: daylight, golden hour, indoor warm, overcast — varied so
    lighting doesn't bake into identity
  - framing: tight headshot through waist-up; a few full-body for proportions
  - wardrobe varied; NO explicit content in the training set — identity
    generalizes to every tier from a SFW set, the reverse bakes in trouble
- **Style diversity (anti-leak):** generate candidates BOTH with and without
  Kroma in the stack. If every training image carries the film look, the
  identity LoRA learns grain as if it were her skin — then double-applies it
  under Kroma at inference.
- **Face consistency curation:** cross-seed renders will drift; keep only
  images a human (Todd) says are the same woman. Optional assist: an
  embedding-distance pass (insightface, exists at ~/Projects/faceswap) to
  flag outliers — assist, never arbiter.
- **Captions:** short natural-prose, one trigger token (`k1ra woman`), then
  ONLY the variable attributes (pose, lighting, wardrobe, framing). Identity
  attributes (face shape, skin tone, hair) stay OUT of captions so they bind
  to the trigger token.

## 5. Training tooling (decision needed — verify before committing)

ComfyBox is Swift/MLX, no Python in the product, and training is explicitly
out of product scope — the trainer is external tooling, used once per
character version.

| option | pros | cons |
|---|---|---|
| **A. ai-toolkit (or diffusion-pipe) on a rented CUDA box (RunPod/Vast)** — recommended | the Krea-2 LoRA ecosystem (SEAsian, Kroma itself) demonstrably comes from CUDA trainers; best-documented path; hours not days | costs a few dollars; weights leave the house (use a throwaway pod, delete after) |
| B. Replicate managed training | zero setup; token already on Bree server | model-coverage for Krea 2 uncertain; less control over ranks/captions |
| C. Local MLX training | data never leaves the Mac | no known Krea-2 trainer for MLX today; building one is a project, not a step |

**Verification step before spending anything:** confirm the chosen trainer
supports Krea-2-Turbo LoRA (the arch is close to Qwen-Image's MMDiT lineage;
kroma-v0.1's key layout — 264 suffix-form pairs across blocks/txtfusion —
tells us what output format to expect). If A fails verification, try the
Kroma author's toolchain (their HF page/model card may name it).

- **Rank:** 32 (identity does not need Kroma's 256; KNP-style checkpoints ran
  fine much lower). Alpha = rank.
- **Steps/LR:** tooling defaults, checkpoint every ~500 steps, pick by
  validation (§7), not by loss.
- **Output check:** keys must load through `loadForKrea2` — the strict
  classification (task #16) will name anything unsupported (e.g. DoRA) at
  load time instead of half-applying.

## 6. Integration

- `krea-kira` / `krea-kira-hq`: add `kira-face @ 0.9` ABOVE Kroma in the list
  (order is documentation; application is additive either way).
- `krea-kira-sfw`: same, without content LoRA (standing rule).
- `krea-film-*` trio: unchanged — stays non-likeness by design.
- Stack budget: 4 LoRAs on q8 weights. July's 3-LoRA mottling incident says
  validate the full stack at production dims before repointing presets
  ([[dont-generalize-validate-at-production-config]]).

## 7. Acceptance (all at production dims/tiers, judged by Todd)

1. **Cross-seed consistency:** 6 fresh seeds, portrait prompt, full stack —
   all six read as the SAME woman (the §1 baseline shows three sisters; this
   is the before/after).
2. **Same-seed A/B:** with/without the identity LoRA — likeness locks to the
   reference without degrading Kroma's film qualities (skin texture, grain,
   tonal depth persist).
3. **Style independence:** identity holds with Kroma removed (plain krea2) —
   proves the LoRA carries face, not style.
4. **Adult-rendering under adversarial conditions:** explicit-tier prompt +
   content LoRA at production scales — subject still unambiguously adult. A
   failure here is a SHIP BLOCKER, not a tuning note.
5. **i2v seed fidelity:** one i2v clip from an identity-locked seed frame —
   face survives motion at strength 0.75.
6. **No stack regressions:** no mottling/artifacts from the 4-LoRA stack at
   q8 (the July failure mode).

## 8. Sequencing

1. Todd rewrites the character sheet + picks the reference face (§2, §3)
2. Candidate generation + curation to 25–40 (§4) — Kira stays paused
3. Trainer verification (§5), then train; checkpoints at 500-step intervals
4. Checkpoint bake-off via §7.1–7.3, pick winner
5. Full acceptance (§7.4–7.6), preset integration (§6), unpause

## 9. Out of scope

Voice/persona (daemon-side), v2v identity transfer, multi-character scenes,
automated retraining cadence. One character, one LoRA, done well.
