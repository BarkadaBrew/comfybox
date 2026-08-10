# PRD: Fine-Tuned Prompt Optimizer

**Status:** future — not scheduled · **Prerequisite:** `motion-tab-prompt-lab.md` shipped and collecting data
**Owner:** TBD · **Drafted:** 2026-08-02

## 1. Problem

ComfyBox rewrites every render prompt through a general instruction model
(currently `dans-pe-v1.3.0-24b-heresy@8bit` via LM Studio) steered by hand-written
system prompts. Those system prompts encode real, expensively-won knowledge —
limb-placement grounding to prevent extra limbs, explicit figure counts, LTX-2.3
motion-sequence phrasing, i2v motion-only discipline. But the approach has ceilings:

- **The rules only work if the model follows them.** A general model drifts,
  ignores clauses, and — observed in the reference author's own workflow — sometimes
  refuses outright and emits refusal text that gets rendered as the prompt.
- **Knowledge lives in prose, not weights.** Each new finding means another
  paragraph in an already-long system prompt, competing for attention with the rest.
- **No feedback.** The optimizer never learns from which of its rewrites produced
  good renders and which the human had to correct.

A model fine-tuned on this pipeline's own accepted prompts would encode the house
style directly, follow it more reliably, and shrink the system prompt to a rubric.

## 2. Why this is a separate project

It is not a UI or plumbing change. It requires a base-model decision, sufficient
data volume, a training environment, an evaluation harness, and a deployment story
with rollback. Bundling it with the Motion tab work would let an unbounded research
task block a bounded product improvement.

## 3. Prerequisite: data

Supplied by `motion-tab-prompt-lab.md`, which is the entire reason that design
captures what it does:

- **Supervised pairs** — `{system, intent, final}` from traces that were human-edited
  or strongly rated. The human-edited ones matter most: the delta between what the
  optimizer produced and what the human shipped is a direct correction signal.
- **Preference pairs** — `{prompt, chosen, rejected}` from Gallery A/B judgments.
- **Grounded rewards** — every trace links to the actual render, its resolved
  parameters, and multi-axis ratings, so a preference is anchored to output quality
  rather than to a recollection of it.

**Open question — volume.** Unknown how many pairs are needed for a worthwhile
delta. Kira's throughput suggests thousands of traces within weeks, but *rated*
and *edited* ones will be far fewer. Establish a floor before committing to train
(a plausible starting hypothesis: ~500 edited/rated supervised pairs, ~200
preference pairs; validate against the eval harness rather than assuming).

## 4. Goals / non-goals

**Goals**
1. Higher rule adherence than the prompted general model — measured, not asserted.
2. Zero refusals on authorized adult content (a known, recurring failure).
3. Smaller, faster optimizer; ideally lower latency than the current 24B.
4. Same interface: the engine still calls a local OpenAI-compatible endpoint.

**Non-goals**
- Replacing the diffusion model or any renderer component.
- Training a general chat model — this is a narrow prompt-rewriting model.
- Removing system prompts entirely; they become a short rubric, not a manual.

## 5. Approach options (to evaluate, not yet decided)

| Option | Cost | Notes |
|---|---|---|
| **LoRA/QLoRA on a small open base** (7–12B) | low–moderate | Most likely fit; trainable locally via MLX. Base must be uncensored for this pipeline. |
| **Full fine-tune of a small model** (~3B) | moderate | Fast inference; may underfit the stylistic nuance. |
| **DPO/ORPO on preference pairs**, on top of an SFT pass | moderate | Uses the A/B data directly; needs the SFT stage first. |
| **Keep prompting, add retrieval** of nearest-neighbour exemplars | very low | Not training at all. **Should be tried first** — may capture most of the benefit; the Prompt Lab already ships static exemplars, and dynamic retrieval is a small step beyond. |

**Recommendation for whoever picks this up:** exhaust the retrieval option before
training anything. It is cheap, reversible, and its result tells you whether the
remaining gap justifies a training project at all.

## 6. Evaluation (required before deployment)

A tuned model that *feels* better is not a result. Needed:

1. **Held-out prompt set** spanning modes (t2v/i2v), content modes, and subjects.
2. **Rule-adherence scoring** — does the rewrite ground limbs, state figure count,
   describe one resolved motion sequence, avoid re-describing the subject for i2v?
   Partly automatable by pattern checks, partly human.
3. **Downstream render A/B** — the real test: same intent, same seed, tuned vs
   prompted optimizer, judged blind in the Gallery. The preference machinery from
   the Prompt Lab is the instrument.
4. **Refusal rate** on the adult set — must be zero.
5. **Latency and memory** versus the incumbent, since it shares a box with a 34GB
   video model and a 22GB image model.

Ship only on a win in (3) with no regression in (4) or (5).

## 7. Deployment

- Serve at the same local OpenAI-compatible endpoint the engine already calls;
  no engine change beyond a configured model name.
- Version the model; record the optimizer model+version **in every prompt trace**
  so output can be attributed after the fact. *(Schema note: add `optimizer_model`
  to the trace record when this project starts — cheap then, a migration later.)*
- Rollback = point the config back at the prompted general model.
- A/B in production by alternating optimizers across renders and comparing ratings.

## 8. Risks

- **Data volume never materializes** — most likely failure. Mitigated by the
  retrieval-first recommendation, which needs far less.
- **Feedback loop collapse** — training on prompts selected by a model trained on
  earlier prompts narrows diversity over generations. Keep raw intents in the
  training mix; refresh from human-authored traces.
- **Overfitting to one aesthetic** — the corpus is one operator's taste on one
  model family. That is largely the *goal*, but it means the tuned optimizer will
  not generalize to a different checkpoint or subject matter.
- **Base-model licensing** for the intended content domain.
- **Opportunity cost** — this competes with rendering-quality work that has, to
  date, produced larger visible gains per hour spent.

---

## Rev 2 — Codex findings resolution (2026-08-03, findings #21–25 in reviews/codex-specs-rereview-2026-08-03.md)

21. **Model provenance now (#21):** traces carry `renderer_model` and
    `optimizer_model` (+ provider/endpoint and immutable revision/digest)
    from day one — baseline data is already model-dependent.
22. **Atomic optimizer profiles (#22):** deployment/rollback unit is a
    versioned profile: model revision, tokenizer, adapter, template/rubric
    id+hash, generation settings, refusal policy. "Change only the model
    name" is retracted.
23. **Reproducible data contract (#23):** training targets `human_final`
    (never the renderer-augmented prompt); a frozen dataset manifest pins
    template hashes; near-duplicate intents/variants/A-B batches are grouped
    into one split BEFORE dedup so they cannot leak across train/held-out.
24. **Ship gate (#24):** evaluation compares incumbent vs static exemplars
    vs dynamic retrieval vs SFT vs SFT+DPO; retrieval must be beaten, not
    skipped. Gate requires explicit rule-adherence improvement, zero
    refusals on a pre-sized test set, and latency/memory bounds.
25. **Causal A/B (#25):** no alternating production traffic. Paired or
    randomized assignment on same intent/config/seed, arm persisted before
    enqueue, blinded Gallery vote, analysis stratified by mode/content mode.
