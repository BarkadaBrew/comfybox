# Prompting Kroma and Chroma

Research brief, 2026-09-16 (Todd: "Research the best way to prompt Chroma and Kroma").
Both are lodestones models, but they sit on different text encoders, and that single
fact decides how each one wants to be prompted.

| | Kroma (v0.3 base / turbo / TDM) | Chroma (Chroma1-HD, Chroma v48–50) |
|---|---|---|
| Lineage | Full fine-tune of Krea 2 Raw | Flux.1-Schnell derived, 8.9B, pruned |
| Text encoder | Qwen3-VL (a language model) | T5-XXL |
| Reads | Sentences, grammar, possessives, spatial relations | Sentences **and** period-separated tags |
| Tag vocabularies | None documented (v0.3 card is silent) | Danbooru + e621 tags trained in |
| Quality/aesthetic tags | Do nothing | `aesthetic 0–10` works, `aesthetic 11` bleeds |
| Negative prompt | Live only above guidance 1.0 (bare base) | Live; use ≥70 tokens or none |
| Weighting syntax | ComfyUI `(term:2)`; **not in ComfyBox** — use `projector_scale` | Standard |
| On this box | `kroma-v0.3-base` resident; all Kira tiers | Registry entry only; no weights installed. Zeta-Chroma (Z-Image based) is experimental, undocumented |

## Kroma

Kroma is Krea 2 with lodestones' film-realism fine-tune baked in. Everything true of
Krea 2 prompting is true of Kroma, with one caveat from the v0.3 release notes:
"Kroma is a fine-tune with its own 'vibe', and prompts do not always translate 1:1
from Krea 2. If a subject stops following the prompt, a small amount of a subject
LoRA nudges it back."

**Write prose. The encoder is a 4B vision-language model that parses grammar.** Every
official Krea 2 sample prompt is 30–150 words of continuous description. Keyword
confetti ("masterpiece, 8k, ultra-detailed") is ignored. Spatial relationships and
possessives survive ("her left hand on his shoulder" lands as written).

**Order the paragraph the way the eye reads a photograph.** The community guide and
Krea's own reverse-engineering system prompt agree on the sequence:
camera and framing → light → subject (pose, then appearance, wardrobe, materials) →
environment → expression and gaze → secondary motion → one style clause. One
paragraph is fine; newlines carry no meaning to the encoder but stop a 400-word
run-on from diluting itself.

**Describe effect, not gear.** "Face close to the lens", "wide establishing shot",
"negative space", "dutch tilt", "shallow focus with a creamy round background",
"warm window light raking across skin, rolling off gently into the shadows". A lens
name on its own is a weak token; the visual consequence of the lens is a strong one.
(House rule already in the assistant brief: Autocord 75/3.5 as an effect.)

**Text in the image goes in double quotes**: `a neon sign reading "OPEN LATE"`.

**Say it positively on the distilled lanes.** On TDM (4 steps) or turbo (8–12 steps)
guidance is 1.0, so there is no negative branch: "no makeup" must become "bare face,
unpainted skin, visible pores". Only the bare base at guidance 3.0–4.0 has a negative
that bites, and there the krea-kira negative (retouching, plastic skin, extra fingers,
orange cast, HDR glow) is the one to use.

**Emphasis.** ComfyUI users weight with `(term:1.1–3.0)`, de-emphasize with
`0.1–0.9`, invert with negative weights; above 3.0 it breaks. ComfyBox does not parse
that syntax. The equivalent dial here is `projector_scale` (1.0 neutral, higher
tightens adherence across the whole prompt), plus plain repetition or a second
sentence for the one thing that must land.

**Settings that match the cards.**
- Turbo / TDM: 8–12 steps (TDM: 4), guidance 1.0–1.5, shift (mu) 1.15, euler.
- Base: Krea 2 Raw settings — 52 steps stock, guidance 3.5, res_2s/beta57 at shift 1.15,
  ≤1024 px on the card (our 1024×1536 lane-2 test took 43 min).
- Krea 2 resolution range 1024–2048 px; 1024×1536 portrait is the sweet spot here.

**Subject adherence.** If Kroma drifts off a described subject, the fix is a small
subject/identity LoRA, not more adjectives. Silveroxides' "unlock" extractions exist to
restore prompt following where the fine-tune refuses; on this box the equivalent is
`krea2_filter_bypass_*` and `Krea2_TextFusion_Refusal_Reduction`, and Todd's ruling is
that the Kroma base needs no identity LoRA for Kira.

## Chroma

Chroma is a different animal: T5-XXL, trained on 5M images curated from 20M
"including anime, furry, artistic stuff, and photos". That mix is why tags work and
why the tag separator changes the style.

**Format: natural-language sentences, then period-separated tags at the end.**
Comma-separated tags pull toward anime/cartoon; periods keep it photographic.
Maximum prompt length is 512 T5 tokens; T5 padding 1.

**Tags need a domain.** T5 does not learn out-of-context trigger words, so write
`artist:name`, `character:name`, and still describe the character's visual features
because captioning gaps are common. Format tags from the dataset help:
`photography_(artwork)`, `digital_media_(artwork)`, `oil_painting_(artwork)`,
`sketch_(artwork)`.

**Aesthetic and quality tags.** `aesthetic 0` to `aesthetic 10` are dataset score
tags (rounded-down pixelprose scores); `aesthetic 11` marks curated AI images and can
cause prompt bleed. `masterpiece; best quality` exist but are weak.

**Negative prompt: long or none.** Chroma has a real CFG branch. The guide is
explicit: "Using a short negative prompt (<70 tokens) is not recommended in any case,
unless it is not used." The official HD example negative: `low quality, ugly,
unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette,
flat colors`.

**Photorealism tricks from the guide.**
- Repeat `The photograph is a.` (×10) at the head for photorealism.
- Open with `A casual snapshot of...` for the amateur/phone look.
- `...cosplayer dressed as <character>...` for a realistic take on a fictional
  character.

**Settings.**
- Chroma1-HD card example: 40 steps, guidance 3.0.
- Standard Chroma: 26 steps, shift 1, beta scheduler 0.6/0.6; photoreal CFG 3.5–4.0.
  CFG skimming or APG lets you push CFG higher without burn.
- Flash variants: 8 steps at cfg 1–1.5 with a second-order substep sampler, or
  16 steps with a multistep sampler.

**Where Chroma stands on this box.** The registry knows `chroma-8.9b-bf16` (T5-XXL,
CFG via approximator) but no Chroma weights are installed. Zeta-Chroma, lodestones'
Z-Image-based prototype with the Flux 2 VAE and x-prediction, has no documentation
beyond training visualizations, and lodestones says to use Chroma for production and
treat Zeta-Chroma as research. If it becomes the art model, it will need its own
prompt study; nothing above transfers except "write sentences".

## What this changes for the assistant

The assistant brief already encodes the Kroma rules (prose over tags, effect over
gear, the two lanes, negative inert at 1.0). Two additions are worth making: the
"say it positively on the distilled lanes" rule, and the ordering
camera → light → subject → environment → expression → style. Chroma rules should
not go into the brief until Chroma weights are on the box.

## Sources

- lodestones/Kroma model card (v0.2 text; v0.3 undocumented) — https://huggingface.co/lodestones/Kroma
- Kroma v0.3 release notes — https://comfyui-wiki.com/en/news/2026-08-31-kroma-v0-3
- Krea 2 prompting guide (community, Qwen3-VL specifics, weighting, block order) — https://gist.github.com/cicalooo/6d26f1d4d54f6d05e87e6fde94b703b7
- Krea 2 Turbo "System Prompt" thread (Krea's reverse-engineering prompt) — https://huggingface.co/krea/Krea-2-Turbo/discussions/4
- Chroma prompting guide (levzzz, chromaforge) — https://github.com/maybleMyers/chromaforge/blob/main/levzzz_chroma_guide.md
- lodestones/Chroma1-HD model card — https://huggingface.co/lodestones/Chroma1-HD
- lodestones/Chroma model card (dataset) — https://huggingface.co/lodestones/Chroma
- Chroma1-HD "Prompt guide?" thread (no official guide exists) — https://huggingface.co/lodestones/Chroma1-HD/discussions/21
- Silveroxides Kroma LoRA extracts — https://huggingface.co/silveroxides/Kroma-LoRA-Extract
- Zeta-Chroma — https://huggingface.co/lodestones/Zeta-Chroma
