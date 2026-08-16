# Text-encoder A/B: heretic vs base Gemma-3-12B q8 — 2026-08-15

**Verdict: base wins. Shipped** — plist `--ltx2-gemma` now points at
`~/LocalModels/gemma-3-12b-it-8bit` (mlx-community, copied from HF cache).
Closes the open question from HANDOFF-ltx-quality-2026-08-02 §5.

## Protocol
i2v, pinkcherry-v18-distill06-int8, 480×480, 97f/24fps, seed 4242, production
env (euler_ancestral_cfg_pp, two_stage=0). Same source still + prompt + seed
per pair; only `--ltx2-gemma` differs. Engine restarted between legs
(same binary throughout — pre-#1479, no RNG-change contamination).

- **Round 1** (mild prompt: head-turn/hair/dolly): visually tied (Todd);
  lapvar 72.6 heretic vs 58.3 base.
- **Round 2** (real production prompt: LTXNUDES oral, guidance 2.0):
  lapvar 94.9 heretic vs 102.2 base. Sharpness deltas flip sign across
  rounds → noise, no consistent sharpness winner.

## Round-2 adherence (the decisive read)
| Prompt element | heretic (A2) | base (B2) |
|---|---|---|
| "takes him deeper" | ✗ disengages mid-clip (retraction artifact) | ✓ holds contact, deeper at end |
| "head tilting back" | partial | ✓ stronger |
| erection stability | tapered by end | fuller, stable |

**Notable:** the "quick retraction" artifact Todd flagged in round 1 (both
encoders, undirected prompt) persisted for heretic even with a directed
prompt, while base mostly avoided it. n=1 seed — evidence, not proof, but
the artifact correlates with the heretic encoder, not only with prompt
under-direction.

## Files
`~/Pictures/ComfyBox/ab-encoder-{A-heretic,B-base,A2-heretic,B2-base}.mp4`,
source still `comfybox-kroma-v0.2-avocado-20260815-162151-05d9.png`.

## Rollback
Point plist arg 15 back at `~/LocalModels/gemma-3-12b-heretic-q8`,
`launchctl bootout/bootstrap gui/501/com.barkadabrew.comfybox`. Heretic dir
kept in place. NOTE: docs/methods/ltx-pinkcherry-comfyui-reference.md still
lists heretic-v2 — correct, that documents the AUTHOR'S ComfyUI oracle, not
our serving recipe.
