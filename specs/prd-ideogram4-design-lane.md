# PRD: Ideogram 4 — the ComfyBox design lane

**Date:** 2026-08-03 · **Owner:** Todd · **Tech spec:** specs/ideogram-4-support.md (rev 3)
**Status:** Lane A eval PASSED same day (typography flawless, 11min/20 steps, 36.5GB peak via mflux fp8)

## 1. Problem / opportunity

ComfyBox renders people and scenes excellently (Z-Image, Krea 2) and video
(LTX), but has NO model that can set type. Todd's layered composition work —
Krita documents and the Decoupage tab — needs posters, labels, lettering,
and graphic elements that current models render as alphabet soup. Ideogram 4
is the best open-weight typography model in existence, its JSON caption
schema places text by bounding box (a layout language, not a vibe), and its
architecture is the Krea 2 family — squarely in ComfyBox's wheelhouse.

**Product statement:** ComfyBox gains a *design lane* — type-perfect,
layout-controlled graphic generation feeding layered composition — fully
local, no hosted API (Todd's explicit constraint).

## 2. Users & jobs

One user (Todd), three jobs:
1. **Decoupage pieces** — labels, lettered panels, ornamental elements to
   cut out and layer. Wants: clean edges, ideally transparency, bbox-placed
   text, style control (palette/medium/era).
2. **Krita layer stock** — posters/signage/graphic backgrounds dropped into
   multi-layer documents via the existing Krita bridge.
3. **One-off design renders** — event posters, covers, product-label
   mockups ("hand-sketched product label" is literally an mflux showcase).

## 3. Product phases

### P0 — port preparation (no user-facing surface)
**Doctrine (Todd 2026-08-03): ComfyUI and mflux are VALIDATION TOOLS ONLY —
oracles and eval rigs, never a serving path, never a stopgap.** Ideogram
renders for real use wait for the native port. P0 is therefore prep:
- `image-design-json` template in the PromptTemplateStore (engine-native,
  useful from P1 day one): local LLM emits valid V4 JSON captions from a
  plain brief, including the **empty-region rule** learned in eval
  (unspecified space grows pseudo-text — describe every region).
- `mflux-save -q 8` bake + a small mflux golden-tensor eval — measurements
  and oracle fixtures for the port, not a usage path.

### P1 — native (the Swift-MLX port, ~1 week)
- Port from mflux's Python-MLX (`src/mflux/models/ideogram4/`), structural
  model `Krea2SingleStreamDiT`, JSON captions ride the ordinary prompt
  string (the Fibo precedent). Our q8 quantization.
- Served by the warm server: model pool entry, presets, traces, Gallery,
  MCP — all the #9/#19 machinery applies for free.
- Admission-gate entry sized from measured peak (Lane A: 36.5GB fp8;
  expect materially less at q8).

### P2 — the design workflow
- **Decoupage integration:** render → auto-matte (rmbg path already in
  the house via Fibo-edit-rmbg / local matting) → cut-out layer piece.
  Transparency-native output if the weights support an alpha mode (check;
  else matting is the path).
- **Krita bridge:** design renders appear as Krita layers like any other
  ComfyBox render.
- Desktop: a Design surface (or Generation-tab mode) with brief → JSON
  caption preview (editable) → render; bbox visualizer later if earned.
- Optimizer feedback loop: rated design renders promote to exemplars
  (the #19 machinery, image task_kind).

### P3 — if earned
- Ideogram LoRAs (mflux lists community formats; unvalidated), style
  presets per era/medium, img2img when mflux gains it upstream.

## 4. Success metrics

- P0: `image-design-json` template emits schema-valid captions from a
  one-line brief (validated by mflux's caption validator); q8 bake measured
  (size, quality delta vs fp8 on the eval poster); golden fixtures exported.
- P1: a usable poster/label from a one-line brief in ≤2 tries, served warm
  in the Gallery with trace lineage; render ≤ Lane A time at q8; zero
  regressions to other models' serving — Todd's judgment.
- P2: brief → cut-out layer in Krita/Decoupage without leaving the flow.

## 5. Non-goals

Hosted API (permanent, Todd 2026-08-03). Commercial use (license).
Multi-user. Replacing Fibo (it keeps edit/rmbg duties). Photoreal people —
that's Krea 2/Z-Image's lane; the design lane is graphics and type.

## 6. Risks

- License is non-commercial — standing constraint, surfaced in model UI.
- fp8→q8 quality delta unmeasured (bake and A/B in P0).
- JSON schema drift upstream; pin the mflux commit the template targets.
- 36.5GB fp8 coexistence with the warm server — P0 renders should avoid
  concurrent LTX jobs until q8 lands (or use the admission gate in P1).
- Hallucinated text in unspecified regions — mitigated by the template
  rule; residual risk on complex layouts.
