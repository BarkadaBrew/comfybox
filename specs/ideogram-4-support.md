# SPEC: Ideogram 4.0 support in ComfyBox

**Status:** rev 3, 2026-08-03 — LOCAL-ONLY, and the port source is
**mflux's Ideogram 4 MLX implementation** (Todd's find:
github.com/filipstrand/mflux, src/mflux/models/ideogram4 — complete:
transformer, scheduler, Qwen3-VL encoder, JSON-caption validation, fp8
loading, quantize via mflux-save, LoRA loading).
**Codex review:** pending (queued behind the audio review)

## 1. What Ideogram 4.0 is (verified 2026-08-03)

Released 2026-06-03 as Ideogram's **first open-weight** frontier model:

- **Architecture:** fully single-stream DiT, 34 layers, **9.3B params**,
  text+image tokens in one unified sequence
- **Text encoder:** Qwen3-VL-8B-Instruct, hidden states from **13
  intermediate layers** (multi-scale semantic features)
- **Weights:** HF `ideogram-ai/ideogram-4-nf4` (nf4 ~5GB, CUDA-only packing)
  and an fp8 variant ("all hardware", no diffusers integration); official
  inference at github.com/ideogram-oss/ideogram4; diffusers docs hint MPS
- **License: "Ideogram 4 Non-Commercial"** — fine for this household's use;
  a hard flag if ComfyBox output ever feeds anything commercial
- **Hosted API:** `POST https://api.ideogram.ai/v1/ideogram-v4/generate`,
  `Api-Key` header; `text_prompt` (auto magic-prompt) XOR `json_prompt`
  (structured V4 contract, no magic-prompt); 36+ resolutions 512×1536 →
  3328×1248; `V4StyleDescription` (aesthetics/art style/lighting/medium/
  photography/hex palette); speeds TURBO/DEFAULT/QUALITY (FLASH pending);
  per-image `is_image_safe` + optional Hive likeness/logo detection —
  i.e. the API is **safety-filtered: SFW-only in practice**
- Signature strength: best-in-class text rendering / typography / design.

**Why the architecture matters to us:** single-stream DiT + Qwen3-VL
multi-layer text conditioning is the Krea 2 recipe. ComfyBox already has
`Krea2SingleStreamDiT`, a shared Qwen3 text-encoder loader, q8 quantization,
the LoRA machinery, and DyPE. A native port reuses most of that skeleton.

## 2. Two lanes (both local)

### Lane A — local evaluation via mflux (hours, running 2026-08-03)
mflux (already installed, household tool) implements ideogram-4-fp8 in
MLX with a CLI: `mflux-generate-ideogram4 --base-model ideogram4
--prompt-file caption.json`. No ComfyUI shims needed; the CivitAI/ComfyUI
route is demoted to fallback.

- **Prompting:** the checkpoint wants structured JSON captions —
  `high_level_description` + `compositional_deconstruction` with per-element
  `bbox` text placement. This is MADE for the decoupage/Krita use case:
  text goes where the layer needs it. Plain prompts underperform.
- **Magic Prompt replacement:** teach the local optimizer (template store —
  a `image-design-json` template) to emit the caption schema; mflux
  validates captions on input.
- **Quantization:** `mflux-save` can produce our own q8/q6 from fp8.

### Lane B — native Swift-MLX port (the destination, ~1 week once A earns it)
Port source is mflux's PYTHON-MLX implementation — the lowest-risk path
(same array semantics, near-mechanical translation; ComfyBox itself began
as this exact kind of port from mflux's Z-Image work).
The ComfyBox-native way ([[comfybox-no-python]]) and the only path to
uncensored/offline/LoRA-capable use:

- Port the 34-layer single-stream DiT from
  `mflux/src/mflux/models/ideogram4/` line-by-line (NOT from the PyTorch
  repo), modeled structurally on `Krea2SingleStreamDiT`.
- Text encoder: Qwen3-VL-8B via the existing shared Qwen3 loader
  (mlx-community conversion likely exists; else convert).
- Weights: fp8 repo dequantized → bf16 (~18.6GB) → our q8 (~10GB), same
  pipeline as every other model. nf4 repo is CUDA-packed, ignore it.
- VAE: whichever they ship (check the repo — likely Qwen-Image-family, which
  we already have for Krea 2).
- **Gate:** Lane A evaluation. If Ideogram 4's output earns a rotation
  place (esp. typography/design work Z-Image/Krea2 can't do), the port is
  justified — and Lane A's ComfyUI setup then supplies the golden tensors
  for port validation, exactly as it did for Kroma and the LTX recipes.

## 2.5 Driving use case (Todd, 2026-08-03)

**Layered Krita workflows and the Decoupage tab.** Ideogram output feeds
composition work, not standalone renders: poster text, design elements,
labels, and graphic pieces that get layered over/under local renders. This
implies:

- Surfaces: the Krita bridge (ComfyBox is already Krita AI Diffusion's
  backend) and `DecoupageView` are first consumers alongside Generation.
- **Transparency:** check whether generate-v4 supports transparent
  backgrounds (Ideogram's design tooling suggests it may); if yes, expose it
  — cut-out-ready elements are the decoupage dream. If no, note the
  workaround (flat-color background + the existing local matting path).
- The `ideogram-design` preset defaults to QUALITY speed and a design-biased
  V4StyleDescription.

## 2.6 Lane A eval result (2026-08-03 — PASSED, Lane B earned)

First local render (fp8, mflux, M3 Max): art-deco COFFEE SHOP poster,
1024x1536, V4_DEFAULT_20.

- **Typography: flawless.** Both requested text elements rendered exactly —
  display lettering with inline highlights, caption-size serif, bbox
  placement honored, style description followed (sunburst, corner motifs,
  letterpress texture). Capability no local model in the house matches.
- **Failure mode observed:** hallucinated pseudo-text ("K OMF SPOM") in an
  UNSPECIFIED empty region. Standing rule for the `image-design-json`
  optimizer template: describe empty regions explicitly ("plain background,
  no text") — unspecified space gets filled. (Same family as the LTX
  anatomy-grounding rule.)
- **Numbers:** 20 steps in 11:06 (28→47 s/step, degrading with memory
  pressure), peak MLX memory 36.5GB. Cannot coexist with the warm server's
  LTX renders unmanaged — the admission-gate problem, properly solved by the
  Lane B port. Interim: try `mflux-save -q 8` for a smaller working set.

## 3. Non-goals

ANY hosted-API integration (Todd 2026-08-03: explicitly rejected — this is
a local-model household); remix/describe/edit-style features until the port
exists; any commercial-use path (license); the CUDA-packed nf4 repo.

## 4. Acceptance

Lane A (eval): int8 loads and renders on MPS; the typography test ("a
poster that says COFFEE SHOP in art-deco lettering") is legible; a design
element works as a Krita/Decoupage layer piece; render time and peak memory
recorded. Transparency support checked (cut-out-ready elements or the
flat-background + local-matting fallback documented).

Lane B (port): golden tensors from Lane A's ComfyUI graph match at each
seam (text-encoder features, DiT block outputs, VAE decode) — the Krea 2
port discipline; then the same typography/decoupage tests pass natively,
served from the warm server with traces/Gallery/preset integration free.

## 5. Risks

- The MPS int8/fp8 path is community-patched (shim + forked nodes) — first-
  run friction expected; GGUF Q8 is the fallback.
- Qwen3-VL-8B encoder VRAM on top of the DiT — fine on 128GB, but measure
  before assuming it coexists with the warm server.
- Non-commercial license needs a standing note in the model card UI.
- Lane B scope creep: the gate exists so a 2-week port is a decision, not a
  drift.

Sources: [developer.ideogram.ai generate-v4](https://developer.ideogram.ai/api-reference/api-reference/generate-v4) · [HF ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4) · [Ideogram API overview](https://developer.ideogram.ai/ideogram-api/api-overview)

## Isolation requirement (Todd 2026-08-04, confirmed)

Ideogram gets its OWN prompt optimization and LoRA management, separate
from all other lanes — architectural necessity, not preference:
- Prompt path: structured JSON captions (bbox/exact-copy/empty-region),
  dedicated image-design-json template + design-only optimizer path
  (shipped in P0), own exemplar set when the feedback loop lands. No
  prose-lane vocabulary or wrappers may leak in.
- LoRAs: new library family tag; picker filters HARD on family
  compatibility (dual-DiT + Qwen3-VL + Flux2 VAE — Z-Image/Krea2/LTX
  LoRAs incompatible at tensor level); separate preset stacks; separate
  Models-tab grouping.
