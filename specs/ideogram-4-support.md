# SPEC: Ideogram 4.0 support in ComfyBox

**Status:** draft, 2026-08-03 · **Requested:** Todd ("spec Ideogram 4 for
ComfyBox support") · **Codex review:** pending (queued behind the audio review)

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

## 2. Two lanes

### Lane A — hosted-API provider (ship first, ~1–2 days)
The evaluation vehicle and immediate capability. Mirrors the existing
Replicate provider pattern (`provider: "local" | "replicate" | "auto"` in
presets already):

1. `IdeogramClient` in `Sources/ZImage/Providers/` — generate-v4 only for
   v1 (remix/describe/edit later if the eval earns them).
2. Key storage: `~/.comfybox/config.json` `providers.ideogram.apiKey`
   (config file, NOT env — per the #9 doctrine), settable from the desktop
   Settings view like other provider endpoints.
3. Routing: `/v1/generate` payload accepts `provider: "ideogram"`; preset
   field `provider: "ideogram"` works today. Server downloads the returned
   image (links expire — download-to-retain is mandatory), writes to the
   normal output dir, records a normal trace (`task_kind: image_render`,
   provenance notes the remote provider + seed + resolution).
4. Prompt handling: default `text_prompt` (their magic-prompt replaces our
   optimizer — do NOT double-optimize; the `enhance` flag stays false on
   this path). `json_prompt` + `V4StyleDescription` exposed as an optional
   preset block for the design/typography use cases.
5. `is_image_safe: false` responses surface as a loud, named error — never a
   silent blank.
6. MCP: `generate_image` gains `provider`; a preset like `ideogram-design`
   makes it one word for the daemon.
7. Trace + Gallery integration come free (it's the same render path).

### Lane B — native MLX port (evaluate first, ~2 weeks if earned)
The ComfyBox-native way ([[comfybox-no-python]]) and the only path to
uncensored/offline/LoRA-capable use:

- Port the 34-layer single-stream DiT to MLX-Swift, modeled on
  `Krea2SingleStreamDiT` (same family; expect differences in AdaLN layout,
  RoPE axes, and the 13-layer text-feature stack — port from
  ideogram-oss/ideogram4 line-by-line like the Krea 2 port).
- Text encoder: Qwen3-VL-8B via the existing shared Qwen3 loader
  (mlx-community conversion likely exists; else convert).
- Weights: fp8 repo dequantized → bf16 (~18.6GB) → our q8 (~10GB), same
  pipeline as every other model. nf4 repo is CUDA-packed, ignore it.
- VAE: whichever they ship (check the repo — likely Qwen-Image-family, which
  we already have for Krea 2).
- **Gate:** run Lane A for a week first. If Ideogram 4's output at DEFAULT/
  QUALITY earns a place in the rotation (esp. typography and design work
  Z-Image/Krea2 can't do), the port is justified; otherwise stop at Lane A.

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

## 3. Non-goals (v1)

Remix/describe/edit endpoints, style-reference images, batch generation,
the FLASH tier, any commercial-use path (license), local nf4.

## 4. Acceptance (Lane A)

1. `provider: "ideogram"` render lands in the Gallery with a normal trace,
   seed and resolution recorded; the file is local (link-expiry proof).
2. Safety-rejected prompt produces a named error, not a blank.
3. No double-optimization: enhance path bypassed, magic-prompt noted in the
   trace as the optimizer.
4. Preset `ideogram-design` renders a typography test ("a poster that says
   COFFEE SHOP in art-deco lettering") legibly — the capability that
   justifies the provider.
4b. A rendered design element imports into Krita via the bridge and into the
   Decoupage tab as a layer piece — the actual workflow, not just the API.
5. Key absent → clean capability-off error at request time, not a crash.

## 5. Risks

- API pricing/rate limits not in the public reference — measure on the real
  key before wiring the daemon to it.
- Link expiry: download must be synchronous with the render call.
- Non-commercial license needs a standing note in the model card UI.
- Lane B scope creep: the gate exists so a 2-week port is a decision, not a
  drift.

Sources: [developer.ideogram.ai generate-v4](https://developer.ideogram.ai/api-reference/api-reference/generate-v4) · [HF ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4) · [Ideogram API overview](https://developer.ideogram.ai/ideogram-api/api-overview)
