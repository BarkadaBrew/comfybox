# FDD: Ideogram 4 design lane — P0 and P1

**Status:** Proposed

**Date:** 2026-08-03

**Product source:** `specs/prd-ideogram4-design-lane.md`

**Technical source:** `specs/ideogram-4-support.md`, rev 3

**Behavioral oracle:** local `mflux` commit `4ac641e7c1016b489d5165c6533371dafa82b2eb`

**Weight source:** `ideogram-ai/ideogram-4-fp8`, pinned snapshot/revision required by the conversion manifest

## 1. Purpose

This document defines P0 caption and bake preparation and the P1 native Swift/MLX implementation of Ideogram 4 in ComfyBox. It incorporates the adversarial review of the PRD and rev 3 technical spec. In particular, it does not assume that Ideogram 4 is one 9.3B model, that mflux's `-q 8` option requantizes the large FP8 components, or that `Krea2SingleStreamDiT` can be adapted by changing a configuration object.

The native implementation must reproduce the pinned mflux implementation stage by stage before q8 quality or serving performance is judged. P0 is allowed to improve the prompt optimizer and gather evidence; it does not ship mflux as a product dependency.

## 2. Verified baseline and scope corrections

The FP8 repository contains four independently loaded components:

| Component | Parameters | Source files | Role |
|---|---:|---:|---|
| Conditional transformer | 9.279B | 8.652 GiB | JSON-caption-conditioned velocity prediction |
| Unconditional transformer | 9.279B | 8.652 GiB | image-only velocity prediction for CFG |
| Qwen3-VL text encoder | 8.145B | 8.176 GiB | 13 tapped hidden-state features |
| Flux2 VAE | 84M | 0.157 GiB | 32-channel latent decode to RGB |
| **Total** | **about 26.8B** | **about 25.64 GiB** | complete inference system |

The 9.3B figure in the source specs describes one transformer, not the complete model. Both transformers are distinct checkpoints and both execute at every denoising step. The text encoder is also material at 8.145B parameters.

The mflux weight definition marks the conditional transformer, unconditional transformer, and text encoder `skip_quantization=True`. Its FP8 linear layer converts the stored FP8 bytes to the execution dtype, multiplies by the per-output-channel `weight_scale`, and then performs the matmul. Consequently, `mflux-save --quantize 8` preserves the three large FP8 components; it does not establish the quality, size, or memory behavior of ComfyBox group-64 q8.

The default P0 and P1 sampler is `V4_DEFAULT_20`. The observed 20-step Lane A render was about 11 minutes and 36.5 GB. `V4_QUALITY_48` is not the default because it is expected to exceed the P0 15-minute target on the measured machine. It remains an explicit opt-in preset.

Supported dimensions in P0/P1 follow the oracle: width and height are independently in 256...2048 and divisible by 16. The hosted API's larger dimension range is not a local-checkpoint contract.

The VAE declares three output channels. Native transparent output is therefore out of scope; any later transparent-background workflow must use a separate matting or segmentation stage.

## 3. Goals and non-goals

### 3.1 Goals

- Convert one-line design briefs into strict, inspectable Ideogram 4 JSON captions without inventing copy.
- Make intended empty areas and prohibited filler explicit in every generated caption.
- Measure what mflux's q8 save actually changes and record disk, resident-memory, speed, and image evidence.
- Port the conditional and unconditional transformers, exact Qwen3-VL taps, prompt input construction, scheduler, CFG, latent packing, and Flux2 VAE decode to Swift/MLX.
- Convert the source FP8 tensors deterministically to BF16 and then to ComfyBox affine group-64 q8 with auditable manifests.
- Integrate model detection, pooling, warm serving, Gallery-visible results, trace lineage, and presets without enabling the P2 Krita or design-editor surface.

### 3.2 Non-goals

- Hosted Ideogram API integration.
- Krita bridge exposure, design-mode UI, bbox editor, inpainting, editing, ControlNet, or transparency/matting.
- LoRA as a P1 acceptance requirement. Weight-key mapping may reserve the seam, but application to both DiTs is deferred until separately golden-tested.
- Reusing Fibo's VAE or claiming that its prompt encoder validates JSON; it currently passes the prompt through as a string.
- Keeping the text encoder, both DiTs, and VAE resident merely to advertise a warm state. Phase-scoped eviction is part of the design.

## 4. P0 — caption template and evidence bake

### 4.1 Prompt optimizer routing

`image-design-json` is a prompt media kind and template ID, not a new `ContentMode`. The request contract is:

```json
{
  "prompt": "poster for a late-night coffee pop-up called MOONBEAN, leave the bottom quarter empty",
  "media_kind": "image-design-json",
  "content_mode": "neutral"
}
```

P0 changes planned by this FDD are:

1. Add `image-design-json` to `PromptTemplateStore.builtins` and explicitly select it from `templateId(contentMode:mediaKind:)` before the ordinary image branch.
2. Give the design path its own user message. It must contain the brief verbatim and must not add Z-Image photographic boilerplate or the `YOUR CONTEXT` / `YOUR PHOTO` fallback wrapper.
3. Do not retrieve generic image exemplars for this media kind. Until exemplars have a distinct `image-design-json` task kind, the design path uses no exemplars.
4. Raise the design response budget from 1,024 to 2,048 tokens.
5. Parse and validate the model response with `Ideogram4Caption` rather than treating `cleanLLMOutput` as validation. One bounded repair attempt may receive the validation messages. If the second response is invalid, return a typed validation failure; never silently send the raw brief to Ideogram as a plain prompt.
6. Preserve the normalized JSON string, template ID, optimizer model, repair count, and validation outcome in trace metadata. The image trace task kind must not be hard-coded as a video task.

### 4.2 Actual `image-design-json` template content

The following is the normative P0 system-template content. The Swift multiline string may differ only in indentation escaping; its instructions and example are acceptance-test input.

```text
You convert a user's image-design brief into one Ideogram 4 structured JSON caption.

Return exactly one JSON object and nothing else. Do not use Markdown, a code fence, comments, or explanatory text. Emit literal Unicode characters rather than \\u escapes.

Use only these top-level keys, in this order when present:
1. "high_level_description" (required string)
2. "style_description" (optional object)
3. "compositional_deconstruction" (required object)

"style_description" must choose exactly one branch and use the exact key order shown:
- Photo: "aesthetics", "lighting", "photo", "medium", then optional "color_palette".
- Art: "aesthetics", "lighting", "medium", "art_style", then optional "color_palette".
Every non-palette field in the chosen branch is a string. A style palette contains at most 16 uppercase "#RRGGBB" strings.

"compositional_deconstruction" must contain exactly these keys in this order:
1. "background": a string describing the complete canvas, including all space not occupied by an element.
2. "elements": an array of element objects.

Every element has type "obj" or "text" and uses one exact key order:
- Text: "type", optional "bbox", "text", "desc", optional "color_palette".
- Object: "type", optional "bbox", "desc", optional "color_palette".

"desc" is required and is a string. "text" is required only for a text element and must reproduce the user's requested visible copy exactly, including spelling, capitalization, punctuation, and line breaks. Never invent names, slogans, dates, prices, labels, logos, or decorative letterforms. If the user did not request visible text, create no text element.

A bbox is optional. When used, it is [y_min, x_min, y_max, x_max] with four integer coordinates from 0 through 1000, inclusive, and min <= max on each axis. Element palettes contain at most 5 uppercase "#RRGGBB" strings.

Empty-region rule: treat every intended empty area as an explicit design element of the background description. State its location, approximate extent, color/material, and that it remains plain and unoccupied. For each empty region, explicitly prohibit text, letters, numerals, symbols, logos, watermarks, labels, marks, objects, and decorative filler. Do not add an object or text bbox that overlaps an intended empty region. Describe the rest of the background too; no part of the canvas may be semantically unspecified.

Do not use unknown keys. Prefer a small number of non-overlapping elements. Use the user's aspect-ratio or placement constraints when supplied. Resolve underspecified visual style conservatively, but never resolve underspecified copy by inventing words.

Example brief: "cream and forest-green MOONBEAN coffee poster, crescent cup above the title, leave the bottom quarter empty"

Example output:
{"high_level_description":"A restrained cream and forest-green promotional poster for MOONBEAN coffee with a crescent-shaped cup above the exact title and a deliberately empty lower quarter.","style_description":{"aesthetics":"Minimal, balanced specialty-coffee identity with crisp edges and generous negative space.","lighting":"Flat graphic illumination with no cast shadows or glow.","medium":"Screen-printed vector-style poster artwork on lightly textured paper.","art_style":"Mid-century geometric commercial graphic design.","color_palette":["#F4EBD8","#174A35"]},"compositional_deconstruction":{"background":"A complete warm cream paper canvas. The upper three quarters support the centered emblem and title with otherwise open cream space. The entire bottom quarter, y=750 through 1000, is a plain uninterrupted cream paper field reserved as intentional empty space: no text, letters, numerals, symbols, logos, watermarks, labels, marks, objects, borders, patterns, shadows, or decorative filler.","elements":[{"type":"obj","bbox":[120,360,430,640],"desc":"A simple forest-green coffee cup whose rim and handle form a clean crescent silhouette.","color_palette":["#174A35"]},{"type":"text","bbox":[470,230,650,770],"text":"MOONBEAN","desc":"The exact uppercase word MOONBEAN in a bold geometric forest-green display face, centered on one line.","color_palette":["#174A35"]}]}}
```

### 4.3 Caption validation contract

`Ideogram4Caption` mirrors the pinned mflux verifier but adds product-policy checks. The distinction is intentional:

| Rule | mflux schema | ComfyBox template policy |
|---|---|---|
| Root is a JSON object; unknown root keys fail | Yes | Yes |
| `compositional_deconstruction` exists | Yes | Yes |
| `high_level_description` exists | No; type-checked only if present | **Required** |
| Root key order | Not checked | Normalizer emits the template order |
| Style has exactly one of `photo` / `art_style` and exact nested order | Yes | Yes |
| Composition key order is `background`, `elements` | Yes | Yes |
| Element type, key order, required `desc`/`text`, bbox, palette limits | Yes | Yes |
| Empty regions are exhaustively described and protected | No | **Required when the brief requests empty/blank/negative space** |
| No invented visible copy | No | **Required** |

JSON is normalized with compact separators while preserving literal Unicode. The strict validator rejects every warning; calling mflux for P0 always supplies `--strict-caption-validation`. Tests include photo and art branches, optional bbox and palette permutations, reversed keys, unknown keys, malformed coordinates, lowercase colors, Unicode, invented text, overlapping empty-region boxes, and absent empty-region prohibitions.

Semantic policy cannot be proved by JSON decoding alone. P0 implements deterministic checks for the requested-empty-space case: the background must contain the region phrase and all prohibited categories, and bboxes must not overlap a normalized region when the brief supplies a measurable band such as “bottom quarter.” “No invented copy” compares every text element's `text` value against explicitly quoted or clearly named copy extracted from the brief; uncertain cases fail for user review rather than guessing.

### 4.4 Local mflux invocation (validation only)

**Doctrine (Todd 2026-08-03): mflux is a VALIDATION TOOL — no alias, no
operator-facing invocation path.** mflux runs are performed only to produce
oracle fixtures and bake evidence for the gates below, invoked directly by
the engineer doing the work:

```
~/Projects/mflux/.venv/bin/mflux-generate-ideogram4 --model ideogram4 \
  --preset V4_DEFAULT_20 --strict-caption-validation \
  --prompt-file caption.json --width 1024 --height 1536 --seed 42 --output out.png
```

Required evidence is the strict-caption smoke render and the stage tensor
dumps — not a shell convenience. Ideogram renders for real use wait for P1.

### 4.5 P0 q8 bake procedure

The bake is an evidence-gathering experiment, not the native q8 conversion:

1. Pin and record the mflux commit, Hugging Face snapshot, macOS/MLX versions, machine RAM, and free disk.
2. Record `du -sh`, per-component safetensors sizes, tensor dtypes, tensor counts, and `weight_scale` counts for the source.
3. Save with a complete command:

   ```sh
   mflux-save --model ideogram4 --quantize 8 --path /path/to/ideogram4-mflux-q8
   ```

4. Repeat the inventory on the saved checkpoint. Assert rather than assume which components changed. The expected result for the pinned code is that both DiTs and the text encoder remain FP8 plus row scales; only eligible VAE modules can change.
5. Render the source and saved paths with the same validated JSON caption, seed, dimensions, and `V4_DEFAULT_20`. Capture output PNGs, wall time, per-step time, peak resident/shared memory, memory after text encoding, memory during both DiT calls, thermal state, and swap.
6. Run at least three briefs: exact typography, explicit bottom-quarter empty space, and the known wavy-coffee-slogan Lane A prompt. A blind reviewer records copy accuracy, empty-region violations, layout, and obvious visual deltas.
7. Write a small machine-readable result manifest next to the external bake artifacts. It contains commands, hashes, inventories, metrics, and image paths. Do not commit multi-gigabyte artifacts.

Expected P0 result:

| Metric | Source FP8 | mflux “q8” save |
|---|---:|---:|
| Large-component storage | about 25.5 GiB | materially unchanged |
| Peak working set at 1024×1536, 20 steps | observed about 36.5 GB | expected about 35–39 GB; measure |
| Runtime | observed about 11 minutes | expected within noise; measure |
| Evidence about native ComfyBox q8 quality | None | **None** — a separate P1 gate |

If the saved model is materially smaller, the inventory must identify the changed tensor families before any claim is made. A metadata field saying “8-bit” is not proof that the large components were quantized.

### 4.6 P0 acceptance criteria

- `image-design-json` is selected by `media_kind` and never falls through to an ordinary image template.
- The normative example and a 20-brief fixture set validate under both Swift and pinned mflux strict validation.
- All requested literal copy is preserved; no fixture gains unrequested visible text.
- Every fixture requesting blank/empty/negative space has an exhaustive background description and no overlapping element bbox.
- Invalid optimizer output gets at most one repair and then a typed failure; raw prose is never silently rendered.
- The shell alias resolves and a strict `V4_DEFAULT_20` render starts with the generated JSON.
- Source-versus-saved component inventories, disk sizes, working-set measurements, timings, commands, hashes, and A/B images are recorded.
- The P0 report explicitly states that the mflux save did or did not transform each large component.

## 5. P1 — native Swift/MLX port

### 5.1 Runtime data flow

```text
validated compact JSON
        |
Qwen chat template + tokenizer (max 2048, fail on overflow)
        |
Qwen3-VL-8B text-only decoder, capture block outputs
0,3,6,9,12,15,18,21,24,27,30,33,35
        |
[B, textLength, 13 x 4096] cached Float32 prompt features
        |
text encoder evicted
        |
packed noise [B, (H/16)(W/16), 128]
   |                                      |
conditional DiT                         unconditional DiT
[zero text tokens; image] + features    image tokens + zero features
   |                                      |
pos_v                                  neg_v
   +-------- v = g*pos_v + (1-g)*neg_v ---+
                     |
            z = z + v*(s-t), 12/20/48 steps
                     |
channel scale/shift + unpack to [B,32,H/8,W/8]
                     |
                Flux2 VAE
                     |
                    RGB
```

Prompt features are cached by normalized JSON, width, height, tokenizer revision, text-encoder revision, and tap list. Evicting the encoder is permitted only after the feature array is evaluated and entered in that cache.

### 5.2 Architecture constants

The native configuration is loaded from the converted manifest and checked against these oracle defaults:

- Transformer: hidden size 4,608; 34 blocks; 18 heads; head dimension 256; FFN intermediate size 12,288; input/output channels 128; AdaLN conditioning width 512; LLM feature width 53,248; RMS epsilon 1e-5 except LLM-conditioning norm 1e-6; RoPE theta 5,000,000; `mrope_section = [24, 20, 20]`.
- Text encoder: Qwen3-VL text decoder with vocab 151,936; hidden size 4,096; 36 blocks; 32 query heads; 8 KV heads; head dimension 128; intermediate size 12,288; max positions 262,144; theta 5,000,000; RMS epsilon 1e-6; prompt cap 2,048 tokens.
- Text taps are **outputs of decoder blocks** 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, and 35. The embedding output is not tap 0 and the final model norm is not applied to the taps.
- Image token grid is `height/16` by `width/16`; each token has 128 channels.
- The VAE is Flux2 `AutoencoderKLFlux2`, 32 latent channels, patch size 2, and 3 RGB output channels.

Configuration loading fails if a component advertises incompatible constants. It must not silently use Krea2 defaults.

### 5.3 Why this is not a Krea2 subclass

Krea2 is a style and primitive reference, not the superclass of the Ideogram transformer:

| Concern | Krea2 | Ideogram 4 requirement |
|---|---|---|
| Blocks / hidden / heads | 28 / 6,144 / 48 Q, 12 KV | 34 / 4,608 / 18 Q=KV |
| Attention projection | Separate Q/K/V, grouped-query attention, sigmoid output gate | Fused QKV, full multi-head attention, no output gate |
| Q/K norm | Krea-specific RMS scale convention | Direct learned RMS weights |
| RoPE | Adjacent-pair rotation, `[32,48,48]`, optional DyPE | Half rotation with exact three-axis selector and fixed theta |
| AdaLN | Six scale/shift/gate vectors at hidden width | Project timestep to 512; four vectors: MSA scale/gate and MLP scale/gate; no shifts; gates use `tanh`, scales add 1 |
| Block normalization | Krea pre-norm layout | Separate attention/FF input and output norms around gated residuals |
| Text features | 12 Qwen taps, tap-axis fusion, width 2,560 each | 13 direct Qwen taps concatenated to 53,248, RMSNorm then one projection |
| Image patching | NCHW latent patchification | Noise is created already packed at 128 channels; unpack only before VAE |
| CFG | One transformer | Two separately weighted transformers per step |

Ideogram code may reuse utility patterns for MLX attention, quantized-linear construction, loading, progress, and memory cleanup. It must have its own transformer, block, RoPE, scheduler, and pipeline types.

### 5.4 Transformer details

#### Input construction

- Tokenize the normalized JSON after the Qwen chat template with a generation prompt. Do not add a second set of special tokens.
- Reject token sequences over 2,048; do not truncate a JSON caption into invalid semantics.
- Left-pad a batch to its longest text sequence, although P1 serving may initially restrict generation to batch size one.
- Text position IDs are `[i,i,i]`. Image positions are `[0,row,column] + 65536` in row-major order.
- Indicator IDs are 3 for text and 2 for image; padding is 0. Segment IDs are 1 for valid text and image tokens and -1 for padding.
- The attention mask is exact segment-ID equality broadcast as `[B,1,L,L]`, matching mflux. This also lets `-1` padding positions see other `-1` padding positions, but valid segment-1 tokens cannot see them, so valid outputs remain isolated. Golden tests lock this behavior rather than silently “fixing” the oracle.
- Conditional input concatenates zero 128-channel text tokens and packed image tokens. Unconditional input contains image tokens only and a zero 53,248-wide LLM-feature tensor.
- Project the image/text token input from 128 to 4,608, add an indicator embedding, RMS-normalize/project masked LLM features from 53,248 to 4,608, and add them to the token stream.

#### AdaLN and block order

The timestep scalar is in `[0,1]` and is multiplied by 10,000 before sinusoidal embedding. The embedding orders sine before cosine, then applies linear → SiLU → linear at width 4,608. `adaln_proj` maps 4,608 to 512 and its result is passed through SiLU for block modulation.

Each block maps the 512-wide conditioning to four 4,608-wide vectors in this order: `scale_msa`, `gate_msa`, `scale_mlp`, `gate_mlp`. Attention input is `attention_norm1(x) * (1 + scale_msa)`. Its normalized output is gated by `tanh(gate_msa)` and added to the residual. The feed-forward branch repeats that shape with `ffn_norm1`, `(1 + scale_mlp)`, `ffn_norm2`, and `tanh(gate_mlp)`. There are no shift vectors.

The final layer applies affine-free LayerNorm, not RMSNorm. It applies a second SiLU to the 512-wide AdaLN input, produces a 4,608-wide scale, multiplies by `1 + scale`, and projects to 128 channels.

#### Three-axis RoPE

For head dimension 256, create 128 inverse frequencies:

```text
inv_freq[i] = 1 / 5_000_000 ^ ((2*i) / 256), i = 0..<128
```

Initialize all frequency slots to use position axis 0. Axis 1 overwrites slots `1,4,...,58`; axis 2 overwrites `2,5,...,59`. Thus the first 60 slots cycle time/row/column and the remaining 68 use axis 0. This is the behavior of the pinned mflux loop; do not reinterpret `[24,20,20]` as three contiguous slices. Duplicate the selected 128 angles to 256 before cosine/sine and rotate the first and second halves, not adjacent pairs. Golden tests lock this seemingly unusual selector to the oracle.

### 5.5 Text-encoder loading strategy

Reuse and generalize `Qwen3TextEncoder`; do not create a second complete Qwen implementation. The existing configuration already represents hidden size, layer count, GQA, head size, intermediate size, and RoPE theta. Add an API whose tap indices explicitly mean decoder-block output indices and which retains only requested layers as it streams forward.

The Ideogram prompt encoder requests the exact 13 taps, stacks them as `[13,B,S,4096]`, transposes to `[B,S,4096,13]`, and reshapes to `[B,S,53248]`. This must be a reshape of that order, not a tap-axis projection borrowed from Krea2. The result is masked, converted to Float32, evaluated, and cached before the text encoder is released.

The loader filters the source `language_model.` prefix, strips it for the shared Swift module, and ignores all vision-tower tensors. It supports the source shard index rather than assuming a single safetensors file. Tokenizer files come from the same pinned snapshot and the cache key includes their hash.

The existing generic hidden-state API includes the embedding in its returned list; using tap values directly against that list would be off by one. Ideogram code may only call the new block-indexed API. Existing Krea2 tap behavior is left unchanged unless its own tests justify a migration.

### 5.6 Latent packing, VAE, and scheduler

#### Packed latent

Create deterministic Float32 normal noise directly as:

```text
[batch, (height/16) * (width/16), 128]
```

There is no Krea2-style forward patchify step. Before decode, apply the pinned 128-channel Ideogram latent scale and shift arrays exactly as mflux, reshape to `[B,H/16,W/16,2,2,32]`, transpose to `[B,32,H/16,2,W/16,2]`, and reshape to `[B,32,H/8,W/8]`. Cast to the VAE execution dtype only after unpacking. Reuse `Flux2VAE`; do not use `Krea2VAE` or `FiboVAE`. Golden tests check constants, channel order, and unpack indices independently.

#### Scheduler and CFG

Implement `Ideogram4LogitNormalSchedule` and `Ideogram4Scheduler` as new types. For `N` steps, form `N+1` linear intervals in `[0,1]`, apply the normal inverse CDF, then:

```text
y       = resolutionMean + std * inverseNormalCDF(interval)
shifted = 1 - sigmoid(y)
t_min   = 1 / (1 + exp(0.5 * 18))
t_max   = 1 / (1 + exp(0.5 * -15))
value   = clamp(shifted, t_min, t_max)
resolutionMean = mu + 0.5 * log((height * width) / (512 * 512))
```

`t_values` use intervals `1...N`; `s_values` use intervals `0...(N-1)`. The denoise loop indexes both arrays in reverse. This reversal also reverses the stored guidance tuple during execution.

| Preset | Stored guidance tuple | `mu` | `std` | Effective execution order |
|---|---|---:|---:|---|
| `V4_QUALITY_48` | 3×3, then 45×7 | 0.0 | 1.5 | 45×7, then 3×3 |
| `V4_DEFAULT_20` | 2×3, then 18×7 | 0.0 | 1.75 | 18×7, then 2×3 |
| `V4_TURBO_12` | 1×3, then 11×7 | 0.5 | 1.75 | 11×7, then 1×3 |

For each step:

```text
v = guidance * positiveVelocity + (1 - guidance) * negativeVelocity
z = z + v * (s - t)
```

A caller choosing a named V4 preset gets its step count, schedule parameters, and guidance tuple as one atomic configuration. The server rejects conflicting scalar `steps` or `guidance` overrides instead of silently discarding them. A future explicit custom mode may use a constant caller-provided guidance and caller-provided step count, matching the mflux Python behavior, but it is not required for P1.

### 5.7 Weight conversion and q8 pipeline

Direct loading of the HF FP8 files is not supported by the ordinary `SafeTensorsReader`, and the existing specialized FP8 path does not implement Ideogram's per-output `weight_scale` convention. Conversion is mandatory.

#### Stage A: FP8 HF to BF16

Add `scripts/convert_ideogram4_fp8.py` with these properties:

1. Accept only an explicit source snapshot and output directory. Require approximately 120 GiB free before retaining FP8, BF16, and q8 stages together.
2. Read all shard indexes for `transformer`, `unconditional_transformer`, and `text_encoder`; copy tokenizer/config/scheduler metadata and VAE weights.
3. For each FP8 `*.weight` with sibling `*.weight_scale`, reproduce mflux:

   ```text
   bf16Weight = fromFP8(rawWeight, BF16) * BF16(weightScale)[..., newAxis]
   ```

   Write the BF16 weight under the original weight key and omit the consumed scale tensor. Preserve already-BF16 embeddings, norms, biases, and non-linear tensors.
4. Filter text-encoder weights to `language_model.*` and either retain that prefix consistently in the intermediate format or record its deterministic removal in the manifest. Never include vision-tower weights.
5. Stream one tensor or bounded shard at a time. The converter may not materialize either 9.3B component in full.
6. Write sharded safetensors indexes atomically. A component is complete only after its index and manifest are fsynced and tensor counts, shapes, dtypes, and hashes reconcile.
7. Emit a manifest containing source repo, immutable snapshot/revision, mflux commit, converter version, input and output SHA-256 values, tensor counts, component parameter counts, dequantization formula, and output schema version.

Expected BF16 storage is about 50 GiB: roughly 17.3 GiB for each DiT, 15.2 GiB for the text encoder, and 0.16 GiB for the VAE. The rev 3 estimate of 18.6 GB covered neither the second DiT nor the text encoder.

The Python converter exports small deterministic conversion fixtures. Swift tests load those fixtures to verify FP8 decoding and scaling without needing the full model.

#### Stage B: BF16 to ComfyBox q8

Add an Ideogram-aware quantization entry point, preferably `Sources/ZImage/Ideogram4/Weights/Ideogram4Quantizer.swift` plus a CLI mode in `Sources/ComfyBox/main.swift`, rather than applying the current Z-Image component loop unchanged.

- Quantize eligible 2-D linear weights in the conditional DiT, unconditional DiT, and Qwen text encoder using ComfyBox affine q8 with group size 64.
- Keep embeddings, RMS/LayerNorm parameters, biases, scalar/vector modulation parameters, and VAE weights in BF16 unless a separately measured policy says otherwise.
- Include both transformer component directory names in enumeration. Omitting `unconditional_transformer` is a hard failure.
- Emit `.scales` and `.biases` according to the existing quantized-linear schema and a per-component count of quantized and skipped layers.
- Fail the command if any expected component has zero eligible quantized layers, if a source key is unconsumed, or if a produced q8 tensor cannot be round-tripped by the Swift loader.
- Preserve the BF16 manifest lineage and add group size, bit width, skipped tensor policy, hashes, and measured reconstruction statistics.

Expected q8 storage is about 27–29 GiB, not 10 GiB. Affine group metadata and the BF16 embedding/norm set make native q8 slightly larger than a raw all-FP8 checkpoint. The reason to use it is native ComfyBox kernel compatibility and predictable loading, not a guaranteed disk reduction relative to source FP8.

Q8 quality is a P1 gate because the P0 mflux save does not requantize the large components. Test the BF16-converted path against the oracle first, then compare q8 against BF16 with stage metrics and blind image review. If q8 text-encoder quality fails, retain the text encoder in BF16 or evaluate q6 for that component; do not weaken acceptance tolerances until images look acceptable.

#### Loader behavior

The loader detects q8 from manifest schema and quantized parameter siblings before applying weights, constructs quantized linear modules with group size 64, and then applies tensors. It validates all expected keys and both DiT identities. No silent fallback from an unrecognized FP8 tensor to BF16 is allowed.

### 5.8 Per-file work breakdown

Paths below are planned names; related small types may be combined where that improves readability without changing responsibility.

| File | Work |
|---|---|
| `Sources/ZImage/Ideogram4/Ideogram4Config.swift` | Codable checked configs, component constants, V4 preset names, dimension limits |
| `Sources/ZImage/Ideogram4/Caption/Ideogram4Caption.swift` | mflux-compatible parse/normalize/schema validation plus separate empty-region and exact-copy policy validation |
| `Sources/ZImage/Telegram/PromptTemplateStore.swift` | Register and select `image-design-json` before generic image routing |
| `Sources/ZImage/Telegram/PromptOptimizer.swift` | Design-specific user message, 2,048-token budget, no generic exemplars/wrapper, strict parse, one repair, typed failure, correct trace metadata |
| `Sources/ZImage/Tokenizer/Tokenizer.swift` | Non-truncating chat-for-generation encode path with explicit 2,048-token overflow error |
| `Sources/ZImage/Flux2/TextEncoder/Qwen3TextEncoder.swift` | Shared streaming block-output tap API; retain only requested states; do not apply final norm to Ideogram taps |
| `Sources/ZImage/Ideogram4/TextEncoder/Ideogram4PromptEncoder.swift` | Chat templating, input/position/indicator/segment tensors, exact 13-tap reshape, cache and eviction boundary |
| `Sources/ZImage/Ideogram4/Transformer/Ideogram4Norm.swift` | Direct-weight RMSNorm and affine-free final LayerNorm semantics |
| `Sources/ZImage/Ideogram4/Transformer/Ideogram4RoPE.swift` | 128-slot three-axis selector, theta 5M, half rotation |
| `Sources/ZImage/Ideogram4/Transformer/Ideogram4Attention.swift` | Fused QKV, 18 full heads, Q/K norm, segment mask, no Krea output gate |
| `Sources/ZImage/Ideogram4/Transformer/Ideogram4Block.swift` | Four-way 512-wide AdaLN modulation, scale/gate ordering, tanh residual gates, Ideogram FFN/norm order |
| `Sources/ZImage/Ideogram4/Transformer/Ideogram4Transformer.swift` | Input/indicator/LLM projections, timestep path, 34 blocks, final layer, debug capture hooks |
| `Sources/ZImage/Ideogram4/Ideogram4Scheduler.swift` | Inverse-normal logit schedule, resolution shift, preset tuples and reverse indexing |
| `Sources/ZImage/Ideogram4/Ideogram4LatentCreator.swift` | Deterministic packed noise, 128-channel scale/shift, exact unpatchify into 32-channel Flux2 latent |
| `Sources/ZImage/Ideogram4/Weights/Ideogram4WeightMapping.swift` | BF16/q8 key mapping for both DiTs, Qwen language model, and Flux2 VAE |
| `Sources/ZImage/Ideogram4/Weights/Ideogram4WeightLoader.swift` | Shard indexes, manifest validation, quantized-module construction, key reconciliation, staged loads |
| `Sources/ZImage/Ideogram4/Weights/Ideogram4Quantizer.swift` | BF16-to-affine-q8 group-64 conversion across all three large components and manifest output |
| `Sources/ZImage/Ideogram4/Ideogram4ModelDetection.swift` | Detect root/subdirectory/index layout, source-vs-converted format, both DiTs, config compatibility |
| `Sources/ZImage/Ideogram4/Ideogram4Initializer.swift` | Ordered phase loads, text-feature cache handoff, component eviction, VAE reuse |
| `Sources/ZImage/Ideogram4/Ideogram4Pipeline.swift` | Request validation, strict captions, two-DiT CFG loop, progress/cancel, decode, result/trace metadata |
| `Sources/ZImage/Flux2/VAE/Flux2VAE.swift` | Reuse as-is if goldens pass; any required public initializer seam gets a focused change and regression tests |
| `Sources/ZImage/Server/ModelPool.swift` | Add `.ideogram4`, detection/load switch cases, staged `PipelineBox`, conservative admission estimate, restore/evict paths |
| `Sources/ZImage/Server/WarmServer.swift` | Pipeline slot, prepare/register/restore, route dispatch, strict request checks, generate response, errors, health and unload behavior |
| `Sources/ZImage/Server/ComfyBridge/ComfyBridgeModelRegistry.swift` | Add Ideogram metadata and model listing; explicitly keep it out of the P1 Krita bridge allow-list |
| `Sources/ZImage/Server/PresetStore.swift` | Add/migrate `ideogram-design`, scheduler name and 1024×1536 default without relying on first-run-only seeding |
| `Sources/ComfyBox/main.swift` | Native Ideogram CLI/diagnostic and conversion/quantization entry points, complete model-family help/dispatch |
| `Sources/ZImage/MCP/MCPToolRegistry.swift` and `Sources/ZImage/MCP/WarmServerClient.swift` | Accept the family/preset through the generic generation tool; expose validation errors and trace ID |
| `scripts/convert_ideogram4_fp8.py` | Streaming FP8-row-scale to BF16 converter and lineage manifest |
| `scripts/export_ideogram4_goldens.py` | Pinned mflux stage exporter described below |
| `Tests/ZImageTests/Ideogram4/*` | Caption, tokenizer, text taps, RoPE, block, latent, scheduler, loading, quantization, pipeline and server tests |

`Sources/ZImage/Krea2/` and `Sources/ZImage/Fibo/` are references and should require no functional modifications for P1. Copying code should be minimized; any reused primitive must keep family-specific semantics visible at its call site.

### 5.9 Model detection, registry, serving, and presets

#### Detection and registry

`Ideogram4ModelDetection` recognizes:

- Source FP8 layout only for conversion diagnostics, not native generation.
- Converted BF16 layout for oracle parity and development.
- ComfyBox q8 layout for product serving.

All layouts require configs plus `transformer`, `unconditional_transformer`, `text_encoder`, `tokenizer`, and `vae`. Detection reports a typed error for a missing branch or incompatible architecture rather than falling through to Flux2/Krea2.

Add `.ideogram4` to every exhaustive `WarmModelFamily` and ModelPool switch: VRAM estimate, detection, load, box creation, registration, restoration, eviction, current-model metadata, LoRA rejection, and response routing. Begin with a conservative 42 GB admission charge and reduce it only after native q8 measurements on the target machine. Report phase-level residency in health diagnostics.

The model registry entry includes family, immutable/default repository ID, q8 requirement, 26.8B total system parameters, 128 packed channels, supported dimensions, default preset, and a noncommercial/license notice sourced from the checkpoint's actual license metadata. Registry or Gallery copy must not invent license terms.

P1 does not add Ideogram to `ComfyBridge`'s Krita execution allow-list. Registry listing and Gallery/warm-server generation are allowed; Krita enablement is a P2 decision.

#### Warm route

The existing `/v1/generate` route remains the generation endpoint. An Ideogram request supplies:

- `model` resolving to family `ideogram4`;
- a schema-valid JSON caption string, preferably returned by `/v1/enhance` with `media_kind=image-design-json`;
- width, height, seed, output path, and `scheduler` equal to one of the three V4 preset names.

The server defaults an absent scheduler to `V4_DEFAULT_20`. Named presets are atomic: conflicting scalar step/guidance fields produce HTTP 400 with a useful message. Invalid JSON, plain text, unsupported dimensions, prompt overflow, source-FP8 weights, and missing components also fail before allocating the denoise working set.

Cancellation is checked between text-encoder layers, between transformer blocks when practical, between conditional and unconditional calls, and between denoise steps. Partial output is never presented as a successful generation.

The result record includes normalized-caption hash, template/optimizer metadata, tokenizer and weight revisions, q8 manifest hash, preset, seed, dimensions, per-phase times, peak memory, output hash/path, and trace ID. This is the lineage required for Gallery display.

#### Preset migration

Add `ideogram-design` with model family Ideogram 4, 1024×1536 dimensions, and scheduler `V4_DEFAULT_20`. The existing store seeds defaults only when the file is absent, so implementation must use a versioned migration/upsert keyed by preset ID. It must not overwrite a user-edited preset with the same ID; a built-in version marker or separate built-in/user overlay is preferred.

Expose `V4_QUALITY_48` and `V4_TURBO_12` as opt-in variants. Do not label `V4_QUALITY_48` as the default until it meets the product time target.

### 5.10 Memory lifecycle and expected working set

The pipeline uses explicit phases:

1. Load/tokenize/encode text; evaluate and cache Float32 53,248-wide prompt features.
2. Evict the text encoder and clear the MLX cache before loading or activating both q8 DiTs.
3. Run the two DiTs sequentially per step. Never retain both velocity outputs longer than the CFG combination requires.
4. Evict DiTs, clear cache, unpack latents, and load/activate the Flux2 VAE.
5. Keep only small configs, tokenizer state, manifests, and bounded prompt features between requests unless the admission gate grants a larger warm residency mode.

Initial planning envelopes, to be replaced with measured values, are:

| Phase | Expected resident/shared working set |
|---|---:|
| q8 weights on disk | about 27–29 GiB |
| Text encode | about 12–18 GB |
| Two-DiT denoise at 1024×1536 | about 26–34 GB |
| VAE decode after DiT eviction | about 3–8 GB |
| Accidental all-component coexistence | about 32–40+ GB; not an accepted lifecycle |

Admission starts at 42 GB because weights, activations, allocator caching, and the 53,248-wide prompt feature tensor are not captured by file size. The P1 acceptance report records peak and phase boundaries on the same hardware as Lane A. Concurrent LTX/Ideogram work is rejected or queued when their combined reservation exceeds the configured budget.

### 5.11 Golden-tensor oracle and test plan

Add `scripts/export_ideogram4_goldens.py`, following the discipline of `scripts/export_audio_codec_goldens.py`:

- Refuse an unclean or unexpected mflux checkout unless an explicit development override is recorded.
- Pin the mflux commit and HF snapshot; record SHA-256 for all model/config/tokenizer inputs.
- Record Python, NumPy, MLX, OS, hardware, precision, seed, normalized JSON, dimensions, preset, and tap list.
- Write a manifest containing every artifact name, shape, dtype, checksum, summary statistics, and the tolerance class expected by Swift.
- Use deterministic 256×256 fixtures for stage tests and at least one 1024×1536 end-to-end quality fixture. Store small arrays directly and deterministic slices plus checksums/statistics for enormous arrays.
- Export to a temporary directory and atomically replace a versioned fixture directory only after reconciliation passes.

Export these stages:

1. Caption warnings, normalized compact JSON, chat-rendered text, token IDs, token count.
2. Text attention mask and position IDs; outputs for each of the 13 block taps; final concatenated LLM features.
3. Full prompt input tensors: position IDs, segment IDs, indicator IDs, image grid, text padding, conditional and unconditional LLM inputs.
4. Seeded packed noise; latent scale and shift arrays; a synthetic index tensor through unpack; VAE input.
5. `t_values`, `s_values`, stored guidance and effective reversed guidance for all three presets at 256×256 and 1024×1536.
6. RoPE inverse frequencies, selector, cosine/sine, and rotated Q/K for a synthetic tensor.
7. Conditional transformer: input projection, LLM projection, timestep embedding, AdaLN input, block 0 attention branch, block 0 FF branch, block outputs 0/16/33, and final velocity.
8. Unconditional transformer: the same decisive block/final stages, proving that the second weight set is loaded and used.
9. First denoise step `positiveVelocity`, `negativeVelocity`, guided velocity, and updated latent; a short multi-step final latent.
10. Unpacked/scaled latent, VAE decoder checkpoints, and RGB output.

Test suites:

| Suite | Main assertions |
|---|---|
| `Ideogram4CaptionTests` | Actual nested ordering, required fields, unknown keys, palettes, bboxes, literal Unicode, strict-vs-policy split, exact copy, empty regions |
| `Ideogram4TokenizerTests` | Chat template, IDs, no double specials, overflow failure |
| `Ideogram4TextEncoderTests` | Tap indices are block outputs, capture order/reshape, no final norm, source prefix filtering |
| `Ideogram4RoPETests` | Exact selector, axis positions, half rotation, theta |
| `Ideogram4TransformerTests` | AdaLN split/order, tanh gates, fused attention, masks, block and final stage goldens |
| `Ideogram4SchedulerTests` | Inverse CDF, clamp endpoints, resolution mean, tuple lengths, reverse execution |
| `Ideogram4LatentTests` | Seeded shape, channel constants, unpack transpose, dimension validation |
| `Ideogram4WeightTests` | Shards, both DiTs, manifest/hashes, unknown/missing keys, q8 construction before load |
| `Ideogram4QuantizationTests` | Small FP8→BF16 fixtures, affine q8 round-trip, nonzero eligible count per large component, reconstruction metrics |
| `Ideogram4PipelineGoldenTests` | Conditional/unconditional/CFG/first-step/short-run stage parity and cancellation |
| `Ideogram4ServerTests` | Detection, admission, prepare/restore/evict, preset conflicts/default, validation errors, response lineage |
| `PresetStoreTests` | Existing-user migration, no duplicate, user override preservation |

Exact integer/string tensors and manifests must match byte-for-byte. Float schedule values target absolute error at most `1e-6`. BF16 model stages use predeclared per-stage absolute/relative and cosine tolerances derived once from the pinned oracle; tolerances cannot be relaxed merely to make a failing stage pass. Q8 is compared first to the Swift BF16 port using reconstruction error, activation cosine similarity, and velocity error, then by blind output review. Pixel-exact equality is not a q8 acceptance criterion.

Only unit tests are run by the implementation agent:

```sh
xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
```

End-to-end model renders, multi-gigabyte conversion, thermal/performance runs, and quality review remain manual because they require the external weights and target hardware.

### 5.12 P1 acceptance criteria

#### Functional parity

- Swift BF16 conversion-path output passes every pinned stage golden through the first denoise update and VAE checkpoints.
- Text taps are exactly 13 decoder-block outputs with the required ordering and a final width of 53,248.
- Conditional and unconditional checkpoints are independently detected, loaded, and exercised; swapping or duplicating them causes a test failure.
- Scheduler arrays, reverse indexing, guidance schedule, CFG formula, packed noise, scale/shift, and unpatchify match the oracle.
- Strict JSON is the default for native Ideogram routes; a plain prompt or invalid caption never begins allocation-heavy inference.

#### Quantization and quality

- The conversion manifest reconciles all tensors and hashes across FP8 → BF16 → q8.
- Every eligible large component reports nonzero q8 layers, and missing `unconditional_transformer` is a hard failure.
- Q8 stage errors stay within frozen thresholds relative to Swift BF16.
- On a fixed 20-prompt review set, q8 preserves all requested exact text on every image where BF16 does and introduces no additional empty-region violation. Overall blind preference/non-inferiority is recorded rather than asserted.
- A usable poster/label is produced in at most two fixed-seed attempts for the PRD's evaluation briefs.

#### Serving and operations

- `ideogram-design` is available to existing and new preset stores and defaults to `V4_DEFAULT_20`, 1024×1536.
- ModelPool admission prevents incompatible concurrent work; cancellation and failure release reservations and component references.
- A warm-server request appears in Gallery with full prompt/model/preset/seed/output lineage and no P2 Krita exposure.
- The 20-step q8 render time is no slower than Lane A's comparable source render by more than 10%, and the report explains hardware/thermal variance.
- Peak native q8 working set is measured per phase. Release requires staying under the configured admission budget with no swap growth during the steady denoise loop.
- All `ZImageTests` pass; manual conversion, one 1024×1536 source/BF16/q8 triplet, and the review-set report are attached to the release evidence.

## 6. Estimates and sequencing

These estimates assume one engineer familiar with Swift/MLX and access to the pinned weights on the target Apple Silicon machine. Long conversion and 11-minute render feedback loops are included as elapsed risk but not counted as parallel engineering throughput.

### 6.1 P0: 2–3 engineering days

| Workstream | Estimate |
|---|---:|
| Caption type, schema/policy validator, normative template, optimizer route and tests | 1.0–1.5 d |
| Alias/invocation smoke and 20-brief caption fixture review | 0.25–0.5 d |
| Source/saved inventory, q8 bake, memory/timing capture and A/B report | 0.75–1.0 d |

### 6.2 P1: 20–28 engineering days, approximately 4–6 solo weeks

| Workstream | Estimate |
|---|---:|
| Pinned mflux golden exporter, manifests and fixtures | 2–3 d |
| Streaming FP8→BF16 converter, q8 quantizer and loader reconciliation | 3–4 d |
| Transformer core: attention, norms, AdaLN, RoPE, blocks and dual-DiT handling | 4–5 d |
| Shared Qwen3 loader extension, tokenizer/input construction and 13-tap conditioning | 2–3 d |
| Scheduler, packed latent, Flux2 VAE reuse, CFG pipeline and lifecycle | 3–4 d |
| Detection, ModelPool admission/eviction and memory instrumentation | 2–3 d |
| WarmServer, registry, MCP/client, preset migration and Gallery lineage | 2–3 d |
| Unit/golden stabilization, q8 quality review and performance evidence | 2–3 d |

The rev 3 “about one week” estimate is not credible for a tested product port. A one-week spike could plausibly reach selected transformer-block goldens, but not conversion, two-DiT parity, q8 validation, staged memory, server integration, migrations, and acceptance evidence.

## 7. Delivery gates

1. **P0 caption gate:** the template/validator suite and external strict-caption smoke pass.
2. **P0 evidence gate:** the q8 bake report confirms actual tensor behavior and removes any unsupported memory claim.
3. **P1 BF16 gate:** source conversion reconciles and Swift matches oracle stage goldens before q8 work is judged.
4. **P1 q8 gate:** all three large components quantize, stage errors are frozen, and blind image review is non-inferior on copy/empty-space behavior.
5. **P1 serving gate:** admission, cancellation, migration, trace lineage, and only-testing unit suite pass; manual target-hardware evidence is attached.

Failure at one gate does not permit bypassing it with end-to-end images. The stage fixtures exist specifically to distinguish caption, tokenizer, text encoder, transformer, scheduler, latent, VAE, quantization, and serving defects.
