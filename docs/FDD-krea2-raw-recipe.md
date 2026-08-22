# FDD: Krea 2 Raw recipe — parity with the reference stack

**Status:** v2 — review resolved
**Author:** Opus (technical architect)
**Date:** 2026-08-22
**PRD:** coffeeshop-server `docs/prds/2026-08-22-krea2-raw-recipe.md` (PR #1654) — authoritative, source-verified 2026-08-22
**Repos:** `BarkadaBrew/comfybox` (engine, Swift/MLX) + `BarkadaBrew/coffeeshop-server` (client, TypeScript)
**Verified worktrees:** engine `/Users/toddwalderman/Projects/zimage-krea2raw` @ `296735d` (clean `origin/main`) · client `/Users/toddwalderman/Projects/coffeeshop-server/.worktrees/krea2raw` @ `34cb3244` (clean `origin/main`)

This document merges four parallel cluster drafts (Engine A — loop/provenance; Engine B — Raw base/LoRA/VAE; Engine C — samplers/two-stage; Client — policy/plumbing/comparison). Where the drafts conflicted at a seam, §2 names the winner and why. Where my own source verification contradicted a draft, §0.2 says so.

---

## v2 changelog

Two reviewers returned findings against v1. Every BLOCK and MAJOR is resolved below — each by a design change, or by a rebuttal with the source I read. Line numbers cited here were re-verified in this session against engine `/Users/toddwalderman/Projects/zimage-krea2raw` @ `296735d` and client `.worktrees/krea2raw` @ `34cb3244`.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCK | `model: "krea2-raw"` is an alias, not a path — `resolve(spec:)` falls through to the HF Krea-2-**Turbo** snapshot, so the reference preset would render Turbo under the name Raw | **Design changed.** `Krea2ModelDetection` gains an explicit spec→directory table (`specDirectory(_:)`, seeded `krea2-raw → ~/LocalModels/krea2-raw`), consulted before the path check; the HF fallback survives **only** for the four exact turbo aliases; every other non-path spec throws `notAKrea2ModelDirectory`. `WarmServer.parseModelSpec` consults the same table instead of growing a second one. §2 D25, §3.5, AC-34a/34b |
| 2 | BLOCK | Removing the `normal\|simple\|sgm_uniform\|ddim_uniform → .flow` mapping 400s every default Krita render; the justification was overstated | **Design changed.** All four names are **kept as declared aliases**, on both `/v1/generate` and the bridge, and the record carries `sigma_schedule: "flow"` **plus** `sigma_schedule_requested: "normal"` so the alias is visible, not silent. Only names with no implementation and no alias are rejected. The three scheduler option lists (`ComfyBridgeObjectInfo.swift:293,334,354`) are corrected in the same commit as the sampler lists. §2 D22, §3.4, AC-16, AC-16a |
| 3 | BLOCK | The workflow's bypass file (`krea2filterbypass_2vector.safetensors`, civitai 3066812) was never acquired; the FDD silently substituted the **Fedor** artifact at the *other* file's strength and built O9, `krea2-reference` and three ACs on it | **Design changed.** WP-E8 now *acquires* the workflow file (the 99-byte body in `fetch.log` is an auth/early-access response — re-run with a valid `CIVITAI_API_KEY`, or Todd downloads it). `krea2-reference` names the workflow's file. AC-47 pins its SHA-256; new AC-47a dumps both tensors and asserts element-wise equality **or** records non-equivalence as an O6 row. Until it lands, §7.1 marks O9/O4b **"predicted — artifact missing"**, and §10's "F1 makes it unnecessary" bullet is struck. §0.2 F1, D10, §3.8, §3.15, §7.1, §10 |
| 4 | BLOCK | `krea2-reference` decoded through the **Qwen** VAE while the reference stack uses Wan 2.1 FP32 — deferring an axis Todd un-deferred, and contradicting D19's own rule | **Design changed. D16 reversed for the reference preset only.** `krea2-reference.vae` is `Wan2_1_VAE_fp32.safetensors` (layout `wanNative`); Qwen stays the default for the existing turbo lanes (the no-regression contract). AC-29 asserts it. §5.4 ablation (1) now runs with the reference decoder. Q5 is rewritten as "do the *daily lanes* adopt Wan after O6". §2 D16, §3.15, §4 AC-29, §9 Q5 |
| 5 | BLOCK | The preset migration named 3 of the **13** live image presets; the other ten refuse on day one (`krea-film-*` carry `kroma-v0.1` in `loras[]`; five `imported-cs-*` are Z-Image and resolve `unknown-family`) | **Design changed.** §3.17 now carries the full 13-row migration table, read from the live store today. `kroma-v0.1.safetensors → .turbo` joins the relativity seeds (WP-E6). `CHECKPOINT_FAMILY_TABLE` gains `zimage-turbo`/`zimage-base` rows so non-Krea presets keep today's path, and `kroma` is required **only** for krea2-family presets. The image/video discriminator is defined. New AC-44a loads the real `presets.json` fixture and asserts zero refusals. §2 D14, §3.17, AC-44a |
| 6 | MAJOR | The FDD used `sampler` as the wire key everywhere, but `GeneratePayload` decodes the sampler as **`scheduler`** — a client emitting `sampler` would render euler silently and AC-15 would pass vacuously | **Design changed.** New D25 pins the contract: **`scheduler` stays the sampler key** (what MCP and the tile pipeline already post); `sampler` is added as a *decoded alias* with mutual exclusion — both present and different ⇒ 400. WP-C2 emits `scheduler`. `applied.stages[].sampler` (response side, new field) keeps its name. AC-15/AC-36 rewritten; new AC-15a is a decode test that `{"sampler":"res_2s"}` cannot silently produce euler. §2 D25, §3.3, §3.4, §3.14, §3.17 |
| 7 | MAJOR | `eta` is **not** a new field — it is already decoded and forwarded to Z-Image `ddim`/`dpmpp-2s-a`; tier-gating it at the family-agnostic decoder would 400 live Z-Image callers, and `eta` carries two meanings | **Design changed.** Validation splits: **name/enum** validation at `decodedGeneratePayload` (family-agnostic); **tier/family gating** inside `runKrea2Generate` and the krea2 arm of `bridgeGenerate`. The dual semantics (DDIM η vs RES4LYF SDE η) are documented on the field. AC-28 gains the Z-Image `scheduler: ddim, eta: 0.5` regression. §2 D18, §3.3, §3.4, AC-28 |
| 8 | MAJOR | The bridge `.raw` arm read `activePreset?.steps` — no such thing exists: `ComfyBridgeGenerateRequest` has no preset field and presets resolve daemon-side | **Design changed.** D13 drops the preset source: `resolvedSteps = request.steps > 0 ? request.steps : Krea2Variant.raw.defaultSteps`, same shape for guidance (Krita always sends both), values recorded in provenance. §2 D13, §3.5 |
| 9 | MAJOR | `LoRAApplicationReport.unbound` was specified at the two `continue` sites — which fire for every module *without* an adapter (thousands of false entries) and never for an offered key that matches nothing | **Design changed.** `unbound = Set(loraWeights.weights.keys) − consumedScaleKeys`, collected at each successful `addLoRA`; the normalize-failure `continue` is counted separately as `shapeRejected`; `strict && !unbound.isEmpty` throws. The new return is `@discardableResult` (six call sites verified). New AC-42a. §2 D9, §3.6 |
| 10 | MAJOR | F1's "**bit-exact** equal to Raw's own projector weights" is false — the delta is F32 `-0.5116999745` where Raw holds bf16 `-0.51171875`; AC-48's "exactly doubled" only holds because of an unstated dtype cast | **Corrected.** §0.2 F1 now states the values agree to ~5 decimals, and that the doubling identity holds **after** the delta is cast to the bf16 parameter dtype at `LoRAPatchSession.swift:133`, on a transformer loaded bf16 (`Krea2WeightLoader.swift:44-46`) whose projector is excluded from quantization (`Krea2Pipeline.swift:192`). AC-48 rewritten to assert exactly that, at exactly that strength. |
| 11 | MAJOR | Kira's 24/7 lane silently moves 9 → 12 steps on the first client deploy, and §7.5 asserted the opposite | **Design changed and announced.** The preset's declared 12 wins (that a preset's `steps` has never been read is the defect class this PRD exists for) — but `preflightRecipePolicy()` is extended from "log refusals" to "emit a **resolution-diff table**": every (preset × lane × mode) whose resolved steps/guidance change versus the pre-C1 resolver, committed in the C1 PR. New AC-38a. §7.5's claim is relabelled "verified for `krea-bree`; **changes** for `krea-kira` and `krea-film-*`". §2 D14, §3.16, §7.5, AC-38a |
| 12 | MAJOR | O4a's "absent kroma is a configuration error" was enforced only in the daemon — the desktop app, the Krita bridge and MCP callers use engine presets and would render silently | **Design changed.** Engine `PresetStore` validates on **save** (400 naming the preset) and on **load** (error log; the preset is flagged invalid in `/v1/presets` so nothing can select it). New AC-44b/44c. §3.15 WP-E20 |
| 13 | MAJOR | `kroma_strength` in the client record was the *requested* policy value — an echo of exactly the kind D8/R5 forbid | **Design changed.** WP-C4 derives it from `applied.loras[]` by matching the family's kroma `file` and reading `scale_applied`; absent ⇒ `0`; no `applied` ⇒ requested under `provenance:'request'`; requested ≠ applied ⇒ `substituted[]` + warn. AC-45 rewritten to read from a fixture produced by the **real** engine serializer (audit #1653 P0-2). §3.10, §3.17, AC-45 |
| 14 | MAJOR | The deploy copied a binary into `~/Projects/zimage.swift/.build/release/` — the tree another session builds from; the next `swift build` there silently replaces the krea2 engine, and our copy wipes that session's in-flight binary | **Design changed.** §7.3 adopts a **versioned binary path**: `~/.comfybox/bin/ComfyBox-<sha>` with `mlx.metallib` beside it and a `current` symlink the plist points at; the prior version is the named rollback; nothing is ever written into another tree's `.build`. `/health` reports `build_sha` so a clobbered binary is detectable (smoke step 7e). The established "fast-forward the live tree to main and build there" flow is recorded as the fallback if Todd prefers it — new Q9. §7.2, §7.3, §7.4 |
| 15 | MAJOR | The FDD retired the global `baseCheckpoint` habit but never said what the **premium lane's** base becomes; the only classified families were Krea 2, so Kira's Z-Image premium checkpoint resolves `unknown-family` and refuses | **Design changed.** The premium lane's base is the preset `model`, same as standard (O4a). `config.comfybox.baseCheckpoint`/`polishCheckpoint` are ignored with a **startup warning naming both**. `CHECKPOINT_FAMILY_TABLE` gains `zimage-turbo`/`zimage-base` rows with their own step/guidance ranges. New AC-37a. §3.16, §3.17 |
| 16 | MAJOR | PRD O2's first acceptance — "the premium lane produces a materially different render from the daily lane at the same seed" — had no AC and no test | **Added.** New AC-9a (`Krea2LaneRenderTests`, §5.3 Raw batch): quality vs render on `raw-accel` at one seed → non-zero pixel diff, `stages[0].steps_run` 10 vs 6, both `provenance:'engine'`; the same pair on `turbo` (12 vs 9) is recorded for the no-regression table. |
| 17 | MAJOR | R1 is no longer "predicted" — mlx-swift converts a Swift `Float` to the array dtype *before* the op, so AC-1 holds by construction; the real trap is the **opposite** direction on the img2img mix | **Relabelled and re-aimed.** R1 is now verified-by-source (`MLXArray+Ops.swift:253-255`, `DType.swift`, `FlowMatchEulerScheduler.swift:76`). WP-E3 now *specifies* that the img2img mix keeps `scheduler.sigmas[startIndex]` as a **float32 MLXArray**, never `.item()` → Swift scalar; the staged re-noise must declare which it uses. §3.3, §4 note, §8 R1 |
| 18 | MINOR | `SchedulerFactory` `.krea2` with `mu: nil` silently yields an unshifted grid; `Krea2Sampling.schedulerConfig` was unspecified | Fixed — `mu` is required (throws), and the synthetic config's seven values are written down. §3.1, AC-14 |
| 19 | MINOR | `total = int(steps/denoise)` must mirror Python double truncation | Fixed — `stage2.denoise` decodes as `Double`, `Int((Double(steps)/denoise).rounded(.towardZero))`, with a parametrised fixture test. §3.14, AC-31 |
| 20 | MINOR | AC-64 asserts decoding into `GenerateResponse`, which is a **private, `Encodable`-only** struct | Fixed — AC-64 is scoped to `ImageJobStatus` and the client decoders; we do **not** add `Decodable`. |
| 21 | MINOR | `client.mode-resolution.test.ts` narrowing was misdescribed | **Partly rebutted, and tightened.** `testLowerStepsPassThrough` (`:40-44`) contains **only** the below-cap assertion (`steps: 4 → 4`) and does survive verbatim; the clamp assertions the reviewer cites at `:48` and `:56` live in `testModeStepsClampedToo` and `testExplicitStepsWinOverModeButStillClamped` — which v1 already counted as "the three clamp tests". §5.1 now names the surviving assertions line by line, and preserves `:49`'s below-cap mode assertion. |
| 22 | MINOR | `Krea2Variant.raw.defaultGuidance = 3.5` would fire for every Raw request that omits `guidance` — including raw-accel with the turbo LoRA — silently doubling model evals | **Design changed** (the reviewer's alternative adopted): engine `Krea2Variant.raw.defaultGuidance = 1.0` (== off). **3.5 belongs to the client's `raw-stock` family policy**, which always sends it explicitly. §3.5, §3.16 |
| 23 | MINOR | Putting `vae` in the pool key makes a 508 MB decoder swap a 22.5 GB, ~67 s full reload | **Design changed.** The pool key is unchanged; `Krea2Pipeline.ensureVAE(path:layout:)` reloads the decoder **in place**, fail-closed, and records the change. AC-59 now asserts the intent (never a silent reuse: the recorded `vae` always matches the request, and a differing request increments a reload counter) rather than a pool eviction. §2 D17, §3.9, AC-59 |
| 24 | MINOR | Several load-bearing citations were wrong | Fixed: `r4_rk_coeff.py` `:1294` ralston_2s / `:1241` ralston_3s / `:1207` ralston_4s; `getPreset` `client.ts:690`; the baseCheckpoint echo `image-gen-tools.ts:871` (and the sidecar twin at `:782`); the async body spread `client.ts:995`; `ImagePreset.steps/guidance/scheduler` `PresetStore.swift:83/87/94`; "all four sites" → five, enumerated; `r4_noise.py` hard mode `:444-445` + `:279-293`. |
| 25 | MINOR | `deis_4m` is named verbatim in PRD O8 yet appeared twice as a schedule-pressure cut candidate | Fixed — `deis_4m` and `ralston_4s` are removed from both cut lists; `res_3s` is the only engineering-owned cut. §2 D20, §6 |
| 26 | MINOR | AC-69 asserted only that the negative-prompt pair *differs*, not that it differs "in the direction requested" | Fixed — AC-69 routes direction to §5.4 ablation row (4), blind-captioned, and states that direction is an **O6 judgement, not a pixel assertion**. |
| 27 | MINOR | §7.1 did not name Raw's actual source (the gated `krea/` fetch failed; the file came from the Comfy-Org mirror); the `bong_tangent`/`beta` fixtures were labelled "independent derivation" when they are reproducible from upstream | Fixed — §7.1 records `Comfy-Org/Krea-2 diffusion_models/krea2_raw_bf16.safetensors`, 26,283,332,608 bytes, official repo gated, with a SHA-256 pin obligation; AC-19/21/31 are relabelled **verified against upstream source (pure-Python re-run 2026-08-22)**. |
| 28 | MINOR | The FDD never answered the PRD's ask to review `origin/bree/sampler-expansion` / `sampler-fixes`, nor the batched-vs-sequential CFG half of open question 2 | **Added as D24, with the branch review done in this session.** Every commit on both branches has landed on `main` **by content** — the scheduler tree via `65c56de`, and the three sampler fixes and the CFG batch>1 fixes verified present in the worktree today (`DDIMScheduler.swift:78-79` alpha guard, `SchedulerFactory.swift:96` `eta: eta ?? 1.0`, `DPMPlusPlus2SAScheduler.swift:91` inverted ratio, `ZImagePipeline.swift:1172-1179` / `ZImageControlPipeline.swift:1083-1121` `modelTimestep`). Nothing to resume. The only unlanded commit that touches our surface is `77ba7f8` (beta via CDF), which D5 supersedes. Batched CFG: **out of scope**, cost acknowledged in R7. |
| 29 | MINOR | Four components had named test files but no acceptance row | Added AC-5a (bridge `.turbo` byte-identical / `.raw` honours steps+CFG+negative), AC-5b (`Krea2Variant` defaults feeding `runKrea2Generate`), AC-59a (the mandatory handoff log line), AC-58 widened to round-trip **all nine** new `ImagePreset` fields. |
| 30 | MINOR | The new `ComfyBoxGenerateOptions` fields had no named operator surface | Fixed — §3.17 WP-C2 now states the surface per field (tool-schema parameter vs preset-only vs CLI-only) and adds a tool-schema test to `client.recipe-wire.test.ts`. AC-68a. |

**One BLOCK is answered by acquisition, not by design (item 3):** the workflow's bypass file must be downloaded before `krea2-reference` can honestly claim parity. That is tracked in §7.1 and §9 Q3/Q4, and it gates WP-E21 the same way T3 does.

---

## 0. Problem & context (measured)

### 0.1 What the PRD established, in one paragraph

PRD §1 and §9 are the record: the daemon has been rendering the most conservative recipe the engine can produce — guidance pinned to 0 (`client.ts:545`), steps clamped to 9 (`client.ts:549`), Euler over Krea 2's native warp because `Krea2Pipeline` has no sampler parameter at all — and nobody chose it. Negative prompts have been inert on every daemon render since the CFG branch shipped. The premium lane and the daily lane are the same recipe. The reference workflow the PRD adopts differs from us on six axes at once (base, acceleration, censorship, sampler/schedule, CFG, VAE) and we match on one (text encoder). PRD §9 records the three findings that reshape the work: O1 is engine work, not plumbing; O2/O3 are client-only; O7's original acceptance contradicted its own source. Todd's scope ruling stands over everything here — **parity on every axis, measurement after parity, outcomes are not optional.**

I have not restated the PRD's evidence. This FDD assumes it.

### 0.2 What I verified today that changes a draft's shape

Four findings, each read from disk in this session. Two of them retire work the drafts had sized as significant.

**F1 — A bypass LoRA of this shape needs no new mechanism. It is not, however, the workflow's file.** Engine B sized O9 as blocked on acquisition with an unbounded `Krea2ConditioningAdapter` (S→M, "do not commit to a number"). The *mechanism* half of that is superseded. The *acquisition* half is not, and v1 blurred them — corrected here.

**What is on disk** (`ls ~/comfybox-models/loras/vault/`, re-checked this session): `krea2_filter_bypass_fedor.safetensors` (1,040 bytes, valid safetensors — civitai 2746817, "Fedor"). **What is not on disk:** `krea2filterbypass_2vector.safetensors`, the file the reference workflow links (civitai 2728234, version 3066812, ~160 B). `scratchpad/fetch.log:3` records the attempt — `bypass lora: 99 bytes (expect ~160) head: {"error":"Early…` — an auth / early-access response, not the artifact. **The reference stack's bypass LoRA has not been obtained.** O9 and the `krea2-reference` preset are therefore written against the workflow's file (§3.8, §3.15), and §7.1 marks them **predicted — artifact missing** until it lands.

The Fedor file is what F1's arithmetic below was read from. It contains exactly one tensor:

```
diffusion_model.txtfusion.projector.diff   F32  [1, 12]   48 bytes
values: [0, 0, 0, 0, 0, 0, 0, 0, -0.51172, -0.89062, 0, 0]
```

This is a **bare-parameter `.diff` delta**, which `LoRAWeightLoader.loadForKrea2` already parses (`:561-563`) and `LoRAPatchSession.apply` already applies fail-loud with a full preflight (`LoRAPatchSession.swift:56-105` — resolves every target or throws `partialApplication` with nothing mutated). The target resolves: `remapKrea2Base` strips `diffusion_model.`, `mapTransformerKey` is a no-op on this path, and `txtfusion.projector.weight` exists in `raw.safetensors` as `F32 [1, 12]` (verified — 430 tensors, download complete at 26 GB). Shapes match exactly.

Three consequences:
- **O9's engine mechanism is zero new code.** The work is a load test, a provenance field, and the strength policy.
- **The 8-bit black-image warning does not apply to this tensor.** `Krea2Pipeline.swift:192` excludes `projector` from quantization (`!path.contains("projector")`), so the target is full-precision on the deployed q8 base. PRD O9's stated risk is retired for this artifact (it remains live for any future bypass that touches a quantized Linear).
- **The file's own semantics are now legible.** `txtfusion.projector` is `Linear(numTxtLayers=12, 1)` (`Krea2Transformer.swift:289-296`) — the learned weighting over the 12 text-encoder layer taps (`selectLayers = [2,5,…,35]`). The delta's two non-zero values **track Raw's own projector weights at columns 8 and 9 to about five decimal places — they are not bit-exact** (corrected in v2; v1 claimed bit-exactness and was wrong). Read from the files this session:

```
raw.safetensors  txtfusion.projector.weight  F32  cols 8,9 = -0.51171875,           -0.890625
fedor  …projector.diff                       F32  cols 8,9 = -0.5116999745368958,  -0.8906000256538391
numpy ==                                                     False                  False
```

Raw's two values are exactly representable in bf16; the delta's are the same numbers typed to five decimals. So the bypass is `w'[8,9] ≈ (1 + s) · w[8,9]` — a scalar amplification of two layer taps, authored against Raw. **The doubling is exact only under a stated dtype chain**, which v1 never named: `LoRAPatchSession.apply` casts the delta to the parameter dtype *before* scaling (`LoRAPatchSession.swift:133` — `current + tensor.asType(current.dtype) * scale`), the Krea 2 transformer is loaded bf16 (`Krea2WeightLoader.swift:44-46`), and the projector is excluded from quantization (`Krea2Pipeline.swift:192`), so `bf16(-0.5116999745) = -0.51171875` and the sum is exactly `2·w` in bf16. At a bf16-disabled load, or at a strength other than 1.0, that identity does not hold and only the ~4e-5 relative agreement does. AC-48 is written against the chain, not against the arithmetic in the abstract.

At the workflow's `s = 1.0` those taps double; at the Fedor author's `s = 3–5` they quadruple to sextuple. **The two published strength recommendations are not near each other, and the file's metadata claim that they are "identical numerical effect" is the file author's assertion, not a verified fact** — which is exactly why WP-E8 now acquires the workflow's file and AC-47a compares the tensors rather than trusting the claim. Calibration: tensor name, shape, dtype and values are verified by reading the file; every prose claim in its `__metadata__` (authorship, "any variant", the refusal-channel explanation) is unverified.

The same file loaded against `kroma-v0.2/turbo.safetensors` also shape-matches (that projector is `[1, 12]` BF16, cols 8,9 = `[-0.53516, -0.98438]`), so it applies to Turbo too — but the doubling identity holds only on Raw.

**F2 — `raw.safetensors` is complete and layout-identical to turbo — and it came from the Comfy-Org mirror.** 430 tensors, same unprefixed key space (`blocks.*`, `txtfusion.*`, `tmlp.*`). Draft B's claim that `Krea2WeightLoader.loadTransformer` needs no change is confirmed against the finished file, not just its header. Only filename resolution changes. **Provenance, corrected in v2:** the official `krea/Krea-2-Raw` repo is gated and the direct fetch produced a 149-byte body (`fetch.log:5`); the 26,283,332,608-byte file on disk is Comfy-Org's `diffusion_models/krea2_raw_bf16.safetensors` (`fetch.log:7`). §7.1 records the source and the SHA-256 pin obligation.

**F3 — The silent-substitution bug Draft B found is live and confirmed.** `Krea2ModelDetection.detect(at:)` requires `turbo.safetensors` by name (`:16-24`); `resolve(spec:)` falls through to `Krea2ModelPaths.resolve(modelDir: nil)` — "newest HF cache snapshot of krea/Krea-2-Turbo" — with no error and no log (`:27-36`). Pointing the engine at `~/LocalModels/krea2-raw` today renders Krea-2-Turbo silently. This is not new work dressed as a fix; it is a bug fix that O4 happens to require.

**F4 — Engine A and Engine C are both right about the EDM range, and the PRD §7 concern is retired.** `SchedulerFactory.resolveSigmas` routes `karras/exponential/beta/beta57` through `flowMatchingSigmaBounds` → `(shiftedSigmaMin, 1.0)`, never the `0.02 … 100` function defaults. No EDM leakage exists on the factory path. What *is* wrong is separate and worse: our `beta` is not ComfyUI's `beta` (§2, D5).

---

## 1. Target architecture

One request shape, one recipe record, two repos. The client owns *policy* (which family, which lane, which kroma strength); the engine owns *physics* (which checkpoint file is actually loaded, which sampler actually ran, which LoRA pairs actually bound) and is the sole authority on what the record says.

```
┌───────────────────────── coffeeshop-server (client) ─────────────────────────┐
│                                                                              │
│  image-gen-tools ──lane──▶ image-lane-map        {intent, tier}              │
│         │                        │                                           │
│         │                        ▼                                           │
│         │              checkpoint-policy.ts   ◀── ComfyBoxPreset             │
│         │              ┌──────────────────────────────────┐                  │
│         │              │ CheckpointFamily                 │  POLICY layer    │
│         │              │   turbo | raw-accel | raw-stock  │  (what we mean   │
│         │              │ resolveRecipe(): ok | refusal    │   to ask for)    │
│         │              │   steps/guidance/sampler/stage2  │                  │
│         │              │   kroma{strength,file}           │                  │
│         │              └──────────────────────────────────┘                  │
│         │                        │  RecipeResolution                         │
│         ▼                        ▼                                           │
│    ComfyBoxClient.generate() ── POST /v1/generate ──────────┐                │
│                                                             │                │
│    buildRenderSettings() ◀── applied{} ◀── response ────────┤                │
│         provenance: 'engine' | 'request'                    │                │
│         substituted[]                                       │                │
└─────────────────────────────────────────────────────────────┼────────────────┘
                                                              │
┌─────────────────────────── comfybox (engine) ───────────────▼────────────────┐
│                                                                              │
│  WarmServer.decodedGeneratePayload  ── validateRecipeNames() ──▶ 400 or pass  │
│         │                               (FAIL LOUD: no coercion to euler)    │
│         ▼                                                                    │
│  runKrea2Generate ──▶ ModelPool.poolKey(spec, quant) ──▶ PoolEntry           │
│         │                        └─ Krea2Pipeline.ensureVAE(path:layout:)     │
│         │                           in-place decoder swap, fail-closed (D17)  │
│         │                                                    │               │
│         │                             Krea2ModelDetection ───┤ PHYSICS layer │
│         │                               Krea2Variant         │ (what is      │
│         │                                 .turbo | .raw      │  actually     │
│         │                               FAIL CLOSED on a     │  loaded)      │
│         │                               dir it cannot name   │               │
│         ▼                                                    ▼               │
│  Krea2Pipeline ─── loadLoRAs (transactional, strict)                         │
│         │            ├── relativity guard  (requiresBase vs variant)         │
│         │            ├── LoRAApplicationReport {offered, bound, unbound}     │
│         │            └── LoRAPatchSession   (.diff — the bypass path)        │
│         │                                                                    │
│         ▼                                                                    │
│  Krea2DenoiseLoop.run(scheduler, evaluate, …)      ◀── SchedulerFactory      │
│    ├─ convert v → x0 when scheduler.modelOutputConvention == .dataPrediction │
│    ├─ 1-row  : step()                              SchedulerKind             │
│    ├─ 2-row  : intermediateStep/finalizeStep        euler heun dpmpp-2m      │
│    ├─ N-row  : TableauScheduler rows                dpmpp-2s-a deis ddim     │
│    ├─ eta SDE injection (T2)                        res_2s res_3s           │
│    └─ bongmath fixed point (T3)                     ralston_2s/3s/4s        │
│         │                                           deis_2m/3m/4m           │
│         ▼                                                                    │
│  generateStaged: stage1 ─latent─▶ re-noise ─▶ stage2 ─▶ ONE vae.decode      │
│         │                                          SigmaScheduleKind         │
│         ▼                                           krea2 flow karras        │
│  RenderRecipe  ──▶ 4 sinks:                         exponential beta beta57  │
│    response.applied · PNG EXIF · /health.last_recipe · async job status      │
│    every field READ BACK from pipeline state, never echoed from the request  │
└──────────────────────────────────────────────────────────────────────────────┘
```

Two invariants hold the design together:

1. **Nothing is silently substituted, anywhere.** A sampler or schedule name with no implementation and no *declared* alias is a 400 — and a declared alias (`normal` → `flow`) is recorded on the render as both names, so even the aliasing is visible (D22). A model directory the engine cannot classify is an error, and so is a **model *alias* nobody has mapped** — neither falls back to the HF turbo snapshot (D25). A LoRA that binds a strict subset of its offered keys throws. A krea2-family preset without a declared kroma strength refuses, in the engine *and* in the daemon. A missing VAE fails the render. Where an artifact we need is simply not on disk, the document says so on its face rather than substituting a lookalike (§7.1, O9).
2. **The record is read back, not echoed.** Every field in `RenderRecipe` is sourced from pipeline state after the fact — resolved enums the factory received, loop-counted evaluations, `loadedLoRAConfigs` (only non-empty after a successful transactional apply), resolved file paths. The client's `provenance: 'engine' | 'request'` makes it visible when that guarantee is absent.

---

## 2. Key design decisions

Each decision names the conflict it resolves, the option that won, and what was rejected.

---

**D1 — One denoise-loop driver, N-row protocol from day one; the existing 1-row and 2-row protocols survive inside it.**

*Conflict:* Engine A designs `Krea2DenoiseLoop.run` against today's 2-evaluation protocol (`intermediateStep`/`finalizeStep`). Engine C needs 3–4 evaluations per step for `ralston_3s`/`ralston_4s`/`res_3s` and adds a `TableauScheduler` protocol with `rows`/`rowSigma`/`rowSample`/`commit`.

*Decision:* One driver, in Engine A's shape (a pure function with an injected `evaluate(latent, sigma) -> velocity`, testable with no weights), carrying **three** dispatch branches from the first commit: 1-row `step()`, 2-row `intermediateStep`/`finalizeStep`, and N-row `TableauScheduler`. Engine C's protocol is added in the same work package that lands the driver (WP-E3), even though its first consumer (WP-E13) arrives later.

*Rejected:* (a) A's 2-row driver now, widened later — the byte-identity gate (AC-1/AC-2) is the most expensive test in the programme and re-running it after a protocol widening costs more than defining the protocol once. (b) C's tableau replacing the existing protocol — `HeunScheduler` and `RES2sScheduler` pass their current tests with bespoke implementations; rewriting them onto a tableau breaks Z-Image byte-identity for no outcome.

*Consequence:* WP-E3 is larger than Engine A sized it (M→M+, the protocol plus the branch), and WP-E13 is smaller.

---

**D2 — `ModelOutputConvention` is `.velocity | .dataPrediction`, and the Z-Image `res_2s` bug is fixed — in its own work package, with an acceptance criterion that names the break.**

*Conflict:* A names the cases `.velocity/.denoised` and explicitly declines to fix `ZImagePipeline` so that "Z-Image is byte-identical" survives as an acceptance criterion. C names them `.velocity/.dataPrediction` and fixes `ZImagePipeline:1226-1233` because `res_2s` is stage 1 of the recipe we are adopting.

*Decision:* Naming goes to C — `.dataPrediction` is unambiguous; "denoised" is what ComfyUI calls the model output in general, including for velocity models. The fix goes to C as well: **`ZImagePipeline` is corrected.** Verified independently: `RES2sScheduler.step` computes `e^{-h}·x + h·φ₁(-h)·m` with `h = -log(σ'/σ)` (`RES2sScheduler.swift:65-68`, `:165-169`), which is the exponential-integrator data-prediction form; `ZImagePipeline.swift:1226-1233` feeds it `-guidedNoise`, a velocity. The solver is dimensionally wrong on the Z-Image path today.

*Rejected:* A's "file it separately." The PRD's entire thesis is that a value nobody passed and nobody could see cost us a week. Knowingly shipping a wrong solver on one family while fixing it on another creates exactly the invisible, family-dependent asymmetry §5 of the PRD outlaws — and it would be discovered by whoever next compares `res_2s` across families.

*Cost, stated plainly:* Z-Image `res_2s` output changes. A's acceptance criterion "the full Z-Image suite passes unchanged and a Z-Image Turbo render is byte-identical" is retained **for the default euler/flow path only**; a new criterion (AC-24) asserts the `res_2s` change is intended, is measured, and is recorded in the changelog. The fix is isolated in WP-E2 so that WP-E3's Krea 2 byte-identity gate is never entangled with it.

---

**D3 — Both sigma grids are reachable; `mu` stays the default; the reference preset states `shift: 1.15` explicitly; provenance records which ran. (Engine C OD-1, the highest-impact open decision.)**

*Conflict:* ComfyUI registers Krea 2 with a **fixed** shift 1.15 (`comfy/supported_models.py:1931-1934`) and the published workflow was authored under it. Our engine computes a **resolution-dependent** `mu` from Krea's own reference `sampling.py` (`Krea2Pipeline.swift:104-124`) — effective shift `e^0.9062 = 2.475` at 1024×1024. Every model-consulting schedule (`flow`, `beta`, `beta57`, `karras`, `exponential`) therefore lands on a different grid than the published run. `bong_tangent` is immune (§2, D6).

*Decision:* option (c). Add an optional `shift: Float?` request field. `nil` (the default) = today's resolution-dependent `mu`, so every existing render is unmoved. A non-nil value overrides `mu` with `log(shift)` for schedule construction. The `krea2-reference` preset sets `shift: 1.15`. `RenderRecipe` records `mu`, `shift`, and `shift_source: "dynamic" | "explicit"`.

*Rejected:* (a) match ComfyUI globally — moves every existing Krea 2 render's schedule and diverges from Krea's own reference sampler, for a parity target that only stage 1 of one preset needs. (b) keep `mu` only — then "exactly as published" is not exactly as published, and O8's acceptance is unmeetable by construction.

*Why this matters more than it looks:* this is the single most likely reason a "parity" render still will not look like the workflow author's, and it is invisible without the provenance field.

---

**D4 — Two-stage requests use a single optional `stage2` object; the *record* uses a `stages[]` array.**

*Conflict:* Engine C OD-3 — `stage2` object (additive, exactly the reference) vs `stages: […]` (generic, matches chroma-generate's `MultiStageScheduler`).

*Decision:* `stage2` on the request (C's recommendation — the recipe has exactly two stages; an array invites per-stage prompt/LoRA/VAE semantics nobody has specified and nothing would test). But the **record** carries `stages: [{…}, {…}]`, one entry per stage that actually ran, one entry for a single-stage render. Request shape and record shape need not match, and a record that changes cardinality between one and two stages is unreadable.

*Rejected:* symmetric `stage2` in the record — makes "what sampler ran" a two-branch lookup for every consumer, including the O6 report.

---

**D5 — `SigmaSchedule.beta` is replaced in place with the ComfyUI-exact algorithm, with a pinned before/after fixture. (Engine C OD-2.)**

Verified independently that the two disagree wildly: ComfyUI's `beta_scheduler` evaluates the beta **PPF**, rounds to an index in the model's discrete 1000-entry sigma table, de-duplicates and appends 0; ours integrates the beta **CDF** and interpolates in log-sigma space. At 6 steps, shift 1.15, our σ₁ is 0.1596 where ComfyUI's is 0.9199.

*Decision:* replace. The current implementation is not the schedule its name claims — shipping it under that name is itself a silent substitution, and the reference recipe's stage 1 cannot run without the fix.

*Rejected:* a new name with the old deprecated — leaves a landmine named `beta` in the dropdown and in the tile pipeline (`src/tile/seamless-processor.ts:90-98`, the only live client caller).

*Obligations this creates:* a changelog note, a pinned before/after fixture in the test suite, and an explicit rollout line (§7) — this is a behaviour change for `beta`/`beta57` callers, announced, not slipped in. ComfyUI's de-duplication can also return fewer than `steps+1` sigmas; `SchedulerFactory` constructs with the **actual** count and `RenderRecipe.steps_effective` reports it (AC-14).

---

**D6 — `bong_tangent` ignores the model entirely; Krea 2's resolution shift is *not* composed on top.**

Verified in the upstream source in the scratchpad: `res4lyf_sigmas.py:4076-4098` accepts `model_sampling` and never references it in the body. The schedule is pure index arithmetic emitting flow sigmas `1.0 → 0.5 → 0.0`. `resolveSigmas` therefore takes `case .bongTangent` and deliberately ignores `config` and `mu`; `RenderRecipe` records `shift_applied: false`. This answers the PRD §7 open question directly and it is not a judgement call.

---

**D7 — Engine `Krea2Variant` and client `CheckpointFamily` are different layers, not competing enums.**

*Conflict:* Engine B introduces `Krea2Variant { turbo, raw }`; the client introduces `CheckpointFamily { turbo, raw-accel, raw-stock }`. Read as competitors they are inconsistent (2 vs 3 values).

*Decision:* they are not competitors. **`Krea2Variant` is a physical fact** read from the checkpoint file the engine actually loaded; it is never requested, only reported. **`CheckpointFamily` is a policy label** the client declares on a preset: what step budget, guidance range, sampler pairing and kroma artifact we intend for that checkpoint. `raw-accel` and `raw-stock` are two policies over one physical variant (`raw`) — accelerated by the turbo LoRA at 0.6, or stock at 52 steps / CFG 3.5.

The seam is a **consistency check, not a mapping**: the client refuses with `family-mismatch` when `/health.model_variant` (or `applied.base_variant`) contradicts the declared family's required variant. Until the engine reports it (WP-E5/WP-E10), the client records `family_verified: false` rather than trusting silently.

*Rejected:* collapsing the two — either the engine would have to know about step budgets (policy in the physics layer) or the client would have to infer the checkpoint from a filename (guessing, which is F3's bug in a different repo).

---

**D8 — One provenance struct, named `RenderRecipe` in Swift, emitted on the wire as `applied`.**

*Conflict:* three names for the same object — A's `Krea2RunRecipe` (response field `recipe`), B's `RenderRecipe`, the client's `applied`.

*Decision:* Swift type `RenderRecipe`; wire field `applied`. The wire name goes to the client because `applied` states the contract — this is what applied, not what you asked for. The struct is the **union** of all three drafts' fields plus C's per-stage block, with `stages[]` per D4.

*Rejected:* A's `recipe` on the wire — reads as "the recipe" without saying whose, and the client's whole read-back design turns on the distinction.

---

**D9 — `LoRAApplicator.applyDynamically` gains `strict: Bool = false` and returns a `LoRAApplicationReport`; only Krea 2 passes `strict: true`.**

Engine B's design, ratified — **with the detection site corrected in v2.** The silent-partial-bind hazard is real: a four-deep stack whose third adapter binds 12 of 264 targets logs nothing today. But v1 specified `unbound` as an accumulation at the two `continue` sites (`LoRAApplicator.swift` — the `guard let pair else { continue }` after the qkv fallback, and the normalize-failure `continue`), and that is the wrong instrument twice over: the first fires for **every** Linear module that has no adapter, which is the overwhelming majority for any sparse LoRA (thousands of false entries), and an *offered key that matches no module* is never visited at either site, so the one case that matters would not appear at all.

*Corrected mechanism:* collect `scaleKey` at each successful `addLoRA` into `consumedScaleKeys`; after the module walk, `unbound = Set(loraWeights.weights.keys).subtracting(consumedScaleKeys)`. The normalize-failure `continue` is counted separately as `shapeRejected` (a real, different fault). `strict && !unbound.isEmpty` throws `partialApplication` naming the keys. `bound < offered` remains the strict *test*; `unbound` is now the strict *evidence*.

The new return is `@discardableResult` — verified six call sites ignore it today (`ZImagePipeline.swift:933`, `ZImageControlPipeline.swift:627`, `Krea2Pipeline.swift:239,279,308`, `Flux2Pipeline.swift:283`), and warnings-as-errors would break every one of them. The `strict: false` default keeps Z-Image/Flux2/Chroma byte-identical while the signature change compiles across all of them.

*Rejected:* B's own alternative — a Krea2-only wrapper that re-walks the module tree to count bindings. Less invasive, but it duplicates the matching logic and will drift from the applicator it is shadowing, which is how the `videoTuning` class of bug happens.

*Sequencing obligation:* this lands as its own commit with a clean build across every family before anything stacks on it (WP-E6).

---

**D10 — O9's mechanism is the existing `.diff` delta path. No `Krea2ConditioningAdapter` is built.**

*Mechanism* superseded by F1; *acquisition* is not. Engine B's conditioning-adapter design was correct reasoning from a 99-byte error page; the artifact of this shape that we can read is a bare-parameter `.diff` delta the engine already handles fail-loud, so **no `Krea2ConditioningAdapter` is built**.

**But the reference stack's file is still missing** (§0.2 F1). v1 substituted Fedor for it and declared O9 satisfied; that is precisely the silent substitution this document exists to prevent, and at the *other* file's recommended strength. Corrected: WP-E8 is XS **plus an acquisition step** — a load test, a shape-mismatch test, a strength-policy constant, a provenance field, **and** obtaining `krea2filterbypass_2vector.safetensors` (re-run the civitai fetch with a valid `CIVITAI_API_KEY`, or Todd downloads it) with AC-47's SHA-256 pin and AC-47a's tensor comparison against Fedor. If the two tensors are equal, Fedor is a verified stand-in and says so; if they differ, Fedor leaves `krea2-reference` and its 3–5 strength becomes an O6 ablation row. Until the file lands, O9 and O4b are **predicted, artifact missing** (§7.1) — and since WP-E21 is already gated on T3 (D19), no preset claims parity in the meantime.

B4a (fail-loud preflight on a JSON error page masquerading as `.safetensors`) is **retained** — it is one guard and one test, and the 99-byte file that motivated it still exists somewhere in Todd's download history. It moves into WP-E6.

---

**D11 — `sigma_schedule: "flow"` stays legal on Krea 2. (Engine A OD-2.)**

It is genuinely a different schedule from the native `krea2` warp (different base grid, different penultimate sigma), so a caller who types it expecting "the default" gets something else. But `RenderRecipe.sigma_schedule` now names what ran, which is the mitigation the PRD asks for; rejecting `flow` on one family would fragment a shared enum for a hazard that provenance already closes.

---

**D12 — `RenderRecipe` is populated for Krea 2 only; other families emit no `applied` block, and that gap is a filed ticket, not a silent asymmetry. (Engine A OD-3.)**

The client's `provenance: 'request'` branch covers absence honestly. Generalising to flux1/flux2/fibo/chroma now triples WP-E10 for outcomes nobody asked for. The obligation: file the follow-up ticket in the same PR that lands WP-E10, and state the asymmetry in the field's doc comment.

---

**D13 — Krita bridge Raw defaults are read from the active preset, not hardcoded. (Engine A OD-1.)**

Three candidate constants exist (workflow: CFG ≤ 2; Krea README: 52 steps / CFG 3.5; the Z-Image Base arm in the same switch: 40 steps). Inventing a fourth is how magic numbers breed.

**v1's answer — "read the active preset" — had no data source and is withdrawn.** Verified: `ComfyBridgeGenerateRequest` (`ComfyBridgeWorkflowParser.swift:11-51`) carries prompt / negative / width / height / steps / guidance / seed / sampler / sigmaSchedule and **no preset field**; `grep activePreset` returns nothing in `WarmServer.swift`; and image presets resolve daemon-side by design (`WarmServer.swift:7553-7555`). There is no preset to read.

*Corrected:* the `.raw` arm takes what Krita actually sends, and falls back to the variant defaults only when it sends nothing —

```swift
resolvedSteps    = request.steps > 0    ? request.steps    : Krea2Variant.raw.defaultSteps      // 30
resolvedGuidance = request.guidance > 0 ? request.guidance : Krea2Variant.raw.defaultGuidance   // 1.0
```

Krita sends both on every render, so in practice the fallbacks are unreachable and the arm is a pass-through with the negative prompt restored. Both values are recorded in provenance. The `.turbo` arm is byte-identical to today (AC-5a).

---

**D14 — Kroma is a first-class preset field, not a `loras[]` entry.**

The client draft's recommendation, ratified: absence from a list is indistinguishable from "off", which is precisely what O4a forbids; a per-render override needs an address; provenance needs a column. A preset that declares `kroma` **and** lists a kroma file in `loras[]` is a `preset-invalid` config error. `file` defaults by family — `raw-accel`/`raw-stock` → `kroma-v0.2-base-lora-rank-384-fro-0985.safetensors` (Raw-relative, verified 3.6 GB in the vault), `turbo` → `kroma-lora-v0.3.safetensors` (Turbo-relative, what `krea-kira` runs today).

`kroma-v0.1.safetensors` (what the three `krea-film-*` presets carry today, Turbo-relative) joins the relativity seeds in WP-E6.

**`kroma` is required only of krea2-family presets.** The live store also holds five Z-Image image presets (`imported-cs-*`); requiring a kroma declaration of them would be nonsense, and refusing them for the absence of one — which is what v1's rule did — takes the daemon's five oldest presets offline on deploy day. `validateImagePreset` therefore keys on the resolved family: krea2 families require `kroma`, `zimage-*` families are validated on their own ranges and keep today's path.

This forces an engine preset-schema change (WP-E20), an **engine-side** validation (also WP-E20 — O4a's rule has to hold for the desktop app, the Krita bridge and MCP, not only for the daemon), and a migration of **all thirteen** live image presets (§3.17), which drives the deploy ordering in §7.

---

**D15 — Kroma-on-Raw is accepted as the dial, and its fidelity gap is recorded per render, not footnoted. (Engine B's open decision.)**

Verified by Draft B and consistent with what I read: the Raw-relative rank-384 extraction carries 256 low-rank pairs and **zero** bare-parameter deltas, while the Turbo-relative `kroma-lora-v0.3` carries 170 (`prenorm`/`postnorm`/`qknorm` scales, `mod.lin`, `first`, `last`). Kroma's norm/modulation changes are unreachable on Raw at any strength with the file we have. "kroma at 1.0 on Raw" is therefore **not** equivalent to `kroma-v0.2-turbo`, independent of the Frobenius truncation — which answers PRD open question 3's remaining half.

*Decision:* accept, and make `RenderRecipe.loras[].deltas_applied` a required field so a Raw+kroma render records `deltas_applied: 0` on its face. Extracting a new Raw-relative kroma that includes deltas is a real but unscoped work item; it is out of scope here (§10).

---

**D16 — `krea2-reference` decodes through Wan 2.1 FP32. The existing turbo lanes keep Qwen-Image as their default. (Engine B's open decision — reversed in v2 for the reference preset.)**

v1 gave `krea2-reference` the Qwen decoder and argued the note.com kroma × VAE interaction would confound O6. That argument does not apply to *this* preset: `krea2-reference` declares `kroma: {strength: 0}`, so there is no kroma to interact with, and §5.4 ablation row (7) already isolates the VAE axis on its own held-seed pair. What v1 actually did was defer an axis Todd explicitly un-deferred (PRD §6: "that deferral is withdrawn"; O7: "the default direction is adoption, not investigation"), while D19 in the same document refuses to ship "a preset named *reference* that is not the reference". Both cannot stand.

*Decision:* `krea2-reference.vae = Wan2_1_VAE_fp32.safetensors`, layout `wanNative`, asserted by AC-29. Every other preset — `krea-bree`, `krea-kira`, `krea2-base`, `krea-film-*` — keeps the model-directory Qwen-Image VAE, which is the no-regression contract. Wan is still never ambient: it is a named preset field that appears in every record, and selecting a VAE that is not on disk still fails the render (AC-56).

What O6 now decides is not whether the *reference* uses Wan — it does, by definition — but whether the **daily lanes** adopt it. §9 Q5 is rewritten accordingly, and §5.4 ablation row (1) ("current vs full reference") runs with the reference decoder in place, so it does not have to be repeated later — the exact cost D19 cites for `bongmath`.

---

**D17 — Do not raise the pool budget. Batch O6 by base instead. (Engine B's open decision.)**

Measured: budget is 40,960 MB (`ModelPool.swift:246`, `COMFYBOX_POOL_BUDGET_MB` unset in the live plist — verified); a `.krea2` entry is estimated at 22,528 MB (`:149-152`); two cannot co-reside; the live log already shows ~20 evict-and-handoff cycles/day at a measured ~67 s each. Adding Raw makes a third `.krea2` spec. Raising the budget to ~48 GB would hold Raw and Turbo warm but would push every video render into a reload, because #218 already vacates the LTX-2 stack for any image load. Mitigation is scheduling — the O6 runner batches every recipe on one base before switching — plus a mandatory handoff log line naming outgoing and incoming **variant** so a slow A/B is attributable rather than mysterious (AC-59a).

**The VAE does not join the pool key** (corrected in v2). v1 added a `vae:` term to `ModelPool.poolKey` (`:490-498`) to stop a Wan request being served by a resident Qwen pipeline. It would have worked, and it would have made a 508 MB decoder swap cost a 22.5 GB eviction and reload — ~67 s — every time the O6 runner crosses ablation row (7), and every time a per-preset VAE differs. Instead, `Krea2Pipeline.ensureVAE(path:layout:)` reloads the decoder **in place** on the resident instance (`Krea2WeightLoader.loadVAE` into the existing `Krea2VAE`), under the same fail-closed checks: a file that is not on disk, or a layout `detectLayout` cannot name, fails the render and leaves the resident decoder untouched. The guarantee AC-59 protects is unchanged and is stated directly — *the recorded `vae` always names what decoded, and a request for a different VAE than the resident one always reloads* — but it is now asserted on the recorded path and a reload counter rather than on a pool eviction.

---

**D18 — The three parity tiers (T1 schedules+samplers, T2 `eta` SDE, T3 `bongmath`) are all in scope, sequenced, and an unimplemented tier is a 400, never a downgrade.**

Todd's scope ruling makes all three mandatory. They are separated because each has its own oracle fixture and can fail independently. `eta != 0` before T2 lands, or `bongmath: true` before T3, returns 400 naming the missing capability.

**Where that gate lives, corrected in v2.** `eta` is **not** a new field: `GeneratePayload` already declares it (`WarmServer.swift:7514`), decodes it (`:7626`) and forwards it to the Z-Image request on both the t2i and img2img paths (`:7760`, `:7832`), where `SchedulerFactory` consumes it for `ddim` and `dpmpp-2s-a`; the CLI exposes `--eta` (`main.swift:159-160`). A tier gate applied at `decodedGeneratePayload` — which runs **before** the family is known — would 400 live Z-Image callers. The same reasoning applies to `bongmath`, `shift` and `stage2`, which are krea2-only.

So validation splits by kind, not by location convenience:

| validation | where | why |
|---|---|---|
| sampler / schedule **names** are known enum members or declared aliases | `decodedGeneratePayload` (`:4193-4204`) — family-agnostic | a name is wrong for every family |
| `stage2.denoise > 0`, ranges, mutual exclusion (`sampler` vs `scheduler`, D25) | same choke point | structural, family-agnostic |
| **tier gating** — `eta != 0` before T2, `bongmath` before T3, `stage2` at all | inside `runKrea2Generate`, and the `.krea2` arm of `bridgeGenerate` | krea2-only capability, and `eta` on Z-Image is a *different, shipped* parameter |

`eta` therefore carries two meanings on one field — DDIM η / DPM++ 2S-A ancestral η on the Z-Image path, RES4LYF SDE η on the Krea 2 path — and that is written on the field's doc comment rather than left to be discovered. AC-28 gains the regression: Z-Image `scheduler: "ddim", eta: 0.5` still succeeds after WP-E4 and WP-E15 land.

---

**D19 — The reference preset `krea2-reference` is created only when T3 lands. (Engine C OD-4.)**

An interim engineering configuration may exist during T1/T2 validation, but it is not the named preset and is not described as parity. `bongmath` is a declared field in the record either way, so a render can always be read for whether it ran.

*Rejected:* shipping the preset with `bongmath: false` declared. It is honest, but it puts a preset named "reference" in the store that is not the reference, and O6's parity run would then have to be repeated.

---

**D20 — `res_3s` is IN; `deis_4m` and `ralston_4s` are IN. (Engine C OD-5.)**

`deis_4m` is named in PRD O8's acceptance verbatim, so it is not optional; `ralston_4s` is its warm-up and comes with it. `res_3s` is C's recommendation and costs ~60 lines plus a fixture once the N-row driver exists.

**Corrected in v2:** v1 listed `deis_4m` and `ralston_4s` as second-tier cut candidates, which frames a mandatory PRD outcome as optional ("Nothing here is optional" — PRD §3). They are removed from the cut list entirely. **`res_3s` is the only engineering-owned cut**; changing anything else in O8's named set goes through §9 Q6.

---

**D21 — The existing `deis` wire name keeps pointing at the same math. (Engine C OD-6.)**

`DEISScheduler.swift:44-81` is a first-order exponential-integrator Euler, not DEIS. The Swift type is renamed `ExponentialEulerScheduler` with a doc comment recording what it is; the `"deis"` wire name is unchanged so the tile pipeline's live behaviour does not move. Repointing `deis` at `deis_2m` would be a deliberate behaviour change for a live caller — the exact thing this FDD exists to prevent.

---

**D22 — Fail-loud parsing applies at enqueue, and a persisted job that fails replay is marked failed with the reason, never silently rendered. (Engine A's open decision.)**

The alias table is preserved in full (`res_2s`, `dpmpp_2m`, `dpmpp_2s_ancestral`, `beta57`, plus the RES4LYF UI prefixes `exponential/` and `multistep/` which are stripped so a workflow value pastes verbatim).

**v1 removed the `normal|simple|sgm_uniform|ddim_uniform → .flow` mapping. That is reversed in v2 — it would 400 every default Krita render on deploy day.** Verified this session: Krita AI Diffusion's built-in sampler presets map their scheduler field to exactly these names — `"Euler": "normal"`, `"DPM++ 2M": "normal"`, `"DPM++ 2M SDE": "normal"`, `"DDIM": "ddim_uniform"`, `"UniPC BH2": "ddim_uniform"`, `"LCM": "sgm_uniform"`, `"Lightning": "sgm_uniform"` (`~/Library/Application Support/krita/pykrita/ai_diffusion/style.py`, `_scheduler_map`) — the workflow parser lifts the string verbatim (`ComfyBridgeWorkflowParser.swift:286`), `bridgeGenerate` forwards it untouched (`WarmServer.swift:3289` `sigmaSchedule: request.sigmaSchedule`), and our own `/object_info` advertises all four in three places (`ComfyBridgeObjectInfo.swift:293,334,354`). Krita renders on this engine as its ComfyUI backend. Rejecting the names we advertise, that our only GUI client sends by default, is not fail-loud — it is an outage.

v1's justification was also overstated. On a discrete-flow model these names are not four unrelated schedules: `normal` and `simple` sample the model's own shifted sigma table at uniform `t`, which is what `.flow` computes. I have not bit-verified that equality against ComfyUI source in this session and do not claim it — which is the reason for the mechanism below rather than a bare "they're the same".

*Decision:* the four names stay as **declared aliases**, on `/v1/generate` and the bridge alike, and the alias is made **visible instead of silent** — the record carries both:

```
"sigma_schedule": "flow",  "sigma_schedule_requested": "normal"
```

That is D11's own argument (provenance closes a name hazard that rejection would close by breaking callers), applied consistently. `ddim_uniform` is kept on the same terms rather than singled out, because Krita's DDIM and UniPC styles send it and it aliases to the same grid. If a later measurement shows any of the four wants its own implementation, it gets one and the alias resolves to it — with the request name still recorded, so nothing about that change is silent either.

What is genuinely **rejected** is a name with no implementation and no alias: `uni_pc`, `dpmpp_2m_sde`, `linear`, `ays`, `res_5s`, `lcm`, `sigmoid_offset`. Those are advertised today (`uni_pc` at `:302,333,353`; `dpmpp_2m_sde` at `:333`; `linear` in the MCP schema) and become euler/flow silently. The `/object_info` **sampler** lists and the **scheduler** lists at `:293,334,354`, and the MCP schema, ship their corrections in the **same commit** as the rejection, never after (AC-17). Every value in Krita's `style.py` maps gets a case in `SamplerNameResolutionTests` and `BridgeKrea2VariantTests` (AC-16a).

---

**D23 — `res2s_c2` is not exposed on the wire. (Engine A's open decision.)** It stays at 0.5 until a recipe needs it. Adding a knob nobody has asked for is surface area with no outcome behind it.

---

**D24 — The stale sampler branches are not resumed: their content is already on `main`. Batched CFG stays out of scope. (PRD §8 Q6 and the remainder of Q2 — answered here because both were addressed to this document.)**

PRD §8 Q6 asks to check `origin/bree/sampler-expansion` (14 commits) and `origin/bree/sampler-fixes` (16) before restarting. Done this session, commit by commit against `origin/main` @ `296735d`:

- The Phase 1–3 scheduler work (`02d1a44`, `2df8205`, `cb273a9`) **is on main** — it landed as `65c56de`, which is why `git diff origin/main...branch` still shows those files as additions from an old merge base.
- The three sampler bug fixes in `2e8bba5` are **on main by content**, verified in the worktree: the DDIM step-0 α-guard (`DDIMScheduler.swift:78-79`), `eta: eta ?? 1.0` forwarding (`SchedulerFactory.swift:96`), and the inverted DPM++ 2S-A ancestral ratio (`DPMPlusPlus2SAScheduler.swift:91`).
- The **CFG batch=2 transformer fixes** the PRD calls out (`683f1fd`, `7e079e0`, `6433454`) are likewise on main by content: `modelTimestep` duplication at `ZImagePipeline.swift:1172-1179` and `ZImageControlPipeline.swift:1083-1121`, and `.expandedDimensions(axis: 1)` throughout `ZImageControlTransformerBlock.swift`.
- The only unlanded commit that touches our surface is `77ba7f8` ("use beta CDF directly, not inverse CDF (ppf)") — which points the **wrong way**: D5 replaces our CDF integrator with ComfyUI's PPF-and-round algorithm. Resuming it would re-litigate a decision we have a pinned fixture for.

*Decision:* **restart, not resume.** Nothing on either branch is needed and neither is rebased on 4 months of main. They stay unmerged; this FDD's §5.2 fixtures supersede their tests.

*Batched vs sequential CFG* (PRD §8 Q2's remaining half): **out of scope.** Krea 2's CFG branch runs two sequential transformer calls (`Krea2Pipeline.swift:343-382`); the batch=2 path those branch commits fixed is Z-Image's, not Krea 2's. Batching would roughly halve CFG wall-clock on the overnight lane at ~2× the activation memory on a pool that already cannot hold two `.krea2` entries (D17). The cost of *not* doing it is stated in R7 and is visible per render as `model_evals_total`. Revisit only if R7's numbers make the overnight lane infeasible.

---

**D25 — The sampler's wire key stays `scheduler`; `sampler` is a decoded alias with mutual exclusion.**

*Conflict:* v1 wrote `sampler` as the wire field everywhere (§3.14's payload, WP-C2's emitted body, AC-15). The engine decodes the sampler as **`scheduler`** (`WarmServer.swift:7511` field, `:7626` `CodingKeys`, `:7730` `parseSchedulerKind(scheduler)`), and both live senders already post that spelling — `MCPToolExecutor.swift:256-261` and the tile pipeline (`src/tile/seamless-processor.ts:90-98`). A client emitting `sampler` against an unchanged decoder would have its key **ignored** and every sampler request would render euler — the precise failure class this document exists to prevent — and AC-15 would have passed vacuously, because an unknown JSON key is not a 400.

*Decision:* `scheduler` remains the sampler key. `sampler` is added to `CodingKeys` as an accepted alias so a value pasted out of a ComfyUI/RES4LYF UI works; both present **and different** is a 400 (`.mutuallyExclusive`, the pattern already at `WarmServer.swift:7847-7855`). The **response** side keeps `applied.stages[].sampler`, which is a new field with no legacy and the clearer name for what it reports. The asymmetry is deliberate and is documented on both structs.

*Rejected:* renaming the request key to `sampler`. It is a live-traffic break on two senders for a cosmetic gain, on the same day we are already changing name resolution.

---

## 3. Component design

Work packages are named `WP-E*` (engine) and `WP-C*` (client) and are sequenced in §6.

### 3.1 WP-E1 — `SigmaScheduleKind.krea2` and the mu seam

**Mechanism.** `Krea2Sampling.timesteps` (`Krea2Pipeline.swift:104-124`) emits `steps+1` points over `linspace(1 → 0)` inclusive of exact 0, warped by `exp(mu)/(exp(mu) + (1/t − 1)^σ)`. `SigmaSchedule.flow` applies the identical warp to a *different* base grid (`linspace(1 → shiftedSigmaMin)` plus a 0.0 sentinel). They are not synonyms; `.flow` must not be Krea 2's default.

**Changes.**
- `SigmaSchedule.swift:4-10` — `+case krea2 = "krea2"`.
- `SigmaSchedule.swift` — `+static func krea2(numSteps: Int, mu: Float, sigmaExp: Float = 1.0) -> [Float]`.
- `Krea2Pipeline.swift:104-124` — `Krea2Sampling.timesteps` keeps its signature and **delegates** to `SigmaSchedule.krea2`, so bit-identity is structural.
- `Krea2Pipeline.swift` — `+Krea2Sampling.mu(seqLen:align:)`, implemented by delegating to `PipelineUtilities.calculateShift(imageSeqLen:baseSeqLen:256, maxSeqLen:6400, baseShift:0.5, maxShift:1.15)`.
- `Krea2Pipeline.swift` — `+Krea2Sampling.schedulerConfig(shift:)`, a synthetic `ZImageSchedulerConfig` carrying the constants baked into `timesteps` (Krea 2 has no `scheduler_config.json`). **Its values are specified, not left implicit** (v1 left them undefined, which made AC-14 and D11 untestable — `flowMatchingSigmaBounds` reads `config.shift` and `numTrainTimesteps` for karras/exponential/beta, and `SigmaSchedule.flow` applies its warp only when `useDynamicShifting == true`):

  ```swift
  static func schedulerConfig(shift: Float? = nil) -> ZImageSchedulerConfig {
    ZImageSchedulerConfig(
      numTrainTimesteps: 1000,
      shift: shift ?? 1.0,        // D3: explicit shift moves the karras/beta bounds with it
      useDynamicShifting: true,   // so .flow and .krea2 both take the mu warp
      baseShift: 0.5, maxShift: 1.15,
      baseImageSeqLen: 256, maxImageSeqLen: 6400)
  }
  ```
  With `shift: nil` the warp comes from `mu` and `flowMatchingSigmaBounds` yields `(0.001, 1.0)`; with `shift: 1.15` (the `krea2-reference` preset) the bounds and the warp move together. Both are pinned by AC-14 and AC-21.
- `ModelConfigs.swift:141-159` — public memberwise `init` for `ZImageSchedulerConfig` (currently `Decodable`-only; tests build it by round-tripping JSON).
- `SchedulerFactory.swift:119-147` — `+case .krea2`. **`mu` is required**: `guard let mu else { throw SchedulerFactoryError.missingMu(.krea2) }`. v1 wrote `mu: mu ?? 0`, which silently produces an *unshifted linear grid* — a silent default in the one document that says there are none. The `kind == .euler && schedule == .flow` fast path is untouched, so Z-Image is bit-unaffected.

### 3.2 WP-E2 — Model-output convention, and the `res_2s` correction

```swift
public enum ModelOutputConvention: Sendable { case velocity, dataPrediction }

public protocol ZImageScheduler {
  /// What quantity `step`/`finalizeStep` expect in `modelOutput`.
  /// Linear-frame solvers integrate dx/dσ = v and take a velocity.
  /// Exponential-frame solvers (res_*, dpmpp_*, ddim) integrate in
  /// h = −log(σ'/σ) and take the data prediction x₀ = x − σ·v.
  var modelOutputConvention: ModelOutputConvention { get }
  …
}
public extension ZImageScheduler {
  var modelOutputConvention: ModelOutputConvention { .velocity }   // every existing scheduler
}
```

`RES2sScheduler` (and later `RES3sScheduler`) override to `.dataPrediction`. Callers convert once per evaluation, **after** CFG combination (CFG is linear in `v`, so the order is safe):

```swift
let out = scheduler.modelOutputConvention == .dataPrediction ? (x - sigma * v) : v
```

For rectified flow (`x_t = t·ε + (1−t)·x₀`, confirmed by the img2img mix at `Krea2ImageToImagePipeline.swift:124`) with the model emitting `v = ε − x₀`, RES2s's update with `modelOutput = x₀` gives `x' = (t'/t)·x + (1 − t'/t)·x₀ = t'ε + (1−t')x₀` — algebraically exact. With velocity it is meaningless.

**Changes:** `ZImageScheduler.swift` (protocol + extension default), `RES2sScheduler.swift` (one line), `ZImagePipeline.swift:1193/1226-1233` (the conversion). Per D2 this is a deliberate, measured behaviour change on the Z-Image `res_2s` path.

### 3.3 WP-E3 — The Krea 2 denoise loop, request fields, and the N-row protocol

**The spine of the programme.** Everything in Engine C except the pure schedules, and O5 entirely, is blocked on this.

```swift
// Sources/ZImage/Krea2/Krea2DenoiseLoop.swift  (new)
enum Krea2DenoiseLoop {
  struct Stats { let stepsRun: Int; let modelEvals: Int }

  /// Pure driver. Knows nothing about transformers or VAEs — `evaluate` is the
  /// real transformer in production and a closed-form velocity field in tests.
  /// `sigma` is Krea 2's t directly: Krea2Pipeline.swift:375 passes ts[i] into
  /// the transformer with no (1000 − t)/1000 renormalisation.
  static func run(
    scheduler: inout any ZImageScheduler,
    initialSample: MLXArray,
    startIndex: Int,
    evaluate: (_ latent: MLXArray, _ sigma: Float) -> MLXArray,   // returns velocity
    noise: SDENoiseInjector?,        // nil until T2
    bongmath: BongMath?,             // nil until T3
    progress: ((Int, Int) -> Void)?
  ) -> (sample: MLXArray, stats: Stats)
}
```

Per-step dispatch, in order:
1. `scheduler.reset()` before step `startIndex` (multistep caches: `DPMPlusPlus2MScheduler.previousOutput`, `RES2sScheduler.firstModelOutput`).
2. `v = evaluate(x, σ_i)`; `out = convert(v)` per WP-E2.
3. If `scheduler is TableauScheduler` → run `rows` evaluations at `rowSigma(i, r)` / `rowSample(i, r, …)`, then `commit`.
4. Else if `scheduler.requiresIntermediateEvaluation` → `intermediateStep`, evaluate at `scheduler.intermediateSigma(i)` (`RES2sScheduler.swift:92-101` returns `σ·exp(−c₂h)` — a genuine substep, **not** `σ_{i+1}`), convert, `finalizeStep`.
5. Else `scheduler.step`.
6. T2: `noise?.inject(&x, step: i)`. T3: `bongmath?.iterate(…)`.
7. `MLX.eval(x)`, `progress?(i+1, total)`, count `modelEvals`.

```swift
// Sources/ZImage/Pipeline/Scheduler/ZImageScheduler.swift  (additive)
public protocol TableauScheduler: ZImageScheduler {
  var rows: Int { get }                                                   // model evals/step
  func rowSigma(timestepIndex: Int, row: Int) -> Float
  mutating func rowSample(timestepIndex: Int, row: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray
  mutating func commit(timestepIndex: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray
}
```

**Request fields**, added as trailing parameters of the existing inits so `main.swift:4656`, `Krea2Img2ImgSpike.swift:23` and `WarmServer.swift:6850,6859` keep compiling:

```swift
public struct Request {          // Krea2Pipeline.Request, Krea2Pipeline.swift:149-179
  … existing …
  public var sampler: SchedulerKind = .euler        // today's behaviour. WIRE KEY IS `scheduler` (D25)
  public var sigmaSchedule: SigmaScheduleKind = .krea2
  public var shift: Float? = nil                    // D3: nil = resolution-dependent mu
  /// RES4LYF SDE eta (T2). NOTE: the SAME wire field `eta` means DDIM η /
  /// DPM++ 2S-A ancestral η on the Z-Image path, where it has shipped since
  /// April. Two meanings, one key — gated per family, never at the decoder (D18).
  public var eta: Float = 0.0                       // T2
  public var bongmath: Bool = false                 // T3
  public var c2: Float = 0.5                        // res_2s / res_3s substep (not on the wire, D23)
}
```

`Krea2Pipeline` also retains what provenance needs and currently discards in `init` (`:184-209`): `public let paths: Krea2ModelPaths`, `public let transformerQuantBits: Int?`, `public let variant: Krea2Variant`.

`generate`/`generateImg2Img` keep their `-> MLXArray` signature; new siblings `generateWithRecipe` / `generateImg2ImgWithRecipe` return `(image: MLXArray, recipe: RenderRecipe)` and the old ones become one-line wrappers. **No shared "last recipe" state** — the server reads the value it was handed.

**The img2img mix keeps its float32 promotion — this is a byte-identity trap in the direction v1 did not name.** Today `Krea2ImageToImagePipeline.swift:123-124` builds `let tStart = MLXArray(ts[startIndex])`, a **float32 0-d array**, so `noise * tStart + sourceLatent * (1 - tStart)` promotes the whole mix to float32 and casts to bf16 once at the end. If the refactor rewrites that with a Swift `Float` scalar (`scheduler.sigmas[startIndex].item(Float.self)`), mlx-swift converts the scalar to the array's dtype **first** (`MLXArray+Ops.swift:253-255` — `lhs.asMLXArray(dtype: rhs.dtype) * rhs`), the mix runs in bf16, and AC-2 fails. **WP-E3 therefore specifies: the mix takes `scheduler.sigmas[startIndex]` as a float32 `MLXArray`, never `.item()`.** The staged re-noise (WP-E17) is new code and may choose either, but must say which in its doc comment and record it. See §8 R1 for why the *other* direction — the euler step itself — is safe.

`Krea2Pipeline.generate:318-366` (noise, conditioning, positions, control tokens) is **unchanged**. Only `:368-389` is replaced. `generateImg2Img` calls the same loop with `startIndex` from `Krea2ImageToImagePipeline.swift:115-116` and mixes at `scheduler.sigmas[startIndex]`.

**Cost is multiplicative and must be reported, not discovered:**
`modelEvals = stepsRun × rows × (guidance > 1 ? 2 : 1)`. `res_2s` + CFG 2.0 at 6 steps is 24 transformer calls against today's 9; `deis_3m` at 2 steps is 6 (see WP-E14 — DEIS never engages at 2 steps). That number is a required `RenderRecipe` field.

### 3.4 WP-E4 — Fail-loud parsing and advertised-list reconciliation

```swift
static func resolveSchedulerKind(_ raw: String?) throws -> SchedulerKind?      // nil == absent
static func resolveSigmaScheduleKind(_ raw: String?) throws -> SigmaScheduleKind?
```

Replacing `parseSchedulerKind`/`parseSigmaScheduleKind` (`WarmServer.swift:7856-7881`), which today coerce anything unrecognised to `.euler`/`.flow` (verified). Both return the **resolved** kind *and* the raw string, so the record can carry `sigma_schedule_requested` (D22). Alias table per D22 — including `normal`/`simple`/`sgm_uniform`/`ddim_uniform`, which are **kept**. New `WarmServerError` cases `unknownSampler(name:valid:)` / `unknownSigmaSchedule(name:valid:)` with messages naming the offending value and the valid set, and matching arms in the exhaustive `switch error` at `:4268-4277` returning 400.

**What validates where** (D18 — v1 put all of it at one choke point, which would have 400'd live Z-Image `eta` callers):

- **Name resolution and structural checks** happen once, at the single decode choke point `decodedGeneratePayload` (`:4193-4204`, used by both `/v1/generate` and `/v1/generate/async`), via `try payload.validateRecipeNames()`: sampler/schedule names resolve or throw; `stage2.denoise > 0`; `scheduler` and `sampler` are not both present with different values (D25). All family-agnostic, all safe before the family is known.
- **Tier and family gating** happens inside `runKrea2Generate` and the `.krea2` arm of `bridgeGenerate`: `eta != 0` before T2, `bongmath: true` before T3, `stage2` on a non-krea2 family, `shift` on a non-krea2 family. `eta` on the Z-Image path is untouched and keeps working (AC-28).

`bridgeGenerate` builds its payload directly (`:3277-3297`) and runs the same name resolution there.

Same commit, all of it (AC-17): the **sampler** lists at `ComfyBridgeObjectInfo.swift:302,333,353` drop `uni_pc`, and `:333` also drops `dpmpp_2m_sde`; the **scheduler** lists at `:293,334,354` are reconciled against `SigmaScheduleKind.allCases` **plus** the kept aliases (so `normal`, `simple`, `sgm_uniform`, `ddim_uniform`, `karras`, `exponential`, `beta` all stay advertised and all resolve, and `bong_tangent`/`beta57` are added). `MCPToolRegistry.swift:122-129` replaces its free-text option lists with `enum` arrays generated from `SchedulerKind.allCases` / `SigmaScheduleKind.allCases`, so the schema cannot drift again (it currently advertises `linear`, which does not exist).

### 3.5 WP-E5 — `Krea2Variant`, fail-closed resolution, pool and bridge

```swift
public enum Krea2Variant: String, Sendable {
  case turbo, raw
  var transformerFilename: String   // "turbo.safetensors" | "raw.safetensors"
  var supportsGuidance: Bool        // false | true
  var defaultSteps: Int             // 9 | 30
  var defaultGuidance: Float        // 1.0 (== off) | 1.0 (== off)   ← see below
  var bridgeStepClamp: Int?         // 12 | nil
}
```

**`Krea2Variant.raw.defaultGuidance` is 1.0, not 3.5** (corrected in v2). v1 set 3.5 — Krea's stock Raw recipe — as the engine default, which fires for *every* Raw request that omits `guidance`. MCP `generate_image` posts guidance only when it is given (`MCPToolExecutor.swift`), and so do desktop callers, so a raw-**accel** render with the turbo LoRA stacked would have silently doubled its model evals and activated an empty negative prompt. An engine default that doubles cost is a surprise, and surprises are what this document is against. **3.5 belongs to the client's `raw-stock` family policy** (§3.16), which always sends it explicitly and records it. `defaultSteps` stays 30: a neutral, CFG-free engine default for a direct Raw call, with 6 / 10 / 52 owned by the client families.

**Resolution order.** Two things are wrong today and v1 only fixed one of them.

*Directory resolution* in `Krea2ModelPaths.detect(at:)`, replacing the single-filename check (F3's bug):
1. `raw.safetensors` → `.raw`; `turbo.safetensors` → `.turbo`. **Both present → throw `ambiguousVariant(dir)`.** Never guess.
2. Neither → read `model_index.json` for `"krea2_variant"` + `"transformer_file"` (escape hatch for a third filename).
3. Still nothing → throw `notAKrea2ModelDirectory(path, reason:)`.

*Spec resolution* in `Krea2ModelDetection.resolve(spec:)` — **new in v2, and the fix for a hole v1's own §3.15 preset fell straight through.** The `krea2-reference` preset names `model: "krea2-raw"`, which is an alias, not a path. Under v1's rule ("the HF-snapshot fallback survives only for a spec that is not an existing filesystem path") that alias is not a path, so `Krea2ModelDetection.resolve` (`:27-36`) would have fallen through to `Krea2ModelPaths.resolve(modelDir: nil)` — "newest HF cache snapshot of **krea/Krea-2-Turbo**" — and the reference preset would have rendered **Turbo under the name Raw**. The engine has no spec→directory mapping for it: the only such table today is `WarmServer.parseModelSpec`'s hardcoded `civitaiPaths`, where `"kroma-v0.2-turbo": "~/LocalModels/kroma-v0.2"` is the sole Krea 2 entry (`WarmServer.swift:4408-4420`).

```swift
public enum Krea2ModelDetection {
  /// Aliases that mean the HF Krea-2-Turbo snapshot, and nothing else.
  static let turboAliases: Set<String> = ["krea2", "krea-2", "krea-2-turbo", "krea/krea-2-turbo"]

  /// Declared spec → directory. Seeded from config (`~/.comfybox/config.json`
  /// `krea2Models`), defaulting to the installs on this machine. THE single
  /// table — `WarmServer.parseModelSpec` consults it instead of growing a second.
  static func specDirectory(_ spec: String) -> URL?   // "krea2-raw" → ~/LocalModels/krea2-raw
                                                      // "kroma-v0.2-turbo" → ~/LocalModels/kroma-v0.2

  public static func resolve(spec: String) throws -> Krea2ModelPaths {
    if isExistingDirectory(spec) { return try detect(at: url(spec)) }        // 1. explicit path
    if let dir = specDirectory(spec) { return try detect(at: dir) }          // 2. declared alias
    if turboAliases.contains(spec.lowercased()) {                           // 3. the ONLY fallback
      return try Krea2ModelPaths.resolve(modelDir: nil)
    }
    throw Krea2ModelPathsError.notAKrea2ModelDirectory(spec, reason: .unmappedSpec)  // 4.
  }
}
```

`isKnownKrea2Model` gains `krea2-raw`/`krea-2-raw` so `detectFamily` (`ModelPool.swift:501-502`) routes them to `.krea2` — but that is now *only* a family hint; the directory comes from the table above, and an alias nobody has declared throws at `loadPipeline` (`:558-563`) instead of silently loading Turbo. AC-34a/34b assert both halves. `Krea2ModelPaths.transformerFile` becomes a stored `let`. `PoolEntry.detectedInfo` (already `Any?`, already per-family) carries the variant back from `loadPipeline`; `poolActivate` stores it in `currentKrea2Variant` beside `currentZImageVariant` (`WarmServer.swift:5913-5916`). The directory probe at `ModelPool.swift:536` is kept — it is what makes Raw work under a custom dir name.

`runKrea2Generate:6799` `payload.steps ?? 9` → `?? variant.defaultSteps`; `:6852,6861` `payload.guidance ?? 1.0` → `?? variant.defaultGuidance`.

**Bridge** (`WarmServer.swift:3222-3228`, verified: pins guidance 0, drops the negative, clamps `min(steps, 12)` for the whole family) splits per D13:

```swift
case .krea2:
  switch await coordinator.currentKrea2Variant {
  case .turbo:   // byte-identical to today
    resolvedSteps = request.steps > 0 ? min(request.steps, 12) : 9
    resolvedGuidance = 0.0
    resolvedNegativePrompt = nil
  case .raw:     // request-sourced, no clamp, live negative (D13 — corrected in v2:
                 // there is no "active preset" on a bridge render; the request has it all)
    resolvedSteps = request.steps > 0 ? request.steps : Krea2Variant.raw.defaultSteps
    resolvedGuidance = request.guidance > 0 ? request.guidance : Krea2Variant.raw.defaultGuidance
    resolvedNegativePrompt = request.negativePrompt
  }
  resolvedSampler = request.sampler
```

Mandatory log line on every base handoff: `krea2 handoff: <outgoing spec/variant> → <incoming spec/variant> (loadTimeMs=…)` (D17).

### 3.6 WP-E6 — LoRA relativity, application report, strict apply

```swift
public struct LoRAApplicationReport: Sendable {
  public let offered: Int          // loraWeights.weights.count
  public let bound: Int            // appliedCount
  public let quantizedBound: Int
  public let deltasApplied: Int    // LoRAPatchSession.preflight's resolved count
  public let shapeRejected: Int    // pairs that matched a module but failed normalizeLoRAPair
  public let unbound: [String]     // OFFERED KEYS THAT BOUND NOTHING. sorted; capped at 32 in logs
}

// LoRAConfiguration gains:
public var requiresBase: Krea2Variant?     // declared, never inferred
// LoRALibraryEntry gains:
public var krea2Relative: Krea2Variant?    // tolerant decode, defaults nil
```

`applyDynamically` gains `strict: Bool = false`, returns the report, and throws `LoRAError.partialApplication` when `strict && !unbound.isEmpty`. Only `Krea2Pipeline.loadLoRAs` and `setControlLoRA` pass `strict: true`.

**`unbound` is computed from the offered keys, not from the module walk** (corrected in v2 — see D9). Inside the walk, every successful `addLoRA` inserts its `scaleKey` into `consumedScaleKeys`; a pair that matched a module but failed `normalizeLoRAPair` increments `shapeRejected` and is **not** consumed. After the walk:

```swift
let unbound = Set(loraWeights.weights.keys).subtracting(consumedScaleKeys).sorted()
if strict && !unbound.isEmpty { throw LoRAError.partialApplication(lora: name, unbound: unbound) }
```

v1 specified accumulation at the two `continue` sites. That is wrong twice: `guard let pair else { continue }` fires for **every** Linear module with no adapter — thousands of false entries for any sparse LoRA — and an offered key matching no module is never visited at either site, so the one failure mode that matters would have been invisible. AC-42a pins it: a LoRA with one key that targets no module reports `bound 263/264` and `unbound == [that key]`, and throws under `strict`.

The new return is **`@discardableResult`**. Verified: six call sites ignore it today — `ZImagePipeline.swift:933`, `ZImageControlPipeline.swift:627`, `Krea2Pipeline.swift:239,279,308`, `Flux2Pipeline.swift:283` — and without the attribute every one of them warns.

The relativity guard sits inside the existing transactional `do` block at `Krea2Pipeline.swift:234-241`, **before** any weight mutation, so the existing rollback (`:242-248`) already restores the base:

```swift
if let required = config.requiresBase ?? libraryEntry?.krea2Relative, required != self.variant {
  throw LoRAError.incompatibleBase(lora: name, requires: required, loaded: self.variant)
}
```

Seeded relativities: `krea2_turbo_lora_rank_64_bf16` → `.raw`; `kroma-v0.2-base-lora-rank-384-fro-0985` → `.raw`; `kroma-lora-v0.3` → `.turbo`; **`kroma-v0.1` → `.turbo`** (added in v2 — it is what the three live `krea-film-*` presets carry, and without a seed their migration cannot be validated).

Also in this WP: `LoRAScanner.detectCompatibilityFromKeys` (`:203-281`) gains a Krea 2 branch (`txtfusion.` + `blocks.N.attn.w{q,k,v,o}` ⇒ `["krea2"]`) — every Krea 2 LoRA in the vault is classified `unknown` today. And B4a: a preflight in `LoRAWeightLoader` throwing `notASafetensorsFile(path, firstBytes:)` when a file is JSON-sniffed **and** header-parse fails (never a size check alone — the real bypass file is 1,040 bytes and must load).

### 3.7 WP-E7 — kroma on Raw (zero new code)

Verified by Draft B against the real files: `kroma-v0.2-base-lora-rank-384-fro-0985.safetensors` yields 256 pairs / 0 deltas / 0 unconsumed, all 256 targets exist in `raw.safetensors`, per-layer rank is dynamic (98–299) and safe because `LoRAConfiguration.effectiveScale(forLayer:)` reads each layer's own dims, and no `.alpha` keys exist so the applied scale passes through verbatim. F32-onto-q8 is handled by `LoRAQuantizedLinear` reading the unpacked shape.

The work is two tests and the `deltas_applied: 0` provenance field that records D15's fidelity gap on the face of every render.

### 3.8 WP-E8 — The bypass LoRA (F1, D10)

No new mechanism. **One acquisition, then five small things.**

**Step 0 — acquire the reference stack's file.** `krea2filterbypass_2vector.safetensors` (civitai model 2728234, version 3066812, ~160 B, the file the workflow links) is **not on disk**; the fetch in `fetch.log:3` returned a 99-byte `{"error":"Early…` body, which is an auth / early-access response, not a 404. Re-run with a valid `CIVITAI_API_KEY`, or Todd downloads it from Buzz. This is a work item, not a footnote: without it, `krea2-reference` cannot honestly claim the censorship axis, and §7.1 carries O9/O4b as **predicted — artifact missing**.

Then:
- A load test pinning the artifact — 1 delta, key `diffusion_model.txtfusion.projector.diff`, `F32 [1,12]`, resolving to `txtfusion.projector.weight` — **with the file's SHA-256 pinned** (AC-47), so a re-download of a different version is caught rather than absorbed.
- **AC-47a — the comparison v1 assumed away.** Dump the workflow file's tensor and Fedor's and assert element-wise equality. The claim that they are "identical numerical effect" is Fedor's `__metadata__`, written by its author, and the two artifacts ship **conflicting strength guidance** (workflow: 1.0 at all times; Fedor: 3.0–5.0). If the tensors are equal, Fedor is a verified stand-in and the record says so. If they differ, Fedor leaves `krea2-reference` and becomes an O6 ablation row at its own strength.
- A shape-mismatch test: the same file against a hypothetical `[1, N≠12]` projector throws `partialApplication`, not a silent skip.
- `krea2Relative: nil` — it shape-matches both variants; the doubling identity is Raw-only and that is documented, not enforced.
- Strength policy: the preset default is **1.0** (the workflow author's figure, and the figure the reference recipe is defined by). The Fedor author's 3–5 is recorded in the preset description as a divergent recommendation, not adopted (§9, Q4).
- Provenance: it appears in `RenderRecipe.loras[]` as `{file, scale_applied, pairs_offered: 0, pairs_bound: 0, deltas_applied: 1}` — and `file` names **which artifact** applied, so a gallery image can always be traced to the workflow's bypass or to the substitute.

### 3.9 WP-E9 — VAE selection and the Wan key map

The Wan 2.1 FP32 file and the Qwen-Image VAE share **zero** key names (Draft B, verified: Wan-native `encoder.downsamples.N.residual.{0,2,3,6}` / `decoder.upsamples.0..14` vs diffusers `AutoencoderKLQwenImage` `up_blocks.N.resnets.M`). Draft B derived and validated the complete bijection — 194/194 mapped with exact shape equality, 194/194 Qwen keys covered, zero ambiguity. That is the risky part, already retired.

- New `Sources/ZImage/Krea2/VAE/Krea2VAEKeyMap.swift`: `VAELayout { qwenDiffusers, wanNative }`, `detectLayout(keys:)`, `canonicalize(_:) -> String?`. ~20 rules, per Draft B's table.
- **Detection is by key sniff, never by filename**: `decoder.upsamples.` ⇒ `.wanNative`; `decoder.up_blocks.` ⇒ `.qwenDiffusers`; neither ⇒ throw `unrecognizedVAELayout(file)`.
- `Krea2WeightLoader.loadVAE` (`:88-120`) takes `layout:` and canonicalizes before the existing 5-D slice / NHWC transpose / gamma flatten.
- **One `Krea2VAE` instance serves both `decode` and `encode`**, so encoder-side selection follows decoder-side automatically — PRD open question 5 answers itself provided we never introduce a second instance (AC-33 asserts identity).
- Selection precedence, each recorded: `payload.vae` → preset `vae` → model dir (`model_index.json` `"vae_file"`, else `vae/diffusion_pytorch_model.safetensors`). A named VAE that does not exist **fails the render**; it never falls back.
- **The VAE is reloaded in place, not keyed into the pool** (D17, corrected in v2). v1 added a `vae:` term to `ModelPool.poolKey` (`:490-498`) to stop a Wan request being served by the resident Qwen pipeline. Correct in effect, expensive in practice: a `.krea2` entry is estimated at 22,528 MB against a 40,960 MB budget (`:149-152`, `:246`), so two cannot co-reside and **every** VAE flip would evict and reload the whole 22.5 GB pipeline at a measured ~67 s — for a 508 MB decoder. Instead `Krea2Pipeline.ensureVAE(path:layout:)` swaps the decoder weights on the resident instance under the same fail-closed checks (missing file or unrecognised layout ⇒ the render fails and the resident decoder is untouched), bumps a reload counter, and updates `RenderRecipe.vae`. The guarantee is unchanged — a request for a VAE other than the resident one **always** reloads, and the record always names what decoded — and AC-59 asserts it on the recorded path plus the counter rather than on a pool eviction.
- `ImagePreset` gains `vae: String?` at **every site — five of them**, not the "four" v1 miscounted: the `PresetStore.swift` stored property, `CodingKeys:147-158`, the custom `init(from:):160-187`, the memberwise `init`, and `ResolvedPreset:229-282`. The `videoTuning` comment at `:153-157` records a field silently dropped from both the custom decoder and the synthesized encoder for months; AC-58 is its round-trip test, widened in v2 to cover all nine new fields.

### 3.10 WP-E10 — `RenderRecipe`, four sinks

```swift
public struct RenderRecipe: Codable, Sendable {   // snake_case on the wire, field `applied`
  // — physics: what was loaded —
  public let baseModel: String            // pool spec, e.g. "kroma-v0.2-turbo"
  public let baseVariant: String          // "raw" | "turbo"  ← from pipeline.variant
  public let baseModelFile: String        // paths.transformerFile.path — unambiguous
  public let quantization: String         // "q8" | "bf16"
  public let vae: String                  // pipeline.vaeSource path + layout label
  public let textEncoder: String          // "qwen3-vl-4b/bf16"
  public let loras: [Applied]             // read back from loadedLoRAConfigs + reports
  public let controlLora: Applied?

  // — geometry & seed —
  public let width: Int; public let height: Int    // POST round-up (Krea2Pipeline.swift:322-323)
  public let seed: UInt64

  // — the schedule grid (D3) —
  public let mu: Float
  public let shift: Float?
  public let shiftSource: String          // "dynamic" | "explicit"

  // — what actually ran, per stage (D4) —
  public let stages: [Stage]
  public let modelEvalsTotal: Int

  public struct Stage: Codable, Sendable {
    public let index: Int                 // 0-based
    public let sampler: String            // resolved SchedulerKind.rawValue
    public let sigmaSchedule: String      // resolved SigmaScheduleKind.rawValue
    public let sigmaScheduleRequested: String?  // D22: the raw name the caller sent when it
                                                // differed ("normal" → "flow"); nil when equal
    public let shiftApplied: Bool         // false for bong_tangent (D6)
    public let stepsRequested: Int
    public let stepsEffective: Int        // beta de-dup can shrink this (D5)
    public let stepsRun: Int              // img2img / stage-2 start mid-schedule
    public let modelEvals: Int
    public let denoise: Float
    public let guidance: Float
    public let negativePrompt: String?    // nil when guidance <= 1 — it did not apply
    public let eta: Float
    public let bongmath: Bool
    public let warmupSampler: String?     // "ralston_3s" for deis_3m below order
    public let warmupSteps: Int
    public let sigmaHead: [Float]         // first 3
    public let sigmaTail: [Float]         // last 3
    public let seed: UInt64
  }

  public struct Applied: Codable, Sendable {
    public let file: String
    public let scaleApplied: Float
    public let relativeTo: String?        // "raw" | "turbo" | null
    public let pairsOffered: Int
    public let pairsBound: Int
    public let shapeRejected: Int         // WP-E6: matched a module, failed normalize
    public let deltasApplied: Int         // D15: kroma-on-Raw records 0 here
    public let role: String?              // "kroma" | "accel" | "bypass" | "control" | null
  }
}
```

Every field is **read back**: `sampler`/`sigmaSchedule` from the resolved enums the factory received; `stepsRun`/`modelEvals` counted by the loop; `loras` from `Krea2Pipeline.loadedLoRAConfigs` (`:147`, only non-empty after a successful transactional apply at `:250`) joined with `LoRAApplicationReport`; `baseModelFile`/`vae` from `Krea2ModelPaths` and the resident decoder.

`Applied.role` is new in v2 and exists for one reason: the client has to report `kroma_strength` **as applied**, and matching a filename by string in the daemon would be a second copy of the family table waiting to drift (§3.17 WP-C4). The engine already knows which configuration slot each LoRA came from, so it labels it once, at the only place that knows the truth.

**Four sinks:**
1. `GenerateResponse` (`WarmServer.swift:7915-7936`, verified — currently `{success, outputPath, durationMs, preemptRefused, etaSec}`) gains `let applied: RenderRecipe?`, defaulted `nil` in the **explicit** init. That discipline is not optional: the comment at `:7922-7925` records that a property-level default drops the parameter from the synthesized memberwise init entirely. Carried through the preempt-refused re-stamp at `:645-647`.
2. **PNG metadata** — `QwenImageIO.ImageMetadata.generation` (`ImageIO.swift:194-225`) gains `applied: RenderRecipe? = nil`. The call at `WarmServer.swift:6868-6879` also starts passing `negativePrompt:`, which it does not today — a CFG render's negative prompt is currently absent from the file.
3. `/health` (`HealthResponse:8094-8121`) gains `lastRecipe: RenderRecipe?` and — separately, for D7 — populates `modelVariant` for the krea2 family (`:6163` sets it only for fibo/flux1/flux2 today; live probe returns `model_family: "krea2"` with no `model_variant`).
4. Async — `ImageJobStatus` (`:4661-4676`) gains `applied: RenderRecipe?` (Optional so persisted pre-upgrade JSON still decodes, same reasoning as `:4671-4673`), via `markSucceeded` (`:4807`) and `toStatus()` (`:4703-4708`).

Other families pass `nil` (D12).

### 3.11 WP-E11 / WP-E12 — `bong_tangent`, and ComfyUI-exact `beta`

**`bong_tangent`** (`res4lyf_sigmas.py:4065-4098`) is two arctan arcs joined at exactly 0.5, `steps+1` values ending at 0, with integer truncation (`int(steps·0.6)`) that makes it non-smooth in `steps`. **Port literally, do not clean up.** Wire name is snake_case `bong_tangent` (matching ComfyUI/RES4LYF, not the April plan's `bong-tangent`) so a value pasted out of the workflow JSON works verbatim. `resolveSigmas` ignores `config` and `mu` (D6). Pinned fixtures from Engine C's independent derivation, e.g. `steps=6` → `1.0, 0.928970, 0.797686, 0.5, 0.185601, 0.056802, 0.0`.

**`beta`** per D5:
```swift
static func discreteFlowSigmaTable(shift: Float, numTrainTimesteps: Int) -> [Float]
  // σ[i] = shift·t/(1+(shift−1)·t), t = (i+1)/T
static func beta(numSteps: Int, shift: Float, numTrainTimesteps: Int,
                 alpha: Float = 0.6, betaParam: Float = 0.6) -> [Float]
  // ts = rint(betaPPF(1 − i/steps, α, β) · (T−1)); dedupe consecutive; append 0
```
`betaPPF` is 60 iterations of bisection on the existing midpoint-rule CDF integrator (`SigmaSchedule.swift:134-149`) — the result is `rint`-ed to an integer index, so quantisation absorbs the error. `beta57` = `(0.5, 0.7)`. `SchedulerFactory` reads the produced count and constructs with the **actual** step count; `stepsEffective` reports it.

This WP also lands the `shift` request field (D3) and threads it into `resolveSigmas` for every model-consulting schedule.

### 3.12 WP-E13 / WP-E14 — Tableau samplers and DEIS multistep

**Tableaus** (linear frame, `h = σ' − σ`, `x_row = x₀ + h·Σⱼ Aᵢⱼ kⱼ`, `x' = x₀ + h·Σⱼ bⱼ kⱼ`), copied verbatim from `r4_rk_coeff.py` — **line labels corrected in v2**, v1 had them shifted: `ralston_2s` (`:1294`), `ralston_3s` (`:1241`), `ralston_4s` (`:1207`); the order ramp that swaps them in is `:1374-1393`. `res_3s` is the same driver in the exponential frame with `phi3` beside the existing `phi1`/`phi2` (`RES2sScheduler.swift:149-161`).

**DEIS.** The PRD names ComfyUI `comfy/k_diffusion/deis.py` as the port source; Engine C verified from the workflow JSON that both sampler nodes are **`ClownsharKSampler_Beta` (RES4LYF)**, not ComfyUI core, and that RES4LYF calls `get_deis_coeff_list(sigmas, order, deis_mode="rhoab")` — the **closed-form** branch (`r4_deis_coeff.py:86-121`), with no `edm2t`, no autograd, no 10,000-point quadrature. **This is easier than the PRD assumed** and it is what the workflow actually runs.

Two behaviours that must be reproduced or the recipe is not the recipe:
- **Order ramp with a ralston warm-up.** `r4_rk_coeff.py:1374-1393`: while `step < order`, the sampler is swapped to `ralston_{order}s`. At the published stage-2 settings (`deis_3m`, 2 steps) **both steps run `ralston_3s`** — 6 forward passes, and true DEIS coefficients never engage. `warmupSampler`/`warmupSteps` in the record exist so nobody rediscovers this.
- **Coefficients in `Double`.** `rhoab` order-3/4 coefficients are differences of cubes and quartics of sigmas that get close together at the schedule tail (0.117 → 0.043 → 0). Compute in `Double`, narrow to `Float` only at use, with a catastrophic-cancellation guard test.

### 3.13 WP-E15 / WP-E16 — `eta` SDE (T2) and `bongmath` (T3)

**T2.** `eta = 0.5` is the workflow's value and the node default; with `noise_mode_sde = "hard"` on a variance-preserving flow model — hard mode is `eta_ratio = eta` at `r4_noise.py:444-445`, feeding the VP split at `:279-293` (citation corrected in v2):
```
σ_up = eta·σ';  σ_res = sqrt(σ'² − σ_up²);  alpha = (1 − σ') + σ_res;  σ_down = σ_res/alpha
x' = alpha · integrate(x, σ → σ_down) + σ_up · noise
```
A per-step re-noise, not decoration. The noise stream is **declared per stage** and deterministic — a stage's stream is seeded from that stage's seed, and two identical payloads produce byte-identical output (AC-27).

**T3.** `bongmath = true` is the node default and runs `RK_Method_Beta.bong_iter` (`r4_rkm.py:655-770`): a 100-iteration fixed point re-deriving `x₀` and every substep row from already-computed rows — **no extra model calls** — gated on `multistep_stages == 0` (`:713`), i.e. inert on true DEIS multistep steps and active on every ralston warm-up step. At the published settings that is *all* of stage 2. Highest uncertainty per line in the programme; it gets its own tier, its own oracle trace, and D19's gate.

### 3.14 WP-E17 — Two stages inside one render (O5)

**Answers PRD open question 4 and 7: one render, and it must be one render.** A client-composed version would round-trip through the VAE (lossy; the reference does not do it), re-tokenise the prompt, and could not express "re-noise the *latent* to σ ≈ 0.117".

Request shape (D4), additive — a payload without `stage2` is byte-identical to today:
```jsonc
{
  // D25: the sampler's wire key is `scheduler` — what GeneratePayload decodes
  // (WarmServer.swift:7511,7626,7730) and what MCPToolExecutor + the tile
  // pipeline already post. `sampler` is accepted as an alias; both present and
  // different is a 400. The RESPONSE reports it as applied.stages[].sampler.
  "scheduler": "res_2s", "sigma_schedule": "beta", "shift": 1.15,
  "steps": 6, "guidance": 1.0, "eta": 0.5, "bongmath": true,
  "stage2": {
    "scheduler": "deis_3m", "sigma_schedule": "bong_tangent",
    "steps": 2, "denoise": 0.2,     // denoise decodes as Double (see below)
    "guidance": 1.0, "eta": 0.5, "bongmath": true,
    "seed": null          // null → stage1 seed &+ 1, recorded either way
  }
}
```

`Krea2Pipeline.generateStaged(_:)` (new file `Krea2StagedGeneration.swift`):
1. Stage 1 runs the loop and stops with the **patchified latent** at σ = 0.
2. **Stage-2 sigmas are a stretch-and-tail, not a truncation.** RES4LYF's `get_sigmas` (`res4lyf_sigmas.py:1397-1429`, the truncation at `:1402`) does `total = int(steps/denoise)` → build at `total` → take `sigmas[-(steps+1):]`. So stage 2 runs on `bong_tangent(10)[-3:]` = `[0.117461, 0.043265, 0.0]` — it starts at σ ≈ 0.117, **not** 0.2. `denoise ≤ 0` is a hard error.

   **The arithmetic type is load-bearing** (v1 left it unstated; every other float on `GeneratePayload` decodes as `Float`). Python's `int(steps/denoise)` truncates a *double*: `2/0.2` is `10.000000000000002` → 10, and a `Float` division of the same pair can land on the other side of the integer and silently select a different tail. **`stage2.denoise` decodes as `Double`**, and `total = Int((Double(steps) / denoise).rounded(.towardZero))`. AC-31 is parametrised over `{steps 2…8} × {denoise 0.1…0.9}` against values dumped from the Python source, so a divergence is a test failure rather than a different picture.
3. Re-noise: `MLXRandom.seed(stage2Seed)`, `ε ~ N(0,1)` at the stage-1 NCHW shape, `x₂ = σ₂[0]·ε + (1 − σ₂[0])·x₁` — the identical rectified-flow mix already at `Krea2ImageToImagePipeline.swift:124`, applied to the latent with **no VAE round-trip**.
4. Stage 2 runs the same loop with its own scheduler, guidance, eta, bongmath.
5. **One** `vae.decode` at the end.

`Krea2ImageToImagePipeline` is left alone: its `strength → startIndex` rule (`:112-116`) is the established img2img contract on `/v1/generate`'s `image_path` path and changing it would move every existing img2img render. Stage 2 is a distinct, differently-specified mechanism; AC-30 pins that they stay distinguishable.

**Distinguishable from the premium polish pass** (PRD O5 acceptance, verbatim): the polish is img2img on a *different checkpoint* at image strength 0.8, composed from two HTTP calls (`client.ts:1159-1290`). Stage 2 is one call, one checkpoint, one LoRA stack, no VAE round-trip, and appears in the record as `stages[1]`. The client records the polish under a separate `polish` key (WP-C4). A record with `stages.length == 2` and a record with `polish` set can never be confused.

### 3.15 WP-E19 / WP-E20 / WP-E21 — Bridge, preset schema, reference preset

**WP-E19:** the bridge Raw branch (D13), §3.5.

**WP-E20:** `ImagePreset` (`PresetStore.swift:61-190`) gains — at **all five** sites (stored property, `CodingKeys`, custom `init(from:)`, memberwise `init`, `ResolvedPreset`), per the `videoTuning` lesson; v1 said "four" and listed five:
```swift
public var checkpointFamily: String?            // "turbo" | "raw-accel" | "raw-stock" | "zimage-turbo" | "zimage-base"
public var kroma: KromaPolicy?                  // { strength: Double, file: String? }
public var vae: String?
public var sampler: String?
public var sigmaSchedule: String?
public var shift: Double?
public var eta: Double?
public var bongmath: Bool?
public var stage2: PresetStage?
```
`ImagePreset` already carries `steps` (`PresetStore.swift:83`), `guidance` (`:87`) and `scheduler` (`:94`) — the daemon simply never reads them (`parsePreset`, `client.ts:711-729`, maps only `id`/`loras`/`negativePrompt`/`model`). WP-C1 fixes the reading half, and §7.5 records what that changes for the live lanes.

**Engine-side validation — O4a is not a daemon-only rule** (added in v2). PRD O4a says "every image preset declares its kroma strength explicitly… an absent value is a configuration error." v1 enforced that only in `validateImagePreset` on the client, which leaves the desktop app, the Krita bridge and MCP — all of which read this same engine preset store — accepting a Krea 2 image preset with no `kroma` and rendering it silently. So `PresetStore` validates too:

- **on save** (`PUT`/`POST /v1/presets`): an image preset whose `model` resolves to a krea2 family and declares no `kroma` is a **400 naming the preset and the field**. `PresetStoreTests.testKrea2ImagePresetRequiresKroma`.
- **on load**: the same preset is logged as an error and flagged `invalid: true` in `GET /v1/presets`, so nothing downstream can select it. `WarmServerPresetValidationTests`.
- non-krea2 image presets (the five `imported-cs-*` Z-Image entries) are unaffected — the requirement is scoped by family, exactly as on the client (D14).

**WP-E21 — `krea2-reference` (derives PRD open question 9 as O4b).** The PRD asks whether "the reference recipe exists as one selectable preset" should be its own outcome. It should, and it is the FDD's job to derive it rather than leave it implicit:

```jsonc
{
  "id": "krea2-reference",
  "name": "Krea 2 Reference (published recipe)",
  "model": "krea2-raw",
  "checkpointFamily": "raw-accel",
  "kroma": { "strength": 0 },
  "loras": [
    { "filename": "krea2_turbo_lora_rank_64_bf16.safetensors", "scale": 0.6 },
    // The workflow's own bypass file. NOT krea2_filter_bypass_fedor — v1 named
    // the substitute here and called the axis satisfied (D10, §0.2 F1). WP-E8
    // acquires this file; until it exists the preset cannot be created, which
    // D19's T3 gate already enforces.
    { "filename": "krea2filterbypass_2vector.safetensors",     "scale": 1.0 }
  ],
  // D16, reversed in v2: the reference stack's decoder IS Wan 2.1 FP32, and this
  // preset declares kroma 0 so there is no kroma x VAE confound to defer for.
  // The existing turbo lanes keep the model-dir Qwen-Image VAE (no regression).
  "vae": "/Users/toddwalderman/LocalModels/vae/Wan2_1_VAE_fp32.safetensors",
  "sampler": "res_2s", "sigmaSchedule": "beta", "shift": 1.15,
  "steps": 6, "guidance": 1.0, "eta": 0.5, "bongmath": true,
  "stage2": { "sampler": "deis_3m", "sigmaSchedule": "bong_tangent",
              "steps": 2, "denoise": 0.2, "eta": 0.5, "bongmath": true }
}
```
Created when T3 lands **and** the workflow's bypass file is on disk (D19, D10). Every default visible in the record. If AC-47a shows Fedor is tensor-identical to the workflow file, Fedor may substitute for it **and the record says which one applied** (`RenderRecipe.loras[].file`); if it is not identical, Fedor is an O6 ablation row at its own 3–5 strength and never appears in this preset.

### 3.16 WP-C1 — Checkpoint-family policy: the pin and the clamp go

Verified today, unchanged from the client draft: `resolveGenerationParams` (`client.ts:544-550`) pins guidance to `DEFAULT_GUIDANCE = 0` and computes `Math.min(rawSteps, TURBO_STEPS=9)`. `this.modeGuidance` is assigned at `:484` and read nowhere. `LANE_MAP` (`image-lane-map.ts:9-13`) is `sketch:4 / render:9 / quality:30` — and `quality: 30` survives today only because the clamp swallows it.

New pure module `src/providers/comfybox/checkpoint-policy.ts`:

```ts
export type CheckpointFamily = 'turbo' | 'raw-accel' | 'raw-stock';
export type LaneIntent = 'draft' | 'daily' | 'max';

export interface FamilyPolicy {
  requiresVariant: 'turbo' | 'raw';              // D7 consistency check
  steps:    { draft: number; daily: number; max: number; allowed: [number, number] };
  guidance: { default: number; allowed: [number, number] };
  sampler?: { name: string; sigmaSchedule: string; shift?: number; eta?: number };
  stage2?:  { steps: number; denoise: number; sampler: string; sigmaSchedule: string; eta?: number };
  requiresAccelLora?: RegExp;
  kromaBaked?: boolean;                          // R4 — see §8
  kromaDefaultFile: string;
}

export type RefusalCode =
  | 'unknown-family' | 'steps-out-of-range' | 'guidance-out-of-range'
  | 'preset-invalid' | 'family-mismatch' | 'accel-lora-missing';

export type RecipeResolution =
  | { ok: true;  recipe: ResolvedRecipe }
  | { ok: false; code: RefusalCode; reason: string };

export function resolveRecipe(input: RecipeInput): RecipeResolution;
export function validateImagePreset(p: ComfyBoxPreset): PresetValidation;
```

| family | requiresVariant | steps draft/daily/max (allowed) | guidance default (allowed) | stage 1 | stage 2 |
|---|---|---|---|---|---|
| `turbo` | krea2 `turbo` | 4 / **9** / 12 (1–16) | **0** (0–1.5) | engine default | none |
| `raw-accel` | krea2 `raw` | 4 / 6 / 10 (2–20) | 1.0 (0–2.5) | `res_2s` + `beta`, shift 1.15, eta 0.5 | 2 @ 0.2, `deis_3m` + `bong_tangent` |
| `raw-stock` | krea2 `raw` | — / 52 / 60 (20–80) | 3.5 (1–7) | engine default | none |
| `zimage-turbo` | z-image `turbo` | 4 / 9 / 16 (1–20) | 3.5 (0–7) | engine default | none |
| `zimage-base` | z-image `base` | 12 / 30 / 40 (8–60) | 4.0 (1–8) | engine default | none |

`turbo` daily 9 and guidance 0 are **exactly today's values** — that is the no-regression contract (AC-35). `raw-accel` comes from the workflow JSON; `raw-stock` from krea-ai/krea-2 README line 50.

**The two `zimage-*` rows are new in v2 and are not scope creep — they are what keeps the daemon running.** The live store holds five Z-Image image presets (`imported-cs-*` on `Tongyi-MAI/Z-Image-Turbo-BF16` / `z-image-turbo-bf16`, 9–16 steps, guidance 3.5–5.0) and `~/.kira/config.json` names a Z-Image checkpoint (`cyberrealisticZImage_v50.safetensors`) as both `baseCheckpoint` and `polishCheckpoint`. Under v1's rule — "a named checkpoint nobody has classified is `unknown-family` and refuses" — every one of those refuses on deploy day. Their ranges are today's observed values, deliberately wide, and `kroma` is not required of them (D14).

**The premium lane's base, stated** (v1 retired the global `baseCheckpoint` habit without saying what replaced it): the premium lane takes the **preset `model`, exactly as the standard lane does** — that is what O4a means by the global habit not carrying over. `config.comfybox.baseCheckpoint` and `polishCheckpoint` are **ignored, with a startup warning naming both keys and the file they point at**, so the operator sees the change rather than discovering it. (The polish pass they feed is already skipped for Krea 2 presets — `client.ts:198` — so for Bree's and Kira's lanes this removes a value that was already inert on the image path.) AC-37a asserts the premium-lane resolution on Kira's live config by name.

**Resolution is a ladder that never reduces a value.** Precedence per field: explicit request > preset > `modeSteps`/`modeGuidance` > family default for the lane intent. Then range-check. Out of range ⇒ `{ok:false}` naming the family, the value, the range and the **source** ("steps 30 from lane 'quality' exceeds turbo range 1–16"). **Never `Math.min`.** `generate()` turns a refusal into the existing fail-closed `{success:false, error}` shape.

**Family resolution, three sources, explicit precedence, no inference:** (1) the preset declares `checkpointFamily`; (2) a static `CHECKPOINT_FAMILY_TABLE` resolves a bare model id — seeded with every id in the live store (`krea2`, `kroma-v0.2-turbo`, `krea2-raw`, `Tongyi-MAI/Z-Image-Turbo-BF16`, `z-image-turbo-bf16`, `cyberrealisticZImage_v50.safetensors`) — with a one-line warning naming the preset, because it is a transition affordance; (3) the engine verifies via `/health.model_variant` or `applied.base_variant` — contradiction ⇒ `family-mismatch`, absence ⇒ `family_verified: false`, never silent trust. A **named** checkpoint nobody has classified is `unknown-family` and refuses; **no model named at all** resolves as `turbo`, which is the documented status quo.

**The image-preset discriminator, defined** (v1 said "image preset" without one, and `mediaKind` is `nil` on all eight Krea entries in the live store): a preset is an image preset when `mediaKind === 'image'` **or** `mediaKind` is absent and `engine === 'zimage'`. The WP-C6 migration sets `mediaKind: 'image'` on all eight so the fallback stops being load-bearing.

`LANE_MAP` becomes `{intent, tier}`; `resolveLane` returns intent; the family maps intent→steps. That is what makes "quality lane on a turbo preset" resolve to 12 rather than refuse at 30.

`preflightRecipePolicy()` runs at daemon startup after the first preset fetch, resolving every lane × content mode × preset. v1 had it log **refusals**. In v2 it emits **two** things, because a refusal is not the only way a deploy surprises someone:

1. **Refusals** — each combination that would fail, naming mode, preset, value, range and the daemon home it read. `modeSteps: { avocado: 20 }` on a turbo preset is exactly such a case and must surface in a startup log, not as a failed render at 2am.
2. **A resolution-diff table** — every combination whose resolved `steps` or `guidance` **changes** versus the pre-C1 resolver (`resolveGenerationParams` on `34cb3244`), with both values side by side. This is committed as a fixture in the C1 PR and reviewed before merge (AC-38a). It exists because the ladder's `preset > modeSteps` precedence changes live behaviour that no refusal would catch — §7.5 lists what it catches today.

### 3.17 WP-C2 / WP-C3 / WP-C4 — Wire, kroma policy, read-back

**WP-C2 (wire).** `ComfyBoxGenerateOptions` gains `sampler`, `sigmaSchedule`, `shift`, `eta`, `bongmath`, `stage2`, `vae`, `kromaStrength`, `lane`. `generate()` emits snake_case **only when set**, so a request naming nothing is byte-identical to today's body (AC-36):

```
scheduler        ← opts.sampler        // D25: the sampler's WIRE KEY is `scheduler`.
sigma_schedule   shift   eta   bongmath   vae
stage2: { steps, denoise, scheduler, sigma_schedule, eta, bongmath }
```

v1 wrote `sampler` on the wire. The engine decodes the sampler as `scheduler` (`WarmServer.swift:7511,7626,7730`) and both live senders already post that spelling, so emitting `sampler` would have had the key **ignored** and every client sampler request would have rendered euler. `renderViaQueue` already spreads `body` into `asyncBody` (`client.ts:995`), so the async transport inherits them; a test pins that. `generatePremium` loses `resolved.steps ?? 30` and `resolved.guidance ?? DEFAULT_GUIDANCE` at `:1227-1228`/`:1276-1277` in favour of the `max`-intent recipe on the **preset's** checkpoint (§3.16).

**The operator surface, per field** (v1 added the options and never said who could reach them — O5/O7/O4a all say "an operator can request"). Added to the `generate_image` tool schema beside the existing `steps`/`lane`/`tier`/`dype` parameters:

| field | surface | why |
|---|---|---|
| `sampler`, `sigma_schedule` | **tool schema** (enum, from the engine's `/object_info`) | O1's acceptance is an operator asking for a sampler |
| `kroma_strength` | **tool schema** (number 0–1.5) | O4a: "an operator can override per render" |
| `stage2` | **tool schema** as a boolean `detail_pass` + optional `detail_denoise` | O5: "an operator can request a detail pass at a given denoise"; the sampler pairing comes from the family policy |
| `vae` | **preset-only** | O7 is a decoder decision, not a per-render whim; the preset is where it is auditable |
| `shift`, `eta`, `bongmath` | **preset-only + CLI** (`scripts/image-compare.mjs`) | parity-tuning knobs; exposing them to an LLM caller invites drift with no outcome behind it |

`client.recipe-wire.test.ts` pins that each exposed parameter reaches the request body unchanged (AC-68a).

**WP-C3 (kroma, D14).** `ComfyBoxPreset` gains `checkpointFamily`, `kroma`, `steps`, `guidance`, `scheduler`. `resolveActiveLoraSet` (`client.ts:565-583`) compiles `kroma.strength > 0` into a prepended `{path, scale}`; `=== 0` contributes nothing. `opts.kromaStrength` overrides, range-checked `[0, 1.5]`. `parsePreset` stays tolerant (video presets need it); `validateImagePreset` runs in `getPreset` (`:686`) and caches, logging once at fetch and refusing at render.

**Migration of the live store — all thirteen image presets.** v1 named three and would have refused the other ten on the first render after deploy. Read from `~/.comfybox/presets.json` this session (24 entries; 11 are video and are untouched):

| preset | model | today's `loras[]` | → `checkpointFamily` | → `kroma` | other |
|---|---|---|---|---|---|
| `krea-kira` | `krea2` | `kroma-lora-v0.3 @ 0.6` | `turbo` | `{strength: 0.6}` | kroma entry leaves `loras[]`; **steps 12 now applies — see §7.5** |
| `krea-kira-hq` | `krea2` | 3 character/style LoRAs | `turbo` | `{strength: 0}` | LoRAs unchanged |
| `krea-kira-sfw` | `krea2` | `SEAsian_Women_Krea-2 @ 0.8` | `turbo` | `{strength: 0}` | LoRAs unchanged |
| `krea-film-apple` | `krea2` | `kroma-v0.1 @ 1.0` + `Filipina_Pinay_Women @ 0.6` | `turbo` | `{strength: 1.0, file: "kroma-v0.1.safetensors"}` | kroma entry leaves `loras[]`; steps 8 now applies |
| `krea-film-banana` | `krea2` | as above + `krea2_innie_vagina @ 0.4` | `turbo` | `{strength: 1.0, file: "kroma-v0.1.safetensors"}` | as above |
| `krea-film-avocado` | `krea2` | as `banana` | `turbo` | `{strength: 1.0, file: "kroma-v0.1.safetensors"}` | as above |
| `krea2-base` | `kroma-v0.2-turbo` | — | `turbo` | `{strength: 0}` on `kromaBaked: true` | records `kroma_strength: "baked"` |
| `krea-bree` | `kroma-v0.2-turbo` | — | `turbo` | `{strength: 0}` on `kromaBaked: true` | records `kroma_strength: "baked"` |
| `imported-cs-neutral` | `Tongyi-MAI/Z-Image-Turbo-BF16` | — | `zimage-turbo` | **not required** (D14) | 9 steps / g 3.5 in range |
| `imported-cs-sensual` | `Tongyi-MAI/Z-Image-Turbo-BF16` | 2 character LoRAs | `zimage-turbo` | not required | 12 / 3.5 |
| `imported-cs-control` | `z-image-turbo-bf16` | — | `zimage-turbo` | not required | 12 / 5.0 |
| `imported-cs-nsfw` | `Tongyi-MAI/Z-Image-Turbo-BF16` | 2 character LoRAs | `zimage-turbo` | not required | 16 / 3.5 |
| `imported-cs-vector` | `Tongyi-MAI/Z-Image-Turbo-BF16` | — | `zimage-turbo` | not required | 16 / 4.0 |

All eight Krea entries also gain `mediaKind: 'image'` (they carry `null` today, which is why §3.16 needs a discriminator at all). `kroma-v0.1.safetensors` is seeded `.turbo`-relative in WP-E6, without which the three `krea-film-*` rows cannot be validated. **AC-44a** loads a checked-in copy of the migrated file and asserts `validateImagePreset` passes for every image entry and `preflightRecipePolicy()` reports **zero** refusals — the whole-store check v1 had no equivalent of.

`krea-bree`/`krea2-base` run `kroma-v0.2-turbo`, where kroma is **baked into the checkpoint**: `kroma: {strength: 0}` on a family carrying `kromaBaked: true` surfaces as `kroma_strength: "baked"` in the record (§8, R4 — the client draft's suggested alternative, adopted, because `0` on a baked checkpoint is honest about the LoRA and misleading about the image).

**WP-C4 (read-back).** `RenderSettingsInput`/`RenderSettings` (`src/image/render-settings.ts:24-66`) gain `sampler`, `sigma_schedule`, `shift`, `eta`, `bongmath`, `vae`, `checkpoint_family`, `model_variant`, `kroma_strength`, `stages[]`, `polish{checkpoint,image_strength}`, and three honesty fields: `provenance: 'engine'|'request'`, `substituted?: string[]`, `family_verified?: boolean`.

Rules: `applied` present ⇒ `provenance:'engine'` and every field comes from it; any field differing from the request is listed in `substituted[]` and logged at warn (**the engine is authoritative about what ran; it is never authoritative about staying quiet**); `applied` absent ⇒ `provenance:'request'` and today's echo, but a reader can now tell.

**`kroma_strength` is derived, not echoed** (added in v2 — v1 had it read the *requested* policy value, which is precisely the echo class D8/R5 and PRD O4a forbid: "the reported settings name the kroma strength that **actually applied**"). The rule:

```
applied present:  entry = applied.loras.find(l => l.role === 'kroma')          // engine-labelled, §3.10
                  kroma_strength = entry ? entry.scale_applied : 0             // absent ⇒ genuinely 0
                  requested !== resolved ⇒ substituted[] + warn
applied absent:   kroma_strength = requested,  provenance: 'request'
kromaBaked family: "baked" (R4), regardless of the above
```

Matching on `role` rather than on a filename keeps the family→file table in one repo. AC-45 asserts it against a fixture produced by the **real engine serializer**, not a hand-written string (audit #1653 P0-2 found a vacuous fixture doing exactly that).

Two existing echoes are removed in the same PR (audit #1653 P2, both still at the cited lines): `image-gen-tools.ts:869` `?? config.comfybox?.baseCheckpoint` — `ensureModelForPreset` returns `null` when activation was *rejected* and the render proceeded on whatever was resident (`client.ts:653-663`); collapsing that into a confident config string is the same defect class as #1639. And `sidecarLoraProvenance` (`client.ts:309-318`) `?? fallbackDefaultLoras` at `:314`, which substitutes the configured default for a deliberate `undefined` meaning "honestly unknown"; the parameter is deleted.

Sealed branch: strips `model`, `preset`, `loras`, `model_variant` (it identifies the checkpoint); keeps `provenance`, `steps`, `guidance`, `sampler`, `sigma_schedule`, `loras_applied` (non-identifying replication params).

### 3.18 WP-C5 — The O6 comparison

Split so the judgement is testable and the I/O is thin.

`src/image/recipe-compare.ts` (pure): `assertNeutralPrompt(prompt): string[]` (PRD §4 requires no age words and no style words — a prompt containing them fails the run rather than producing a comparison nobody can trust); `planComparison(spec): ComparisonPlan` (one seed × one prompt × N named recipes, each recipe *is* a request); `blindCaptionOrder(items, rng)` (the captioner sees an order uncorrelated with the recipe list; the mapping is held out of the caption call); `buildComparisonReport(plan, results, captions)` → markdown with a settings table (one row per recipe, columns from `RenderSettings` including `provenance`), the images, the blind captions, and a **recipes-that-refused** section — a refusal is a result.

`scripts/image-compare.mjs` (CLI): drives the plan serially, captions via `callVision` (`src/vision/vision-client.ts:19`) with `resolveVisionEndpoint` (`src/vision/vision-endpoint.ts:58`) — the same path `look` uses, so a comparison cannot silently run on a second vision stack (audit P1) — checkpoints per recipe so a mid-run failure does not discard completed renders, and **orders recipes to batch by base** (D17).

**CLI first, MCP second.** An N-recipe sweep at raw-accel step counts with CFG exceeds `generate_image`'s 600 s tool timeout and the 180 s ThreadMailbox cap (`image-gen-tools.ts:449-458`). An MCP `compare_recipes` that enqueues and returns a report path is a follow-up.

---

## 4. Technical acceptance criteria

Each is numbered, mapped to a PRD outcome, testable, and names the test that proves it. "Production config" means the deployed 8-bit transformer (`ModelPool.swift:562`), 1024×1024, and the real files on disk — never a convenient setting.

### Regression gates (the merge conditions)

| # | Outcome | Criterion | Test |
|---|---|---|---|
| 1 | O1 | **Default Krea 2 t2i is byte-identical.** kroma-v0.2 q8, 1024², 9 steps, guidance 1.0, seed 44821, no LoRAs, no sampler field → PNG SHA-256 equals a fixture hash captured from `296735d` before the refactor. | `Krea2SamplerParityTests.testDefaultT2IByteIdentical` (integration) |
| 2 | O1 | **Default img2img is byte-identical** under the same protocol at `strength 0.3`. | `Krea2SamplerParityTests.testDefaultImg2ImgByteIdentical` |
| 3 | O1 | **Schedule equivalence.** `SigmaSchedule.krea2(numSteps:mu:)` equals the pre-change `Krea2Sampling.timesteps` element-for-element (exact `==` on Float) for `{4,6,9,12,20,52} × {256,1024,4096,6400,9216}`, including the trailing exact `0.0` and count `steps+1`. | `Krea2SigmaScheduleTests.testMatchesPreChangeOracle` |
| 4 | O1 | **mu equivalence.** `Krea2Sampling.mu(seqLen:align:16)` `==` `PipelineUtilities.calculateShift(…, 256, 6400, 0.5, 1.15)` exactly, over the same seqLen set. | `Krea2SigmaScheduleTests.testMuMatchesCalculateShift` |
| 5 | O4 | **No turbo regression on the live dir.** A fixed-seed `/v1/generate` on `~/LocalModels/kroma-v0.2` at 9 steps / guidance 1 is byte-identical before and after the whole programme. | `Krea2SamplerParityTests.testLiveTurboDirUnchanged` |
| 5a | O2, O3 | **The bridge splits by variant.** On `.turbo` the constructed `GeneratePayload` is field-for-field equal to the one `296735d` builds for a fixed `ComfyBridgeGenerateRequest` (steps clamp 12, guidance 0, negative dropped). On `.raw` the same request yields `steps == request.steps` (no clamp), `guidance == request.guidance`, `negativePrompt` preserved; with steps/guidance absent it yields 30 / 1.0; an unknown sampler name throws rather than becoming euler. | `BridgeKrea2VariantTests` |
| 5b | O2 | **Variant defaults reach the generate path.** `runKrea2Generate` on a `.raw` pipeline with no `steps` and no `guidance` uses 30 / 1.0 (**not** 3.5 — §3.5), and `applied.stages[0]` reports both. On `.turbo`, 9 / 1.0, unchanged. | `Krea2VariantDefaultsTests` |
| 6 | O1 | **Z-Image default path untouched.** Full `ZImageTests` + `ZImageIntegrationTests` pass, and a Z-Image Turbo render at defaults (euler/flow) is byte-identical pre/post. | existing suites + `ZImageTurboParityTests` |
| 7 | O2 | `resolveRecipe({family:'turbo'})` with no steps/mode/preset returns `steps: 9, guidance: 0` — byte-identical to `resolveGenerationParams` on `34cb3244`. | `checkpoint-policy.test.ts` + surviving `client.mode-resolution.test.ts` |
| 8 | O1 | `generate()` with no recipe fields produces a request body deep-equal to the body produced on `34cb3244` for the same options, on both transports. | `client.recipe-wire.test.ts` |

*Verified by source, not predicted (corrected in v2).* v1 flagged criterion 1 as resting on an unchecked MLX promotion assumption. Read this session: `MLXArray+Ops.swift:253-255` defines `Float * MLXArray` as `lhs.asMLXArray(dtype: rhs.dtype) * rhs`, so today's `(tp - tc) * v` computes the difference in Swift **float32** and rounds once to bf16 — identical to `FlowMatchEulerScheduler.step`'s `(sigmas[i+1] - sigmas[i]).asType(sample.dtype)` over a float32 `sigmas` array (`FlowMatchEulerScheduler.swift:39,76`). **AC-1 holds by construction.** The pre-planned remedy is retained in case a future MLX release changes the promotion rule, and a "close enough" result is still not acceptable.

*The trap is the other direction, and it is real:* AC-2. See §3.3 — the img2img mix must keep `scheduler.sigmas[startIndex]` as a **float32 `MLXArray`**; taking `.item(Float.self)` would move the whole mix into bf16 and fail AC-2. That sentence is part of AC-2's test description.

### O1 — sampler and schedule are requestable and honoured

| # | Criterion | Test |
|---|---|---|
| 9 | Two renders, one seed, one prompt, differing only in the sampler (`euler` vs `res_2s`) produce a non-zero max-abs pixel difference, and each `applied.stages[0].sampler` names the one requested. *(PRD O1 acceptance, verbatim.)* | `Krea2SamplerParityTests.testSamplerChangesOutput` |
| 9a | **The premium lane is materially different from the daily lane** *(PRD O2 acceptance line 1, which v1 had no criterion for)*. At production config on `raw-accel`, one seed, one prompt: the `max` lane and the `daily` lane produce a non-zero pixel difference, `applied.stages[0].steps_run` is 10 vs 6, and both records carry `provenance: 'engine'`. The same pair on `turbo` (12 vs 9) is rendered and recorded in the no-regression table. | `Krea2LaneRenderTests` (integration, §5.3 Raw batch) |
| 10 | **RES2s receives x₀, not velocity.** With `evaluate` returning the exact rectified-flow velocity of a fixed `(x₀, ε)`, `res_2s` over the krea2 schedule reconstructs `x₀` to ≤1e-5 relative; feeding velocity instead (pre-change behaviour) fails the same assertion by >1.0. | `ModelOutputConventionTests.testRES2sReconstructsX0` |
| 11 | Every `SchedulerKind` reports a convention; only `res_2s`/`res_3s` are `.dataPrediction`. | `ModelOutputConventionTests.testConventionTable` |
| 12 | **CFG works under every sampler and its cost is reported.** For each `SchedulerKind` at `guidance 2.0`, the loop completes and `applied.stages[0].model_evals == steps_run × rows × 2`. | `Krea2DenoiseLoopTests.testCFGEvalCount` |
| 13 | **Multistep state does not leak.** Two consecutive `dpmpp_2m` renders at one seed are byte-identical. | `Krea2DenoiseLoopTests.testResetBetweenRuns` |
| 14 | Non-flow schedules stay in the flow domain: for `karras/exponential/beta/beta57` under `Krea2Sampling.schedulerConfig`, every sigma ∈ [0,1] and `sigmas[0] == 1.0`. | `Krea2SigmaScheduleTests.testNoEDMLeakage` |

### O8 — fail loud, and the reference pairing exists

| # | Criterion | Test |
|---|---|---|
| 15 | `POST /v1/generate {"scheduler":"uni_pc"}` returns **400** naming `uni_pc` and listing valid samplers; `successful_render_count` unchanged; no file written. Same for `/v1/generate/async`, and same for `{"sampler":"uni_pc"}` via the D25 alias. | `SamplerNameResolutionTests` + `WarmServerRejectionTests` |
| 15a | **The sampler key cannot be silently ignored** (D25). `{"sampler":"res_2s"}` decodes to `SchedulerKind.res2s` and the render reports `applied.stages[0].sampler == "res_2s"` — it can never produce euler. `{"scheduler":"res_2s","sampler":"euler"}` is a **400** (`mutuallyExclusive`); `{"scheduler":"res_2s","sampler":"res_2s"}` succeeds. | `GeneratePayloadDecodeTests.testSamplerKeyAlias` |
| 16 | Aliases survive: `res_2s`, `dpmpp_2m`, `dpmpp_2s_ancestral`, `beta57`, the RES4LYF UI prefixes `exponential/res_2s` / `multistep/deis_3m`, **and the four ComfyUI schedule names `normal`, `simple`, `sgm_uniform`, `ddim_uniform` → `.flow`** (D22, reversed from v1). Each aliased request records `sigma_schedule: "flow"` **and** `sigma_schedule_requested: "<raw>"`. | `SamplerNameResolutionTests.testAliasTable` |
| 16a | **Every Krita default renders.** For each value in Krita AI Diffusion's `style.py` `_scheduler_map` (`normal`, `karras`, `ddim_uniform`, `sgm_uniform`) and `_sampler_map` (`euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_2m_sde_gpu`, `dpmpp_sde_gpu`, `uni_pc_bh2`, `lcm`), a bridge request either resolves **or** returns a 400 that names the value — and every name our own `/object_info` advertises resolves. The default style (`Euler` → sampler `euler`, scheduler `normal`) renders end to end on both variants. | `BridgeKrea2VariantTests.testKritaStyleMatrix` |
| 17 | **No phantom names anywhere.** `GET /object_info` contains no `uni_pc`/`dpmpp_2m_sde`, every string in every `sampler_name` list (`:302,333,353`) **and every `scheduler` list** (`:293,334,354`) resolves without throwing, and the union of the advertised scheduler lists equals `SigmaScheduleKind.allCases` ∪ the declared aliases. The MCP `generate_image` schema's enums equal `SchedulerKind.allCases`/`SigmaScheduleKind.allCases` (it advertises `linear` today, which does not exist). | `ComfyBridgeObjectInfoSamplerTests` + `MCPGenerateSchemaTests` |
| 18 | A persisted queue job that fails replay validation is marked **failed with the reason recorded**, never rendered and never silently dropped. | `PersistedQueueRecoveryTests.testInvalidRecipeJobFailsLoud` |
| 19 | **`bong_tangent` matches upstream exactly** for `steps ∈ {2,6,8,9,10,12,20}` to 1e-6: `steps+1` elements, starts 1.0, exactly 0.5 at the join, ends 0.0. Fixture values are **verified against upstream source** (`res4lyf_sigmas.py:4065-4098`, pure-Python re-run 2026-08-22) — `steps=6` → `[1.0, 0.928970, 0.797686, 0.5, 0.185601, 0.056802, 0.0]` — not an independent derivation awaiting confirmation. | `BongTangentScheduleTests` |
| 20 | **`bong_tangent` is shift-free** — identical sigmas for shift 1.0 / 1.15 / 2.475 and for `useDynamicShifting` true/false. | `BongTangentScheduleTests.testIgnoresModelSampling` |
| 21 | **`beta` matches ComfyUI at Krea 2's registered shift**: `beta(6, shift: 1.15)` == `[1.0, 0.919919, 0.751973, 0.535879, 0.304782, 0.104360, 0.0]` to 1e-5 (**verified against upstream source**, ComfyUI `samplers.py:456-468`, pure-Python re-run 2026-08-22); `beta(9)`, `beta(30)`, `beta57` match their fixtures. Built through `Krea2Sampling.schedulerConfig(shift: 1.15)` — the synthetic config's seven values are pinned in §3.1, so this criterion tests one thing. | `BetaScheduleComfyParityTests` |
| 22 | **`beta` de-duplication is reported, not hidden**: a step count producing fewer than `steps+1` sigmas yields `steps_effective < steps_requested` and the render succeeds. | `BetaScheduleComfyParityTests.testDedupeReported` |
| 23 | **DEIS coefficients match `rhoab`** to 1e-6 for orders 2/3/4 on a fixed sigma array, including the `min(i+1, max_order)` ramp; computed in `Double`. | `DEISMultistepSchedulerTests.testRhoabCoefficients` |
| 24 | **`deis_3m` warm-up is faithful**: `maxOrder 3`, 2 steps → `warmup_sampler == "ralston_3s"`, `warmup_steps == 2`, **6** model evaluations. At 8 steps → `warmup_steps == 3` and steps 3–7 use order-3 coefficients. | `DEISMultistepSchedulerTests.testOrderRamp` |
| 25 | **Order of accuracy is real.** On a synthetic linear ODE with closed-form solution, halving steps multiplies terminal error by ≈2^p with p≥2 (`ralston_2s`,`res_2s`), p≥3 (`ralston_3s`,`res_3s`,`deis_3m`), p≥4 (`ralston_4s`,`deis_4m`). | `ExplicitRKSchedulerTests` + `DEISMultistepSchedulerTests` |
| 26 | **Step-trace parity with RES4LYF** to 1e-4 relative after every step, for `{res_2s + beta, 6}` and `{deis_3m + bong_tangent, 2 @ denoise 0.2}`, at each tier: eta 0/bongmath off (T1), eta 0.5 (T2), bongmath true (T3). | `RES4LYFTraceParityTests` (§5.2) |
| 27 | **T2/T3 determinism.** Two staged renders with identical payloads including seeds are byte-identical; changing only `stage2.seed` changes the output. | `Krea2StagedRenderTests.testDeterminism` |
| 28 | **An unimplemented tier is a 400 — on the Krea 2 path only.** `eta != 0` before T2, `bongmath: true` before T3, `stage2` on a non-krea2 family, `scheduler: "res_5s"`, `sigma_schedule: "ays"`, `stage2.denoise: 0` each return 400 naming the unsupported value. No request in the matrix silently becomes euler/flow. **And the regression that v1's design would have broken:** a Z-Image render with `{"scheduler":"ddim","eta":0.5}` still succeeds after WP-E4 and WP-E15 land, and `--eta` on the CLI still works — `eta` has shipped on that path since April (`WarmServer.swift:7514,7626,7760,7832`; `main.swift:159-160`) and the tier gate lives inside `runKrea2Generate`, not at the decoder (D18). | `StagedPayloadDecodeTests.testRejectionMatrix` + `ZImageEtaRegressionTests` |
| 29 | **The published recipe is expressible in one payload**, and every published default appears in `applied` — including `vae` naming `Wan2_1_VAE_fp32.safetensors` with layout `wanNative` (D16, reversed in v2), `base_variant == "raw"`, the turbo LoRA at 0.6, and the bypass entry naming the **workflow's** file. | `Krea2StagedRenderTests.testReferenceRecipeRoundTrip` |

### O5 — the second-stage detail pass

| # | Criterion | Test |
|---|---|---|
| 30 | **Two stages, one render, no VAE round-trip**: exactly one `vae.decode` and zero `vae.encode` (call counter), and the output differs from stage-1-only at the same seed. `applied.stages.length == 2`; a client polish render sets `polish` and leaves `stages.length == 1`. | `Krea2StagedRenderTests.testSingleDecode` + `render-settings.test.ts` |
| 31 | **Stage-2 sigmas are the stretched tail**: `{steps:2, denoise:0.2, bong_tangent}` produces exactly `[0.117461, 0.043265, 0.0]` — not `[0.2, 0.1, 0]`, not a slice of stage 1 (**verified against upstream source**, `res4lyf_sigmas.py:1402`, pure-Python re-run 2026-08-22). Parametrised over `{steps 2…8} × {denoise 0.1…0.9}` against dumped Python values, with `denoise` decoded as `Double` and `total = Int((Double(steps)/denoise).rounded(.towardZero))` — `2/0.2` is `10.000000000000002` in double and a `Float` division can land on the other side of the integer. | `Krea2StagedSigmaTests.testStretchAndTail` |
| 32 | **Stage 2 at denoise → 0 is a no-op**: `denoise: 0.01` gives mean absolute pixel difference < 2/255 from stage-1-only. | `Krea2StagedRenderTests.testDenoiseZeroNoop` |

### O4 / O4a — Raw as a base, and kroma as policy

| # | Criterion | Test |
|---|---|---|
| 33 | `detect(at: ~/LocalModels/krea2-raw)` returns `variant == .raw`, `transformerFile == raw.safetensors`; the pipeline loads all **430** tensors with no `.shapeMismatch`. | `Krea2VariantDetectionTests` + `Krea2RawLoadTests` (integration) |
| 34 | **Fail closed (F3 regression).** `resolve(spec:)` on an *existing* directory with no recognisable DiT **throws `notAKrea2ModelDirectory`** — it does not return the HF-cache turbo snapshot. Asserted on the **error type**, not "not turbo". A dir with both files throws `ambiguousVariant`. | `Krea2VariantDetectionTests.testFailsClosed` |
| 34a | **An unmapped alias throws; it does not become Turbo** (the hole v1's own reference preset fell through). `resolve(spec: "krea2-raw")` with no `specDirectory` entry throws `notAKrea2ModelDirectory(reason: .unmappedSpec)`. With the entry present it returns `.raw` and `transformerFile == raw.safetensors`. The four turbo aliases (`krea2`, `krea-2`, `krea-2-turbo`, `krea/krea-2-turbo`) — **and only those four** — still reach the HF snapshot; a fifth invented alias throws. | `Krea2VariantDetectionTests.testSpecResolution` |
| 34b | **The name and the physics agree end to end.** After loading `model: "krea2-raw"`, `GET /health` reports `model == "krea2-raw"` **and** `model_variant == "raw"`, and `applied.base_model_file` ends in `raw.safetensors`. | `Krea2RecipeProvenanceTests.testRawVariantReported` (integration) |
| 35 | `resolveRecipe(steps: 30, family:'turbo')` → `{ok:false, code:'steps-out-of-range'}` whose reason names family, value, range and source. **It never returns 9.** `steps: 20` on `raw-accel` returns 20. `guidance: 2.0` on `raw-accel` returns 2.0; on `turbo` refuses. | `checkpoint-policy.test.ts` |
| 36 | Precedence asserted with all four sources populated and distinct (request > preset > mode > family-default-for-intent), for both steps and guidance. | `checkpoint-policy.test.ts` |
| 37 | `resolveLane('quality')` returns `{intent:'max', tier:'premium'}` and **no absolute step count**; end-to-end the quality lane resolves to 12 on turbo and 10 on raw-accel. | `image-lane-map.test.ts` |
| 37a | **The premium lane resolves by name on the live config.** `preflightRecipePolicy()` run against `~/.kira/config.json` reports the premium/`max` lane's resolution for each image preset — the checkpoint it will use (the preset `model`), the family, the resolved steps/guidance — or its refusal with a reason. `config.comfybox.baseCheckpoint`/`polishCheckpoint` produce **one startup warning naming both keys and the file**, and are not used for the image path. `generatePremium` on the new ladder resolves through the same recipe as `generate`. | `client.preflight.test.ts` + `client.premium-lane.test.ts` |
| 38 | `preflightRecipePolicy()` on a config with `modeSteps:{avocado:20}` + a turbo preset returns exactly one refusal naming mode, preset and range, and the log names the daemon home. | `client.preflight.test.ts` |
| 38a | **Nothing changes silently on deploy day.** `preflightRecipePolicy()` also emits a resolution-diff table: every (preset × lane × content mode) whose resolved `steps` or `guidance` differs from `resolveGenerationParams` on `34cb3244`, both values side by side. The table is committed as a fixture in the C1 PR. It **must** contain `krea-kira × render × *: steps 9 → 12` and `krea-film-* × render × *: steps 9 → 8` (§7.5), and must contain no row for `krea-bree`. | `client.preflight.test.ts` |
| 39 | **Turbo LoRA binds completely**: `loadForKrea2(krea2_turbo_lora_rank_64_bf16)` → 264 pairs, 7 deltas, rank 64, no throw; applied to Raw, `report.bound == report.offered` and `unbound.isEmpty`. | `Krea2LoRAKeyMappingTests` + `LoRAApplicationReportTests` |
| 40 | **Kroma binds completely on Raw**, 256 pairs / 0 deltas, and `effectiveScale(forLayer:)` returns exactly 1.0 for a rank-98 and a rank-299 layer (no alpha ⇒ no rank normalisation). | `Krea2LoRAKeyMappingTests` + `LoRAAlphaScalingTests` |
| 41 | **Relativity is enforced**: `kroma-lora-v0.3` (declared `.turbo`) on a `.raw` pipeline throws `incompatibleBase` and `loadedLoRAConfigs.isEmpty` afterwards (rollback held). | `Krea2LoRARelativityTests` |
| 42 | **No silent partial apply**: a four-deep stack either applies all four with `bound == offered` each, or throws. An artificially truncated LoRA in position 3 throws `partialApplication` with the base restored. `strict:false` (Z-Image/Flux2) does not throw. | `LoRAApplicationReportTests` |
| 42a | **`unbound` names offered keys, not unvisited modules.** A LoRA whose 264 keys include one that targets no module in the transformer reports `offered 264, bound 263, unbound == ["<that key>"]` — **not** thousands of entries for the modules that had no adapter, and not an empty list. A pair that matches a module but fails `normalizeLoRAPair` increments `shapeRejected` and appears in neither. Under `strict: true` the first case throws `partialApplication` naming the key. | `LoRAApplicationReportTests.testUnboundIsOfferedMinusConsumed` |
| 43 | **Turbo-LoRA strength is a live dial**: renders at 0.0 / 0.6 / 1.0 on Raw at one seed produce three distinguishable images and `scale_applied` matches each request. | `Krea2RawLoRAGradientTests` (integration) |
| 44 | **Kroma is a declared field.** A **krea2-family** image preset with no `kroma` fails `validateImagePreset` with `preset-invalid`, is logged once at fetch, and the render refuses. `{strength:0}` validates and contributes no LoRA. `{strength:0.6}` prepends the family-correct file. A preset declaring `kroma` **and** listing a kroma file in `loras[]` refuses. A Turbo-relative kroma on a raw family refuses, naming both. A `zimage-*` family preset with no `kroma` **passes** (D14). | `client.preset-policy.test.ts` |
| 44a | **The real store migrates clean.** A checked-in copy of the migrated `presets.json` (all 24 entries, 13 image) passes `validateImagePreset` for every image entry, and `preflightRecipePolicy()` reports **zero** refusals over every lane × content mode × preset. Run against the fixture, and re-run by hand against the live file before the C6 deploy. | `client.preset-policy.test.ts` |
| 44b | **The engine enforces it too** (O4a is not daemon-only). `PUT /v1/presets` with a krea2-family image preset lacking `kroma` returns **400** naming the preset and the field; a `zimage-*` preset is accepted. | `WarmServerPresetValidationTests` |
| 44c | An invalid preset already on disk is logged at error on load and appears in `GET /v1/presets` with `invalid: true`, so the desktop app and the bridge cannot select it. | `PresetStoreTests.testKrea2ImagePresetRequiresKroma` |
| 45 | **`kroma_strength` is read back, not echoed.** Given an `applied` block produced by the **real** engine serializer in which the kroma LoRA carries `role: "kroma"` and `scale_applied: 0.3`, the record reads `kroma_strength: 0.3` and `provenance: 'engine'` — even when the request asked for the preset's 0.6, in which case `substituted` contains the difference and a warn is logged. No kroma entry in `applied.loras[]` ⇒ `0`. No `applied` ⇒ the requested value under `provenance: 'request'`. A `kromaBaked` family records `"baked"`, never `0`. | `client.preset-policy.test.ts` + `client.provenance-readback.test.ts` |
| 46 | A `raw-accel` preset whose resolved stack contains no acceleration LoRA refuses with `accel-lora-missing`, **before** any swap or generate call fires (spy counts zero). | `client.preset-policy.test.ts` |

### O9 — the bypass LoRA

| # | Criterion | Test |
|---|---|---|
| 47 | `loadForKrea2(krea2filterbypass_2vector.safetensors)` — **the workflow's file**, SHA-256 pinned in the fixture — yields **0 pairs, 1 delta**, key resolving to `txtfusion.projector.weight`, `F32 [1,12]`, no throw. Fixture-gated: the test skips with a named message until WP-E8's acquisition step lands, and §7.1 carries O9 as *predicted — artifact missing* while it does. The same assertions hold for `krea2_filter_bypass_fedor.safetensors`, which **is** on disk. | `Krea2BypassLoRATests.testLoadsAsDelta` |
| 47a | **The substitution is measured, not assumed.** Dump both files' single tensor and assert element-wise equality between `krea2filterbypass_2vector` and `krea2_filter_bypass_fedor`. Equal ⇒ Fedor is a verified stand-in and `krea2-reference` may name either, with `RenderRecipe.loras[].file` recording which. Not equal ⇒ the test records the difference, Fedor leaves the reference preset, and its 3–5 strength becomes an O6 ablation row (§5.4). Either outcome is a pass; *not running it* is the failure. | `Krea2BypassLoRATests.testEquivalentToReferenceFile` |
| 48 | **The doubling holds under the stated dtype chain, and the chain is part of the assertion.** On a bf16-loaded Raw transformer whose projector is excluded from quantization (`Krea2Pipeline.swift:192`), applying the delta at strength **1.0** gives `txtfusion.projector.weight` columns 8 and 9 == `bf16(2·w)` and all ten other columns bit-unchanged — because `LoRAPatchSession.apply` casts the delta to the parameter dtype **before** scaling (`LoRAPatchSession.swift:133`). The raw values are **not** bit-equal (F32 `-0.5116999745` vs bf16 `-0.51171875`, ~4e-5 relative), so the same test at strength 0.5 asserts the ~4e-5 bound, not equality. v1 claimed unqualified exactness; that is corrected. | `Krea2BypassLoRATests.testDoublesTwoColumns` |
| 49 | A JSON error page named `.safetensors` throws `notASafetensorsFile` quoting the payload — not a `SafeTensorsReader` internal error, not a silent skip. **And the real 1,040-byte bypass file still loads** (no bare size floor). | `LoRALoaderTests.testJSONErrorPageRejected` + AC-47 |
| 50 | Bypass at 1.0 on the **deployed q8** base produces a non-black image (mean luminance > 0.02) at a fixed seed, and stacks with the turbo LoRA without either being dropped. | `Krea2BypassRenderTests` (integration, production config) |
| 51 | A bypass file whose delta shape does not match the loaded projector throws `partialApplication`, never applies partially. | `Krea2BypassLoRATests.testShapeMismatchThrows` |

### O7 — the decoder

| # | Criterion | Test |
|---|---|---|
| 52 | `Krea2VAEKeyMap.canonicalize` maps all **194** Wan-native keys onto Krea2VAE paths with exact shape equality and covers all 194 Qwen keys — asserted against both real files, both directions. | `Krea2VAEKeyMapTests` |
| 53 | `detectLayout` returns `.wanNative` for `Wan2_1_VAE_fp32.safetensors`, `.qwenDiffusers` for the kroma-v0.2 VAE, and **throws** for a third file. | `Krea2VAEKeyMapTests.testLayoutSniff` |
| 54 | Loading the Wan file into `Krea2VAE` completes with no `.shapeMismatch`; a fixed latent decoded through Qwen vs Wan gives two images that **differ** (weights are live) and are both well-formed (no NaN, range within [0,1]). | `Krea2VAELayoutLoadTests` |
| 55 | `latentsMean/latentsStd` equal the Qwen-Image `vae/config.json` values to 4 decimals and are **unchanged** across layouts. *(Pins the lineage inference; if Wan-AI's published constants differ, this is where it surfaces.)* | `Krea2VAELayoutLoadTests.testLatentNormalization` |
| 56 | Selecting a VAE not on disk **fails the render** with a path-naming error — never falls back to the model dir's VAE. | `Krea2VAESelectionTests` |
| 57 | img2img on a Wan-selected pipeline encodes through the **same** `Krea2VAE` instance (asserted by identity), so encoder and decoder can never disagree. | `Krea2VAESelectionTests.testSingleInstance` |
| 58 | **All nine new preset fields** — `checkpointFamily`, `kroma`, `vae`, `sampler`, `sigmaSchedule`, `shift`, `eta`, `bongmath`, `stage2` — survive write → read → resolve and appear in `ResolvedPreset`, asserted field by field (the `videoTuning` regression class; v1 covered `vae` only). | `PresetStoreTests.testNewImageFieldsRoundTrip` |
| 59 | **A VAE change is never a silent reuse.** Requesting Wan while a Qwen decoder is resident increments the in-place reload counter and the record's `vae` names the Wan file; requesting the resident one does not reload. The pipeline instance is unchanged across the swap (the pool is not evicted — D17), and a request naming a VAE not on disk fails the render with the resident decoder untouched. | `ModelPoolHandoffTests` + `Krea2VAESelectionTests` |
| 59a | **The handoff log line exists and names both variants** (D17). A base switch emits exactly one `krea2 handoff: <outgoing spec/variant> → <incoming spec/variant> (loadTimeMs=…)`, with both variants non-empty. | `ModelPoolHandoffTests.testHandoffLogLine` |

### Provenance (PRD §4) — spans O1/O3/O4/O4a/O5/O7/O9

| # | Criterion | Test |
|---|---|---|
| 60 | **Read-back, not echo.** A render requesting 30 steps on a preset with two LoRAs returns `applied.stages[0].steps_requested == 30`, `applied.loras` equal to `loadedLoRAConfigs` (file + scale) joined with the bind counts, `base_model_file` the resolved path, `base_variant` the loaded variant, `vae` the resolved path, `quantization == "q8"`. A LoRA that fails to load errors the render and **writes no image and no partial record**. | `Krea2RecipeProvenanceTests` (integration) |
| 61 | **The PNG carries the same record.** EXIF `UserComment` JSON contains `sampler`, `sigma_schedule`, `shift`, `steps_effective`, `steps_run`, `model_evals`, `guidance`, `vae`, `base_variant`, `base_model_file`, `loras[]` with `deltas_applied`, and — when `guidance > 1` — `negative_prompt`; when `guidance <= 1`, `negative_prompt` is **absent** (it did not apply). | `Krea2RecipeProvenanceTests.testPNGMetadata` |
| 62 | `/health.last_recipe`, `/health.model_variant` and the async job's `applied` carry the same values as the sync response for the same render. | `Krea2RecipeProvenanceTests.testAllFourSinks` |
| 63 | **img2img reports honestly**: `strength 0.3` / 20 steps reports `steps_requested 20`, `steps_run 14`, `model_evals 14` at guidance 1. | `Krea2DenoiseLoopTests.testImg2ImgStepAccounting` |
| 64 | Persisted pre-upgrade JSON lacking `applied` still decodes into **`ImageJobStatus`** (the persisted queue type), and the client's response/health decoders accept a body with no `applied` and report `provenance: 'request'`. *(v1 also named `GenerateResponse`; it is a `private struct … Encodable` at `WarmServer.swift:7915` — there is nothing to decode into, and we are **not** adding `Decodable` for a test.)* | `Krea2RecipeProvenanceTests.testBackwardCompatibleDecode` + `client.provenance-readback.test.ts` |
| 65 | A response with `applied` → `provenance:'engine'` and every settings field sourced from it. `applied.steps` differing from the request → `substituted: ['steps 6→9']` **and a warn log**. No `applied` → `provenance:'request'` and today's values. | `client.provenance-readback.test.ts` |
| 66 | `buildRenderSettings` with `model: undefined` reports `model: undefined` — never a config checkpoint. `sidecarLoraProvenance({appliedLoras: undefined})` returns `loras: undefined`. | `image-gen-provenance.test.ts` |
| 67 | Under seal: `model`, `preset`, `loras`, `model_variant` absent; `provenance`, `steps`, `guidance`, `sampler`, `sigma_schedule`, `loras_applied` retained. | `render-settings.test.ts` |
| 68 | No source file contains `Math.min(` applied to a step count, and no tool description contains a literal step count for a lane. | `image-gen-provenance.test.ts` (static assertion) |
| 68a | **The operator surface reaches the wire.** Each parameter exposed on the `generate_image` schema (`sampler`, `sigma_schedule`, `kroma_strength`, `detail_pass`, `detail_denoise`) arrives in the request body unchanged and under the right key — `sampler` → **`scheduler`** (D25), `detail_pass` → `stage2`. A parameter the schema does not expose (`shift`, `eta`, `bongmath`, `vae`) is reachable from a preset and from the compare CLI, and is absent from the tool schema. | `client.recipe-wire.test.ts` |

### O3 — negative prompts, and O6 — the comparison

| # | Criterion | Test |
|---|---|---|
| 69 | Two renders at one seed on `raw-accel`, identical but for the negative prompt at CFG 2.0, differ; `applied.stages[0].guidance == 2.0` and `negative_prompt` is present; `model_evals` is 2× the CFG-1 run, so the extra time is visible to the caller. **Direction** (PRD O3: "differ in the direction requested") is judged, not asserted: this pair **is** §5.4 ablation row (4), captioned blind, and the O6 report records whether the caption of the negative-prompt render lacks the negated attribute. Direction is an O6 judgement; a pixel test cannot make it. | `Krea2CFGRenderTests` (integration) + §5.4 |
| 70 | `assertNeutralPrompt` rejects a prompt containing an age word or a style word; `planComparison` refuses to build a plan from it. | `recipe-compare.test.ts` |
| 71 | `blindCaptionOrder` produces an order uncorrelated with the recipe list under a seeded RNG, and its mapping reconstructs the pairing exactly. | `recipe-compare.test.ts` |
| 72 | `buildComparisonReport` renders one row per recipe **including refused ones**, with the refusal reason in place of the image. | `recipe-compare.test.ts` |
| 73 | Every preset id in `KIRA_RENDER_SETS` and in the checked-in O6 recipe file has a `kiraRenderSetPolicy` row. | `render-sets.test.ts` |

### Z-Image `res_2s` (D2's declared break)

| # | Criterion | Test |
|---|---|---|
| 74 | **The Z-Image `res_2s` change is intended, measured and recorded.** A Z-Image render with `scheduler: res_2s` differs pre/post; the post-fix run satisfies AC-10's x₀ reconstruction on the Z-Image path; the changelog names it. Z-Image at defaults (euler/flow) is untouched (AC-6). | `ZImageRES2sCorrectionTests` |

---

## 5. Test plan

### 5.1 Unit — no weights, no GPU, in the default gate

`xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests` and `node scripts/run-tests.mjs`.

This is where most of the correctness lives, deliberately. Three techniques carry it:

- **Delegation makes equivalence structural.** `Krea2Sampling.timesteps` delegating to `SigmaSchedule.krea2` means AC-3 is a pinned-oracle test against a copy of the pre-change body inlined in the test, not a hope.
- **An injected `evaluate` closure removes the model.** `Krea2DenoiseLoop.run` takes a closed-form velocity field, so the multi-eval × CFG matrix, the reset discipline, the `startIndex` accounting and the x₀ reconstruction (AC-10, 12, 13, 63) all run in milliseconds with no weights.
- **A synthetic linear ODE gives order of accuracy without an oracle.** `denoised(x,σ) = A·x + b` makes the flow ODE closed-form, so AC-25 measures convergence order exactly. This is the strongest offline correctness signal available and it is worth more than any single fixture.

Header-only file tests need no GPU either: `Krea2VAEKeyMapTests` (AC-52/53) reads both real VAE headers, and `Krea2LoRAKeyMappingTests`/`Krea2BypassLoRATests` read the vault files, all fixture-gated with `XCTSkipUnless(FileManager.default.fileExists(…))` — the pattern `RealVAEExactTests.swift` already uses.

New/extended engine test files: `Scheduler/{Krea2SigmaScheduleTests, ModelOutputConventionTests, BongTangentScheduleTests, BetaScheduleComfyParityTests, ExplicitRKSchedulerTests, DEISMultistepSchedulerTests, RES3sSchedulerTests, SchedulerFactoryTests(ext)}`, `Krea2/{Krea2DenoiseLoopTests, Krea2RequestTests(ext), Krea2StagedSigmaTests, Krea2VariantDetectionTests, Krea2VariantDefaultsTests, Krea2LoRARelativityTests, Krea2BypassLoRATests}`, `Weights/{Krea2LoRAKeyMappingTests, LoRAApplicationReportTests, LoRAAlphaScalingTests(ext), Krea2VAEKeyMapTests, Krea2VAELayoutLoadTests, LoRALoaderTests(ext)}`, `Server/{SamplerNameResolutionTests, GeneratePayloadDecodeTests(ext), StagedPayloadDecodeTests, ZImageEtaRegressionTests, Krea2RecipeProvenanceTests, ComfyBridgeObjectInfoSamplerTests, BridgeKrea2VariantTests, PresetStoreTests(ext), WarmServerPresetValidationTests, ModelPoolHandoffTests(ext), PersistedQueueRecoveryTests}`, `Krea2/Krea2LaneRenderTests`, `MCP/MCPGenerateSchemaTests`.

New/extended client test files: `checkpoint-policy.test.ts`, `client.recipe-wire.test.ts`, `client.preset-policy.test.ts`, `client.provenance-readback.test.ts`, `client.preflight.test.ts`, `client.premium-lane.test.ts`, `client.mode-resolution.test.ts` (**narrowed, not deleted**, and named assertion by assertion in v2 — `testGuidanceAlwaysZero` (`:21-23`), `testStepsDefaultTo9` (`:29`) and `testLowerStepsPassThrough` (`:42`, `steps: 4 → 4`, the below-cap case) survive **verbatim** as the no-regression lock; `testModeStepsClampedToo`'s below-cap assertion (`:49`, mode steps 8 → 8) is kept and its two clamp assertions (`:48`, `:50`) move; `testStepsClampedDownFrom30` and `testExplicitStepsWinOverModeButStillClamped` become refusal assertions in `checkpoint-policy.test.ts`; the file header records where each moved), `image-lane-map.test.ts`, `image-gen-provenance.test.ts`, `render-settings.test.ts` (ext), `recipe-compare.test.ts`, `render-sets.test.ts`.

**One standing warning:** audit #1653 P0-2 found a **vacuous fixture** in `render-settings.test.ts` — a hand-written string that asserted nothing. The extension must produce its fixtures from the **real** producers.

### 5.2 Parity vs oracle — pinned fixtures and step traces

ComfyUI/RES4LYF are validation oracles, never backends. The fixture generator is a one-off Python script under `scripts/oracles/`, not engine code, so it does not violate the no-Python-in-ComfyBox rule.

**Sigma fixtures** (`Tests/ZImageTests/Fixtures/Scheduler/comfy_sigmas.json`, `res4lyf_deis_coeffs.json`) — `beta`/`beta57` at 6/9/30 steps under `ModelSamplingDiscreteFlow(shift=1.15)`, `bong_tangent` at 2/6/8/9/10/12/20, `rhoab` coefficient lists for orders 2/3/4. The `bong_tangent` and `beta` values quoted in AC-19/21/31 are **already verified against upstream source** by a pure-Python re-run on 2026-08-22 (`res4lyf_sigmas.py:4065-4098`, ComfyUI `samplers.py:456-468`) — v1 labelled them "Engine C's independent derivation", which understated them. The scripted dump is therefore **confirmatory**: **a mismatch between the pinned values and the dump means the port source moved and this FDD is stale** — that is the check, and it is still not a formality.

**Step traces — the load-bearing test (AC-26).** Run RES4LYF's `rk_sampler_beta` against a *scripted denoiser* that both stacks evaluate identically and that needs no weights: `denoised(x, σ) = 0.5·tanh(x) + 0.25·σ − 0.1·x`, latent 1×16×8×8 so the JSON stays small. Export per step: `σ`, `σ_down`, `σ_up`, `alpha_ratio`, every row's `x` and `k`, **the injected noise tensor** (so the Swift side never has to reproduce torch's RNG), and `x_next`. Three runs per recipe — eta 0/bongmath off, eta 0.5, eta 0.5 + bongmath — giving T1, T2 and T3 each an independent, falsifiable gate. This validates tableaus, DEIS coefficients, the eta split and the bongmath fixed point end-to-end with zero weights and zero GPU.

**Same-seed end-to-end vs ComfyUI: assessed and scoped out.** Not feasible as a bit comparison — initial noise comes from torch's generator vs `MLXRandom.seed`, the text encoder is a separate implementation, and the VAE decode differs numerically. Claiming otherwise would be an uncalibrated claim. The fallback if the traces leave doubt: export ComfyUI's initial noise latent and conditioning to `.safetensors`, inject both, and compare **latents** (not pixels) after each stage. A day of plumbing for a test §5.2 largely subsumes — **defer**, and build only if the traces pass while real renders still look wrong.

### 5.3 Integration — live engine, production config

`ZImageIntegrationTests`, run by hand, at the real dims / real quantisation / real files. Never at 256×256 or 2 steps: the two regressions this project has already paid for came from testing at convenient settings.

Baselines for AC-1/2/5/6 are captured **before any change lands, on the same binary**, and committed as hashes.

Ordering matters for cost (D17): every integration render on one base runs before the base switches. A full pass is roughly — turbo baselines (AC-1,2,5,6,74) → Raw loads and gradients (AC-33,43,50,60,61,62,69) → VAE pair (AC-54,57) → staged reference recipe (AC-29,30,32,27).

### 5.4 The O6 protocol

Runs **after** parity, exactly as PRD §4 specifies, via `scripts/image-compare.mjs`:

- One seed, held across every variant. One prompt with no age words and no style words (`assertNeutralPrompt` enforces, AC-70).
- **Run the FULL adopted stack against the current one first.** Ablate afterwards, only to learn which axis carried the gain.
- One variable per render thereafter. Recipes ordered to batch by base (D17), checkpointed per recipe.
- Blind captioning by the vision model through the same endpoint `look` uses; the caption is the measurement.
- Every image carries its settings, and the report's settings table shows `provenance` per row so an engine-echoed row is never read as an engine-verified one.
- Refusals are rows in the report, not omissions.
- The write-up records the result even when it says the old recipe won.

Ablation set, in the order the outcomes make them meaningful: **(1)** current (kroma-v0.2-turbo, 9 steps, euler, guidance 0, Qwen VAE) vs **full reference including the Wan 2.1 FP32 decoder** (D16 — because the reference preset now carries it, row (1) does not have to be re-run after O7 resolves); then **(2)** turbo-LoRA strength 0.0 / 0.6 / 1.0 on Raw; **(3)** kroma 0.0 / 0.5 / 1.0 on Raw; **(4)** CFG 1.0 vs 2.0 with the same negative; **(5)** sampler `euler` vs `res_2s+beta` at held steps; **(6)** single-stage vs +stage-2; **(7)** Qwen-Image VAE vs Wan 2.1 FP32 (the held-seed pair O7 requires); **(8)** bypass 0.0 vs 1.0, run and judged **separately** so the content-policy axis does not confound the realism question (PRD §5).

---

## 6. Work packages & sequencing

Sizes: XS ≤ ½ day · S ≈ 1 day · M ≈ 2–4 days · L > 1 week. Every WP is one PR.

| WP | Repo | Outcome(s) | Depends on | Size | PR title |
|---|---|---|---|---|---|
| **Phase 0 — foundations, all parallel** ||||||
| E1 | engine | O1 | — | S | `feat(krea2): SigmaScheduleKind.krea2 — native warp as a first-class schedule` |
| E2 | engine | O1, O8 | — | S | `fix(scheduler): declare model-output convention; res_2s takes x₀, not velocity` |
| E4 | engine | O1, O8 | — | S | `fix(server): reject unknown sampler/schedule names (keep the ComfyUI aliases); reconcile advertised lists` |
| E5 | engine | O4 | — | M | `feat(krea2): Krea2Variant — Raw as a second base, fail-closed model-dir resolution` |
| E18 | engine | O8 | — | M | `test(scheduler): RES4LYF/ComfyUI oracle fixtures + step-trace harness` |
| C1 | client | O2, O3, O4a | — | M | `feat(comfybox): checkpoint-family policy — unpin guidance, remove the step clamp` |
| **Phase 1 — the spine** ||||||
| **E3** | engine | **O1** | E1, E2 | **M+** | `refactor(krea2): denoise loop onto ZImageScheduler; N-row protocol; request fields` |
| E6 | engine | O4, O9 | E5 | M | `feat(lora): relativity guard + LoRAApplicationReport + strict apply` |
| E9 | engine | O7 | E5 | M | `feat(krea2): VAE selection + Wan 2.1 key map` |
| E11 | engine | O8 | E1 | S | `feat(scheduler): bong_tangent (RES4LYF-exact, shift-free)` |
| E12 | engine | O8 | E1 | M | `fix(scheduler): ComfyUI-exact beta/beta57 + explicit shift field` |
| C2 | client | O1, O3, O5, O8 | C1 | S | `feat(comfybox): plumb sampler/schedule/shift/eta/stage2/vae to /v1/generate` |
| **Phase 2 — Raw is runnable** ||||||
| E7 | engine | O4 | E6 | XS | `test(krea2): kroma-v0.2 Raw-relative LoRA binds on Raw (256/256, 0 deltas)` |
| E8 | engine | O9 | E6 | XS + **acquisition** | `feat(krea2): bypass LoRA via the existing .diff path + strength policy` — the code is XS; **step 0 is obtaining `krea2filterbypass_2vector.safetensors`** (§3.8), which is not a coding task and is the actual long pole. E21 is blocked on it as well as on T3 |
| E13 | engine | O8 | E3 | M | `feat(scheduler): N-row tableau driver + ralston_2s/3s/4s + res_3s` |
| E19 | engine | O2, O3 | E5 | S | `fix(bridge): Krita krea2 arm is variant-aware; Raw honours steps/CFG/negative` |
| E20 | engine | O4a, O7, O8 | E9 | S | `feat(presets): checkpointFamily, kroma, vae, sampler fields on ImagePreset` |
| E10 | engine | all | E3, E5, E6, E9 | M | `feat(server): RenderRecipe on response, PNG, /health, async status` |
| **Phase 3 — the reference pairing** ||||||
| E14 | engine | O8 | E13, E18 | M | `feat(scheduler): deis_2m/3m/4m (rhoab, order ramp, ralston warm-up)` |
| E15 | engine | O8 | E13, E18 | M | `feat(scheduler): eta SDE (T2) — hard-mode VP split + per-stage noise stream` |
| E17 | engine | O5, O8 | E3, E10 | M | `feat(krea2): two-stage recipe inside one render (stage2, stretch-and-tail)` |
| C3 | client | O4a | E20 | M | `feat(comfybox): kroma as a declared preset field + preset validation` |
| C4 | client | all | E10 | M | `feat(image): read applied recipe back; provenance flag; drop the two echoes` |
| **Phase 4 — parity** ||||||
| E16 | engine | O8 | E13, E18 | M | `feat(scheduler): bongmath fixed point (T3)` |
| E21 | engine | O4b | E8, E14, E15, E16, E17, E20 | S | `feat(presets): krea2-reference — the published recipe as one preset` |
| C6 | client | O4a | C3 | S | `chore(kira): lane preset migration + render-set policy rows + preflight wiring` |
| **Phase 5 — measurement, last** ||||||
| C5 | client | **O6** | C2, C4, E21 | M | `feat(image): recipe comparison library + image-compare CLI` |

**Critical path:** E1/E2 → **E3** → E13 → E14/E15/E16 → E17 → E21 → C5. Roughly 8 engine WPs deep.

**What runs in parallel and matters:** C1 (the client unpin, O2/O3) has **no engine dependency** and should start on day one — it is the cheapest outcome in the programme and it is a prerequisite for exercising anything else at real settings. E5 (variant + fail-closed) likewise: it fixes a live silent-substitution bug (F3) and can ship before the loop refactor. E18 (oracle fixtures) should start early because E11/E12/E13/E14/E15/E16 all consume it and a missing fixture blocks a finished implementation.

**Merge gates.** E3 does not merge without AC-1 and AC-2 green (byte-identity). E6 does not merge without a clean build across Z-Image/Flux2/Chroma (D9). E10 does not merge without AC-60 (read-back, not echo). Nothing in Phase 4 merges without its tier's trace fixture (AC-26).

**If schedule pressure appears:** `res_3s` is the **only** engineering-owned cut. `deis_4m` is named verbatim in PRD O8's acceptance and `ralston_4s` is its warm-up; both are in, and changing that is a PRD scope change requiring Todd (§9, Q6), not a schedule decision. (v1 listed them as a second-tier cut, which framed a mandatory outcome as optional — corrected, D20.)

---

## 7. Rollout

### 7.1 Assets — status verified today

| Asset | Path | Status |
|---|---|---|
| Krea 2 Raw DiT | `~/LocalModels/krea2-raw/raw.safetensors` | **Complete**, 26,283,332,608 B, 430 tensors verified. **Source: Comfy-Org/Krea-2 `diffusion_models/krea2_raw_bf16.safetensors`** — the official `krea/Krea-2-Raw` repo is gated and the direct fetch returned a 149-byte body (`fetch.log:5,7`). WP-E5 pins its SHA-256 in the fixture before first use |
| Raw model dir borrows | `~/LocalModels/krea2-raw/{text_encoder,tokenizer,vae}` | Symlinked to the Krea-2-Turbo HF snapshot — **same pattern as `~/LocalModels/kroma-v0.2`**, verified |
| Turbo LoRA (rank 64) | `~/comfybox-models/loras/vault/krea2_turbo_lora_rank_64_bf16.safetensors` | Present, 469 MB |
| Kroma Raw-relative | `~/comfybox-models/loras/vault/kroma-v0.2-base-lora-rank-384-fro-0985.safetensors` | Present, 3.6 GB |
| Kroma Turbo-relative | `~/comfybox-models/loras/kroma-lora-v0.3.safetensors` | Present (live, `krea-kira`) |
| Bypass — **the workflow's file** | `krea2filterbypass_2vector.safetensors` (civitai 2728234 / version 3066812) | **MISSING.** The fetch returned a 99-byte `{"error":"Early…` auth response (`fetch.log:3`). **WP-E8 acquires it.** Until then **O9 and O4b are predicted, artifact missing** — the mechanism is verified (F1), the artifact is not |
| Bypass — the Fedor variant | `~/comfybox-models/loras/vault/krea2_filter_bypass_fedor.safetensors` | Present and valid, 1,040 B (civitai 2746817). A *different* artifact whose equivalence to the workflow's file is its author's `__metadata__` claim, and whose published strength guidance (3–5) conflicts with the workflow's (1.0). AC-47a measures it; it is **not** what `krea2-reference` names |
| Wan 2.1 FP32 VAE | `~/LocalModels/vae/Wan2_1_VAE_fp32.safetensors` | Present, 508 MB |

**Raw model dir layout** (already correct on disk — nothing to build):
```
~/LocalModels/krea2-raw/
├── raw.safetensors                → the only file that differs from kroma-v0.2
├── text_encoder -> …/models--krea--Krea-2-Turbo/snapshots/1161245…/text_encoder
├── tokenizer    -> …/tokenizer
└── vae          -> …/vae
```
`Krea2ModelPaths` after WP-E5 resolves `raw.safetensors` → `.raw`. Note `Krea2ModelPaths.resolve` must keep tolerating symlinked subdirectories, which it does today.

**Disk:** Raw adds 26 GB to an internal volume at 88% with ~221 GiB free (PRD §5). Acceptable, but it is the tight constraint — no third full checkpoint without a plan.

### 7.2 Branch strategy — and the blocker that is not ours to clear

**The live engine tree `~/Projects/zimage.swift` is not a development tree for this work.** Verified today: it is on `fix/face-anchor-icref-broadcast`, **1 commit ahead of `origin/main`** (`c39136a`, the face-anchor mask pad for the IC-control broadcast crash-loop) and carries **uncommitted work from another session** — modified `Sources/ZImage/LTX2/LTX2VideoGenerator.swift`, `Sources/ZImage/LTX2/VAE/LTX2AudioVAE.swift`, `Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift`, plus an untracked `LTX2HiFiGANVocoder.swift`.

Rules, in order:

1. **All development happens in `~/Projects/zimage-krea2raw`**, branched from `origin/main`. Nobody on this programme runs `git checkout`, `git stash`, `git commit` or `git push` in `~/Projects/zimage.swift`. (Memory: a prior subagent's stash/checkout comparison poisoned a commit. Not repeating it.)
2. **The uncommitted LTX2 work is a hard deploy blocker owned by its session or by Todd.** It must be committed (or deliberately discarded by its owner) before any binary built from another branch replaces the live one, or the LTX2 HiFiGAN vocoder work is lost when the tree is next touched. **This FDD does not clear it and no WP is blocked on clearing it — but no *deploy* can proceed until it is.**
3. **`fix/face-anchor-icref-broadcast` merges to `main` first.** Its LTX fix is a crash-loop fix on the engine the Kira scheduler renders on 24/7; a krea2 binary built from `main` without it would regress video. Our branch rebases on `main` after that merge, and the first deploy build is verified to contain `c39136a` (`git log --oneline | grep c39136a` in the build tree, and an LTX2 i2v smoke render).
4. **The deployed binary does not live in anyone's `.build/`.** v1's deploy step copied our binary over `~/Projects/zimage.swift/.build/release/ComfyBox` — the very tree rule 1 forbids us to touch, and the tree another session builds from. That is a two-way hazard: the next `swift build` in that tree silently replaces the krea2 engine with an LTX-branch build carrying none of these WPs, and our `cp` destroys whatever that session had just built. §7.3 moves the deploy target to a versioned path outside every worktree.

### 7.3 Deploy mechanics

The live engine is `launchd com.barkadabrew.comfybox` running `~/Projects/zimage.swift/.build/release/ComfyBox serve --model ~/LocalModels/kroma-v0.2`. Kira's scheduler renders on it 24/7.

**The deploy target changes** (v2). Today the plist runs `~/Projects/zimage.swift/.build/release/ComfyBox` — a path inside a live development worktree. Two sessions writing to it is how the krea2 engine gets silently replaced by an LTX-branch build, and how another session's in-flight binary gets destroyed by our `cp`. The binary moves outside every worktree, keeps its own `mlx.metallib`, and carries its sha in its name so a swap is auditable:

```
~/.comfybox/bin/
├── ComfyBox-<sha>          ← built + codesigned, one per deploy
├── mlx.metallib            ← copied beside it (a clean .build never regenerates it)
└── current -> ComfyBox-<sha>
```
`com.barkadabrew.comfybox`'s `ProgramArguments[0]` becomes `~/.comfybox/bin/current`. **Rollback is `ln -sfn ComfyBox-<previous-sha> current` + bootout/bootstrap** — a named path, not "the previous file, kept for the duration". Repointing the plist is a one-time change Todd approves (§9 Q9); if he prefers the established flow instead, the alternative is written there and is equally acceptable — it is the *shared mutable `.build/`* that is not.

```
1. Tell Todd BEFORE starting — the pause is visible and a codesign prompt can block.
2. Pause the queue.                      (persists across restarts — ALWAYS resume)
3. Build in ~/Projects/zimage-krea2raw:
   xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' \
              -derivedDataPath .build/xcode
   # DO NOT rm -rf .build — it kills mlx.metallib and swift build won't regenerate it
   #   (recovery: copy from the desktop app bundle)
4. Verify the build carries the LTX fix:  git log --oneline -20 | grep c39136a
5. /bin/cp -f the binary to ~/.comfybox/bin/ComfyBox-<sha>, copy mlx.metallib beside it,
   re-codesign, then repoint `current`.  NEVER write into another worktree's .build.
   # in-place rebuild → codesign SIGKILL .ips reports are benign
6. launchctl bootout + bootstrap com.barkadabrew.comfybox
7. Smoke, in this order:
   a. LTX2 i2v render (proves the face-anchor fix survived)
   b. Krea 2 turbo t2i at seed 44821 → SHA-256 against the AC-1 fixture
   c. GET /health — model_family, model_variant, last_recipe present
   d. POST /v1/generate {"scheduler":"uni_pc"} → expect 400
   e. GET /health — build_sha equals the sha we just deployed
      (new field; without it a clobbered binary is undetectable from outside)
   f. A Krita render on the default style (sampler euler, scheduler "normal") → 200
      — the D22 regression, checked on the live app, not only in tests
8. Resume the queue. Confirm it is running.
```

**Deploy ordering across the two repos is not optional** (client-draft R2, verified against `PresetStore.swift:157-190`): the engine's preset **encoder** drops unknown fields on any write, so a desktop-app preset save between two deploys silently erases new preset fields — the same failure the `videoTuning` comment records.

```
WP-E20 (engine preset schema)  →  presets.json edit  →  WP-C3/C6 (daemon deploy)
```
Never the reverse, and never a desktop preset save in between. Client deploys follow the standard flow: push branch → `ssh todd@10.0.100.232` (fish) → `local-merge.sh` gate → webhook self-deploy + daemon restarts.

### 7.4 Rollback

Most of this programme is additive and rolls back by not selecting the new preset. **Three changes are not additive** and each needs an explicit line:

| Change | Rollback |
|---|---|
| **Fail-loud parsing** (WP-E4). Anything currently sending a name that has **no alias** starts getting 400s. Known senders: **Krita AI Diffusion** — which sends `normal`/`ddim_uniform`/`sgm_uniform` on every default style (`style.py` `_scheduler_map`) and is why D22 **keeps** those four aliases rather than rejecting them; `src/tile/seamless-processor.ts:90-98`; persisted queue jobs accepted under the old parser; and anything sending `uni_pc`/`dpmpp_2m_sde`, which we advertise today and which do not exist. Verification on deploy is smoke step 7f — a real Krita render on the default style, not only the test matrix. | Revert WP-E4 alone — it has no dependents. Watch `serve.err.log` for `unknownSampler`/`unknownSigmaSchedule` in the first 24 h, and check the Krita render before resuming the queue. |
| **`beta`/`beta57` replacement** (WP-E12, D5). Behaviour change for the tile pipeline and the CLI. | Revert WP-E12; `bong_tangent` (E11) is independent and stays. The before/after fixture is committed so the difference is inspectable without a rebuild. |
| **Z-Image `res_2s` correction** (WP-E2, D2). | Revert WP-E2 — but then WP-E3 must not merge, because Krea 2's `res_2s` inherits the same conversion. These two revert together or not at all. |

Binary rollback is `ln -sfn ~/.comfybox/bin/ComfyBox-<previous-sha> current` + bootout/bootstrap (§7.3) — a named path that no other session writes to. Preset rollback: `~/.comfybox/presets.json` is copied to a timestamped backup before the migration edit.

### 7.5 The aesthetic-continuity call Todd owns

PRD §5: *"every image in the gallery to date carries kroma's look. Changing the base changes how the characters render from identical prompts. That is a continuity decision for Todd."*

This FDD is built so that call is a **preset edit, not an engineering change**, and so it can be made after seeing the O6 images rather than before:

- Bree's lane (`krea-bree`, `kroma-v0.2-turbo`) and Kira's lane (`krea-kira`, `krea2` + `kroma-lora-v0.3 @ 0.6`) **keep their checkpoints and their LoRA stacks**. They gain `checkpointFamily: 'turbo'` and an explicit `kroma` declaration (O4a).

- **But "otherwise render exactly as today" is false for Kira and for the film presets, and v1 asserted it. Corrected, and announced.** Today `resolveLane('render')` passes an explicit `steps: 9` (`image-lane-map.ts:9-13`, `image-gen-tools.ts:626-629`), `resolveGenerationParams` clamps it (`client.ts:549`), and `parsePreset` never reads a preset's `steps` at all (`client.ts:711-729`) — so `krea-kira` renders at **9** while its own preset says **12**, and `krea-film-*` render at 9 while theirs say **8**. Under WP-C1 the lane carries an *intent*, not an absolute, and the preset's declared value wins. So on the first client deploy:

  | preset | steps today | steps after C1 | guidance today | after |
  |---|---|---|---|---|
  | `krea-kira` (24/7 lane) | 9 | **12** | 0 | 1 — numerically identical, the engine's CFG test is `> 1.0` |
  | `krea-kira-hq`, `krea-kira-sfw` | 9 | 9 | 0 | 1 (as above) |
  | `krea-film-apple/banana/avocado` | 9 | **8** | 0 | 0 |
  | `krea-bree` | 9 | 9 | 0 | 1 (as above) |
  | `imported-cs-*` (Z-Image) | 9 | their declared 9–16 | 0 | their declared 3.5–5.0 |

  That a preset's declared `steps` has never been read is the same defect class the PRD is built on, so honouring it is the point rather than a side effect — but nobody should discover it from an overnight batch. AC-38a makes `preflightRecipePolicy()` emit this exact table at startup, and the table is committed as a fixture in the C1 PR. `~/.kira/config.json` also carries `modeSteps: 10` for all four modes, which today clamps to 9 and after C1 loses to the preset; that is in the table too. **The first Kira overnight cycle after the C1 deploy is watched, not assumed** (R3).
- `krea2-reference` is a new, additional preset. Nothing selects it by default.
- Moving a daily lane to Raw afterwards is one preset field (`model` + `checkpointFamily`), gated by the family policy that already knows the right step/CFG/sampler budget for it. The same is true of the decoder: `krea2-reference` decodes through Wan 2.1 FP32 (D16), the daily lanes keep Qwen, and moving them is one preset field after O6 answers Q5.

---

## 8. Risks & mitigations

**R1 — Byte-identity of the default path is the whole merge gate for E3. The half v1 worried about is now verified; the half it did not name is the live one.** *Verified by source:* mlx-swift converts a Swift `Float` to the array's dtype **before** the op (`MLXArray+Ops.swift:253-255`), so today's `(tp - tc) * v` and `FlowMatchEulerScheduler`'s `(sigmas[i+1] - sigmas[i]).asType(sample.dtype)` over a float32 sigma array are the same computation — AC-1 holds by construction. *The live risk is the opposite direction:* the img2img mix promotes to **float32** today because `tStart` is a 0-d float32 `MLXArray` (`Krea2ImageToImagePipeline.swift:123-124`); rewriting it with a Swift scalar would silently move the mix into bf16 and fail AC-2. *Mitigation:* §3.3 specifies the float32 `MLXArray` form explicitly, AC-2's test description repeats it, and AC-1/AC-2 remain E3's merge condition rather than a follow-up. If a future MLX release changes the promotion rule, the pre-planned one-line remedy stands. **A "close enough" result is still not acceptable.**

**R2 — Fail-loud is a live-traffic behaviour change.** *Mitigation:* the alias table is preserved in full and extended with the RES4LYF UI prefixes; the error names the valid set; the `/object_info` and MCP fixes ship in the same commit as the rejection, never after; persisted jobs fail loud with a recorded reason (AC-18) rather than either rendering wrong or vanishing.

**R3 — The client's day-one behaviour changes in two ways, and only one of them is a refusal.** (a) *Refusals:* `modeSteps` and the `quality` lane carry values that survive today only because of the clamp. Lane-intent resolution means the *lane* never refuses; the startup preflight names every remaining config combination that would (AC-38); the residue is hand-written `modeSteps` in `~/.bree/config.json` and `~/.kira/config.json` (both carry 10 for every mode). (b) *Silent resolution changes*, which v1 missed entirely: honouring preset `steps` moves Kira's 24/7 lane 9 → 12 and the three film presets 9 → 8 (§7.5). A refusal log would not have caught either. *Mitigation:* the preflight now also emits a **resolution-diff table** against the pre-C1 resolver, committed as a fixture in the C1 PR (AC-38a). **The first Kira overnight cycle after deploy is watched, not assumed.**

**R4 — `kroma_strength: 0` on `kroma-v0.2-turbo` is honest about the LoRA and misleading about the image**, because kroma is baked into that checkpoint. *Mitigation adopted* (the client draft's alternative, not its default): the family entry carries `kromaBaked: true` and the record reads `kroma_strength: "baked"`. AC-45.

**R5 — Provenance is only as honest as the engine.** If `applied` echoed the request instead of reading back, `provenance: 'engine'` would be a stronger claim than the data supports, and the client cannot prove otherwise. *Mitigation:* AC-60 is an engine-side obligation with a named test, and every field's source is specified in §3.10. This is the one place where a lazy implementation would quietly defeat the whole PRD.

**R6 — A third resident `.krea2` spec deepens existing pool thrash.** Measured: ~20 handoffs/day at ~67 s each between two bases already. *Mitigation:* D17 (don't raise the budget, batch by base in O6, mandatory handoff log naming the variant). Residual: A/B work is materially slower and the scheduler's throughput drops when lanes alternate bases.

**R7 — Cost is multiplicative and compounds three ways.** `res_2s` 6 steps = 12 evals; CFG 2 doubles to 24; `deis_3m`'s ralston warm-up adds 6; bongmath adds ~100 cheap tensor passes per warm-up step; the Raw checkpoint is ~50% slower per eval; and a base swap is ~67 s. The published "8 steps" is ~36 forward passes at CFG 2. *Mitigation:* `model_evals_total` is a required record field so nobody is surprised, and the overnight lane's throughput assumptions are re-derived from that number rather than from step count. Todd's latency ruling (PRD §5) is assumed to extend to render recipes; §9 Q1 asks him to confirm.

**R8 — `bongmath` is 100 iterations of a fixed point nobody here fully understands.** Highest uncertainty per line. *Mitigation:* its own tier, its own trace fixture, and the knowledge that it is inert on true multistep steps (`r4_rkm.py:713`) — which at the published settings means it affects *all* of stage 2 and none of the DEIS math. Do not skip it and call the recipe adopted (D19).

**R9 — The reference is a 7,000-line framework and we port the slice two recipes traverse.** `eta` and `bongmath` were both non-neutral defaults nobody had read; there may be another. *Mitigation:* the step traces are generated from the real framework at the real settings, so a mis-scoped path shows up as a numeric mismatch rather than as a plausible-looking image.

**R10 — Kroma-on-Raw is not `kroma-v0.2-turbo`** (D15). Any "we still have kroma" claim must be qualified. *Mitigation:* `deltas_applied: 0` on the face of every such render.

**R11 — Sigma-grid divergence from ComfyUI at the model level** (D3). Our effective shift at 1024² is 2.475 vs ComfyUI's 1.15. This is the single most likely reason a "parity" render still will not look like the author's. *Mitigation:* the `shift` field, the reference preset stating 1.15, and `shift_source` in the record. `bong_tangent` is immune.

**R12 — The reference stack's bypass file is not on disk, and the file that is on disk is a different artifact whose equivalence is an unverified authorial claim** (F1, D10). Its arithmetic is verified; its `__metadata__` claims about equivalence to the workflow's file, about "any variant", and about which columns are "refusal channels" are the file author's, and the two artifacts ship strength guidance that differs by 5×. v1 treated this as a footnote and built O9, `krea2-reference` and three ACs on the substitute — a silent substitution of a parity axis, which is the failure mode this document exists to prevent. *Mitigation:* WP-E8 acquires the workflow's file (step 0), AC-47 pins its SHA-256, AC-47a **measures** the equivalence rather than assuming it, §7.1 carries O9/O4b as *predicted — artifact missing* until then, and §9 Q3/Q4 put the content-policy and strength calls with Todd.

**R13 — q8 + a four-deep stack is untested at depth.** Each `applyDynamically` on a `QuantizedLinear` dequantizes and `LoRAPatchSession.commitQuantized` requantizes per patched weight; four adapters compound both error and time. *Mitigation:* AC-42 covers correctness; render-time impact is measured, not predicted.

**R14 — The O6 sweep is long, serial, and competes with Kira's queue.** *Mitigation:* run against a paused queue or schedule it; checkpoint per recipe so a mid-run failure does not discard completed renders.

---

## 9. Open questions for Todd

Only decisions an architect cannot make. Everything else is settled in §2.

**Q1 — Does the latency ruling extend to render recipes?** PRD §5 flags that Todd removed latency as a constraint for overnight production in the *vision-captioner* context and assumes it carries over. The compounded cost is R7: the reference recipe is ~36 forward passes at CFG 2 on a ~50%-slower checkpoint, against today's 9. If it does not carry over, this programme should stop before Phase 3, not after.

**Q2 — Do Bree's and Kira's daily lanes move to Raw, and when?** §7.5 is built so this is a preset edit after seeing O6, and this FDD does **not** move them. But Todd should say whether the intent is "Raw becomes the house base once O6 says so" or "Raw is a preset that exists alongside the current lanes indefinitely" — it changes what the O6 write-up needs to answer.

**Q3 — Is the bypass LoRA in the stack at all?** PRD §5: *"a content-policy artefact, not just a quality one. Its inclusion is Todd's call."* F1 makes it cheap and safe to run and shows exactly what it does (two layer-tap weights doubled). The engineering is done either way; whether `krea2-reference` ships with it is not an engineering decision.

**Q4 — Bypass strength: 1.0 or the Fedor author's 3–5?** These are not near each other — 1.0 doubles the two taps, 5.0 sextuples them. The file's metadata claims numerical equivalence with the workflow's file "at the same strength," which is an unverified assertion from a source whose two published recommendations differ by 5×. This FDD takes 1.0 (the workflow author's figure, and what the reference recipe is defined by) and records the divergence. Confirm, or make it an O6 ablation row.

**Q5 — Do the *daily lanes* adopt Wan 2.1 FP32 after O6?** Reframed in v2. The reference preset's decoder is no longer an open question: `krea2-reference` decodes through Wan 2.1 FP32, because that is what the reference stack is and PRD O7 says "the default direction is adoption" (D16). What is open is whether `krea-bree`, `krea-kira` and the film presets follow after §5.4's held-seed pair (row 7) — a one-preset-field change each, and a continuity decision (§7.5) as much as a quality one. No answer is needed before the first parity render; an answer is needed before the daily lanes move.

**Q6 — If schedule pressure appears, is cutting `deis_4m` acceptable?** It is named verbatim in PRD O8's acceptance and is not used by the published recipe. Cutting it is a PRD scope change, not an engineering call — and `ralston_4s` is its warm-up, so they go together. (`res_3s` is the only piece ours to cut; D20.)

**Q7 — Who clears the uncommitted LTX2 work in `~/Projects/zimage.swift`?** §7.2. It blocks every deploy and this programme will not touch it. Todd or the owning session needs to commit or discard it, and `fix/face-anchor-icref-broadcast` needs to merge to `main`, before the first krea2 binary swap.

---

**Q9 — Move the live engine binary out of `~/Projects/zimage.swift/.build/release/`?** §7.3. Today the launchd plist runs a binary inside a live development worktree that another session builds from, so a `swift build` there and a deploy from here can each silently undo the other. v2 proposes `~/.comfybox/bin/ComfyBox-<sha>` with a `current` symlink (rollback becomes a named path, and `/health.build_sha` makes a clobbered binary detectable). The alternative — the established flow: merge to `main`, fast-forward the live tree, build *there* — is equally safe and needs no plist change, but it couples every krea2 deploy to whoever owns that tree's uncommitted work (Q7). **Either is fine; what cannot stand is two sessions writing to one `.build/`.** Todd picks.

**Q8 — Raise `COMFYBOX_POOL_BUDGET_MB` to ~48000 for the A/B period?** D17 recommends no (it would push every video render into a reload). The cost of "no" is ~67 s per Raw/Turbo alternation during O6. Todd's call if he wants both bases warm.

---

## 10. Out of scope

- **Prompt vocabulary work** — parked per PRD §6 until the stack is at parity, because with CFG live and the decoder selectable it is a different lever than the one we were reasoning about.
- **Any change to the video pipeline.** The face-anchor branch merge (§7.2) is a prerequisite, not a change we make.
- **Retraining or retiring kroma.** It becomes a stackable dial under O4; whether we keep it is decided by O6.
- **Extracting a new Raw-relative kroma that includes the 170 norm/modulation deltas** (D15). Real, unscoped, nobody has sized it.
- **Generalising `RenderRecipe` to flux1/flux2/fibo/chroma** (D12) — filed as a follow-up ticket in the WP-E10 PR.
- **RES4LYF's guides, style transfer, implicit/pseudo-implicit samplers, overshoot, noise-scaling, non-gaussian noise, `sampler_mode` unsample/resample, the RES 5S/2M family, `sigmoid_offset`, `ays`, `lcm`.**
- **chroma-generate's `mlx_schedulers.py` as a port source.** PRD §7's own caveat: its `bong_tangent`, `deis` and RES family are approximations of different algorithms, and its schedules were never fed into its own timestep grid. It stays a naming/UX map only.
- **An MCP `compare_recipes` tool** — CLI first (§3.18); the MCP wrapper that enqueues and returns a report path is a follow-up.
- **Same-seed bit-comparison against ComfyUI end to end** (§5.2) — assessed, not feasible, deferred with a stated fallback.
- ~~**Todd's Buzz/CivitAI acquisition of the workflow's exact bypass filename.** F1 makes it unnecessary~~ — **struck in v2.** It is not unnecessary: the file is the censorship axis of the reference stack, it is not on disk (`fetch.log:3`), and the artifact v1 substituted for it is a different file at a conflicting strength. Acquiring it is **step 0 of WP-E8**, in scope, and it gates `krea2-reference` the same way T3 does (D10, D19, §7.1).
