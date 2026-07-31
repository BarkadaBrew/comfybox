# Krea-2 DyPE (NTK) — Design

**Date:** 2026-07-31
**Status:** Approved, ready for implementation planning
**Goal:** Give the Krea-2 pipeline high-resolution position handling (NTK), make it
reachable from Kira, and have the gallery HQ rerender use it.

## Problem

Kira's "HQ 2K rerender" upsizes to 2048px and renders on `krea2` (the `krea-kira`
and `krea-kira-hq` presets are both `"model": "krea2"`). Those renders get no
high-resolution position handling at all, so structure drifts as resolution
climbs past what the model trained on.

## Findings that shaped this design

Three facts, each verified against the code, that are not obvious from the outside:

1. **DyPE is absent from Krea-2.** `grep -rn 'dyPE\|DyPE' Sources/ZImage/Krea2/`
   returns nothing. Krea-2 has its own `Krea2Rope`
   (`Sources/ZImage/Krea2/Transformer/Krea2Transformer.swift:106-135`) with a
   fixed `cfg.theta`, entirely separate from `ZImageRopeEmbedder` where DyPE lives.
   The `dyPE` field rides along on the request and nothing reads it.

2. **YaRN is unimplemented.** `ZImageRopeEmbedder.swift:141-145` prints
   `"[DyPE] WARNING: YaRN method requested but not yet implemented"` and falls back
   to NTK. The four config knobs `beta0`/`beta1`/`gamma0`/`gamma1` exist only in
   `ModelConfigs.swift:30-56` — declared, defaulted, assigned, read by nothing.
   Every DyPE render ever produced by this codebase, on any model, used NTK.

3. **The uncommitted `.ntk` → `.yarn` change is a runtime no-op.** It produces
   identical pixels and adds a warning line per hi-res render. See Rollback below.

## Scope decision

NTK to Krea-2 first. YaRN is deferred: its gain over NTK is unmeasured here, and
there is no baseline to judge it against until Krea-2 has NTK working. The visible
win is Krea-2 going from *no* high-res handling to the same coherence Z-Image
already has.

## Design

### 1. Per-axis NTK in `Krea2Rope`

`Krea2Transformer.swift:106-120`. `make(pos:axes:theta:)` gains a
`scales: [Float] = [1, 1, 1]` parameter. Per axis, when `scale > 1`, substitute:

```swift
let ntkTheta = theta * pow(scale, Float(d) / Float(d - 2))
```

This is the formula in `ZImageRopeEmbedder.computeNTKFreqTable:80`, unchanged.
Axis 0 never scales. The default argument keeps every existing caller
byte-identical.

**Why axis 0 is safe to fix:** `Krea2Sampling.buildPositions`
(`Krea2Pipeline.swift:71-82`) places text tokens at the origin `[0,0,0]` and image
tokens at `[0, row, col]`. Axis 0 is unused by image tokens, and holding it vanilla
preserves text-image alignment — the same invariant `ZImageRopeEmbedder` maintains
via its `captionFreqs` path.

### 2. Scale derivation

Base resolution 1024px ÷ align 16 = **64 tokens** per axis, matching Z-Image's
`baseResolution / (latentDivisor * patchSize)` (`ZImageTransformer2D.swift:245`).

```
hScale = Float(hTok) / 64
wScale = Float(wTok) / 64
```

`hTok`/`wTok` are already computed in `Krea2Pipeline.generate`
(`Krea2Pipeline.swift:255-256`). Derive the scales there and pass them to the
transformer, which calls `Krea2Rope.make` once per forward pass
(`Krea2Transformer.swift:438`).

### 3. Request plumbing

`Krea2Pipeline.Request` (`Krea2Pipeline.swift:124-132`) carries
prompt/width/height/steps/seed/controlImagePixels — **no `dyPE` field**. Add one
(`dyPE: DyPEConfig = .disabled`).

The server-side adapter that builds this Request from a `ZImageGenerationRequest`
currently drops `request.dyPE`. That adapter lives in the Flux-2/Krea-2 dispatch
region of `WarmServer.swift` (near the `Krea2Pipeline.generateImg2Img` comment at
`:5368`); **locate it precisely during implementation** — it is not constructed via
a literal `Krea2Pipeline.Request(` spelling, so grep for the call site by dispatch
path rather than by constructor name.

Once `dyPE` reaches the pipeline, the existing auto-enable branches
(`WarmServer.swift:6249` txt2img, `:6338` img2img) start working for
`model: "krea2"` with no further change.

### 4. Kira access

`coffeeshop-server/src/tools/image-gen-tools.ts` declares no `dype` in the
`generate_image` schema (~`:287-300`), so the `dype: 'ntk'` that
`src/daemon.ts:476` already sends is silently dropped at the tool boundary. Add the
schema field plus the param mapping (~`:613`). No change to `daemon.ts` is needed —
its intent is already correct.

### 5. HQ buttons

**Telegram (Kirabot):** `daemon.ts:12052` (`hqrerender:` callback) →
`hq2kDims` (`:416`) → 2048px → `:490` sends the result. This needs **no change**.
It already upsizes past 1024, so it inherits DyPE the moment §3 lands. The button
starts working by fixing the layer beneath it.

**CoffeeShop Desktop:** there is a second HQ button in the desktop app
(`/Applications/CoffeeShop Desktop.app` = `ComfyBoxDesktop`, bundle id
`com.barkadabrew.comfybox.desktop`). **Its location is not yet identified** —
`GalleryView.swift` has no HQ/rerender button, and `ComfyBoxDesktopApp.swift:678`
only loads a prompt into Generate and switches tabs. Find it after the engine work
lands, then confirm it routes through a >1024 request so it inherits DyPE the same
way the Telegram button does. Tracked as an explicit open item, not a blocker.

## Testing

`Krea2Rope.make` is pure MLX with no model weights, so the scale math is
unit-testable without loading Krea-2 — cheap to run, no memory pressure.

- `scale == 1.0` produces output byte-identical to today (regression guard for
  every existing render path).
- `scale > 1.0` lowers frequencies for that axis (NTK spreads them for the wider
  position range).
- Axis 0 is unchanged under any `scales` value (text-image alignment invariant).
- Scale derivation: 2048px → `hTok/wTok == 128` → scale `2.0`; 1024px → scale `1.0`.

Follow the structure of `Tests/ZImageTests/Server/DyPEAutoEnableTests.swift`.

**Build note:** full `xcodebuild` runs are memory-hungry (cold builds recompile
~100 mlx-swift `.metal` kernels) and can OOM alongside other work. Run targeted
tests with `-only-testing:` and coordinate before running the full suite.

## Rollback / cleanup

Revert the uncommitted `.ntk` → `.yarn` edit at `WarmServer.swift:6249` and `:6338`.
It changes no pixels (YaRN is a stub) and adds a per-render warning line. Keep
`DyPEAutoEnableTests.swift` with its expectations flipped back to `.ntk`, since the
auto-enable branch is exactly what Krea-2 will now depend on and deserves coverage.

## Deferred, deliberately

- **YaRN.** Revisit once Krea-2 NTK 2K output exists to compare against. Requires
  implementing the interpolation blend and beta/gamma damping, and validating the
  numerics against the reference implementation.
- **`ImageToImagePipeline.swift:383`**, a third auto-enable site still on `.ntk`.
  It is guarded by `if request.dyPE.enabled`, so server-built requests pass through
  untouched; only callers that bypass `WarmServer` with DyPE off reach it. Harmless
  today, worth aligning when convenient.
- **Locating the desktop HQ button** (see §5).
