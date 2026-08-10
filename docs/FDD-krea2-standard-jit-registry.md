# FDD: Krea2 + Sulphur2 warm standard, JIT model/LoRA loading, LoRA registry + suggestion engine

**Repo:** `BarkadaBrew/comfybox` (`~/Projects/zimage.swift`, Swift/MLX)
**Author:** Opus (technical architect)  **Date:** 2026-07-20
**Status:** v1 — design for review (grounded in current `main` + LTX2 WIP working tree)
**Directive (Todd, 2026-07-20):** Standardize ComfyBox on **Krea2-Turbo (q8, image) + Sulphur2 (LTX-2.3 distil, video)** as the always-warm working standard. Every other model **and** every non-core LoRA goes to cold storage / HuggingFace and is pulled just-in-time on request, then idle-unloaded / LRU-evicted against a VRAM+disk budget. Add a LoRA **registry + prompt-based suggestion engine** with **JIT HF download**, and surface the selected preset/LoRAs to **Krita**.

---

## v2 changelog (resolves Fable review — VERDICT: BLOCK, 5 must-fix)

Fable verified as good and **kept unchanged**: `ModelPool` reuse accuracy, the honest #218 non-simultaneous framing, the suggestion-engine shape (§3.4), and the Krita surface (§3.5). The five blockers are fixed as follows (details in the cited sections):

- **B1 — Nearline reuse is broken for HF LoRAs.** Confirmed against `NearlineLibrary.swift`: `scan()` sets `state.items = found` from configured attached-storage **roots only** (drops any HF item with no root file → loses `lastUsedAt`/`stagedPath`), `stage()` throws `sourceMissing` without an attached `path`, and it stages into `~/.comfybox/loras` while our pull lands files in the LoRA library `root/vault/` (cache-dir mismatch). **FIX:** dropped the Nearline-reuse plan. **§3.3 now specs a small, dedicated `VaultCacheTracker`** (state `~/.comfybox/lora-cache.json`) that budgets exactly `root/vault/`, is never rebuilt by a root walk, and evicts LRU non-pinned pulled files. Nearline stays only for the attached-storage catalog.
- **B2 — Krea2 re-warm + video-active gate unspecified (OOM hazard).** **§3.1 now specs** (a) a `LaneCoordinator` that owns re-warm, **lazy by default** (re-warm pinned Krea2 on the next image request after video, not proactively), and drops the "re-warms in seconds" claim (honest: loads+quantizes ~13.5GB transformer + ~8GB Qwen3-VL, tens of seconds); (b) an explicit **VIDEO-ACTIVE GATE** — a `videoLaneBusy` flag that **defers all pinned re-warm and all JIT image `load`s** (returns 503/queued) until LTX2 is idle, so `evictIfNeeded` can never pass its (LTX-blind) budget math mid-render; (c) `releaseAll(keepPinned:)` preserves `activeId` iff it points at a kept entry, and its named caller is the `LaneCoordinator` video-handoff path (else the param is dropped).
- **B3 — catalog/scan integration bugs.** **§3.2/§4 now:** (a) merge matches by **`hfFilename`/filename, not id**, and mandates catalog `id == slugify(filename)` as a validation rule — no duplicate rows after pull+rescan; (b) preservation lives **inside `scan()`** (rows with `hfRepo != nil` and file absent are kept, not removed); (c) the evictor **clears `relativePath` and flips `local:false`** (keeps the row, no dead path); (d) `relativePath`/`sizeBytes` are **non-optional core fields** — specced the optionality refactor (`relativePath: String?`, `sizeBytes: UInt64?`) and its use sites (`resolve()`, `scan()`, `sizeFormatted`); (e) pull writes **repo-prefixed filenames** (`{repoSlug}__{filename}`) so generic ilkerzgi style names don't collide in `importFile`'s filename-dedupe.
- **B4 — budget headroom + pool-key drift.** **§6 now:** under `warmStandard`, raise the default pool budget to **`65536` (64GB)** so a JIT FIBO/Klein fits alongside pinned Krea2, **and** add a "pool contains only pinned/active, nothing evictable → allow the load anyway" escape in `evictIfNeeded` (analogous to the existing pool-empty escape), logged. **Pool-key unified:** `warmStandard` pins Krea2 with `quantization:"q8"` and `registerExisting` is changed to pass the real quantization (not hardcoded `nil`) so the pinned key (`...-q8`) matches the re-warm `load(..., quantization:"q8")` lookup — no duplicate 22.5GB Krea2, no dead pin.
- **B5 — suggestion nits + rollout.** **§3.4:** intent matching is **word-boundary** (`\bkira\b`/`\bpinay\b`), killing "shakira"/"akira" false positives; after a pull, **re-verify compatibility from the scanned file** and **quarantine on mismatch** (catalog `base_model` is unverified seed metadata); the content-mode ladder is corrected to **avocado → banana → apple, neutral == apple** (aligns `min_content_mode` gating). **§7:** Phase 0 is no longer labeled "no behavior change" — it carries the `relativePath` optionality refactor and can change `entry(for:)`/`compatible(with:)` once a catalog exists; its gate is now **"existing swap/resolve behavior byte-identical when no catalog file is present."**

---

## 1. Summary & motivation

ComfyBox today loads exactly **one** image model at startup (`Coordinator.prepare()` reads `configuration.modelSpec`, loads its pipeline, and registers it as the single active `ModelPool` entry). Everything else — Z-Image/CyberRealistic, Flux2-Klein, FIBO, Chroma, alternate LTX bases (10Eros, etc.), upscalers — either sits resident when selected or is loaded ad hoc. LoRAs live in a local `library.json` index (`LoRALibrary`) that only knows about files already **on disk**; there is no catalog of not-yet-downloaded HF LoRAs and no automatic "which LoRA fits this prompt" logic. The result is unpredictable VRAM/disk pressure (the 128GB machine can't hold image + LTX2 video simultaneously — #218) and manual LoRA juggling.

This design makes the **warm standard explicit and pinned**, demotes everything else to **JIT + LRU**, turns the existing `LoRALibrary` into a **registry that also describes catalog-only (HF) LoRAs**, adds a **suggestion engine** that turns a prompt + context into a ranked LoRA set and a resolved preset, and exposes that selection over a **Krita-facing REST surface**. Crucially, **most primitives already exist** — this FDD is mostly wiring and policy, not new subsystems:

| Need | Already exists (reused) |
|------|-------------------------|
| Multi-model warm pool + LRU eviction + VRAM budget | `ModelPool` actor (`Server/ModelPool.swift`) |
| Single warm model at startup | `Coordinator.prepare()` + `ModelPool.registerExisting` (`WarmServer.swift`) |
| LoRA index/scan/metadata/quarantine | `LoRALibrary` / `LoRAScanner` / `LoRALibraryEntry` (`LoRA/`) |
| HF download of a LoRA on demand | `LoRASource.huggingFace` → `LoRAWeightLoader.resolveSource` → `ModelResolution.resolve` (`Weights/`) |
| Disk-budget LRU (attached storage) — *pattern reference, not reused for HF; see B1* | `NearlineLibrary` (`Server/NearlineLibrary.swift`) + `/v1/nearline/*` |
| Preset → param bundle w/ LoRAs | `PresetStore` / `ImagePreset` / `ResolvedPreset` (`Server/PresetStore.swift`) |
| Hot LoRA swap | `/v1/lora/swap` → `coordinator.enqueueSwap` + `stageNearlineLoras` |
| MCP surface | `MCPToolExecutor` (18+ tools mapped to `/v1/*`) |

---

## 2. Current state (cited)

- **Warm model, single.** `Coordinator.prepare()` (`WarmServer.swift:4114`) branches by family (`isKrea2/isChroma/isFibo/isFlux2` else flux1), builds the pipeline (`Krea2Pipeline(paths:, quantizeTransformer:8)` for Krea2), and `await modelPool.registerExisting(...)` marks it active. Only `configuration.modelSpec` is warmed.
- **Pool + eviction.** `ModelPool` (actor) keys entries by `poolKey(modelSpec, quantization)`, tracks `vramEstimateMB` per entry (`VRAMEstimates.estimate`; `.krea2 = 22528`), enforces `budgetMB` (default **40960**, env `COMFYBOX_POOL_BUDGET_MB`), and evicts LRU **non-active** entries (`evictIfNeeded`, `releaseLRUInactive`). `releaseAll()` fully vacates the image side for the LTX2 video handoff (#218 — image+video cannot co-reside). **There is no pinning and no idle-unload timer.**
- **LoRA registry (local only).** `LoRALibrary` persists `library.json` (root `~/Models/loras` or `COMFYBOX_MODELS`), scanning `.safetensors` via `LoRAScanner.analyze` (detects `modelCompatibility` incl. `krea2`/`ltx`/`z-image`/`klein-9b`/`chroma`, `format`, `rank`, `alpha`, `triggerwords` from kohya `ss_tag_frequency` or a `*.civitai.json` sidecar). `LoRALibraryEntry` already carries `tags`, `category`, `sourceURL`, `civitaiModelId`, `recommendedScale`, `scaleRange`, `quarantined`. **Every code path assumes the file is present locally** — `resolve(id)` returns `root/relativePath`; there is no "known but not downloaded" state.
- **HF download primitive.** `LoRAConfiguration.huggingFace(modelId:filename:scale:)` + `LoRAWeightLoader.downloadFromHuggingFace` resolve via `ModelResolution.resolve` (swift-transformers `HubApi(downloadBase:)`, honoring `HF_HUB_CACHE`/`HF_HOME`; snapshots land in `~/.cache/huggingface/hub/models--ORG--REPO/snapshots/<commit>/`). So **JIT HF fetch already works**, but the result lands in the HF cache, not the library, and is not tracked for disk-budget eviction.
- **Nearline JIT (attached storage).** `NearlineLibrary` stages models/LoRAs from attached volumes (default `/Volumes/Seagate 22T/MacMigrate/Models`) into a local working set, **LRU-evicting against `cacheLimitGB` (default 200)**, state in `~/.comfybox/nearline.json`; `stageNearlineLoras(in:)` auto-stages a swap payload's LoRAs before applying. This is the exact JIT+LRU disk pattern to extend to HF.
- **Presets.** `PresetStore` (`~/.comfybox/presets.json`) resolves an `ImagePreset` (fields incl. `model`, `mode`, `loras:[LoraReference{filename,scale}]`, `injectedKeywords`, `steps`, `guidance`, `width/height`, `upscale`) onto `PresetDefaults`. `/v1/presets`, `/v1/presets/resolve`.
- **Styles.** `GET /v1/styles` serves `ComfyBoxStylePresets` (static prompt styles) — distinct from LoRAs; the style **lane** here is about LoRAs, not these text styles.
- **MCP.** `MCPToolExecutor` maps `swap_loras`→`/v1/lora/swap`, `list_loras`→`/v1/loras`, `lora_library`, `lora_scan`, `load_model`→`/v1/model/load`, `switch_model`→`/v1/model/activate`, `model_pool`, `unload_model`, `list_models`, `list_styles`, `apply_style`, `nearline_*`.

---

## 3. Design

### 3.1 Model standardization — pinned warm standard + JIT/idle-unload for the rest

**Goal state:** Krea2-Turbo (q8) is **pinned-warm** for image at all times; Sulphur2 is the **warm video target**; every other checkpoint is demoted to JIT load → idle-unload / LRU-evict.

Changes, smallest-surface-first:

1. **Config: `warmStandard`** (new block in `ComfyBoxServerConfig`, gated `warmStandard.enabled`, env `COMFYBOX_WARM_STANDARD=1`):
   ```
   warmStandard: {
     enabled: false,                 // opt-in; off = today's single-model startup
     image:  "krea/Krea-2-Turbo",    // pinned image model (q8)
     imageQuantization: "q8",
     video:  "sulphur2",             // warm video target (LTX-2.3 distil)
     idleUnloadSeconds: 900,         // non-pinned pool entries evicted after idle
     pinnedModels: ["krea/Krea-2-Turbo"]
   }
   ```
   When `enabled`, `Coordinator.prepare()` sets `configuration.modelSpec = warmStandard.image` (Krea2 path already exists) and marks the registered entry **pinned** (see #2). `video` is handled by the LTX2/`VideoGeneratorHolder` lane, unchanged except it reads its base from `warmStandard.video`.

2. **`ModelPool` pinning + key unification (B4).** Add `var pinned: Bool` to `PoolEntry`; `registerExisting(..., quantization:, pinned:)` and `load(..., pinned:)`. Pinned entries are **excluded** from `evictIfNeeded`, `releaseLRUInactive`, and the new idle sweep. `unload(modelId:)` refuses pinned entries (`ModelPoolError.cannotUnloadPinned`).
   **Pool-key drift fix:** `registerExisting` today hardcodes `quantization: nil`, so the startup Krea2 keys as `krea-krea-2-turbo`, but a re-warm via `load(warmStandard.image, quantization:"q8")` keys as `...-q8` — a mismatched pin that either duplicates 22.5GB or matches nothing. `registerExisting` is changed to take the real `quantization` and compute `poolKey(for: spec, quantization: "q8")`, and `warmStandard.imageQuantization` is threaded through both the startup register and every re-warm `load`, so the pinned key is identical everywhere.

3. **`releaseAll(keepPinned:)` (B2c).** Signature `releaseAll(keepPinned: Bool = false) -> Int`. Default `false` = today's byte-identical vacate-everything (the #218 LTX2 handoff). When `true`, kept pinned entries are retained **and `activeId` is preserved iff it points at a kept entry** (else set to `nil`, as today). **Named caller:** the new `LaneCoordinator` (below) is the *only* caller that passes `keepPinned:` — and it passes `false` for the video handoff (LTX2 needs the full ~65GB, so even pinned Krea2 must go). The `keepPinned:true` path exists for the memory-pressure guard that wants to shed only non-pinned image models without dropping Krea2; if that guard is not wired in v1, **the param is dropped and `releaseAll()` stays as-is** (decision at implementation — no speculative API).

4. **Idle-unload timer.** A lightweight repeating task calls `ModelPool.evictIdle(olderThan:)`, releasing any **non-pinned, non-active** entry whose `lastUsed` exceeds `idleUnloadSeconds`. LRU-on-budget (`evictIfNeeded`) is the other half and already exists.

5. **JIT load for demoted models.** No new load path — `/v1/model/load` → `modelPool.load(modelSpec:)` already family-detects, VRAM-estimates, evicts, and loads Z-Image/Flux2/FIBO/Chroma/Krea2. Demotion = not warming them at startup; the suggestion engine / explicit `load_model` pulls them on demand. Alternate **LTX video bases** (10Eros v1.4 fp8, etc.) load via `--ltx2-weights` / `VideoGeneratorHolder`, not the image pool.

6. **`LaneCoordinator` — re-warm ownership + VIDEO-ACTIVE GATE (B2a/B2b) — the OOM-critical piece.** A small coordinator (in the existing `Coordinator`, not a new service) owns the image↔video lane and a `videoLaneBusy: Bool` flag, set true for the duration of an LTX2 render (from `VideoGeneratorHolder` acquisition to release).
   - **Gate:** while `videoLaneBusy`, **every** pinned re-warm **and every JIT image `load`** is deferred — the request returns `503 { retry_after }` (or queues, for internal callers) rather than entering `evictIfNeeded`. This is mandatory: the image pool's budget math is **blind to LTX-2** (`ModelPool.swift` — the pool never accounts for the ~65GB video stack), so an image load admitted mid-video would pass its budget check and OOM the box. The gate, not the budget, is what prevents co-residence.
   - **Re-warm trigger (lazy, honest cost):** after a video render releases, Krea2 is **not** eagerly re-loaded. Re-warm happens **on the next image request** (or explicit `load_model`): the request path checks the pinned entry is present, and if the video handoff evicted it, calls `load(warmStandard.image, quantization: imageQuantization, pinned: true)` first. This costs **tens of seconds** (load + q8-quantize ~13.5GB `SingleStreamDiT` transformer + load ~8GB Qwen3-VL text encoder + VAE — the same `Krea2Pipeline(paths:, quantizeTransformer:8)` cost as cold startup); there is **no "seconds" fast path**. The lazy trigger means the cost is paid once, only when image is actually needed again, and never concurrently with video.

**Note — image ↔ video co-residence (hard constraint, #218).** Krea2 (~22.5GB weights) + LTX2 Sulphur2 (~65GB weight floor, activation-bound higher) cannot both be resident on 128GB during a video render. "Both always warm" is therefore **not simultaneous**: Krea2 is pinned-warm whenever the video lane is idle; a video request trips the gate, `releaseAll()` vacates the image side (including pinned Krea2), LTX2 loads, and image re-warms lazily on its next request per #6. `warmStandard` records the *intent* (what to re-warm), not a promise of co-residence or of a cheap reload.

### 3.2 LoRA registry (catalog + local, two lanes)

Extend `LoRALibrary` from "index of local files" to "registry that also describes catalog-only HF LoRAs," keeping `library.json` back-compatible.

- **Core-field optionality refactor (B3d) — required, not additive.** `LoRALibraryEntry.relativePath: String` and `sizeBytes: UInt64` are **non-optional** today, and `resolve(id)` unconditionally does `root.appendingPathComponent(entry.relativePath)`. A catalog-only row has no file, so these become **`relativePath: String?`** and **`sizeBytes: UInt64?`**. Use-site changes: `resolve(id)` throws `LoRALibraryError.notDownloaded(id)` when `relativePath == nil`; `scan()` and `importFile` set both when a file materializes; `sizeFormatted`/`primaryCompatibility` and any `entry.relativePath` reader tolerate nil. This is the one genuinely invasive change and is called out in the Phase 0 gate (§7).
- **New fields on `LoRALibraryEntry`** (all `decodeIfPresent`; old `library.json` still loads):
  - `hfRepo: String?`, `hfFilename: String?` — HF source (repo + file-within-repo) for JIT pull; `hfFilename` is the **match key** for catalog↔scan reconciliation (B3a).
  - `type: LoRAType?` — enum `identity | control | realism | style | character | nsfw`. Distinct from `modelCompatibility` (which base) and `category` (dir name).
  - `lane: LoRALane?` — `realism | style`.
  - `intentTags: [String]?` — suggestion keywords, separate from freeform `tags`.
  - `pinned: Bool?` — never disk-evicted (core Kira stack).
  - `baseModel: String?` — normalized base ("krea2","z-image","ltx"); on **scan of a pulled file it is re-derived from `LoRAScanner` and overrides the catalog seed** (B5 compatibility re-verify).
  - `local: Bool` (computed): `relativePath != nil` and file exists.
- **Catalog seed file `~/.comfybox/lora-catalog.json`** (Todd's schema, §4), merged **inside `scan()`**, not as a post-scan pass (B3b):
  - **Validation on load:** every catalog entry must satisfy `id == LoRAScanner.slugify(hfFilename)` (B3a). A mismatch is a hard config error logged and skipped — this guarantees a pulled file (whose scanned id is `slugify(filename)`) reconciles to the same row instead of spawning a duplicate.
  - **Reconciliation is by filename, never by id.** During `scan()`, a scanned file matches a catalog row when `scannedFilename == row.hfFilename` (or `slugify` equality); the row is updated in place (fills `relativePath`/`sizeBytes`/re-derived `baseModel`), preserving user-edited `tags`/`notes`/`recommendedScale`/`triggerwords` exactly as the existing preserve-user-fields logic does.
  - **Catalog preservation inside `scan()` (B3b):** `scan()`'s current "removed = existing not seen on disk" logic would delete every catalog-only row (no file yet). Change: rows with `hfRepo != nil && file absent` are **kept** as `local:false`, never counted as removed. This lives inside `scan()` so no post-scan merge can recreate rows and lose user metadata.
- **Seed content** (from the known Krea-2 HF LoRAs):
  - *Realism lane* (base `krea2`, pinned): `KNPV4.1` (identity), `Filipina_Pinay_Women` (identity/realism, rec scale ~0.35, **required trigger "Pinay"**), `krea2_innie_vagina` (nsfw anatomy). These are the "Kira" stack.
  - *Style lane* (base `krea2`): the `ilkerzgi/fal-Krea-2-Style-LoRAs` collection (decoupage and other styles) + any other style LoRAs, `type:style`, `lane:style`, `pinned:false` (JIT + evictable).

### 3.3 JIT LoRA download → vault cache → LRU evict (core pinned)

- **Pull path (B3e — repo-prefixed filename).** New `POST /v1/loras/pull {id | hf_repo, filename}` → resolve the registry row → `LoRAWeightLoader.resolveSource(.huggingFace(repo, filename))` (existing) → the safetensors lands in the HF cache → **copy it into `root/vault/` under a repo-prefixed name `{slugify(repo)}__{filename}`**. `importFile` dedupes by filename only, and the ilkerzgi style collection has generic names (`decoupage.safetensors`, …) that would collide across repos; the prefix makes the on-disk name unique per source. The catalog row's `hfFilename` stays the bare filename (match key); the scanned `filename`/`relativePath` carry the prefixed name. Post-copy the row flips `local:true` with `relativePath` set. Idempotent: an already-local row returns immediately.
- **Auto-pull on use.** Add `stageCatalogLoras(in:)` alongside the existing `stageNearlineLoras(in:)` pre-swap hook — it pulls any referenced catalog-only (HF) LoRA before the swap applies, so `swap_loras` / a resolved preset can name a not-yet-downloaded LoRA and it materializes transparently.
- **Disk budget + LRU eviction — dedicated `VaultCacheTracker` (B1), NOT Nearline.** Nearline cannot be reused: its `scan()` rebuilds `state.items` from attached-storage **roots only** (`NearlineLibrary.swift:144` `state.items = found`), so an HF-registered item with no root file is silently dropped on the next scan (loses `lastUsedAt`/`stagedPath` → untracked orphan); its `stage()` throws `sourceMissing` without an attached `path`; and it stages into `~/.comfybox/loras`, a different directory than our `root/vault/` pulls. Instead, a small dedicated tracker:
  - **`VaultCacheTracker`** (`LoRA/VaultCacheTracker.swift`), state `~/.comfybox/lora-cache.json`: `{ items: [{ filename, sizeBytes, lastUsedAt, pinned }], cacheLimitGB }`. Same NSLock + atomic-write house style as `PresetStore`/`NearlineLibrary`, but **never rebuilt by a filesystem walk** — items are added on pull, `lastUsedAt` bumped on every apply, removed only on eviction. It budgets exactly `root/vault/`.
  - **`loraCacheLimitGB`** default 100. On over-budget, evict LRU **non-pinned** items: delete the file in `root/vault/`, then **on the matching `LoRALibraryEntry` clear `relativePath` (→ nil) and `sizeBytes` (→ nil), keeping the row (B3c)** — `local:false`, re-pullable, no dead path. Pinned items (Kira stack) are never evicted; user-placed files (never added to the tracker) are never touched.
- **Offline/known-good.** `POST /v1/loras/pin` forces pull + `pinned:true` in both the entry and the tracker. The Kira stack is pinned + pre-pulled on first run so it is always local even if HF later goes unreachable.

### 3.4 Prompt-based suggestion engine

New module `LoRA/LoRASuggestionEngine.swift` — pure, deterministic, testable (no I/O; takes a registry snapshot + request, returns a ranked plan).

**Input:** `{ prompt, negativePrompt?, lane?, hasControlImage: Bool, baseModel?, contentMode? }` (contentMode = neutral/banana/avocado from the daemon).
**Output:** `SuggestionPlan { presetId?, model, loras:[{id, scale, triggersToInject:[String]}], injectedKeywords:[String], rationale:[String] }` — directly mappable to an `ImagePreset`/`ResolvedPreset`.

**Content-mode ladder (B5):** the daemon's modes rank **avocado (explicit) → banana (suggestive) → apple (SFW)**, with **neutral == apple**. `min_content_mode` on a catalog row gates NSFW LoRAs: a row with `min_content_mode: "banana"` is eligible only when the active mode is banana or avocado. The engine compares against this exact ordering (not a made-up one).

**Ranking rules (first match wins per slot; combine across slots):**
1. **Base/model** = `warmStandard.image` (Krea2-Turbo) unless the request/base says otherwise.
2. **Identity/Kira slot:** prompt matches `\bkira\b` or `\bpinay\b` (**word-boundary, case-insensitive** — not substring, so "shakira"/"akira"/"pinays" don't false-trigger), keyed off `intentTags` → force the realism Kira stack `KNPV4.1@1.0 + Filipina_Pinay_Women@0.35 (+ krea2_innie_vagina when active mode is banana or avocado, i.e. ≥ its `min_content_mode`)`, and **inject required trigger "Pinay"** (face-drift guard, per known-good recipe). No realism LoRA added (net-neutral/softens — known-good).
3. **Style slot:** style keywords ("decoupage","collage","watercolor","anime",… matched word-boundary against style-lane `intentTags`) → add the top-ranked matching `lane:style` LoRA(s) at `recommendedScale`, capped at N (default 2).
4. **Realism default:** photographic intent, no explicit style, no Kira → realism-lane default (identity-neutral), Krea2 base.
5. **Control slot:** `hasControlImage` → add the `type:control` LoRA compatible with the active base (e.g. Krea2 depth Control-LoRA, per `FDD-krea2-depth-controlnet.md`).
6. **Compatibility gate:** every selected LoRA must pass `modelCompatibility ∋ baseModel` (reuse `LoRALibrary.compatible(with:)` / `LoRACompatibility`). **Catalog `base_model` is unverified seed metadata** — the gate uses the value re-derived from `LoRAScanner` after the file is pulled (see below), never the raw seed. Incompatible candidates are dropped with a rationale line.
7. **Scoring** within a slot: exact `intentTag` hit > `tags` hit > `triggerwords` hit > word-boundary name hit; ties broken by `recommendedScale` presence then id. Deterministic.

**Post-pull compatibility re-verification + quarantine (B5).** When `stageCatalogLoras`/`/v1/loras/pull` materializes a file, `scan()` re-derives `modelCompatibility`/`baseModel` from the actual safetensors (`LoRAScanner.detectCompatibility`). If the scanned base contradicts the catalog seed (e.g. seed says `krea2`, file scans as `z-image`), the entry is **auto-quarantined** (`LoRALibrary.quarantine(id, reason:)`, already exists) and excluded from suggestions until a human clears it — so a wrong catalog row can never silently apply an incompatible LoRA to Krea2.

The plan is emitted as a transient preset; the caller may `POST /v1/presets` to persist it or pass it straight to `/v1/generate`.

### 3.5 Krita exposure (interface sketch)

Krita's *AI Diffusion* plugin talks to a ComfyUI-style backend; ComfyBox already has a `ComfyBridge` (`comfyBridge.route(request)` runs before the native routes). Two integration options; **recommend (B)** to keep Krita thin:

- **(A) Native-backend mode:** Krita points at ComfyBox's ComfyUI-compatible bridge; a small custom node/graph calls the new `/v1/loras/suggest` then `/v1/generate`. Heavier (needs a Krita-side graph).
- **(B) Sidecar "auto-preset" REST (recommended):** a thin Krita plugin/docker calls one endpoint and renders the result:
  ```
  POST /v1/suggest        { prompt, negative_prompt?, lane?, control_image?: base64, content_mode? }
    → 200 { model, loras:[{id,scale,triggers}], injected_keywords, resolved_preset, rationale }
  POST /v1/generate       (existing) with the resolved_preset  → image
  GET  /v1/loras?lane=…&local=…   (registry browse for a Krita dropdown)
  POST /v1/loras/pull     { id }   (Krita "download this style" button)
  ```
  Krita shows: a **prompt box**, a **lane toggle (Realism / Style)**, and a **style picker** populated from `GET /v1/loras?lane=style` (catalog + local, with a cloud icon on `local:false`); on render it calls `/v1/suggest` → `/v1/generate`. Selecting a not-yet-local style triggers `/v1/loras/pull` (JIT) with a progress toast. The server owns all model/LoRA policy; Krita stays a thin client. The suggestion response is identical to what the daemon/MCP `suggest_loras` tool returns, so Krita and Bree share one brain.

---

## 4. Data model — `lora-catalog.json` (registry seed)

Human-editable seed merged **inside `scan()`** (§3.2). **Hard validation rule: `id == LoRAScanner.slugify(filename)`** (B3a) so a pulled+rescanned file reconciles to the same row (`slugify` lowercases, maps `_`/space→`-`, collapses/trims `-`, and **keeps `.`** — e.g. `KNPV4.1.safetensors → "knpv4.1"`). Reconciliation matches on **filename/`hfFilename`, never id**. Todd's schema, made concrete with valid ids:

```jsonc
{
  "version": 1,
  "loras": [
    {
      "id": "knpv4.1",                            // == slugify("KNPV4.1.safetensors")
      "hf_repo": "krea/KNP",                      // JIT source (repo)
      "filename": "KNPV4.1.safetensors",          // file within repo; the MATCH KEY
      "type": "identity",                         // identity|control|realism|style|character|nsfw
      "lane": "realism",
      "base_model": "krea2",                      // seed hint only — RE-DERIVED from file on pull
      "triggers": [],                             // activation words → triggerwords
      "intent_tags": ["kira", "face", "identity"],
      "tags": ["kira-stack"],
      "recommended_scale": 1.0,
      "size_bytes": null,                          // null until pulled/scanned
      "pinned": true,                             // never disk-evicted
      "local_path": null                          // null = catalog-only; set on pull
    },
    {
      "id": "filipina-pinay-women",               // == slugify("Filipina_Pinay_Women.safetensors")
      "hf_repo": "krea/Filipina_Pinay_Women", "filename": "Filipina_Pinay_Women.safetensors",
      "type": "identity", "lane": "realism", "base_model": "krea2",
      "triggers": ["Pinay"],                      // REQUIRED trigger (face-drift guard)
      "intent_tags": ["kira", "pinay", "filipina"],
      "recommended_scale": 0.35, "pinned": true, "size_bytes": null, "local_path": null
    },
    {
      "id": "krea2-innie-vagina",                 // == slugify("krea2_innie_vagina.safetensors")
      "hf_repo": "krea/krea2_innie_vagina", "filename": "krea2_innie_vagina.safetensors",
      "type": "nsfw", "lane": "realism", "base_model": "krea2",
      "intent_tags": ["kira", "explicit", "anatomy"], "recommended_scale": 0.8,
      "pinned": true, "min_content_mode": "banana", "size_bytes": null, "local_path": null
    },
    {
      "id": "decoupage",                          // == slugify("decoupage.safetensors")
      "hf_repo": "ilkerzgi/fal-Krea-2-Style-LoRAs", "filename": "decoupage.safetensors",
      "type": "style", "lane": "style", "base_model": "krea2",
      "intent_tags": ["decoupage", "collage", "paper"], "recommended_scale": 0.9,
      "pinned": false, "size_bytes": null, "local_path": null
    }
    // …remaining ilkerzgi styles seeded the same way
  ]
}
```

Field mapping into `LoRALibraryEntry`: `hf_repo→hfRepo`, `filename→hfFilename` (bare, match key), `type→type`, `lane→lane`, `base_model→baseModel` (seed; overridden by scan), `triggers→triggerwords`, `intent_tags→intentTags`, `recommended_scale→recommendedScale`, `pinned→pinned`. `size_bytes→sizeBytes?` and `local_path→relativePath?` are **both nil until pulled** (the on-disk `filename`/`relativePath` carry the repo-prefixed name `{slugify(repo)}__{filename}`, B3e; `hfFilename` stays bare). `min_content_mode` is a new optional gate (§3.4). `id == slugify(filename)` is enforced at load; violators are skipped with an error.

---

## 5. API surface (new / changed)

| Method + path | Status | Purpose |
|---|---|---|
| `POST /v1/suggest` | **new** | prompt (+ optional control image, lane, content_mode) → ranked LoRAs + resolved preset + rationale |
| `POST /v1/loras/pull` | **new** | JIT-download a catalog LoRA from HF into the library; returns the now-local entry |
| `POST /v1/loras/pin` / `POST /v1/loras/unpin` | **new** | pull+protect / release protection (pinned = never evicted) |
| `GET /v1/loras` | **changed** | add `lane`, `type`, `local`, `pinned`, `source` filters; include catalog-only entries |
| `POST /v1/loras/scan` | **changed** | after scan, merge `lora-catalog.json` (idempotent) |
| `POST /v1/model/load` | unchanged | already the JIT load path for demoted models |
| `GET /v1/model/pool` | **changed** | expose `pinned` + `idle_unload_seconds` per entry |
| `GET /v1/config` / `PUT /v1/config` | **changed** | add `warmStandard` + `loraCacheLimitGB` |
| MCP: `suggest_loras`, `pull_lora`, `pin_lora` | **new** | thin wrappers over the above, for Bree/Channels + Krita parity |

All new endpoints follow the existing snake_case DTO + `decode`/`.json` conventions in `WarmServer.swift`.

---

## 6. VRAM / disk budget & eviction policy

**VRAM (models) — `ModelPool`, budget `COMFYBOX_POOL_BUDGET_MB`. Under `warmStandard`, default raised 40960 → 65536 (B4).** With the old 40960 default, pinned Krea2 (22528) leaves only ~18.4GB, so a JIT **FIBO bf16 (22528) could never load** (`evictIfNeeded` finds no evictable non-pinned entry → `budgetExceeded`) and **Klein-9b (18432) fits only at the exact boundary**. Two coordinated fixes:
- **Raise the budget** to 65536 (64GB) when `warmStandard.enabled` (still env-overridable). 22528 pinned + up to ~43GB JIT headroom comfortably fits FIBO/Klein/Chroma alongside Krea2. (This is image-only headroom; it does **not** imply co-residence with LTX2 — the §3.1 video gate still forbids that.)
- **"Only pinned/active, nothing evictable" escape in `evictIfNeeded`:** analogous to the existing "pool is empty — allow anyway" branch, when the sole reason eviction can't free space is that all entries are pinned or active, **admit the load anyway** (single over-budget model, logged as a warning) rather than throwing `budgetExceeded`. This guarantees a demanded JIT model still loads even if the budget math is tight against the pin.
- Krea2-Turbo q8 (~22.5GB) is **pinned** → excluded from `evictIfNeeded` / `releaseLRUInactive` / `evictIdle`. Its pool key uses `quantization:"q8"` everywhere (B4 key-unify, §3.1 #2).
- Demoted image models load on demand; `evictIfNeeded` LRU-evicts non-pinned non-active entries; `evictIdle(olderThan: idleUnloadSeconds)` sheds idle ones.
- Video (LTX2) is **not** in the image pool and its ~65GB is invisible to this budget — which is exactly why the §3.1 VIDEO-ACTIVE GATE (not the budget) prevents image loads during video. A video request uses `releaseAll()` (image fully vacated), then image re-warms Krea2 **lazily on its next request** (§3.1 #6).

**Disk (LoRAs) — dedicated `VaultCacheTracker`, `loraCacheLimitGB` (default 100) (B1):**
- HF-pulled LoRAs cached in `root/vault/` under repo-prefixed names; tracked in `~/.comfybox/lora-cache.json` with `lastUsedAt` (never rebuilt by a filesystem walk — the flaw that made Nearline unusable here).
- Over budget → evict LRU **non-pinned** files; the file is deleted and the registry row is **demoted** (`relativePath`→nil, `sizeBytes`→nil, `local:false`), never removed → re-pullable, no dead path (B3c).
- Kira stack (`pinned:true`) never evicted; user-placed files (never added to the tracker) never touched.

**Disk (HF model snapshots):** unchanged HF cache under `HF_HUB_CACHE`; out of scope for automatic eviction in v1 (manual + Nearline handles the big attached-storage catalog). Flag for a future `modelCacheLimitGB`.

---

## 7. Migration / rollout (incremental, opt-in, default = today)

1. **Phase 0 — registry fields + `relativePath`/`sizeBytes` optionality refactor + catalog merge (B5 — NOT "no behavior change").** Add the new `LoRALibraryEntry` fields **and** the core-field optionality refactor (§3.2 B3d), plus the scan-internal catalog merge. This is invasive: once a catalog exists, `entry(for:)`/`compatible(with:)` can return a catalog row that today would 404, and `resolve()` on it throws `notDownloaded` ("pull first") instead of a path. **Gate:** with **no `lora-catalog.json` present**, `/v1/loras`, `entry(for:)`, `compatible(with:)`, and every existing swap/resolve path must be **byte-identical to today** (regression-tested). Behavior differences are permitted **only** when a catalog is seeded.
2. **Phase 1 — JIT pull + LRU cache.** `/v1/loras/pull`, `stageCatalogLoras`, `loraCacheLimitGB`. Pin + pre-pull the Kira stack on first run. Verify pull→apply→evict→re-pull round-trips; pinned never evicted.
3. **Phase 2 — suggestion engine + `/v1/suggest` + MCP.** Pure module + endpoint; **off by default** (callers opt in). Presets/generate unchanged when unused.
4. **Phase 3 — warm standard (`warmStandard.enabled`, default false).** Pinning + idle-unload. Flip to true only after Phase 0–2 soak. Default-off means startup stays single-model until Todd flips it.
5. **Phase 4 — Krita sidecar** (option B) against the now-stable `/v1/suggest` + `/v1/loras`.

Each phase is independently revertible (config flag or absent catalog). No phase changes the default generation output until `warmStandard.enabled` / suggestion are explicitly turned on.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Image+video can't co-reside on 128GB (#218); pool budget is LTX-blind → OOM if an image load slips in during video.** | The §3.1 **VIDEO-ACTIVE GATE** (`videoLaneBusy`) defers all pinned re-warm + JIT image loads until video is idle — the gate, not the budget, enforces exclusivity. `releaseAll()` vacates image for LTX2; Krea2 re-warms **lazily on the next image request** (tens of seconds, honestly costed — no "seconds" claim). |
| **Budget headroom under pinning (B4):** pinned Krea2 vs a JIT FIBO/Klein. | Default budget raised to 64GB under `warmStandard`; plus a "only pinned/active → admit anyway" escape in `evictIfNeeded` so a demanded JIT model never dead-locks on the pin. |
| **Pool-key drift (B4):** `registerExisting` pinned `nil`-quant vs `load` `q8`-quant → duplicate/dead pin. | `registerExisting` takes the real quantization; `warmStandard.imageQuantization` threaded through startup + every re-warm → one identical key. |
| **HF unreachable / offline** when a JIT LoRA is needed. | Kira stack pinned + pre-pulled on first run (always local). Suggestion engine degrades to local-only entries and emits a rationale line; `/v1/suggest` returns a `warnings[]` field. |
| **Evicting a LoRA mid-render / an in-use one.** | `lastUsedAt` bump on apply; eviction skips active-swap LoRAs (same guard as Nearline); pinned never evicted; eviction only deletes the vault copy, never the registry row. |
| **Wrong-lane / bad suggestion** (e.g. style LoRA on a photo). | Deterministic rules + compatibility gate; `rationale[]` is always returned; lane is an explicit override in `/v1/suggest`; Krita shows the chosen LoRAs before render. |
| **Style LoRAs trained on non-turbo Krea-2-Raw** degrade on q8 turbo. | Same empirical GO/NO-GO discipline as `FDD-krea2-depth-controlnet.md`: gate each style on a q8-turbo smoke test; `recommended_scale` tunable per entry; net-negative ones flagged in catalog. |
| **Scanner assumes local files** (catalog rows have none). | `local:false` rows are registry-only; `resolve(id)` on a non-local entry throws a clear "pull first" error; `scan()` skips missing files for catalog rows instead of removing them. |
| **Disk fill from many style pulls.** | `loraCacheLimitGB` LRU cap; Nearline's free-space floor pattern; pinned set is small + bounded. |
| **`releaseAll(keepPinned:)` regressions** the #218 video handoff. | Default `keepPinned:false` preserves today's vacate-everything behavior for the LTX2 path; only new callers pass `true`. Covered by a regression test. |

---

## 9. Test plan / acceptance criteria

**Unit (pure, fast):**
1. **Suggestion ranking + word boundary (B5)** — table-driven: "kira on the beach" → Kira stack + "Pinay" injected; **"shakira"/"akira" → NO Kira trigger**; "decoupage portrait" → style-lane decoupage; "realistic headshot" → realism default; control image present → control LoRA; incompatible base → dropped with rationale. Content-mode ladder: `krea2_innie_vagina` selected at banana/avocado, dropped at apple/neutral. Deterministic.
2. **Catalog merge (B3a/B3b)** — `id != slugify(filename)` is rejected at load; a pull+rescan of a catalog file yields **exactly one** row (no duplicate); merge preserves user-edited `tags`/`notes`/`recommendedScale`; catalog-only rows survive `scan()` (not counted removed).
3. **Pin protection** — `evictIfNeeded`/`evictIdle`/`VaultCacheTracker` evict never touch `pinned:true` (model + LoRA).
4. **`ModelPool` pinning + key-unify + budget escape (B4)** — pinned entry survives budget pressure and idle sweep; `unload` on pinned throws; **`registerExisting(quantization:"q8")` and `load(...,"q8")` produce the same pool key** (no duplicate Krea2); with only pinned+active present, a JIT FIBO load is **admitted via the escape** (not `budgetExceeded`); `releaseAll(keepPinned:true)` keeps the entry and preserves `activeId` iff it points at a kept entry, `releaseAll()` (default) drops everything byte-for-byte as today.
5. **`VaultCacheTracker` (B1/B3c)** — over-budget evicts LRU non-pinned; evicted row is **demoted** (`relativePath`/`sizeBytes` → nil, row kept); state is never rebuilt from a filesystem walk; pinned + user-placed files never evicted.
6. **Optionality refactor (B3d)** — `resolve()` on a `relativePath == nil` row throws `notDownloaded`; `sizeFormatted`/`compatible(with:)` tolerate nil; a legacy `library.json` (non-null fields) still round-trips.
7. **Preset mapping** — a `SuggestionPlan` round-trips through `ImagePreset`→`ResolvedPreset` with correct `loras`/`injectedKeywords`.

**Integration (guarded, may need HF / GPU — mark accordingly):**
8. **JIT pull round-trip + repo-prefix (B3e)** — pull two different-repo LoRAs with the **same bare filename** → both land in `root/vault/` under distinct `{repo}__name` names (no collision) → `local:true` → apply → evict over budget → rows demote to `local:false` → re-pull succeeds.
9. **Post-pull quarantine (B5)** — a catalog row seeded `base_model:"krea2"` whose file scans as `z-image` is **auto-quarantined** and excluded from suggestions.
10. **Warm-standard startup + VIDEO-ACTIVE GATE (B2)** — `warmStandard.enabled` warms Krea2 pinned (key `...-q8`); `model_pool` shows pinned+active. While `videoLaneBusy`, an image `load_model` / pinned re-warm returns **503/deferred**, not an OOM; after LTX2 releases, the **next** image request lazily re-warms Krea2 (measured cost logged, expected tens of seconds). A JIT Z-Image evicts on idle but never Krea2.
11. **`/v1/suggest` → `/v1/generate`** — end-to-end for the three canonical prompts (Kira / style / realism) produces on-model output (VLM rubric spot-check).

**Acceptance:** with **no catalog file**, the build is behaviorally byte-identical to today (Phase 0 crit gate); with flags on, Krea2 stays pinned-warm (single unified key), non-core models/LoRAs are JIT+evictable via `VaultCacheTracker`, image loads are provably gated off during video (no mid-render OOM), the Kira stack is always local + suggested only on word-boundary "kira"/"pinay", and `/v1/suggest` returns a resolved preset Krita can render.

---

## 10. Out of scope (v1)

- Training or fine-tuning LoRAs.
- Video-LoRA suggestion (LTX act-LoRAs) — image lanes only in v1; the engine leaves `mediaKind:video` to the existing video presets (`src/kira/video-presets.ts` server-side).
- Automatic HF **model** snapshot eviction (`modelCacheLimitGB`) — LoRAs only in v1; big models stay on Nearline/attached storage.
- Building the Krita plugin itself — this FDD sketches the interface; the plugin is a separate deliverable.
- Auto-quantization of JIT-pulled models (they load at their native/detected precision).
- Changing the #218 image↔video handoff mechanism (only parameterized, not redesigned).
