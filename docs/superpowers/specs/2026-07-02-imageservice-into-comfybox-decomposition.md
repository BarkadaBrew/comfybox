# Consolidating Coffee Shop Image Service into ComfyBox — Decomposition

**Date:** 2026-07-02
**Status:** Draft decomposition (awaiting user confirmation on two forks)
**Goal:** Move *all* functionality of `~/Projects/coffeeshop-image-service` (Node backend + Electron/React UI) into ComfyBox (Swift WarmServer + ComfyBoxDesktop SwiftUI app), then **retire the old image service**. The Swift desktop app becomes the unified front end for both the ComfyBox image engine and the "Coffee Shop" creative layer.

---

## Confirmed constraints

- **No Python, ever.** ComfyBox is self-standing Swift/MLX. It must **not** spawn `mflux-*` or any Python. Every surviving mflux capability is reimplemented natively. (See memory `comfybox-no-python`.)
- **LoRA training is dropped.** `mflux-train` retires with mflux. ComfyBox keeps native inference + LoRA *bake/apply* (already native); training is out of scope, possibly a separate offline tool someday.
- **Model priorities (confirmed).**
  - **Z-Image Turbo** — default, already native. No work.
  - **Zeta-Chroma** (art creation) — the model Todd actually wants for art. **NOT the existing `Sources/ZImage/Chroma` family** (that one is lodestone's Flux-based Chroma — "not heavy", leave as-is). Zeta-Chroma (`lodestones/Zeta-Chroma`) is built *on Z-Image*: same DiT lineage + Qwen3-4B encoder + **Flux 2 VAE** (all already native in ComfyBox), with three deltas to implement: a **DeCo head**, **fm-x0 prediction** (vs Z-Image's velocity — a scheduler mode), and a **pixel-space** variant that skips the VAE. So it's a **Z-Image *variant* port, not a new family** — small relative to the others. Caveat: it's in **early upstream pretraining (WIP, Apache 2.0)**; port once lodestone stabilizes weights/arch.
  - **Fibo, Flux2 (= FLUX.2 Klein)** — already native. **Klein = the Flux2 family** (`Flux2Pipeline` is "Flux 2 Klein", IDs `black-forest-labs/FLUX.2-klein-{4B,9B,base}`). Text-to-image ✅ and img2img ✅ (`inputImagePath` + `denoise<1.0`). **GAP: Klein inpainting (masked fill)** — no `mask` param / fill path today. **Higher priority than Chroma** (Todd's call). Contained add: reuse the Z-Image masked-denoise loop (`ZImageControlPipeline`: per-step re-noise + mask composite) on the Flux2 latent/VAE, + a `mask` input on `Flux2GenerationConfig`. Not a model port — a feature on an existing one.
  - **Existing Flux-based Chroma** — **already fully supported** (ModelPool `.chroma`, detection, CLI). Todd rarely uses it → **low priority, nothing to do.**
  - **Qwen-Image (+edit) incl. Krea-2** — the main new *family* port (Krea-2 is Qwen-based).
  - **Depth (Depth Anything) + concept/IP-adapter** — preprocessors.
  - **FLUX editing** (Kontext/Fill/Redux/in-context/CatVTON) — optional.
  - **FLUX.1** (dev/schnell/krea) — **deprioritized**; Todd rarely uses it now.
- **Sequencing = backend-first.** Grow WarmServer into a superset of the image-service API, cut the Electron backend over, then port the UI views, then delete the Electron app. The UI always has a stable target.

---

## Current state (measured)

| | Backend | UI |
|---|---|---|
| **Image service (retiring)** | `src/` Node/TS, ~9,240 LOC, HTTP on **:7861**, launchd daemon | Electron/React renderer, ~7,829 LOC, 11 views |
| **ComfyBox (target)** | `Sources/ZImage/Server` Swift WarmServer on **:7870** (native MLX, mflux-free) | `Sources/ComfyBoxDesktop` SwiftUI, 4 tabs + DAM (SQLite) |

**Key relationship to reverse:** the image service currently delegates *some* generation to ComfyBox's warm server (`src/zimage-warm-client.ts` / `warm-worker.ts` → `localhost:7862`). After consolidation ComfyBox is no longer a sub-worker — it *is* the daemon. The `warm-worker` concept disappears.

**Persistence to migrate** (image service stores under `~/.coffeeshop/` + a data dir): `providers.json`, config, queue-state, media-library index, presets, characters, projects, shot-templates, smart-tabs. ComfyBox stores config under `~/.comfybox/` and its DAM at `~/.comfybox/dam.sqlite3`.

---

## API surface delta (image-service routes → ComfyBox)

ComfyBox **already has** (some names differ): `/health`, `/v1/generate`, `/v1/queue` (bridge), `/v1/models`, `/v1/loras` (+scan/swap), `/v1/upscale`, `/v1/video/generate` + status, `/v1/model/*` pool, `/v1/styles`, `/ws`.

ComfyBox **must gain** to reach parity:

| Route(s) | Subsystem | Notes |
|---|---|---|
| `/v1/mflux`, `/mflux/{bake,save,depth,info}` | **Native tools** (no bridge) | Reframed as native ops. `bake`/`save`/`info` ComfyBox already has natively; `depth` needs a native depth model. **`train` and `upgrade` are dropped** (Python-only). |
| `/v1/characters` | **Character registry** | ComfyBox has a stub that 404s today. |
| `/v1/presets`, `/v1/presets/resolve` | **Presets** | ComfyBox desktop has local `presets.json`; server-side presets + resolve are new. |
| `/v1/projects` | **Projects** | New. |
| `/v1/shot-templates` | **Shot templates** | New. |
| `/v1/smart-tabs` (+ `/`) | **Smart tabs** | New. |
| `/v1/content-modes` | **Content modes** (neutral/banana/avocado) | Guidance boost + style variants. Telegram bot already has a content-mode notion — unify. |
| `/v1/enhance` (+`/preview`,`/status`) | **Prompt enhancer** | LLM via OpenAI-compatible endpoint. ComfyBox desktop *calls* `/v1/enhance` already but the server route doesn't exist — this closes that gap. |
| `/v1/assets` (+`/folders`,`/rescan`,`/export/immich`) | **Media library** | Overlaps ComfyBox DAM; unify onto DAM + add folders/rescan/Immich export. |
| `/v1/immich/status` | **Immich integration** | External photo server export. |
| `/v1/events` (SSE) | **Event stream** | ComfyBox uses WebSocket (`/ws`) for progress; either add SSE or migrate UI to WS. |
| `/v1/jobs` (+`/batch`,`/explore`) | **Job model** | Richer than current queue; maps onto dual-lane queue. |
| `/v1/config`, `/v1/prompt-config`, `/v1/catalog`, `/v1/stats`, `/v1/memory`, `/v1/audit-log`, `/v1/providers/status`, `/v1/uploads/image` | **Misc infra** | Config, catalog, memory guard, audit log, provider status, uploads. |
| dual-lane queue (`chat` / `hq`) | **Queue** | Replace ComfyBox's single FIFO coordinator with two lanes + pause/resume/mutate/duplicate. Careful: the FIFO actor is load-bearing and was just hardened. |

---

## Decomposition into sub-projects (backend-first order)

Each is its own spec → plan → build cycle. Ordered by dependency and risk.

**P0 — Foundation & config unification.** Reconcile `~/.coffeeshop/` and `~/.comfybox/` config models; decide canonical port (keep :7870, alias :7861 during transition); add `/v1/config`, `/v1/providers/status`, `/v1/catalog`, `/v1/stats`, `/v1/memory`, `/v1/audit-log`; data-migration script for existing presets/characters/etc.

**P1 — Native model-family ports (no Python).** Each is its own spec-sized effort; ComfyBox already has the pipeline scaffolding (families share `SafeTensorsReader`, `ModelPool`, scheduler stack — Chroma's VAE-reuse is the template). Sub-items, **in confirmed priority order**:
- **P1a — Qwen-Image (+edit) incl. Krea-2** — the main new *family* port. Qwen is only a text encoder in ComfyBox today; this adds Qwen *image generation*. Replaces `-qwen`,`-qwen-edit`.
- **P1b — Depth (Depth Anything) + concept/IP-adapter preprocessors** — replaces `-generate-depth`,`-save-depth`,`-concept*`.
- **P1c — FLUX editing** (Kontext / Fill / Redux / in-context / CatVTON) — optional; replaces `-kontext`,`-fill`,`-redux`,`-in-context*`.
- **P1-Klein-fill — Klein (Flux2) generative fill: inpaint + outpaint** — **prioritized** (Todd; key for Krita). **Outpaint = inpaint with the mask on extended canvas** — one feature, not two. The whole delivery path already exists for Z-Image and works via Krita: ComfyBridge carries `inpaintImageId`+`maskImageId` and advertises the inpaint nodes; WarmServer passes `maskImage/maskData/denoise/maskGrow/maskFeather/maskCropX,Y`; `ZImageControlPipeline` consumes them. **The only gap is Flux2/Klein has no mask path** (`Flux2GenerationConfig` has no `mask`). Work = add a `mask` input to `Flux2GenerationConfig` + port the Z-Image masked-denoise loop (per-step re-noise + composite) onto the Flux2 latent/VAE, then route Klein through the existing bridge inpaint path. Test seam quality on outpaint (mask over blank margin). Replaces `mflux-generate-fill`.
- **P1d — FLUX.1** (dev/schnell/krea) — **deprioritized** (rarely used now); port late or skip. Replaces `mflux-generate`.
- **P1e — Zeta-Chroma** (art) — **LOW PRIORITY, deferred until the model matures** (early upstream pretraining). A **Z-Image variant**, not a new family: reuses native Z-Image DiT + Qwen3-4B encoder + Flux2 VAE; new code is the DeCo head, an fm-x0 scheduler mode, and a pixel-space (VAE-less) decode path. **A prior Python/MLX port exists as reference** (`~/Projects/coffeeshop-server/scripts/zeta-chroma-mlx.py`, `chroma-generate/ZETA_CHROMA_IMPLEMENTATION_PLAN.md`) — re-port to native Swift.
- Already native, no port: **Z-Image/Turbo (default)**, existing Chroma (Flux-based, low priority), Fibo, Flux2, SeedVR2/ESRGAN upscale, quantize/save, inpaint/fill (Z-Image), ControlNet (Z-Image), info.
- **Dropped:** `mflux-train` (training), `mflux-upgrade` (Python version mgmt).

**P2 — Dual-lane queue.** Extend the WarmServer coordinator to two lanes (chat/hq) with pause/resume/cancel/mutate/duplicate and the `/v1/jobs` model. Must preserve the leaked-continuation safety the current actor has.

**P3 — Creative layer.** Character registry, presets + resolve, projects, shot-templates, smart-tabs, content modes. Mostly CRUD over JSON stores → Swift `Codable` + files (or DAM's SQLite).

**P4 — Configurable local-AI providers + prompt enhancer + events.**
- **AI provider registry (new, UI-configurable):** a set of *named capabilities* — `promptOptimization` (now), `vision` and `captioning` (future) — each independently configured to an OpenAI-compatible local endpoint (base URL, model id, api key). Current preference: a **"Dan's" model in LM Studio** for prompt optimization. Backed by `~/.coffeeshop`/`~/.comfybox` `providers.json` (image service already has `getSharedProviderConfigPath()` → `~/.coffeeshop/providers.json` — reuse/migrate). Must be extensible: adding a new capability = a new config entry + a Settings panel row, no code fork.
- **Prompt enhancer:** `/v1/enhance*` consumes the `promptOptimization` provider. ComfyBox desktop already *calls* `/v1/enhance` but the route doesn't exist server-side — this closes that gap.
- **Events:** the SSE (`/v1/events`) vs unify-on-WebSocket decision (defer to P4 spec).

**P5 — Media library unification.** Fold `/v1/assets*` onto the ComfyBox DAM; add folders, rescan, sidecar parity, Immich export + status.

**P6 — Backend cutover.** Point any remaining image-service consumers (Telegram bot, Obsidian Coffee Shop plugin, `warm-worker` callers) at ComfyBox; retire the Node daemon + launchd plist.

**P7 — UI port + UX upgrades (not a 1:1 clone).** Bring the Electron views into ComfyBoxDesktop as SwiftUI (Dashboard, Queue, Gallery, Presets, Characters, Models&LoRAs, Config, Server, plus the current Generate/Compare). Reference the React components for layout/behavior, not code. **Three areas are explicitly to be *improved*, not just ported (Todd):**
- **LoRA management** — beyond the current library list: browse/search/filter the LoRA library (uses `LoRALibrary`/`LoRAScanner`/compatibility already in-tree), trigger/adjust the multi-LoRA stack with live scales (incl. negative/slider LoRAs per the MCP scale change), see compatibility/quarantine state, import + metadata. Replaces the Electron `ModelsAndLoras` + `LoraImportDialog`.
- **Prompt management** — a real prompt workflow: history, reusable snippets/wildcards, the configurable prompt-optimizer (P4 provider) inline with before/after preview, per-preset prompt+negative, and prompt metadata surfaced from the DAM. Replaces scattered prompt fields.
- **DAM functions** — richer than today's basic Gallery: folders, robust FTS search over prompt/seed/model (fixing the current FTS-dup + annotation-loss bugs from the remediation pass), rating/favorite/tags, compare, sidecar metadata for app-generated images, Immich export. Builds on `DAMStore`.

The rest of the views are straight ports. **Note:** MfluxTools view is dropped (Python tools retired); its still-native actions (bake/save/quantize/depth) resurface inside the LoRA/model management UI.

**P8 — Electron retirement.** Delete the Electron renderer + main process; keep the old repo archived or reduce it to the mflux Python venv if that's where mflux lives.

---

## Open decisions to confirm

1. ~~mflux strategy~~ — **Resolved: native only, no Python. Training dropped.**
2. ~~"Krea2"~~ — **Resolved: Krea-2 is a Qwen-based model; folds into the Qwen-Image port (P1a).**
3. ~~Model-port priority~~ — **Resolved: Qwen-Image/Krea-2 (P1a) → depth/concept (P1b) → FLUX editing (P1c, optional) → FLUX.1 (P1d, deprioritized). Z-Image Turbo + Chroma already native.**
4. ~~AI provider config~~ — **Resolved: prompt-opt = `dans-pe-v1.3.0-24b-heresy@8bit` via LM Studio (`http://localhost:1234/v1`); vision + captioning stubbed for later.**
5. **Event transport** — add SSE `/v1/events` for compatibility, or migrate the new UI to the existing WebSocket? (Defer to P4.)
6. **Queue model** — replace the just-hardened single-FIFO actor with dual lanes, or run dual lanes *on top of* it? (Defer to P2, but flag: don't regress the continuation-safety work.)
7. **Content modes** — the ComfyBox Telegram bot and the image service each have their own content-mode notion; pick one canonical implementation.

---

## Relationship to the in-flight remediation branch

A separate multi-agent remediation is currently fixing review findings on `claude/queue-progress-telemetry` (desktop/Telegram snake_case decode bugs, server security, control-pipeline CFG cache, LoRA alpha, CI, and the ComfyBridge/Krita integration). **That work should land first** — it stabilizes exactly the surfaces (WarmServer response contract, DAM, desktop client, queue telemetry) that P0–P2 here build on. This consolidation starts from the post-remediation baseline.
