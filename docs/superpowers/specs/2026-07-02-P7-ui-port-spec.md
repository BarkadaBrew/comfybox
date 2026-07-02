# P7 — Electron → SwiftUI UI Port (spec)

**Date:** 2026-07-02
**Parent:** `2026-07-02-imageservice-into-comfybox-decomposition.md` (P7)
**Status:** Draft spec — planning the full UI port before building (Todd: "plan the full UI port first").
**Goal:** Replace the Coffee Shop image service's Electron/React UI (~7,800 LOC, 9 sidebar views) with native SwiftUI in **ComfyBoxDesktop**, so the Electron app can be retired. **Improve, don't clone** — LoRA management, prompt management, and DAM get real UX upgrades, not 1:1 ports. Build **view-by-view**, gated on backend availability (some views' backends already ship; others wait on P3/P4/P5).

---

## Design stance

Native macOS app, not a re-skinned webview. Reference `frontend-design` at implementation time. Principles:
- SwiftUI `NavigationSplitView` shell with a sidebar (mirrors the Electron sidebar), `@Observable` state via `EngineService` + `DAMStore`.
- Talks to the warm server over HTTP (`WarmServerClient`/`EngineService`), plus the **local DAM** (SQLite) for assets. No pipeline linkage.
- Match platform idioms (toolbars, inspectors, `Table`, drag-and-drop, quicklook) rather than reproducing web layouts.
- All new view logic covered by model-free Swift Testing (the existing 89-test desktop suite is the pattern; no server/model needed).

---

## View inventory & mapping

| Electron view (LOC) | Existing SwiftUI | Action | Backend | Gate |
|---|---|---|---|---|
| **Dashboard** (496) | — (missing) | **New** — status, queue summary, live progress (uses the new `current_job_id`/`progress_percent`), recent renders, provider/health tiles | `/health`, `/v1/models`, DAM | **Now** |
| **Gallery** (1429) | `GalleryView`, `AssetDetailView`, `ComparisonGridView` | **Upgrade** (DAM area) | DAM (local) | **Now** |
| **Queue** (149) | `QueuePanel` | **Upgrade** — determinate progress, cancel/reorder | `/v1/queue`, `/health` | **Now** (dual-lane after P2) |
| **Presets** (264) | `PresetView`, `PresetManager`, `PresetForm` | **Upgrade** | local + `/v1/presets` | **Now** (local); server presets after **P3** |
| **Characters** (327) | `CharacterLibraryView` | **Upgrade** | `/v1/characters` | **P3** (route absent today) |
| **Models & LoRAs** (591) | `ModelSelector`, `LoRAPicker` | **Upgrade** (LoRA area) + fold in native bake/save/depth | `/v1/models`, `/v1/loras*` | **Now** (bake/depth after those native ops land) |
| **mflux Tools** (631) | — | **Drop** — Python retired; native bake/save/depth resurface inside Models & LoRAs | — | — |
| **Server** (423) | — (missing) | **New** — model pool load/activate/unload, server status, shutdown, logs | `/v1/model/*`, `/health`, `/v1/shutdown` | **Now** |
| **Config** (1146) | `SettingsView` | **Upgrade** — full config incl. **AI-provider registry** editor | `/v1/config`, `/v1/providers/status` | **Now** (routes deployed in P0) |

Components → SwiftUI equivalents: `JobSubmitForm`→`GenerationView` (exists), `JobCard`/`StatusBadge`→list cells, `LoraImportDialog`→LoRA import sheet, `BakeDialog`→bake sheet (native op), `PresetForm`→preset editor, `ImmichStatus`→DAM export (P5), `Sidebar`→`NavigationSplitView` sidebar.

---

## The three "improve, don't clone" areas

**1. LoRA management** (upgrade `LoRAPicker`/`ModelSelector` → a real manager).
- Browse/search/filter the LoRA library (backed by in-tree `LoRALibrary`/`LoRAScanner`/`LoRACompatibility`).
- Multi-LoRA stack with live scale sliders incl. **negative/slider LoRAs** (per the MCP scale change), compatibility + quarantine badges, trigger words, import + metadata edit.
- Fold in native **bake/save/quantize/depth** actions (the surviving mflux-tools functions).

**2. Prompt management** (new surface, not just a text field).
- Prompt history, reusable snippets/wildcards, per-preset prompt+negative.
- Inline **prompt optimization** via the configurable provider (P0 registry → P4 `/v1/enhance`) with before/after preview.
- Prompt metadata surfaced from the DAM (reuse a render's exact prompt/seed).

**3. DAM functions** (upgrade `GalleryView`/`AssetDetailView`).
- Folders, robust FTS search over prompt/seed/model, rating/favorite/tags, compare, sidecar metadata for app-generated images (the remediation already fixed the FTS-dup + annotation-loss bugs).
- Immich export + status (**P5**).

---

## Build order (interleaved with backend phases)

**UI-Now (backends already shipped):** Config/provider-registry Settings panel → Server view → Dashboard → Queue upgrade → Models&LoRAs (LoRA manager) → Gallery/DAM upgrade → Presets (local).
- **First increment: the AI-provider Settings panel** — small, exercises the deployed `/v1/config` + `/v1/providers/status`, establishes the port pattern, closes P0's deferred UI item.

**UI-Gated (wait on backend phase):**
- Characters → after **P3**.
- Prompt-optimization inline UI → after **P4** (`/v1/enhance`); the provider *config* is already editable now.
- DAM Immich export, media `/v1/assets*` unification → after **P5**.
- Dual-lane Queue affordances → after **P2**.

**Retire:** delete the Electron renderer + main process once the UI-Now set + P3/P4/P5-gated views reach parity (**P8**).

---

## Shared infrastructure to add
- `EngineService`: config get/put (`/v1/config`), provider status, model-pool ops, server control — extend the existing client.
- A small `ConfigStore` (`@Observable`) wrapping `ComfyBoxServerConfig` fetched from `/v1/config` for the Settings UI.
- Sidebar-driven `NavigationSplitView` shell replacing the current 4-tab layout; preserve existing views as destinations.

## Acceptance (per view)
- Renders from live server/DAM data; no hardcoded ports/paths (uses `AppConfig`/`/v1/config`).
- Errors surfaced to the UI (not swallowed — the remediation's lesson).
- Model-free Swift Testing for view models/state; no server/weights in tests.
- The Electron view it replaces is verified redundant before P8 deletion.

## Decisions (confirmed)
1. **Shell:** full sidebar `NavigationSplitView` (matches Electron, scales to 9+ destinations).
2. **Prompt management:** inspector panel on Generate + a history sheet.
3. **First increment:** the AI-provider Settings panel.

## Governing principle: parity proof, then expansion (Todd)
For **every** ported view: first build to **parity** with the Electron original and prove it (the SwiftUI view does what the web view did against live data), **then** layer the "improve, don't clone" expansions. Parity is the checkpoint that de-risks each port before investing in enhancements; don't start an Electron view's deletion (P8) until its SwiftUI replacement has passed parity.
