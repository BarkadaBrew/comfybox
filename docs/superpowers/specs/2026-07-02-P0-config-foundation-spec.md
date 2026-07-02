# P0 — Config Foundation & Port Reconciliation (spec)

**Date:** 2026-07-02
**Parent:** `2026-07-02-imageservice-into-comfybox-decomposition.md` (P0)
**Status:** Draft spec — awaiting review before implementation
**Goal:** Establish one config model, one canonical port, and one AI-provider registry in ComfyBox so later phases (mflux-free model ports, queue, creative layer, media library) have a stable foundation — *before* any feature moves off the Node service. No feature migration in P0; this is plumbing + a migration path.

---

## Current state (measured)

**Image service (`~/Projects/coffeeshop-image-service`):**
- Data dir: `~/.coffeeshop/image-service/`
- Config: `~/.coffeeshop/image-service/config.json` — big `ImageServiceConfig` (port **7861**, presets, immich, replicate, enhancer, warmWorker, smartTabs, …)
- **Shared providers: `~/.coffeeshop/providers.json`** (env override `COFFEESHOP_PROVIDER_CONFIG`) — the AI-provider/endpoint config (Replicate, LM Studio, etc.)
- Queue state + media library index live under the data dir.

**ComfyBox:**
- Desktop `AppConfig`: `~/.comfybox/config.json` — `serverHost`, `serverPort` **7870**.
- WarmServer coded default port: **7862** (`WarmServer.swift:25`).
- Image-service `warm-worker` calls ComfyBox at **`localhost:7862`**.
- DAM: `~/.comfybox/dam.sqlite3`.

### ⚠️ Live port inconsistency (fix in P0)
Three ports are in play — **7861** (image service), **7862** (WarmServer default + what warm-worker calls), **7870** (what the desktop app expects). Desktop config says 7870 but the server defaults to 7862, so desktop↔server only works if a config file overrides the port. This must collapse to **one canonical port**.

---

## P0 scope

1. **Canonical port.** Pick one ComfyBox HTTP port and make server default + desktop default + docs agree. **Recommendation: 7870** (already the desktop externalized default; a857040). Keep 7862 accepted for one release as a compatibility alias so the image-service warm-worker and any Krita config keep working during transition; log a deprecation when it's used. Retire 7861 with the Node service.

2. **Config location + model.** Standardize on **`~/.comfybox/`** as ComfyBox's home. Extend the server side beyond the desktop `AppConfig` (host/port only) to a fuller `ComfyBoxConfig` that will absorb the fields later phases need (output dir, default model, provider registry, queue, media, immich, replicate). Add `/v1/config` GET/PUT so the UI can read/write it. Decide: **one `config.json`** with sub-sections, vs. separate files per concern. *Recommendation: one `config.json` + the separate `providers.json` (below), mirroring the image service's split so provider secrets stay isolated.*

3. **AI-provider registry** (ties to `comfybox-configurable-ai-providers`). A **`providers` section inside `~/.comfybox/config.json`** (migrated from `~/.coffeeshop/providers.json`) modeling a map of **named capabilities → endpoint config**:
   - `promptOptimization` → `{ baseUrl: "http://localhost:1234/v1", model: "dans-pe-v1.3.0-24b-heresy@8bit", apiKey?: "" }` (LM Studio).
   - `vision`, `captioning` → present in the schema but unconfigured/optional.
   - Plus the existing `replicate` block (for video). Adding a capability = a new key + a Settings row; no code fork.
   Expose `/v1/providers/status` (already an image-service route) so the UI can show which capabilities are configured/reachable.

4. **Ancillary read-only routes** the UI/consolidation need early, thin passthroughs over the above: `/v1/catalog` (models/modes), `/v1/stats`, `/v1/memory`, `/v1/audit-log`. (Enhancer route `/v1/enhance*` is P4, but its provider config lands here.)

5. **Migration.** A one-shot migrator (CLI subcommand `comfybox migrate-config` or automatic on first launch) that: reads `~/.coffeeshop/providers.json` → writes `~/.comfybox/providers.json`; maps the still-relevant `ImageServiceConfig` fields (output dir, replicate, immich, enhancer) into `ComfyBoxConfig`; leaves the old files untouched (non-destructive). Presets/characters/media are **not** migrated in P0 (their phases own that).

---

## Explicitly out of scope for P0
Dual-lane queue (P2), creative-layer stores (P3), enhancer execution + events (P4), media-library unification (P5), any model port. P0 only creates the config surfaces they will populate.

## Decisions (confirmed)
1. **Canonical port = 7870** — server default + desktop default. `7862` accepted for one release as a deprecation-logged alias (warm-worker / Krita configs). `7861` retires with the Node service.
2. **Single combined `~/.comfybox/config.json`** — one file with sub-sections, including the AI-provider registry inline (no separate `providers.json`). Migrator folds `~/.coffeeshop/providers.json` into this file's provider section.
3. **Automatic migration on first launch** — non-destructive (reads `~/.coffeeshop`, writes `~/.comfybox/config.json`, leaves originals untouched). No separate CLI command required.

## Acceptance
- Server and desktop default to the same port with no config file present; 7862 still works with a deprecation log.
- `~/.comfybox/providers.json` exists with the `promptOptimization` entry pointing at the LM Studio model; `/v1/providers/status` reports it.
- `/v1/config` round-trips (GET returns current, PUT persists, reload reflects).
- Migrator copies `providers.json` from `~/.coffeeshop` without modifying the originals.
- New unit tests: config load/default/round-trip, provider-registry parse, migrator mapping. No server/model/network needed.
