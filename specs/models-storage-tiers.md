# Models Tab: Storage Tiers (online / nearline / archive / delete)

**Status:** DRAFT rev 1 — 2026-08-04. For Codex review.
**Origin:** Todd, after the 2026-08-04 manual 203G HF-cache migration to Bolt:
"spec changes to the models tab to manage online, nearline, archive, delete
for models and loras."

## Problem

Model storage lifecycle is manual and invisible. The home SSD hit 99%
(31GB free) because model weights accumulate in hot paths (HF cache,
`~/.comfybox/loras`, `~/LocalModels`) with no tooling to see what is
load-bearing versus dormant, and no safe way to demote weight without
shell surgery. Today's migration was hand-verified (`lsof`, serve.log,
launch args) — that knowledge should live in the product, not in a
session transcript.

## Prior art (extend, don't replace)

- `NearlineLibrary.swift` + `~/.comfybox/nearline.json`: root-scanning
  catalog (kind/name/path/sizeMB, cacheLimitGB) over legacy roots
  (Seagate 22T). Becomes the catalog backbone.
- LoRA library index (168 entries, usage-tracked) + DAM sqlite.
- RenderTraceStore: per-render config lines name every LoRA/model used —
  the last-used signal.
- Asset catalog service (:7871) — models/LoRAs are assets too; storage
  tier joins the one-index-over-everything model.
- Today's mover runbook: rsync → size-verify → rm → symlink (proven on
  203G; interruption-safe per item).

## Tiers

| tier | location | loadable | latency |
|---|---|---|---|
| **online** | home SSD hot path | yes | instant |
| **nearline** | Bolt (or registered root), symlinked into the hot path | yes, transparently | first-load over TB; fine for cold/rare models |
| **archive** | Bolt archive dir, NOT symlinked | no — explicit restore required | n/a |
| **delete** | gone | — | audit-logged, type-to-confirm |

Tier is a property of the ITEM in the catalog, derived from filesystem
truth (real dir / symlink / archive-dir presence), never a parallel
database that can drift.

## Safety rules (all learned 2026-08-04, all MUST)

1. **Serving-path guard.** An item referenced by the live config, launch
   args, presets.json, or loaded ModelPool CANNOT be archived/deleted;
   demote-to-nearline shows a latency warning instead (e.g. krea2
   reloads after EVERY video eviction — 65s from SSD; nearline would tax
   every image render). Pinned items (krea2, active Gemma, seedvr2,
   preset LoRAs) surface a "in use by X" chip.
2. **Foreign trees are off-limits.** LM Studio (`~/.cache/lm-studio`),
   mlx-vlm, and any non-ComfyBox model manager's directory are shown
   read-only ("managed by LM Studio") — never moved, linked, or deleted.
3. **Move = rsync → size-verify (≥95%) → rm → symlink**, executed as a
   queued job with progress, paced off-render (admission-aware, same
   idle-window discipline as engine deploys). Interruption loses nothing.
4. **Symlink integrity check** at server startup and root-mount events:
   broken link (Bolt unmounted) → item flagged `offline` in the tab +
   health warning, never a silent load failure. (Bolt symlinks already
   bit us once: Finder-alias vs symlink for ~/Pictures/ComfyBox.)
5. **Delete** requires typed-name confirmation in the UI, writes an
   audit-log line, and is only offered from archive tier (two-step:
   nothing goes from online to gone in one click).

## Signals

- **last-used**: from RenderTraceStore config lines (models, video
  LoRAs) + LoRA library usage counts (image LoRAs). Displayed per item;
  drives suggestions.
- **suggestions**: unused ≥30d + >5GB → "suggest nearline"; unused ≥90d
  → "suggest archive". Suggestions only — no auto-moves in v1.
- **free-space watermark**: header gauge for the home volume; <50GB
  turns amber with a one-click "review suggestions" flow.

## API (extends nearline routes)

- `GET /v1/models/storage` — full catalog: {id, kind (model|lora|encoder|
  upscaler), name, path, tier, sizeMB, lastUsed, pinned, pinReason,
  managedBy}.
- `POST /v1/models/storage/move` {id, tier} → 202 + job id (progress via
  existing job polling); rejects per safety rules with the reason.
- `POST /v1/models/storage/restore` {id} — archive → nearline.
- `DELETE /v1/models/storage/{id}` — archive tier only + confirm token.
- Roots config: registered move targets (Bolt Models dir default;
  Seagate legacy root read-only source).

## UI (ModelsView)

- Columns: name, kind, tier badge, size, last-used, in-use chip.
- Row actions per tier (guarded as above); bulk-select for LoRAs.
- Move-queue panel with per-item progress (GB copied / total).
- Free-space gauge + suggestions banner.
- Filter: kind, tier, unused-30d.

## Non-goals (v1)

- Auto-tiering / eviction policies (suggestions only).
- Managing LM Studio or other foreign trees (read-only visibility).
- HF re-download flows (archive keeps bytes; nothing depends on the hub).
- Dedup across roots.

## Build order

1. Server: catalog assembly (filesystem truth + trace/library signals) +
   GET route. Pure, testable.
2. Server: move job (productize the 2026-08-04 mover) + guards.
3. Desktop: ModelsView columns/badges/actions + move queue.
4. Suggestions + watermark polish.

Estimate: P1–P2 ≈ 2–3 days server, P3–P4 ≈ 1–2 days desktop.
