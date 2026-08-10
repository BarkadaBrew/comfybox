# Models Tab: Storage Tiers (online / nearline / archive / delete)

**Status:** DRAFT rev 2 — 2026-08-04. Rev 1 findings from Codex review
(reviews/codex-specs-review-2026-08-04.md) ALL folded: 4 blockers, 8 majors,
1 minor. Build-ready pending Todd sign-off.
**Origin:** Todd, after the 2026-08-04 manual 203G HF-cache migration:
"spec changes to the models tab to manage online, nearline, archive, delete
for models and loras."

## Problem

Model storage lifecycle is manual and invisible. The home SSD hit 99%
because weights accumulate in hot paths with no tooling for load-bearing
vs dormant, and no safe demotion path. Today's migration was hand-verified;
that knowledge belongs in the product.

## Items and identity (Codex #3, #6 — BLOCKER/MAJOR)

The unit of management is a **logical item**, never a bare file path:

- **Item kinds:** `hfModel` (the whole `models--Org--Name` tree, including
  `snapshots/` and its `blobs/` — HF snapshots symlink into blobs, so the
  package moves as ONE unit and shared-blob references are never broken),
  `localModel` (a plain directory package, e.g. `~/LocalModels/*`),
  `lora` (single .safetensors), `component` (encoder/upscaler file).
- **Identity:** stable logical ID = `{manager, packageName, contentId}`
  where contentId is a cheap content fingerprint (total size + newest mtime
  + per-file manifest hash of names/sizes; full hashing optional). Bare
  filenames are display only — duplicate names across roots are distinct
  items with distinct placements.
- **Placements:** an item has 1..n placements `{root, path, role}`. During
  a move it legitimately has two. Tier is DERIVED per rules below, with
  placement precedence (hot-path real dir > hot-path symlink > archive dir),
  and every catalog response carries a `catalogRevision` for optimistic
  concurrency.
- **Reconciliation ownership:** this catalog is the single writer for
  model/LoRA storage state; the asset-catalog service (:7871) consumes it
  read-only. The legacy `NearlineLibrary` scan is absorbed (below).

## Tiers and availability (Codex #7, #8 — MAJOR)

| tier | definition | loadable |
|---|---|---|
| online | real files in a hot path | instant |
| nearline | hot path is a symlink to a registered root | yes; first-load over the link |
| archive | placement only under a root's archive dir; no hot-path link | no — restore required |

- `delete` is a terminal ACTION, not a tier; deleted items persist as
  catalog tombstones (id, name, size, deletedAt, actor) for audit and UI
  history.
- **Availability is orthogonal to tier:** `available | offline(rootDown) |
  broken(linkTargetMissing) | inMotion(jobId)`. When a root is unmounted,
  its items keep their last-known catalog entries flagged `offline` —
  a scan NEVER erases entries because a volume is absent, and
  archive/delete/reconciliation commits are refused while any involved
  root is inaccessible. Link integrity is validated at startup, on mount
  events, AND at load time (a resolver hitting a broken link reports
  `broken`, never a silent load failure).

## Transition state machine (Codex #7 — MAJOR)

States: `stable → copying → verifying → committing → stable`, with
`failed` and `recoveryRequired` off-ramps. Allowed edges:
online↔nearline, nearline↔archive, online→archive, archive→nearline
(restore), archive→deleted. Same-item jobs are serialized (one in-flight
transition per item); cancel is honored only in `copying` (destination
partial is cleaned); `verifying`/`committing` run to completion or
`recoveryRequired`. A durable journal (`~/.comfybox/storage-journal.jsonl`)
records every phase edge; startup replays the journal to resolve
interrupted jobs (partial copy → delete partial, resume `stable`;
interrupted commit → finish or roll back per journal).

## Move protocol (Codex #1 — BLOCKER)

1. Copy to a **temporary destination** (`<dest>.partial-<jobId>`) via
   rsync; require exit 0.
2. **Manifest verification**: per-file name+size equality against the
   source manifest (not an aggregate ≥95% heuristic); optional sampled
   hashing for items above a configurable size.
3. fsync/flush, then atomic **rename** of the temp dir to the final
   destination.
4. **Atomic hot-path swap**: build the new symlink at a temp name and
   `rename(2)` over the hot path (never rm-then-ln — no window where the
   serving path is absent). Journal before and after.
5. Source removal LAST, only after the swap commits and the commit-time
   reference recheck (below) passes.

## Render/move exclusion (Codex #2 — BLOCKER)

An **item lease** shared by every model/LoRA resolver (ModelPool, LoRA
merge, upsampler load, audio VAE bind) and every storage job:

- Renders take a read lease at RESOLVE time and hold it through load.
- Storage jobs take a write intent: copying may proceed concurrently with
  read leases (loads continue from the old placement), but `committing`
  requires zero active read leases on the old placement AND a recheck
  that no queued/reserved render references the item — otherwise the
  commit WAITS (bounded) then defers the job.
- A render request arriving for an item queued for archive/delete
  CANCELS the pending job and wins (policy: renders beat housekeeping);
  the cancellation is journaled and surfaced in the move queue UI.

## Foreign-tree confinement (Codex #4 — BLOCKER)

- All mutating operations resolve **canonical, no-follow paths** and
  verify containment inside an allowlisted managed root (registered by
  path + volume UUID) as the FINAL pre-commit check, not just at planning.
- LM Studio, mlx-vlm, and other managers' trees are cataloged read-only
  (`managedBy`) and are never valid move sources, destinations, or delete
  targets, even when reached via symlink traversal.
- Archive deletion rejects symlinks outright and deletes only the
  catalog-owned real object it re-verifies at commit.

## NearlineLibrary migration (Codex #5 — MAJOR)

The legacy nearline (authoritative-on-external + staging-cache + LRU
eviction over `nearline.json`) is REPLACED: its roots become read-only
registered roots, its item list is imported as archive-tier placements,
its staging cache is drained (staged copies re-cataloged as online items
with a demotion suggestion), and `/v1/nearline/*` routes become thin
adapters over the new catalog for one release, then retire. Auto-staging
is removed — promotion is always an explicit restore.

## Pins and acknowledgements (Codex #9 — MAJOR)

- The dependency set for the serving-path guard covers: loaded pool,
  QUEUED and reserved renders, presets (including alias resolution),
  dynamic LoRA references, launch args, and env-resolved paths (e.g. the
  upsampler). Guard responses enumerate resolved dependents.
- Demoting a hot-reload-path item (krea2 reloads ~65s after every video
  eviction) requires a **server-minted expiring acknowledgement token**
  bound to {itemId, catalogRevision}; bulk/API callers cannot bypass the
  latency warning. `allowedTransitions` in the DTO carries the reasons.

## Storage jobs API (Codex #10, #11, #12 — MAJOR)

- `GET /v1/models/storage` → items: {id, kind, name, tier, availability,
  placements, sizeMB, lastUsed, pinned, pinReasons, managedBy,
  allowedTransitions, catalogRevision}.
- `POST /v1/models/storage/move` {itemId, targetTier, expectedRevision,
  idempotencyKey, ack?} → 202 {jobId} or 409 (revision) / 423 (guard).
- `GET /v1/models/storage/jobs/{id}` → durable DTO: {itemId, source,
  target, phase, bytesCopied, bytesTotal, filesDone, startedAt, error,
  cancellable, terminal}. Jobs persist across server restarts (journal).
- `POST .../restore` {itemId, ...} — archive → nearline.
- `DELETE .../{itemId}` requires a **delete token**: minted by a preflight
  call scoped to {itemId, contentId, canonicalPath, catalogRevision,
  typedName}, single-use, short-lived; all guards re-run at consumption.
  Refusals AND successes audit-logged with actor, bytes, canonical path.
- Roots config: {path, volumeUUID, role: onlineRoot|moveTarget|readOnly,
  nearlineDir, archiveDir, writable}. Capacity admission before any copy
  (destination free ≥ size + headroom; restores respect the home
  watermark); source==target and aliased-path moves rejected.

## Signals (Codex #13 — MINOR)

Render traces gain a structured `storage_item_ids` field at submit
(resolvers report the logical IDs they bound). last-used = last render
START that bound the item; usage survives relocation because it keys on
logical ID, not path. Items with no structured signal show
`unknown (pre-2026-08 traces)`, distinct from `never used`. Suggestions
(unused ≥30d & >5GB → nearline; ≥90d → archive) remain suggestion-only
in v1.

## UI (unchanged from rev 1, plus)

Tier badge + availability chip; move-queue panel driven by the durable
job DTO; suggestion banner; free-space gauge; delete = typed-name flow
backed by the token protocol.

## Non-goals (v1)

Auto-tiering; managing foreign trees; HF re-download flows; cross-root
dedup (shared-blob INTEGRITY is in scope via package-unit moves; dedup
OPTIMIZATION is not).

## Build order (revised)

1. Catalog core: item model, placements, availability, journal, lease.
2. Move job engine (protocol above) + guards + roots/capacity.
3. Routes + NearlineLibrary adapter + trace signal emission.
4. ModelsView UI.
5. Suggestions/watermark polish.

Estimate: P1–P3 ≈ 4–5 days server, P4–P5 ≈ 2 days desktop.
