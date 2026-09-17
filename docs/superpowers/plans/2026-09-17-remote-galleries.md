# Remote Galleries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configurable remote galleries (a folder on a portable drive, and an Immich server) that selected assets are MOVED to, with full metadata, hidden when the remote is absent.

**Architecture:** Desktop-side. A `RemoteGalleryConfig` list in `desktop-config.json` plus a keychain entry per Immich remote; a `RemoteGalleryRegistry` that resolves reachability; a `RemoteGalleryTransfer` pipeline that copies, verifies, relocates the catalog row and only then deletes the local file, journalled in a new `asset_transfers` table so every step is re-runnable; an `AssetMediaResolver` per remote kind so bytes come from `file://`, Immich, or the engine as appropriate.

**Tech Stack:** Swift 6, SwiftUI, SQLite3 C API, swift-testing (`import Testing`), `CryptoKit` for digests, `URLSession` for Immich.

**Spec:** `docs/FDD-remote-galleries.md` (reviewed by Codex 2026-09-17; §8 records what changed).

## Global Constraints

- **Never delete a local file before the catalog points at a verified remote copy.** Order is copy → verify digest → relocate row → delete local, journalled at each step.
- **A move is never `deleteAsset`.** That drops FTS rows, collections, folder mappings and lineage edges. Use the dedicated relocate transaction.
- `assets.absolute_path` stays `NOT NULL UNIQUE` and keeps the last known local path. Storage state lives in new columns.
- Migrations are additive and idempotent; an older build must still open the database.
- Secrets never enter `desktop-config.json`; Immich keys go to the login keychain.
- Tests first, and each test seen failing before its implementation.
- `swift build` / `xcodebuild` one at a time (`comfybox-metallib-clean-build`).

---

### Task R1: Catalog cleanup — deleted vs unattached  ✅ DONE

**Files:** `Sources/ComfyBoxDesktop/DAM/DAMStore.swift`, `Tests/ComfyBoxDesktopTests/DAMStoreMissingFilesTests.swift`

- [x] `DAMStore.classifyMissingFile(path:mountedVolumeRoots:) -> MissingFileClass` (`.orphan` / `.unattached`), whole-component volume match.
- [x] `scanMissingFiles() -> MissingFileReport` (orphans, unattached), excluding secured and hosted-elsewhere rows.
- [x] `purgeMissing(ids:) -> Int` — reviewed purge, not bound by the unattended circuit breaker, refuses rows whose file exists.
- [x] `vacuumStaleLocations() -> Int` — drops `asset_locations` rows with no asset.
- [x] `pruneOrphans` no longer counts unattached rows toward its ceiling.
- [x] 8 tests, all passing.

### Task R2: Maintenance UI for the purge

**Files:**
- Modify: `Sources/ComfyBoxDesktop/DAM/GalleryMaintenance.swift` — add `scanMissingFiles()` / `purgeMissing(ids:)` / `vacuumStaleLocations()` pass-throughs beside the thumbnail-orphan pair.
- Modify: `Sources/ComfyBoxDesktop/Views/GalleryMaintenanceView.swift` — a "Missing files" section: scan button, count, a list of filenames and paths, "Purge n rows" (destructive, confirmed), and a separate line for unattached rows that says which volume is absent and offers nothing.
- Test: `Tests/ComfyBoxDesktopTests/GalleryMaintenanceMissingFilesTests.swift`

- [ ] **Step 1:** Write the failing test: a maintenance instance over a temp store reports the same split as the store, and `purgeMissing` deletes only the orphans.
- [ ] **Step 2:** Run it, see it fail (no such methods).
- [ ] **Step 3:** Add the pass-throughs; run again.
- [ ] **Step 4:** Wire the view section (no test — the repo has no SwiftUI harness; keep view logic in pure statics if any appears).
- [ ] **Step 5:** Commit.

### Task R3: `RemoteGalleryConfig` + settings

**Files:**
- Create: `Sources/ComfyBoxDesktop/RemoteGallery/RemoteGalleryConfig.swift` — the struct from FDD §3.1, `Codable`, plus `locationHost` (`"remote:<id>"`) and `galleryRoot` (`<rootPath>/ComfyBoxGallery`).
- Modify: `Sources/ComfyBoxDesktop/Views/SettingsView.swift` — `remoteGalleries: [RemoteGalleryConfig]?` on `DesktopSettings` (pattern: `archiveRoots` `:90`, `watchedServices` `:73`), a Remote Galleries section, add/edit/remove sheets.
- Create: `Sources/ComfyBoxDesktop/RemoteGallery/RemoteGalleryKeychain.swift` — Immich key storage.
- Test: `Tests/ComfyBoxDesktopTests/RemoteGalleryConfigTests.swift`

- [ ] Round-trip encode/decode including an absent list (old config file), `locationHost` format, `galleryRoot` composition, and that a `.folder` config with no `rootPath` is rejected by the validator.
- [ ] Keychain read/write/delete round trip, and that no key text ever reaches `DesktopSettings`.

### Task R4: Folder gallery format

**Files:**
- Create: `Sources/ComfyBoxDesktop/RemoteGallery/FolderGalleryIndex.swift` — `GalleryIndex {schemaVersion, id, name, createdAt, assets: [IndexEntry]}`, atomic write (temp + rename), read, append, and `create(at:name:)` which lays out `media/`, `sidecars/`, `thumbnails/`.
- Create: `Sources/ComfyBoxDesktop/RemoteGallery/RemoteSidecar.swift` — full `CatalogAsset` + recipe + `sensitive` flag.
- Test: `Tests/ComfyBoxDesktopTests/FolderGalleryIndexTests.swift`

- [ ] Create in a temp dir, append two entries, reload, expect both; an interrupted write (temp left behind) never corrupts the index; a duplicate filename gets a numeric suffix.

### Task R5: Transfer journal + relocate

**Files:**
- Modify: `Sources/ComfyBoxCatalog/CatalogSchema.swift` — additive migration: `assets.storage_state`, `assets.primary_host`, table `asset_transfers`.
- Modify: `Sources/ComfyBoxCatalog/CatalogStore.swift` — `relocateAsset(id:host:path:)`, `transferJournal` read/write, `pendingTransfers()`.
- Test: `Tests/ComfyBoxCatalogTests/CatalogRelocateTests.swift`

- [ ] Relocating keeps FTS rows, collections, folder mappings and edges (assert each), sets `storage_state`/`primary_host`, writes exactly one location row, and leaves `absolute_path` intact.
- [ ] Journal transitions and `pendingTransfers()` filtering; an old database without the columns migrates on open.

### Task R6: The send pipeline

**Files:**
- Create: `Sources/ComfyBoxDesktop/RemoteGallery/RemoteGalleryTransfer.swift` — `send(assets:to:progress:)` implementing FDD §3.3 exactly, `recoverPending()` for launch.
- Modify: `Sources/ComfyBoxDesktop/DAM/AssetIngestor.swift` — per-path transfer lock, `knownPaths` removal, re-adoption of a `remote` row whose file reappears.
- Test: `Tests/ComfyBoxDesktopTests/RemoteGalleryTransferTests.swift`

- [ ] A successful send moves the file, writes sidecar and thumbnail, relocates the row, and deletes the local copy.
- [ ] **A digest mismatch leaves the local file in place** and journals `failed`.
- [ ] Recovery from each journal state does the right thing (three tests).
- [ ] A secured asset leaves the vault, clears `secured_assets`, and its sidecar says `sensitive: true`.
- [ ] The ingestor re-adopts rather than duplicating when a moved file reappears.

### Task R7: Reachability, resolver, and the tab

**Files:**
- Create: `Sources/ComfyBoxDesktop/RemoteGallery/RemoteGalleryRegistry.swift` — status per remote, volume-UUID matching, Immich ping, timer + focus refresh.
- Create: `Sources/ComfyBoxDesktop/DAM/AssetMediaResolver.swift` — folder / Immich / engine implementations.
- Modify: `Sources/ComfyBoxDesktop/DAM/CatalogBrowser.swift` — use the resolver; add absent-remote ids to `hiddenAssetIDs`.
- Modify: `Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift` — the Remote Gallery tab scopes the gallery to remotes with a picker.
- Test: `Tests/ComfyBoxDesktopTests/RemoteGalleryRegistryTests.swift`, `AssetMediaResolverTests.swift`

- [ ] A folder remote is reachable only when its volume is mounted; an absent remote's assets are hidden; a resolver returns `file://` for folder, Immich URLs for Immich, the engine route for legacy server trees.

### Task R8: Immich client

**Files:**
- Create: `Sources/ComfyBoxDesktop/RemoteGallery/ImmichClient.swift` — version check, upload (SHA-1 `x-immich-checksum`, `sidecarData`), album create/add (`PUT /api/albums/{id}/assets` with `{ids:[…]}`), verify by `GET /api/assets/{id}`.
- Test: `Tests/ComfyBoxDesktopTests/ImmichClientTests.swift` (stubbed `URLProtocol`, no live server)

- [ ] Upload request shape; `200` duplicate treated as present; album add payload; a major-version mismatch refuses; verification failure aborts before any local delete.

### Task R9: Gallery UI — "Send to Remote"

**Files:**
- Modify: `Sources/ComfyBoxDesktop/Views/GalleryView.swift` — toolbar action beside Archive (`:544`) and a context-menu item using the established selection idiom (`isSelectMode && selectedIds.contains(asset.id) ? selectedAssetsList : [asset]`), a destination picker of reachable remotes, a confirmation that says the local copy will be deleted, and a progress sheet.
- Test: `Tests/ComfyBoxDesktopTests/GalleryViewSendToRemoteTests.swift` — pure statics only (no SwiftUI harness in this repo): destination list filtering, the confirmation string, and the selection-to-send mapping.

- [ ] Only reachable remotes are offered; the confirmation names the count and the remote; sending is refused while a transfer for the same asset is pending.

## Self-review notes

- Spec coverage: §3.1 → R3, §3.2 → R4, §3.3 → R5+R6, §3.4 → R8, §3.5 → R7, §3.6 → R1+R2, §1's "select assets and send" → R9.
- Types referenced across tasks: `RemoteGalleryConfig` (R3) is consumed by R6, R7, R8, R9; `FolderGalleryIndex` (R4) by R6; `relocateAsset` (R5) by R6; `RemoteGalleryRegistry` (R7) by R9.
- Open questions from the FDD (Immich album name, sends during a render, whether the tab also lists Kira/Bree server trees) do not block R1–R6.
