# FDD — Remote galleries: portable drives and Immich

**Status:** draft for review · **Owner:** ComfyBox Desktop · **Date:** 2026-09-17

## 1. Requirement

Todd, 2026-09-17:

> In the Desktop app there is a tab for remote gallery. I would like to wire through configuration where the remote galleries are and I want to create a remote gallery on a USB SSD thumb drive. One is attached right now. That would be a gallery that I can send images and videos to but not as an archive. Sensitive content, private content or content I want to make portable would go there. This would require the creation of a gallery on that drive and the ability to configure remote gallery locations where ever they might live and the ability to select assets from the main gallery and send them to remote.

> in addition we can use the immich store on the server as an additional remote gallery

> the current gallery is corrupted and wont fix the DB of the deleted entries, it assumes it is unattached storage that the thumbnail cant find and wont clean itself

Decisions taken with Todd the same day:

| Question | Decision |
|---|---|
| First drive | `/Volumes/Vault` — USB, case-sensitive APFS, 2 TB, 686 GB free |
| What happens to the local copy | **Move.** The remote becomes the only copy |
| What travels with the asset | Everything: the media file plus its recipe, prompt and catalog metadata |
| Drive unplugged | The remote and its assets are **hidden** until it is back |

A remote gallery is **not an archive**. `.cbarchive` bundles (`GalleryArchiver`) stay what they are: cold storage, restored as a unit. A remote gallery holds plain media files that any machine can browse, and the Desktop shows them as ordinary gallery assets while the remote is reachable.

## 2. Current state

- **The tab renders the main gallery.** `AppTab.remoteGallery` (`ComfyBoxDesktopApp.swift:83`) resolves to the same `galleryDetail` (`:488`) as `AppTab.gallery`. `RemoteGalleryView` / `RemoteGalleryService` were deleted in the 2026-07-31 "one gallery" convergence. There is no remote configuration anywhere in the app.
- **"Remote" today means a catalog row whose location host is not `mac`.** `CatalogBrowser.resolve` (`DAM/CatalogBrowser.swift:128`) splits a page into `localPaths` / `remotePaths` using `CatalogStore.locations(of:scope:)`, and streams remote bytes from the engine (`GET /v1/gallery/file?path=`).
- **The catalog is one SQLite file**, `~/.comfybox/dam.sqlite3`, shared by the DAM and `ComfyBoxCatalog`. `AssetLocation {host, path, mtime}` (`CatalogModels.swift:60`) is the location record.
- **The live database is in the state Todd describes.** 411 asset rows, 188 of them naming a file that no longer exists, **zero** location rows for any live asset, and 885 location rows left over from an earlier database (`kira` 883, `mac` 2). `DAMStore.pruneOrphans` refuses the sweep because its circuit breaker (`pruneCircuitBreakerFraction = 0.05`, floor 5 — `DAMStore.swift:229`) allows at most 20 deletions out of 411. That breaker is right for an unattended sweep on every gallery load, and wrong as the only cleanup path there is.

## 3. Design

### 3.1 Configured remotes (`RemoteGalleryConfig`)

A new list in the desktop-only settings document `~/.comfybox/desktop-config.json`, following the `archiveRoots` / `watchedServices` pattern (`SettingsView.swift:90`, `:73`):

```swift
struct RemoteGalleryConfig: Codable, Identifiable, Hashable {
    let id: String              // stable uuid; the location host is "remote:<id>"
    var name: String            // "Vault SSD", "Immich"
    var kind: Kind              // .folder | .immich
    var enabled: Bool
    // .folder
    var rootPath: String?       // "/Volumes/Vault" — the gallery lives at <rootPath>/ComfyBoxGallery
    var volumeUUID: String?     // identifies the drive across mount points/renames
    // .immich
    var baseURL: String?        // "http://10.0.100.232:2283"
    var albumName: String?      // album the sends land in; created on first send
    var albumID: String?        // cached after creation
}
```

Secrets never enter this file: an Immich API key lives in the login keychain under service `com.barkadabrew.comfybox`, account `remote-gallery-<id>` (`KeychainStore`, with the data-protection caveat in `comfybox-keychain-durable-fix`). Settings gains a **Remote Galleries** section: a list, an add sheet per kind (folder picker for `.folder`, host/album/key form for `.immich`), edit, remove. Removing a remote never touches its contents.

### 3.2 On-disk layout of a folder remote

Self-describing and portable, so the drive is useful on a machine with no ComfyBox:

```
<rootPath>/ComfyBoxGallery/
  gallery.json                       # index: schemaVersion, id, name, createdAt, assets[]
  media/2026/09/<filename>           # the original file, original name
  sidecars/2026/09/<filename>.json   # the full catalog row + recipe
  thumbnails/<assetID>.jpg
```

`gallery.json` is written atomically (temp + rename). `media/` keeps the original filename; a collision takes a numeric suffix via the existing `GalleryView.uniqueDestination(inDirectory:filename:)` helper. The sidecar carries every field of `CatalogAsset` plus the prompt, recipe and lineage edges, so a send is reversible by hand and a future "bring back" task has everything it needs.

### 3.3 Sending (a move, ordered so a crash can never strand an asset)

Codex review, 2026-09-17: the first draft deleted the local file before rewriting the catalog, so a crash in between left a row pointing at nothing. It also assumed `asset_locations` could carry primary ownership, which it cannot: `assets.absolute_path` is `NOT NULL UNIQUE` (`CatalogSchema.swift:107`) and `CatalogBrowser` resolves it first (`CatalogBrowser.swift:172`). Both are fixed here.

**Schema additions** (catalog migration, additive):

- `assets.storage_state TEXT` — `local` (default, today's behaviour), `remote`, `missing`.
- `assets.primary_host TEXT` — `mac`, or `remote:<id>` once moved. `absolute_path` keeps the **last known local path** and is never nulled, so the unique index and every existing query keep working.
- `asset_transfers` — the durable journal: `(asset_id, remote_id, state, remote_path, sha256, started_at, updated_at)` with state in `copying`, `verified`, `relocated`, `done`, `failed`.

**Per asset, in this order:**

1. Journal `copying`. Compute the source SHA-256 (and, for Immich, SHA-1 for its duplicate header).
2. Copy media to `media/…/<name>.part`, then the sidecar and thumbnail.
3. Re-read the written file, compare digests. Mismatch → delete the `.part`, journal `failed`, **local copy untouched**.
4. Rename `.part` to final, append to `gallery.json`. Journal `verified`.
5. **`CatalogStore.relocateAsset(id:host:path:)`** — a dedicated transaction that writes `storage_state`, `primary_host` and the `asset_locations` row and nothing else. It must never call `deleteAsset`, which would take FTS rows, collections, folder mappings and lineage edges with it (`DAMStore.swift:605`). Journal `relocated`.
6. Only now delete the local media, its JSON sidecar and its cached thumbnail. Journal `done`.

**Recovery on launch** reads the journal: `copying` → delete the stray `.part`, drop the entry, local copy is authoritative; `verified` → redo step 5 then 6; `relocated` → redo step 6. Every state is safe to re-run.

Deletion is a real `removeItem`, not a move to Trash, and the UI says so.

**Secured (vault) assets.** Securing moves the media into `~/.comfybox/secure.noindex`, drops the thumbnail and records the id in `secured_assets`, which `hiddenAssetIDs` uses to hide it (`AssetIngestor.swift:326`, `CatalogBrowser.swift:60`). A send of a secured asset moves it out of the vault and **clears its `secured_assets` row**, otherwise it would stay invisible on the remote too. The remote gallery is the privacy boundary from then on; the sidecar records `sensitive: true`, and no thumbnail is left in the Mac's cache.

**The watcher and the backfill must not undo a move.** `AssetIngestor` polls the output directory every 5 seconds against an in-memory `knownPaths` set (`AssetIngestor.swift:80`), and `ComfyBoxGallery backfill` rescans the same trees. A transfer takes a per-path lock with the ingestor for the length of the move, and drops the path from `knownPaths` when it completes. A row with `storage_state = remote` is a tombstone for that path: if the same file reappears locally, the ingestor **re-adopts the existing row** (back to `local`) instead of creating a second row for the same asset.

### 3.4 Immich remote

The same pipeline with an HTTP tail. Immich 2.3.1 runs at `http://10.0.100.232:2283`; the client reads `GET /api/server/version` on connect and refuses a major version it has not been built against, because the asset endpoints are versioned (Codex review).

- **Upload:** `POST /api/assets`, `x-api-key`, multipart (`assetData`, `deviceAssetId` = catalog id, `deviceId` = `comfybox-desktop`, `fileCreatedAt`, `fileModifiedAt`), plus the ComfyBox sidecar as `sidecarData`. `201` = created, `200` = duplicate; the body is `{id, status}`.
- **Duplicate detection** uses Immich's own `x-immich-checksum` header, which is **SHA-1**, not the catalog's SHA-256. Both digests are computed for an Immich send.
- **Album:** `POST /api/albums` once (named by config), then `PUT /api/albums/{albumID}/assets` with `{ids: [...]}`.
- **Verification before the local delete** is `GET /api/assets/{id}` returning the asset with the expected size and checksum. A `duplicate` status counts as present.
- Location: host `remote:<id>`, path `immich://<assetID>`.

### 3.5 Browsing, and hiding an absent remote

`RemoteGalleryRegistry` (`@Observable @MainActor`) resolves each configured remote to `reachable` / `absent` / `disabled`:

- `.folder`: the root is mounted (matched by `volumeUUID` first, so a rename or a different mount point still resolves).
- `.immich`: `GET /api/server/ping` within a 2-second budget, re-checked on a timer and on tab focus.

**Byte resolution needs a real abstraction, not a host map.** Today every remote path is turned into `http://127.0.0.1:7870/v1/gallery/file?path=` (`CatalogBrowser.swift:225`), and the engine refuses paths outside its allowed output directory (`AssetMediaSource.swift:10`) — so a `/Volumes/...` asset would never load. `AssetMediaResolver` gains one implementation per kind: folder remotes open `file://` directly, Immich uses its thumbnail and original endpoints with the key, and the existing server trees keep the engine route.

An asset whose remote is **absent** joins `hiddenAssetIDs`, the same carve-out the vault uses, so nothing about it renders in either tab. The Remote Gallery tab becomes the gallery scoped to remotes: a picker of the reachable remotes, with the same grid, lightbox and search.

### 3.6 Catalog cleanup (prerequisite — §1's third complaint)

Three changes, smallest first:

1. **Distinguish "deleted" from "unattached".** A missing file whose volume is mounted, and whose location host is this Mac or nothing, is a genuine orphan. A missing file under an **unmounted volume** or an **absent remote** is unattached and must never be pruned. `DAMStore.pruneOrphans` gains that test, so the unattended sweep stops counting unattached rows toward its ceiling.
2. **An explicit, reviewed purge.** `GalleryMaintenance` gains `scanMissingFiles()` (report: count, ids, paths, reclaimable thumbnail bytes) and `purgeMissing(ids:)`. The Maintenance view lists them and purges only what the operator confirms. An explicit purge is not bound by the unattended breaker; the breaker keeps guarding the automatic sweep.
3. **Vacuum stale location rows.** Delete `asset_locations` rows whose `asset_id` has no `assets` row (885 of them today).

## 4. Out of scope

- Bringing an asset back from a remote (a later task; the sidecar carries what it needs).
- Editing, re-rendering or deriving from a remote asset.
- Two-way sync, conflict resolution, or more than one Mac writing one remote gallery.
- Encrypting the drive. macOS encrypts the volume if Todd wants that; this feature writes plain files.
- The Immich web UI, its users, or its own albums beyond the one configured album.

## 5. Risks

| Risk | Mitigation |
|---|---|
| A move deletes the only copy | Checksum verified on the remote before any local delete; per-asset, so a failure never cascades |
| The drive is pulled mid-send | `.part` naming; an interrupted asset is never indexed and the local copy still exists |
| An absent remote looks like data loss | The remote's assets are hidden, never pruned; the tab says which remotes are absent |
| Immich deduplicates an upload | `status: duplicate` is treated as present; the local delete still proceeds only after the asset is confirmed |
| Case-sensitive APFS on the drive | Filenames are written exactly as stored; collisions get a numeric suffix rather than a case-fold overwrite |
| The purge deletes wanted rows | Explicit selection, a listed preview, and unattached rows are excluded by construction |
| A crash mid-move | The journal makes every state re-runnable, and the local copy is deleted only after the catalog points at a verified remote |
| The watcher re-adopts a moved file | Per-path lock during the move; `storage_state = remote` is a tombstone the ingestor honours |

## 6. Rollout

1. Catalog cleanup (§3.6) — safe on its own, fixes the live database.
2. Config + Settings UI (§3.1).
3. Folder remote: create, send, index (§3.2, §3.3).
4. Browse + hide-when-absent (§3.5).
5. Immich remote (§3.4).

## 7. Open questions

- **OQ-1** Immich album name. Default `ComfyBox` unless Todd names it.
- **OQ-2** Should a send be allowed while the engine is rendering? It is disk IO, not GPU, so the proposal is yes, with no GPU slot taken.
- **OQ-3** Does the Remote Gallery tab also keep showing Kira's and Bree's server trees (today's `host != mac` rows), or only configured remotes? The proposal is both: server trees are reachable remotes named by the catalog, listed alongside configured ones.

## 8. Review

Codex (read-only, 2026-09-17) reviewed the draft against the code and raised ten findings. Two criticals (delete-before-relocate; `absolute_path` cannot be surrendered) and five highs/mediums (secured rows, backfill re-adoption, relocate-not-delete, media resolver, ingestor concurrency, Immich API details) are folded into §3.3–§3.5 above. Its remaining note, that `DesktopSettings` lives inside `Views/SettingsView.swift` and that `ComfyBoxGallery` on :7871 is a catalog search service rather than a byte service, is corrected in §2 and §3.1.
