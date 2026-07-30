# The ComfyBox Gallery — one catalog for everything ComfyBox makes

Date: 2026-07-30
Status: design, approved for planning
Repos touched: `zimage.swift` (ComfyBox engine + desktop), `coffeeshop-server` (Kira daemon, Studio bot, Bree MCP)

## Problem

Media is produced by five paths into the ComfyBox Gallery and its two downstream
copies, and indexed by four partial, mutually-ignorant records — three of which
are rolling windows. Nothing can answer a question about work older than a
couple of days.

Measured 2026-07-30:

| index | scope | rows | cap | facets present |
|---|---|---|---|---|
| `~/.comfybox/dam.sqlite3` | Mac output dir | 1072 | none | prompt 943; `realm`/`content_mode`/`character_name`/`source` **all NULL** |
| `~/.kira/render-journal.jsonl` | Kira stream renders | 283 | **500 lines** | lane, tier, seed meta |
| `~/.kira/studio/history.json` | Studio bot renders | ~500 | **500 entries** | character, contentMode, seed |
| `~/.bree/studio/history.json` | Bree renders | ~500 | **500 entries** | character, contentMode, seed |
| `~/.kira/studio/catalog.json` | — | — | — | counts and bytes only |

**The ComfyBox Gallery is the home.** It holds Kira's content and every asset
ComfyBox generates, for whichever application requested it. `~/Pictures/ComfyBox`
(Mac, 1459+ files) is that home; the server trees — `~/.kira/studio` (2068
images + 496 videos + 3901 sidecars) and `~/.bree/studio` (29 images + 435
videos + 1273 sidecars) — are downstream distribution copies of it, not peers.

That is why `source` (already a column, currently 100% NULL) matters as much as
`realm`: it records **which application asked for the render** — Kira's
scheduler, the Studio bot, Bree's MCP, the desktop app, the Krita bridge, a
workflow run. The home indexes all of them.

Three symptoms follow directly:

1. **Kira's memory of her own work is ~2 days.** `render-journal.jsonl` is her
   only archive and it is capped at 500 lines against ~2500 assets.
2. **The Mac DAM cannot answer "who / which register / where from."** Those
   columns exist and are 100% empty, because `AssetIngestor` (in the desktop
   app) *infers* metadata after the fact from a sidecar or embedded PNG fields,
   and only while the app is running. The engine, which knows all of it at
   render time, never tells anyone. `dam.sqlite3` therefore starts 2026-07-27.
3. **The same asset is two rows or zero rows.** Every Kira/Studio asset is
   rendered to the Mac output dir and copied to a server tree under a different
   name (`recovered_from` in the sidecars records the pairing). No index knows
   the two are one thing.

Root cause, stated once: **the index is written by a process that has to guess,
instead of by the process that knows** — even though the data it needs is
already embedded in the files (see "Metadata carriers" below). The DAM columns
are empty for lack of *parsing*, not for lack of data.

## Decisions taken

1. **One schema serves all consumers.** Todd (desktop UI), Bree (MCP, on par
   with Todd), Kira (MCP, hard-scoped to her own realm), CoffeeShop Studio
   (Telegram bot — both producer and consumer).
2. **Canonical store on the Mac**, extending the existing `dam.sqlite3`. Aligns
   with the Kira-off-Linux migration: when the daemon moves to the Mac, nothing
   about the catalog changes.
3. **Facets + FTS, with a vision backfill** only for rows that have no prompt
   text. No embeddings, no vector store.
4. **The 2026-07-07 provenance contract is extended** to name the catalog as a
   third approved location for raw prompt text, held to the same rule.
5. **The file is the store of record; the catalog is derived and rebuildable.**
   Embedded EXIF/XMP/IPTC and Finder tags are authoritative; the catalog is an
   index over them, and gallery edits write back to them.
6. **The gallery is a ComfyBox service in its own process** — a
   `ComfyBox gallery-serve` subcommand under its own launchd agent, not routes
   inside the GPU engine. Same repo, same binary, separate lifecycle.
7. **`kira` is the only realm exception.** Two values: `kira` and `shared`.
8. **The ComfyBox Gallery is the home** for Kira's content and for everything
   ComfyBox generates, whichever application requested it. Bree's vault is out
   of scope — see "Deferred".

## Non-goals

- No embeddings or vector search. Revisit only if facets + FTS demonstrably fail.
- No new gallery UI framework — facets land in the existing `GalleryView`.
- No media moves. Locations are recorded; files stay exactly where they are.
- **Bree's vault is untouched — not indexed, not read, not written.** It is
  hers alone. See "Deferred".
- No change to `/v1/gallery/list` or `/v1/gallery/file`; `RemoteGalleryService`
  keeps working untouched.
- The three rolling records (`render-journal.jsonl`, both `history.json`) keep
  their current jobs and caps. They become hot caches in front of the catalog,
  not things to migrate away.

## Architecture

### Metadata carriers — the file is the store of record

The catalog is a **derived, rebuildable index**, not a store of record. Deleting
it and reconstructing it from the media must produce the same rows. Everything
below follows from that.

Carriers actually present today, ranked by durability:

| carrier | written by | survives copy to Linux | readable outside ComfyBox |
|---|---|---|---|
| EXIF / XMP / IPTC embedded | engine at render; `SidecarService.embed` on demand | **yes** | Adobe, Finder, Spotlight, exiftool |
| JSON sidecar | Studio / Kira daemon, server-side | n/a (written there) | no |
| Finder tags (`com.apple.metadata:_kMDItemUserTags`) | `FinderTags.swift` | **no** (xattr is lost) | Finder, Spotlight |
| catalog DB | this design | derived | no |

Verified on a live render:

```
EXIF:ImageDescription  full prompt
EXIF:Software          ComfyBox
EXIF:UserComment       {"prompt":…,"loras":[{name,scale}…],"seed":…,"steps":9,
                        "guidance":0,"model":"krea-2-turbo","width":…,"height":…}
XMP:CreatorTool        ComfyBox
XMP-dc:Description     full prompt
XMP-dc:Subject         krea-2-turbo
```

plus `SidecarService.keywords(tags:character:contentMode:)` →
`IPTC:Keywords` + `XMP-dc:Subject`.

Two rules follow:

1. **Write-back is mandatory.** Ratings, tags, captions and favorites set in the
   gallery are written back to XMP/IPTC and Finder tags, so Adobe, Finder and
   Spotlight keep seeing them and so they survive a catalog rebuild.
2. **Finder tags are Mac-realm only.** They do not ride the copy to the server,
   so color labels are never treated as authoritative for a server-side asset.

### Realm — one exception, not a taxonomy

`realm` has exactly two values: **`kira`** and **`shared`**. Kira's renders stamp
`kira`; everything else — Todd's, Bree's, Studio's — is `shared`. Kira's tool
sees only `kira`; the other three consumers see everything. Kira is the only
isolation exception, so the column encodes that and nothing more.

`character_name` is a **different** column: who is *depicted* (Kira, Divine,
Rhea, Bea, Soraya, Mahal). It does not track realm — a Kira-realm render can
depict anyone, and a shared render can depict Kira.

Realm cannot be derived from a Mac path, because every realm's renders pass
through the same engine output dir. **The caller stamps it.** For backfill it is
`kira` when the asset has a twin under a Kira studio tree or a matching journal
record, and `shared` otherwise.

### Identity — one asset, many locations

Merge key is `sha256` (column already exists). `recovered_from` and the filename
UUID are fallback joins. **Verification item for the plan:** confirm the
server copy is a byte-identical file copy rather than a re-encode — if any path
re-encodes or resizes, `sha256` cannot be the primary merge key and
`recovered_from` becomes primary. A new
`asset_locations(asset_id, host, path, mtime)` table holds the Mac original and
the server copy as two rows against one asset. `assets.absolute_path` stays the
Mac primary so every existing `DAMStore` query keeps working.

### Service boundary — `ComfyBox gallery-serve`

The gallery is a ComfyBox service, but **not inside the GPU process**.

Today the Mac Gallery and the server Gallery differ because they are two
independent *implementations* of the same idea — a SQLite DAM in the desktop
app, and a `FileManager.enumerator` in `WarmServer.swift:1233`. Nothing makes
them agree but the filesystem. One service with many views fixes that: the
desktop Gallery tab, the Telegram browser, Bree's MCP and Kira's MCP all become
clients.

It runs as a `ComfyBox gallery-serve` subcommand under its own launchd agent, on
its own port, owning the catalog and all read/write routes. Rationale:

- `ComfyBox serve` loads models, gates at 40GB, and gets rebuilt and re-signed;
  restarting it orphans in-flight GPU jobs. Coupling the gallery to it means
  browsing dies during model loads, and every gallery fix requires bouncing the
  renderer and re-signing the engine binary.
- Same repo and binary keeps this consistent with retiring the Node image
  service into ComfyBox — it adds a process, not a project.
- The desktop still reads the SQLite directly, so the local UI takes no HTTP hop.

### Write path — the engine reports, the gallery records

The engine does not own the archive. At render completion it **reports** the
render to `gallery-serve` over localhost, which writes the row into
`~/.comfybox/dam.sqlite3` (already WAL, already multi-process safe,
`DAMStore.swift:57`). Render requests gain an optional `catalog` object so the
caller stamps facets the engine cannot know — `realm`, `lane`, `arc`, `theme`,
`stock`, `genre`, `family`, `style`, `character`, `sealed`. The engine passes
them through verbatim beside what it already knows (prompt, negative, seed,
steps, guidance, model, preset, LoRAs, dimensions, output path).

If `gallery-serve` is down, the report is dropped, not retried into a queue —
the asset is still on disk with full embedded metadata, and the next backfill
sweep picks it up. This is the payoff of treating the file as the store of
record: the reporting path is allowed to be lossy.

`AssetIngestor` remains the **fallback** for assets that appear with no catalog
write — drag-ins, mflux output, recovered clips. It is not removed.

CoffeeShop Studio needs no new write path: it already writes a rich sidecar per
render. It gains a catalog row written alongside that sidecar.

### Schema — extend `assets`, do not fork it

New columns:

```
realm TEXT              -- kira | shared   (isolation column; kira is the only exception)
lane TEXT               -- still | shoot | tile | kira | film-erotic | video
arc TEXT, theme TEXT
stock TEXT, genre TEXT, family TEXT, style TEXT   -- mirrors RenderSeedMeta 1:1
preset TEXT, loras TEXT (json), render_id TEXT
caption TEXT, caption_source TEXT                  -- vision backfill
sealed INTEGER NOT NULL DEFAULT 0
duration_ms INTEGER, fps REAL, frames INTEGER      -- video
```

`content_mode` (= tier), `character_name`, `source`, `sha256`, `rating`,
`favorite` already exist and finally get populated.

New tables: `asset_locations(asset_id, host, path, mtime)`.

FTS5 extends from `(prompt, negative_prompt)` to `(prompt, negative_prompt,
caption)`.

The `stock`/`genre`/`family`/`style` names are taken verbatim from
`RenderSeedMeta` in `render-journal.ts` so journal records map 1:1 with no
translation layer.

### Provenance and permissions

The contract from 2026-07-07 (documented in `studio-log.ts`) is extended: the
catalog is a **third** approved location for raw prompt text, subject to the
same rule.

- Catalog file `0600` inside a `0700` directory on both boxes. This fixes a
  pre-existing gap: `~/.comfybox/dam.sqlite3` is currently `-rw-r--r--` inside
  a `drwxr-xr-x` directory while already holding 943 raw prompts.
- **`sealed` rows store facets only** — no `prompt`, no `negative_prompt`, no
  `caption`, and no FTS entry. This mirrors the existing sidecar behaviour for
  sealed (Bree) turns.
- **`sealed` must gate embedding too, not only the catalog row.** The engine
  writes the raw prompt into every file's `EXIF:ImageDescription`,
  `EXIF:UserComment` and `XMP-dc:Description` — a carrier that travels with the
  file everywhere, including into the world-readable Mac output dir. This
  predates the design and is not introduced by it, but it is the same class of
  exposure the contract exists to prevent.
  **Verification item for the plan:** determine what the engine currently embeds
  for Bree's sealed renders, and gate it if it embeds prompt text. Do not assume
  either behaviour.
- The amendment is written into the `studio-log.ts` header comment, where the
  rule is enforced, so the contract stays documented at the point of use.

### Query surface — one API, four bindings

Served by `gallery-serve`, not by the engine:

- `GET /v1/catalog/search` — `q`, `realm`, `lane`, `tier`, `character`, `stock`,
  `genre`, `arc`, `kind`, `min_rating`, `since`, `until`, `order`, `limit`,
  `offset`. Returns rows with their locations and a thumbnail URL.
- `GET /v1/catalog/facets` — value counts per facet, so a UI can be browsed
  rather than only searched.
- `GET /v1/catalog/asset/{id}`.

Bindings:

| consumer | binding | scope |
|---|---|---|
| Todd | facet rail + saved searches in `GalleryView`, via `DAMStore` locally (no HTTP hop) | both realms |
| Bree | MCP `search_gallery`, beside her existing `gallery_recent` | both realms |
| Studio | `GenerationHistory.search` falls through to the catalog past its 500-entry window | both realms |
| Kira | MCP `search_gallery` | **`realm = kira` only, locked at the tool boundary** |

Kira's realm lock is applied service-side, not as a parameter she supplies. She
cannot express a query that reaches a `shared` asset. This preserves the
isolation boundary already set for the backroom: no persona or content bleed.

### Kira's memory

`render-journal.jsonl` keeps its job unchanged — 500-line rolling window,
prompt-bound digest, taste loop, mode clamp. The catalog becomes the durable
archive behind it: `recallRenders` falls through to a catalog query when the
answer is older than the window. That is what takes her from two days of memory
to the full archive.

The existing tier ceiling applies to catalog results exactly as it applies to
journal records — no explicit intent text or avocado path reaches an apple-mode
prompt.

## Backfill

Cheap and almost entirely GPU-free, because the data already exists in two
places: embedded in the files, and in the server sidecars.

1. **`exiftool` sweep over the gallery home and both server trees — the primary source.** Recovers
   prompt, LoRAs with scales, seed, steps, guidance, model, dimensions and
   `Software` from `EXIF:UserComment` / `ImageDescription` / `XMP`. Because EXIF
   rides the file copy, this covers the server trees as well as the Mac.
2. **Server sidecars supply what EXIF lacks** — `tier`, `lane`, `category`,
   `checkpoint`, `preset`, `sealed` (3901 Kira + 1273 Bree). Journal lines (283)
   and both `history.json` files fill `lane`/`arc`/`theme` for recent renders.
3. **Finder tags** (`_kMDItemUserTags`) supply existing color labels for Mac
   assets. Mac-realm only — the xattr does not survive the copy.
4. **Deduplication** — `sha256`, falling back to `recovered_from` and the
   filename UUID, merges each gallery-home asset with its server copies into
   one asset with several locations.
5. **Vision captions last, and only** where `prompt IS NULL AND caption IS NULL`
   after steps 1–2 — a far smaller set than the 129 rows estimated before the
   EXIF sweep was known to exist. Low-priority queue that never competes with a
   Kira render cycle.

Backfill is idempotent and re-runnable, and — per "the file is the store of
record" — reconstructs the catalog from scratch if it is deleted.

## Testing

- **Realm isolation** — property test: for any combination of filters, a
  kira-scoped query never returns a row whose realm is not `kira`.
- **Sealed rows carry no text** — a sealed render produces a row with facets and
  with `prompt`, `negative_prompt`, `caption` all NULL and no FTS entry.
- **Render reporting** — a render carrying `catalog` facets produces a row with
  those facets non-null. This is the exact failure that left the current columns
  empty; it must be asserted, not assumed.
- **Lossy reporting is safe** — with `gallery-serve` stopped, a render still
  lands on disk with full embedded metadata and is picked up by the next
  backfill sweep. No queue, no retry, no lost asset.
- **Rebuildability** — delete the catalog, re-run backfill, get the same rows.
  This is the property that makes the catalog safe to treat as derived.
- **Write-back round-trip** — a rating/tag set in the gallery is readable by
  `exiftool` and by Finder afterwards, and survives a catalog rebuild.
- **Backfill idempotence** — run twice: unchanged row count, no duplicate
  locations.
- **Deduplication** — a Mac original and its server copy resolve to one asset
  with two locations, not two assets.
- **Journal ↔ catalog agreement** — every journal record resolves to a catalog
  row.
- **Fallback ingest** — an asset appearing with no catalog write is still
  indexed by `AssetIngestor`.
- **Permissions** — catalog file is `0600` in a `0700` directory after init on
  both boxes.
- **Vault is never touched** — after backfill, write-back and a full test run,
  no path under `~/Documents/Vaults/BarkadaAI` has been read or written.
  Asserted, not assumed.

## Deferred

**Bree's vault** (`~/Documents/Vaults/BarkadaAI` on the server — 1848 images +
230 videos, 5.9 GB). Out of scope: it is Bree's alone. The intended future shape
is to normalize it as a **remote gallery** — a peer surface the catalog can
browse rather than a tree it ingests — which keeps ownership with Bree while
making it reachable. Nothing in this design reads or writes it.

One observation to carry into that work, measured 2026-07-30 and recorded only
so it is not rediscovered as a surprise: PR #826/#833 stripped prompt text from
the vault-adjacent sidecar, but embedded metadata rides the file itself. Over a
120-file sample of vault PNGs, 81 carry PNG text chunks and 67 carry a large
`eXIf` block, including IPTC captions and JSON generation parameters. If those
files are ever stripped, the catalog is the natural place to hold the provenance
they would lose.

## Sequencing and rollback

This is two implementation plans, split at the engine boundary.

**Plan A — nothing signed, nothing restarted.**

1. Schema migration + permissions tightening (additive; no reader changes).
2. Backfill: `exiftool` sweep, sidecars, journals, dedup.
3. `ComfyBox gallery-serve` subcommand + launchd agent + `DAMStore` query
   methods.
4. Consumers: desktop facets, Bree MCP tool, Kira MCP tool, Studio fallthrough.

The `gallery-serve` subcommand ships in the same binary as the engine, so step 3
does involve a build — but **the running engine is not replaced or restarted**.
The new agent launches the freshly built binary in a separate process while
`com.barkadabrew.comfybox` keeps running the checkpointed one. Rollback is
`launchctl bootout` of the new agent; nothing else changes.

**Plan B — the one deploy that touches the engine.**

5. Render reporting from the engine, `sealed` embed gating, metadata write-back.

This is the only piece requiring the running engine to be replaced: rebuild,
Developer-ID re-sign, and a restart — the riskiest action available per
`~/.comfybox/CHECKPOINT-video-known-good-20260730.md`. It deploys on its own,
with the existing rollback (binary backup `ComfyBox.bak-KNOWN-GOOD-20260730`,
restart under `~/.kira/coordination/comfybox-restart.lock`).

Plan A alone already delivers the ask: Kira stops being capped at two days of
memory, and every gallery becomes searchable across the gallery home and both server trees. Plan B is
what stops the index from drifting again. Every step is independently
revertible; the catalog is additive throughout, so reverting any step leaves the
existing four indexes working exactly as they do today.
