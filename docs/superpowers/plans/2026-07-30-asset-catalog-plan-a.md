# Asset Catalog — Plan A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one searchable catalog over every asset ComfyBox has made — served by its own process — without replacing or restarting the running engine.

**Architecture:** A dependency-free `ComfyBoxCatalog` library owns the SQLite schema (an additive migration of the existing `~/.comfybox/dam.sqlite3`), the backfill readers, and the query layer. A tiny `ComfyBoxGallery` executable serves it over HTTP on its own launchd agent. The Kira daemon, Bree's MCP and the Studio bot become HTTP clients; the desktop keeps reading the same SQLite directly.

**Tech Stack:** Swift 5.9 (Foundation + system SQLite3 + Network framework, no third-party deps), `exiftool` 13.55 at `/opt/homebrew/bin/exiftool`, TypeScript/Node on the coffeeshop-server side (`node scripts/run-tests.mjs`).

**Spec:** `docs/superpowers/specs/2026-07-30-asset-catalog-design.md`

## Global Constraints

- **Never rebuild the engine binary.** `.build/release/ComfyBox` is the running, checkpointed engine (`~/.comfybox/CHECKPOINT-video-known-good-20260730.md`). No task in this plan may run a build that writes to it. Build only the new targets by name: `swift build -c release --product ComfyBoxGallery`.
- **Never restart `com.barkadabrew.comfybox`.** No `launchctl bootout`/`bootstrap`/`kickstart` against that label in this plan.
- **Never `rm -rf .build`** — it destroys `mlx.metallib`, which `swift build` will not regenerate.
- **`ComfyBoxCatalog` has zero dependencies** beyond Foundation, SQLite3 and Network. It must not import or link `ZImage` or MLX.
- **Schema changes are additive only.** New columns append to the end of `assets`; `DAMStore` reads by explicit column list (`DAMStore.swift:223`, `:519`), so appended columns cannot break it. Never drop, rename or reorder an existing column. Never touch `folders`, `asset_folders` or `secured_assets`.
- **Realm values are exactly `kira` and `shared`.** No third value.
- **Kira's realm lock is applied service-side**, never as a client-supplied parameter.
- **Never read or write anything under `~/Documents/Vaults/BarkadaAI`.** Out of scope.
- **Catalog file is `0600` inside a `0700` directory.**
- **`sealed` rows store no text**: `prompt`, `negative_prompt`, `prompt_raw` and `caption` are NULL and no FTS row is written.
- Before any `systemctl restart kira-daemon`, read `~/.kira/coordination/README.md` and touch the lock.
- Server deploys go through `scripts/local-merge.sh <branch>` then `scripts/kira-update.sh` on `todd@10.0.100.232` (fish shell — wrap commands in `bash -c`).

## Deviation from the spec, and why

The spec chose "`ComfyBox gallery-serve` subcommand, own process". Building that subcommand requires building the `ComfyBox` product, which writes `.build/release/ComfyBox` — the running engine. That contradicts Plan A's promise.

This plan keeps the intent (same repo, own process, own lifecycle) and moves the seam: all logic lives in `ComfyBoxCatalog`, served by a separate `ComfyBoxGallery` executable. In Plan B, when the engine is rebuilt anyway, `main.swift` gains:

```swift
case "gallery-serve":
  GalleryServer.runCLIEntryPoint(args: Array(args.dropFirst()))
  return
```

which delegates to the same library. No code is duplicated and no decision is reversed.

## File Structure

**New — `Sources/ComfyBoxCatalog/`** (library, no dependencies)

| file | responsibility |
|---|---|
| `CatalogSchema.swift` | additive migration + seed collections. The only file that writes DDL. |
| `CatalogModels.swift` | `CatalogAsset`, `CatalogCollection`, `AssetEdge`, `CatalogQuery`, `CatalogFacets`. Pure value types. |
| `CatalogStore.swift` | actor over SQLite: upsert, search, facets, collections, edges. No I/O beyond the DB. |
| `MetadataReader.swift` | `exiftool` + JSON-sidecar + container-probe readers. Filesystem in, values out. |
| `CatalogBackfill.swift` | sweep orchestration: walk trees, dedup, file into collections, build edges. |
| `CollectionRules.swift` | the lane/source/preset → collection rule table. Pure, table-driven. |
| `HTTPKit.swift` | minimal `NWListener` HTTP server: request parse, routing, JSON/binary responses. |
| `GalleryServer.swift` | the routes themselves + `runCLIEntryPoint(args:)`. |

**New — `Sources/ComfyBoxGallery/main.swift`** — 10 lines, calls `GalleryServer.runCLIEntryPoint`.

**New — `Tests/ComfyBoxCatalogTests/`** — one test file per source file above.

**Modified — `Package.swift`** — add both targets and the test target; add `ComfyBoxCatalog` to `ComfyBoxDesktop`'s dependencies.

**coffeeshop-server (new):** `src/catalog/client.ts`, `src/catalog/client.test.ts`, `src/kira/gallery-tools.ts`, `src/kira/gallery-tools.test.ts`, `src/tools/gallery-search-tools.ts`, `src/tools/gallery-search-tools.test.ts`.

**coffeeshop-server (modified):** `src/studio/generation-history.ts` (search fallthrough).

**Desktop (modified):** `Sources/ComfyBoxDesktop/Views/GalleryView.swift` (facet + collection rail).

---

### Task 1: Package targets and an empty catalog library that builds

**Files:**
- Modify: `Package.swift`
- Create: `Sources/ComfyBoxCatalog/CatalogModels.swift`
- Create: `Sources/ComfyBoxGallery/main.swift`
- Test: `Tests/ComfyBoxCatalogTests/CatalogModelsTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces: targets `ComfyBoxCatalog` (library) and `ComfyBoxGallery` (executable); types `CatalogAsset`, `CatalogCollection`, `AssetEdge`, `CatalogRealm`.

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxCatalogTests/CatalogModelsTests.swift`:

```swift
import XCTest
@testable import ComfyBoxCatalog

final class CatalogModelsTests: XCTestCase {
    func testRealmHasExactlyTwoValues() {
        XCTAssertEqual(CatalogRealm.allCases.map(\.rawValue).sorted(), ["kira", "shared"])
    }

    func testRealmDefaultsToSharedForUnknownInput() {
        XCTAssertEqual(CatalogRealm(rawValue: "todd") ?? .shared, .shared)
        XCTAssertEqual(CatalogRealm(rawValue: "kira"), .kira)
    }

    func testSealedAssetCarriesNoText() {
        let a = CatalogAsset(
            id: "a1", kind: "image", filename: "x.png", absolutePath: "/tmp/x.png",
            realm: .shared, sealed: true,
            prompt: "secret", negativePrompt: "neg", promptRaw: "raw", caption: "cap"
        )
        XCTAssertNil(a.prompt)
        XCTAssertNil(a.negativePrompt)
        XCTAssertNil(a.promptRaw)
        XCTAssertNil(a.caption)
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter CatalogModelsTests`
Expected: FAIL — no such module `ComfyBoxCatalog`.

- [ ] **Step 3: Add the targets**

In `Package.swift`, add to `products`:

```swift
    .executable(name: "ComfyBoxGallery", targets: ["ComfyBoxGallery"]),
```

and to `targets`:

```swift
    .target(
      name: "ComfyBoxCatalog",
      dependencies: [],
      path: "Sources/ComfyBoxCatalog"
    ),
    .executableTarget(
      name: "ComfyBoxGallery",
      dependencies: ["ComfyBoxCatalog"],
      path: "Sources/ComfyBoxGallery"
    ),
    .testTarget(
      name: "ComfyBoxCatalogTests",
      dependencies: ["ComfyBoxCatalog"],
      path: "Tests/ComfyBoxCatalogTests"
    ),
```

Also add `"ComfyBoxCatalog"` to the existing `ComfyBoxDesktop` target's `dependencies` array (it currently reads `dependencies: ["ZImage"]`).

- [ ] **Step 4: Write the models**

Create `Sources/ComfyBoxCatalog/CatalogModels.swift`:

```swift
// CatalogModels.swift — pure value types for the asset catalog.
//
// No I/O, no SQLite, no Foundation-beyond-Date. Everything here is safe to
// construct in a test without touching a database or the filesystem.

import Foundation

/// The isolation domain. Kira is the ONLY exception; everything else — Todd's
/// work, Bree's, the Studio bot's — is `shared`. Never add a third value.
public enum CatalogRealm: String, CaseIterable, Sendable, Codable {
    case kira
    case shared
}

/// A relation between two assets.
public enum AssetRelation: String, Sendable, Codable {
    /// `from` (a clip) was animated from `to` (a still).
    case i2vSource = "i2v_source"
    /// `from` (a clip) is a component of `to` (a scene / montage / storyboard).
    case memberOf = "member_of"
}

public struct AssetEdge: Sendable, Equatable, Codable {
    public let fromAssetID: String
    public let toAssetID: String
    public let relation: AssetRelation

    public init(fromAssetID: String, toAssetID: String, relation: AssetRelation) {
        self.fromAssetID = fromAssetID
        self.toAssetID = toAssetID
        self.relation = relation
    }
}

/// A body of work. `parentID == nil` is a root; a child's parent must itself be
/// a root (two levels, enforced in CatalogStore).
public struct CatalogCollection: Sendable, Equatable, Codable {
    public let id: String
    public let slug: String
    public let name: String
    public let parentID: String?
    /// nil = shared vocabulary anything may contribute to.
    /// .kira = exists only in her realm and is visible only to her.
    public let realm: CatalogRealm?
    public let description: String?

    public init(id: String = UUID().uuidString, slug: String, name: String,
                parentID: String? = nil, realm: CatalogRealm? = nil,
                description: String? = nil) {
        self.id = id
        self.slug = slug
        self.name = name
        self.parentID = parentID
        self.realm = realm
        self.description = description
    }
}

/// Where an asset's bytes live. One asset, many locations — the gallery home
/// plus any downstream server copies.
public struct AssetLocation: Sendable, Equatable, Codable {
    public let host: String
    public let path: String
    public let mtime: Date

    public init(host: String, path: String, mtime: Date) {
        self.host = host
        self.path = path
        self.mtime = mtime
    }
}

public struct CatalogAsset: Sendable, Equatable {
    public let id: String
    public let kind: String            // "image" | "video"
    public let filename: String
    public let absolutePath: String
    public let sha256: String?
    public let fileSize: Int64
    public let width: Int?
    public let height: Int?
    public let createdAt: Date

    // Isolation and provenance
    public let realm: CatalogRealm
    public let source: String?         // which application requested it
    public let sealed: Bool

    // Text (always nil when sealed)
    public let prompt: String?
    public let negativePrompt: String?
    public let promptRaw: String?
    public let caption: String?
    public let captionSource: String?

    // Generation facets
    public let seed: Int?
    public let steps: Int?
    public let guidance: Double?
    public let modelFamily: String?
    public let preset: String?
    public let loras: String?          // JSON array as stored
    public let renderID: String?
    public let contentMode: String?    // the fruit tier
    public let characterName: String?

    // Seed metadata — names mirror RenderSeedMeta 1:1
    public let lane: String?
    public let arc: String?
    public let theme: String?
    public let stock: String?
    public let genre: String?
    public let family: String?
    public let style: String?

    // Video
    public let mode: String?           // "i2v" | "t2v"
    public let durationMs: Int?
    public let fps: Double?
    public let frames: Int?
    public let resolution: String?
    public let aspectRatio: String?

    // User annotation
    public let rating: Int
    public let favorite: Bool

    public init(
        id: String = UUID().uuidString, kind: String = "image",
        filename: String, absolutePath: String,
        sha256: String? = nil, fileSize: Int64 = 0,
        width: Int? = nil, height: Int? = nil, createdAt: Date = Date(),
        realm: CatalogRealm = .shared, source: String? = nil, sealed: Bool = false,
        prompt: String? = nil, negativePrompt: String? = nil,
        promptRaw: String? = nil, caption: String? = nil, captionSource: String? = nil,
        seed: Int? = nil, steps: Int? = nil, guidance: Double? = nil,
        modelFamily: String? = nil, preset: String? = nil, loras: String? = nil,
        renderID: String? = nil, contentMode: String? = nil, characterName: String? = nil,
        lane: String? = nil, arc: String? = nil, theme: String? = nil,
        stock: String? = nil, genre: String? = nil, family: String? = nil, style: String? = nil,
        mode: String? = nil, durationMs: Int? = nil, fps: Double? = nil, frames: Int? = nil,
        resolution: String? = nil, aspectRatio: String? = nil,
        rating: Int = 0, favorite: Bool = false
    ) {
        self.id = id; self.kind = kind
        self.filename = filename; self.absolutePath = absolutePath
        self.sha256 = sha256; self.fileSize = fileSize
        self.width = width; self.height = height; self.createdAt = createdAt
        self.realm = realm; self.source = source; self.sealed = sealed
        // Sealed rows carry facets only — the text is dropped at construction so
        // no caller can smuggle it past the rule by forgetting to check.
        self.prompt = sealed ? nil : prompt
        self.negativePrompt = sealed ? nil : negativePrompt
        self.promptRaw = sealed ? nil : promptRaw
        self.caption = sealed ? nil : caption
        self.captionSource = sealed ? nil : captionSource
        self.seed = seed; self.steps = steps; self.guidance = guidance
        self.modelFamily = modelFamily; self.preset = preset; self.loras = loras
        self.renderID = renderID; self.contentMode = contentMode
        self.characterName = characterName
        self.lane = lane; self.arc = arc; self.theme = theme
        self.stock = stock; self.genre = genre; self.family = family; self.style = style
        self.mode = mode; self.durationMs = durationMs; self.fps = fps; self.frames = frames
        self.resolution = resolution; self.aspectRatio = aspectRatio
        self.rating = rating; self.favorite = favorite
    }
}
```

Create `Sources/ComfyBoxGallery/main.swift`:

```swift
// ComfyBoxGallery — the gallery service process.
//
// Deliberately separate from the ComfyBox engine binary: the engine is a GPU
// process that loads models, gets rebuilt and re-signed, and whose restart
// orphans in-flight jobs. The gallery must stay up across all of that.

import ComfyBoxCatalog

GalleryServer.runCLIEntryPoint(args: Array(CommandLine.arguments.dropFirst()))
```

Until Task 7 lands `GalleryServer`, stub it so the target compiles. Create `Sources/ComfyBoxCatalog/GalleryServer.swift`:

```swift
import Foundation

public enum GalleryServer {
    /// Replaced with the real server in Task 7.
    public static func runCLIEntryPoint(args: [String]) {
        FileHandle.standardError.write(Data("gallery server not yet implemented\n".utf8))
        exit(1)
    }
}
```

- [ ] **Step 5: Run the tests**

Run: `swift test --filter CatalogModelsTests`
Expected: PASS, 3 tests.

- [ ] **Step 6: Verify the engine binary was not rebuilt**

```bash
shasum -a 256 .build/release/ComfyBox
```

Expected: `7422710b…` — the checkpointed hash, unchanged. If this differs, STOP and restore from `.build/release/ComfyBox.bak-KNOWN-GOOD-20260730`.

- [ ] **Step 7: Commit**

```bash
git add Package.swift Sources/ComfyBoxCatalog Sources/ComfyBoxGallery Tests/ComfyBoxCatalogTests
git commit -m "feat(catalog): add ComfyBoxCatalog library and ComfyBoxGallery executable targets"
```

---

### Task 2: Additive schema migration and seed collections

**Files:**
- Create: `Sources/ComfyBoxCatalog/CatalogSchema.swift`
- Test: `Tests/ComfyBoxCatalogTests/CatalogSchemaTests.swift`

**Interfaces:**
- Consumes: `CatalogRealm` (Task 1).
- Produces: `CatalogSchema.migrate(db:)`, `CatalogSchema.seedCollections` (`[CatalogCollection]`), `CatalogSchema.newColumns` (`[(String, String)]` of name/type).

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxCatalogTests/CatalogSchemaTests.swift`:

```swift
import XCTest
import SQLite3
@testable import ComfyBoxCatalog

final class CatalogSchemaTests: XCTestCase {
    private var db: OpaquePointer!
    private var path: String!

    override func setUp() {
        super.setUp()
        path = NSTemporaryDirectory() + "catalog-test-\(UUID().uuidString).sqlite3"
        XCTAssertEqual(sqlite3_open(path, &db), SQLITE_OK)
    }

    override func tearDown() {
        sqlite3_close(db)
        try? FileManager.default.removeItem(atPath: path)
        super.tearDown()
    }

    /// Recreate the CURRENT production schema, so we test migration of a real
    /// pre-existing database rather than a greenfield one.
    private func createLegacySchema() {
        let sql = """
            CREATE TABLE assets (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'image',
                filename TEXT NOT NULL, absolute_path TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL DEFAULT 0, sha256 TEXT,
                width INTEGER, height INTEGER,
                created_at REAL NOT NULL, modified_at REAL NOT NULL,
                ingested_at REAL NOT NULL, orphaned INTEGER NOT NULL DEFAULT 0,
                prompt TEXT, negative_prompt TEXT, seed INTEGER, steps INTEGER,
                guidance REAL, model_family TEXT,
                rating INTEGER NOT NULL DEFAULT 0, favorite INTEGER NOT NULL DEFAULT 0,
                content_mode TEXT, character_name TEXT, source TEXT
            );
            CREATE TABLE folders (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE asset_folders (asset_id TEXT PRIMARY KEY, folder_id TEXT NOT NULL);
            CREATE VIRTUAL TABLE assets_fts USING fts5(id UNINDEXED, prompt, negative_prompt);
            """
        XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK)
    }

    private func columns(of table: String) -> [String] {
        var stmt: OpaquePointer?
        var out: [String] = []
        sqlite3_prepare_v2(db, "PRAGMA table_info(\(table))", -1, &stmt, nil)
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 1) { out.append(String(cString: c)) }
        }
        sqlite3_finalize(stmt)
        return out
    }

    private func tableNames() -> [String] {
        var stmt: OpaquePointer?
        var out: [String] = []
        sqlite3_prepare_v2(db, "SELECT name FROM sqlite_master WHERE type='table'", -1, &stmt, nil)
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 0) { out.append(String(cString: c)) }
        }
        sqlite3_finalize(stmt)
        return out
    }

    func testMigrationAddsNewColumnsWithoutDisturbingLegacyOnes() throws {
        createLegacySchema()
        let before = columns(of: "assets")
        try CatalogSchema.migrate(db: db)
        let after = columns(of: "assets")

        // Every legacy column survives, in its original position.
        XCTAssertEqual(Array(after.prefix(before.count)), before,
                       "legacy columns must not move — DAMStore reads by explicit list")
        for (name, _) in CatalogSchema.newColumns {
            XCTAssertTrue(after.contains(name), "missing new column \(name)")
        }
    }

    func testMigrationIsIdempotent() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        let once = columns(of: "assets")
        try CatalogSchema.migrate(db: db)
        XCTAssertEqual(columns(of: "assets"), once)
    }

    func testMigrationCreatesNewTablesAndLeavesLegacyTablesAlone() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        let tables = tableNames()
        for t in ["asset_locations", "collections", "asset_collections", "asset_edges"] {
            XCTAssertTrue(tables.contains(t), "missing table \(t)")
        }
        XCTAssertTrue(tables.contains("folders"))
        XCTAssertTrue(tables.contains("asset_folders"))
        XCTAssertEqual(columns(of: "asset_folders"), ["asset_id", "folder_id"],
                       "legacy folder tables must be untouched")
    }

    func testFTSGainsCaptionColumn() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        XCTAssertEqual(columns(of: "assets_fts"), ["id", "prompt", "negative_prompt", "caption"])
    }

    func testSeedCollectionsAreTwoLevelsAndWellFormed() {
        let seeds = CatalogSchema.seedCollections
        let roots = Set(seeds.filter { $0.parentID == nil }.map(\.id))
        for c in seeds where c.parentID != nil {
            XCTAssertTrue(roots.contains(c.parentID!),
                          "\(c.slug) parent must be a root — two levels only")
        }
        XCTAssertEqual(Set(seeds.map(\.slug)).count, seeds.count, "slugs must be unique")
        // Kira's genres are hers; the cross-producer bodies are shared.
        XCTAssertNil(seeds.first { $0.slug == "decoupage" }?.realm)
        XCTAssertNil(seeds.first { $0.slug == "photography" }?.realm)
        XCTAssertEqual(seeds.first { $0.slug == "kira-dreams-memories" }?.realm, .kira)
        XCTAssertEqual(seeds.first { $0.slug == "kira-autocord" }?.realm, .kira)
    }

    func testSeedingIsIdempotent() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        try CatalogSchema.migrate(db: db)
        var stmt: OpaquePointer?
        sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM collections", -1, &stmt, nil)
        XCTAssertEqual(sqlite3_step(stmt), SQLITE_ROW)
        XCTAssertEqual(Int(sqlite3_column_int(stmt, 0)), CatalogSchema.seedCollections.count)
        sqlite3_finalize(stmt)
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter CatalogSchemaTests`
Expected: FAIL — `CatalogSchema` is undefined.

- [ ] **Step 3: Write the schema**

Create `Sources/ComfyBoxCatalog/CatalogSchema.swift`:

```swift
// CatalogSchema.swift — the ONLY file that writes DDL.
//
// Migration is strictly additive. `DAMStore` in the desktop app reads `assets`
// with an explicit column list (DAMStore.swift:223, :519), so columns appended
// at the end are invisible to it and cannot break it. Never drop, rename or
// reorder an existing column; never touch folders / asset_folders /
// secured_assets, which are legacy and deliberately left alone.

import Foundation
import SQLite3

public enum CatalogSchemaError: Error, LocalizedError {
    case execFailed(String, String)

    public var errorDescription: String? {
        switch self {
        case let .execFailed(sql, msg): return "SQL failed: \(msg) — \(sql)"
        }
    }
}

public enum CatalogSchema {

    /// Columns appended to `assets`, in order. Name → SQLite type.
    public static let newColumns: [(String, String)] = [
        ("realm", "TEXT"),
        ("sealed", "INTEGER NOT NULL DEFAULT 0"),
        ("lane", "TEXT"),
        ("arc", "TEXT"),
        ("theme", "TEXT"),
        ("stock", "TEXT"),
        ("genre", "TEXT"),
        ("family", "TEXT"),
        ("style", "TEXT"),
        ("preset", "TEXT"),
        ("loras", "TEXT"),
        ("render_id", "TEXT"),
        ("caption", "TEXT"),
        ("caption_source", "TEXT"),
        ("prompt_raw", "TEXT"),
        ("mode", "TEXT"),
        ("duration_ms", "INTEGER"),
        ("fps", "REAL"),
        ("frames", "INTEGER"),
        ("resolution", "TEXT"),
        ("aspect_ratio", "TEXT"),
    ]

    /// The seeded bodies of work. Ids are stable literals so re-seeding is a
    /// no-op and so the rule table can reference them by id.
    public static let seedCollections: [CatalogCollection] = [
        // Shared — any producer contributes.
        CatalogCollection(id: "col-decoupage", slug: "decoupage", name: "Decoupage",
                          description: "Tile designs — desktop tool, tile engine, Krita, Kira."),
        CatalogCollection(id: "col-photography", slug: "photography", name: "Photography",
                          description: "Art photography from any producer."),
        CatalogCollection(id: "col-adult", slug: "adult", name: "Adult",
                          parentID: nil, description: "Adult entertainment."),
        // Kira's realm — hers alone.
        CatalogCollection(id: "col-kira-still-life", slug: "kira-still-life",
                          name: "Still Life", realm: .kira,
                          description: "Her still-life practice."),
        CatalogCollection(id: "col-kira-autocord", slug: "kira-autocord",
                          name: "Autocord Photography", parentID: "col-photography",
                          realm: .kira, description: "Her TLR practice, any film stock, any genre."),
        CatalogCollection(id: "col-kira-decoupage", slug: "kira-decoupage",
                          name: "Decoupage Designs", parentID: "col-decoupage",
                          realm: .kira, description: "Her tile designs."),
        CatalogCollection(id: "col-kira-nightlife", slug: "kira-nightlife",
                          name: "Nightlife", realm: .kira,
                          description: "Voyeuristic, fun, sometimes crazy."),
        CatalogCollection(id: "col-kira-erotic-portraiture", slug: "kira-erotic-portraiture",
                          name: "Erotic Portraiture", parentID: "col-adult",
                          realm: .kira, description: "Erotic film photography of her and her friends."),
        CatalogCollection(id: "col-kira-adult-scenes", slug: "kira-adult-scenes",
                          name: "Adult Scenes", parentID: "col-adult",
                          realm: .kira, description: "Stills and clips alike."),
        CatalogCollection(id: "col-kira-dreams-memories", slug: "kira-dreams-memories",
                          name: "Dreams & Memories", realm: .kira,
                          description: "The t2v work she makes to describe a dream or a memory."),
    ]

    public static func migrate(db: OpaquePointer?) throws {
        // Appended columns. A duplicate-column error means an earlier run already
        // added it, which is success — hence `try?`, matching the existing
        // `source` migration idiom at DAMStore.swift:623.
        for (name, type) in newColumns {
            _ = try? exec(db, "ALTER TABLE assets ADD COLUMN \(name) \(type)")
        }
        _ = try? exec(db, "UPDATE assets SET realm = 'shared' WHERE realm IS NULL")

        try exec(db, """
            CREATE TABLE IF NOT EXISTS asset_locations (
                asset_id TEXT NOT NULL,
                host TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime REAL NOT NULL,
                PRIMARY KEY (asset_id, host, path)
            )
            """)
        try exec(db, "CREATE INDEX IF NOT EXISTS idx_locations_path ON asset_locations(path)")

        try exec(db, """
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                parent_id TEXT,
                realm TEXT,
                description TEXT,
                created_at REAL NOT NULL
            )
            """)
        try exec(db, """
            CREATE TABLE IF NOT EXISTS asset_collections (
                asset_id TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                manual INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (asset_id, collection_id)
            )
            """)
        try exec(db, "CREATE INDEX IF NOT EXISTS idx_asset_collections_col ON asset_collections(collection_id)")

        try exec(db, """
            CREATE TABLE IF NOT EXISTS asset_edges (
                from_asset_id TEXT NOT NULL,
                to_asset_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                PRIMARY KEY (from_asset_id, to_asset_id, relation)
            )
            """)
        try exec(db, "CREATE INDEX IF NOT EXISTS idx_edges_to ON asset_edges(to_asset_id)")

        try exec(db, "CREATE INDEX IF NOT EXISTS idx_assets_realm ON assets(realm, created_at DESC)")

        try migrateFTS(db: db)
        try seed(db: db)
    }

    /// FTS5 has no ALTER. Rebuild the virtual table with the caption column and
    /// repopulate from `assets`, which is the source of truth for the text.
    private static func migrateFTS(db: OpaquePointer?) throws {
        if ftsHasCaption(db: db) { return }
        try exec(db, "DROP TABLE IF EXISTS assets_fts")
        try exec(db, """
            CREATE VIRTUAL TABLE assets_fts USING fts5(
                id UNINDEXED, prompt, negative_prompt, caption
            )
            """)
        try exec(db, """
            INSERT INTO assets_fts (id, prompt, negative_prompt, caption)
            SELECT id, COALESCE(prompt, ''), COALESCE(negative_prompt, ''), COALESCE(caption, '')
            FROM assets WHERE sealed = 0
            """)
    }

    private static func ftsHasCaption(db: OpaquePointer?) -> Bool {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "PRAGMA table_info(assets_fts)", -1, &stmt, nil) == SQLITE_OK else {
            return false
        }
        defer { sqlite3_finalize(stmt) }
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 1), String(cString: c) == "caption" { return true }
        }
        return false
    }

    private static func seed(db: OpaquePointer?) throws {
        let now = Date().timeIntervalSince1970
        for c in seedCollections {
            let sql = """
                INSERT OR IGNORE INTO collections (id, slug, name, parent_id, realm, description, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw CatalogSchemaError.execFailed(sql, String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }
            bindText(stmt, 1, c.id)
            bindText(stmt, 2, c.slug)
            bindText(stmt, 3, c.name)
            bindText(stmt, 4, c.parentID)
            bindText(stmt, 5, c.realm?.rawValue)
            bindText(stmt, 6, c.description)
            sqlite3_bind_double(stmt, 7, now)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw CatalogSchemaError.execFailed(sql, String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    @discardableResult
    static func exec(_ db: OpaquePointer?, _ sql: String) throws -> Bool {
        var errMsg: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if rc != SQLITE_OK {
            let msg = errMsg.flatMap { String(cString: $0) } ?? "unknown"
            sqlite3_free(errMsg)
            throw CatalogSchemaError.execFailed(sql, msg)
        }
        return true
    }

    static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    static func bindText(_ stmt: OpaquePointer?, _ idx: Int32, _ value: String?) {
        if let v = value {
            sqlite3_bind_text(stmt, idx, (v as NSString).utf8String, -1, transient)
        } else {
            sqlite3_bind_null(stmt, idx)
        }
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `swift test --filter CatalogSchemaTests`
Expected: PASS, 6 tests.

- [ ] **Step 5: Dry-run the migration against a COPY of production**

Never migrate the live DB in a test step.

```bash
cp ~/.comfybox/dam.sqlite3 /tmp/dam-migration-check.sqlite3
swift run --product ComfyBoxGallery 2>/dev/null || true   # target must build
sqlite3 /tmp/dam-migration-check.sqlite3 "SELECT COUNT(*) FROM assets;"
```

Expected: `1072`. Keep this copy — Task 5 migrates it for real.

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxCatalog/CatalogSchema.swift Tests/ComfyBoxCatalogTests/CatalogSchemaTests.swift
git commit -m "feat(catalog): additive schema migration, edges/collections tables, seed collections"
```

---

### Task 3: The rule table that files assets into collections

**Files:**
- Create: `Sources/ComfyBoxCatalog/CollectionRules.swift`
- Test: `Tests/ComfyBoxCatalogTests/CollectionRulesTests.swift`

**Interfaces:**
- Consumes: `CatalogAsset`, `CatalogRealm`, `CatalogSchema.seedCollections`.
- Produces: `CollectionRules.defaultCollectionIDs(for:) -> [String]`.

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxCatalogTests/CollectionRulesTests.swift`:

```swift
import XCTest
@testable import ComfyBoxCatalog

final class CollectionRulesTests: XCTestCase {

    private func asset(realm: CatalogRealm = .kira, lane: String? = nil,
                       source: String? = nil, family: String? = nil,
                       mode: String? = nil, kind: String = "image") -> CatalogAsset {
        CatalogAsset(kind: kind, filename: "x", absolutePath: "/tmp/x",
                     realm: realm, source: source, lane: lane, family: family, mode: mode)
    }

    func testKiraLanesFileIntoHerGenres() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "still")),
                       ["col-kira-still-life"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "shoot")),
                       ["col-kira-autocord"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "film-erotic")),
                       ["col-kira-erotic-portraiture"])
    }

    func testHerTileWorkJoinsBothHerGenreAndTheSharedBody() {
        let ids = CollectionRules.defaultCollectionIDs(for: asset(lane: "tile"))
        XCTAssertEqual(Set(ids), ["col-kira-decoupage", "col-decoupage"],
                       "her decoupage is hers AND part of the shared body of work")
    }

    func testNightlifeFamilySplitsOutOfTheKiraLane() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "kira", family: "nightlife")),
                       ["col-kira-nightlife"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "kira")),
                       ["col-kira-adult-scenes"])
    }

    func testDreamsAndMemoriesIsT2VOnly() {
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(lane: "video", mode: "t2v", kind: "video")),
            ["col-kira-dreams-memories"])
        // An i2v clip is not a dream — it belongs to whatever genre its scene is.
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(lane: "kira", mode: "i2v", kind: "video")),
            ["col-kira-adult-scenes"])
    }

    func testSharedProducersFileIntoSharedBodiesOnly() {
        for src in ["desktop-decoupage", "tile-engine", "krita"] {
            XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, source: src)),
                           ["col-decoupage"], "\(src) contributes to the shared decoupage body")
        }
    }

    func testASharedAssetNeverFilesIntoAKiraCollection() {
        let kiraCollectionIDs = Set(CatalogSchema.seedCollections
            .filter { $0.realm == .kira }.map(\.id))
        for lane in ["still", "shoot", "tile", "kira", "film-erotic", "video"] {
            let ids = Set(CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, lane: lane)))
            XCTAssertTrue(ids.isDisjoint(with: kiraCollectionIDs),
                          "shared asset on lane \(lane) leaked into a kira collection")
        }
    }

    func testUnknownInputFilesNowhereRatherThanGuessing() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "no-such-lane")), [])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset()), [])
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter CollectionRulesTests`
Expected: FAIL — `CollectionRules` is undefined.

- [ ] **Step 3: Write the rules**

Create `Sources/ComfyBoxCatalog/CollectionRules.swift`:

```swift
// CollectionRules.swift — derived filing: lane / source / family / mode → the
// collections an asset lands in when nobody names one.
//
// Pure and table-driven on purpose. This is the layer that lets the Studio bot,
// Krita and the desktop tool file into the same bodies of work as Kira WITHOUT
// adopting her scheduler's vocabulary — each producer keeps its own words and
// the table translates.
//
// Precedence lives in CatalogStore: manual > explicit > derived. This function
// is only the derived tier, and returns [] rather than guessing.

import Foundation

public enum CollectionRules {

    /// Kira's lane → her genre. `kira` lane is disambiguated by `family`.
    private static let kiraLaneToCollection: [String: String] = [
        "still": "col-kira-still-life",
        "shoot": "col-kira-autocord",
        "tile": "col-kira-decoupage",
        "film-erotic": "col-kira-erotic-portraiture",
        "kira": "col-kira-adult-scenes",
    ]

    /// Her genres that are also part of a shared, cross-producer body of work.
    private static let alsoShared: [String: String] = [
        "col-kira-decoupage": "col-decoupage",
    ]

    /// Non-Kira producers → the shared body they contribute to.
    private static let sourceToSharedCollection: [String: String] = [
        "desktop-decoupage": "col-decoupage",
        "tile-engine": "col-decoupage",
        "krita": "col-decoupage",
        "studio-tile": "col-decoupage",
    ]

    public static func defaultCollectionIDs(for asset: CatalogAsset) -> [String] {
        switch asset.realm {
        case .kira:
            return kiraCollectionIDs(for: asset)
        case .shared:
            // A shared asset can only ever land in a shared collection.
            guard let source = asset.source,
                  let id = sourceToSharedCollection[source] else { return [] }
            return [id]
        }
    }

    private static func kiraCollectionIDs(for asset: CatalogAsset) -> [String] {
        guard let lane = asset.lane else { return [] }

        // t2v on the video lane is Dreams & Memories — the one genre that is
        // only video. An i2v clip belongs to the genre of the scene it animates.
        if lane == "video" {
            return asset.mode == "t2v" ? ["col-kira-dreams-memories"] : []
        }

        if lane == "kira", asset.family == "nightlife" {
            return ["col-kira-nightlife"]
        }

        guard let own = kiraLaneToCollection[lane] else { return [] }
        if let shared = alsoShared[own] { return [own, shared] }
        return [own]
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `swift test --filter CollectionRulesTests`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/ComfyBoxCatalog/CollectionRules.swift Tests/ComfyBoxCatalogTests/CollectionRulesTests.swift
git commit -m "feat(catalog): table-driven collection filing rules"
```

---

### Task 4: Metadata readers — exiftool, JSON sidecars, container probe

**Files:**
- Create: `Sources/ComfyBoxCatalog/MetadataReader.swift`
- Test: `Tests/ComfyBoxCatalogTests/MetadataReaderTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces: `struct FileMetadata` and `MetadataReader.readEmbedded(path:) -> FileMetadata?`, `MetadataReader.readSidecar(jsonData:) -> FileMetadata?`, `MetadataReader.parseUserComment(_:) -> FileMetadata`.

**Background the implementer needs:** images written by the engine carry the full generation record in `EXIF:UserComment` as JSON, plus the prompt in `EXIF:ImageDescription` and `XMP-dc:Description`. Videos carry **none of this** — an `.mp4` has only QuickTime `CreateDate`, `Duration` and dimensions — so video metadata comes exclusively from the JSON sidecar written server-side. Sidecars sometimes record `"duration": null`, so duration is always probed from the container rather than trusted.

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxCatalogTests/MetadataReaderTests.swift`:

```swift
import XCTest
@testable import ComfyBoxCatalog

final class MetadataReaderTests: XCTestCase {

    /// Verbatim shape of a real EXIF:UserComment written by the engine.
    func testParsesEngineUserCommentJSON() throws {
        let json = """
        {"width":896,"prompt":"a woman by a window","loras":[{"name":"KNPV4.1_pre","scale":1},\
        {"scale":0.35,"name":"Filipina_Pinay_Women"}],"seed":2090500631,"steps":9,\
        "height":1664,"guidance":0,"model":"krea-2-turbo"}
        """
        let m = MetadataReader.parseUserComment(json)
        XCTAssertEqual(m.prompt, "a woman by a window")
        XCTAssertEqual(m.seed, 2090500631)
        XCTAssertEqual(m.steps, 9)
        XCTAssertEqual(m.width, 896)
        XCTAssertEqual(m.height, 1664)
        XCTAssertEqual(m.guidance, 0)
        XCTAssertEqual(m.modelFamily, "krea-2-turbo")
        XCTAssertNotNil(m.loras)
        XCTAssertTrue(m.loras!.contains("KNPV4.1_pre"))
    }

    func testGarbageUserCommentYieldsEmptyMetadataNotACrash() {
        let m = MetadataReader.parseUserComment("not json at all {{{")
        XCTAssertNil(m.prompt)
        XCTAssertNil(m.seed)
    }

    /// Verbatim shape of a real IMAGE sidecar from ~/.kira/studio/metadata.
    func testParsesImageSidecar() throws {
        let json = """
        {"character":"kira","category":"generated","provider":"comfybox","tier":"standard",
         "generated_at":"2026-07-22T05:35:45.468Z","durationMs":84944,"width":576,"height":1024,
         "seed":1387857967,"content_mode":"avocado","model":"krea2","preset":"krea-kira",
         "guidance":0,"sealed":false,
         "loras":[{"path":"KNPV4.1_pre.safetensors","scale":1},{"path":"Filipina_Pinay_Women.safetensors","scale":0.35}]}
        """
        let m = try XCTUnwrap(MetadataReader.readSidecar(jsonData: Data(json.utf8)))
        XCTAssertEqual(m.characterName, "kira")
        XCTAssertEqual(m.contentMode, "avocado")
        XCTAssertEqual(m.preset, "krea-kira")
        XCTAssertEqual(m.modelFamily, "krea2")
        XCTAssertEqual(m.seed, 1387857967)
        XCTAssertFalse(m.sealed)
    }

    /// Verbatim shape of a real VIDEO sidecar — note source_image and prompt_raw.
    func testParsesVideoSidecarIncludingSourceImage() throws {
        let json = """
        {"character":"bree","mode":"i2v","duration":null,"provider":"comfybox","model":"ltx",
         "resolution":"480p","aspect_ratio":"9:16","content_mode":"avocado",
         "generated_at":"2026-07-11T12:17:40.506Z","prompt":"optimized text","prompt_raw":"original text",
         "source_image":"/home/todd/.kira/studio/gallery/Bree/generated/1783770983068_x.png"}
        """
        let m = try XCTUnwrap(MetadataReader.readSidecar(jsonData: Data(json.utf8)))
        XCTAssertEqual(m.mode, "i2v")
        XCTAssertEqual(m.resolution, "480p")
        XCTAssertEqual(m.aspectRatio, "9:16")
        XCTAssertEqual(m.prompt, "optimized text")
        XCTAssertEqual(m.promptRaw, "original text")
        XCTAssertEqual(m.sourceImagePath,
                       "/home/todd/.kira/studio/gallery/Bree/generated/1783770983068_x.png")
        XCTAssertNil(m.durationMs, "sidecar duration is null — must NOT be invented")
    }

    func testSealedSidecarIsFlagged() throws {
        let json = #"{"character":"bree","sealed":true,"content_mode":"apple"}"#
        let m = try XCTUnwrap(MetadataReader.readSidecar(jsonData: Data(json.utf8)))
        XCTAssertTrue(m.sealed)
    }

    func testUnreadableSidecarReturnsNilRatherThanThrowing() {
        XCTAssertNil(MetadataReader.readSidecar(jsonData: Data("]]not json[[".utf8)))
        XCTAssertNil(MetadataReader.readSidecar(jsonData: Data()))
    }

    func testSidecarPathMirrorsMediaPath() {
        XCTAssertEqual(
            MetadataReader.sidecarPath(
                forMedia: "/home/todd/.kira/studio/gallery/Kira/generated/x.png",
                galleryRoot: "/home/todd/.kira/studio/gallery",
                metadataRoot: "/home/todd/.kira/studio/metadata"),
            "/home/todd/.kira/studio/metadata/Kira/generated/x.json")
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter MetadataReaderTests`
Expected: FAIL — `MetadataReader` is undefined.

- [ ] **Step 3: Write the reader**

Create `Sources/ComfyBoxCatalog/MetadataReader.swift`:

```swift
// MetadataReader.swift — pull generation facts out of the artifacts on disk.
//
// Three sources, in descending durability:
//   1. Embedded EXIF/XMP (IMAGES ONLY) — rides the file everywhere, including
//      the copy to the Linux server. `EXIF:UserComment` holds the full JSON
//      generation record; ImageDescription/XMP hold the prompt.
//   2. JSON sidecar (images AND video) — server-side mirror tree. For VIDEO
//      this is the ONLY source: an .mp4 carries no generation metadata at all.
//   3. Container probe — duration/fps/frames, because sidecars sometimes say
//      "duration": null and must not be trusted for it.
//
// Every read is best-effort: a missing tool, a missing file or garbage JSON
// yields fewer facts, never a throw.

import Foundation

public struct FileMetadata: Sendable, Equatable {
    public var prompt: String?
    public var negativePrompt: String?
    public var promptRaw: String?
    public var seed: Int?
    public var steps: Int?
    public var guidance: Double?
    public var width: Int?
    public var height: Int?
    public var modelFamily: String?
    public var preset: String?
    public var loras: String?
    public var characterName: String?
    public var contentMode: String?
    public var lane: String?
    public var mode: String?
    public var resolution: String?
    public var aspectRatio: String?
    public var durationMs: Int?
    public var sealed: Bool = false
    public var sourceImagePath: String?
    public var software: String?

    public init() {}
}

public enum MetadataReader {

    public static let exiftoolPath = "/opt/homebrew/bin/exiftool"

    // MARK: - Embedded (images)

    /// Read embedded EXIF/XMP via exiftool. Returns nil when exiftool is absent
    /// or the file has nothing — never throws.
    public static func readEmbedded(path: String) -> FileMetadata? {
        guard FileManager.default.isExecutableFile(atPath: exiftoolPath) else { return nil }
        let args = ["-j", "-n",
                    "-EXIF:UserComment", "-EXIF:ImageDescription", "-EXIF:Software",
                    "-XMP-dc:Description", "-IPTC:Keywords", "-XMP-dc:Subject",
                    path]
        guard let out = runTool(exiftoolPath, args),
              let arr = try? JSONSerialization.jsonObject(with: out) as? [[String: Any]],
              let obj = arr.first else { return nil }

        var m = FileMetadata()
        if let uc = obj["UserComment"] as? String {
            m = parseUserComment(uc)
        }
        if m.prompt == nil {
            m.prompt = (obj["ImageDescription"] as? String) ?? (obj["Description"] as? String)
        }
        m.software = obj["Software"] as? String
        return m
    }

    /// Parse the engine's `EXIF:UserComment` JSON blob.
    public static func parseUserComment(_ raw: String) -> FileMetadata {
        var m = FileMetadata()
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return m }
        m.prompt = obj["prompt"] as? String
        m.negativePrompt = obj["negative_prompt"] as? String
        m.seed = intValue(obj["seed"])
        m.steps = intValue(obj["steps"])
        m.guidance = doubleValue(obj["guidance"])
        m.width = intValue(obj["width"])
        m.height = intValue(obj["height"])
        m.modelFamily = obj["model"] as? String
        if let loras = obj["loras"], let d = try? JSONSerialization.data(withJSONObject: loras) {
            m.loras = String(data: d, encoding: .utf8)
        }
        return m
    }

    // MARK: - Sidecar (images and video)

    public static func readSidecar(jsonData: Data) -> FileMetadata? {
        guard !jsonData.isEmpty,
              let obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        else { return nil }

        var m = FileMetadata()
        m.prompt = obj["prompt"] as? String
        m.promptRaw = obj["prompt_raw"] as? String
        m.negativePrompt = obj["negative_prompt"] as? String
        m.seed = intValue(obj["seed"])
        m.steps = intValue(obj["steps"])
        m.guidance = doubleValue(obj["guidance"])
        m.width = intValue(obj["width"])
        m.height = intValue(obj["height"])
        m.modelFamily = obj["model"] as? String
        m.preset = obj["preset"] as? String
        m.characterName = obj["character"] as? String
        m.contentMode = obj["content_mode"] as? String
        m.mode = obj["mode"] as? String
        m.resolution = obj["resolution"] as? String
        m.aspectRatio = obj["aspect_ratio"] as? String
        m.sourceImagePath = obj["source_image"] as? String
        m.sealed = (obj["sealed"] as? Bool) ?? false
        if let loras = obj["loras"], let d = try? JSONSerialization.data(withJSONObject: loras) {
            m.loras = String(data: d, encoding: .utf8)
        }
        // `duration` is deliberately NOT read: it is null in real sidecars.
        // Duration comes from probeContainer.
        return m
    }

    /// Map a media path in the gallery tree to its sidecar in the mirror tree.
    public static func sidecarPath(forMedia media: String,
                                   galleryRoot: String,
                                   metadataRoot: String) -> String? {
        guard media.hasPrefix(galleryRoot) else { return nil }
        let rel = String(media.dropFirst(galleryRoot.count))
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let base = (rel as NSString).deletingPathExtension
        return (metadataRoot as NSString).appendingPathComponent(base + ".json")
    }

    // MARK: - Container probe (video)

    public struct ContainerInfo: Sendable, Equatable {
        public let durationMs: Int?
        public let fps: Double?
        public let frames: Int?
        public let width: Int?
        public let height: Int?
    }

    /// Probe duration/fps/frames from the container itself. Uses exiftool so we
    /// take no new dependency; ffprobe is not assumed to be installed.
    public static func probeContainer(path: String) -> ContainerInfo? {
        guard FileManager.default.isExecutableFile(atPath: exiftoolPath) else { return nil }
        let args = ["-j", "-n", "-QuickTime:Duration", "-VideoFrameRate",
                    "-ImageWidth", "-ImageHeight", path]
        guard let out = runTool(exiftoolPath, args),
              let arr = try? JSONSerialization.jsonObject(with: out) as? [[String: Any]],
              let obj = arr.first else { return nil }

        let seconds = doubleValue(obj["Duration"])
        let fps = doubleValue(obj["VideoFrameRate"])
        let ms = seconds.map { Int(($0 * 1000).rounded()) }
        let frames: Int? = {
            guard let s = seconds, let f = fps, f > 0 else { return nil }
            return Int((s * f).rounded())
        }()
        return ContainerInfo(durationMs: ms, fps: fps, frames: frames,
                             width: intValue(obj["ImageWidth"]),
                             height: intValue(obj["ImageHeight"]))
    }

    // MARK: - Helpers

    private static func runTool(_ launchPath: String, _ args: [String]) -> Data? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return data.isEmpty ? nil : data
    }

    private static func intValue(_ any: Any?) -> Int? {
        if let i = any as? Int { return i }
        if let d = any as? Double { return Int(d) }
        if let s = any as? String { return Int(s) }
        return nil
    }

    private static func doubleValue(_ any: Any?) -> Double? {
        if let d = any as? Double { return d }
        if let i = any as? Int { return Double(i) }
        if let s = any as? String { return Double(s) }
        return nil
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `swift test --filter MetadataReaderTests`
Expected: PASS, 7 tests.

- [ ] **Step 5: Sanity-check against one real file each**

```bash
swift run --product ComfyBoxGallery --help 2>/dev/null || true
/opt/homebrew/bin/exiftool -j -n -EXIF:UserComment "$(ls -t ~/Pictures/ComfyBox/*.png | head -1)" | head -5
/opt/homebrew/bin/exiftool -j -n -QuickTime:Duration -VideoFrameRate "$(ls -t ~/Pictures/ComfyBox/*.mp4 | head -1)"
```

Expected: the PNG returns a `UserComment` JSON string; the MP4 returns `Duration` and `VideoFrameRate` and **no** generation fields. That asymmetry is the whole reason video is handled separately — confirm it before moving on.

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxCatalog/MetadataReader.swift Tests/ComfyBoxCatalogTests/MetadataReaderTests.swift
git commit -m "feat(catalog): embedded/sidecar/container metadata readers"
```

---

### Task 5: CatalogStore — upsert, realm-scoped search, collections, edges

**Files:**
- Create: `Sources/ComfyBoxCatalog/CatalogStore.swift`
- Test: `Tests/ComfyBoxCatalogTests/CatalogStoreTests.swift`

**Interfaces:**
- Consumes: `CatalogSchema`, `CatalogAsset`, `CatalogCollection`, `AssetEdge`, `CollectionRules`.
- Produces: `actor CatalogStore` with
  `open(path:) async throws -> CatalogStore`,
  `upsert(_ asset: CatalogAsset, explicitCollectionIDs: [String]) throws`,
  `search(_ query: CatalogQuery) throws -> [CatalogAsset]`,
  `facets(scope: CatalogRealm?) throws -> CatalogFacets`,
  `collections(visibleTo: CatalogRealm?) throws -> [CatalogCollection]`,
  `createCollection(_:by:) throws`, `renameCollection(id:name:by:) throws`,
  `retireCollection(id:by:) throws`, `file(assetID:into:by:) throws`,
  `addEdge(_:) throws`, `edges(for:) throws -> [AssetEdge]`,
  `addLocation(assetID:_:) throws`, `locations(of:) throws -> [AssetLocation]`;
  plus `struct CatalogQuery` and `struct CatalogFacets`.

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxCatalogTests/CatalogStoreTests.swift`:

```swift
import XCTest
@testable import ComfyBoxCatalog

final class CatalogStoreTests: XCTestCase {
    private var path: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        path = NSTemporaryDirectory() + "store-test-\(UUID().uuidString).sqlite3"
        store = try await CatalogStore.open(path: path)
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: path)
        try await super.tearDown()
    }

    private func make(_ id: String, realm: CatalogRealm, lane: String? = nil,
                      tier: String? = nil, kind: String = "image",
                      prompt: String? = nil, sealed: Bool = false,
                      collection: String? = nil, family: String? = nil,
                      mode: String? = nil, source: String? = nil) -> CatalogAsset {
        CatalogAsset(id: id, kind: kind, filename: "\(id).png",
                     absolutePath: "/tmp/\(id).png",
                     realm: realm, source: source, sealed: sealed,
                     prompt: prompt, contentMode: tier, lane: lane,
                     family: family, mode: mode)
    }

    // MARK: - Realm isolation (the invariant that matters most)

    func testKiraScopedSearchNeverReturnsASharedRow() async throws {
        try await store.upsert(make("k1", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared, lane: "shoot"), explicitCollectionIDs: [])

        // Every filter combination we expose must respect the lock.
        let variations: [CatalogQuery] = [
            CatalogQuery(scope: .kira),
            CatalogQuery(scope: .kira, lane: "shoot"),
            CatalogQuery(scope: .kira, collectionID: "col-photography"),
            CatalogQuery(scope: .kira, kind: "image"),
            CatalogQuery(scope: .kira, minRating: 0),
            CatalogQuery(scope: .kira, orderBy: .oldest),
        ]
        for q in variations {
            let rows = try await store.search(q)
            XCTAssertFalse(rows.contains { $0.realm != .kira },
                           "leaked a non-kira row for query \(q)")
        }
    }

    func testScopeNilSeesEverything() async throws {
        try await store.upsert(make("k1", realm: .kira), explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared), explicitCollectionIDs: [])
        let rows = try await store.search(CatalogQuery(scope: nil))
        XCTAssertEqual(Set(rows.map(\.id)), ["k1", "s1"])
    }

    // MARK: - Mode clamp

    func testTierCeilingHidesTextAndPathButKeepsCounts() async throws {
        try await store.upsert(make("a", realm: .kira, tier: "avocado", prompt: "explicit text"),
                               explicitCollectionIDs: [])
        try await store.upsert(make("n", realm: .kira, tier: "neutral", prompt: "a tulip"),
                               explicitCollectionIDs: [])

        let clamped = try await store.search(CatalogQuery(scope: .kira, ceiling: "apple"))
        let avocado = try XCTUnwrap(clamped.first { $0.id == "a" })
        XCTAssertNil(avocado.prompt, "text above the ceiling must not surface")
        XCTAssertEqual(avocado.absolutePath, "", "path above the ceiling must not surface")
        XCTAssertEqual(avocado.contentMode, "avocado", "the tier LABEL is metadata and stays")

        let neutral = try XCTUnwrap(clamped.first { $0.id == "n" })
        XCTAssertEqual(neutral.prompt, "a tulip")
        XCTAssertEqual(neutral.absolutePath, "/tmp/n.png")
    }

    func testNoCeilingMeansNoClamp() async throws {
        try await store.upsert(make("a", realm: .kira, tier: "avocado", prompt: "explicit text"),
                               explicitCollectionIDs: [])
        let rows = try await store.search(CatalogQuery(scope: .kira, ceiling: nil))
        XCTAssertEqual(rows.first?.prompt, "explicit text")
    }

    // MARK: - Sealed

    func testSealedRowStoresNoTextAndIsNotFullTextSearchable() async throws {
        try await store.upsert(make("sealed1", realm: .shared, prompt: "hunting-phrase", sealed: true),
                               explicitCollectionIDs: [])
        let byID = try await store.search(CatalogQuery(scope: nil))
        XCTAssertNil(byID.first { $0.id == "sealed1" }?.prompt)
        let byText = try await store.search(CatalogQuery(scope: nil, text: "hunting-phrase"))
        XCTAssertTrue(byText.isEmpty, "a sealed row must not be reachable by its text")
    }

    // MARK: - Collections

    func testDerivedFilingHappensOnUpsert() async throws {
        try await store.upsert(make("t1", realm: .kira, lane: "tile"), explicitCollectionIDs: [])
        let hers = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        let shared = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-decoupage"))
        XCTAssertEqual(hers.map(\.id), ["t1"])
        XCTAssertEqual(shared.map(\.id), ["t1"], "her tile work is in the shared body too")
    }

    func testAnAssetInTwoCollectionsIsReturnedByEitherAndCountedOnce() async throws {
        try await store.upsert(make("x", realm: .kira, lane: "shoot"),
                               explicitCollectionIDs: ["col-kira-erotic-portraiture"])
        let a = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-autocord"))
        let b = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-erotic-portraiture"))
        XCTAssertEqual(a.map(\.id), ["x"])
        XCTAssertEqual(b.map(\.id), ["x"])
        XCTAssertEqual(a.count, 1)
    }

    func testCollectionQueryMatchesRootAndChildren() async throws {
        try await store.upsert(make("auto", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        let roll = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-photography"))
        XCTAssertEqual(roll.map(\.id), ["auto"],
                       "asking for Photography must include Autocord Still Life")
    }

    func testDepthCapRejectsAThirdLevel() async throws {
        let child = CatalogCollection(id: "c-child", slug: "child", name: "Child",
                                      parentID: "col-kira-autocord", realm: .kira)
        await XCTAssertThrowsErrorAsync(try await store.createCollection(child, by: .kira))
    }

    func testKiraCannotRestructureASharedCollection() async throws {
        await XCTAssertThrowsErrorAsync(
            try await store.renameCollection(id: "col-decoupage", name: "Hers Now", by: .kira))
        await XCTAssertThrowsErrorAsync(
            try await store.retireCollection(id: "col-photography", by: .kira))
        await XCTAssertThrowsErrorAsync(
            try await store.createCollection(
                CatalogCollection(slug: "new-shared", name: "New Shared", realm: nil), by: .kira))
    }

    func testKiraCanCurateHerOwnRealm() async throws {
        let mine = CatalogCollection(id: "c-mine", slug: "kira-rainy-days",
                                     name: "Rainy Days", realm: .kira)
        try await store.createCollection(mine, by: .kira)
        try await store.renameCollection(id: "c-mine", name: "Rain", by: .kira)
        let visible = try await store.collections(visibleTo: .kira)
        XCTAssertEqual(visible.first { $0.id == "c-mine" }?.name, "Rain")
        try await store.retireCollection(id: "c-mine", by: .kira)
        XCTAssertNil(try await store.collections(visibleTo: .kira).first { $0.id == "c-mine" })
    }

    func testKiraCannotFileASharedRow() async throws {
        try await store.upsert(make("s1", realm: .shared), explicitCollectionIDs: [])
        await XCTAssertThrowsErrorAsync(
            try await store.file(assetID: "s1", into: "col-kira-still-life", by: .kira))
    }

    func testCollectionsVisibleToKiraExcludeOtherRealmsPrivateOnes() async throws {
        try await store.createCollection(
            CatalogCollection(id: "c-secret", slug: "x", name: "X", realm: .kira), by: .kira)
        let shared = try await store.collections(visibleTo: .shared)
        XCTAssertNil(shared.first { $0.id == "c-secret" },
                     "her private collections are hers")
        XCTAssertNotNil(shared.first { $0.id == "col-decoupage" })
    }

    // MARK: - Edges

    func testI2VSourceResolvesInBothDirections() async throws {
        try await store.upsert(make("still", realm: .kira), explicitCollectionIDs: [])
        try await store.upsert(make("clip", realm: .kira, kind: "video", mode: "i2v"),
                               explicitCollectionIDs: [])
        try await store.addEdge(AssetEdge(fromAssetID: "clip", toAssetID: "still", relation: .i2vSource))

        let fromClip = try await store.edges(for: "clip")
        let fromStill = try await store.edges(for: "still")
        XCTAssertEqual(fromClip.map(\.toAssetID), ["still"])
        XCTAssertEqual(fromStill.map(\.fromAssetID), ["clip"],
                       "from the still, find its clips")
    }

    func testEdgesAreIdempotent() async throws {
        try await store.upsert(make("a", realm: .kira), explicitCollectionIDs: [])
        try await store.upsert(make("b", realm: .kira), explicitCollectionIDs: [])
        let e = AssetEdge(fromAssetID: "a", toAssetID: "b", relation: .memberOf)
        try await store.addEdge(e)
        try await store.addEdge(e)
        XCTAssertEqual(try await store.edges(for: "a").count, 1)
    }

    // MARK: - Locations

    func testOneAssetManyLocations() async throws {
        try await store.upsert(make("a", realm: .kira), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "a",
            AssetLocation(host: "mac", path: "/Users/t/Pictures/ComfyBox/a.png", mtime: Date()))
        try await store.addLocation(assetID: "a",
            AssetLocation(host: "kira", path: "/home/todd/.kira/studio/gallery/Kira/a.png", mtime: Date()))
        XCTAssertEqual(try await store.locations(of: "a").count, 2)
    }

    // MARK: - Facets

    func testFacetCountsAreRealmScoped() async throws {
        try await store.upsert(make("k1", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("k2", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared, lane: "shoot"), explicitCollectionIDs: [])
        let f = try await store.facets(scope: .kira)
        XCTAssertEqual(f.lane["shoot"], 2)
    }
}

/// XCTAssertThrowsError has no async form in this toolchain.
func XCTAssertThrowsErrorAsync(_ expression: @autoclosure () async throws -> Void,
                               file: StaticString = #filePath, line: UInt = #line) async {
    do {
        try await expression()
        XCTFail("expected an error", file: file, line: line)
    } catch {
        // expected
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter CatalogStoreTests`
Expected: FAIL — `CatalogStore` is undefined.

- [ ] **Step 3: Write the store**

Create `Sources/ComfyBoxCatalog/CatalogStore.swift`. This is the largest file in the plan; write it in the order the tests exercise it.

```swift
// CatalogStore.swift — the query layer over ~/.comfybox/dam.sqlite3.
//
// Two rules are enforced HERE, in the store, rather than in any caller:
//
//   1. REALM LOCK. `CatalogQuery.scope` is set by the service from which tool
//      is calling — never from client input. When scope == .kira the WHERE
//      clause carries `realm = 'kira'` unconditionally, so no combination of
//      other filters can widen it.
//   2. MODE CLAMP. `ceiling` withholds prompt/caption/path for rows above the
//      active chat mode while KEEPING the tier label, matching the precedent in
//      render-journal.ts (counts are metadata; text and paths are not).
//
// Both are properties of the returned rows, not of the caller's discipline.

import Foundation
import SQLite3

public enum CatalogError: Error, LocalizedError {
    case prepareFailed(String)
    case stepFailed(String)
    case depthCapExceeded
    case notPermitted(String)
    case noSuchCollection(String)

    public var errorDescription: String? {
        switch self {
        case let .prepareFailed(m): return "prepare failed: \(m)"
        case let .stepFailed(m): return "step failed: \(m)"
        case .depthCapExceeded: return "collections are two levels deep at most"
        case let .notPermitted(m): return "not permitted: \(m)"
        case let .noSuchCollection(id): return "no such collection: \(id)"
        }
    }
}

public enum CatalogOrder: String, Sendable {
    case newest, oldest, rating
}

public struct CatalogQuery: Sendable, CustomStringConvertible {
    /// nil = every realm. Set by the SERVICE, never by client input.
    public var scope: CatalogRealm?
    /// The active chat-mode ceiling; nil = no clamp.
    public var ceiling: String?
    public var text: String?
    public var collectionID: String?
    public var lane: String?
    public var tier: String?
    public var character: String?
    public var source: String?
    public var stock: String?
    public var genre: String?
    public var arc: String?
    public var kind: String?
    public var mode: String?
    public var minDurationMs: Int?
    public var maxDurationMs: Int?
    public var minRating: Int?
    public var since: Date?
    public var until: Date?
    public var orderBy: CatalogOrder = .newest
    public var limit: Int = 50
    public var offset: Int = 0

    public init(scope: CatalogRealm? = nil, ceiling: String? = nil, text: String? = nil,
                collectionID: String? = nil, lane: String? = nil, tier: String? = nil,
                character: String? = nil, source: String? = nil, stock: String? = nil,
                genre: String? = nil, arc: String? = nil, kind: String? = nil,
                mode: String? = nil, minDurationMs: Int? = nil, maxDurationMs: Int? = nil,
                minRating: Int? = nil, since: Date? = nil, until: Date? = nil,
                orderBy: CatalogOrder = .newest, limit: Int = 50, offset: Int = 0) {
        self.scope = scope; self.ceiling = ceiling; self.text = text
        self.collectionID = collectionID; self.lane = lane; self.tier = tier
        self.character = character; self.source = source; self.stock = stock
        self.genre = genre; self.arc = arc; self.kind = kind; self.mode = mode
        self.minDurationMs = minDurationMs; self.maxDurationMs = maxDurationMs
        self.minRating = minRating; self.since = since; self.until = until
        self.orderBy = orderBy; self.limit = limit; self.offset = offset
    }

    public var description: String {
        "CatalogQuery(scope: \(scope?.rawValue ?? "all"), collection: \(collectionID ?? "-"), lane: \(lane ?? "-"))"
    }
}

public struct CatalogFacets: Sendable, Equatable {
    public var lane: [String: Int] = [:]
    public var tier: [String: Int] = [:]
    public var character: [String: Int] = [:]
    public var source: [String: Int] = [:]
    public var stock: [String: Int] = [:]
    public var genre: [String: Int] = [:]
    public var kind: [String: Int] = [:]
    public var mode: [String: Int] = [:]
    public var collection: [String: Int] = [:]
    public init() {}
}

/// The fruit tiers, least to most explicit. Used only for ceiling comparison.
public let CATALOG_TIER_ORDER: [String] = ["neutral", "apple", "banana", "avocado"]

public func tierRank(_ tier: String?) -> Int {
    guard let t = tier?.lowercased(), let i = CATALOG_TIER_ORDER.firstIndex(of: t) else { return 0 }
    return i
}

public actor CatalogStore {
    private var db: OpaquePointer?
    public let dbPath: String

    private init(db: OpaquePointer, dbPath: String) {
        self.db = db
        self.dbPath = dbPath
    }

    public static func open(path: String? = nil) async throws -> CatalogStore {
        let resolved: String
        if let p = path {
            resolved = p
        } else {
            let dir = NSString(string: "~/.comfybox").expandingTildeInPath
            try FileManager.default.createDirectory(
                atPath: dir, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            resolved = (dir as NSString).appendingPathComponent("dam.sqlite3")
        }

        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(resolved, &handle, flags, nil) == SQLITE_OK, let h = handle else {
            let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            if let x = handle { sqlite3_close(x) }
            throw CatalogError.prepareFailed(msg)
        }

        let store = CatalogStore(db: h, dbPath: resolved)
        try await store.initialize()
        return store
    }

    private func initialize() throws {
        try CatalogSchema.exec(db, "PRAGMA journal_mode=WAL")
        try CatalogSchema.migrate(db: db)
        tightenPermissions()
    }

    /// Catalog holds raw prompt text under the extended 2026-07-07 provenance
    /// contract: 0600 file inside a 0700 directory. Applied on every open so a
    /// file created by another process is corrected too.
    private func tightenPermissions() {
        let fm = FileManager.default
        try? fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: dbPath)
        for suffix in ["-wal", "-shm"] {
            try? fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: dbPath + suffix)
        }
        try? fm.setAttributes([.posixPermissions: 0o700],
                              ofItemAtPath: (dbPath as NSString).deletingLastPathComponent)
    }

    deinit { if let db = db { sqlite3_close(db) } }
}
```

- [ ] **Step 4: Add upsert, with derived + explicit filing**

Append to `CatalogStore.swift`, inside the actor:

```swift
    // MARK: - Write

    /// Insert or update an asset and (re-)apply its non-manual collection
    /// membership. Manual filings (`manual = 1`) are never removed here —
    /// precedence is manual > explicit > derived.
    public func upsert(_ asset: CatalogAsset, explicitCollectionIDs: [String]) throws {
        let sql = """
            INSERT INTO assets (
                id, kind, filename, absolute_path, file_size, sha256, width, height,
                created_at, modified_at, ingested_at, orphaned,
                prompt, negative_prompt, seed, steps, guidance, model_family,
                rating, favorite, content_mode, character_name, source,
                realm, sealed, lane, arc, theme, stock, genre, family, style,
                preset, loras, render_id, caption, caption_source, prompt_raw,
                mode, duration_ms, fps, frames, resolution, aspect_ratio
            ) VALUES (
                ?1,?2,?3,?4,?5,?6,?7,?8,
                ?9,?9,?9,0,
                ?10,?11,?12,?13,?14,?15,
                ?16,?17,?18,?19,?20,
                ?21,?22,?23,?24,?25,?26,?27,?28,?29,
                ?30,?31,?32,?33,?34,?35,
                ?36,?37,?38,?39,?40,?41
            )
            ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind, filename=excluded.filename,
                absolute_path=excluded.absolute_path, file_size=excluded.file_size,
                sha256=COALESCE(excluded.sha256, assets.sha256),
                width=COALESCE(excluded.width, assets.width),
                height=COALESCE(excluded.height, assets.height),
                prompt=excluded.prompt, negative_prompt=excluded.negative_prompt,
                seed=COALESCE(excluded.seed, assets.seed),
                steps=COALESCE(excluded.steps, assets.steps),
                guidance=COALESCE(excluded.guidance, assets.guidance),
                model_family=COALESCE(excluded.model_family, assets.model_family),
                content_mode=COALESCE(excluded.content_mode, assets.content_mode),
                character_name=COALESCE(excluded.character_name, assets.character_name),
                source=COALESCE(excluded.source, assets.source),
                realm=excluded.realm, sealed=excluded.sealed,
                lane=COALESCE(excluded.lane, assets.lane),
                arc=COALESCE(excluded.arc, assets.arc),
                theme=COALESCE(excluded.theme, assets.theme),
                stock=COALESCE(excluded.stock, assets.stock),
                genre=COALESCE(excluded.genre, assets.genre),
                family=COALESCE(excluded.family, assets.family),
                style=COALESCE(excluded.style, assets.style),
                preset=COALESCE(excluded.preset, assets.preset),
                loras=COALESCE(excluded.loras, assets.loras),
                render_id=COALESCE(excluded.render_id, assets.render_id),
                caption=excluded.caption, caption_source=excluded.caption_source,
                prompt_raw=excluded.prompt_raw,
                mode=COALESCE(excluded.mode, assets.mode),
                duration_ms=COALESCE(excluded.duration_ms, assets.duration_ms),
                fps=COALESCE(excluded.fps, assets.fps),
                frames=COALESCE(excluded.frames, assets.frames),
                resolution=COALESCE(excluded.resolution, assets.resolution),
                aspect_ratio=COALESCE(excluded.aspect_ratio, assets.aspect_ratio)
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }

        bindText(stmt, 1, asset.id)
        bindText(stmt, 2, asset.kind)
        bindText(stmt, 3, asset.filename)
        bindText(stmt, 4, asset.absolutePath)
        sqlite3_bind_int64(stmt, 5, asset.fileSize)
        bindText(stmt, 6, asset.sha256)
        bindInt(stmt, 7, asset.width)
        bindInt(stmt, 8, asset.height)
        sqlite3_bind_double(stmt, 9, asset.createdAt.timeIntervalSince1970)
        bindText(stmt, 10, asset.prompt)
        bindText(stmt, 11, asset.negativePrompt)
        bindInt(stmt, 12, asset.seed)
        bindInt(stmt, 13, asset.steps)
        bindDouble(stmt, 14, asset.guidance)
        bindText(stmt, 15, asset.modelFamily)
        sqlite3_bind_int(stmt, 16, Int32(asset.rating))
        sqlite3_bind_int(stmt, 17, asset.favorite ? 1 : 0)
        bindText(stmt, 18, asset.contentMode)
        bindText(stmt, 19, asset.characterName)
        bindText(stmt, 20, asset.source)
        bindText(stmt, 21, asset.realm.rawValue)
        sqlite3_bind_int(stmt, 22, asset.sealed ? 1 : 0)
        bindText(stmt, 23, asset.lane)
        bindText(stmt, 24, asset.arc)
        bindText(stmt, 25, asset.theme)
        bindText(stmt, 26, asset.stock)
        bindText(stmt, 27, asset.genre)
        bindText(stmt, 28, asset.family)
        bindText(stmt, 29, asset.style)
        bindText(stmt, 30, asset.preset)
        bindText(stmt, 31, asset.loras)
        bindText(stmt, 32, asset.renderID)
        bindText(stmt, 33, asset.caption)
        bindText(stmt, 34, asset.captionSource)
        bindText(stmt, 35, asset.promptRaw)
        bindText(stmt, 36, asset.mode)
        bindInt(stmt, 37, asset.durationMs)
        bindDouble(stmt, 38, asset.fps)
        bindInt(stmt, 39, asset.frames)
        bindText(stmt, 40, asset.resolution)
        bindText(stmt, 41, asset.aspectRatio)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw CatalogError.stepFailed(String(cString: sqlite3_errmsg(db)))
        }

        try reindexFTS(asset)
        try applyDerivedFiling(asset, explicitCollectionIDs: explicitCollectionIDs)
    }

    /// A sealed row is never full-text indexed — that is what makes it
    /// unreachable by its own prompt.
    private func reindexFTS(_ asset: CatalogAsset) throws {
        try execBind("DELETE FROM assets_fts WHERE id = ?1") { [weak self] s in
            self?.bindText(s, 1, asset.id)
        }
        guard !asset.sealed else { return }
        let hasText = (asset.prompt ?? asset.caption ?? asset.promptRaw) != nil
        guard hasText else { return }
        try execBind("""
            INSERT INTO assets_fts (id, prompt, negative_prompt, caption)
            VALUES (?1, ?2, ?3, ?4)
            """) { [weak self] s in
            guard let self else { return }
            self.bindText(s, 1, asset.id)
            // prompt_raw rides the prompt column so both phrasings are findable.
            self.bindText(s, 2, [asset.prompt, asset.promptRaw]
                .compactMap { $0 }.joined(separator: " "))
            self.bindText(s, 3, asset.negativePrompt ?? "")
            self.bindText(s, 4, asset.caption ?? "")
        }
    }

    private func applyDerivedFiling(_ asset: CatalogAsset, explicitCollectionIDs: [String]) throws {
        // Replace only non-manual memberships; manual filings survive.
        try execBind("DELETE FROM asset_collections WHERE asset_id = ?1 AND manual = 0") { [weak self] s in
            self?.bindText(s, 1, asset.id)
        }
        let derived = CollectionRules.defaultCollectionIDs(for: asset)
        let ids = Set(explicitCollectionIDs.isEmpty ? derived : explicitCollectionIDs)
        for cid in ids {
            // A shared asset can never enter a kira collection, whatever the caller says.
            if asset.realm == .shared, try collectionRealm(cid) == .kira { continue }
            try execBind("""
                INSERT OR IGNORE INTO asset_collections (asset_id, collection_id, manual)
                VALUES (?1, ?2, 0)
                """) { [weak self] s in
                guard let self else { return }
                self.bindText(s, 1, asset.id)
                self.bindText(s, 2, cid)
            }
        }
    }
```

- [ ] **Step 5: Add search with the realm lock and the mode clamp**

Append inside the actor:

```swift
    // MARK: - Read

    public func search(_ query: CatalogQuery) throws -> [CatalogAsset] {
        var wheres: [String] = []
        var binds: [(Int32) -> Void] = []
        var n: Int32 = 0
        func bind(_ f: @escaping (Int32) -> Void) { n += 1; binds.append(f) }

        // THE REALM LOCK. Unconditional, first, and not reachable from client input.
        if let scope = query.scope {
            wheres.append("a.realm = ?\(n + 1)")
            bind { [weak self] i in self?.bindText(self?.pending, i, scope.rawValue) }
        }
        if let t = query.text, !t.isEmpty {
            wheres.append("a.id IN (SELECT id FROM assets_fts WHERE assets_fts MATCH ?\(n + 1))")
            bind { [weak self] i in self?.bindText(self?.pending, i, t) }
        }
        if let c = query.collectionID {
            // Matches the collection AND its children — asking for Photography
            // returns Autocord Still Life too.
            wheres.append("""
                a.id IN (SELECT asset_id FROM asset_collections
                         WHERE collection_id = ?\(n + 1)
                            OR collection_id IN (SELECT id FROM collections WHERE parent_id = ?\(n + 1)))
                """)
            bind { [weak self] i in self?.bindText(self?.pending, i, c) }
        }
        func eq(_ column: String, _ value: String?) {
            guard let v = value else { return }
            wheres.append("a.\(column) = ?\(n + 1)")
            bind { [weak self] i in self?.bindText(self?.pending, i, v) }
        }
        eq("lane", query.lane); eq("content_mode", query.tier)
        eq("character_name", query.character); eq("source", query.source)
        eq("stock", query.stock); eq("genre", query.genre); eq("arc", query.arc)
        eq("kind", query.kind); eq("mode", query.mode)

        func cmp(_ column: String, _ op: String, _ value: Int?) {
            guard let v = value else { return }
            wheres.append("a.\(column) \(op) ?\(n + 1)")
            bind { [weak self] i in sqlite3_bind_int64(self?.pending, i, Int64(v)) }
        }
        cmp("duration_ms", ">=", query.minDurationMs)
        cmp("duration_ms", "<=", query.maxDurationMs)
        cmp("rating", ">=", query.minRating)

        func date(_ column: String, _ op: String, _ value: Date?) {
            guard let v = value else { return }
            wheres.append("a.\(column) \(op) ?\(n + 1)")
            bind { [weak self] i in sqlite3_bind_double(self?.pending, i, v.timeIntervalSince1970) }
        }
        date("created_at", ">=", query.since)
        date("created_at", "<=", query.until)

        let whereSQL = wheres.isEmpty ? "" : "WHERE " + wheres.joined(separator: " AND ")
        let order: String
        switch query.orderBy {
        case .newest: order = "a.created_at DESC"
        case .oldest: order = "a.created_at ASC"
        case .rating: order = "a.rating DESC, a.created_at DESC"
        }

        let sql = """
            SELECT a.id, a.kind, a.filename, a.absolute_path, a.sha256, a.file_size,
                   a.width, a.height, a.created_at, a.realm, a.source, a.sealed,
                   a.prompt, a.negative_prompt, a.prompt_raw, a.caption, a.caption_source,
                   a.seed, a.steps, a.guidance, a.model_family, a.preset, a.loras,
                   a.render_id, a.content_mode, a.character_name,
                   a.lane, a.arc, a.theme, a.stock, a.genre, a.family, a.style,
                   a.mode, a.duration_ms, a.fps, a.frames, a.resolution, a.aspect_ratio,
                   a.rating, a.favorite
            FROM assets a
            \(whereSQL)
            ORDER BY \(order)
            LIMIT ?\(n + 1) OFFSET ?\(n + 2)
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt); pending = nil }
        pending = stmt

        for (i, f) in binds.enumerated() { f(Int32(i + 1)) }
        sqlite3_bind_int(stmt, n + 1, Int32(query.limit))
        sqlite3_bind_int(stmt, n + 2, Int32(query.offset))

        var rows: [CatalogAsset] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(clamp(rowToAsset(stmt), to: query.ceiling))
        }
        return rows
    }

    /// THE MODE CLAMP. Above the ceiling the tier LABEL survives — it is
    /// metadata — while text and the file path do not. Matches render-journal.ts.
    private func clamp(_ a: CatalogAsset, to ceiling: String?) -> CatalogAsset {
        guard let ceiling, tierRank(a.contentMode) > tierRank(ceiling) else { return a }
        return CatalogAsset(
            id: a.id, kind: a.kind, filename: "", absolutePath: "",
            sha256: a.sha256, fileSize: a.fileSize, width: a.width, height: a.height,
            createdAt: a.createdAt, realm: a.realm, source: a.source, sealed: a.sealed,
            prompt: nil, negativePrompt: nil, promptRaw: nil, caption: nil, captionSource: nil,
            seed: a.seed, steps: a.steps, guidance: a.guidance, modelFamily: a.modelFamily,
            preset: a.preset, loras: a.loras, renderID: a.renderID,
            contentMode: a.contentMode, characterName: a.characterName,
            lane: a.lane, arc: a.arc, theme: a.theme, stock: a.stock,
            genre: a.genre, family: a.family, style: a.style,
            mode: a.mode, durationMs: a.durationMs, fps: a.fps, frames: a.frames,
            resolution: a.resolution, aspectRatio: a.aspectRatio,
            rating: a.rating, favorite: a.favorite)
    }
```

**Note for the implementer:** the `pending` statement handle above is a small
piece of bookkeeping so the bind closures can reach the prepared statement.
Declare it alongside `db`:

```swift
    private var pending: OpaquePointer?
```

- [ ] **Step 6: Add collections, edges, locations, facets, and the row/bind helpers**

Append inside the actor:

```swift
    // MARK: - Collections

    public func collections(visibleTo realm: CatalogRealm?) throws -> [CatalogCollection] {
        var out: [CatalogCollection] = []
        let sql = """
            SELECT id, slug, name, parent_id, realm, description FROM collections
            WHERE realm IS NULL OR ?1 IS NULL OR realm = ?1
            ORDER BY parent_id IS NOT NULL, name
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, realm?.rawValue)
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(CatalogCollection(
                id: text(stmt, 0) ?? "", slug: text(stmt, 1) ?? "", name: text(stmt, 2) ?? "",
                parentID: text(stmt, 3),
                realm: text(stmt, 4).flatMap { CatalogRealm(rawValue: $0) },
                description: text(stmt, 5)))
        }
        return out
    }

    public func createCollection(_ c: CatalogCollection, by actor: CatalogRealm?) throws {
        try requireOwnership(of: c.realm, by: actor, action: "create")
        if let parent = c.parentID {
            guard let parentRow = try collectionRow(parent) else {
                throw CatalogError.noSuchCollection(parent)
            }
            guard parentRow.parentID == nil else { throw CatalogError.depthCapExceeded }
        }
        try execBind("""
            INSERT INTO collections (id, slug, name, parent_id, realm, description, created_at)
            VALUES (?1,?2,?3,?4,?5,?6,?7)
            """) { [weak self] s in
            guard let self else { return }
            self.bindText(s, 1, c.id); self.bindText(s, 2, c.slug); self.bindText(s, 3, c.name)
            self.bindText(s, 4, c.parentID); self.bindText(s, 5, c.realm?.rawValue)
            self.bindText(s, 6, c.description)
            sqlite3_bind_double(s, 7, Date().timeIntervalSince1970)
        }
    }

    public func renameCollection(id: String, name: String, by actor: CatalogRealm?) throws {
        guard let row = try collectionRow(id) else { throw CatalogError.noSuchCollection(id) }
        try requireOwnership(of: row.realm, by: actor, action: "rename")
        try execBind("UPDATE collections SET name = ?2 WHERE id = ?1") { [weak self] s in
            guard let self else { return }
            self.bindText(s, 1, id); self.bindText(s, 2, name)
        }
    }

    public func retireCollection(id: String, by actor: CatalogRealm?) throws {
        guard let row = try collectionRow(id) else { throw CatalogError.noSuchCollection(id) }
        try requireOwnership(of: row.realm, by: actor, action: "retire")
        try execBind("DELETE FROM asset_collections WHERE collection_id = ?1") { [weak self] s in
            self?.bindText(s, 1, id)
        }
        try execBind("DELETE FROM collections WHERE id = ?1") { [weak self] s in
            self?.bindText(s, 1, id)
        }
    }

    /// Manual filing. Wins over derived filing and survives re-ingest.
    public func file(assetID: String, into collectionID: String, by actor: CatalogRealm?) throws {
        guard let row = try collectionRow(collectionID) else {
            throw CatalogError.noSuchCollection(collectionID)
        }
        if let actor, try assetRealm(assetID) != actor {
            throw CatalogError.notPermitted("\(actor.rawValue) may not file an asset outside its realm")
        }
        // Contributing to a SHARED collection is allowed; restructuring it is not.
        if row.realm != nil { try requireOwnership(of: row.realm, by: actor, action: "file into") }
        try execBind("""
            INSERT INTO asset_collections (asset_id, collection_id, manual) VALUES (?1, ?2, 1)
            ON CONFLICT(asset_id, collection_id) DO UPDATE SET manual = 1
            """) { [weak self] s in
            guard let self else { return }
            self.bindText(s, 1, assetID); self.bindText(s, 2, collectionID)
        }
    }

    /// nil actor = the service itself (backfill, desktop). A realm-scoped actor
    /// may only touch collections of its own realm.
    private func requireOwnership(of target: CatalogRealm?, by actor: CatalogRealm?,
                                  action: String) throws {
        guard let actor else { return }
        guard target == actor else {
            throw CatalogError.notPermitted("\(actor.rawValue) may not \(action) a \(target?.rawValue ?? "shared") collection")
        }
    }

    private func collectionRow(_ id: String) throws -> CatalogCollection? {
        var stmt: OpaquePointer?
        let sql = "SELECT id, slug, name, parent_id, realm, description FROM collections WHERE id = ?1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return CatalogCollection(
            id: text(stmt, 0) ?? "", slug: text(stmt, 1) ?? "", name: text(stmt, 2) ?? "",
            parentID: text(stmt, 3),
            realm: text(stmt, 4).flatMap { CatalogRealm(rawValue: $0) },
            description: text(stmt, 5))
    }

    private func collectionRealm(_ id: String) throws -> CatalogRealm? {
        try collectionRow(id)?.realm
    }

    private func assetRealm(_ id: String) throws -> CatalogRealm? {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT realm FROM assets WHERE id = ?1", -1, &stmt, nil) == SQLITE_OK
        else { throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db))) }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return text(stmt, 0).flatMap { CatalogRealm(rawValue: $0) }
    }

    // MARK: - Edges and locations

    public func addEdge(_ e: AssetEdge) throws {
        try execBind("""
            INSERT OR IGNORE INTO asset_edges (from_asset_id, to_asset_id, relation)
            VALUES (?1, ?2, ?3)
            """) { [weak self] s in
            guard let self else { return }
            self.bindText(s, 1, e.fromAssetID); self.bindText(s, 2, e.toAssetID)
            self.bindText(s, 3, e.relation.rawValue)
        }
    }

    /// Every edge touching this asset, in either direction — so a still finds
    /// its clips and a clip finds its still with one call.
    public func edges(for assetID: String) throws -> [AssetEdge] {
        var stmt: OpaquePointer?
        let sql = """
            SELECT from_asset_id, to_asset_id, relation FROM asset_edges
            WHERE from_asset_id = ?1 OR to_asset_id = ?1
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, assetID)
        var out: [AssetEdge] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let r = text(stmt, 2).flatMap({ AssetRelation(rawValue: $0) }) else { continue }
            out.append(AssetEdge(fromAssetID: text(stmt, 0) ?? "",
                                 toAssetID: text(stmt, 1) ?? "", relation: r))
        }
        return out
    }

    public func addLocation(assetID: String, _ loc: AssetLocation) throws {
        try execBind("""
            INSERT OR REPLACE INTO asset_locations (asset_id, host, path, mtime)
            VALUES (?1, ?2, ?3, ?4)
            """) { [weak self] s in
            guard let self else { return }
            self.bindText(s, 1, assetID); self.bindText(s, 2, loc.host); self.bindText(s, 3, loc.path)
            sqlite3_bind_double(s, 4, loc.mtime.timeIntervalSince1970)
        }
    }

    public func locations(of assetID: String) throws -> [AssetLocation] {
        var stmt: OpaquePointer?
        let sql = "SELECT host, path, mtime FROM asset_locations WHERE asset_id = ?1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, assetID)
        var out: [AssetLocation] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(AssetLocation(host: text(stmt, 0) ?? "", path: text(stmt, 1) ?? "",
                                     mtime: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))))
        }
        return out
    }

    /// Asset id for a known file path — used by backfill to resolve
    /// `source_image` into an `i2v_source` edge.
    public func assetID(forPath path: String) throws -> String? {
        var stmt: OpaquePointer?
        let sql = """
            SELECT id FROM assets WHERE absolute_path = ?1
            UNION SELECT asset_id FROM asset_locations WHERE path = ?1 LIMIT 1
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, path)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return text(stmt, 0)
    }

    public func assetID(forSHA256 sha: String) throws -> String? {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT id FROM assets WHERE sha256 = ?1 LIMIT 1", -1, &stmt, nil) == SQLITE_OK
        else { throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db))) }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, sha)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return text(stmt, 0)
    }

    // MARK: - Facets

    public func facets(scope: CatalogRealm?) throws -> CatalogFacets {
        var f = CatalogFacets()
        func count(_ column: String, into keyPath: WritableKeyPath<CatalogFacets, [String: Int]>) throws {
            let sql = """
                SELECT \(column), COUNT(*) FROM assets
                WHERE \(column) IS NOT NULL AND (?1 IS NULL OR realm = ?1)
                GROUP BY \(column)
                """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }
            bindText(stmt, 1, scope?.rawValue)
            while sqlite3_step(stmt) == SQLITE_ROW {
                guard let k = text(stmt, 0) else { continue }
                f[keyPath: keyPath][k] = Int(sqlite3_column_int(stmt, 1))
            }
        }
        try count("lane", into: \.lane)
        try count("content_mode", into: \.tier)
        try count("character_name", into: \.character)
        try count("source", into: \.source)
        try count("stock", into: \.stock)
        try count("genre", into: \.genre)
        try count("kind", into: \.kind)
        try count("mode", into: \.mode)

        // Collections need the join.
        let sql = """
            SELECT ac.collection_id, COUNT(*) FROM asset_collections ac
            JOIN assets a ON a.id = ac.asset_id
            WHERE ?1 IS NULL OR a.realm = ?1
            GROUP BY ac.collection_id
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, scope?.rawValue)
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let k = text(stmt, 0) else { continue }
            f.collection[k] = Int(sqlite3_column_int(stmt, 1))
        }
        return f
    }

    // MARK: - Row and bind helpers

    private func rowToAsset(_ s: OpaquePointer?) -> CatalogAsset {
        CatalogAsset(
            id: text(s, 0) ?? "", kind: text(s, 1) ?? "image",
            filename: text(s, 2) ?? "", absolutePath: text(s, 3) ?? "",
            sha256: text(s, 4), fileSize: sqlite3_column_int64(s, 5),
            width: optInt(s, 6), height: optInt(s, 7),
            createdAt: Date(timeIntervalSince1970: sqlite3_column_double(s, 8)),
            realm: CatalogRealm(rawValue: text(s, 9) ?? "shared") ?? .shared,
            source: text(s, 10), sealed: sqlite3_column_int(s, 11) != 0,
            prompt: text(s, 12), negativePrompt: text(s, 13), promptRaw: text(s, 14),
            caption: text(s, 15), captionSource: text(s, 16),
            seed: optInt(s, 17), steps: optInt(s, 18), guidance: optDouble(s, 19),
            modelFamily: text(s, 20), preset: text(s, 21), loras: text(s, 22),
            renderID: text(s, 23), contentMode: text(s, 24), characterName: text(s, 25),
            lane: text(s, 26), arc: text(s, 27), theme: text(s, 28), stock: text(s, 29),
            genre: text(s, 30), family: text(s, 31), style: text(s, 32),
            mode: text(s, 33), durationMs: optInt(s, 34), fps: optDouble(s, 35),
            frames: optInt(s, 36), resolution: text(s, 37), aspectRatio: text(s, 38),
            rating: Int(sqlite3_column_int(s, 39)), favorite: sqlite3_column_int(s, 40) != 0)
    }

    private func execBind(_ sql: String, _ binder: (OpaquePointer?) -> Void) throws {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        binder(stmt)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw CatalogError.stepFailed(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func bindText(_ s: OpaquePointer?, _ i: Int32, _ v: String?) {
        if let v { sqlite3_bind_text(s, i, (v as NSString).utf8String, -1, CatalogSchema.transient) }
        else { sqlite3_bind_null(s, i) }
    }
    private func bindInt(_ s: OpaquePointer?, _ i: Int32, _ v: Int?) {
        if let v { sqlite3_bind_int64(s, i, Int64(v)) } else { sqlite3_bind_null(s, i) }
    }
    private func bindDouble(_ s: OpaquePointer?, _ i: Int32, _ v: Double?) {
        if let v { sqlite3_bind_double(s, i, v) } else { sqlite3_bind_null(s, i) }
    }
    private func text(_ s: OpaquePointer?, _ i: Int32) -> String? {
        guard let c = sqlite3_column_text(s, i) else { return nil }
        return String(cString: c)
    }
    private func optInt(_ s: OpaquePointer?, _ i: Int32) -> Int? {
        sqlite3_column_type(s, i) == SQLITE_NULL ? nil : Int(sqlite3_column_int64(s, i))
    }
    private func optDouble(_ s: OpaquePointer?, _ i: Int32) -> Double? {
        sqlite3_column_type(s, i) == SQLITE_NULL ? nil : sqlite3_column_double(s, i)
    }
```

- [ ] **Step 7: Run the tests**

Run: `swift test --filter CatalogStoreTests`
Expected: PASS, 17 tests. If the realm-isolation test fails, stop and fix it before anything else — every consumer depends on it.

- [ ] **Step 8: Commit**

```bash
git add Sources/ComfyBoxCatalog/CatalogStore.swift Tests/ComfyBoxCatalogTests/CatalogStoreTests.swift
git commit -m "feat(catalog): CatalogStore with service-side realm lock and mode clamp"
```

---

### Task 6: Backfill — sweep three trees, dedup, file, build edges

**Files:**
- Create: `Sources/ComfyBoxCatalog/CatalogBackfill.swift`
- Test: `Tests/ComfyBoxCatalogTests/CatalogBackfillTests.swift`

**Interfaces:**
- Consumes: `CatalogStore`, `MetadataReader`, `CollectionRules`.
- Produces: `struct BackfillTree`, `struct BackfillReport`, `CatalogBackfill.run(store:trees:host:) async throws -> BackfillReport`, `CatalogBackfill.realm(forTreeID:)`.

**Background:** three trees. The gallery home `~/Pictures/ComfyBox` (Mac) is where everything is rendered; `~/.kira/studio` and `~/.bree/studio` (server) are downstream copies with the JSON sidecars. The same asset appears in more than one, so it must dedup to one row with several `asset_locations`. Video has no embedded metadata, so its facts come only from the sidecar plus a container probe.

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxCatalogTests/CatalogBackfillTests.swift`:

```swift
import XCTest
@testable import ComfyBoxCatalog

final class CatalogBackfillTests: XCTestCase {
    private var root: String!
    private var dbPath: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        root = NSTemporaryDirectory() + "bf-\(UUID().uuidString)"
        dbPath = root + "/catalog.sqlite3"
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        store = try await CatalogStore.open(path: dbPath)
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: root)
        try await super.tearDown()
    }

    /// Write `bytes` to <tree>/<relative> and return the absolute path.
    @discardableResult
    private func write(_ relative: String, bytes: String, tree: String) throws -> String {
        let path = root + "/" + tree + "/" + relative
        try FileManager.default.createDirectory(
            atPath: (path as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true)
        try Data(bytes.utf8).write(to: URL(fileURLWithPath: path))
        return path
    }

    private func writeSidecar(_ relative: String, json: String, tree: String) throws {
        _ = try write(relative, bytes: json, tree: tree)
    }

    private func trees() -> [BackfillTree] {
        [
            BackfillTree(id: "home", realm: nil, host: "mac",
                         mediaRoot: root + "/home", metadataRoot: nil),
            BackfillTree(id: "kira", realm: .kira, host: "kira",
                         mediaRoot: root + "/kira/gallery", metadataRoot: root + "/kira/metadata"),
        ]
    }

    func testAssetsAreIndexedFromEveryTree() async throws {
        try write("a.png", bytes: "AAA", tree: "home")
        try write("Kira/generated/b.png", bytes: "BBB", tree: "kira/gallery")
        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.assetsIndexed, 2)
        XCTAssertEqual(try await store.search(CatalogQuery(scope: nil)).count, 2)
    }

    func testIdenticalBytesInTwoTreesBecomeOneAssetWithTwoLocations() async throws {
        try write("dup.png", bytes: "SAME BYTES", tree: "home")
        try write("Kira/generated/renamed.png", bytes: "SAME BYTES", tree: "kira/gallery")
        let report = try await CatalogBackfill.run(store: store, trees: trees())

        let rows = try await store.search(CatalogQuery(scope: nil))
        XCTAssertEqual(rows.count, 1, "same bytes = one asset")
        XCTAssertEqual(report.duplicatesMerged, 1)
        XCTAssertEqual(try await store.locations(of: rows[0].id).count, 2)
    }

    func testRealmComesFromTheTreeThatHoldsTheCopy() async throws {
        try write("only-home.png", bytes: "H", tree: "home")
        try write("shared-bytes.png", bytes: "S", tree: "home")
        try write("Kira/generated/shared-bytes.png", bytes: "S", tree: "kira/gallery")
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let kira = try await store.search(CatalogQuery(scope: .kira))
        let shared = try await store.search(CatalogQuery(scope: .shared))
        XCTAssertEqual(kira.count, 1, "a Mac asset with a Kira twin is hers")
        XCTAssertEqual(shared.count, 1, "a Mac asset with no twin defaults to shared")
    }

    func testSidecarSuppliesFacetsAndFilesIntoACollection() async throws {
        try write("Kira/generated/tile1.png", bytes: "T", tree: "kira/gallery")
        try writeSidecar("Kira/generated/tile1.json", json: """
            {"character":"kira","content_mode":"neutral","lane":"tile","preset":"krea-kira","sealed":false}
            """, tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let hers = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        XCTAssertEqual(hers.count, 1)
        XCTAssertEqual(hers.first?.contentMode, "neutral")
        XCTAssertEqual(hers.first?.preset, "krea-kira")
    }

    func testSealedSidecarProducesARowWithNoText() async throws {
        try write("Kira/generated/s.png", bytes: "S1", tree: "kira/gallery")
        try writeSidecar("Kira/generated/s.json", json: """
            {"character":"bree","sealed":true,"prompt":"must not be stored","content_mode":"apple"}
            """, tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let rows = try await store.search(CatalogQuery(scope: nil))
        XCTAssertNil(rows.first?.prompt)
        XCTAssertTrue(try await store.search(CatalogQuery(scope: nil, text: "must not be stored")).isEmpty)
    }

    func testSourceImageBecomesAnI2VEdge() async throws {
        let still = try write("Kira/generated/still.png", bytes: "STILL", tree: "kira/gallery")
        try write("Kira/video/clip.mp4", bytes: "CLIP", tree: "kira/gallery")
        try writeSidecar("Kira/video/clip.json", json: """
            {"character":"kira","mode":"i2v","content_mode":"banana","source_image":"\(still)"}
            """, tree: "kira/metadata")
        let report = try await CatalogBackfill.run(store: store, trees: trees())

        XCTAssertEqual(report.edgesCreated, 1)
        let clipID = try XCTUnwrap(try await store.assetID(forPath: root + "/kira/gallery/Kira/video/clip.mp4"))
        let edges = try await store.edges(for: clipID)
        XCTAssertEqual(edges.first?.relation, .i2vSource)
    }

    func testAnUnresolvableSourceImageIsSkippedNotInvented() async throws {
        try write("Kira/video/orphan.mp4", bytes: "O", tree: "kira/gallery")
        try writeSidecar("Kira/video/orphan.json", json: """
            {"mode":"i2v","source_image":"/nowhere/gone.png"}
            """, tree: "kira/metadata")
        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.edgesCreated, 0)
        XCTAssertEqual(report.assetsIndexed, 1, "the clip is still indexed")
    }

    func testBackfillIsIdempotent() async throws {
        try write("a.png", bytes: "A", tree: "home")
        try write("Kira/generated/b.png", bytes: "B", tree: "kira/gallery")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let firstCount = try await store.search(CatalogQuery(scope: nil, limit: 500)).count
        let firstLocations = try await store.locations(of:
            try XCTUnwrap(try await store.assetID(forPath: root + "/home/a.png"))).count

        _ = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(try await store.search(CatalogQuery(scope: nil, limit: 500)).count, firstCount)
        XCTAssertEqual(try await store.locations(of:
            try XCTUnwrap(try await store.assetID(forPath: root + "/home/a.png"))).count, firstLocations)
    }

    func testRebuildFromScratchProducesTheSameRows() async throws {
        try write("Kira/generated/x.png", bytes: "X", tree: "kira/gallery")
        try writeSidecar("Kira/generated/x.json", json: #"{"character":"kira","lane":"shoot","content_mode":"neutral"}"#,
                         tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let before = try await store.search(CatalogQuery(scope: nil, limit: 500)).map(\.filename).sorted()

        // Delete the catalog entirely and rebuild from the media alone.
        store = nil
        try FileManager.default.removeItem(atPath: dbPath)
        store = try await CatalogStore.open(path: dbPath)
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let after = try await store.search(CatalogQuery(scope: nil, limit: 500)).map(\.filename).sorted()
        XCTAssertEqual(after, before, "the catalog must be reconstructible from the files")
    }

    /// Video cannot fall back on the file, so its rebuild is asserted separately.
    func testVideoRebuildsFromItsSidecar() async throws {
        try write("Kira/video/v.mp4", bytes: "V", tree: "kira/gallery")
        try writeSidecar("Kira/video/v.json", json: """
            {"character":"kira","mode":"t2v","lane":"video","content_mode":"banana",
             "resolution":"480p","aspect_ratio":"9:16","duration":null}
            """, tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        store = nil
        try FileManager.default.removeItem(atPath: dbPath)
        store = try await CatalogStore.open(path: dbPath)
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let rows = try await store.search(CatalogQuery(scope: .kira, kind: "video"))
        XCTAssertEqual(rows.first?.mode, "t2v")
        XCTAssertEqual(rows.first?.resolution, "480p")
        XCTAssertEqual(rows.first?.aspectRatio, "9:16")
        let dreams = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-dreams-memories"))
        XCTAssertEqual(dreams.count, 1, "t2v files into Dreams & Memories")
    }

    func testTheVaultIsNeverTouched() async throws {
        // A tree list may never include a vault path. This is a guard on the
        // caller contract, asserted so a future edit cannot quietly add one.
        for t in trees() {
            XCTAssertFalse(t.mediaRoot.contains("Vaults"), "no tree may point into a vault")
            XCTAssertFalse(t.metadataRoot?.contains("Vaults") ?? false)
        }
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter CatalogBackfillTests`
Expected: FAIL — `CatalogBackfill` is undefined.

- [ ] **Step 3: Write the backfill**

Create `Sources/ComfyBoxCatalog/CatalogBackfill.swift`:

```swift
// CatalogBackfill.swift — reconstruct the catalog from what is on disk.
//
// Order matters and is not arbitrary:
//   pass 1  index every media file in every tree, deduping by sha256 so an
//           asset copied to a server tree becomes ONE row with several
//           locations rather than two rows.
//   pass 2  resolve `source_image` from video sidecars into i2v_source edges.
//           This is a second pass because the still may be indexed after the
//           clip that references it.
//
// Realm is recovered from which tree holds the copy: an asset with a twin under
// a Kira tree is hers; anything else is shared. It cannot come from the Mac
// path, because every realm renders through the same output directory.
//
// NOTHING here may point at ~/Documents/Vaults/BarkadaAI. The vault is out of
// scope and is never read.

import Foundation
import CryptoKit

public struct BackfillTree: Sendable {
    public let id: String
    /// nil = realm is decided by whether a twin exists elsewhere.
    public let realm: CatalogRealm?
    public let host: String
    public let mediaRoot: String
    public let metadataRoot: String?

    public init(id: String, realm: CatalogRealm?, host: String,
                mediaRoot: String, metadataRoot: String?) {
        self.id = id; self.realm = realm; self.host = host
        self.mediaRoot = mediaRoot; self.metadataRoot = metadataRoot
    }
}

public struct BackfillReport: Sendable, Equatable {
    public var filesScanned = 0
    public var assetsIndexed = 0
    public var duplicatesMerged = 0
    public var edgesCreated = 0
    public var sidecarsRead = 0
    public var skipped = 0
}

public enum CatalogBackfill {

    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "tiff", "heic"]
    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]

    public static func run(store: CatalogStore, trees: [BackfillTree]) async throws -> BackfillReport {
        var report = BackfillReport()

        // Pass 1 — index, dedup, file.
        // sha256 → asset id, so the second sighting of the same bytes adds a
        // location instead of a row.
        var bySHA: [String: String] = [:]
        // Deferred i2v links: (clip path, source image path).
        var pendingEdges: [(String, String)] = []

        for tree in trees {
            for path in mediaFiles(under: tree.mediaRoot) {
                report.filesScanned += 1
                let ext = (path as NSString).pathExtension.lowercased()
                let kind = videoExtensions.contains(ext) ? "video" : "image"

                guard let data = FileManager.default.contents(atPath: path) else {
                    report.skipped += 1
                    continue
                }
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()

                // Metadata: embedded for images (video containers carry none),
                // sidecar for both — the ONLY source for video.
                var meta = kind == "image" ? (MetadataReader.readEmbedded(path: path) ?? FileMetadata())
                                           : FileMetadata()
                if let mroot = tree.metadataRoot,
                   let sidecar = MetadataReader.sidecarPath(forMedia: path,
                                                            galleryRoot: tree.mediaRoot,
                                                            metadataRoot: mroot),
                   let sdata = FileManager.default.contents(atPath: sidecar),
                   let smeta = MetadataReader.readSidecar(jsonData: sdata) {
                    report.sidecarsRead += 1
                    meta = merge(embedded: meta, sidecar: smeta)
                }
                if kind == "video", let info = MetadataReader.probeContainer(path: path) {
                    meta.durationMs = info.durationMs
                }

                let attrs = try? FileManager.default.attributesOfItem(atPath: path)
                let mtime = (attrs?[.modificationDate] as? Date) ?? Date()
                let size = (attrs?[.size] as? NSNumber)?.int64Value ?? Int64(data.count)

                if let existingID = bySHA[sha] ?? (try await store.assetID(forSHA256: sha)) {
                    // Same bytes seen before — a downstream copy.
                    report.duplicatesMerged += 1
                    try await store.addLocation(assetID: existingID,
                        AssetLocation(host: tree.host, path: path, mtime: mtime))
                    // A twin in a realm-bearing tree settles the realm.
                    if let realm = tree.realm {
                        try await store.setRealm(realm, forAssetID: existingID)
                    }
                    if kind == "video", let src = meta.sourceImagePath {
                        pendingEdges.append((path, src))
                    }
                    continue
                }

                let asset = CatalogAsset(
                    kind: kind,
                    filename: (path as NSString).lastPathComponent,
                    absolutePath: path,
                    sha256: sha, fileSize: size,
                    width: meta.width, height: meta.height,
                    createdAt: (attrs?[.creationDate] as? Date) ?? mtime,
                    realm: tree.realm ?? .shared,
                    source: meta.software.map { $0.lowercased() },
                    sealed: meta.sealed,
                    prompt: meta.prompt, negativePrompt: meta.negativePrompt,
                    promptRaw: meta.promptRaw,
                    seed: meta.seed, steps: meta.steps, guidance: meta.guidance,
                    modelFamily: meta.modelFamily, preset: meta.preset, loras: meta.loras,
                    contentMode: meta.contentMode, characterName: meta.characterName,
                    lane: meta.lane,
                    mode: meta.mode, durationMs: meta.durationMs,
                    resolution: meta.resolution, aspectRatio: meta.aspectRatio)

                try await store.upsert(asset, explicitCollectionIDs: [])
                try await store.addLocation(assetID: asset.id,
                    AssetLocation(host: tree.host, path: path, mtime: mtime))
                bySHA[sha] = asset.id
                report.assetsIndexed += 1

                if kind == "video", let src = meta.sourceImagePath {
                    pendingEdges.append((path, src))
                }
            }
        }

        // Pass 2 — i2v edges, now that every still is indexed.
        for (clipPath, sourcePath) in pendingEdges {
            guard let clipID = try await store.assetID(forPath: clipPath),
                  let stillID = try await store.assetID(forPath: sourcePath) else { continue }
            try await store.addEdge(AssetEdge(fromAssetID: clipID, toAssetID: stillID,
                                              relation: .i2vSource))
            report.edgesCreated += 1
        }

        return report
    }

    /// The sidecar is authoritative where it speaks; embedded fills the gaps.
    static func merge(embedded: FileMetadata, sidecar: FileMetadata) -> FileMetadata {
        var m = embedded
        m.prompt = sidecar.prompt ?? m.prompt
        m.promptRaw = sidecar.promptRaw ?? m.promptRaw
        m.negativePrompt = sidecar.negativePrompt ?? m.negativePrompt
        m.seed = sidecar.seed ?? m.seed
        m.steps = sidecar.steps ?? m.steps
        m.guidance = sidecar.guidance ?? m.guidance
        m.width = sidecar.width ?? m.width
        m.height = sidecar.height ?? m.height
        m.modelFamily = sidecar.modelFamily ?? m.modelFamily
        m.preset = sidecar.preset ?? m.preset
        m.loras = sidecar.loras ?? m.loras
        m.characterName = sidecar.characterName ?? m.characterName
        m.contentMode = sidecar.contentMode ?? m.contentMode
        m.lane = sidecar.lane ?? m.lane
        m.mode = sidecar.mode ?? m.mode
        m.resolution = sidecar.resolution ?? m.resolution
        m.aspectRatio = sidecar.aspectRatio ?? m.aspectRatio
        m.sourceImagePath = sidecar.sourceImagePath ?? m.sourceImagePath
        m.sealed = sidecar.sealed || m.sealed
        return m
    }

    static func mediaFiles(under root: String) -> [String] {
        let fm = FileManager.default
        guard let en = fm.enumerator(at: URL(fileURLWithPath: root),
                                     includingPropertiesForKeys: nil,
                                     options: [.skipsHiddenFiles]) else { return [] }
        var out: [String] = []
        for case let url as URL in en {
            let ext = url.pathExtension.lowercased()
            guard imageExtensions.contains(ext) || videoExtensions.contains(ext) else { continue }
            out.append(url.path)
        }
        return out.sorted()
    }
}
```

- [ ] **Step 4: Add the one store method backfill needs**

Append inside `CatalogStore`:

```swift
    /// Settle an asset's realm once a twin is found in a realm-bearing tree.
    /// Backfill-only: realm is otherwise stamped by the caller at render time.
    public func setRealm(_ realm: CatalogRealm, forAssetID id: String) throws {
        try execBind("UPDATE assets SET realm = ?2 WHERE id = ?1") { [weak self] s in
            guard let self else { return }
            self.bindText(s, 1, id); self.bindText(s, 2, realm.rawValue)
        }
        // Filing depends on realm, so re-derive it. Fetch the row by id —
        // a limit-1 search would almost never contain it.
        if let row = try asset(id: id) {
            try applyDerivedFiling(row, explicitCollectionIDs: [])
        }
    }

    /// Fetch one row by id, unscoped and unclamped. Internal helper for
    /// backfill; consumers go through `search`, which applies the realm lock.
    public func asset(id: String) throws -> CatalogAsset? {
        var q = CatalogQuery(scope: nil, limit: 1)
        q.limit = 1
        let sql = """
            SELECT a.id, a.kind, a.filename, a.absolute_path, a.sha256, a.file_size,
                   a.width, a.height, a.created_at, a.realm, a.source, a.sealed,
                   a.prompt, a.negative_prompt, a.prompt_raw, a.caption, a.caption_source,
                   a.seed, a.steps, a.guidance, a.model_family, a.preset, a.loras,
                   a.render_id, a.content_mode, a.character_name,
                   a.lane, a.arc, a.theme, a.stock, a.genre, a.family, a.style,
                   a.mode, a.duration_ms, a.fps, a.frames, a.resolution, a.aspect_ratio,
                   a.rating, a.favorite
            FROM assets a WHERE a.id = ?1
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return rowToAsset(stmt)
    }
```

- [ ] **Step 5: Run the tests**

Run: `swift test --filter CatalogBackfillTests`
Expected: PASS, 11 tests.

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxCatalog/CatalogBackfill.swift Sources/ComfyBoxCatalog/CatalogStore.swift Tests/ComfyBoxCatalogTests/CatalogBackfillTests.swift
git commit -m "feat(catalog): backfill sweep with sha256 dedup, sidecar facets and i2v edges"
```

---

### Task 7: HTTP server and the gallery routes

**Files:**
- Create: `Sources/ComfyBoxCatalog/HTTPKit.swift`
- Modify: `Sources/ComfyBoxCatalog/GalleryServer.swift` (replace the Task 1 stub)
- Test: `Tests/ComfyBoxCatalogTests/GalleryServerTests.swift`

**Interfaces:**
- Consumes: `CatalogStore`, `CatalogQuery`, `CatalogFacets`.
- Produces: `HTTPKit.Request`, `HTTPKit.Response`, `HTTPKit.Server`; `GalleryServer.handle(request:store:) async -> HTTPKit.Response`, `GalleryServer.runCLIEntryPoint(args:)`.

**Route contract.** `scope` is derived from the `X-Catalog-Actor` header, never from a query parameter — that is the realm lock. Header value `kira` scopes to her realm; anything else (or absent) is unscoped.

| route | returns |
|---|---|
| `GET /v1/catalog/search` | `{items: [...], count}` |
| `GET /v1/catalog/facets` | `{lane: {...}, tier: {...}, collection: {...}, …}` |
| `GET /v1/catalog/collections` | `{items: [{id, slug, name, parent_id, realm, count}]}` |
| `GET /v1/catalog/asset/{id}` | one row plus `locations` and `edges` |
| `GET /healthz` | `{ok: true}` |

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxCatalogTests/GalleryServerTests.swift`:

```swift
import XCTest
@testable import ComfyBoxCatalog

final class GalleryServerTests: XCTestCase {
    private var path: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        path = NSTemporaryDirectory() + "srv-\(UUID().uuidString).sqlite3"
        store = try await CatalogStore.open(path: path)
        try await store.upsert(CatalogAsset(id: "k1", filename: "k1.png", absolutePath: "/tmp/k1.png",
                                            realm: .kira, prompt: "a tulip", contentMode: "neutral",
                                            lane: "still"), explicitCollectionIDs: [])
        try await store.upsert(CatalogAsset(id: "k2", filename: "k2.png", absolutePath: "/tmp/k2.png",
                                            realm: .kira, prompt: "a nightclub", contentMode: "avocado",
                                            lane: "kira"), explicitCollectionIDs: [])
        try await store.upsert(CatalogAsset(id: "s1", filename: "s1.png", absolutePath: "/tmp/s1.png",
                                            realm: .shared, prompt: "a tulip", contentMode: "neutral"),
                               explicitCollectionIDs: [])
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: path)
        try await super.tearDown()
    }

    private func get(_ target: String, actor: String? = nil, ceiling: String? = nil) async throws -> [String: Any] {
        var headers: [String: String] = [:]
        if let actor { headers["x-catalog-actor"] = actor }
        if let ceiling { headers["x-catalog-ceiling"] = ceiling }
        let req = HTTPKit.Request(method: "GET", target: target, headers: headers, body: Data())
        let res = await GalleryServer.handle(request: req, store: store)
        XCTAssertEqual(res.status, 200, "unexpected status for \(target)")
        return try XCTUnwrap(JSONSerialization.jsonObject(with: res.body) as? [String: Any])
    }

    func testHealthz() async throws {
        let body = try await get("/healthz")
        XCTAssertEqual(body["ok"] as? Bool, true)
    }

    func testSearchWithoutActorSeesEveryRealm() async throws {
        let body = try await get("/v1/catalog/search?limit=100")
        XCTAssertEqual(body["count"] as? Int, 3)
    }

    func testKiraActorHeaderScopesToHerRealm() async throws {
        let body = try await get("/v1/catalog/search?limit=100", actor: "kira")
        let items = try XCTUnwrap(body["items"] as? [[String: Any]])
        XCTAssertEqual(Set(items.compactMap { $0["id"] as? String }), ["k1", "k2"])
    }

    /// The lock cannot be widened from the wire, whatever the caller sends.
    func testRealmQueryParameterCannotOverrideTheActorHeader() async throws {
        for attempt in ["realm=shared", "realm=", "scope=shared", "realm=shared&realm=shared"] {
            let body = try await get("/v1/catalog/search?limit=100&\(attempt)", actor: "kira")
            let items = try XCTUnwrap(body["items"] as? [[String: Any]])
            XCTAssertTrue(items.allSatisfy { ($0["realm"] as? String) == "kira" },
                          "leaked with query \(attempt)")
        }
    }

    func testCeilingHeaderClampsTextAndPath() async throws {
        let body = try await get("/v1/catalog/search?limit=100", actor: "kira", ceiling: "apple")
        let items = try XCTUnwrap(body["items"] as? [[String: Any]])
        let avocado = try XCTUnwrap(items.first { ($0["id"] as? String) == "k2" })
        XCTAssertNil(avocado["prompt"])
        XCTAssertNil(avocado["path"])
        XCTAssertEqual(avocado["tier"] as? String, "avocado", "the label is metadata and survives")
    }

    func testFullTextSearch() async throws {
        let body = try await get("/v1/catalog/search?q=tulip&limit=100", actor: "kira")
        XCTAssertEqual(body["count"] as? Int, 1)
    }

    func testCollectionsAreRealmFiltered() async throws {
        let all = try await get("/v1/catalog/collections")
        let hers = try await get("/v1/catalog/collections", actor: "kira")
        let allSlugs = Set((all["items"] as! [[String: Any]]).compactMap { $0["slug"] as? String })
        let herSlugs = Set((hers["items"] as! [[String: Any]]).compactMap { $0["slug"] as? String })
        XCTAssertTrue(allSlugs.contains("decoupage"))
        XCTAssertTrue(herSlugs.contains("kira-still-life"))
        XCTAssertTrue(herSlugs.contains("decoupage"), "shared vocabulary is visible to her")
    }

    func testAssetDetailIncludesLocationsAndEdges() async throws {
        try await store.addLocation(assetID: "k1",
            AssetLocation(host: "mac", path: "/tmp/k1.png", mtime: Date()))
        try await store.addEdge(AssetEdge(fromAssetID: "k2", toAssetID: "k1", relation: .i2vSource))
        let body = try await get("/v1/catalog/asset/k1", actor: "kira")
        XCTAssertEqual((body["locations"] as? [[String: Any]])?.count, 1)
        XCTAssertEqual((body["edges"] as? [[String: Any]])?.count, 1)
    }

    func testKiraCannotFetchASharedAssetById() async throws {
        let req = HTTPKit.Request(method: "GET", target: "/v1/catalog/asset/s1",
                                  headers: ["x-catalog-actor": "kira"], body: Data())
        let res = await GalleryServer.handle(request: req, store: store)
        XCTAssertEqual(res.status, 404, "a shared row must be invisible to her, not merely filtered")
    }

    func testUnknownRouteIs404() async throws {
        let req = HTTPKit.Request(method: "GET", target: "/nope", headers: [:], body: Data())
        let res = await GalleryServer.handle(request: req, store: store)
        XCTAssertEqual(res.status, 404)
    }

    func testQueryParsingHandlesPercentEncodingAndMissingValues() {
        let q = HTTPKit.queryParameters(of: "/v1/catalog/search?q=film%20noir&lane=&limit=10")
        XCTAssertEqual(q["q"], "film noir")
        XCTAssertEqual(q["lane"], "")
        XCTAssertEqual(q["limit"], "10")
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter GalleryServerTests`
Expected: FAIL — `HTTPKit` is undefined.

- [ ] **Step 3: Write HTTPKit**

Create `Sources/ComfyBoxCatalog/HTTPKit.swift`:

```swift
// HTTPKit.swift — the smallest HTTP/1.1 server that serves this catalog.
//
// Deliberately hand-rolled rather than reusing the engine's WarmServer: this
// library must not depend on ZImage/MLX, so the gallery process stays small,
// starts instantly, and can be rebuilt without touching the engine binary.

import Foundation
import Network

public enum HTTPKit {

    public struct Request: Sendable {
        public let method: String
        public let target: String
        /// Header names are lowercased on parse.
        public let headers: [String: String]
        public let body: Data

        public init(method: String, target: String, headers: [String: String], body: Data) {
            self.method = method; self.target = target
            self.headers = headers.reduce(into: [:]) { $0[$1.key.lowercased()] = $1.value }
            self.body = body
        }

        public var path: String { target.split(separator: "?", maxSplits: 1).first.map(String.init) ?? target }
        public var query: [String: String] { HTTPKit.queryParameters(of: target) }
    }

    public struct Response: Sendable {
        public let status: Int
        public let contentType: String
        public let body: Data

        public init(status: Int, contentType: String = "application/json", body: Data) {
            self.status = status; self.contentType = contentType; self.body = body
        }

        public static func json(_ object: Any, status: Int = 200) -> Response {
            let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
            return Response(status: status, body: data)
        }

        public static func error(_ status: Int, _ message: String) -> Response {
            json(["error": message], status: status)
        }

        var wireData: Data {
            let reason = status == 200 ? "OK" : (status == 404 ? "Not Found" : "Error")
            var head = "HTTP/1.1 \(status) \(reason)\r\n"
            head += "Content-Type: \(contentType)\r\n"
            head += "Content-Length: \(body.count)\r\n"
            head += "Connection: close\r\n\r\n"
            return Data(head.utf8) + body
        }
    }

    /// Parse `?a=1&b=two%20words`. A key with no `=` yields "".
    public static func queryParameters(of target: String) -> [String: String] {
        guard let qIndex = target.firstIndex(of: "?") else { return [:] }
        let raw = String(target[target.index(after: qIndex)...])
        var out: [String: String] = [:]
        for pair in raw.split(separator: "&") {
            let parts = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            let key = String(parts[0]).removingPercentEncoding ?? String(parts[0])
            let value = parts.count > 1
                ? (String(parts[1]).replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? "")
                : ""
            out[key] = value
        }
        return out
    }

    static func parse(_ data: Data) -> Request? {
        guard let text = String(data: data, encoding: .utf8),
              let headerEnd = text.range(of: "\r\n\r\n") else { return nil }
        let headLines = text[..<headerEnd.lowerBound].components(separatedBy: "\r\n")
        guard let requestLine = headLines.first else { return nil }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { return nil }

        var headers: [String: String] = [:]
        for line in headLines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        let body = Data(text[headerEnd.upperBound...].utf8)
        return Request(method: String(parts[0]), target: String(parts[1]),
                       headers: headers, body: body)
    }

    /// Bound to loopback only. The gallery holds raw prompt text under the
    /// provenance contract; it is not a LAN service.
    public final class Server: @unchecked Sendable {
        private let port: UInt16
        private let handler: @Sendable (Request) async -> Response
        private var listener: NWListener?

        public init(port: UInt16, handler: @escaping @Sendable (Request) async -> Response) {
            self.port = port
            self.handler = handler
        }

        public func start() throws {
            let params = NWParameters.tcp
            params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1",
                                                              port: NWEndpoint.Port(rawValue: port)!)
            let l = try NWListener(using: params)
            l.newConnectionHandler = { [weak self] conn in self?.accept(conn) }
            l.start(queue: .global(qos: .userInitiated))
            listener = l
        }

        private func accept(_ conn: NWConnection) {
            conn.start(queue: .global(qos: .userInitiated))
            conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, _, _ in
                guard let self, let data, let req = HTTPKit.parse(data) else {
                    conn.cancel(); return
                }
                Task {
                    let res = await self.handler(req)
                    conn.send(content: res.wireData,
                              completion: .contentProcessed { _ in conn.cancel() })
                }
            }
        }
    }
}
```

- [ ] **Step 4: Write the routes**

Replace `Sources/ComfyBoxCatalog/GalleryServer.swift` entirely:

```swift
// GalleryServer.swift — the gallery service's routes.
//
// THE REALM LOCK LIVES HERE, at the boundary: `scope` comes from the
// X-Catalog-Actor header the daemon sets per tool, never from a query
// parameter. A client cannot widen its own scope by asking.

import Foundation

public enum GalleryServer {

    public static let defaultPort: UInt16 = 7871

    public static func handle(request: HTTPKit.Request, store: CatalogStore) async -> HTTPKit.Response {
        // Scope and ceiling are HEADERS, set by the trusted daemon per caller.
        let scope: CatalogRealm? = request.headers["x-catalog-actor"] == "kira" ? .kira : nil
        let ceiling = request.headers["x-catalog-ceiling"]

        do {
            switch (request.method, request.path) {
            case ("GET", "/healthz"):
                return .json(["ok": true])

            case ("GET", "/v1/catalog/search"):
                let q = try query(from: request, scope: scope, ceiling: ceiling)
                let rows = try await store.search(q)
                return .json(["count": rows.count, "items": rows.map(dict(for:))])

            case ("GET", "/v1/catalog/facets"):
                let f = try await store.facets(scope: scope)
                return .json([
                    "lane": f.lane, "tier": f.tier, "character": f.character,
                    "source": f.source, "stock": f.stock, "genre": f.genre,
                    "kind": f.kind, "mode": f.mode, "collection": f.collection,
                ])

            case ("GET", "/v1/catalog/collections"):
                let cols = try await store.collections(visibleTo: scope)
                let counts = try await store.facets(scope: scope).collection
                return .json(["items": cols.map { c in
                    [
                        "id": c.id, "slug": c.slug, "name": c.name,
                        "parent_id": c.parentID as Any,
                        "realm": c.realm?.rawValue as Any,
                        "description": c.description as Any,
                        "count": counts[c.id] ?? 0,
                    ] as [String: Any]
                }])

            case ("GET", let p) where p.hasPrefix("/v1/catalog/asset/"):
                let id = String(p.dropFirst("/v1/catalog/asset/".count))
                var q = CatalogQuery(scope: scope, ceiling: ceiling, limit: 1000)
                q.limit = 1000
                let rows = try await store.search(q)
                guard let row = rows.first(where: { $0.id == id }) else {
                    // 404 rather than 403: a row outside the caller's realm does
                    // not exist as far as that caller is concerned.
                    return .error(404, "no such asset")
                }
                var body = dict(for: row)
                body["locations"] = try await store.locations(of: id).map {
                    ["host": $0.host, "path": $0.path, "mtime": $0.mtime.timeIntervalSince1970]
                }
                body["edges"] = try await store.edges(for: id).map {
                    ["from": $0.fromAssetID, "to": $0.toAssetID, "relation": $0.relation.rawValue]
                }
                return .json(body)

            default:
                return .error(404, "no such route")
            }
        } catch {
            return .error(500, error.localizedDescription)
        }
    }

    private static func query(from request: HTTPKit.Request,
                              scope: CatalogRealm?, ceiling: String?) throws -> CatalogQuery {
        let p = request.query
        func s(_ k: String) -> String? {
            guard let v = p[k], !v.isEmpty else { return nil }
            return v
        }
        func i(_ k: String) -> Int? { s(k).flatMap(Int.init) }
        func d(_ k: String) -> Date? { s(k).flatMap(Double.init).map(Date.init(timeIntervalSince1970:)) }

        var q = CatalogQuery(scope: scope, ceiling: ceiling)
        // NOTE: `realm` is deliberately NOT read from the query string.
        q.text = s("q")
        q.collectionID = s("collection")
        q.lane = s("lane"); q.tier = s("tier"); q.character = s("character")
        q.source = s("source"); q.stock = s("stock"); q.genre = s("genre")
        q.arc = s("arc"); q.kind = s("kind"); q.mode = s("mode")
        q.minDurationMs = i("min_duration"); q.maxDurationMs = i("max_duration")
        q.minRating = i("min_rating")
        q.since = d("since"); q.until = d("until")
        q.orderBy = CatalogOrder(rawValue: s("order") ?? "newest") ?? .newest
        q.limit = min(i("limit") ?? 50, 500)
        q.offset = i("offset") ?? 0
        return q
    }

    private static func dict(for a: CatalogAsset) -> [String: Any] {
        var out: [String: Any] = [
            "id": a.id, "kind": a.kind, "realm": a.realm.rawValue,
            "created_at": a.createdAt.timeIntervalSince1970,
        ]
        // A clamped row has an empty path and no text; omit rather than send "".
        if !a.absolutePath.isEmpty { out["path"] = a.absolutePath }
        if !a.filename.isEmpty { out["filename"] = a.filename }
        func put(_ k: String, _ v: Any?) { if let v { out[k] = v } }
        put("prompt", a.prompt); put("prompt_raw", a.promptRaw); put("caption", a.caption)
        put("tier", a.contentMode); put("character", a.characterName); put("source", a.source)
        put("lane", a.lane); put("arc", a.arc); put("theme", a.theme)
        put("stock", a.stock); put("genre", a.genre); put("family", a.family); put("style", a.style)
        put("preset", a.preset); put("model", a.modelFamily); put("seed", a.seed)
        put("mode", a.mode); put("duration_ms", a.durationMs); put("fps", a.fps)
        put("frames", a.frames); put("resolution", a.resolution); put("aspect_ratio", a.aspectRatio)
        put("width", a.width); put("height", a.height)
        out["rating"] = a.rating
        out["favorite"] = a.favorite
        out["sealed"] = a.sealed
        return out
    }

    // MARK: - CLI

    public static func runCLIEntryPoint(args: [String]) {
        var port = defaultPort
        var dbPath: String? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--port": if i + 1 < args.count, let p = UInt16(args[i + 1]) { port = p; i += 1 }
            case "--db": if i + 1 < args.count { dbPath = args[i + 1]; i += 1 }
            case "--help", "-h":
                print("Usage: ComfyBoxGallery [--port \(defaultPort)] [--db PATH]")
                exit(0)
            default: break
            }
            i += 1
        }

        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                let store = try await CatalogStore.open(path: dbPath)
                let server = HTTPKit.Server(port: port) { req in
                    await handle(request: req, store: store)
                }
                try server.start()
                FileHandle.standardError.write(Data("gallery listening on 127.0.0.1:\(port)\n".utf8))
            } catch {
                FileHandle.standardError.write(Data("gallery failed to start: \(error)\n".utf8))
                exit(1)
            }
        }
        sem.wait()   // run forever; launchd owns the lifecycle
    }
}
```

- [ ] **Step 5: Run the tests**

Run: `swift test --filter GalleryServerTests`
Expected: PASS, 11 tests. The `testRealmQueryParameterCannotOverrideTheActorHeader` case is the one that matters most — it is the wire-level proof of the realm lock.

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxCatalog/HTTPKit.swift Sources/ComfyBoxCatalog/GalleryServer.swift Tests/ComfyBoxCatalogTests/GalleryServerTests.swift
git commit -m "feat(catalog): loopback HTTP gallery service with header-derived realm lock"
```

---

### Task 8: Migrate the real database, run the real backfill, install the agent

**Files:**
- Create: `~/Library/LaunchAgents/com.barkadabrew.comfybox-gallery.plist`
- Create: `scripts/gallery-backfill.sh`

**Interfaces:**
- Consumes: the `ComfyBoxGallery` binary and `CatalogBackfill`.
- Produces: a running service on `127.0.0.1:7871` and a populated catalog.

This is the first task that touches live data. Every step is reversible.

- [ ] **Step 1: Back up the live catalog**

```bash
cp ~/.comfybox/dam.sqlite3 ~/.comfybox/dam.sqlite3.bak-precatalog-$(date +%s)
ls -la ~/.comfybox/dam.sqlite3.bak-precatalog-*
```

Expected: a ~3.9 MB backup exists. **Rollback for this entire task is restoring that file.**

- [ ] **Step 2: Build only the gallery product**

```bash
shasum -a 256 .build/release/ComfyBox > /tmp/engine-hash-before.txt
swift build -c release --product ComfyBoxGallery
shasum -a 256 .build/release/ComfyBox > /tmp/engine-hash-after.txt
diff /tmp/engine-hash-before.txt /tmp/engine-hash-after.txt && echo "ENGINE UNCHANGED"
```

Expected: `ENGINE UNCHANGED`. If the hashes differ, STOP: restore `.build/release/ComfyBox` from `ComfyBox.bak-KNOWN-GOOD-20260730` before doing anything else.

- [ ] **Step 3: Add the backfill script**

Create `scripts/gallery-backfill.sh`:

```bash
#!/usr/bin/env bash
# Rebuild the catalog from what is on disk.
#
# Reads three trees: the ComfyBox gallery home on this Mac, and the two server
# studio trees mounted or synced locally. NEVER reads ~/Documents/Vaults —
# Bree's vault is out of scope.
set -euo pipefail

BIN="${BIN:-.build/release/ComfyBoxGallery}"
HOME_TREE="${HOME_TREE:-$HOME/Pictures/ComfyBox}"

if [[ ! -x "$BIN" ]]; then
  echo "build first: swift build -c release --product ComfyBoxGallery" >&2
  exit 1
fi

exec "$BIN" backfill --home "$HOME_TREE" "$@"
```

```bash
chmod +x scripts/gallery-backfill.sh
```

- [ ] **Step 4: Add the `backfill` subcommand to the gallery CLI**

In `GalleryServer.runCLIEntryPoint`, before the port parsing loop, add:

```swift
        if args.first == "backfill" {
            runBackfillCLI(args: Array(args.dropFirst()))
            return
        }
```

and add to `GalleryServer`:

```swift
    /// One-shot backfill. Trees are passed explicitly so no path is ever
    /// implied — in particular, no vault path can be reached by default.
    static func runBackfillCLI(args: [String]) {
        var home = NSString(string: "~/Pictures/ComfyBox").expandingTildeInPath
        var kiraRoot: String? = nil
        var breeRoot: String? = nil
        var dbPath: String? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--home": if i + 1 < args.count { home = args[i + 1]; i += 1 }
            case "--kira-studio": if i + 1 < args.count { kiraRoot = args[i + 1]; i += 1 }
            case "--bree-studio": if i + 1 < args.count { breeRoot = args[i + 1]; i += 1 }
            case "--db": if i + 1 < args.count { dbPath = args[i + 1]; i += 1 }
            default: break
            }
            i += 1
        }

        var trees: [BackfillTree] = [
            BackfillTree(id: "home", realm: nil, host: "mac", mediaRoot: home, metadataRoot: nil)
        ]
        if let k = kiraRoot {
            trees.append(BackfillTree(id: "kira", realm: .kira, host: "kira",
                                      mediaRoot: k + "/gallery", metadataRoot: k + "/metadata"))
            trees.append(BackfillTree(id: "kira-video", realm: .kira, host: "kira",
                                      mediaRoot: k + "/video", metadataRoot: k + "/metadata"))
        }
        if let b = breeRoot {
            trees.append(BackfillTree(id: "bree", realm: .shared, host: "bree",
                                      mediaRoot: b + "/gallery", metadataRoot: b + "/metadata"))
            trees.append(BackfillTree(id: "bree-video", realm: .shared, host: "bree",
                                      mediaRoot: b + "/video", metadataRoot: b + "/metadata"))
        }

        for t in trees where t.mediaRoot.contains("Vaults") {
            FileHandle.standardError.write(Data("refusing to read a vault path: \(t.mediaRoot)\n".utf8))
            exit(2)
        }

        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                let store = try await CatalogStore.open(path: dbPath)
                let report = try await CatalogBackfill.run(store: store, trees: trees)
                print("""
                    scanned:    \(report.filesScanned)
                    indexed:    \(report.assetsIndexed)
                    merged:     \(report.duplicatesMerged)
                    sidecars:   \(report.sidecarsRead)
                    edges:      \(report.edgesCreated)
                    skipped:    \(report.skipped)
                    """)
            } catch {
                FileHandle.standardError.write(Data("backfill failed: \(error)\n".utf8))
                exit(1)
            }
            sem.signal()
        }
        sem.wait()
    }
```

Rebuild: `swift build -c release --product ComfyBoxGallery`

- [ ] **Step 5: Dry-run the backfill against the COPY, not the live catalog**

```bash
cp ~/.comfybox/dam.sqlite3 /tmp/catalog-dryrun.sqlite3
.build/release/ComfyBoxGallery backfill --db /tmp/catalog-dryrun.sqlite3
sqlite3 /tmp/catalog-dryrun.sqlite3 \
  "SELECT realm, COUNT(*) FROM assets GROUP BY realm;
   SELECT COUNT(*) FROM asset_locations;
   SELECT COUNT(*) FROM asset_edges;
   SELECT c.slug, COUNT(*) FROM asset_collections ac JOIN collections c ON c.id=ac.collection_id GROUP BY c.slug;"
```

Expected: every row has a realm (no NULLs), locations ≥ asset count, and collection counts that look plausible. Record the numbers — Step 7 must reproduce them.

- [ ] **Step 6: Run the backfill on the live catalog**

```bash
.build/release/ComfyBoxGallery backfill
sqlite3 ~/.comfybox/dam.sqlite3 "SELECT COUNT(*) FROM assets; SELECT COUNT(*) FROM assets WHERE realm IS NULL;"
```

Expected: asset count ≥ 1072 (the pre-existing rows plus everything the sweep found), and **0** rows with a NULL realm.

- [ ] **Step 7: Verify idempotence on live data**

```bash
sqlite3 ~/.comfybox/dam.sqlite3 "SELECT COUNT(*) FROM assets;" > /tmp/count1.txt
.build/release/ComfyBoxGallery backfill
sqlite3 ~/.comfybox/dam.sqlite3 "SELECT COUNT(*) FROM assets;" > /tmp/count2.txt
diff /tmp/count1.txt /tmp/count2.txt && echo "IDEMPOTENT"
```

Expected: `IDEMPOTENT`.

- [ ] **Step 8: Verify permissions**

```bash
ls -l ~/.comfybox/dam.sqlite3 ~/.comfybox/dam.sqlite3-wal 2>/dev/null
ls -ld ~/.comfybox
```

Expected: `-rw-------` on the database and its WAL, `drwx------` on the directory. This closes the world-readable gap the spec recorded.

- [ ] **Step 9: Install the launchd agent**

Create `~/Library/LaunchAgents/com.barkadabrew.comfybox-gallery.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>com.barkadabrew.comfybox-gallery</string>
	<key>ProgramArguments</key>
	<array>
		<string>/Users/toddwalderman/Projects/zimage.swift/.build/release/ComfyBoxGallery</string>
		<string>--port</string>
		<string>7871</string>
	</array>
	<key>KeepAlive</key>
	<true/>
	<key>RunAtLoad</key>
	<true/>
	<key>StandardOutPath</key>
	<string>/Users/toddwalderman/.comfybox/gallery.log</string>
	<key>StandardErrorPath</key>
	<string>/Users/toddwalderman/.comfybox/gallery.err.log</string>
	<key>WorkingDirectory</key>
	<string>/Users/toddwalderman/Projects/zimage.swift</string>
</dict>
</plist>
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.barkadabrew.comfybox-gallery.plist
curl -s http://127.0.0.1:7871/healthz
```

Expected: `{"ok":true}`. Note this bootstraps only the **new** label — `com.barkadabrew.comfybox` is untouched.

- [ ] **Step 10: Verify the engine is still serving**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7870/v1/gallery/list?limit=1
launchctl print gui/$(id -u)/com.barkadabrew.comfybox | grep -E 'state|pid'
```

Expected: `200`, and the engine still `running` with its original pid. If the pid changed, the engine was restarted — record it and check `~/.comfybox/serve.err.log`.

- [ ] **Step 11: Commit**

```bash
git add scripts/gallery-backfill.sh Sources/ComfyBoxCatalog/GalleryServer.swift
git commit -m "feat(catalog): backfill CLI, launchd agent, live migration of dam.sqlite3"
```

---

### Task 9: Catalog client + Kira's tools (realm-locked, mode-clamped, share-gated)

**Repo:** `~/Projects/coffeeshop-server`

**Files:**
- Create: `src/catalog/client.ts`
- Create: `src/catalog/client.test.ts`
- Create: `src/kira/gallery-tools.ts`
- Create: `src/kira/gallery-tools.test.ts`

**Interfaces:**
- Consumes: the HTTP routes from Task 7.
- Produces: `CatalogClient` (`search`, `facets`, `collections`, `asset`), and `createKiraGalleryTools(): BuiltInTool[]` exposing `search_gallery`, `share_from_gallery`, `curate_collection`.

**Background the implementer needs:** `BuiltInTool` is the daemon's tool shape — `{name, description, parameters, execute(params)}` — see `src/kira/suggestions.ts` for a worked example. The mode clamp already exists in `src/kira/render-journal.ts`: tier-labeled counts surface at any chat mode; intent text and paths only at or below the ceiling. Reuse `normalizeSchedulerContentMode` from `src/kira/stream-mode.ts` for tier vocabulary.

- [ ] **Step 1: Write the failing client test**

Create `src/catalog/client.test.ts`:

```ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CatalogClient } from './client';

/** Capture what the client puts on the wire. */
function stubFetch(capture: { url?: string; headers?: Record<string, string> }, body: any) {
    return async (url: string, init?: any) => {
        capture.url = url;
        capture.headers = init?.headers ?? {};
        return { ok: true, status: 200, json: async () => body } as any;
    };
}

test('kira actor header is always sent and never overridable by caller params', async () => {
    const cap: { url?: string; headers?: Record<string, string> } = {};
    const c = new CatalogClient('http://127.0.0.1:7871', 'kira', stubFetch(cap, { items: [], count: 0 }));
    // A caller trying to widen scope must have no effect on the wire.
    await c.search({ q: 'tulip', realm: 'shared', scope: 'shared' } as any);
    assert.equal(cap.headers!['x-catalog-actor'], 'kira');
    assert.ok(!cap.url!.includes('realm='), `realm leaked into the query: ${cap.url}`);
    assert.ok(!cap.url!.includes('scope='), `scope leaked into the query: ${cap.url}`);
});

test('ceiling rides as a header, not a parameter', async () => {
    const cap: { url?: string; headers?: Record<string, string> } = {};
    const c = new CatalogClient('http://127.0.0.1:7871', 'kira', stubFetch(cap, { items: [], count: 0 }));
    await c.search({ q: 'x' }, { ceiling: 'banana' });
    assert.equal(cap.headers!['x-catalog-ceiling'], 'banana');
});

test('a down service yields an empty result, never a throw', async () => {
    const dead = async () => { throw new Error('ECONNREFUSED'); };
    const c = new CatalogClient('http://127.0.0.1:7871', 'kira', dead as any);
    assert.deepEqual(await c.search({ q: 'x' }), { items: [], count: 0, unavailable: true });
});

test('query params are encoded', async () => {
    const cap: { url?: string; headers?: Record<string, string> } = {};
    const c = new CatalogClient('http://127.0.0.1:7871', null, stubFetch(cap, { items: [], count: 0 }));
    await c.search({ q: 'film noir', collection: 'col-kira-autocord', limit: 5 });
    assert.ok(cap.url!.includes('q=film%20noir'), cap.url);
    assert.ok(cap.url!.includes('collection=col-kira-autocord'), cap.url);
    assert.ok(cap.url!.includes('limit=5'), cap.url);
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd ~/Projects/coffeeshop-server && node scripts/run-tests.mjs src/catalog/client.test.ts`
Expected: FAIL — cannot find `./client`.

- [ ] **Step 3: Write the client**

Create `src/catalog/client.ts`:

```ts
/**
 * client.ts — HTTP client for the ComfyBox gallery service (ComfyBoxGallery,
 * 127.0.0.1:7871 on the Mac).
 *
 * The ACTOR is fixed at construction, not per call, and rides as a header.
 * That is the realm lock: a caller cannot widen its own scope by passing a
 * parameter, because `realm`/`scope` are stripped before the query is built.
 *
 * Every failure is soft. The gallery is an index over files that still exist;
 * if the service is down, a search returns nothing rather than breaking a
 * conversation or a render cycle.
 */

export type CatalogActor = 'kira' | null;

export interface CatalogSearchParams {
    q?: string;
    collection?: string;
    lane?: string;
    tier?: string;
    character?: string;
    source?: string;
    stock?: string;
    genre?: string;
    arc?: string;
    kind?: 'image' | 'video';
    mode?: 'i2v' | 't2v';
    min_duration?: number;
    max_duration?: number;
    min_rating?: number;
    order?: 'newest' | 'oldest' | 'rating';
    limit?: number;
    offset?: number;
}

export interface CatalogItem {
    id: string;
    kind: string;
    realm: string;
    created_at: number;
    path?: string;
    filename?: string;
    prompt?: string;
    tier?: string;
    lane?: string;
    stock?: string;
    genre?: string;
    mode?: string;
    duration_ms?: number;
    rating: number;
    favorite: boolean;
}

export interface CatalogSearchResult {
    items: CatalogItem[];
    count: number;
    unavailable?: boolean;
}

/** Params a caller must never be able to set — the realm lock. */
const FORBIDDEN_PARAMS = new Set(['realm', 'scope', 'actor']);

export const DEFAULT_CATALOG_URL = 'http://127.0.0.1:7871';

export class CatalogClient {
    constructor(
        private readonly baseUrl: string = DEFAULT_CATALOG_URL,
        private readonly actor: CatalogActor = null,
        private readonly fetchImpl: typeof fetch = fetch,
    ) {}

    private headers(opts?: { ceiling?: string }): Record<string, string> {
        const h: Record<string, string> = {};
        if (this.actor) h['x-catalog-actor'] = this.actor;
        if (opts?.ceiling) h['x-catalog-ceiling'] = opts.ceiling;
        return h;
    }

    private qs(params: Record<string, any>): string {
        const parts: string[] = [];
        for (const [k, v] of Object.entries(params)) {
            if (FORBIDDEN_PARAMS.has(k)) continue;      // the lock
            if (v === undefined || v === null || v === '') continue;
            parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
        }
        return parts.length ? `?${parts.join('&')}` : '';
    }

    private async get(path: string, opts?: { ceiling?: string }): Promise<any | null> {
        try {
            const res: any = await this.fetchImpl(this.baseUrl + path, { headers: this.headers(opts) });
            if (!res || res.ok === false) return null;
            return await res.json();
        } catch {
            return null;
        }
    }

    async search(params: CatalogSearchParams, opts?: { ceiling?: string }): Promise<CatalogSearchResult> {
        const body = await this.get('/v1/catalog/search' + this.qs(params as any), opts);
        if (!body) return { items: [], count: 0, unavailable: true };
        return { items: body.items ?? [], count: body.count ?? 0 };
    }

    async asset(id: string, opts?: { ceiling?: string }): Promise<any | null> {
        return this.get(`/v1/catalog/asset/${encodeURIComponent(id)}`, opts);
    }

    async collections(): Promise<any[]> {
        const body = await this.get('/v1/catalog/collections');
        return body?.items ?? [];
    }

    async facets(): Promise<Record<string, Record<string, number>>> {
        return (await this.get('/v1/catalog/facets')) ?? {};
    }
}
```

- [ ] **Step 4: Run the client tests**

Run: `node scripts/run-tests.mjs src/catalog/client.test.ts`
Expected: PASS, 4 tests.

- [ ] **Step 5: Write the failing tool test**

Create `src/kira/gallery-tools.test.ts`:

```ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createKiraGalleryTools, __setCatalogClientForTest } from './gallery-tools';

function fakeClient(items: any[]) {
    return {
        lastOpts: undefined as any,
        async search(_p: any, opts?: any) { this.lastOpts = opts; return { items, count: items.length }; },
        async asset(id: string, opts?: any) {
            this.lastOpts = opts;
            return items.find((i) => i.id === id) ?? null;
        },
        async collections() { return [{ id: 'col-kira-autocord', slug: 'kira-autocord', name: 'Autocord', realm: 'kira' }]; },
        async facets() { return {}; },
    };
}

function tool(name: string) {
    const t = createKiraGalleryTools().find((x) => x.name === name);
    assert.ok(t, `missing tool ${name}`);
    return t!;
}

test('search_gallery passes the ACTIVE chat mode as the ceiling', async () => {
    const c = fakeClient([{ id: 'a', tier: 'neutral', path: '/tmp/a.png', rating: 0, favorite: false }]);
    __setCatalogClientForTest(c as any, () => 'apple');
    await tool('search_gallery').execute({ q: 'tulip' });
    assert.equal(c.lastOpts.ceiling, 'apple');
});

test('share_from_gallery RE-CHECKS the ceiling at send time', async () => {
    // Found while the mode allowed it; the mode has since dropped.
    const c = fakeClient([{ id: 'a', tier: 'avocado', path: '/tmp/a.png', rating: 0, favorite: false }]);
    __setCatalogClientForTest(c as any, () => 'apple');
    const res: any = await tool('share_from_gallery').execute({ asset_id: 'a' });
    assert.equal(res.success, false, 'must not surface an above-ceiling asset');
    assert.match(res.error, /mode/i);
});

test('share_from_gallery succeeds at or below the ceiling', async () => {
    const c = fakeClient([{ id: 'a', tier: 'banana', path: '/tmp/a.png', rating: 0, favorite: false }]);
    __setCatalogClientForTest(c as any, () => 'avocado');
    const res: any = await tool('share_from_gallery').execute({ asset_id: 'a' });
    assert.equal(res.success, true);
    assert.equal(res.path, '/tmp/a.png');
});

test('share_from_gallery refuses an asset with no path (already clamped upstream)', async () => {
    const c = fakeClient([{ id: 'a', tier: 'avocado', rating: 0, favorite: false }]);
    __setCatalogClientForTest(c as any, () => 'avocado');
    const res: any = await tool('share_from_gallery').execute({ asset_id: 'a' });
    assert.equal(res.success, false);
});

test('search_gallery reports counts even when nothing is shareable', async () => {
    const c = fakeClient([{ id: 'a', tier: 'avocado', rating: 0, favorite: false }]);
    __setCatalogClientForTest(c as any, () => 'neutral');
    const res: any = await tool('search_gallery').execute({ q: 'x' });
    assert.equal(res.count, 1, 'tier-labeled counts are metadata and always surface');
    assert.equal(res.items[0].tier, 'avocado');
    assert.equal(res.items[0].path, undefined);
});

test('no gallery tool accepts a realm or scope parameter', () => {
    for (const t of createKiraGalleryTools()) {
        const props = Object.keys(t.parameters?.properties ?? {});
        for (const forbidden of ['realm', 'scope', 'actor']) {
            assert.ok(!props.includes(forbidden),
                `${t.name} exposes ${forbidden} — the realm lock must not be expressible`);
        }
    }
});

test('a down service degrades to an empty result', async () => {
    const dead = {
        async search() { return { items: [], count: 0, unavailable: true }; },
        async asset() { return null; },
        async collections() { return []; },
        async facets() { return {}; },
    };
    __setCatalogClientForTest(dead as any, () => 'banana');
    const res: any = await tool('search_gallery').execute({ q: 'x' });
    assert.equal(res.count, 0);
    assert.equal(res.success, true, 'unavailability is not an error she has to handle');
});
```

- [ ] **Step 6: Run it and watch it fail**

Run: `node scripts/run-tests.mjs src/kira/gallery-tools.test.ts`
Expected: FAIL — cannot find `./gallery-tools`.

- [ ] **Step 7: Write the tools**

Create `src/kira/gallery-tools.ts`:

```ts
/**
 * gallery-tools.ts — Kira's access to her own archive.
 *
 * Three tools: find her work, share a piece of it, and organize her own
 * genres. All three are locked to her realm SERVICE-side (the client sends a
 * fixed actor header; no parameter here can widen it), and all three inherit
 * the mode clamp already established in render-journal.ts:
 *
 *   tier-labeled COUNTS are metadata and surface at any chat mode;
 *   intent text and file paths surface only at or below the active ceiling.
 *
 * The ceiling is re-checked at SHARE time, not only at search time — the mode
 * can change between her finding something and her sending it, and those are
 * two different moments.
 *
 * Before this existed her memory of her own work was render-journal.jsonl:
 * 500 lines, roughly two days, against thousands of assets.
 */

import type { BuiltInTool } from '../tool-types';
import { CatalogClient, DEFAULT_CATALOG_URL, type CatalogItem } from '../catalog/client';
import { getActiveChatCeiling } from './stream-mode';

const TIER_ORDER = ['neutral', 'apple', 'banana', 'avocado'] as const;

function tierRank(tier?: string): number {
    const i = TIER_ORDER.indexOf((tier ?? '').toLowerCase() as any);
    return i < 0 ? 0 : i;
}

let client: CatalogClient | null = null;
let ceilingFn: () => string = getActiveChatCeiling;

function catalog(): CatalogClient {
    if (!client) client = new CatalogClient(process.env.COMFYBOX_GALLERY_URL || DEFAULT_CATALOG_URL, 'kira');
    return client;
}

/** Test seam only. */
export function __setCatalogClientForTest(c: CatalogClient, ceiling: () => string): void {
    client = c;
    ceilingFn = ceiling;
}

export function createKiraGalleryTools(): BuiltInTool[] {
    return [
        {
            name: 'search_gallery',
            description: [
                'Search YOUR OWN archive — everything you have ever made, not just the last',
                'couple of days. Find by words in the prompt, by genre collection (your Still',
                'Life, Autocord Photography, Decoupage Designs, Nightlife, Erotic Portraiture,',
                'Adult Scenes, Dreams & Memories), by film stock, by tier, by date, by stills',
                'or clips. Results respect the current conversation mode: you always see what',
                'exists and its tier, but the text and the file only come back for work at or',
                'below the mode you are in.',
            ].join(' '),
            parameters: {
                type: 'object',
                properties: {
                    q: { type: 'string', description: 'Words to look for in the prompt or caption.' },
                    collection: { type: 'string', description: 'A collection id, e.g. col-kira-autocord. Matches its children too.' },
                    tier: { type: 'string', enum: [...TIER_ORDER], description: 'neutral | apple | banana | avocado' },
                    kind: { type: 'string', enum: ['image', 'video'] },
                    stock: { type: 'string', description: 'Film stock, for Autocord work.' },
                    limit: { type: 'number', description: 'Max results (default 20).' },
                },
                required: [],
            },
            async execute(params: Record<string, any>) {
                const ceiling = ceilingFn();
                const res = await catalog().search(
                    {
                        q: params.q, collection: params.collection, tier: params.tier,
                        kind: params.kind, stock: params.stock,
                        limit: Math.min(Number(params.limit) || 20, 100),
                    },
                    { ceiling },
                );
                return {
                    success: true,
                    count: res.count,
                    ceiling,
                    ...(res.unavailable ? { note: 'the gallery service is not answering right now' } : {}),
                    items: res.items.map((i: CatalogItem) => ({
                        id: i.id, kind: i.kind, tier: i.tier, lane: i.lane,
                        stock: i.stock, genre: i.genre, created_at: i.created_at,
                        // path/prompt are absent when the service clamped them.
                        ...(i.path ? { path: i.path } : {}),
                        ...(i.prompt ? { prompt: i.prompt } : {}),
                    })),
                };
            },
        },
        {
            name: 'share_from_gallery',
            description: [
                'Pull one piece out of your archive to show right now. Give the asset id from',
                'search_gallery. The current conversation mode is checked again at this moment —',
                'if the mode has changed since you found it, the file is withheld.',
            ].join(' '),
            parameters: {
                type: 'object',
                properties: {
                    asset_id: { type: 'string', description: 'The id from search_gallery.' },
                },
                required: ['asset_id'],
            },
            async execute(params: Record<string, any>) {
                const ceiling = ceilingFn();
                // Re-fetch rather than trust anything the caller carried over.
                const row = await catalog().asset(String(params.asset_id), { ceiling });
                if (!row) return { success: false, error: 'no such asset in your gallery' };
                if (tierRank(row.tier) > tierRank(ceiling)) {
                    return {
                        success: false,
                        error: `that one is ${row.tier} and the conversation mode is ${ceiling} right now`,
                    };
                }
                if (!row.path) {
                    return { success: false, error: 'that asset has no file available right now' };
                }
                return { success: true, id: row.id, kind: row.kind, tier: row.tier, path: row.path };
            },
        },
        {
            name: 'curate_collection',
            description: [
                'Organize your own bodies of work. List your collections, or file a piece of',
                'your work into one. These are yours — you can shape them. You contribute to the',
                'shared bodies (Decoupage, Photography) but do not restructure them.',
            ].join(' '),
            parameters: {
                type: 'object',
                properties: {
                    action: { type: 'string', enum: ['list', 'file'], description: 'list your collections, or file an asset into one' },
                    asset_id: { type: 'string', description: 'For action=file.' },
                    collection: { type: 'string', description: 'Collection id, for action=file.' },
                },
                required: ['action'],
            },
            async execute(params: Record<string, any>) {
                if (params.action === 'list') {
                    return { success: true, collections: await catalog().collections() };
                }
                if (!params.asset_id || !params.collection) {
                    return { success: false, error: 'action=file needs asset_id and collection' };
                }
                const ok = await catalog().file(String(params.asset_id), String(params.collection));
                return ok
                    ? { success: true, filed: params.asset_id, into: params.collection }
                    : { success: false, error: 'could not file that — check the ids' };
            },
        },
    ];
}
```

- [ ] **Step 8: Add the `file` call to the client and the route to the server**

In `src/catalog/client.ts`, add to `CatalogClient`:

```ts
    async file(assetId: string, collectionId: string): Promise<boolean> {
        try {
            const res: any = await this.fetchImpl(this.baseUrl + '/v1/catalog/file', {
                method: 'POST',
                headers: { ...this.headers(), 'content-type': 'application/json' },
                body: JSON.stringify({ asset_id: assetId, collection_id: collectionId }),
            });
            return !!res && res.ok !== false;
        } catch {
            return false;
        }
    }
```

In `GalleryServer.handle`, add before `default:`:

```swift
            case ("POST", "/v1/catalog/file"):
                guard let obj = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
                      let assetID = obj["asset_id"] as? String,
                      let collectionID = obj["collection_id"] as? String else {
                    return .error(400, "asset_id and collection_id required")
                }
                do {
                    // `scope` is the actor: nil = the service itself, .kira = her.
                    try await store.file(assetID: assetID, into: collectionID, by: scope)
                    return .json(["ok": true])
                } catch {
                    return .error(403, error.localizedDescription)
                }
```

- [ ] **Step 9: Add `getActiveChatCeiling` if it does not already exist**

Check first: `grep -n "getActiveChatCeiling\|activeChatMode" src/kira/stream-mode.ts`. If absent, add to `src/kira/stream-mode.ts`:

```ts
/**
 * The tier ceiling for the CURRENT conversation — the same value the render
 * journal clamps against. Content at or below this may surface text and paths;
 * above it, only tier-labeled counts.
 */
export function getActiveChatCeiling(): string {
    return normalizeSchedulerContentMode(process.env.KIRA_CHAT_MODE) || 'apple';
}
```

If the daemon already exposes an active chat mode accessor, call that instead and delete this shim — do not introduce a second source of truth for the mode.

- [ ] **Step 10: Run the tests**

Run: `node scripts/run-tests.mjs src/kira/gallery-tools.test.ts src/catalog/client.test.ts`
Expected: PASS, 11 tests.

- [ ] **Step 11: Register the tools**

Find where `createSuggestionTools()` is registered (`grep -rn "createSuggestionTools" src/ --include=*.ts | grep -v test`) and register `createKiraGalleryTools()` in the same place, the same way.

- [ ] **Step 12: Commit**

```bash
git add src/catalog src/kira/gallery-tools.ts src/kira/gallery-tools.test.ts src/kira/stream-mode.ts
git commit -m "feat(kira): catalog client and realm-locked, mode-clamped gallery tools"
```

---

### Task 10: Bree's tool and the Studio history fallthrough

**Repo:** `~/Projects/coffeeshop-server`

**Files:**
- Create: `src/tools/gallery-search-tools.ts`
- Create: `src/tools/gallery-search-tools.test.ts`
- Modify: `src/studio/generation-history.ts`
- Modify: `src/studio/generation-history.test.ts`

**Interfaces:**
- Consumes: `CatalogClient` (Task 9).
- Produces: `createGallerySearchTools(): BuiltInTool[]` exposing `search_gallery` for Bree (unscoped), and `GenerationHistory.searchWithFallthrough(keyword, catalog)`.

**Background:** Bree is a consumer on par with Todd — her client is constructed with actor `null`, so she sees both realms. The Studio bot's `GenerationHistory.search` only reaches its own 500-entry window; past that it should ask the catalog.

- [ ] **Step 1: Write the failing tests**

Create `src/tools/gallery-search-tools.test.ts`:

```ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGallerySearchTools, __setGalleryClientForTest } from './gallery-search-tools';

function fake(items: any[]) {
    return {
        lastHeadersActor: undefined as any,
        async search() { return { items, count: items.length }; },
        async asset(id: string) { return items.find((i) => i.id === id) ?? null; },
        async collections() { return []; },
        async facets() { return { tier: { neutral: 3 } }; },
    };
}

test('Bree sees both realms — her tool does not scope', async () => {
    const c = fake([
        { id: 'k', realm: 'kira', tier: 'neutral', path: '/tmp/k.png', rating: 0, favorite: false },
        { id: 's', realm: 'shared', tier: 'neutral', path: '/tmp/s.png', rating: 0, favorite: false },
    ]);
    __setGalleryClientForTest(c as any);
    const t = createGallerySearchTools().find((x) => x.name === 'search_gallery')!;
    const res: any = await t.execute({ q: 'x' });
    assert.equal(res.count, 2);
    assert.deepEqual(res.items.map((i: any) => i.realm).sort(), ['kira', 'shared']);
});

test('facets are exposed so a caller can browse rather than guess', async () => {
    __setGalleryClientForTest(fake([]) as any);
    const t = createGallerySearchTools().find((x) => x.name === 'gallery_facets')!;
    const res: any = await t.execute({});
    assert.equal(res.facets.tier.neutral, 3);
});
```

Add to `src/studio/generation-history.test.ts`:

```ts
test('search falls through to the catalog past the 500-entry window', async () => {
    const h = new GenerationHistory('/tmp/gh-fallthrough-' + Date.now() + '.json', 500);
    h.append({ id: 'local-1', prompt: 'a local tulip', character: 'kira',
               outputPath: '/tmp/a.png', createdAt: new Date().toISOString() } as any);

    const catalog = {
        async search() {
            return { items: [{ id: 'old-1', prompt: 'an old tulip', path: '/tmp/old.png' }], count: 1 };
        },
    };
    const res = await h.searchWithFallthrough('tulip', catalog as any);
    assert.equal(res.local.length, 1);
    assert.equal(res.archive.length, 1);
    assert.equal(res.archive[0].id, 'old-1');
});

test('fallthrough survives a missing catalog', async () => {
    const h = new GenerationHistory('/tmp/gh-nocat-' + Date.now() + '.json', 500);
    const res = await h.searchWithFallthrough('anything', null as any);
    assert.deepEqual(res.archive, []);
});
```

- [ ] **Step 2: Run them and watch them fail**

Run: `node scripts/run-tests.mjs src/tools/gallery-search-tools.test.ts src/studio/generation-history.test.ts`
Expected: FAIL — missing module, and `searchWithFallthrough` undefined.

- [ ] **Step 3: Write Bree's tools**

Create `src/tools/gallery-search-tools.ts`:

```ts
/**
 * gallery-search-tools.ts — gallery search for consumers who see everything.
 *
 * Bree is a consumer on par with Todd: her client is constructed with a NULL
 * actor, so the service applies no realm scope. This is platform code and
 * lives outside src/kira/ deliberately — the Kira-scoped variant is
 * src/kira/gallery-tools.ts and the two must not be merged, because the whole
 * difference between them is the actor.
 */

import type { BuiltInTool } from '../tool-types';
import { CatalogClient, DEFAULT_CATALOG_URL } from '../catalog/client';

let client: CatalogClient | null = null;

function catalog(): CatalogClient {
    if (!client) client = new CatalogClient(process.env.COMFYBOX_GALLERY_URL || DEFAULT_CATALOG_URL, null);
    return client;
}

/** Test seam only. */
export function __setGalleryClientForTest(c: CatalogClient): void {
    client = c;
}

export function createGallerySearchTools(): BuiltInTool[] {
    return [
        {
            name: 'search_gallery',
            description: [
                'Search the whole ComfyBox gallery — every image and clip generated by any',
                'application, going back to the beginning, not just what is recent. Filter by',
                'words, collection, character, tier, film stock, stills vs clips, or date.',
            ].join(' '),
            parameters: {
                type: 'object',
                properties: {
                    q: { type: 'string', description: 'Words to look for in the prompt or caption.' },
                    collection: { type: 'string', description: 'Collection id; matches its children too.' },
                    character: { type: 'string', description: 'Who is depicted.' },
                    tier: { type: 'string', enum: ['neutral', 'apple', 'banana', 'avocado'] },
                    kind: { type: 'string', enum: ['image', 'video'] },
                    source: { type: 'string', description: 'Which application requested the render.' },
                    limit: { type: 'number', description: 'Max results (default 20).' },
                },
                required: [],
            },
            async execute(params: Record<string, any>) {
                const res = await catalog().search({
                    q: params.q, collection: params.collection, character: params.character,
                    tier: params.tier, kind: params.kind, source: params.source,
                    limit: Math.min(Number(params.limit) || 20, 100),
                });
                return {
                    success: true,
                    count: res.count,
                    ...(res.unavailable ? { note: 'gallery service unavailable' } : {}),
                    items: res.items,
                };
            },
        },
        {
            name: 'gallery_facets',
            description: 'Value counts per facet across the gallery, so it can be browsed rather than only searched.',
            parameters: { type: 'object', properties: {}, required: [] },
            async execute() {
                return { success: true, facets: await catalog().facets() };
            },
        },
    ];
}
```

- [ ] **Step 4: Add the Studio fallthrough**

Add to the `GenerationHistory` class in `src/studio/generation-history.ts`:

```ts
    /**
     * Keyword search across BOTH the local 500-entry window and the durable
     * catalog behind it. The window stays the hot cache; the catalog is the
     * archive. A missing or unreachable catalog degrades to local-only rather
     * than failing the caller.
     */
    async searchWithFallthrough(
        keyword: string,
        catalog: { search(params: any): Promise<{ items: any[]; count: number }> } | null,
    ): Promise<{ local: GenerationRecord[]; archive: any[] }> {
        const local = this.search(keyword);
        if (!catalog) return { local, archive: [] };
        try {
            const res = await catalog.search({ q: keyword, limit: 50 });
            const localPaths = new Set(local.map((r) => r.outputPath));
            // Do not show the same asset twice — the window already has it.
            return { local, archive: (res.items ?? []).filter((i: any) => !localPaths.has(i.path)) };
        } catch {
            return { local, archive: [] };
        }
    }
```

- [ ] **Step 5: Run the tests**

Run: `node scripts/run-tests.mjs src/tools/gallery-search-tools.test.ts src/studio/generation-history.test.ts`
Expected: PASS.

- [ ] **Step 6: Register Bree's tools and run the full gate**

Register `createGallerySearchTools()` wherever the other platform tool factories are registered (`grep -rn "createSuggestionTools\|BuiltInTool\[\]" src/tool-registry.ts`).

Run: `node scripts/run-tests.mjs`
Expected: the full suite passes (~720 tests). If `async-media-turn.test.ts` fails, re-run once — it is a known ~1-in-3 flake unrelated to this work.

- [ ] **Step 7: Commit**

```bash
git add src/tools/gallery-search-tools.ts src/tools/gallery-search-tools.test.ts src/studio/generation-history.ts src/studio/generation-history.test.ts
git commit -m "feat(gallery): Bree's unscoped search tools and Studio history fallthrough"
```

---

### Task 11: Converge the desktop on one gallery over the whole catalog

**Repo:** `~/Projects/zimage.swift`

**Files:**
- Create: `Sources/ComfyBoxDesktop/DAM/CatalogBrowser.swift`
- Modify: `Sources/ComfyBoxDesktop/Views/GalleryView.swift`
- Test: `Tests/ComfyBoxDesktopTests/CatalogBrowserTests.swift`

**Interfaces:**
- Consumes: `CatalogStore`, `CatalogQuery`, `CatalogFacets`, `CatalogCollection` from `ComfyBoxCatalog`.
- Produces: `@Observable final class CatalogBrowser` with `load()`, `apply(filter:)`, `collections`, `facets`, `items`, and `isRemote(_:) -> Bool` / `streamURL(for:) -> URL?`.

**Why this task exists.** The desktop currently has **two** disagreeing galleries: `GalleryView` (local `DAMStore`, Mac files only) and `RemoteGalleryView` (`/v1/gallery/list`, a bare directory listing with no metadata). That split is the "Mac Gallery and server Gallery are different" problem living inside one app. `CatalogBrowser` replaces both readers with one query over the catalog; a row's locations decide whether its bytes open from disk or stream from the engine's existing `/v1/gallery/file`.

`DAMStore` is **not** deleted — `AssetIngestor` still writes through it as the fallback path, and its rating/favorite/secure operations still work. Only the browsing reader changes.

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxDesktopTests/CatalogBrowserTests.swift`:

```swift
import XCTest
import ComfyBoxCatalog
@testable import ComfyBoxDesktop

@MainActor
final class CatalogBrowserTests: XCTestCase {
    private var path: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        path = NSTemporaryDirectory() + "browser-\(UUID().uuidString).sqlite3"
        store = try await CatalogStore.open(path: path)
        try await store.upsert(CatalogAsset(id: "local", filename: "l.png",
                                            absolutePath: "/tmp/l.png", realm: .shared,
                                            contentMode: "neutral"), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "local",
            AssetLocation(host: "mac", path: "/tmp/l.png", mtime: Date()))
        try await store.upsert(CatalogAsset(id: "remote", filename: "r.png",
                                            absolutePath: "/home/todd/.kira/studio/gallery/Kira/r.png",
                                            realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "remote",
            AssetLocation(host: "kira", path: "/home/todd/.kira/studio/gallery/Kira/r.png", mtime: Date()))
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: path)
        try await super.tearDown()
    }

    func testDesktopSeesBothRealms() async throws {
        let b = CatalogBrowser(store: store)
        await b.load()
        XCTAssertEqual(Set(b.items.map(\.id)), ["local", "remote"],
                       "Todd's surface is unscoped — one gallery over everything")
    }

    func testARowOnlyOnTheServerIsMarkedRemoteAndGetsAStreamURL() async throws {
        let b = CatalogBrowser(store: store, engineBaseURL: "http://127.0.0.1:7870")
        await b.load()
        let remote = try XCTUnwrap(b.items.first { $0.id == "remote" })
        let local = try XCTUnwrap(b.items.first { $0.id == "local" })
        XCTAssertTrue(await b.isRemote(remote))
        XCTAssertFalse(await b.isRemote(local))
        let url = try XCTUnwrap(await b.streamURL(for: remote))
        XCTAssertTrue(url.absoluteString.hasPrefix("http://127.0.0.1:7870/v1/gallery/file?path="))
    }

    func testCollectionFilterNarrowsTheGrid() async throws {
        let b = CatalogBrowser(store: store)
        await b.apply(filter: CatalogQuery(collectionID: "col-kira-autocord"))
        XCTAssertEqual(b.items.map(\.id), ["remote"])
    }

    func testFacetsPopulateForTheRail() async throws {
        let b = CatalogBrowser(store: store)
        await b.load()
        XCTAssertEqual(b.facets.lane["shoot"], 1)
        XCTAssertFalse(b.collections.isEmpty)
    }

    func testTheDesktopIsNotRealmScoped() async throws {
        let b = CatalogBrowser(store: store)
        await b.apply(filter: CatalogQuery(lane: "shoot"))
        XCTAssertEqual(b.items.first?.realm, .kira,
                       "a kira row must be reachable from Todd's surface")
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `swift test --filter CatalogBrowserTests`
Expected: FAIL — `CatalogBrowser` is undefined.

- [ ] **Step 3: Write the browser**

Create `Sources/ComfyBoxDesktop/DAM/CatalogBrowser.swift`:

```swift
// CatalogBrowser.swift — one gallery reader over the whole catalog.
//
// Replaces two disagreeing readers: GalleryView's local DAMStore fetch (Mac
// files only) and RemoteGalleryService's /v1/gallery/list (a bare directory
// listing with no metadata). Those two ARE the "Mac gallery and server gallery
// are different" problem, inside one app.
//
// Todd's surface is deliberately UNSCOPED — he sees both realms. Only Kira's
// MCP tool carries a realm lock.
//
// DAMStore is not replaced: AssetIngestor still writes through it, and ratings,
// favorites and the secure-vault moves still go there. Only browsing changes.

import Foundation
import ComfyBoxCatalog

@Observable
@MainActor
public final class CatalogBrowser {
    private let store: CatalogStore
    private let engineBaseURL: String

    public private(set) var items: [CatalogAsset] = []
    public private(set) var collections: [CatalogCollection] = []
    public private(set) var facets = CatalogFacets()
    public private(set) var isLoading = false
    public var error: String?

    public init(store: CatalogStore, engineBaseURL: String = "http://127.0.0.1:7870") {
        self.store = store
        self.engineBaseURL = engineBaseURL
    }

    public func load() async {
        await apply(filter: CatalogQuery(limit: 500))
    }

    public func apply(filter: CatalogQuery) async {
        isLoading = true
        defer { isLoading = false }
        error = nil
        do {
            var q = filter
            q.scope = nil          // Todd's surface sees everything.
            if q.limit == 50 { q.limit = 500 }
            items = try await store.search(q)
            collections = try await store.collections(visibleTo: nil)
            facets = try await store.facets(scope: nil)
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// True when no location for this asset exists on this Mac.
    public func isRemote(_ asset: CatalogAsset) async -> Bool {
        guard let locs = try? await store.locations(of: asset.id) else {
            return !FileManager.default.fileExists(atPath: asset.absolutePath)
        }
        let localPaths = locs.filter { $0.host == "mac" }.map(\.path)
        if localPaths.contains(where: { FileManager.default.fileExists(atPath: $0) }) { return false }
        return !FileManager.default.fileExists(atPath: asset.absolutePath)
    }

    /// Stream URL for a server-side asset, via the engine route the spec
    /// deliberately left unchanged.
    public func streamURL(for asset: CatalogAsset) async -> URL? {
        let locs = (try? await store.locations(of: asset.id)) ?? []
        let path = locs.first(where: { $0.host != "mac" })?.path ?? asset.absolutePath
        var c = URLComponents(string: engineBaseURL + "/v1/gallery/file")
        c?.queryItems = [URLQueryItem(name: "path", value: path)]
        return c?.url
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `swift test --filter CatalogBrowserTests`
Expected: PASS, 5 tests.

- [ ] **Step 5: Wire the rail into GalleryView**

In `Sources/ComfyBoxDesktop/Views/GalleryView.swift`, add the browser alongside the existing state (near `@State private var sidecar = SidecarService()` at line 55):

```swift
    @State private var browser: CatalogBrowser?
    @State private var selectedCollectionID: String?
    @State private var selectedLane: String?
```

Add a leading sidebar to the view body, and drive the grid from `browser?.items` when a browser is present:

```swift
    /// Facet + collection rail. Collections come first because they are the
    /// bodies of work; lanes below them are the finer cut.
    @ViewBuilder
    private var catalogRail: some View {
        if let browser {
            List(selection: $selectedCollectionID) {
                Section("Collections") {
                    ForEach(browser.collections.filter { $0.parentID == nil }, id: \.id) { root in
                        DisclosureGroup {
                            ForEach(browser.collections.filter { $0.parentID == root.id }, id: \.id) { child in
                                Label("\(child.name)  (\(browser.facets.collection[child.id] ?? 0))",
                                      systemImage: "folder")
                                    .tag(child.id as String?)
                            }
                        } label: {
                            Label("\(root.name)  (\(browser.facets.collection[root.id] ?? 0))",
                                  systemImage: "folder.fill")
                                .tag(root.id as String?)
                        }
                    }
                }
                Section("Lane") {
                    ForEach(browser.facets.lane.sorted(by: { $0.key < $1.key }), id: \.key) { lane, count in
                        Button("\(lane)  (\(count))") {
                            selectedLane = lane
                            Task { await browser.apply(filter: CatalogQuery(lane: lane)) }
                        }
                    }
                }
            }
            .frame(minWidth: 200)
            .onChange(of: selectedCollectionID) { _, newValue in
                Task { await browser.apply(filter: CatalogQuery(collectionID: newValue)) }
            }
        }
    }
```

Initialize it where the view first appears:

```swift
    .task {
        guard browser == nil else { return }
        if let store = try? await CatalogStore.open() {
            let b = CatalogBrowser(store: store)
            await b.load()
            browser = b
        }
    }
```

- [ ] **Step 6: Build the desktop app and confirm it still compiles**

```bash
swift build -c release --product ComfyBoxDesktop
```

Expected: builds clean. This does not touch `.build/release/ComfyBox`.

- [ ] **Step 7: Commit**

```bash
git add Sources/ComfyBoxDesktop/DAM/CatalogBrowser.swift Sources/ComfyBoxDesktop/Views/GalleryView.swift Tests/ComfyBoxDesktopTests/CatalogBrowserTests.swift
git commit -m "feat(desktop): one gallery over the whole catalog with a collection/facet rail"
```

---

### Task 12: End-to-end verification and deploy

**Files:**
- Create: `Tests/ComfyBoxCatalogTests/InvariantTests.swift`
- Create: `docs/methods/gallery-catalog-runbook.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the invariant suite and an operational runbook with rollback.

- [ ] **Step 1: Write the invariant tests**

Create `Tests/ComfyBoxCatalogTests/InvariantTests.swift`:

```swift
import XCTest
@testable import ComfyBoxCatalog

/// The properties the whole design rests on. If one of these fails, the
/// catalog is wrong regardless of what any other test says.
final class InvariantTests: XCTestCase {
    private var path: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        path = NSTemporaryDirectory() + "inv-\(UUID().uuidString).sqlite3"
        store = try await CatalogStore.open(path: path)
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: path)
        try await super.tearDown()
    }

    /// Exhaustive: no combination of filters lets a kira-scoped query see a
    /// shared row. Generated rather than hand-listed so a new filter added
    /// later is covered without anyone remembering to add a case.
    func testRealmLockHoldsUnderEveryFilterCombination() async throws {
        for i in 0..<20 {
            try await store.upsert(CatalogAsset(
                id: "k\(i)", kind: i % 2 == 0 ? "image" : "video",
                filename: "k\(i)", absolutePath: "/tmp/k\(i)",
                realm: .kira, prompt: "shared word \(i)",
                contentMode: ["neutral", "apple", "banana", "avocado"][i % 4],
                characterName: "Kira", lane: ["still", "shoot", "tile", "kira"][i % 4]),
                explicitCollectionIDs: [])
            try await store.upsert(CatalogAsset(
                id: "s\(i)", kind: i % 2 == 0 ? "image" : "video",
                filename: "s\(i)", absolutePath: "/tmp/s\(i)",
                realm: .shared, prompt: "shared word \(i)",
                contentMode: ["neutral", "apple", "banana", "avocado"][i % 4],
                characterName: "Kira", lane: ["still", "shoot", "tile", "kira"][i % 4]),
                explicitCollectionIDs: [])
        }

        let texts: [String?] = [nil, "shared", "word"]
        let lanes: [String?] = [nil, "still", "shoot", "tile", "kira"]
        let tiers: [String?] = [nil, "neutral", "apple", "banana", "avocado"]
        let kinds: [String?] = [nil, "image", "video"]
        let ceilings: [String?] = [nil, "neutral", "avocado"]

        for t in texts {
            for l in lanes {
                for tier in tiers {
                    for k in kinds {
                        for c in ceilings {
                            let q = CatalogQuery(scope: .kira, ceiling: c, text: t,
                                                 lane: l, tier: tier, kind: k, limit: 500)
                            let rows = try await store.search(q)
                            XCTAssertTrue(rows.allSatisfy { $0.realm == .kira },
                                          "leak: text=\(t ?? "-") lane=\(l ?? "-") tier=\(tier ?? "-") kind=\(k ?? "-") ceiling=\(c ?? "-")")
                        }
                    }
                }
            }
        }
    }

    /// Above the ceiling: label survives, text and path do not — at every tier.
    func testModeClampAtEveryCeiling() async throws {
        for (i, tier) in ["neutral", "apple", "banana", "avocado"].enumerated() {
            try await store.upsert(CatalogAsset(
                id: "a\(i)", filename: "a\(i)", absolutePath: "/tmp/a\(i)",
                realm: .kira, prompt: "text \(i)", contentMode: tier),
                explicitCollectionIDs: [])
        }
        for ceiling in ["neutral", "apple", "banana", "avocado"] {
            let rows = try await store.search(CatalogQuery(scope: .kira, ceiling: ceiling, limit: 100))
            XCTAssertEqual(rows.count, 4, "counts are metadata and always surface")
            for row in rows {
                if tierRank(row.contentMode) > tierRank(ceiling) {
                    XCTAssertNil(row.prompt, "\(row.contentMode!) leaked text at ceiling \(ceiling)")
                    XCTAssertEqual(row.absolutePath, "")
                    XCTAssertNotNil(row.contentMode, "the tier label must survive")
                } else {
                    XCTAssertNotNil(row.prompt)
                }
            }
        }
    }

    func testPermissionsAreTightenedOnOpen() async throws {
        let attrs = try FileManager.default.attributesOfItem(atPath: path)
        let perms = (attrs[.posixPermissions] as? NSNumber)?.intValue ?? 0
        XCTAssertEqual(perms & 0o077, 0, "catalog must not be group- or world-readable")
    }
}
```

- [ ] **Step 2: Run the whole Swift suite**

```bash
swift test --filter ComfyBoxCatalogTests
swift test --filter CatalogBrowserTests
```

Expected: all pass. The combination test in Step 1 runs 900 queries; if it is slow, that is fine — it is the guarantee everything else depends on.

- [ ] **Step 3: Verify the vault was never touched**

The vault is out of scope; this proves it rather than assuming it.

```bash
ssh todd@10.0.100.232 'bash -c "find ~/Documents/Vaults/BarkadaAI -newermt \"-1 day\" -type f | head"'
grep -rn "Vaults" Sources/ComfyBoxCatalog/ Sources/ComfyBoxGallery/ || echo "NO VAULT REFERENCES IN CATALOG CODE"
```

Expected: no recently-modified vault files attributable to this work, and `NO VAULT REFERENCES IN CATALOG CODE` (the only permitted mention is the refusal guard in `runBackfillCLI`, which greps as `contains("Vaults")` — if that is the sole hit, that is correct).

- [ ] **Step 4: Verify the engine was never disturbed**

```bash
shasum -a 256 .build/release/ComfyBox
launchctl print gui/$(id -u)/com.barkadabrew.comfybox | grep -E 'state|pid'
curl -s -o /dev/null -w 'engine %{http_code}\n' http://127.0.0.1:7870/v1/gallery/list?limit=1
curl -s -o /dev/null -w 'gallery %{http_code}\n' http://127.0.0.1:7871/healthz
```

Expected: the checkpoint hash `7422710b…`, the engine `running` with an unchanged pid, and both services answering `200`.

- [ ] **Step 5: Prove the catalog is rebuildable from the files**

```bash
cp ~/.comfybox/dam.sqlite3 /tmp/catalog-before-rebuild.sqlite3
sqlite3 /tmp/catalog-before-rebuild.sqlite3 \
  "SELECT COUNT(*), COUNT(DISTINCT realm) FROM assets;" > /tmp/rebuild-before.txt

rm -f /tmp/catalog-rebuilt.sqlite3
.build/release/ComfyBoxGallery backfill --db /tmp/catalog-rebuilt.sqlite3
sqlite3 /tmp/catalog-rebuilt.sqlite3 "SELECT COUNT(*), COUNT(DISTINCT realm) FROM assets;" > /tmp/rebuild-after.txt
diff /tmp/rebuild-before.txt /tmp/rebuild-after.txt && echo "REBUILDABLE"
```

Expected: `REBUILDABLE`. A difference here means some facts exist only in the catalog and nowhere on disk — for **video** that is expected until Plan B adds an embedded carrier, so check whether the delta is video-only:

```bash
sqlite3 /tmp/catalog-rebuilt.sqlite3 "SELECT kind, COUNT(*) FROM assets GROUP BY kind;"
```

Record the result in the runbook either way.

- [ ] **Step 6: Deploy the server side**

```bash
cd ~/Projects/coffeeshop-server
git push origin HEAD
ssh todd@10.0.100.232 'bash -c "cd ~/coffeeshop-server && scripts/local-merge.sh <branch>"'
```

Wait for the gate (~720 tests). Then, taking the coordination lock first:

```bash
ssh todd@10.0.100.232 'bash -c "cat ~/.kira/coordination/README.md"'
ssh todd@10.0.100.232 'bash -c "echo \"catalog client + gallery tools\" > ~/.kira/coordination/kira-daemon-restart.lock"'
ssh todd@10.0.100.232 'bash -c "cd ~/coffeeshop-server && scripts/kira-update.sh"'
ssh todd@10.0.100.232 'bash -c "rm -f ~/.kira/coordination/kira-daemon-restart.lock"'
```

**Note:** restarting the Kira daemon orphans any in-flight ComfyBox GPU job. Check for one first (`curl -s http://127.0.0.1:7870/v1/queue | head`) and wait for it to finish if the queue is busy.

- [ ] **Step 7: Write the runbook**

Create `docs/methods/gallery-catalog-runbook.md`:

```markdown
# Gallery catalog — runbook

## What runs where

- `com.barkadabrew.comfybox-gallery` → `.build/release/ComfyBoxGallery --port 7871`,
  loopback only. Owns `~/.comfybox/dam.sqlite3` (0600 in a 0700 directory).
- `com.barkadabrew.comfybox` → the GPU engine on 7870. **Unrelated lifecycle.**
  Restarting one never requires restarting the other.

## Health

    curl -s http://127.0.0.1:7871/healthz          # {"ok":true}
    tail -20 ~/.comfybox/gallery.err.log

## Rebuild the catalog from the files

    .build/release/ComfyBoxGallery backfill

Idempotent and re-runnable. Safe at any time; it only writes the catalog.

## Rollback

1. Server side: `git revert` the daemon commit, `scripts/kira-update.sh`
   (take `~/.kira/coordination/kira-daemon-restart.lock` first).
2. Gallery service: `launchctl bootout gui/$(id -u)/com.barkadabrew.comfybox-gallery`.
   Nothing else depends on it; the desktop falls back to `DAMStore`.
3. Catalog contents: restore `~/.comfybox/dam.sqlite3.bak-precatalog-*`.

The engine binary is never modified by this work, so there is no engine rollback.

## Known limits (Plan B closes these)

- Renders are indexed by the periodic backfill, not reported at render time,
  so a brand-new asset appears on the next sweep rather than immediately.
- `.mp4` files carry no embedded metadata, so a video whose sidecar is lost is
  unattributable. Plan B writes an XMP block into the container.
- Ratings and tags set in the desktop are not yet written back to XMP/Finder.
```

- [ ] **Step 8: Commit**

```bash
git add Tests/ComfyBoxCatalogTests/InvariantTests.swift docs/methods/gallery-catalog-runbook.md
git commit -m "test(catalog): realm-lock and mode-clamp invariants; add operational runbook"
```

---

## Spec coverage

| spec requirement | task |
|---|---|
| Additive schema, legacy columns untouched | 2 |
| `realm` = kira \| shared, kira the only exception | 1, 2, 5, 12 |
| Realm lock service-side, not client-supplied | 5, 7, 9, 12 |
| Mode clamp on retrieval | 5, 12 |
| Mode clamp re-checked at share time | 9 |
| Sealed rows carry no text, no FTS entry | 1, 5, 6 |
| Collections: two levels, many-to-many | 2, 3, 5 |
| Derived filing from lane/source, manual overrides | 3, 5 |
| Collections span realms, rows do not | 5, 7 |
| Kira curates her realm, not shared collections | 5, 9 |
| Her genres seeded from her lanes | 2, 3 |
| Video: sidecar-only metadata, container-probed duration | 4, 6 |
| `i2v_source` edges from `source_image` | 6 |
| `member_of` edges for composed video | 5 (table + API); backfill of scene members is Plan B |
| One asset, many locations, sha256 dedup | 5, 6 |
| exiftool backfill over three trees | 6, 8 |
| Rebuildable from the files | 6, 12 |
| Query API + facets + collections tree | 7 |
| Four consumers bound | 9, 10, 11 |
| Desktop is the display surface, one gallery | 11 |
| Catalog 0600 in a 0700 directory | 5, 8, 12 |
| Vault untouched | 6, 8, 12 |
| Engine never rebuilt or restarted | 1, 8, 12 |

**Deliberately deferred to Plan B**, and named here so it is not mistaken for an omission: render-time reporting from the engine, `sealed` embed gating, metadata write-back to XMP/Finder, the XMP block for `.mp4` containers, vision captions for rows with no text, and backfill of `member_of` edges for scene/montage composition (the table, API and tests exist; only the scene-manifest reader is deferred, because scene composition metadata lives in the storyboard/montage paths rather than in the sidecar).
