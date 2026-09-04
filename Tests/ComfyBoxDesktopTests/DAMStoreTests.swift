// DAMStoreTests.swift — Tests for DAMStore actor

import Testing
import Foundation
import SQLite3
import ComfyBoxCatalog
@testable import ComfyBoxDesktop

@Suite("DAMStore")
struct DAMStoreTests {
    @Test("opens database successfully")
    func openDatabase() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let count = try await store.assetCount()
        #expect(count == 0)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("insert and fetch asset")
    func insertAndFetch() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "test-1", filename: "test.png")
        try await store.insertAsset(asset)
        let count = try await store.assetCount()
        #expect(count == 1)
        let fetched = try await store.fetchAssets(limit: 10)
        #expect(fetched.count == 1)
        #expect(fetched[0].id == "test-1")
        #expect(fetched[0].filename == "test.png")
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("insert multiple assets and fetch in order")
    func multipleAssets() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        for i in 0..<5 {
            let asset = DAMAsset(
                id: "asset-\(i)", filename: "img-\(i).png",
                absolutePath: "/tmp/img-\(i).png",
                createdAt: Date(timeIntervalSince1970: Double(1_700_000_000 + i * 100))
            )
            try await store.insertAsset(asset)
        }
        let count = try await store.assetCount()
        #expect(count == 5)
        let fetched = try await store.fetchAssets(limit: 10)
        #expect(fetched.count == 5)
        #expect(fetched[0].id == "asset-4") // Newest first
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("allAssetPaths returns inserted paths")
    func allPaths() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "path-test", filename: "path-test.png")
        try await store.insertAsset(asset)
        let paths = try await store.allAssetPaths()
        #expect(paths.contains(asset.absolutePath))
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("full-text search matches prompt")
    func ftsSearch() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "fts-test", filename: "fts.png", prompt: "a golden sunset over the ocean")
        try await store.insertAsset(asset)
        let results = try await store.searchPrompts(query: "sunset")
        #expect(!results.isEmpty)
        #expect(results[0].id == "fts-test")
        let noResults = try await store.searchPrompts(query: "xylophone")
        #expect(noResults.isEmpty)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("fetch respects limit and offset")
    func limitOffset() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        for i in 0..<10 {
            let asset = DAMAsset(
                id: "lo-\(i)", filename: "lo-\(i).png",
                absolutePath: "/tmp/lo-\(i).png",
                createdAt: Date(timeIntervalSince1970: Double(1_700_000_000 + i * 100))
            )
            try await store.insertAsset(asset)
        }
        let page1 = try await store.fetchAssets(limit: 3, offset: 0)
        #expect(page1.count == 3)
        #expect(page1[0].id == "lo-9")
        let page2 = try await store.fetchAssets(limit: 3, offset: 3)
        #expect(page2.count == 3)
        #expect(page2[0].id == "lo-6")
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("insert or replace updates existing asset")
    func insertOrReplace() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "upsert-1", rating: 0)
        try await store.insertAsset(asset)
        let updated = DAMAsset(id: "upsert-1", filename: asset.filename, absolutePath: asset.absolutePath, rating: 5)
        try await store.insertAsset(updated)
        let count = try await store.assetCount()
        #expect(count == 1)
        let fetched = try await store.fetchAssets()
        #expect(fetched[0].rating == 5)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("FTS index does not accumulate duplicate rows on repeated updates")
    func ftsNoDuplicates() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "fts-dup", filename: "dup.png", prompt: "cerulean dragonfly over water")
        try await store.insertAsset(asset)
        // Simulate repeated rating/favorite updates via insertAsset.
        for rating in 1...3 {
            let updated = TestData.makeAsset(
                id: "fts-dup", filename: "dup.png",
                prompt: "cerulean dragonfly over water", rating: rating
            )
            try await store.insertAsset(updated)
        }
        let results = try await store.searchPrompts(query: "dragonfly")
        #expect(results.count == 1)
        #expect(results[0].id == "fts-dup")
        #expect(results[0].rating == 3)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("re-ingesting a path preserves id, rating, and favorite")
    func reingestPreservesAnnotations() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let original = TestData.makeAsset(
            id: "original-id", filename: "keeper.png",
            prompt: "an annotated masterpiece", rating: 4, favorite: true
        )
        try await store.insertAsset(original)

        // Simulate a re-ingest: fresh UUID, same absolute path, no metadata.
        let reingested = DAMAsset(
            filename: "keeper.png",
            absolutePath: original.absolutePath,
            fileSize: 999
        )
        let stored = try await store.insertAsset(reingested)
        #expect(stored.id == "original-id")

        let fetched = try await store.fetchAssets()
        #expect(fetched.count == 1)
        #expect(fetched[0].id == "original-id")
        #expect(fetched[0].rating == 4)
        #expect(fetched[0].favorite)
        // File metadata refreshed, generation metadata preserved.
        #expect(fetched[0].fileSize == 999)
        #expect(fetched[0].prompt == "an annotated masterpiece")
        // FTS still resolves to the single surviving row.
        let results = try await store.searchPrompts(query: "masterpiece")
        #expect(results.count == 1)
        #expect(results[0].id == "original-id")
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("re-ingesting a path with nil source preserves the existing row's source")
    func reingestPreservesSource() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let original = DAMAsset(
            id: "original-id", filename: "sourced.png",
            absolutePath: "/tmp/test-images/sourced.png",
            source: "kira"
        )
        try await store.insertAsset(original)

        // Colliding path, fresh id, no source — as a poller re-ingest would produce.
        let reingested = DAMAsset(
            filename: "sourced.png",
            absolutePath: original.absolutePath,
            fileSize: 999
        )
        let stored = try await store.insertAsset(reingested)
        #expect(stored.id == "original-id")
        #expect(stored.source == "kira")

        let fetched = try await store.fetchAssets()
        #expect(fetched.count == 1)
        #expect(fetched[0].source == "kira")
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("fetchAsset(byPath:) returns tracked asset or nil")
    func fetchByPath() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "by-path", filename: "by-path.png")
        try await store.insertAsset(asset)
        let found = try await store.fetchAsset(byPath: asset.absolutePath)
        #expect(found?.id == "by-path")
        let missing = try await store.fetchAsset(byPath: "/tmp/does-not-exist.png")
        #expect(missing == nil)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("deleteAsset removes the row and its FTS entry")
    func deleteAsset() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "del-1", filename: "del.png", prompt: "obsidian tower at dusk")
        try await store.insertAsset(asset)
        try await store.deleteAsset(id: "del-1")
        let count = try await store.assetCount()
        #expect(count == 0)
        let ftsResults = try await store.searchPrompts(query: "obsidian")
        #expect(ftsResults.isEmpty)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("deleteAsset of unknown id is a no-op")
    func deleteUnknownAsset() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "keep-1", filename: "keep.png")
        try await store.insertAsset(asset)
        try await store.deleteAsset(id: "no-such-id")
        let count = try await store.assetCount()
        #expect(count == 1)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("deleteAssets removes only the given ids")
    func deleteMultipleAssets() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        for i in 0..<5 {
            let asset = DAMAsset(
                id: "bulk-\(i)", filename: "bulk-\(i).png",
                absolutePath: "/tmp/bulk-\(i).png",
                prompt: "bulk prompt \(i)"
            )
            try await store.insertAsset(asset)
        }
        try await store.deleteAssets(ids: ["bulk-1", "bulk-3"])
        let remaining = try await store.fetchAssets(limit: 10)
        let ids = Set(remaining.map(\.id))
        #expect(ids == ["bulk-0", "bulk-2", "bulk-4"])
        // FTS rows for deleted assets are gone too.
        let fts1 = try await store.searchPrompts(query: "\"bulk prompt 1\"")
        #expect(fts1.isEmpty)
        let fts2 = try await store.searchPrompts(query: "\"bulk prompt 2\"")
        #expect(fts2.count == 1)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("folders: create, list, rename, delete")
    func folderCRUD() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)

        let folder = try await store.createFolder(name: "Portraits")
        #expect(!folder.id.isEmpty)
        #expect(folder.name == "Portraits")

        _ = try await store.createFolder(name: "Landscapes")
        var folders = try await store.listFolders()
        #expect(folders.map(\.name).sorted() == ["Landscapes", "Portraits"])

        try await store.renameFolder(id: folder.id, name: "People")
        folders = try await store.listFolders()
        #expect(folders.map(\.name).sorted() == ["Landscapes", "People"])

        try await store.deleteFolder(id: folder.id)
        folders = try await store.listFolders()
        #expect(folders.map(\.name) == ["Landscapes"])
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("folders: assign, reassign, and unfile assets")
    func folderAssignment() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        for i in 0..<3 {
            try await store.insertAsset(TestData.makeAsset(id: "a\(i)", filename: "a\(i).png"))
        }
        let folder = try await store.createFolder(name: "Picks")

        try await store.assignAssets(ids: ["a0", "a1"], toFolder: folder.id)
        var assignments = try await store.folderAssignments()
        #expect(assignments == ["a0": folder.id, "a1": folder.id])

        let counts = try await store.folderCounts()
        #expect(counts[folder.id] == 2)

        try await store.assignAssets(ids: ["a0"], toFolder: nil)
        assignments = try await store.folderAssignments()
        #expect(assignments == ["a1": folder.id])
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("folders: deleting a folder unfiles assets; deleting an asset drops its mapping")
    func folderCleanup() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        try await store.insertAsset(TestData.makeAsset(id: "x1", filename: "x1.png"))
        try await store.insertAsset(TestData.makeAsset(id: "x2", filename: "x2.png"))
        let folder = try await store.createFolder(name: "Temp")
        try await store.assignAssets(ids: ["x1", "x2"], toFolder: folder.id)

        try await store.deleteAsset(id: "x1")
        var assignments = try await store.folderAssignments()
        #expect(assignments == ["x2": folder.id])

        try await store.deleteFolder(id: folder.id)
        assignments = try await store.folderAssignments()
        #expect(assignments.isEmpty)
        // Assets themselves survive folder deletion.
        let count = try await store.assetCount()
        #expect(count == 1)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("folders: membership survives asset re-ingest (INSERT OR REPLACE)")
    func folderSurvivesReingest() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(id: "r1", filename: "r1.png")
        try await store.insertAsset(asset)
        let folder = try await store.createFolder(name: "Keep")
        try await store.assignAssets(ids: ["r1"], toFolder: folder.id)

        try await store.insertAsset(asset)  // re-ingest same asset
        let assignments = try await store.folderAssignments()
        #expect(assignments["r1"] == folder.id)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("pruneOrphans removes rows whose file is gone, keeps present files")
    func pruneOrphans() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)

        // One real file, two rows pointing at nonexistent files.
        let realPath = (tmpDir as NSString).appendingPathComponent("real-\(UUID().uuidString).png")
        try Data([0x89, 0x50]).write(to: URL(fileURLWithPath: realPath))
        try await store.insertAsset(DAMAsset(id: "real", filename: "real.png", absolutePath: realPath, prompt: "kept"))
        try await store.insertAsset(DAMAsset(id: "ghost1", filename: "g1.png", absolutePath: "/nope/g1.png", prompt: "gone one"))
        try await store.insertAsset(DAMAsset(id: "ghost2", filename: "g2.png", absolutePath: "/nope/g2.png"))
        // A ghost that's filed into a folder — mapping must go too.
        let folder = try await store.createFolder(name: "F")
        try await store.assignAssets(ids: ["ghost1"], toFolder: folder.id)

        let removed = try await store.pruneOrphans()
        #expect(Set(removed) == ["ghost1", "ghost2"])

        let remaining = try await store.fetchAssets(limit: 10)
        #expect(remaining.map(\.id) == ["real"])
        // FTS and folder mapping for the pruned rows are gone.
        #expect(try await store.searchPrompts(query: "gone").isEmpty)
        #expect(try await store.folderAssignments().isEmpty)

        try? FileManager.default.removeItem(atPath: realPath)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("pruneOrphans keeps secured assets even though their path is in the vault")
    func pruneKeepsSecured() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let vaultPath = (tmpDir as NSString).appendingPathComponent("vault-\(UUID().uuidString).png")
        try Data([0x89]).write(to: URL(fileURLWithPath: vaultPath))
        try await store.insertAsset(DAMAsset(id: "sec", filename: "sec.png", absolutePath: vaultPath))
        try await store.secureAsset(id: "sec", securedPath: vaultPath, originalPath: "/tmp/orig.png")

        let removed = try await store.pruneOrphans()
        #expect(removed.isEmpty)
        #expect(try await store.assetCount() == 1)
        try? FileManager.default.removeItem(atPath: vaultPath)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    /// The catalog migrated INTO dam.sqlite3, so this one table now also holds
    /// rows for Kira's and Bree's server trees — roughly 1,300 of the 2,994 rows
    /// in the live database name a path that exists only on a server. For those,
    /// "the file isn't here" is the normal case, not evidence of an orphan, and
    /// the gallery calls pruneOrphans on every unfiltered load.
    @Test("pruneOrphans keeps rows whose bytes live on another host")
    func pruneKeepsAssetsHostedElsewhere() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")

        // Same file, both stores — exactly the production arrangement.
        let store = try await DAMStore.open(path: dbPath)
        try await store.insertAsset(DAMAsset(id: "onserver", filename: "s.png",
                                             absolutePath: "/home/todd/.kira/studio/gallery/s.png"))
        try await store.insertAsset(DAMAsset(id: "reallygone", filename: "g.png",
                                             absolutePath: "/nope/g.png"))
        let catalog = try await CatalogStore.open(path: dbPath)
        try await catalog.addLocation(assetID: "onserver",
            AssetLocation(host: "kira", path: "/home/todd/.kira/studio/gallery/s.png", mtime: Date()))

        let removed = try await store.pruneOrphans()
        #expect(removed == ["reallygone"])
        #expect(try await store.fetchAssets(limit: 10).map(\.id) == ["onserver"])

        try? FileManager.default.removeItem(atPath: dbPath)
    }

    /// A mass disappearance is almost always an unmounted volume or a revoked
    /// folder permission, not deletions — and this sweep runs unattended on
    /// every unfiltered gallery load. It must refuse rather than proceed.
    @Test("pruneOrphans refuses en-masse and deletes nothing")
    func pruneCircuitBreaker() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)

        // 100 rows, 6 of them missing — over both the 5% ceiling and the floor.
        for i in 0..<94 {
            let path = (tmpDir as NSString).appendingPathComponent("live-\(UUID().uuidString).png")
            try Data([0x89]).write(to: URL(fileURLWithPath: path))
            try await store.insertAsset(DAMAsset(id: "live\(i)", filename: "l.png", absolutePath: path))
        }
        for i in 0..<6 {
            try await store.insertAsset(DAMAsset(id: "gone\(i)", filename: "g.png",
                                                 absolutePath: "/nope/\(i).png"))
        }

        await #expect(throws: DAMStoreError.self) { try await store.pruneOrphans() }
        #expect(try await store.assetCount() == 100, "a refused sweep deletes nothing")
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    /// The live schema has no foreign keys, so nothing cascades on its own.
    @Test("deleteAsset also clears catalog collections, locations and edges")
    func deleteClearsCatalogTables() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        try await store.insertAsset(DAMAsset(id: "doomed", filename: "d.png", absolutePath: "/nope/d.png"))

        let catalog = try await CatalogStore.open(path: dbPath)
        try await catalog.file(assetID: "doomed", into: "col-photography", by: nil)
        try await catalog.addLocation(assetID: "doomed",
            AssetLocation(host: "kira", path: "/home/todd/d.png", mtime: Date()))
        #expect(try await catalog.locations(of: "doomed", scope: nil).count == 1)

        try await store.deleteAsset(id: "doomed")
        #expect(try await catalog.locations(of: "doomed", scope: nil).isEmpty)
        #expect(try await catalog.collectionCounts(scope: nil)["col-photography"] == 0)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("searchPrompts preserves source (regression: SELECT was missing a.source)")
    func searchPromptsPreservesSource() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = DAMAsset(
            id: "source-test", filename: "source-test.png",
            absolutePath: "/tmp/source-test.png",
            prompt: "a kira render of a sunlit meadow",
            source: "kira"
        )
        try await store.insertAsset(asset)
        let results = try await store.searchPrompts(query: "meadow")
        #expect(results.count == 1)
        #expect(results[0].source == "kira")
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("allAssetIds returns every id, including secured ones")
    func allAssetIds() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let securedPath = (tmpDir as NSString).appendingPathComponent("secured-\(UUID().uuidString).png")
        try Data([0x89]).write(to: URL(fileURLWithPath: securedPath))
        try await store.insertAsset(DAMAsset(id: "plain-1", filename: "plain.png", absolutePath: "/tmp/plain.png"))
        try await store.insertAsset(DAMAsset(id: "sec-1", filename: "sec.png", absolutePath: securedPath))
        try await store.secureAsset(id: "sec-1", securedPath: securedPath, originalPath: "/tmp/orig.png")

        let ids = try await store.allAssetIds()
        #expect(ids == ["plain-1", "sec-1"])
        try? FileManager.default.removeItem(atPath: securedPath)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("allAssetIds returns an empty set on a fresh DB")
    func allAssetIdsEmpty() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let ids = try await store.allAssetIds()
        #expect(ids.isEmpty)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    // MARK: - Targeted id-based queries (#265)
    //
    // `assets(withIDs:)`, `assetIDs(withIDs:)` and `assetLocations(limit:offset:)`
    // exist so a caller that already knows which handful of rows it needs
    // (a restore's live-row lookup, a crash-recovery sweep, a folder's
    // member set, a paged missing-file scan) never has to materialize the
    // whole `assets` table to filter it in memory — see #265.

    @Test("assets(withIDs:) returns exactly the requested rows, skipping ids that don't exist")
    func assetsByIDs() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        for i in 0..<5 {
            try await store.insertAsset(DAMAsset(id: "id-\(i)", filename: "f-\(i).png", absolutePath: "/tmp/f-\(i).png"))
        }
        let fetched = try await store.assets(withIDs: ["id-1", "id-3", "no-such-id"])
        #expect(Set(fetched.map(\.id)) == ["id-1", "id-3"])
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("assets(withIDs:) with an empty set returns no rows")
    func assetsByIDsEmpty() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        try await store.insertAsset(TestData.makeAsset(id: "solo"))
        let fetched = try await store.assets(withIDs: [])
        #expect(fetched.isEmpty)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("assets(withIDs:) chunks an IN (...) list over 500 ids and returns every match")
    func assetsByIDsChunking() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        var ids: Set<String> = []
        for i in 0..<600 {
            let id = "chunk-\(i)"
            ids.insert(id)
            try await store.insertAsset(DAMAsset(id: id, filename: "c-\(i).png", absolutePath: "/tmp/c-\(i).png"))
        }
        let fetched = try await store.assets(withIDs: ids)
        #expect(fetched.count == 600)
        #expect(Set(fetched.map(\.id)) == ids)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("assetIDs(withIDs:) returns only the ids that currently exist")
    func assetIDsByIDs() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        for i in 0..<5 {
            try await store.insertAsset(DAMAsset(id: "id-\(i)", filename: "f-\(i).png", absolutePath: "/tmp/f-\(i).png"))
        }
        let existing = try await store.assetIDs(withIDs: ["id-1", "id-3", "missing-id"])
        #expect(existing == ["id-1", "id-3"])
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("assetIDs(withIDs:) with an empty set returns an empty set")
    func assetIDsByIDsEmpty() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        try await store.insertAsset(TestData.makeAsset(id: "solo"))
        let existing = try await store.assetIDs(withIDs: [])
        #expect(existing.isEmpty)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("assetIDs(withIDs:) chunks an IN (...) list over 500 ids and finds every existing one")
    func assetIDsByIDsChunking() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        var ids: Set<String> = []
        for i in 0..<600 {
            let id = "chunk-\(i)"
            ids.insert(id)
            try await store.insertAsset(DAMAsset(id: id, filename: "c-\(i).png", absolutePath: "/tmp/c-\(i).png"))
        }
        let queried = ids.union(["definitely-not-there"])
        let existing = try await store.assetIDs(withIDs: queried)
        #expect(existing == ids)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("assetLocations pages through every row exactly once, in a stable order")
    func assetLocationsPaging() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        var expected: [String: String] = [:]
        for i in 0..<250 {
            let id = "loc-\(String(format: "%03d", i))"
            let path = "/tmp/loc-\(i).png"
            expected[id] = path
            try await store.insertAsset(DAMAsset(id: id, filename: "loc-\(i).png", absolutePath: path))
        }

        var seenIDs: [String] = []
        var seenPaths: [String: String] = [:]
        var offset = 0
        let pageSize = 37 // not a divisor of 250 — exercises a partial last page
        while true {
            let page = try await store.assetLocations(limit: pageSize, offset: offset)
            if page.isEmpty { break }
            #expect(page.count <= pageSize)
            for (id, path) in page { seenPaths[id] = path }
            seenIDs.append(contentsOf: page.map(\.id))
            offset += page.count
            if page.count < pageSize { break }
        }

        #expect(seenIDs.count == 250, "every row visited")
        #expect(Set(seenIDs).count == 250, "no row visited twice")
        #expect(seenIDs == seenIDs.sorted(), "pages walk in a stable order")
        #expect(seenPaths == expected)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("assetLocations on an empty store returns no rows on the first page")
    func assetLocationsEmpty() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let page = try await store.assetLocations(limit: 100, offset: 0)
        #expect(page.isEmpty)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("preserves all asset fields through insert and fetch")
    func allFields() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let asset = TestData.makeAsset(
            id: "fields-test", filename: "fields.png", prompt: "test prompt",
            seed: 42, steps: 20, guidance: 7.5, modelFamily: "sdxl",
            rating: 4, favorite: true, contentMode: "banana",
            characterName: "Alice", width: 768, height: 512
        )
        try await store.insertAsset(asset)
        let fetched = try await store.fetchAssets()
        let f = fetched[0]
        #expect(f.prompt == "test prompt")
        #expect(f.seed == 42)
        #expect(f.steps == 20)
        #expect(f.guidance == 7.5)
        #expect(f.modelFamily == "sdxl")
        #expect(f.rating == 4)
        #expect(f.favorite)
        #expect(f.contentMode == "banana")
        #expect(f.characterName == "Alice")
        #expect(f.width == 768)
        #expect(f.height == 512)
        try? FileManager.default.removeItem(atPath: dbPath)
    }
}

@Suite("DAMStoreError")
struct DAMStoreErrorTests {
    @Test("openFailed includes path and message")
    func openFailed() {
        let error = DAMStoreError.openFailed("/bad/path", "no such file")
        #expect(error.errorDescription?.contains("/bad/path") == true)
        #expect(error.errorDescription?.contains("no such file") == true)
    }

    @Test("prepareFailed includes message")
    func prepareFailed() {
        let error = DAMStoreError.prepareFailed("syntax error")
        #expect(error.errorDescription?.contains("syntax error") == true)
    }

    @Test("insertFailed includes message")
    func insertFailed() {
        let error = DAMStoreError.insertFailed("constraint violation")
        #expect(error.errorDescription?.contains("constraint violation") == true)
    }

    @Test("execFailed includes SQL and message")
    func execFailed() {
        let error = DAMStoreError.execFailed("CREATE TABLE", "table exists")
        #expect(error.errorDescription?.contains("CREATE TABLE") == true)
        #expect(error.errorDescription?.contains("table exists") == true)
    }

    @Test("stepFailed includes the sqlite code and message")
    func stepFailed() {
        let error = DAMStoreError.stepFailed(SQLITE_IOERR, "disk I/O error")
        #expect(error.errorDescription?.contains("\(SQLITE_IOERR)") == true)
        #expect(error.errorDescription?.contains("disk I/O error") == true)
    }
}

/// #263: `while sqlite3_step(stmt) == SQLITE_ROW` cannot distinguish
/// "the result set is finished" (`SQLITE_DONE`) from "the step itself
/// failed partway through" (`SQLITE_BUSY`, `SQLITE_IOERR`,
/// `SQLITE_CORRUPT`, …) — both fall out of the loop identically, silently
/// truncating the result. `allAssetIds()`/`allAssetPaths()` and their
/// destructive-chain neighbours (`securedAssetIds()`,
/// `fetchAssets()`, `pruneOrphans()`'s `assetIDsHostedElsewhere()`) now
/// route through `drainSQLiteRows`, which returns the terminal step code
/// instead of swallowing it. Fault-injecting a real mid-iteration SQLite
/// error into an actor-encapsulated connection is impractical to do
/// deterministically, so this pins the shared draining primitive directly
/// with a stubbed `step` closure, per the fallback the ticket allows.
@Suite("drainSQLiteRows")
struct DrainSQLiteRowsTests {
    @Test("walks every row and returns SQLITE_DONE on a clean finish")
    func cleanFinish() {
        var remaining = [SQLITE_ROW, SQLITE_ROW, SQLITE_ROW, SQLITE_DONE]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        #expect(rc == SQLITE_DONE)
        #expect(seen == 3)
    }

    @Test("stops on a mid-iteration error and reports the failing code, not SQLITE_DONE")
    func midIterationErrorStopsAndReportsCode() {
        // Two good rows, then the step call itself fails — the historical
        // bug treated this identically to "no more rows".
        var remaining = [SQLITE_ROW, SQLITE_ROW, SQLITE_IOERR]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        #expect(rc == SQLITE_IOERR)
        #expect(rc != SQLITE_DONE)
        #expect(seen == 2, "onRow must not fire for the failing step")
    }

    @Test("reports SQLITE_BUSY too — any non-ROW, non-DONE code must not look like end-of-set")
    func busyIsNotTreatedAsDone() {
        var remaining = [SQLITE_ROW, SQLITE_BUSY]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        #expect(rc == SQLITE_BUSY)
        #expect(seen == 1)
    }

    @Test("an empty result set still reports SQLITE_DONE, not an error")
    func emptyResultIsNotAnError() {
        var remaining = [SQLITE_DONE]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        #expect(rc == SQLITE_DONE)
        #expect(seen == 0)
    }
}
