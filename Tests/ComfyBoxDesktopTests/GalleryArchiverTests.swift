// GalleryArchiverTests.swift — Tests for GalleryArchiver.archive (T3),
// GalleryArchiver.restore (T4), and GalleryArchiver.resumePendingRemovals (T7).
//
// Archive: bundle well-formedness, sidecar/folder/secured handling, and
// crash-safety (nothing destroyed before the commit point).
// Restore: full round-trip field equality, the four id-collision cases,
// filename-collision renaming, folder re-creation, partial restore,
// idempotence, and progress reporting.
// resumePendingRemovals: finishes an interrupted removal (crash between
// steps 8 and 10 of §5.1), is idempotent, and leaves incomplete bundles
// (which never committed) entirely untouched.

import Testing
import Foundation
import SQLite3
@testable import ComfyBoxDesktop

@Suite("GalleryArchiver")
struct GalleryArchiverTests {
    /// Create an isolated temp environment: watch dir, thumbnail dir,
    /// archive root, and store — mirrors AssetIngestorTests.makeEnvironment().
    @MainActor
    private func makeEnvironment() async throws -> (
        archiver: GalleryArchiver,
        ingestor: AssetIngestor,
        store: DAMStore,
        watchDir: String,
        thumbDir: String,
        archiveRoot: String,
        dbPath: String
    ) {
        let base = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("archiver-test-\(UUID().uuidString)")
        let watchDir = (base as NSString).appendingPathComponent("output")
        let thumbDir = (base as NSString).appendingPathComponent("thumbnails")
        let archiveRoot = (base as NSString).appendingPathComponent("archives")
        try FileManager.default.createDirectory(atPath: watchDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: thumbDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: archiveRoot, withIntermediateDirectories: true)
        let dbPath = (base as NSString).appendingPathComponent("dam.sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let ingestor = AssetIngestor(store: store, watchDirectory: watchDir, thumbnailDirectory: thumbDir)
        let archiver = GalleryArchiver(store: store, ingestor: ingestor)
        return (archiver, ingestor, store, watchDir, thumbDir, archiveRoot, dbPath)
    }

    /// Write a minimal file with the given name into the directory.
    @discardableResult
    private func writeFile(_ name: String, in dir: String, contents: String = "not-a-real-png") -> String {
        let path = (dir as NSString).appendingPathComponent(name)
        FileManager.default.createFile(atPath: path, contents: Data(contents.utf8))
        return path
    }

    /// Simulate a generated thumbnail (real thumbnail generation needs a real image).
    @MainActor
    private func makeThumbnail(for id: String, ingestor: AssetIngestor) {
        let path = ingestor.thumbnailPath(for: id)
        FileManager.default.createFile(atPath: path, contents: Data("thumb".utf8))
    }

    // MARK: - Bundle well-formed + DB emptied + thumbnails gone

    @Test("archive copies assets into a well-formed bundle, empties the DB, and clears thumbnails")
    @MainActor
    func archiveWellFormedBundle() async throws {
        let env = try await makeEnvironment()
        var assets: [DAMAsset] = []
        for i in 1...3 {
            let path = writeFile("render\(i).png", in: env.watchDir, contents: "image-\(i)")
            let stored = try await env.ingestor.ingestFile(at: path)
            makeThumbnail(for: stored.id, ingestor: env.ingestor)
            assets.append(stored)
        }

        let request = GalleryArchiver.ArchiveRequest(
            name: "My Archive", destinationRoot: env.archiveRoot, assets: assets
        )
        let result = try await env.archiver.archive(request)

        let fm = FileManager.default
        #expect(fm.fileExists(atPath: result.bundlePath))
        #expect(result.bundlePath.hasSuffix(".cbarchive"))
        #expect(result.archived == 3)
        #expect(result.skippedSecured == 0)
        #expect(result.failed.isEmpty)

        let manifestPath = (result.bundlePath as NSString).appendingPathComponent("manifest.json")
        let entriesPath = (result.bundlePath as NSString).appendingPathComponent("entries.jsonl")
        let incompletePath = (result.bundlePath as NSString).appendingPathComponent("INCOMPLETE.json")
        #expect(fm.fileExists(atPath: manifestPath))
        #expect(fm.fileExists(atPath: entriesPath))
        #expect(!fm.fileExists(atPath: incompletePath))

        let manifest = try ArchiveManifest.decode(Data(contentsOf: URL(fileURLWithPath: manifestPath)))
        #expect(manifest.assetCount == 3)

        for asset in assets {
            let assetPath = (result.bundlePath as NSString)
                .appendingPathComponent("assets/\(asset.id)/\(asset.filename)")
            #expect(fm.fileExists(atPath: assetPath))
            #expect(!fm.fileExists(atPath: env.ingestor.thumbnailPath(for: asset.id)))
        }

        let dbCount = try await env.store.assetCount()
        #expect(dbCount == 0)
        #expect(env.archiver.phase == .idle)
        #expect(!env.archiver.isRunning)
    }

    // MARK: - Sidecar copied

    @Test("sidecar JSON is copied into the bundle alongside the asset")
    @MainActor
    func sidecarCopied() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("render.png", in: env.watchDir)
        writeFile("render.json", in: env.watchDir, contents: #"{"prompt":"test"}"#)
        let stored = try await env.ingestor.ingestFile(at: imagePath)

        let request = GalleryArchiver.ArchiveRequest(
            name: "Sidecar Test", destinationRoot: env.archiveRoot, assets: [stored]
        )
        let result = try await env.archiver.archive(request)

        let sidecarPath = (result.bundlePath as NSString)
            .appendingPathComponent("assets/\(stored.id)/render.json")
        #expect(FileManager.default.fileExists(atPath: sidecarPath))
    }

    // MARK: - Folder membership + manifest.folders

    @Test("archived entries carry folderId and manifest.folders lists the folder")
    @MainActor
    func folderMembershipRecorded() async throws {
        let env = try await makeEnvironment()
        let path1 = writeFile("pick1.png", in: env.watchDir, contents: "one")
        let path2 = writeFile("pick2.png", in: env.watchDir, contents: "two")
        let asset1 = try await env.ingestor.ingestFile(at: path1)
        let asset2 = try await env.ingestor.ingestFile(at: path2)

        let folder = try await env.store.createFolder(name: "Picks")
        try await env.store.assignAssets(ids: [asset1.id, asset2.id], toFolder: folder.id)

        let request = GalleryArchiver.ArchiveRequest(
            name: "Folder Test", destinationRoot: env.archiveRoot, assets: [asset1, asset2]
        )
        let result = try await env.archiver.archive(request)

        let entriesPath = (result.bundlePath as NSString).appendingPathComponent("entries.jsonl")
        var seenFolderIds: [String: String?] = [:]
        _ = try ArchiveJSONL.read(at: entriesPath) { entry in
            seenFolderIds[entry.id] = entry.folderId
        }
        #expect(seenFolderIds[asset1.id] == folder.id)
        #expect(seenFolderIds[asset2.id] == folder.id)

        let manifestPath = (result.bundlePath as NSString).appendingPathComponent("manifest.json")
        let manifest = try ArchiveManifest.decode(Data(contentsOf: URL(fileURLWithPath: manifestPath)))
        #expect(manifest.folders.count == 1)
        #expect(manifest.folders.first?.name == "Picks")
    }

    // MARK: - Folder deletion

    @Test("ArchiveRequest.folder is deleted once all its assets are archived")
    @MainActor
    func folderDeletedOnFullArchive() async throws {
        let env = try await makeEnvironment()
        let path1 = writeFile("a.png", in: env.watchDir, contents: "one")
        let path2 = writeFile("b.png", in: env.watchDir, contents: "two")
        let asset1 = try await env.ingestor.ingestFile(at: path1)
        let asset2 = try await env.ingestor.ingestFile(at: path2)

        let folder = try await env.store.createFolder(name: "ToArchive")
        try await env.store.assignAssets(ids: [asset1.id, asset2.id], toFolder: folder.id)

        let request = GalleryArchiver.ArchiveRequest(
            name: "Folder Delete Test", destinationRoot: env.archiveRoot,
            assets: [asset1, asset2], folder: folder
        )
        _ = try await env.archiver.archive(request)

        let folders = try await env.store.listFolders()
        #expect(!folders.contains(where: { $0.id == folder.id }))
    }

    // MARK: - Secured asset excluded

    @Test("secured assets are excluded from the archive and their row survives")
    @MainActor
    func securedAssetExcluded() async throws {
        let env = try await makeEnvironment()
        let pathA = writeFile("visible.png", in: env.watchDir, contents: "visible")
        let pathB = writeFile("secret.png", in: env.watchDir, contents: "secret")
        let assetA = try await env.ingestor.ingestFile(at: pathA)
        let assetB = try await env.ingestor.ingestFile(at: pathB)

        try await env.store.secureAsset(
            id: assetB.id, securedPath: "/secure/secret.png", originalPath: assetB.absolutePath
        )

        let request = GalleryArchiver.ArchiveRequest(
            name: "Secured Test", destinationRoot: env.archiveRoot, assets: [assetA, assetB]
        )
        let result = try await env.archiver.archive(request)

        #expect(result.skippedSecured == 1)
        #expect(result.archived == 1)

        let remainingIds = try await env.store.allAssetIds()
        #expect(remainingIds.contains(assetB.id))
        #expect(!remainingIds.contains(assetA.id))
    }

    // MARK: - Failure leaves everything intact

    @Test("a failed archive (unwritable destination) leaves the DB and source files untouched")
    @MainActor
    func failureLeavesEverythingIntact() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir)
        let stored = try await env.ingestor.ingestFile(at: path)

        let unwritableRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("unwritable-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: unwritableRoot, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: unwritableRoot)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: unwritableRoot)
            try? FileManager.default.removeItem(atPath: unwritableRoot)
        }

        let countBefore = try await env.store.assetCount()

        let request = GalleryArchiver.ArchiveRequest(
            name: "Doomed", destinationRoot: unwritableRoot, assets: [stored]
        )
        await #expect(throws: (any Error).self) {
            try await env.archiver.archive(request)
        }

        let countAfter = try await env.store.assetCount()
        #expect(countAfter == countBefore)
        #expect(FileManager.default.fileExists(atPath: stored.absolutePath))
        #expect(env.archiver.lastError != nil)
        #expect(!env.archiver.isRunning)
    }

    // MARK: - Restore: full round-trip

    @Test("restore full round-trip: every field matches and the id is preserved when free")
    @MainActor
    func restoreFullRoundTrip() async throws {
        let env = try await makeEnvironment()
        var assets: [DAMAsset] = []
        for i in 1...3 {
            let contents = "image-\(i)"
            let path = writeFile("render\(i).png", in: env.watchDir, contents: contents)
            let asset = DAMAsset(
                id: "asset-\(i)",
                kind: "image",
                filename: "render\(i).png",
                absolutePath: path,
                fileSize: Int64(contents.utf8.count),
                sha256: "sha-\(i)",
                width: 1024,
                height: 1536,
                createdAt: Date(timeIntervalSince1970: 1_700_000_000 + Double(i)),
                modifiedAt: Date(timeIntervalSince1970: 1_700_000_100 + Double(i)),
                ingestedAt: Date(timeIntervalSince1970: 1_700_000_200 + Double(i)),
                orphaned: false,
                prompt: "prompt \(i)",
                negativePrompt: "negative \(i)",
                seed: 1000 + i,
                steps: 20 + i,
                guidance: 3.5,
                modelFamily: "z-image-turbo",
                rating: i,
                favorite: i.isMultiple(of: 2),
                contentMode: "apple",
                characterName: "kira",
                source: "kira"
            )
            let stored = try await env.store.insertAsset(asset)
            makeThumbnail(for: stored.id, ingestor: env.ingestor)
            assets.append(stored)
        }

        let archiveResult = try await env.archiver.archive(
            .init(name: "RoundTrip", destinationRoot: env.archiveRoot, assets: assets)
        )

        let restoreResult = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(restoreResult.restored == 3)
        #expect(restoreResult.skipped == 0)
        #expect(restoreResult.reIdentified == 0)
        #expect(restoreResult.renamed == 0)
        #expect(restoreResult.failed.isEmpty)

        for original in assets {
            let restoredPath = (env.watchDir as NSString).appendingPathComponent(original.filename)
            #expect(FileManager.default.fileExists(atPath: restoredPath))
            let restored = try await env.store.fetchAsset(byPath: restoredPath)
            let restoredAsset = try #require(restored)
            #expect(restoredAsset.id == original.id)
            #expect(restoredAsset.rating == original.rating)
            #expect(restoredAsset.favorite == original.favorite)
            #expect(restoredAsset.prompt == original.prompt)
            #expect(restoredAsset.negativePrompt == original.negativePrompt)
            #expect(restoredAsset.seed == original.seed)
            #expect(restoredAsset.steps == original.steps)
            #expect(restoredAsset.guidance == original.guidance)
            #expect(restoredAsset.modelFamily == original.modelFamily)
            #expect(restoredAsset.contentMode == original.contentMode)
            #expect(restoredAsset.characterName == original.characterName)
            #expect(restoredAsset.source == original.source)
            #expect(restoredAsset.width == original.width)
            #expect(restoredAsset.height == original.height)
            #expect(restoredAsset.createdAt.timeIntervalSince1970 == original.createdAt.timeIntervalSince1970)
        }
    }

    // MARK: - Restore: version policy

    @Test("restore rejects a manifest written by a newer schema version before doing any work")
    @MainActor
    func restoreRejectsUnsupportedSchemaVersion() async throws {
        let env = try await makeEnvironment()
        let bundlePath = (env.archiveRoot as NSString).appendingPathComponent("Future.cbarchive")
        try FileManager.default.createDirectory(atPath: bundlePath, withIntermediateDirectories: true)
        let futureManifest: [String: Any] = [
            "schemaVersion": 99, "archiveId": "x", "name": "Future", "createdAt": 0,
            "producer": "test", "assetCount": 0, "totalBytes": 0, "folders": [],
        ]
        let data = try JSONSerialization.data(withJSONObject: futureManifest)
        try data.write(to: URL(fileURLWithPath: (bundlePath as NSString).appendingPathComponent("manifest.json")))
        FileManager.default.createFile(
            atPath: (bundlePath as NSString).appendingPathComponent("entries.jsonl"), contents: nil
        )

        await #expect(throws: ArchiveError.unsupportedSchemaVersion(99)) {
            try await env.archiver.restore(.init(bundlePath: bundlePath))
        }
        let count = try await env.store.assetCount()
        #expect(count == 0)
    }

    // MARK: - Restore: id collision — skip

    @Test("restore skips entirely when the live asset already occupies the archived id and path, preserving its rating")
    @MainActor
    func restoreSkipsWhenAlreadyPresent() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "SkipTest", destinationRoot: env.archiveRoot, assets: [asset])
        )

        // Live asset reappears at the same id and the same destination path
        // (as if it was never removed), with a rating the user bumped since.
        let destPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        writeFile("render.png", in: env.watchDir, contents: "render-bytes-again")
        let bumped = DAMAsset(id: asset.id, filename: "render.png", absolutePath: destPath, fileSize: 999, rating: 5)
        try await env.store.insertAsset(bumped)

        let restoreResult = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(restoreResult.skipped == 1)
        #expect(restoreResult.restored == 0)
        #expect(restoreResult.reIdentified == 0)
        #expect(restoreResult.failed.isEmpty)

        let live = try await env.store.fetchAsset(byPath: destPath)
        #expect(live?.rating == 5)
        let assetCount = try await env.store.assetCount()
        #expect(assetCount == 1)
    }

    // MARK: - Restore: id collision — overwrite

    @Test("overwriteExistingMetadata reverts the live rating to the archived value")
    @MainActor
    func restoreOverwriteRevertsRating() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        let ratedAsset = DAMAsset(
            id: asset.id, filename: asset.filename, absolutePath: asset.absolutePath,
            fileSize: asset.fileSize, rating: 2
        )
        let stored = try await env.store.insertAsset(ratedAsset)
        makeThumbnail(for: stored.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "OverwriteTest", destinationRoot: env.archiveRoot, assets: [stored])
        )

        let destPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        writeFile("render.png", in: env.watchDir, contents: "render-bytes-again")
        let bumped = DAMAsset(id: asset.id, filename: "render.png", absolutePath: destPath, fileSize: 999, rating: 5)
        try await env.store.insertAsset(bumped)

        let restoreResult = try await env.archiver.restore(
            .init(bundlePath: archiveResult.bundlePath, overwriteExistingMetadata: true)
        )
        #expect(restoreResult.restored == 1)
        #expect(restoreResult.skipped == 0)

        let live = try await env.store.fetchAsset(byPath: destPath)
        #expect(live?.rating == 2)
        let assetCount = try await env.store.assetCount()
        #expect(assetCount == 1)
    }

    // MARK: - Restore: id collision — orphaned row (same id/path, file missing)

    @Test("orphaned row (same id, same recorded path, file missing): the skip branch still restores the file for real")
    @MainActor
    func restoreReclaimsOrphanedRowOnSkipBranch() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "OrphanSkipTest", destinationRoot: env.archiveRoot, assets: [asset])
        )

        // A live row re-appears at the archived id and the same recorded
        // path, but the backing file was deleted out from under the DB
        // (e.g. removed from Finder — the exact state pruneOrphans() exists
        // to clean up).
        let destPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        let orphan = DAMAsset(id: asset.id, filename: "render.png", absolutePath: destPath, fileSize: 999, rating: 5)
        try await env.store.insertAsset(orphan)
        #expect(!FileManager.default.fileExists(atPath: destPath))

        let restoreResult = try await env.archiver.restore(
            .init(bundlePath: archiveResult.bundlePath, overwriteExistingMetadata: false)
        )
        #expect(restoreResult.restored == 1)
        #expect(restoreResult.skipped == 0)
        #expect(restoreResult.reIdentified == 0)
        #expect(restoreResult.failed.isEmpty)

        #expect(FileManager.default.fileExists(atPath: destPath))
        let live = try await env.store.fetchAsset(byPath: destPath)
        #expect(live?.id == asset.id)
        let assetCount = try await env.store.assetCount()
        #expect(assetCount == 1)
    }

    @Test("orphaned row (same id, same recorded path, file missing): the overwrite branch restores the file and reverts metadata")
    @MainActor
    func restoreReclaimsOrphanedRowOnOverwriteBranch() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        let ratedAsset = DAMAsset(
            id: asset.id, filename: asset.filename, absolutePath: asset.absolutePath,
            fileSize: asset.fileSize, rating: 2
        )
        let stored = try await env.store.insertAsset(ratedAsset)
        makeThumbnail(for: stored.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "OrphanOverwriteTest", destinationRoot: env.archiveRoot, assets: [stored])
        )

        let destPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        let orphan = DAMAsset(id: asset.id, filename: "render.png", absolutePath: destPath, fileSize: 999, rating: 5)
        try await env.store.insertAsset(orphan)
        #expect(!FileManager.default.fileExists(atPath: destPath))

        let restoreResult = try await env.archiver.restore(
            .init(bundlePath: archiveResult.bundlePath, overwriteExistingMetadata: true)
        )
        #expect(restoreResult.restored == 1)
        #expect(restoreResult.skipped == 0)
        #expect(restoreResult.reIdentified == 0)
        #expect(restoreResult.failed.isEmpty)

        #expect(FileManager.default.fileExists(atPath: destPath))
        let live = try await env.store.fetchAsset(byPath: destPath)
        #expect(live?.rating == 2)
        let assetCount = try await env.store.assetCount()
        #expect(assetCount == 1)
    }

    // MARK: - Restore: id collision — re-ID

    @Test("restore mints a fresh id when the archived id is taken by an asset at a different path; both rows survive")
    @MainActor
    func restoreReIdentifiesOnDifferentPath() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "ReIDTest", destinationRoot: env.archiveRoot, assets: [asset])
        )

        let otherPath = writeFile("unrelated.png", in: env.watchDir, contents: "unrelated-bytes")
        let collidingAsset = DAMAsset(id: asset.id, filename: "unrelated.png", absolutePath: otherPath)
        try await env.store.insertAsset(collidingAsset)

        let restoreResult = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(restoreResult.reIdentified == 1)
        #expect(restoreResult.restored == 0)
        #expect(restoreResult.skipped == 0)

        let count = try await env.store.assetCount()
        #expect(count == 2)

        let restoredPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        #expect(FileManager.default.fileExists(atPath: restoredPath))
        #expect(FileManager.default.fileExists(atPath: otherPath))
        let restoredRow = try await env.store.fetchAsset(byPath: restoredPath)
        #expect(restoredRow != nil)
        #expect(restoredRow?.id != asset.id)
    }

    // MARK: - Restore: filename collision

    @Test("a different, DB-tracked file already at the destination filename is renamed, not overwritten")
    @MainActor
    func restoreRenamesOnFilenameCollision() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "CollisionTest", destinationRoot: env.archiveRoot, assets: [asset])
        )

        // A different, unrelated, DB-tracked file now occupies render.png.
        let unrelatedPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        let unrelatedContents = "a-totally-different-file-of-a-different-size"
        FileManager.default.createFile(atPath: unrelatedPath, contents: Data(unrelatedContents.utf8))
        let unrelatedAsset = DAMAsset(
            id: "unrelated-id", filename: "render.png", absolutePath: unrelatedPath,
            fileSize: Int64(unrelatedContents.utf8.count)
        )
        try await env.store.insertAsset(unrelatedAsset)

        let restoreResult = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(restoreResult.renamed == 1)
        #expect(restoreResult.restored == 1)

        let originalPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        let renamedPath = (env.watchDir as NSString).appendingPathComponent("render (restored).png")
        #expect(FileManager.default.fileExists(atPath: originalPath))
        #expect(FileManager.default.fileExists(atPath: renamedPath))

        let count = try await env.store.assetCount()
        #expect(count == 2)
    }

    // MARK: - Restore: folder re-creation

    @Test("folder re-creation: an absent folder is created with the archived id")
    @MainActor
    func restoreRecreatesAbsentFolderWithArchivedId() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("pick.png", in: env.watchDir, contents: "pick")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)
        let folder = try await env.store.createFolder(name: "Picks", id: "folder-fixed-id")
        try await env.store.assignAssets(ids: [asset.id], toFolder: folder.id)

        let archiveResult = try await env.archiver.archive(
            .init(name: "FolderCreate", destinationRoot: env.archiveRoot, assets: [asset], folder: folder)
        )
        let foldersAfterArchive = try await env.store.listFolders()
        #expect(!foldersAfterArchive.contains(where: { $0.id == folder.id }))

        _ = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))

        let folders = try await env.store.listFolders()
        let restoredFolder = try #require(folders.first(where: { $0.id == "folder-fixed-id" }))
        #expect(restoredFolder.name == "Picks")

        let assignments = try await env.store.folderAssignments()
        let restoredAssetRow = try await env.store.fetchAsset(
            byPath: (env.watchDir as NSString).appendingPathComponent("pick.png")
        )
        let resolved = try #require(restoredAssetRow)
        #expect(assignments[resolved.id] == "folder-fixed-id")
    }

    @Test("folder re-creation: a folder present with the same name but a different id is reused, not duplicated")
    @MainActor
    func restoreReusesFolderBySameName() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("pick.png", in: env.watchDir, contents: "pick")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)
        let folder = try await env.store.createFolder(name: "Picks", id: "archived-folder-id")
        try await env.store.assignAssets(ids: [asset.id], toFolder: folder.id)

        let archiveResult = try await env.archiver.archive(
            .init(name: "FolderReuse", destinationRoot: env.archiveRoot, assets: [asset], folder: folder)
        )

        // The user re-created "Picks" by hand before restoring; it got a new id.
        let handMadeFolder = try await env.store.createFolder(name: "Picks")
        #expect(handMadeFolder.id != "archived-folder-id")

        _ = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))

        let folders = try await env.store.listFolders()
        #expect(folders.filter { $0.name.caseInsensitiveCompare("Picks") == .orderedSame }.count == 1)
        #expect(folders.contains(where: { $0.id == handMadeFolder.id }))
        #expect(!folders.contains(where: { $0.id == "archived-folder-id" }))

        let assignments = try await env.store.folderAssignments()
        let restoredAssetRow = try await env.store.fetchAsset(
            byPath: (env.watchDir as NSString).appendingPathComponent("pick.png")
        )
        let resolved = try #require(restoredAssetRow)
        #expect(assignments[resolved.id] == handMadeFolder.id)
    }

    // MARK: - Restore: partial restore

    @Test("partial restore: a one-id subset restores exactly one row and one file")
    @MainActor
    func restorePartialSubset() async throws {
        let env = try await makeEnvironment()
        var assets: [DAMAsset] = []
        for i in 1...3 {
            let path = writeFile("pick\(i).png", in: env.watchDir, contents: "pick-\(i)")
            let stored = try await env.ingestor.ingestFile(at: path)
            makeThumbnail(for: stored.id, ingestor: env.ingestor)
            assets.append(stored)
        }
        let archiveResult = try await env.archiver.archive(
            .init(name: "PartialTest", destinationRoot: env.archiveRoot, assets: assets)
        )

        let target = assets[1]
        let result = try await env.archiver.restore(
            .init(bundlePath: archiveResult.bundlePath, assetIds: [target.id])
        )
        #expect(result.restored == 1)
        #expect(result.skipped == 0)
        #expect(result.failed.isEmpty)

        let count = try await env.store.assetCount()
        #expect(count == 1)

        let restoredPath = (env.watchDir as NSString).appendingPathComponent(target.filename)
        #expect(FileManager.default.fileExists(atPath: restoredPath))
        for other in assets where other.id != target.id {
            let otherPath = (env.watchDir as NSString).appendingPathComponent(other.filename)
            #expect(!FileManager.default.fileExists(atPath: otherPath))
        }
    }

    // MARK: - Restore: poller registration (C1)

    @Test("a restored file is registered with the ingestor so the next poller scan does not re-ingest it, losing source")
    @MainActor
    func restoreRegistersWithIngestorPoller() async throws {
        let env = try await makeEnvironment()
        let sourced = DAMAsset(
            id: "kira-asset", filename: "kira-render.png",
            absolutePath: writeFile("kira-render.png", in: env.watchDir, contents: "kira-bytes"),
            source: "kira"
        )
        let stored = try await env.store.insertAsset(sourced)
        makeThumbnail(for: stored.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "PollerTest", destinationRoot: env.archiveRoot, assets: [stored])
        )
        #expect(try await env.store.assetCount() == 0)

        _ = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))

        let restoredPath = (env.watchDir as NSString).appendingPathComponent("kira-render.png")
        #expect(FileManager.default.fileExists(atPath: restoredPath))

        // Back-date the file so `isFileStable`'s "at least 1 second old"
        // gate can't itself hide a missing markKnown call.
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-5)], ofItemAtPath: restoredPath
        )

        let ingestedCountBefore = env.ingestor.ingestedCount
        await env.ingestor.scanForNewFiles()

        // No spurious re-ingest happened: `markKnown` (called from
        // `performCopyAndInsert` with the restore's final destination path)
        // kept the poller from treating the restored file as new.
        #expect(env.ingestor.ingestedCount == ingestedCountBefore)

        let count = try await env.store.assetCount()
        #expect(count == 1)
        let row = try await env.store.fetchAsset(byPath: restoredPath)
        #expect(row?.id == "kira-asset")
        #expect(row?.source == "kira")
    }

    // MARK: - Restore: idempotence

    @Test("restore idempotence survives a rename: restoring twice after a planted filename collision reports skipped, not reIdentified, on the second run")
    @MainActor
    func restoreIdempotentAfterRename() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "RenameIdempotenceTest", destinationRoot: env.archiveRoot, assets: [asset])
        )

        // Plant a different, unrelated, DB-tracked file at the destination
        // filename so the first restore is forced to rename — reproducing
        // "restore lands under a different name than it was archived under".
        let collisionPath = (env.watchDir as NSString).appendingPathComponent("render.png")
        let unrelatedContents = "a-totally-different-file-of-a-different-size"
        FileManager.default.createFile(atPath: collisionPath, contents: Data(unrelatedContents.utf8))
        let unrelatedAsset = DAMAsset(
            id: "unrelated-id", filename: "render.png", absolutePath: collisionPath,
            fileSize: Int64(unrelatedContents.utf8.count)
        )
        try await env.store.insertAsset(unrelatedAsset)

        let first = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(first.restored == 1)
        #expect(first.renamed == 1)
        #expect(first.reIdentified == 0)
        let renamedPath = (env.watchDir as NSString).appendingPathComponent("render (restored).png")
        #expect(FileManager.default.fileExists(atPath: renamedPath))
        let countAfterFirst = try await env.store.assetCount()
        #expect(countAfterFirst == 2)   // the unrelated asset + the restored one

        // Second restore: the archived id now lives at "render (restored).png",
        // which no longer matches the freshly-recomputed destination candidate
        // ("render.png", still occupied by the unrelated asset) — the exact
        // path mismatch that used to be misread as "id taken by someone else".
        let second = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(second.skipped == 1)
        #expect(second.restored == 0)
        #expect(second.reIdentified == 0)
        #expect(second.renamed == 0)
        #expect(second.failed.isEmpty)

        let countAfterSecond = try await env.store.assetCount()
        #expect(countAfterSecond == countAfterFirst)

        // No third file (e.g. "render (restored 2).png") appears.
        let doubleRenamedPath = (env.watchDir as NSString).appendingPathComponent("render (restored 2).png")
        #expect(!FileManager.default.fileExists(atPath: doubleRenamedPath))
        let contents = try FileManager.default.contentsOfDirectory(atPath: env.watchDir)
        #expect(Set(contents) == ["render.png", "render (restored).png"])
    }

    @Test("restoring twice reports everything skipped on the second run and leaves assetCount unchanged")
    @MainActor
    func restoreIsIdempotent() async throws {
        let env = try await makeEnvironment()
        var assets: [DAMAsset] = []
        for i in 1...2 {
            let path = writeFile("dup\(i).png", in: env.watchDir, contents: "dup-\(i)")
            let stored = try await env.ingestor.ingestFile(at: path)
            makeThumbnail(for: stored.id, ingestor: env.ingestor)
            assets.append(stored)
        }
        let archiveResult = try await env.archiver.archive(
            .init(name: "IdempotenceTest", destinationRoot: env.archiveRoot, assets: assets)
        )

        let first = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(first.restored == 2)
        let countAfterFirst = try await env.store.assetCount()

        let second = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(second.skipped == 2)
        #expect(second.restored == 0)
        #expect(second.reIdentified == 0)
        #expect(second.renamed == 0)

        let countAfterSecond = try await env.store.assetCount()
        #expect(countAfterSecond == countAfterFirst)
    }

    // MARK: - Restore: progress callback

    @Test("progress callback fires per asset with the entry-count total and ends at done == total")
    @MainActor
    func restoreProgressCallback() async throws {
        let env = try await makeEnvironment()
        var assets: [DAMAsset] = []
        for i in 1...3 {
            let path = writeFile("prog\(i).png", in: env.watchDir, contents: "prog-\(i)")
            let stored = try await env.ingestor.ingestFile(at: path)
            makeThumbnail(for: stored.id, ingestor: env.ingestor)
            assets.append(stored)
        }
        let archiveResult = try await env.archiver.archive(
            .init(name: "ProgressTest", destinationRoot: env.archiveRoot, assets: assets)
        )

        var calls: [(done: Int, total: Int)] = []
        let result = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath)) { done, total in
            calls.append((done, total))
        }
        #expect(!calls.isEmpty)
        #expect(calls.allSatisfy { $0.total == 3 })
        #expect(calls.last?.done == 3)
        #expect(result.restored == 3)
    }

    // MARK: - resumePendingRemovals

    /// Writes `{"assetIds": ids}` to `PENDING_REMOVAL.json` in `bundleRoot`,
    /// matching the (file-private) marker format `GalleryArchiver.archive`
    /// itself writes at step 7.
    private func writePendingRemovalMarker(ids: [String], in bundleRoot: String) throws {
        let path = (bundleRoot as NSString).appendingPathComponent("PENDING_REMOVAL.json")
        let json = try JSONSerialization.data(withJSONObject: ["assetIds": ids])
        try json.write(to: URL(fileURLWithPath: path))
    }

    @Test("resumePendingRemovals finishes an interrupted removal: live id removed, thumbnail gone, marker deleted")
    @MainActor
    func resumePendingRemovalsFinishesInterruptedRemoval() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)

        // A normal archive fully commits: source + thumbnail removed, marker
        // gone. Reconstruct the "crash between step 8 and step 10" state on
        // top of that genuinely valid, committed bundle: the asset row and
        // its thumbnail reappear (as if source removal never finished) and
        // PENDING_REMOVAL.json is rewritten to list it.
        let archiveResult = try await env.archiver.archive(
            .init(name: "ResumeTest", destinationRoot: env.archiveRoot, assets: [asset])
        )
        let incompletePath = (archiveResult.bundlePath as NSString).appendingPathComponent("INCOMPLETE.json")
        #expect(!FileManager.default.fileExists(atPath: incompletePath))

        try await env.store.insertAsset(asset)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)
        try writePendingRemovalMarker(ids: [asset.id], in: archiveResult.bundlePath)

        await env.archiver.resumePendingRemovals(in: [env.archiveRoot])

        let idsAfterFirstResume = try await env.store.allAssetIds()
        #expect(!idsAfterFirstResume.contains(asset.id))
        #expect(!FileManager.default.fileExists(atPath: env.ingestor.thumbnailPath(for: asset.id)))
        let pendingRemovalPath = (archiveResult.bundlePath as NSString).appendingPathComponent("PENDING_REMOVAL.json")
        #expect(!FileManager.default.fileExists(atPath: pendingRemovalPath))

        // Idempotent: running again with the marker (and the asset) already
        // gone is a no-op, and does not throw.
        await env.archiver.resumePendingRemovals(in: [env.archiveRoot])
        let idsAfterSecondResume = try await env.store.allAssetIds()
        #expect(idsAfterSecondResume.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: pendingRemovalPath))
    }

    @Test("resumePendingRemovals leaves a bundle with INCOMPLETE.json untouched, even if PENDING_REMOVAL.json is also present")
    @MainActor
    func resumePendingRemovalsSkipsIncompleteBundle() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)

        let bundlePath = (env.archiveRoot as NSString).appendingPathComponent("Doomed.cbarchive")
        try FileManager.default.createDirectory(atPath: bundlePath, withIntermediateDirectories: true)
        let incompletePath = (bundlePath as NSString).appendingPathComponent("INCOMPLETE.json")
        FileManager.default.createFile(atPath: incompletePath, contents: Data("{}".utf8))
        try writePendingRemovalMarker(ids: [asset.id], in: bundlePath)

        await env.archiver.resumePendingRemovals(in: [env.archiveRoot])

        let idsAfterResume = try await env.store.allAssetIds()
        #expect(idsAfterResume.contains(asset.id))
        #expect(FileManager.default.fileExists(atPath: asset.absolutePath))
        let pendingRemovalPath = (bundlePath as NSString).appendingPathComponent("PENDING_REMOVAL.json")
        #expect(FileManager.default.fileExists(atPath: pendingRemovalPath))
        #expect(FileManager.default.fileExists(atPath: incompletePath))
    }

    @Test("resumePendingRemovals leaves the marker in place when the removal fails, and a later clean pass finishes the job")
    @MainActor
    func resumePendingRemovalsLeavesMarkerOnDeleteFailure() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)

        let archiveResult = try await env.archiver.archive(
            .init(name: "FailureTest", destinationRoot: env.archiveRoot, assets: [asset])
        )
        // Reconstruct the interrupted-removal state, as in the happy-path test above.
        try await env.store.insertAsset(asset)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)
        try writePendingRemovalMarker(ids: [asset.id], in: archiveResult.bundlePath)
        let pendingRemovalPath = (archiveResult.bundlePath as NSString).appendingPathComponent("PENDING_REMOVAL.json")

        // Force the row DELETE inside `ingestor.deleteAsset` to fail
        // deterministically, without racing and without touching
        // `DAMStore.swift`: a second raw connection to the same on-disk
        // database (opened directly via the system SQLite3 module — the
        // same one `DAMStore` itself imports, available to any target on
        // the SDK with no extra package wiring) takes the single WAL writer
        // lock with `BEGIN EXCLUSIVE` and holds it. `DAMStore` never sets a
        // busy_timeout, so the store's own write attempt hits SQLITE_BUSY
        // immediately rather than blocking. WAL mode's whole point is that
        // readers are never blocked by a pending writer, so
        // `allAssetIds`/`assetCount`/`fetchAssets` — the snapshot reads —
        // are unaffected; only the DELETE fails. (An attempt at the same
        // thing via the "user immutable" file flag on the `-wal` file did
        // not reproduce a failure in practice — SQLite apparently satisfies
        // that particular write from data already resident in its own
        // in-process cache/mapping rather than re-touching the file in a
        // way the flag intercepts — so this lock-based construction is used
        // instead.)
        var blocker: OpaquePointer?
        #expect(sqlite3_open_v2(env.dbPath, &blocker, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK)
        #expect(sqlite3_exec(blocker, "BEGIN EXCLUSIVE", nil, nil, nil) == SQLITE_OK)

        await env.archiver.resumePendingRemovals(in: [env.archiveRoot])

        // Failed pass: marker survives, and the DB row (the thing whose
        // removal actually failed) is still there. The pre-DB-write file
        // and thumbnail removals in `ingestor.deleteAsset` already ran by
        // this point regardless — that partial-file-cleanup ordering is
        // `ingestor.deleteAsset`'s own behavior, reused as-is, not
        // something T7 introduces or is responsible for making atomic.
        #expect(FileManager.default.fileExists(atPath: pendingRemovalPath))
        let idsAfterFailedResume = try await env.store.allAssetIds()
        #expect(idsAfterFailedResume.contains(asset.id))

        // Release the lock and re-run: idempotent recovery finishes the job.
        sqlite3_exec(blocker, "ROLLBACK", nil, nil, nil)
        sqlite3_close(blocker)

        await env.archiver.resumePendingRemovals(in: [env.archiveRoot])

        let idsAfterCleanResume = try await env.store.allAssetIds()
        #expect(!idsAfterCleanResume.contains(asset.id))
        #expect(!FileManager.default.fileExists(atPath: pendingRemovalPath))
    }

    // MARK: - resumePendingRemovals: isRunning lock (I1)

    /// Polls `archiver.phase` (up to ~200 cooperative yields) until it
    /// matches `expected`, then reports whether it was observed. Used to
    /// catch a genuinely in-flight async operation without a fixed sleep.
    @MainActor
    private func waitForPhase(_ expected: GalleryArchiver.Phase, on archiver: GalleryArchiver) async -> Bool {
        for _ in 0..<200 {
            if archiver.phase == expected { return true }
            await Task.yield()
        }
        return false
    }

    @Test("resumePendingRemovals holds the same isRunning lock as archive/restore: phase is .removingSources while the sweep is in flight, back to .idle after")
    @MainActor
    func resumePendingRemovalsSetsPhase() async throws {
        let env = try await makeEnvironment()
        let path = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let asset = try await env.ingestor.ingestFile(at: path)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)
        let archiveResult = try await env.archiver.archive(
            .init(name: "PhaseTest", destinationRoot: env.archiveRoot, assets: [asset])
        )
        try await env.store.insertAsset(asset)
        makeThumbnail(for: asset.id, ingestor: env.ingestor)
        try writePendingRemovalMarker(ids: [asset.id], in: archiveResult.bundlePath)

        #expect(env.archiver.phase == .idle)
        #expect(!env.archiver.isRunning)

        async let sweep: Void = env.archiver.resumePendingRemovals(in: [env.archiveRoot])

        let observedRemoving = await waitForPhase(.removingSources, on: env.archiver)
        #expect(observedRemoving)

        await sweep
        #expect(env.archiver.phase == .idle)
        #expect(!env.archiver.isRunning)
    }

    @Test("a sweep skipped while the archiver is already running (concurrent restore) leaves PENDING_REMOVAL.json intact")
    @MainActor
    func resumePendingRemovalsSkippedWhileArchiverBusy() async throws {
        let env = try await makeEnvironment()

        // Bundle A: reconstruct the "interrupted removal" state the sweep would finish.
        let pathA = writeFile("render.png", in: env.watchDir, contents: "render-bytes")
        let assetA = try await env.ingestor.ingestFile(at: pathA)
        makeThumbnail(for: assetA.id, ingestor: env.ingestor)
        let archiveResultA = try await env.archiver.archive(
            .init(name: "BusyTest", destinationRoot: env.archiveRoot, assets: [assetA])
        )
        try await env.store.insertAsset(assetA)
        makeThumbnail(for: assetA.id, ingestor: env.ingestor)
        try writePendingRemovalMarker(ids: [assetA.id], in: archiveResultA.bundlePath)
        let pendingRemovalPathA = (archiveResultA.bundlePath as NSString).appendingPathComponent("PENDING_REMOVAL.json")

        // Bundle B: archived separately so a restore of it can be kept
        // genuinely in flight (a real suspension on the store actor)
        // concurrently with the sweep call below.
        let pathB = writeFile("other.png", in: env.watchDir, contents: "other-bytes")
        let assetB = try await env.ingestor.ingestFile(at: pathB)
        makeThumbnail(for: assetB.id, ingestor: env.ingestor)
        let archiveResultB = try await env.archiver.archive(
            .init(name: "BusyTestOther", destinationRoot: env.archiveRoot, assets: [assetB])
        )

        async let busyRestore = env.archiver.restore(.init(bundlePath: archiveResultB.bundlePath))

        let observedRestoring = await waitForPhase(.restoring, on: env.archiver)
        #expect(observedRestoring)

        // The sweep must not race restore()'s in-flight work — it should
        // see isRunning and skip entirely, leaving the marker untouched.
        await env.archiver.resumePendingRemovals(in: [env.archiveRoot])

        #expect(FileManager.default.fileExists(atPath: pendingRemovalPathA))
        let idsWhileSkipped = try await env.store.allAssetIds()
        #expect(idsWhileSkipped.contains(assetA.id))

        _ = try await busyRestore
    }
}
