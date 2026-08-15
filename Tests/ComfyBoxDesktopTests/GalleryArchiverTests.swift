// GalleryArchiverTests.swift — Tests for GalleryArchiver.archive (T3) and
// GalleryArchiver.restore (T4).
//
// Archive: bundle well-formedness, sidecar/folder/secured handling, and
// crash-safety (nothing destroyed before the commit point).
// Restore: full round-trip field equality, the four id-collision cases,
// filename-collision renaming, folder re-creation, partial restore,
// idempotence, and progress reporting. `resumePendingRemovals` is T7 and is
// not implemented here.

import Testing
import Foundation
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
        archiveRoot: String
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
        return (archiver, ingestor, store, watchDir, thumbDir, archiveRoot)
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

    // MARK: - Restore: idempotence

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
}
