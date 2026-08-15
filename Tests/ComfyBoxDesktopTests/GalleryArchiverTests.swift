// GalleryArchiverTests.swift — Tests for GalleryArchiver.archive (T3)
//
// Archive-only per the FDD task brief for T3: bundle well-formedness,
// sidecar/folder/secured handling, and crash-safety (nothing destroyed
// before the commit point). Restore / resumePendingRemovals are later tasks.

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
}
