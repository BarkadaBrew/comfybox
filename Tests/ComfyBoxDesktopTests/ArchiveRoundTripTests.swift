// ArchiveRoundTripTests.swift — Integration test for GalleryArchiver
// archive() + restore() together (T4 §9.6).
//
// Uses real PNGs (TestData.writeRealPNG) so AssetIngestor's thumbnail
// generation actually produces non-empty JPEGs — generateThumbnail no-ops
// on fake (non-decodable) image bytes.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("ArchiveRoundTrip")
struct ArchiveRoundTripTests {
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
            .appendingPathComponent("archive-roundtrip-test-\(UUID().uuidString)")
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

    private func withAnnotations(_ asset: DAMAsset, rating: Int, favorite: Bool) -> DAMAsset {
        DAMAsset(
            id: asset.id, kind: asset.kind, filename: asset.filename,
            absolutePath: asset.absolutePath, fileSize: asset.fileSize, sha256: asset.sha256,
            width: asset.width, height: asset.height,
            createdAt: asset.createdAt, modifiedAt: asset.modifiedAt, ingestedAt: asset.ingestedAt,
            orphaned: asset.orphaned, prompt: asset.prompt, negativePrompt: asset.negativePrompt,
            seed: asset.seed, steps: asset.steps, guidance: asset.guidance, modelFamily: asset.modelFamily,
            rating: rating, favorite: favorite, contentMode: asset.contentMode,
            characterName: asset.characterName, source: asset.source
        )
    }

    @Test("archive then restore round-trips 5 real assets: emptied gallery, well-formed bundle, 5 rows, 5 files, 5 non-empty thumbnails, folder re-created")
    @MainActor
    func fullRoundTrip() async throws {
        let env = try await makeEnvironment()

        var assets: [DAMAsset] = []
        for i in 1...5 {
            let path = (env.watchDir as NSString).appendingPathComponent("photo\(i).png")
            #expect(TestData.writeRealPNG(at: path, width: 32, height: 32))
            let stored = try await env.ingestor.ingestFile(at: path)
            assets.append(stored)
        }

        // Rate + favorite a couple.
        var annotated: [DAMAsset] = []
        for (index, asset) in assets.enumerated() {
            switch index {
            case 0:
                annotated.append(try await env.store.insertAsset(withAnnotations(asset, rating: 5, favorite: true)))
            case 1:
                annotated.append(try await env.store.insertAsset(withAnnotations(asset, rating: 3, favorite: false)))
            default:
                annotated.append(asset)
            }
        }

        // File two into a folder.
        let folder = try await env.store.createFolder(name: "Favorites")
        try await env.store.assignAssets(ids: [annotated[0].id, annotated[1].id], toFolder: folder.id)

        // Sanity: thumbnails exist and are non-empty before archiving.
        for asset in annotated {
            let attrs = try FileManager.default.attributesOfItem(atPath: env.ingestor.thumbnailPath(for: asset.id))
            let size = try #require(attrs[.size] as? Int)
            #expect(size > 0)
        }

        let archiveResult = try await env.archiver.archive(
            .init(name: "RoundTrip", destinationRoot: env.archiveRoot, assets: annotated)
        )
        #expect(archiveResult.archived == 5)
        #expect(archiveResult.failed.isEmpty)

        // Gallery empty, bundle well-formed.
        let emptiedCount = try await env.store.assetCount()
        #expect(emptiedCount == 0)
        for asset in annotated {
            #expect(!FileManager.default.fileExists(atPath: asset.absolutePath))
        }

        let manifestPath = (archiveResult.bundlePath as NSString).appendingPathComponent("manifest.json")
        let manifest = try ArchiveManifest.decode(Data(contentsOf: URL(fileURLWithPath: manifestPath)))
        #expect(manifest.assetCount == 5)
        #expect(manifest.folders.count == 1)
        #expect(manifest.folders.first?.name == "Favorites")
        let entriesPath = (archiveResult.bundlePath as NSString).appendingPathComponent("entries.jsonl")
        #expect(FileManager.default.fileExists(atPath: entriesPath))

        // Restore.
        let restoreResult = try await env.archiver.restore(.init(bundlePath: archiveResult.bundlePath))
        #expect(restoreResult.restored == 5)
        #expect(restoreResult.skipped == 0)
        #expect(restoreResult.reIdentified == 0)
        #expect(restoreResult.failed.isEmpty)

        let restoredCount = try await env.store.assetCount()
        #expect(restoredCount == 5)

        var restoredFolderId: String?
        for original in annotated {
            let restoredPath = (env.watchDir as NSString).appendingPathComponent(original.filename)
            #expect(FileManager.default.fileExists(atPath: restoredPath))

            let restored = try await env.store.fetchAsset(byPath: restoredPath)
            let restoredAsset = try #require(restored)
            #expect(restoredAsset.id == original.id)
            #expect(restoredAsset.rating == original.rating)
            #expect(restoredAsset.favorite == original.favorite)
            #expect(restoredAsset.prompt == original.prompt)
            #expect(restoredAsset.width == original.width)
            #expect(restoredAsset.height == original.height)

            let thumbPath = env.ingestor.thumbnailPath(for: restoredAsset.id)
            let attrs = try FileManager.default.attributesOfItem(atPath: thumbPath)
            let size = try #require(attrs[.size] as? Int)
            #expect(size > 0)

            if original.id == annotated[0].id || original.id == annotated[1].id {
                let assignments = try await env.store.folderAssignments()
                restoredFolderId = assignments[restoredAsset.id]
                #expect(restoredFolderId != nil)
            }
        }

        let folders = try await env.store.listFolders()
        #expect(folders.count == 1)
        #expect(folders.first?.name == "Favorites")
        #expect(folders.first?.id == restoredFolderId)
    }
}
