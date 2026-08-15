// GalleryMaintenanceTests.swift — Tests for orphan-thumbnail scan/purge

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("GalleryMaintenance")
struct GalleryMaintenanceTests {
    /// Create an isolated temp environment: watch dir, thumbnail dir, store, ingestor, maintenance.
    @MainActor
    private func makeEnvironment() async throws -> (
        maintenance: GalleryMaintenance, ingestor: AssetIngestor, store: DAMStore,
        watchDir: String, thumbDir: String
    ) {
        let base = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("maintenance-test-\(UUID().uuidString)")
        let watchDir = (base as NSString).appendingPathComponent("output")
        let thumbDir = (base as NSString).appendingPathComponent("thumbnails")
        let secureDir = (base as NSString).appendingPathComponent("vault")
        try FileManager.default.createDirectory(atPath: watchDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: thumbDir, withIntermediateDirectories: true)
        let dbPath = (base as NSString).appendingPathComponent("dam.sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let ingestor = AssetIngestor(store: store, watchDirectory: watchDir, thumbnailDirectory: thumbDir)
        ingestor.secureDirectory = secureDir
        let maintenance = GalleryMaintenance(store: store, ingestor: ingestor)
        return (maintenance, ingestor, store, watchDir, thumbDir)
    }

    private func writeFile(_ name: String, in dir: String, contents: String = "thumb-bytes") -> String {
        let path = (dir as NSString).appendingPathComponent(name)
        FileManager.default.createFile(atPath: path, contents: Data(contents.utf8))
        return path
    }

    @Test("scan finds a stale thumbnail among live ones and reports its size")
    @MainActor
    func scanFindsOrphan() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("render.png", in: env.watchDir)
        let stored = try await env.ingestor.ingestFile(at: imagePath)

        // A second live asset.
        let imagePath2 = writeFile("render2.png", in: env.watchDir)
        let stored2 = try await env.ingestor.ingestFile(at: imagePath2)

        // Two live thumbnails (fake ingest can't produce real ones — simulate).
        writeFile("\(stored.id).jpg", in: env.thumbDir)
        writeFile("\(stored2.id).jpg", in: env.thumbDir)

        // One stale thumbnail with no matching row.
        let stalePath = writeFile("stale-id.jpg", in: env.thumbDir, contents: "0123456789")

        let report = try await env.maintenance.scanOrphanThumbnails()

        #expect(report.orphanCount == 1)
        #expect(report.orphanPaths == [stalePath])
        #expect(report.reclaimableBytes == 10)
        #expect(report.scanned == 3)
    }

    @Test("scan and purge ignore non-jpg files and subdirectories")
    @MainActor
    func scanIgnoresNonJPGAndSubdirs() async throws {
        let env = try await makeEnvironment()

        // A stale thumbnail (the only real orphan).
        let stalePath = writeFile("stale-id.jpg", in: env.thumbDir)
        // Non-.jpg files that should be ignored entirely.
        _ = writeFile("notes.txt", in: env.thumbDir)
        _ = writeFile("stale-id.png", in: env.thumbDir)
        // A subdirectory that happens to be named like a .jpg — must be skipped.
        let subdirPath = (env.thumbDir as NSString).appendingPathComponent("weird.jpg")
        try FileManager.default.createDirectory(atPath: subdirPath, withIntermediateDirectories: true)

        let report = try await env.maintenance.scanOrphanThumbnails()

        #expect(report.orphanPaths == [stalePath])
        #expect(report.scanned == 1)

        let result = await env.maintenance.purgeOrphanThumbnails(report)
        #expect(result.deleted == 1)
        #expect(!FileManager.default.fileExists(atPath: stalePath))
        // Untouched.
        #expect(FileManager.default.fileExists(atPath: subdirPath))
        #expect(FileManager.default.fileExists(atPath: (env.thumbDir as NSString).appendingPathComponent("notes.txt")))
        #expect(FileManager.default.fileExists(atPath: (env.thumbDir as NSString).appendingPathComponent("stale-id.png")))
    }

    @Test("purge deletes exactly the reported paths and leaves matched thumbnails")
    @MainActor
    func purgeDeletesExactlyReportedPaths() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("render.png", in: env.watchDir)
        let stored = try await env.ingestor.ingestFile(at: imagePath)
        let livePath = writeFile("\(stored.id).jpg", in: env.thumbDir)
        let stalePath = writeFile("stale-id.jpg", in: env.thumbDir)

        let report = try await env.maintenance.scanOrphanThumbnails()
        #expect(report.orphanPaths == [stalePath])

        let result = await env.maintenance.purgeOrphanThumbnails(report)

        #expect(result.deleted == 1)
        #expect(result.bytesFreed == report.reclaimableBytes)
        #expect(!FileManager.default.fileExists(atPath: stalePath))
        #expect(FileManager.default.fileExists(atPath: livePath))
    }

    @Test("empty thumbnail directory yields zero orphans without throwing")
    @MainActor
    func emptyDirectoryYieldsZeroOrphans() async throws {
        let env = try await makeEnvironment()
        let report = try await env.maintenance.scanOrphanThumbnails()
        #expect(report.orphanCount == 0)
        #expect(report.scanned == 0)
        #expect(report.reclaimableBytes == 0)

        let result = await env.maintenance.purgeOrphanThumbnails(report)
        #expect(result.deleted == 0)
        #expect(result.bytesFreed == 0)
    }

    @Test("a secured asset's thumbnail is not reported as orphaned")
    @MainActor
    func securedAssetStaysInLiveSet() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("render.png", in: env.watchDir)
        let stored = try await env.ingestor.ingestFile(at: imagePath)

        // secureAsset deletes the row's thumbnail (a thumbnail of a secured
        // image defeats the point) but keeps the row — and thus the id —
        // in the store. Simulate a thumbnail that (re)appeared afterwards
        // under that same id: it must not be flagged as orphaned, because
        // the id is still live.
        _ = try await env.ingestor.secureAsset(stored)
        let thumbPath = writeFile("\(stored.id).jpg", in: env.thumbDir)

        let report = try await env.maintenance.scanOrphanThumbnails()
        #expect(report.orphanCount == 0)
        #expect(FileManager.default.fileExists(atPath: thumbPath))
    }
}
