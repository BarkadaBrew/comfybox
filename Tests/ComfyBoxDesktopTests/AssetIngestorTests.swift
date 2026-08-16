// AssetIngestorTests.swift — Tests for AssetIngestor delete pipeline

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("AssetIngestor")
struct AssetIngestorTests {
    /// Create an isolated temp environment: watch dir, thumbnail dir, and store.
    @MainActor
    private func makeEnvironment() async throws -> (ingestor: AssetIngestor, store: DAMStore, watchDir: String, dbPath: String) {
        let base = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ingestor-test-\(UUID().uuidString)")
        let watchDir = (base as NSString).appendingPathComponent("output")
        let thumbDir = (base as NSString).appendingPathComponent("thumbnails")
        try FileManager.default.createDirectory(atPath: watchDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: thumbDir, withIntermediateDirectories: true)
        let dbPath = (base as NSString).appendingPathComponent("dam.sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let ingestor = AssetIngestor(store: store, watchDirectory: watchDir, thumbnailDirectory: thumbDir)
        return (ingestor, store, watchDir, dbPath)
    }

    /// Write a minimal file with the given name into the directory.
    private func writeFile(_ name: String, in dir: String, contents: String = "not-a-real-png") -> String {
        let path = (dir as NSString).appendingPathComponent(name)
        FileManager.default.createFile(atPath: path, contents: Data(contents.utf8))
        return path
    }

    @Test("deleteAsset removes file, sidecar, thumbnail, and database row")
    @MainActor
    func deleteRemovesEverything() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("render.png", in: env.watchDir)
        let sidecarPath = writeFile("render.json", in: env.watchDir, contents: #"{"prompt":"test"}"#)

        let stored = try await env.ingestor.ingestFile(at: imagePath)
        // Simulate a generated thumbnail (the fake PNG can't produce a real one).
        let thumbPath = env.ingestor.thumbnailPath(for: stored.id)
        FileManager.default.createFile(atPath: thumbPath, contents: Data("thumb".utf8))

        try await env.ingestor.deleteAsset(stored)

        let fm = FileManager.default
        #expect(!fm.fileExists(atPath: imagePath))
        #expect(!fm.fileExists(atPath: sidecarPath))
        #expect(!fm.fileExists(atPath: thumbPath))
        let count = try await env.store.assetCount()
        #expect(count == 0)
    }

    @Test("deleteAsset tolerates a file already missing on disk")
    @MainActor
    func deleteMissingFile() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("gone.png", in: env.watchDir)
        let stored = try await env.ingestor.ingestFile(at: imagePath)
        try FileManager.default.removeItem(atPath: imagePath)

        try await env.ingestor.deleteAsset(stored)

        let count = try await env.store.assetCount()
        #expect(count == 0)
    }

    @Test("a deleted path can be ingested again as a fresh asset")
    @MainActor
    func deletedPathReingests() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("again.png", in: env.watchDir)
        let first = try await env.ingestor.ingestFile(at: imagePath)
        try await env.ingestor.deleteAsset(first)

        _ = writeFile("again.png", in: env.watchDir, contents: "second-render")
        let second = try await env.ingestor.ingestFile(at: imagePath)

        #expect(second.id != first.id)
        let count = try await env.store.assetCount()
        #expect(count == 1)
    }

    // MARK: - Folder import

    @Test("importFolder ingests all images with metadata merged from sidecars")
    @MainActor
    func importFolder() async throws {
        let env = try await makeEnvironment()
        // A source folder unrelated to the watch dir, with images, a sidecar,
        // a nested subfolder, and a non-image to ignore.
        let source = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("import-src-\(UUID().uuidString)")
        let nested = (source as NSString).appendingPathComponent("sub")
        try FileManager.default.createDirectory(atPath: nested, withIntermediateDirectories: true)
        _ = writeFile("a.png", in: source)
        _ = writeFile("b.jpg", in: source)
        _ = writeFile("notes.txt", in: source)          // ignored
        _ = writeFile("c.png", in: nested)               // recursive
        // Sidecar metadata for a.png, merged as if rendered locally.
        _ = writeFile("a.json", in: source,
                      contents: #"{"prompt": "imported sunset", "seed": 7, "steps": 20}"#)

        var progressSeen: [Int] = []
        let summary = try await env.ingestor.importFolder(at: source) { done, total in
            progressSeen.append(done)
            #expect(total == 3)
        }

        #expect(summary.imported == 3)
        #expect(summary.total == 3)
        #expect(progressSeen.last == 3)

        let count = try await env.store.assetCount()
        #expect(count == 3)

        // Metadata from the sidecar came through.
        let hits = try await env.store.searchPrompts(query: "sunset")
        #expect(hits.count == 1)
        #expect(hits.first?.seed == 7)
        #expect(hits.first?.steps == 20)
    }

    @Test("importFolder is idempotent — re-importing adds nothing")
    @MainActor
    func importFolderIdempotent() async throws {
        let env = try await makeEnvironment()
        let source = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("import-src-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: source, withIntermediateDirectories: true)
        _ = writeFile("x.png", in: source)
        _ = writeFile("y.png", in: source)

        let first = try await env.ingestor.importFolder(at: source)
        #expect(first.imported == 2)
        let second = try await env.ingestor.importFolder(at: source)
        #expect(second.imported == 0)
        #expect(second.skipped == 2)

        let count = try await env.store.assetCount()
        #expect(count == 2)
    }

    // MARK: - Deleting a file this Mac does not own

    /// The local case, unchanged: where trashing is unavailable (a sandboxed
    /// test runner, a volume with no Trash) a file on THIS Mac may still be
    /// removed outright. That is the fallback's original and only purpose.
    @Test("a failed trash on a local file still removes it")
    func localFallbackStillRemoves() throws {
        var removed: [String] = []
        try AssetIngestor.trashOrRemove(
            atPath: "/tmp/local.png",
            fileExists: { _ in true },
            trash: { _ in throw CocoaError(.fileWriteUnknown) },
            remove: { removed.append($0) },
            isLocal: { _ in true })
        #expect(removed == ["/tmp/local.png"])
    }

    /// THE ONE THAT MATTERS. Roughly 1,300 of the live catalog's 2,994 rows name
    /// a path under /Volumes/todd — an smbfs mount of the server. `trashItem`
    /// does not work there, which is the NORMAL outcome, and the old fallback
    /// answered that by calling `removeItem`: a permanent, unrecoverable delete
    /// of production media, from a dialog whose button reads "Move to Trash".
    @Test("a failed trash on a server file deletes nothing and says so")
    func remoteFileIsNeverHardDeleted() {
        var removed: [String] = []
        let path = "/Volumes/todd/.kira/studio/video/clip.mp4"
        #expect(throws: AssetIngestor.DeletionRefusal.notOnThisMac(path)) {
            try AssetIngestor.trashOrRemove(
                atPath: path,
                fileExists: { _ in true },
                trash: { _ in throw CocoaError(.fileWriteUnknown) },
                remove: { removed.append($0) },
                isLocal: { _ in false })
        }
        #expect(removed.isEmpty, "a file on another host was permanently deleted")
    }

    /// A successful trash never reaches either branch.
    @Test("a file that trashes cleanly is not removed twice")
    func successfulTrashDoesNotFallThrough() throws {
        var removed: [String] = []
        var trashed: [String] = []
        try AssetIngestor.trashOrRemove(
            atPath: "/Volumes/todd/x.png",
            fileExists: { _ in true },
            trash: { trashed.append($0) },
            remove: { removed.append($0) },
            isLocal: { _ in false })
        #expect(trashed == ["/Volumes/todd/x.png"])
        #expect(removed.isEmpty)
    }

    /// The locality check itself. `/tmp` is on this Mac; an unresolvable path
    /// under /Volumes is assumed not to be, so the refusal stands on a guess
    /// rather than a hard delete proceeding on one.
    @Test("volume locality fails closed")
    func localityFailsClosed() {
        #expect(AssetIngestor.isOnThisMac(NSTemporaryDirectory()))
        #expect(AssetIngestor.isOnThisMac("/tmp"))
        #expect(!AssetIngestor.isOnThisMac("/Volumes/definitely-not-mounted-\(UUID().uuidString)/x.png"))
    }

    /// And the whole path: `deleteAsset` must leave the database row alone when
    /// the file survives, so the grid keeps pointing at media that still exists.
    @Test("deleteAsset keeps the row when the file could not be trashed")
    @MainActor
    func deleteAbortsRatherThanOrphanTheRow() async throws {
        let env = try await makeEnvironment()
        let imagePath = writeFile("server.png", in: env.watchDir)
        let stored = try await env.ingestor.ingestFile(at: imagePath)

        // Re-point the row at a path that reads as another host's disk.
        let remotePath = "/Volumes/todd/.kira/studio/gallery/server.png"
        let remoteAsset = DAMAsset(
            id: stored.id, kind: stored.kind, filename: stored.filename,
            absolutePath: remotePath, fileSize: stored.fileSize,
            createdAt: stored.createdAt, modifiedAt: stored.modifiedAt,
            ingestedAt: stored.ingestedAt, prompt: stored.prompt)

        // `fileExists` is false for an unmounted path, so drive the decision
        // directly — the store row is what this test is about.
        #expect(throws: AssetIngestor.DeletionRefusal.self) {
            try AssetIngestor.trashOrRemove(
                atPath: remoteAsset.absolutePath,
                fileExists: { _ in true },
                trash: { _ in throw CocoaError(.fileWriteUnknown) },
                remove: { _ in Issue.record("permanently deleted a server file") },
                isLocal: { _ in false })
        }
        #expect(try await env.store.assetCount() == 1)
    }

    // MARK: - regenerateAllThumbnails

    @Test("regenerateAllThumbnails overwrites an existing (stale) thumbnail")
    @MainActor
    func regenerateAllThumbnailsOverwritesStale() async throws {
        let env = try await makeEnvironment()
        let imagePath = (env.watchDir as NSString).appendingPathComponent("real.png")
        #expect(TestData.writeRealPNG(at: imagePath, width: 32, height: 32))

        let stored = try await env.ingestor.ingestFile(at: imagePath)
        let thumbPath = env.ingestor.thumbnailPath(for: stored.id)

        // Overwrite whatever ingestFile produced with a sentinel that a
        // fixed implementation must clear before regenerating — this is
        // the @testable guard on the generateThumbnail early-return trap.
        let sentinel = Data("stale".utf8)
        try sentinel.write(to: URL(fileURLWithPath: thumbPath))
        #expect(FileManager.default.contents(atPath: thumbPath) == sentinel)

        let summary = await env.ingestor.regenerateAllThumbnails(for: [stored])

        #expect(summary.total == 1)
        #expect(summary.regenerated == 1)
        #expect(summary.missingSource == 0)
        #expect(summary.failed == 0)

        let finalContents = FileManager.default.contents(atPath: thumbPath)
        #expect(finalContents != sentinel)
        #expect((finalContents?.count ?? 0) > 0)
    }

    @Test("regenerateAllThumbnails counts a missing source without throwing and keeps processing others")
    @MainActor
    func regenerateAllThumbnailsMissingSource() async throws {
        let env = try await makeEnvironment()

        let goodPath = (env.watchDir as NSString).appendingPathComponent("good.png")
        #expect(TestData.writeRealPNG(at: goodPath, width: 16, height: 16))
        let goodAsset = try await env.ingestor.ingestFile(at: goodPath)

        let missingPath = (env.watchDir as NSString).appendingPathComponent("missing.png")
        #expect(TestData.writeRealPNG(at: missingPath, width: 16, height: 16))
        let missingAsset = try await env.ingestor.ingestFile(at: missingPath)
        try FileManager.default.removeItem(atPath: missingPath)

        let summary = await env.ingestor.regenerateAllThumbnails(for: [goodAsset, missingAsset])

        #expect(summary.total == 2)
        #expect(summary.missingSource == 1)
        #expect(summary.regenerated == 1)
        #expect(summary.failed == 0)

        let goodThumb = env.ingestor.thumbnailPath(for: goodAsset.id)
        let goodSize = (try? FileManager.default.attributesOfItem(atPath: goodThumb)[.size] as? Int) ?? 0
        #expect((goodSize ?? 0) > 0)
    }

    @Test("regenerateAllThumbnails reports progress ending at done == total")
    @MainActor
    func regenerateAllThumbnailsProgress() async throws {
        let env = try await makeEnvironment()

        var assets: [DAMAsset] = []
        for i in 0..<5 {
            let path = (env.watchDir as NSString).appendingPathComponent("img\(i).png")
            #expect(TestData.writeRealPNG(at: path, width: 8, height: 8))
            assets.append(try await env.ingestor.ingestFile(at: path))
        }

        var doneValues: [Int] = []
        let summary = await env.ingestor.regenerateAllThumbnails(for: assets) { done, total in
            doneValues.append(done)
            #expect(total == 5)
        }

        #expect(summary.total == 5)
        #expect(doneValues.count == 5)
        #expect(doneValues.sorted() == [1, 2, 3, 4, 5])
        #expect(doneValues.last == 5)
    }
}
