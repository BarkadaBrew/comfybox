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
}
