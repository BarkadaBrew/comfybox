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
}
