// AssetSecurityTests.swift — Securing sensitive DAM assets

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("DAM asset security")
struct AssetSecurityTests {
    private func makeStore() async throws -> (DAMStore, String) {
        let dbPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        return (try await DAMStore.open(path: dbPath), dbPath)
    }

    @Test("secure and unsecure round-trip updates path and membership")
    func secureRoundTrip() async throws {
        let (store, dbPath) = try await makeStore()
        try await store.insertAsset(TestData.makeAsset(id: "s1", filename: "s1.png"))

        try await store.secureAsset(
            id: "s1",
            securedPath: "/secure/s1.png",
            originalPath: "/tmp/s1.png"
        )
        let securedIds = try await store.securedAssetIds()
        #expect(securedIds == ["s1"])
        let moved = try await store.fetchAsset(byPath: "/secure/s1.png")
        #expect(moved?.id == "s1")

        let restoredPath = try await store.unsecureAsset(id: "s1")
        #expect(restoredPath == "/tmp/s1.png")
        let after = try await store.securedAssetIds()
        #expect(after.isEmpty)
        let back = try await store.fetchAsset(byPath: "/tmp/s1.png")
        #expect(back?.id == "s1")
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("secured assets are excluded from prompt search and history")
    func searchExclusion() async throws {
        let (store, dbPath) = try await makeStore()
        try await store.insertAsset(TestData.makeAsset(
            id: "vis", filename: "vis.png", prompt: "sunlit meadow"))
        try await store.insertAsset(DAMAsset(
            id: "sec", filename: "sec.png", absolutePath: "/tmp/sec.png",
            prompt: "sunlit secret garden"))

        try await store.secureAsset(id: "sec", securedPath: "/secure/sec.png", originalPath: "/tmp/sec.png")

        let hits = try await store.searchPrompts(query: "sunlit")
        #expect(hits.map(\.id) == ["vis"])

        let history = try await store.promptHistory(limit: 10)
        #expect(history.map(\.prompt) == ["sunlit meadow"])

        // Unsecuring brings it back into search.
        _ = try await store.unsecureAsset(id: "sec")
        let after = try await store.searchPrompts(query: "sunlit")
        #expect(Set(after.map(\.id)) == ["vis", "sec"])
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("unsecuring an unknown id returns nil")
    func unknownUnsecure() async throws {
        let (store, dbPath) = try await makeStore()
        let restored = try await store.unsecureAsset(id: "nope")
        #expect(restored == nil)
        try? FileManager.default.removeItem(atPath: dbPath)
    }

    @Test("ingestor secures an asset: file + sidecar moved, thumbnail gone")
    func ingestorSecure() async throws {
        let tmp = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("secure-test-\(UUID().uuidString)")
        let watchDir = (tmp as NSString).appendingPathComponent("out")
        let thumbDir = (tmp as NSString).appendingPathComponent("thumbs")
        let secureDir = (tmp as NSString).appendingPathComponent("vault")
        try FileManager.default.createDirectory(atPath: watchDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: thumbDir, withIntermediateDirectories: true)

        let dbPath = (tmp as NSString).appendingPathComponent("dam.sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let ingestor = await AssetIngestor(store: store, watchDirectory: watchDir, thumbnailDirectory: thumbDir)
        await MainActor.run { ingestor.secureDirectory = secureDir }

        // A tiny valid PNG + sidecar + fake thumbnail.
        let imagePath = (watchDir as NSString).appendingPathComponent("secret.png")
        let png = Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")!
        try png.write(to: URL(fileURLWithPath: imagePath))
        let sidecarPath = (watchDir as NSString).appendingPathComponent("secret.json")
        try Data(#"{"prompt": "hidden"}"#.utf8).write(to: URL(fileURLWithPath: sidecarPath))

        let asset = try await ingestor.ingestFile(at: imagePath)
        let thumbPath = await MainActor.run { ingestor.thumbnailPath(for: asset.id) }
        #expect(FileManager.default.fileExists(atPath: thumbPath))

        let secured = try await ingestor.secureAsset(asset)
        #expect(!FileManager.default.fileExists(atPath: imagePath))
        #expect(!FileManager.default.fileExists(atPath: sidecarPath))
        #expect(!FileManager.default.fileExists(atPath: thumbPath))
        #expect(FileManager.default.fileExists(atPath: secured.absolutePath))
        #expect(secured.absolutePath.hasPrefix(secureDir))
        let ids = try await store.securedAssetIds()
        #expect(ids == [asset.id])

        // Unsecure restores the file (and sidecar) and regenerates a thumbnail.
        let restored = try await ingestor.unsecureAsset(secured)
        #expect(restored.absolutePath == imagePath)
        #expect(FileManager.default.fileExists(atPath: imagePath))
        #expect(FileManager.default.fileExists(atPath: sidecarPath))
        #expect(FileManager.default.fileExists(atPath: thumbPath))
        let after = try await store.securedAssetIds()
        #expect(after.isEmpty)

        try? FileManager.default.removeItem(atPath: tmp)
    }
}
