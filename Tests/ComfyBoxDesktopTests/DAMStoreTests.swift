// DAMStoreTests.swift — Tests for DAMStore actor

import Testing
import Foundation
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
}
