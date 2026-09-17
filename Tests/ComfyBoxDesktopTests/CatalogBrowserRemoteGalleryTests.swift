// CatalogBrowserRemoteGalleryTests.swift — what the gallery shows when an
// asset lives on a remote gallery (FDD-remote-galleries §3.5).

import Testing
import Foundation
import ComfyBoxCatalog
@testable import ComfyBoxDesktop

// MARK: - Remote galleries (FDD-remote-galleries §3.5)

@Suite("CatalogBrowser: remote galleries")
@MainActor
struct CatalogBrowserRemoteGalleryTests {

    private func makeStore() async throws -> (CatalogStore, String) {
        let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("cb-\(UUID().uuidString).sqlite3")
        return (try await CatalogStore.open(path: path), path)
    }

    @Test("an asset on an attached drive shows, reading from the drive")
    func attachedDriveShows() async throws {
        let (store, dbPath) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        let driveRoot = (NSTemporaryDirectory() as NSString).appendingPathComponent("drive-\(UUID().uuidString)")
        let mediaDir = (driveRoot as NSString).appendingPathComponent("media/2026/09")
        try FileManager.default.createDirectory(atPath: mediaDir, withIntermediateDirectories: true)
        let onDrive = (mediaDir as NSString).appendingPathComponent("kira.png")
        FileManager.default.createFile(atPath: onDrive, contents: Data("bytes".utf8))
        defer { try? FileManager.default.removeItem(atPath: driveRoot) }

        try await store.upsert(CatalogAsset(id: "a1", filename: "kira.png", absolutePath: "/gone/kira.png"),
                               explicitCollectionIDs: [])
        try await store.relocateAsset(id: "a1", host: "remote:r1", path: "media/2026/09/kira.png")

        let browser = CatalogBrowser(store: store)
        browser.remoteGalleryRoots = ["remote:r1": driveRoot]
        await browser.apply(filter: CatalogQuery(limit: 10))
        #expect(browser.items.map(\.id) == ["a1"])
        #expect(browser.localPath(forID: "a1") == onDrive, "it reads from the drive, not the old local path")
        #expect(browser.localPath(forID: "a1") != nil, "an attached drive is as good as local")
    }

    @Test("an asset on a drive that is not attached is hidden, not broken")
    func absentDriveHides() async throws {
        let (store, dbPath) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        try await store.upsert(CatalogAsset(id: "a1", filename: "kira.png", absolutePath: "/gone/kira.png"),
                               explicitCollectionIDs: [])
        try await store.relocateAsset(id: "a1", host: "remote:r1", path: "media/2026/09/kira.png")

        let browser = CatalogBrowser(store: store)
        browser.remoteGalleryRoots = [:]   // nothing attached
        await browser.apply(filter: CatalogQuery(limit: 10))
        #expect(browser.items.isEmpty, "Todd's choice: hide it until the drive is back")
    }

    @Test("a local asset is unaffected by remote configuration")
    func localUnaffected() async throws {
        let (store, dbPath) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("local-\(UUID().uuidString).png")
        FileManager.default.createFile(atPath: path, contents: Data("bytes".utf8))
        defer { try? FileManager.default.removeItem(atPath: path) }
        try await store.upsert(CatalogAsset(id: "a1", filename: "local.png", absolutePath: path),
                               explicitCollectionIDs: [])

        let browser = CatalogBrowser(store: store)
        browser.remoteGalleryRoots = ["remote:r1": "/Volumes/NoSuchDrive"]
        await browser.apply(filter: CatalogQuery(limit: 10))
        #expect(browser.items.map(\.id) == ["a1"])
        #expect(browser.localPath(forID: "a1") == path)
    }
}

@Suite("Remote galleries: a file that comes back")
@MainActor
struct RemoteGalleryReadoptionTests {

    @Test("a moved file that reappears on this Mac is local again")
    func reappearsLocally() async throws {
        let dbPath = (NSTemporaryDirectory() as NSString).appendingPathComponent("re-\(UUID().uuidString).sqlite3")
        let store = try await CatalogStore.open(path: dbPath)
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        let localPath = (NSTemporaryDirectory() as NSString).appendingPathComponent("back-\(UUID().uuidString).png")
        FileManager.default.createFile(atPath: localPath, contents: Data("bytes".utf8))
        defer { try? FileManager.default.removeItem(atPath: localPath) }

        try await store.upsert(CatalogAsset(id: "a1", filename: "back.png", absolutePath: localPath),
                               explicitCollectionIDs: [])
        try await store.relocateAsset(id: "a1", host: "remote:r1", path: "media/2026/09/back.png")

        let browser = CatalogBrowser(store: store)
        browser.remoteGalleryRoots = [:]        // the drive is not attached
        await browser.apply(filter: CatalogQuery(limit: 10))

        #expect(browser.items.map(\.id) == ["a1"], "the file is here, so it shows")
        #expect(browser.localPath(forID: "a1") == localPath)
        #expect(try await store.storageState(of: "a1")?.storageState == "local",
                "and the catalog is corrected, so it is not hidden again next time")
    }
}
