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

@Suite("CatalogBrowser: scoped to one remote")
@MainActor
struct CatalogBrowserRemoteScopeTests {

    @Test("the tab shows one remote's contents and nothing else")
    func scopedToOneRemote() async throws {
        let dbPath = (NSTemporaryDirectory() as NSString).appendingPathComponent("sc-\(UUID().uuidString).sqlite3")
        let store = try await CatalogStore.open(path: dbPath)
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        // One asset on the drive, one still local.
        let driveRoot = (NSTemporaryDirectory() as NSString).appendingPathComponent("drv-\(UUID().uuidString)")
        let mediaDir = (driveRoot as NSString).appendingPathComponent("media")
        try FileManager.default.createDirectory(atPath: mediaDir, withIntermediateDirectories: true)
        let onDrive = (mediaDir as NSString).appendingPathComponent("moved.png")
        FileManager.default.createFile(atPath: onDrive, contents: Data("bytes".utf8))
        defer { try? FileManager.default.removeItem(atPath: driveRoot) }

        let localPath = (NSTemporaryDirectory() as NSString).appendingPathComponent("still-\(UUID().uuidString).png")
        FileManager.default.createFile(atPath: localPath, contents: Data("bytes".utf8))
        defer { try? FileManager.default.removeItem(atPath: localPath) }

        try await store.upsert(CatalogAsset(id: "moved", filename: "moved.png", absolutePath: "/gone/moved.png"),
                               explicitCollectionIDs: [])
        try await store.relocateAsset(id: "moved", host: "remote:r1", path: "media/moved.png")
        try await store.upsert(CatalogAsset(id: "local", filename: "still.png", absolutePath: localPath),
                               explicitCollectionIDs: [])

        let browser = CatalogBrowser(store: store)
        browser.remoteGalleryRoots = ["remote:r1": driveRoot]
        browser.restrictToRemoteHost = "remote:r1"
        await browser.apply(filter: CatalogQuery(limit: 50))

        #expect(browser.items.map(\.id) == ["moved"], "only the drive's contents")
        #expect(browser.remoteHostByAsset["moved"] == "remote:r1", "and the grid can label it")
    }

    @Test("ids on a host come back newest first, and a host with nothing is empty")
    func idsOnHost() async throws {
        let dbPath = (NSTemporaryDirectory() as NSString).appendingPathComponent("ids-\(UUID().uuidString).sqlite3")
        let store = try await CatalogStore.open(path: dbPath)
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        try await store.upsert(CatalogAsset(id: "a", filename: "a.png", absolutePath: "/tmp/a.png",
                                            createdAt: Date(timeIntervalSince1970: 1_000)),
                               explicitCollectionIDs: [])
        try await store.upsert(CatalogAsset(id: "b", filename: "b.png", absolutePath: "/tmp/b.png",
                                            createdAt: Date(timeIntervalSince1970: 2_000)),
                               explicitCollectionIDs: [])
        try await store.relocateAsset(id: "a", host: "remote:r1", path: "media/a.png")
        try await store.relocateAsset(id: "b", host: "remote:r1", path: "media/b.png")

        #expect(try await store.assetIDs(onHost: "remote:r1") == ["b", "a"])
        #expect(try await store.assetIDs(onHost: "remote:nope").isEmpty)
    }
}
