// RemoteGalleryTransferTests.swift — sending an asset to a remote gallery
// (FDD-remote-galleries §3.3).
//
// A send MOVES: the remote becomes the only copy. So the order is copy →
// verify the digest → relocate the catalog row → delete the local file, and
// every failure leaves the local copy exactly where it was.

import Testing
import Foundation
import ComfyBoxCatalog
@testable import ComfyBoxDesktop

@Suite("RemoteGalleryTransfer")
@MainActor
struct RemoteGalleryTransferTests {

    // MARK: - Fixtures

    private struct World {
        let store: DAMStore
        let catalog: CatalogStore
        let ingestor: AssetIngestor
        let transfer: RemoteGalleryTransfer
        let remote: RemoteGalleryConfig
        let localDir: String
        let dbPath: String
        let driveRoot: String
    }

    private func makeWorld() async throws -> World {
        let base = (NSTemporaryDirectory() as NSString).appendingPathComponent("xfer-\(UUID().uuidString)")
        let localDir = (base as NSString).appendingPathComponent("local")
        let thumbs = (base as NSString).appendingPathComponent("thumbs")
        let drive = (base as NSString).appendingPathComponent("drive")
        for dir in [localDir, thumbs, drive] {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }
        let dbPath = (base as NSString).appendingPathComponent("dam.sqlite3")
        let catalog = try await CatalogStore.open(path: dbPath)
        let store = try await DAMStore.open(path: dbPath)
        let ingestor = AssetIngestor(store: store, watchDirectory: localDir, thumbnailDirectory: thumbs)
        let remote = RemoteGalleryConfig.folder(name: "Test Drive", rootPath: drive, volumeUUID: nil)
        let transfer = RemoteGalleryTransfer(store: store, catalog: catalog, ingestor: ingestor)
        return World(store: store, catalog: catalog, ingestor: ingestor, transfer: transfer,
                     remote: remote, localDir: localDir, dbPath: dbPath, driveRoot: drive)
    }

    /// A real file on disk plus its catalog + DAM rows.
    @discardableResult
    private func makeAsset(_ world: World, id: String, filename: String,
                           bytes: String = "pretend this is a png") async throws -> DAMAsset {
        let path = (world.localDir as NSString).appendingPathComponent(filename)
        FileManager.default.createFile(atPath: path, contents: Data(bytes.utf8))
        let sidecar = ((path as NSString).deletingPathExtension) + ".json"
        FileManager.default.createFile(atPath: sidecar, contents: Data(#"{"prompt":"a barista"}"#.utf8))
        FileManager.default.createFile(atPath: world.ingestor.thumbnailPath(for: id), contents: Data("thumb".utf8))

        let asset = DAMAsset(id: id, kind: "image", filename: filename, absolutePath: path,
                             fileSize: Int64(bytes.utf8.count), prompt: "a barista at a window")
        try await world.store.insertAsset(asset)
        try await world.catalog.upsert(
            CatalogAsset(id: id, filename: filename, absolutePath: path, prompt: "a barista at a window"),
            explicitCollectionIDs: [])
        return asset
    }

    // MARK: - The happy path

    @Test("a send moves the file, its sidecar and its thumbnail, and relocates the row")
    func sendMoves() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "kira-01.png")

        let outcome = await world.transfer.send(assets: [asset], to: world.remote)
        #expect(outcome.sent == ["a1"])
        #expect(outcome.failed.isEmpty)

        // Gone from the Mac.
        #expect(!FileManager.default.fileExists(atPath: asset.absolutePath), "the local copy is deleted")
        #expect(!FileManager.default.fileExists(atPath: world.ingestor.thumbnailPath(for: "a1")))

        // Present on the drive, under its own name, with a sidecar.
        let galleryRoot = try #require(world.remote.galleryRoot)
        let index = try FolderGalleryIndex.load(fromGalleryRoot: galleryRoot)
        let entry = try #require(index.entries.first)
        #expect(entry.assetID == "a1")
        #expect(entry.filename == "kira-01.png")
        #expect(FileManager.default.fileExists(atPath: index.absolutePath(for: entry.mediaPath)))
        #expect(FileManager.default.fileExists(atPath: index.absolutePath(for: entry.sidecarPath)))
        #expect(!FileManager.default.fileExists(atPath: index.absolutePath(for: entry.mediaPath) + ".part"),
                "no temp file is left behind")

        // The catalog knows where it went, and kept the asset.
        let state = try await world.catalog.storageState(of: "a1")
        #expect(state?.storageState == "remote")
        #expect(state?.primaryHost == world.remote.locationHost)
        let locations = try await world.catalog.locations(of: "a1", scope: nil)
        #expect(locations.first?.path == entry.mediaPath)
        #expect(try await world.catalog.asset(id: "a1", visibleTo: nil, ceiling: nil) != nil, "the row survives")
        #expect(try await world.catalog.pendingTransfers().isEmpty, "the journal is clear")
    }

    @Test("the sidecar on the drive carries the prompt, so the drive is readable alone")
    func sidecarCarriesMetadata() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "kira-01.png")
        _ = await world.transfer.send(assets: [asset], to: world.remote)

        let galleryRoot = try #require(world.remote.galleryRoot)
        let index = try FolderGalleryIndex.load(fromGalleryRoot: galleryRoot)
        let entry = try #require(index.entries.first)
        let data = try Data(contentsOf: URL(fileURLWithPath: index.absolutePath(for: entry.sidecarPath)))
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("a barista at a window"))
        #expect(text.contains("kira-01.png"))
    }

    @Test("two assets with the same filename both survive the move")
    func filenameCollision() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let first = try await makeAsset(world, id: "a1", filename: "shot.png", bytes: "first")
        let secondDir = (world.localDir as NSString).appendingPathComponent("sub")
        try FileManager.default.createDirectory(atPath: secondDir, withIntermediateDirectories: true)
        let secondPath = (secondDir as NSString).appendingPathComponent("shot.png")
        FileManager.default.createFile(atPath: secondPath, contents: Data("second".utf8))
        let second = DAMAsset(id: "a2", kind: "image", filename: "shot.png", absolutePath: secondPath)
        try await world.store.insertAsset(second)
        try await world.catalog.upsert(CatalogAsset(id: "a2", filename: "shot.png", absolutePath: secondPath),
                                       explicitCollectionIDs: [])

        let outcome = await world.transfer.send(assets: [first, second], to: world.remote)
        #expect(outcome.sent.count == 2)

        let galleryRoot = try #require(world.remote.galleryRoot)
        let index = try FolderGalleryIndex.load(fromGalleryRoot: galleryRoot)
        #expect(Set(index.entries.map(\.mediaPath)).count == 2, "the second one is not overwritten")
        for entry in index.entries {
            #expect(FileManager.default.fileExists(atPath: index.absolutePath(for: entry.mediaPath)))
        }
    }

    // MARK: - Failure leaves the local copy alone

    @Test("a missing local file fails that asset and touches nothing else")
    func missingLocalFile() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let good = try await makeAsset(world, id: "a1", filename: "good.png")
        let ghost = DAMAsset(id: "a2", kind: "image", filename: "ghost.png",
                             absolutePath: (world.localDir as NSString).appendingPathComponent("ghost.png"))
        try await world.store.insertAsset(ghost)

        let outcome = await world.transfer.send(assets: [ghost, good], to: world.remote)
        #expect(outcome.sent == ["a1"])
        #expect(outcome.failed.map(\.assetID) == ["a2"])
        #expect(try await world.catalog.pendingTransfers().isEmpty, "a failed send leaves no pending work")
        let state = try await world.catalog.storageState(of: "a2")
        #expect(state?.storageState != "remote", "a failed send never claims the asset moved")
    }

    @Test("an unwritable destination fails without deleting anything")
    func unwritableDestination() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "kira-01.png")
        // A remote whose root cannot be created: a file sits where the
        // directory would go.
        let blocked = (NSTemporaryDirectory() as NSString).appendingPathComponent("blocked-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: blocked, contents: Data("not a directory".utf8))
        defer { try? FileManager.default.removeItem(atPath: blocked) }
        let remote = RemoteGalleryConfig.folder(name: "Blocked", rootPath: blocked, volumeUUID: nil)

        let outcome = await world.transfer.send(assets: [asset], to: remote)
        #expect(outcome.sent.isEmpty)
        #expect(outcome.failed.count == 1)
        #expect(FileManager.default.fileExists(atPath: asset.absolutePath), "the local copy is untouched")
    }

    // MARK: - Recovery

    @Test("recovery finishes a transfer that was interrupted after the copy")
    func recoverVerified() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "kira-01.png")

        // Simulate a crash after the media landed and was verified.
        let galleryRoot = try #require(world.remote.galleryRoot)
        var index = try FolderGalleryIndex.create(at: galleryRoot, name: world.remote.name)
        let mediaPath = FolderGalleryIndex.mediaRelativePath(filename: asset.filename, sentAt: Date(), taken: [])
        let destination = index.absolutePath(for: mediaPath)
        try FileManager.default.createDirectory(atPath: (destination as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: asset.absolutePath, toPath: destination)
        try index.append(FolderGalleryIndex.Entry(assetID: "a1", mediaPath: mediaPath,
                                                  sidecarPath: mediaPath + ".json", thumbnailPath: nil,
                                                  kind: "image", filename: asset.filename,
                                                  fileSize: 1, sha256: nil, sentAt: Date()))
        try await world.catalog.beginTransfer(assetID: "a1", remoteID: world.remote.id, remotePath: mediaPath)
        try await world.catalog.setTransferState(assetID: "a1", state: .verified)

        await world.transfer.recoverPending(remotes: [world.remote])

        #expect(!FileManager.default.fileExists(atPath: asset.absolutePath), "the local copy is finally removed")
        #expect(try await world.catalog.storageState(of: "a1")?.storageState == "remote")
        #expect(try await world.catalog.pendingTransfers().isEmpty)
    }

    @Test("recovery of an interrupted copy keeps the local copy and clears the stray part file")
    func recoverCopying() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "kira-01.png")

        let galleryRoot = try #require(world.remote.galleryRoot)
        let index = try FolderGalleryIndex.create(at: galleryRoot, name: world.remote.name)
        let mediaPath = FolderGalleryIndex.mediaRelativePath(filename: asset.filename, sentAt: Date(), taken: [])
        let part = index.absolutePath(for: mediaPath) + ".part"
        try FileManager.default.createDirectory(atPath: (part as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: part, contents: Data("half".utf8))
        try await world.catalog.beginTransfer(assetID: "a1", remoteID: world.remote.id, remotePath: mediaPath)

        await world.transfer.recoverPending(remotes: [world.remote])

        #expect(FileManager.default.fileExists(atPath: asset.absolutePath), "the local copy is authoritative")
        #expect(!FileManager.default.fileExists(atPath: part), "the half-written file is cleared")
        #expect(try await world.catalog.pendingTransfers().isEmpty)
        #expect(try await world.catalog.storageState(of: "a1")?.storageState != "remote")
    }

    @Test("recovery refuses to delete the local copy when the remote file is gone")
    func recoveryRefusesMissingRemote() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "kira-01.png")

        // A journal row that claims a verified copy which is not there.
        let galleryRoot = try #require(world.remote.galleryRoot)
        _ = try FolderGalleryIndex.create(at: galleryRoot, name: world.remote.name)
        try await world.catalog.beginTransfer(assetID: "a1", remoteID: world.remote.id,
                                              remotePath: "media/2026/09/kira-01.png", sha256: "deadbeef")
        try await world.catalog.setTransferState(assetID: "a1", state: .verified)

        await world.transfer.recoverPending(remotes: [world.remote])

        #expect(FileManager.default.fileExists(atPath: asset.absolutePath),
                "a journal row is not proof: the local copy stays")
        #expect(try await world.catalog.pendingTransfers().isEmpty)
        #expect(try await world.catalog.storageState(of: "a1")?.storageState != "remote")
    }

    @Test("recovery refuses when the remote file is there but changed")
    func recoveryRefusesChangedRemote() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "kira-01.png")

        let galleryRoot = try #require(world.remote.galleryRoot)
        let index = try FolderGalleryIndex.create(at: galleryRoot, name: world.remote.name)
        let mediaPath = "media/2026/09/kira-01.png"
        let destination = index.absolutePath(for: mediaPath)
        try FileManager.default.createDirectory(atPath: (destination as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: destination, contents: Data("different bytes".utf8))
        try await world.catalog.beginTransfer(assetID: "a1", remoteID: world.remote.id,
                                              remotePath: mediaPath, sha256: "not-the-digest-of-those-bytes")
        try await world.catalog.setTransferState(assetID: "a1", state: .verified)

        await world.transfer.recoverPending(remotes: [world.remote])
        #expect(FileManager.default.fileExists(atPath: asset.absolutePath), "a checksum mismatch keeps the local copy")
    }

    // MARK: - Vault assets

    @Test("a secured asset leaves the vault and stops being hidden")
    func securedAsset() async throws {
        let world = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: world.dbPath) }
        let asset = try await makeAsset(world, id: "a1", filename: "private.png")
        try await world.store.secureAsset(id: "a1", securedPath: asset.absolutePath, originalPath: asset.absolutePath)
        #expect(try await world.store.securedAssetIds().contains("a1"))

        let outcome = await world.transfer.send(assets: [asset], to: world.remote)
        #expect(outcome.sent == ["a1"])
        #expect(!(try await world.store.securedAssetIds().contains("a1")),
                "otherwise it stays invisible on the remote too")

        let galleryRoot = try #require(world.remote.galleryRoot)
        let index = try FolderGalleryIndex.load(fromGalleryRoot: galleryRoot)
        let entry = try #require(index.entries.first)
        let text = String(decoding: try Data(contentsOf: URL(fileURLWithPath: index.absolutePath(for: entry.sidecarPath))),
                          as: UTF8.self)
        #expect(text.contains("\"sensitive\":true") || text.contains("\"sensitive\" : true"),
                "the sidecar records that this was vault content")
    }
}

// MARK: - Immich sends (stubbed server)

/// A stub with its own state, so this suite cannot race ImmichClientTests
/// (both suites run in parallel; `.serialized` only orders tests WITHIN a suite).
final class ImmichTransferStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: (@Sendable (URLRequest) -> (Int, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let (code, data) = Self.handler?(request) ?? (200, Data("{}".utf8))
        let response = HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    static func reset() { handler = nil }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ImmichTransferStubProtocol.self]
        return URLSession(configuration: config)
    }
}

@Suite("RemoteGalleryTransfer: Immich", .serialized)
@MainActor
struct RemoteGalleryImmichTransferTests {

    private func makeWorld() async throws -> (RemoteGalleryTransfer, DAMStore, CatalogStore, AssetIngestor, String, String) {
        let base = (NSTemporaryDirectory() as NSString).appendingPathComponent("imx-\(UUID().uuidString)")
        let localDir = (base as NSString).appendingPathComponent("local")
        let thumbs = (base as NSString).appendingPathComponent("thumbs")
        for dir in [localDir, thumbs] {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }
        let dbPath = (base as NSString).appendingPathComponent("dam.sqlite3")
        let catalog = try await CatalogStore.open(path: dbPath)
        let store = try await DAMStore.open(path: dbPath)
        let ingestor = AssetIngestor(store: store, watchDirectory: localDir, thumbnailDirectory: thumbs)
        let transfer = RemoteGalleryTransfer(store: store, catalog: catalog, ingestor: ingestor) { remote in
            ImmichClient(baseURL: remote.normalizedBaseURL ?? "", apiKey: "test-key",
                         session: ImmichTransferStubProtocol.session())
        }
        return (transfer, store, catalog, ingestor, localDir, dbPath)
    }

    private func makeAsset(_ store: DAMStore, _ catalog: CatalogStore, dir: String, id: String) async throws -> DAMAsset {
        let path = (dir as NSString).appendingPathComponent("\(id).png")
        FileManager.default.createFile(atPath: path, contents: Data("bytes".utf8))
        let asset = DAMAsset(id: id, kind: "image", filename: "\(id).png", absolutePath: path)
        try await store.insertAsset(asset)
        try await catalog.upsert(CatalogAsset(id: id, filename: "\(id).png", absolutePath: path),
                                 explicitCollectionIDs: [])
        return asset
    }

    private let remote = RemoteGalleryConfig.immich(name: "Immich", baseURL: "http://10.0.100.232:2283", albumName: "ComfyBox")

    @Test("a confirmed upload moves the asset and records its Immich id")
    func uploadMoves() async throws {
        ImmichTransferStubProtocol.reset()
        ImmichTransferStubProtocol.handler = { request in
            let path = request.url?.path ?? ""
            if path == "/api/server/version" { return (200, Data(#"{"major":2,"minor":3,"patch":1}"#.utf8)) }
            if path == "/api/assets", request.httpMethod == "POST" {
                return (201, Data(#"{"id":"immich-9","status":"created"}"#.utf8))
            }
            if path == "/api/assets/immich-9" { return (200, Data(#"{"id":"immich-9"}"#.utf8)) }
            if path == "/api/albums", request.httpMethod == "POST" { return (201, Data(#"{"id":"album-1"}"#.utf8)) }
            return (200, Data("{}".utf8))
        }
        let (transfer, store, catalog, _, dir, dbPath) = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        let asset = try await makeAsset(store, catalog, dir: dir, id: "a1")

        let outcome = await transfer.send(assets: [asset], to: remote)
        #expect(outcome.sent == ["a1"])
        #expect(outcome.albumID == "album-1")
        #expect(!FileManager.default.fileExists(atPath: asset.absolutePath), "the local copy is gone")
        let locations = try await catalog.locations(of: "a1", scope: nil)
        #expect(locations.first?.path == "immich://immich-9")
        #expect(try await catalog.pendingTransfers().isEmpty)
    }

    @Test("an upload the server cannot confirm keeps the local copy")
    func unconfirmedKeepsLocal() async throws {
        ImmichTransferStubProtocol.reset()
        ImmichTransferStubProtocol.handler = { request in
            let path = request.url?.path ?? ""
            if path == "/api/server/version" { return (200, Data(#"{"major":2,"minor":3,"patch":1}"#.utf8)) }
            if path == "/api/assets", request.httpMethod == "POST" {
                return (201, Data(#"{"id":"immich-9","status":"created"}"#.utf8))
            }
            // The verification says it is not there.
            return (404, Data(#"{"message":"not found"}"#.utf8))
        }
        let (transfer, store, catalog, _, dir, dbPath) = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        let asset = try await makeAsset(store, catalog, dir: dir, id: "a1")

        let outcome = await transfer.send(assets: [asset], to: remote)
        #expect(outcome.sent.isEmpty)
        #expect(outcome.failed.count == 1)
        #expect(FileManager.default.fileExists(atPath: asset.absolutePath), "nothing is deleted on an unconfirmed upload")
        #expect(try await catalog.pendingTransfers().isEmpty)
    }

    @Test("the upload carries the metadata field Immich requires")
    func metadataFieldPresent() async throws {
        ImmichTransferStubProtocol.reset()
        nonisolated(unsafe) var uploadBody = ""
        ImmichTransferStubProtocol.handler = { request in
            let path = request.url?.path ?? ""
            if path == "/api/server/version" { return (200, Data(#"{"major":2,"minor":3,"patch":1}"#.utf8)) }
            if path == "/api/assets", request.httpMethod == "POST" {
                if let stream = request.httpBodyStream {
                    stream.open()
                    var data = Data()
                    let size = 4096
                    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
                    while stream.hasBytesAvailable {
                        let read = stream.read(buffer, maxLength: size)
                        if read <= 0 { break }
                        data.append(buffer, count: read)
                    }
                    buffer.deallocate(); stream.close()
                    uploadBody = String(decoding: data, as: UTF8.self)
                }
                return (201, Data(#"{"id":"immich-9","status":"created"}"#.utf8))
            }
            return (200, Data(#"{"id":"immich-9"}"#.utf8))
        }
        let (transfer, store, catalog, _, dir, dbPath) = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        let asset = try await makeAsset(store, catalog, dir: dir, id: "a1")
        _ = await transfer.send(assets: [asset], to: remote)
        #expect(uploadBody.contains("name=\"metadata\""),
                "AssetMediaCreateDto requires it in Immich 2.3.1 — without it the upload is rejected")
    }

    @Test("a checksum the server does not match keeps the local copy")
    func checksumMismatchKeepsLocal() async throws {
        ImmichTransferStubProtocol.reset()
        ImmichTransferStubProtocol.handler = { request in
            let path = request.url?.path ?? ""
            if path == "/api/server/version" { return (200, Data(#"{"major":2,"minor":3,"patch":1}"#.utf8)) }
            if path == "/api/assets", request.httpMethod == "POST" {
                return (201, Data(#"{"id":"immich-9","status":"created"}"#.utf8))
            }
            // The server holds SOMETHING under that id, but not these bytes.
            return (200, Data(#"{"id":"immich-9","checksum":"c29tZXRoaW5nIGVsc2U="}"#.utf8))
        }
        let (transfer, store, catalog, _, dir, dbPath) = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        let asset = try await makeAsset(store, catalog, dir: dir, id: "a1")

        let outcome = await transfer.send(assets: [asset], to: remote)
        #expect(outcome.sent.isEmpty)
        #expect(FileManager.default.fileExists(atPath: asset.absolutePath))
    }

    @Test("a server version this build does not speak stops the whole send")
    func versionGateStopsSend() async throws {
        ImmichTransferStubProtocol.reset()
        ImmichTransferStubProtocol.handler = { _ in (200, Data(#"{"major":9,"minor":0,"patch":0}"#.utf8)) }
        let (transfer, store, catalog, _, dir, dbPath) = try await makeWorld()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        let asset = try await makeAsset(store, catalog, dir: dir, id: "a1")

        let outcome = await transfer.send(assets: [asset], to: remote)
        #expect(outcome.sent.isEmpty)
        #expect(outcome.failed.first?.reason.contains("not supported") == true)
        #expect(FileManager.default.fileExists(atPath: asset.absolutePath))
    }
}
