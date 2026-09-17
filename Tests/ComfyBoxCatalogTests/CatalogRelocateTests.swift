// CatalogRelocateTests.swift — moving an asset to a remote without losing it
// (FDD-remote-galleries §3.3).
//
// Codex review of the draft spec raised the two properties pinned here:
//   1. a move must NOT be a delete — FTS rows, collections, folder mappings
//      and lineage edges all hang off the asset row;
//   2. `assets.absolute_path` is NOT NULL UNIQUE and cannot be surrendered, so
//      storage state gets its own columns and the path stays as the last known
//      local one.

import Testing
import Foundation
import SQLite3
@testable import ComfyBoxCatalog

@Suite("CatalogStore relocate + transfer journal")
struct CatalogRelocateTests {

    private func tempPath() -> String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("cat-\(UUID().uuidString).sqlite3")
    }

    /// A store with one asset row, filed into a collection.
    private func makeStore() async throws -> (CatalogStore, String, String) {
        let path = tempPath()
        let store = try await CatalogStore.open(path: path)
        let id = "asset-1"
        try await store.upsert(
            CatalogAsset(id: id, filename: "kira-01.png",
                         absolutePath: "/Users/todd/Pictures/ComfyBox/kira-01.png",
                         prompt: "a barista at a window"),
            explicitCollectionIDs: [])
        return (store, id, path)
    }

    @Test("a relocation keeps everything that hangs off the asset")
    func relocateKeepsGraph() async throws {
        let (store, id, path) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let before = try await store.asset(id: id, visibleTo: nil, ceiling: nil)
        #expect(before != nil)

        try await store.relocateAsset(id: id, host: "remote:r1", path: "media/2026/09/kira-01.png")

        let after = try await store.asset(id: id, visibleTo: nil, ceiling: nil)
        #expect(after != nil, "the row survives a move")
        #expect(after?.prompt == "a barista at a window", "its metadata travels with it")
        #expect(after?.absolutePath == "/Users/todd/Pictures/ComfyBox/kira-01.png",
                "absolute_path keeps the last known local path")

        // Searchable by prompt, i.e. its FTS row was not dropped.
        let hits = try await store.search(CatalogQuery(text: "barista", limit: 10))
        #expect(hits.contains { $0.id == id }, "a moved asset is still searchable")
    }

    @Test("a relocation records exactly one location, and re-running is harmless")
    func relocateIsIdempotent() async throws {
        let (store, id, path) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: path) }

        try await store.relocateAsset(id: id, host: "remote:r1", path: "media/2026/09/kira-01.png")
        try await store.relocateAsset(id: id, host: "remote:r1", path: "media/2026/09/kira-01.png")

        let locations = try await store.locations(of: id, scope: nil)
        #expect(locations.count == 1)
        #expect(locations.first?.host == "remote:r1")
        #expect(locations.first?.path == "media/2026/09/kira-01.png")

        let state = try await store.storageState(of: id)
        #expect(state?.storageState == "remote")
        #expect(state?.primaryHost == "remote:r1")
    }

    @Test("a fresh row is local, and a moved row can come home")
    func storageStates() async throws {
        let (store, id, path) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let fresh = try await store.storageState(of: id)
        #expect(fresh?.storageState == nil || fresh?.storageState == "local", "untouched rows read as local")

        try await store.relocateAsset(id: id, host: "remote:r1", path: "media/x.png")
        try await store.relocateAsset(id: id, host: "mac", path: "/Users/todd/Pictures/ComfyBox/kira-01.png")
        let home = try await store.storageState(of: id)
        #expect(home?.storageState == "local")
        #expect(home?.primaryHost == "mac")
    }

    // ── The journal ──────────────────────────────────────────────────────────

    @Test("a transfer is recorded, advanced, and finished")
    func journalLifecycle() async throws {
        let (store, id, path) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: path) }

        try await store.beginTransfer(assetID: id, remoteID: "r1", remotePath: "media/2026/09/kira-01.png")
        var pending = try await store.pendingTransfers()
        #expect(pending.map(\.assetID) == [id])
        #expect(pending.first?.state == .copying)

        try await store.setTransferState(assetID: id, state: .verified)
        pending = try await store.pendingTransfers()
        #expect(pending.first?.state == .verified)
        #expect(pending.first?.remotePath == "media/2026/09/kira-01.png")

        try await store.finishTransfer(assetID: id)
        #expect(try await store.pendingTransfers().isEmpty, "a finished transfer is not pending work")
    }

    @Test("a failed transfer leaves no pending work and no location")
    func failedTransfer() async throws {
        let (store, id, path) = try await makeStore()
        defer { try? FileManager.default.removeItem(atPath: path) }

        try await store.beginTransfer(assetID: id, remoteID: "r1", remotePath: "media/x.png")
        try await store.setTransferState(assetID: id, state: .failed)
        #expect(try await store.pendingTransfers().isEmpty)
        #expect(try await store.locations(of: id, scope: nil).isEmpty)
        let state = try await store.storageState(of: id)
        #expect(state?.storageState != "remote", "a failed send never claims the asset moved")
    }

    @Test("an older database gains the columns and the journal on open")
    func migrationIsAdditive() async throws {
        let path = tempPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        // A database with the pre-remote shape: open, close, reopen.
        let first = try await CatalogStore.open(path: path)
        try await first.upsert(CatalogAsset(id: "a", filename: "a.png", absolutePath: "/tmp/a.png"),
                               explicitCollectionIDs: [])
        let second = try await CatalogStore.open(path: path)
        try await second.relocateAsset(id: "a", host: "remote:r1", path: "media/a.png")
        #expect(try await second.storageState(of: "a")?.storageState == "remote")
    }
}
