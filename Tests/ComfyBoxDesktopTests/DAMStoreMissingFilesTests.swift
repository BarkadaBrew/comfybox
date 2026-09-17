// DAMStoreMissingFilesTests.swift — telling a deleted file apart from an
// unattached one (FDD-remote-galleries §3.6).
//
// Todd 2026-09-17: "the current gallery is corrupted and wont fix the DB of
// the deleted entries, it assumes it is unattached storage that the thumbnail
// cant find and wont clean itself." His live catalog: 411 rows, 188 with no
// file, so the unattended sweep's 5% ceiling (20) refused every time and
// nothing was ever cleaned.

import Testing
import Foundation
import SQLite3
import ComfyBoxCatalog
@testable import ComfyBoxDesktop

@Suite("DAMStore missing files")
struct DAMStoreMissingFilesTests {

    private func tempDBPath() -> String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
    }

    /// An asset row pointing at `path`.
    private func asset(_ id: String, at path: String) -> DAMAsset {
        DAMAsset(id: id, kind: "image", filename: (path as NSString).lastPathComponent, absolutePath: path)
    }

    /// Insert a raw `asset_locations` row the way the catalog backfill would.
    private func insertLocation(dbPath: String, assetID: String, host: String, path: String) throws {
        var db: OpaquePointer?
        #expect(sqlite3_open(dbPath, &db) == SQLITE_OK)
        defer { sqlite3_close(db) }
        let sql = "INSERT OR REPLACE INTO asset_locations (asset_id, host, path, mtime) VALUES (?,?,?,0)"
        var stmt: OpaquePointer?
        #expect(sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK)
        defer { sqlite3_finalize(stmt) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, assetID, -1, transient)
        sqlite3_bind_text(stmt, 2, host, -1, transient)
        sqlite3_bind_text(stmt, 3, path, -1, transient)
        #expect(sqlite3_step(stmt) == SQLITE_DONE)
    }

    private func locationCount(dbPath: String) -> Int {
        var db: OpaquePointer?
        guard sqlite3_open(dbPath, &db) == SQLITE_OK else { return -1 }
        defer { sqlite3_close(db) }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM asset_locations", -1, &stmt, nil) == SQLITE_OK else { return -1 }
        defer { sqlite3_finalize(stmt) }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int(stmt, 0)) : -1
    }

    // ── The classifier ───────────────────────────────────────────────────────

    @Test("a missing file on a mounted volume is a genuine orphan")
    func orphanOnMountedVolume() {
        let verdict = DAMStore.classifyMissingFile(
            path: "/Users/todd/Pictures/ComfyBox/gone.png",
            mountedVolumeRoots: ["/", "/Volumes/Vault"])
        #expect(verdict == .orphan)
    }

    @Test("a missing file on an unmounted volume is unattached, never an orphan")
    func unattachedVolume() {
        let verdict = DAMStore.classifyMissingFile(
            path: "/Volumes/Bolt/ComfyBox/clip.mp4",
            mountedVolumeRoots: ["/", "/Volumes/Vault"])
        #expect(verdict == .unattached)
        // The same path once the drive is back.
        #expect(DAMStore.classifyMissingFile(
            path: "/Volumes/Bolt/ComfyBox/clip.mp4",
            mountedVolumeRoots: ["/", "/Volumes/Bolt"]) == .orphan)
    }

    @Test("the volume name is matched whole, not by prefix")
    func volumePrefixIsNotAMatch() {
        // "/Volumes/Vault2" is a different drive from "/Volumes/Vault".
        #expect(DAMStore.classifyMissingFile(
            path: "/Volumes/Vault2/ComfyBoxGallery/media/x.png",
            mountedVolumeRoots: ["/", "/Volumes/Vault"]) == .unattached)
    }

    // ── The unattended sweep ─────────────────────────────────────────────────

    @Test("unattached rows do not count toward the circuit breaker")
    func breakerIgnoresUnattached() async throws {
        let path = tempDBPath()
        let store = try await DAMStore.open(path: path)
        defer { try? FileManager.default.removeItem(atPath: path) }

        // 30 rows on a drive that is not mounted, plus one real deletion here.
        for i in 0..<30 {
            try await store.insertAsset(asset("gone-\(i)", at: "/Volumes/NoSuchDrive-\(UUID().uuidString)/media/\(i).png"))
        }
        try await store.insertAsset(asset("deleted-here", at: NSTemporaryDirectory() + "definitely-not-here-\(UUID().uuidString).png"))

        // Before the fix this threw: 31 candidates against a ceiling of 5.
        let pruned = try await store.pruneOrphans()
        #expect(pruned == ["deleted-here"])
        #expect(try await store.assetCount() == 30, "the unattached rows survive untouched")
    }

    // ── The explicit, reviewed purge ─────────────────────────────────────────

    @Test("scanMissingFiles reports what is gone, and why")
    func scanReportsClassification() async throws {
        let path = tempDBPath()
        let store = try await DAMStore.open(path: path)
        defer { try? FileManager.default.removeItem(atPath: path) }

        let present = (NSTemporaryDirectory() as NSString).appendingPathComponent("present-\(UUID().uuidString).png")
        FileManager.default.createFile(atPath: present, contents: Data("x".utf8))
        defer { try? FileManager.default.removeItem(atPath: present) }

        try await store.insertAsset(asset("here", at: present))
        try await store.insertAsset(asset("gone", at: NSTemporaryDirectory() + "gone-\(UUID().uuidString).png"))
        try await store.insertAsset(asset("unplugged", at: "/Volumes/NoSuchDrive-\(UUID().uuidString)/x.png"))

        let report = try await store.scanMissingFiles()
        #expect(report.orphans.map(\.id) == ["gone"])
        #expect(report.unattached.map(\.id) == ["unplugged"])
        #expect(!report.orphans.contains { $0.id == "here" })
        #expect(report.orphans.first?.absolutePath.hasSuffix(".png") == true)
    }

    @Test("purgeMissing deletes exactly the ids it is given, however many")
    func explicitPurgeIgnoresTheBreaker() async throws {
        let path = tempDBPath()
        let store = try await DAMStore.open(path: path)
        defer { try? FileManager.default.removeItem(atPath: path) }

        var ids: [String] = []
        for i in 0..<40 {
            let id = "gone-\(i)"
            ids.append(id)
            try await store.insertAsset(asset(id, at: NSTemporaryDirectory() + "no-\(UUID().uuidString).png"))
        }
        try await store.insertAsset(asset("keep", at: NSTemporaryDirectory() + "no-\(UUID().uuidString).png"))

        // 40 of 41 rows — far past the unattended ceiling, and deliberate.
        let purged = try await store.purgeMissing(ids: ids)
        #expect(purged == 40)
        #expect(try await store.assetCount() == 1)
        let left = try await store.fetchAssets(limit: 10)
        #expect(left.first?.id == "keep")
    }

    @Test("purgeMissing refuses an id whose file is still there")
    func purgeRefusesLiveFiles() async throws {
        let path = tempDBPath()
        let store = try await DAMStore.open(path: path)
        defer { try? FileManager.default.removeItem(atPath: path) }

        let present = (NSTemporaryDirectory() as NSString).appendingPathComponent("present-\(UUID().uuidString).png")
        FileManager.default.createFile(atPath: present, contents: Data("x".utf8))
        defer { try? FileManager.default.removeItem(atPath: present) }
        try await store.insertAsset(asset("here", at: present))

        let purged = try await store.purgeMissing(ids: ["here"])
        #expect(purged == 0, "a row with a file is never a missing-file purge candidate")
        #expect(try await store.assetCount() == 1)
    }

    // ── Stale location rows ──────────────────────────────────────────────────

    @Test("vacuumStaleLocations drops location rows with no asset, and keeps the rest")
    func vacuumStaleLocations() async throws {
        let path = tempDBPath()
        // The catalog owns this table; open it first so the migration runs.
        _ = try await CatalogStore.open(path: path)
        let store = try await DAMStore.open(path: path)
        defer { try? FileManager.default.removeItem(atPath: path) }

        try await store.insertAsset(asset("live", at: NSTemporaryDirectory() + "live-\(UUID().uuidString).png"))
        try insertLocation(dbPath: path, assetID: "live", host: "mac", path: "/tmp/live.png")
        try insertLocation(dbPath: path, assetID: "vanished", host: "kira", path: "/home/todd/.kira/studio/gallery/old.png")
        try insertLocation(dbPath: path, assetID: "vanished-2", host: "kira", path: "/home/todd/.kira/studio/gallery/old2.png")

        let removed = try await store.vacuumStaleLocations()
        #expect(removed == 2, "885 rows like these are in Todd's live database")
        #expect(locationCount(dbPath: path) == 1)
    }
}
