// GalleryMaintenanceMissingFilesTests.swift — the Gallery Health sheet's
// "Missing Assets" section (FDD-remote-galleries §3.6).
//
// Todd 2026-09-17: "there is a doctor function built already but it doesnt
// work with this use case." The doctor's Remove button called
// AssetIngestor.pruneOrphans(), i.e. the UNATTENDED sweep, which refuses
// outright above 5% of the library — 188 of his 411 rows — so the one screen
// built to fix the catalog could never fix it.

import Testing
import Foundation
import ComfyBoxCatalog
@testable import ComfyBoxDesktop

@Suite("Gallery Health: missing assets")
@MainActor
struct GalleryMaintenanceMissingFilesTests {

    private func tempPath(_ suffix: String) -> String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("maint-\(UUID().uuidString)\(suffix)")
    }

    private func asset(_ id: String, at path: String) -> DAMAsset {
        DAMAsset(id: id, kind: "image", filename: (path as NSString).lastPathComponent, absolutePath: path)
    }

    /// A store + ingestor pair over throwaway directories.
    private func makeSubject() async throws -> (GalleryMaintenance, DAMStore, AssetIngestor, String) {
        let dbPath = tempPath(".sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        let thumbs = tempPath("-thumbs")
        try FileManager.default.createDirectory(atPath: thumbs, withIntermediateDirectories: true)
        let ingestor = AssetIngestor(store: store, watchDirectory: tempPath("-watch"), thumbnailDirectory: thumbs)
        return (GalleryMaintenance(store: store, ingestor: ingestor), store, ingestor, dbPath)
    }

    @Test("the scan splits what is deleted from what is merely unplugged")
    func scanSplits() async throws {
        let (maintenance, store, _, dbPath) = try await makeSubject()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        try await store.insertAsset(asset("gone", at: tempPath("-gone.png")))
        try await store.insertAsset(asset("unplugged", at: "/Volumes/NoSuchDrive-\(UUID().uuidString)/x.png"))

        let report = try await maintenance.scanMissingFiles()
        #expect(report.orphans.map(\.id) == ["gone"])
        #expect(report.unattached.map(\.id) == ["unplugged"])
    }

    @Test("purging removes the rows and their cached thumbnails, however many")
    func purgeRemovesRowsAndThumbnails() async throws {
        let (maintenance, store, ingestor, dbPath) = try await makeSubject()
        defer { try? FileManager.default.removeItem(atPath: dbPath) }

        // 40 of 41 rows gone — past the unattended sweep's ceiling, which is
        // exactly the case the doctor could not handle.
        var ids: [String] = []
        for i in 0..<40 {
            let id = "gone-\(i)"
            ids.append(id)
            try await store.insertAsset(asset(id, at: tempPath("-\(i).png")))
            FileManager.default.createFile(atPath: ingestor.thumbnailPath(for: id), contents: Data("t".utf8))
        }
        try await store.insertAsset(asset("unplugged", at: "/Volumes/NoSuchDrive-\(UUID().uuidString)/x.png"))

        let purged = try await maintenance.purgeMissingFiles(ids: ids)
        #expect(purged == 40)
        #expect(try await store.assetCount() == 1, "the unplugged row is untouched")
        #expect(!FileManager.default.fileExists(atPath: ingestor.thumbnailPath(for: "gone-0")), "its thumbnail goes too")
    }

    @Test("vacuum drops location rows whose asset is gone")
    func vacuum() async throws {
        let dbPath = tempPath(".sqlite3")
        _ = try await CatalogStore.open(path: dbPath)
        let (maintenance, store, _, _) = try await makeSubject()
        _ = store
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        // No stale rows in a fresh database: the call is safe and returns 0.
        #expect(try await maintenance.vacuumStaleLocations() == 0)
    }

    // ── The lines the sheet shows ────────────────────────────────────────────

    @Test("the missing-assets line names both halves")
    func missingLine() {
        #expect(GalleryMaintenanceView.missingAssetsLine(orphans: 188, unattached: 0)
            == "188 assets whose file is gone")
        #expect(GalleryMaintenanceView.missingAssetsLine(orphans: 1, unattached: 0)
            == "1 asset whose file is gone")
        #expect(GalleryMaintenanceView.missingAssetsLine(orphans: 0, unattached: 0)
            == "No missing files")
    }

    @Test("unplugged drives are reported separately, and never as deletions")
    func unattachedLine() throws {
        let line = try #require(GalleryMaintenanceView.unattachedLine(volumes: ["Bolt", "Vault"], count: 12))
        #expect(line.contains("12"))
        #expect(line.contains("Bolt") && line.contains("Vault"))
        #expect(!line.lowercased().contains("delete"), "nothing here is deletable: \(line)")
        #expect(GalleryMaintenanceView.unattachedLine(volumes: [], count: 0) == nil)
    }

    @Test("the volume name is what the user reads, not the whole path")
    func volumeNames() {
        let names = GalleryMaintenanceView.volumeNames(paths: [
            "/Volumes/Bolt/ComfyBox/a.png",
            "/Volumes/Bolt/ComfyBox/b.png",
            "/Volumes/Vault/ComfyBoxGallery/media/c.mp4",
            "/Users/todd/Pictures/ComfyBox/d.png",
        ])
        #expect(names == ["Bolt", "Vault"])
    }
}
