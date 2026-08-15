// GalleryMaintenanceViewTests.swift — Tests for GalleryMaintenanceView's pure
// formatting helpers and its in-view "missing assets" preview count. The
// sheet itself is a SwiftUI view (no snapshot/unit harness in this repo,
// confirmed by ArchiveSheetTests' own note) — this covers the non-view logic
// it adds: report/result/confirmation line formatting, and
// `countMissingAssets`, which mirrors `DAMStore.pruneOrphans()`'s selection
// without touching the database.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("GalleryMaintenanceView formatting")
struct GalleryMaintenanceViewFormattingTests {

    // MARK: - Orphan thumbnails

    @Test("orphan report line uses singular wording for one thumbnail")
    func orphanReportLineSingular() {
        let line = GalleryMaintenanceView.orphanReportLine(count: 1, bytes: 1_024)
        #expect(line.contains("1 orphan thumbnail ·"))
        #expect(!line.contains("thumbnails"))
        #expect(line.contains("reclaimable"))
    }

    @Test("orphan report line uses plural wording for many thumbnails")
    func orphanReportLinePlural() {
        let line = GalleryMaintenanceView.orphanReportLine(count: 148, bytes: 4_200_000)
        #expect(line.contains("148 orphan thumbnails ·"))
        #expect(line.contains("MB"))
        #expect(line.contains("reclaimable"))
    }

    @Test("orphan result line reports deleted count and freed bytes")
    func orphanResultLine() {
        let line = GalleryMaintenanceView.orphanResultLine(deleted: 148, bytesFreed: 4_200_000)
        #expect(line.hasPrefix("Deleted 148"))
        #expect(line.contains("freed"))
        #expect(line.contains("MB"))
    }

    @Test("orphan confirm title includes count and size")
    func orphanConfirmTitle() {
        let title = GalleryMaintenanceView.orphanConfirmTitle(count: 148, bytes: 4_200_000)
        #expect(title.contains("148 orphan thumbnails"))
        #expect(title.contains("MB"))
        #expect(title.hasSuffix("?"))
    }

    @Test("orphan confirm title singular wording for one thumbnail")
    func orphanConfirmTitleSingular() {
        let title = GalleryMaintenanceView.orphanConfirmTitle(count: 1, bytes: 512)
        #expect(title.contains("1 orphan thumbnail ("))
        #expect(!title.contains("thumbnails"))
    }

    // MARK: - Missing assets

    @Test("missing assets line uses singular/plural wording")
    func missingAssetsLine() {
        #expect(GalleryMaintenanceView.missingAssetsLine(count: 1) == "1 asset whose file is gone")
        #expect(GalleryMaintenanceView.missingAssetsLine(count: 12) == "12 assets whose file is gone")
        #expect(GalleryMaintenanceView.missingAssetsLine(count: 0) == "0 assets whose file is gone")
    }

    @Test("missing removed line reports the count")
    func missingRemovedLine() {
        #expect(GalleryMaintenanceView.missingRemovedLine(count: 12) == "Removed 12")
        #expect(GalleryMaintenanceView.missingRemovedLine(count: 0) == "Removed 0")
    }

    // MARK: - Regenerate all thumbnails

    @Test("regen summary line reports total/regenerated only when clean")
    func regenSummaryLineClean() {
        let summary = AssetIngestor.ThumbnailRegenSummary(total: 50, regenerated: 50, missingSource: 0, failed: 0)
        let line = GalleryMaintenanceView.regenSummaryLine(summary)
        #expect(line == "Regenerated 50 of 50")
    }

    @Test("regen summary line appends missingSource and failed clauses when present")
    func regenSummaryLineWithIssues() {
        let summary = AssetIngestor.ThumbnailRegenSummary(total: 50, regenerated: 45, missingSource: 3, failed: 2)
        let line = GalleryMaintenanceView.regenSummaryLine(summary)
        #expect(line.contains("Regenerated 45 of 50"))
        #expect(line.contains("3 missing source"))
        #expect(line.contains("2 failed"))
    }

    // MARK: - Incomplete archives

    @Test("incomplete archives line uses singular/plural wording")
    func incompleteArchivesLine() {
        #expect(GalleryMaintenanceView.incompleteArchivesLine(count: 1) == "1 incomplete archive")
        #expect(GalleryMaintenanceView.incompleteArchivesLine(count: 3) == "3 incomplete archives")
        #expect(GalleryMaintenanceView.incompleteArchivesLine(count: 0) == "0 incomplete archives")
    }

    @Test("incomplete discarded line reports the count")
    func incompleteDiscardedLine() {
        #expect(GalleryMaintenanceView.incompleteDiscardedLine(count: 3) == "Discarded 3")
    }

    @Test("incomplete confirm title uses singular/plural wording")
    func incompleteConfirmTitle() {
        #expect(GalleryMaintenanceView.incompleteConfirmTitle(count: 1) == "Discard 1 incomplete archive?")
        #expect(GalleryMaintenanceView.incompleteConfirmTitle(count: 3) == "Discard 3 incomplete archives?")
    }
}

@Suite("GalleryMaintenanceView.countMissingAssets")
struct GalleryMaintenanceViewMissingAssetsTests {
    @MainActor
    private func makeStore() async throws -> (store: DAMStore, base: String) {
        let base = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("maintenance-view-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: base, withIntermediateDirectories: true)
        let dbPath = (base as NSString).appendingPathComponent("dam.sqlite3")
        let store = try await DAMStore.open(path: dbPath)
        return (store, base)
    }

    @Test("counts only assets whose file no longer exists")
    @MainActor
    func countsMissingOnly() async throws {
        let env = try await makeStore()

        let livePath = (env.base as NSString).appendingPathComponent("live.png")
        FileManager.default.createFile(atPath: livePath, contents: Data("live".utf8))
        _ = try await env.store.insertAsset(DAMAsset(filename: "live.png", absolutePath: livePath))

        // Never written to disk — file is "gone" from the moment it's inserted.
        let goneOnePath = (env.base as NSString).appendingPathComponent("gone1.png")
        let goneTwoPath = (env.base as NSString).appendingPathComponent("gone2.png")
        _ = try await env.store.insertAsset(DAMAsset(filename: "gone1.png", absolutePath: goneOnePath))
        _ = try await env.store.insertAsset(DAMAsset(filename: "gone2.png", absolutePath: goneTwoPath))

        let count = try await GalleryMaintenanceView.countMissingAssets(store: env.store)
        #expect(count == 2)
    }

    @Test("excludes secured assets even when their file is gone")
    @MainActor
    func excludesSecured() async throws {
        let env = try await makeStore()

        let securedPath = (env.base as NSString).appendingPathComponent("secured.png")
        let inserted = try await env.store.insertAsset(DAMAsset(filename: "secured.png", absolutePath: securedPath))
        try await env.store.secureAsset(id: inserted.id, securedPath: "vault/secured.png", originalPath: securedPath)

        let count = try await GalleryMaintenanceView.countMissingAssets(store: env.store)
        #expect(count == 0)
    }

    @Test("empty store yields zero without throwing")
    @MainActor
    func emptyStoreYieldsZero() async throws {
        let env = try await makeStore()
        let count = try await GalleryMaintenanceView.countMissingAssets(store: env.store)
        #expect(count == 0)
    }
}
