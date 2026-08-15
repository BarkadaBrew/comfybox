// GalleryMaintenanceView.swift — Gallery Health maintenance sheet (T11)
//
// Launched from the Gallery toolbar (stethoscope icon) — not from Settings,
// which is a fixed 500x480 preferences window with no DAMStore/AssetIngestor
// (see §7.6 of the FDD). Four independent Scan -> report -> act sections,
// each backed by an already-shipped piece of maintenance machinery:
//   - Orphan thumbnails: GalleryMaintenance.scanOrphanThumbnails/purgeOrphanThumbnails
//   - Missing assets: no dry-run API exists on AssetIngestor, so the preview
//     count is computed here by mirroring DAMStore.pruneOrphans()'s own
//     selection (secured assets excluded, existence stat'd off-main) without
//     touching the database; the actual removal still goes through
//     ingestor.pruneOrphans().
//   - Regenerate all thumbnails: AssetIngestor.regenerateAllThumbnails(for:progress:)
//   - Incomplete archives: a private ArchiveStore instance (not the one the
//     App layer owns — this sheet only needs a scan+discard, not the shared
//     browser state), reusing its manifest-header scan.
//
// Controller ruling: purge and regenerate are disabled while
// `archiver?.isRunning` is true — a restore in flight writes DAM rows and
// thumbnails together, and racing it with a purge/regenerate could delete or
// overwrite a thumbnail whose row hasn't landed yet. GalleryMaintenance and
// AssetIngestor deliberately don't enforce this themselves (they're also
// used from contexts with no archiver at all), so the guard lives here.

import SwiftUI

struct GalleryMaintenanceView: View {
    let store: DAMStore
    let ingestor: AssetIngestor
    var archiver: GalleryArchiver?

    @Environment(\.dismiss) private var dismiss

    private var maintenance: GalleryMaintenance { GalleryMaintenance(store: store, ingestor: ingestor) }

    /// A restore in flight writes thumbnails whose rows may not exist yet —
    /// purge and regenerate must wait it out.
    private var actionsLocked: Bool { archiver?.isRunning ?? false }

    // MARK: - Orphan thumbnails

    @State private var orphanReport: ThumbnailOrphanReport?
    @State private var orphanScanning = false
    @State private var orphanPurging = false
    @State private var orphanResult: (deleted: Int, bytesFreed: Int64)?
    @State private var orphanError: String?
    @State private var showOrphanConfirm = false

    // MARK: - Missing assets

    @State private var missingCount: Int?
    @State private var missingScanning = false
    @State private var missingRemoving = false
    @State private var missingRemoved: Int?
    @State private var missingError: String?

    // MARK: - Regenerate all thumbnails

    @State private var assetTotal: Int?
    @State private var regenScanning = false
    @State private var regenRunning = false
    @State private var regenProgress: (done: Int, total: Int)?
    @State private var regenSummary: AssetIngestor.ThumbnailRegenSummary?
    @State private var regenError: String?

    // MARK: - Incomplete archives

    @State private var archiveStore = ArchiveStore()
    @State private var incompleteScanned = false
    @State private var incompleteScanning = false
    @State private var incompleteDiscarding = false
    @State private var incompleteDiscardResult: Int?
    @State private var incompleteError: String?
    @State private var showIncompleteConfirm = false

    private var incompleteArchives: [ArchiveStore.Summary] {
        archiveStore.archives.filter(\.isIncomplete)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Gallery Health", systemImage: "stethoscope")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
            }
            .padding()

            Divider()

            Form {
                orphanSection
                missingSection
                regenSection
                incompleteSection
            }
            .formStyle(.grouped)
        }
        .frame(width: 520, height: 460)
        .confirmationDialog(
            orphanReport.map { Self.orphanConfirmTitle(count: $0.orphanCount, bytes: $0.reclaimableBytes) } ?? "",
            isPresented: $showOrphanConfirm,
            titleVisibility: .visible
        ) {
            Button("Clean Up", role: .destructive) { Task { await purgeOrphans() } }
            Button("Cancel", role: .cancel) {}
        }
        .confirmationDialog(
            Self.incompleteConfirmTitle(count: incompleteArchives.count),
            isPresented: $showIncompleteConfirm,
            titleVisibility: .visible
        ) {
            Button("Discard", role: .destructive) { Task { await discardIncomplete() } }
            Button("Cancel", role: .cancel) {}
        }
    }

    // MARK: - Orphan thumbnails section

    private var orphanSection: some View {
        Section("Orphan Thumbnails") {
            if let report = orphanReport {
                Text(Self.orphanReportLine(count: report.orphanCount, bytes: report.reclaimableBytes))
                    .foregroundStyle(.secondary)
            }
            if let result = orphanResult {
                Text(Self.orphanResultLine(deleted: result.deleted, bytesFreed: result.bytesFreed))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let orphanError {
                Text(orphanError).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Button(orphanReport == nil ? "Scan" : "Rescan") {
                    Task { await scanOrphans() }
                }
                .disabled(orphanScanning)

                if let report = orphanReport, orphanResult == nil {
                    Button("Clean Up") { showOrphanConfirm = true }
                        .disabled(orphanPurging || actionsLocked || report.orphanCount == 0)
                        .help(actionsLocked ? "A restore is in progress — try again once it finishes." : "")
                }

                if orphanScanning || orphanPurging {
                    ProgressView().controlSize(.small)
                }
            }
        }
    }

    private func scanOrphans() async {
        orphanScanning = true
        orphanError = nil
        orphanResult = nil
        defer { orphanScanning = false }
        do {
            orphanReport = try await maintenance.scanOrphanThumbnails()
        } catch {
            orphanError = error.localizedDescription
        }
    }

    private func purgeOrphans() async {
        guard let report = orphanReport else { return }
        orphanPurging = true
        defer { orphanPurging = false }
        let result = await maintenance.purgeOrphanThumbnails(report)
        orphanResult = result
        orphanReport = nil
    }

    // MARK: - Missing assets section

    private var missingSection: some View {
        Section("Missing Assets") {
            if let missingCount {
                Text(Self.missingAssetsLine(count: missingCount))
                    .foregroundStyle(.secondary)
            }
            if let missingRemoved {
                Text(Self.missingRemovedLine(count: missingRemoved))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let missingError {
                Text(missingError).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Button(missingCount == nil ? "Scan" : "Rescan") {
                    Task { await scanMissing() }
                }
                .disabled(missingScanning)

                if let missingCount, self.missingRemoved == nil {
                    Button("Remove") {
                        Task { await removeMissing() }
                    }
                    .disabled(missingRemoving || missingCount == 0)
                }

                if missingScanning || missingRemoving {
                    ProgressView().controlSize(.small)
                }
            }
        }
    }

    private func scanMissing() async {
        missingScanning = true
        missingError = nil
        missingRemoved = nil
        defer { missingScanning = false }
        do {
            missingCount = try await Self.countMissingAssets(store: store)
        } catch {
            missingError = error.localizedDescription
        }
    }

    private func removeMissing() async {
        missingRemoving = true
        defer { missingRemoving = false }
        do {
            missingRemoved = try await ingestor.pruneOrphans()
            missingCount = nil
        } catch {
            missingError = error.localizedDescription
        }
    }

    /// Mirrors `DAMStore.pruneOrphans()`'s selection (secured assets
    /// excluded, existence checked via `FileManager`) without touching the
    /// database — a pure preview count for the confirmation UI. No dry-run
    /// API exists on `AssetIngestor`/`DAMStore`, so this is computed
    /// in-view; the stat loop runs off-main.
    static func countMissingAssets(store: DAMStore) async throws -> Int {
        let secured = try await store.securedAssetIds()
        let total = try await store.assetCount()
        let all = try await store.fetchAssets(limit: total, offset: 0)
        return await Task.detached(priority: .utility) {
            var count = 0
            let fm = FileManager.default
            for asset in all where !secured.contains(asset.id) {
                if !fm.fileExists(atPath: asset.absolutePath) { count += 1 }
            }
            return count
        }.value
    }

    // MARK: - Regenerate all thumbnails section

    private var regenSection: some View {
        Section("Regenerate All Thumbnails") {
            if let assetTotal {
                Text("\(assetTotal) asset\(assetTotal == 1 ? "" : "s")")
                    .foregroundStyle(.secondary)
            }
            if let progress = regenProgress {
                ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
                Text("\(progress.done) of \(progress.total)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let summary = regenSummary {
                Text(Self.regenSummaryLine(summary))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let regenError {
                Text(regenError).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Button(assetTotal == nil ? "Scan" : "Rescan") {
                    Task { await scanAssetCount() }
                }
                .disabled(regenScanning || regenRunning)

                if let assetTotal, regenProgress == nil {
                    Button("Regenerate") {
                        Task { await regenerateAll() }
                    }
                    .disabled(regenRunning || actionsLocked || assetTotal == 0)
                    .help(actionsLocked ? "A restore is in progress — try again once it finishes." : "")
                }

                if regenScanning {
                    ProgressView().controlSize(.small)
                }
            }
        }
    }

    private func scanAssetCount() async {
        regenScanning = true
        regenError = nil
        regenSummary = nil
        defer { regenScanning = false }
        do {
            assetTotal = try await store.assetCount()
        } catch {
            regenError = error.localizedDescription
        }
    }

    private func regenerateAll() async {
        guard let total = assetTotal else { return }
        regenRunning = true
        regenSummary = nil
        regenProgress = (0, total)
        defer { regenRunning = false }
        do {
            let assets = try await store.fetchAssets(limit: total, offset: 0)
            let summary = await ingestor.regenerateAllThumbnails(for: assets) { done, total in
                regenProgress = (done, total)
            }
            regenSummary = summary
        } catch {
            regenError = error.localizedDescription
        }
        regenProgress = nil
    }

    // MARK: - Incomplete archives section

    private var incompleteSection: some View {
        Section("Incomplete Archives") {
            if incompleteScanned {
                Text(Self.incompleteArchivesLine(count: incompleteArchives.count))
                    .foregroundStyle(.secondary)
            }
            if let incompleteDiscardResult {
                Text(Self.incompleteDiscardedLine(count: incompleteDiscardResult))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let incompleteError {
                Text(incompleteError).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Button(incompleteScanned ? "Rescan" : "Scan") {
                    Task { await scanIncomplete() }
                }
                .disabled(incompleteScanning)

                if incompleteScanned, incompleteDiscardResult == nil, !incompleteArchives.isEmpty {
                    Button("Discard") { showIncompleteConfirm = true }
                        .disabled(incompleteDiscarding)
                }

                if incompleteScanning || incompleteDiscarding {
                    ProgressView().controlSize(.small)
                }
            }
        }
    }

    private func scanIncomplete() async {
        incompleteScanning = true
        incompleteError = nil
        incompleteDiscardResult = nil
        defer { incompleteScanning = false; incompleteScanned = true }
        await archiveStore.reload()
    }

    private func discardIncomplete() async {
        incompleteDiscarding = true
        defer { incompleteDiscarding = false }
        var discarded = 0
        for summary in incompleteArchives {
            // Belt and suspenders: only ever remove a directory that is
            // itself a `.cbarchive` bundle — never anything else on disk.
            guard summary.bundlePath.hasSuffix(".cbarchive") else { continue }
            if (try? FileManager.default.removeItem(atPath: summary.bundlePath)) != nil {
                discarded += 1
            }
        }
        incompleteDiscardResult = discarded
        await archiveStore.reload()
    }

    // MARK: - Pure formatting helpers (unit-tested)

    private static func byteString(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB, .useKB, .useBytes]
        return formatter.string(fromByteCount: bytes)
    }

    static func orphanReportLine(count: Int, bytes: Int64) -> String {
        let word = count == 1 ? "thumbnail" : "thumbnails"
        return "\(count) orphan \(word) · \(byteString(bytes)) reclaimable"
    }

    static func orphanResultLine(deleted: Int, bytesFreed: Int64) -> String {
        "Deleted \(deleted) · freed \(byteString(bytesFreed))"
    }

    static func orphanConfirmTitle(count: Int, bytes: Int64) -> String {
        let word = count == 1 ? "thumbnail" : "thumbnails"
        return "Delete \(count) orphan \(word) (\(byteString(bytes)))?"
    }

    static func missingAssetsLine(count: Int) -> String {
        "\(count) asset\(count == 1 ? "" : "s") whose file is gone"
    }

    static func missingRemovedLine(count: Int) -> String {
        "Removed \(count)"
    }

    static func regenSummaryLine(_ summary: AssetIngestor.ThumbnailRegenSummary) -> String {
        var parts = ["Regenerated \(summary.regenerated) of \(summary.total)"]
        if summary.missingSource > 0 {
            parts.append("\(summary.missingSource) missing source")
        }
        if summary.failed > 0 {
            parts.append("\(summary.failed) failed")
        }
        return parts.joined(separator: " · ")
    }

    static func incompleteArchivesLine(count: Int) -> String {
        "\(count) incomplete archive\(count == 1 ? "" : "s")"
    }

    static func incompleteDiscardedLine(count: Int) -> String {
        "Discarded \(count)"
    }

    static func incompleteConfirmTitle(count: Int) -> String {
        "Discard \(count) incomplete archive\(count == 1 ? "" : "s")?"
    }
}
