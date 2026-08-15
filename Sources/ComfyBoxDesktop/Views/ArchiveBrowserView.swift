// ArchiveBrowserView.swift — Archive browser tab (T10)
//
// HStack sidebar+grid mirroring GalleryView's shape: a 190pt-wide List of
// .cbarchive bundles (from ArchiveStore) on the left, and a LazyVGrid of the
// SELECTED bundle's archived-asset thumbnails on the right. Entries are
// streamed from that bundle's entries.jsonl page-by-page (never the whole
// file eagerly) via `loadEntriesPage`. Restore Selected/Restore All run
// through the shared GalleryArchiver the app already owns; Delete Archive
// trashes the bundle directory outright (it's a plain folder on disk, not a
// DAMStore-tracked asset). Export as Zip (T12) shells `ditto` via
// ArchiveStore.exportAsZip — destination chosen with an NSSavePanel on
// the main actor, the Process work off it.

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ArchiveBrowserView: View {
    let store: DAMStore
    let ingestor: AssetIngestor
    let archiver: GalleryArchiver
    let archives: ArchiveStore

    static let entriesPageSize = 500

    @State private var selectedBundleId: String?
    @State private var selectedBundle: ArchiveStore.Summary?
    @State private var entries: [ArchivedAsset] = []
    @State private var hasMoreEntries = false
    @State private var isLoadingEntries = false
    @State private var selectedAssetIds: Set<String> = []

    @State private var restoreToOriginalLocations = false
    @State private var overwriteExistingMetadata = false
    @State private var restoreProgress: (done: Int, total: Int)?
    @State private var restoreSummary: String?

    @State private var errorMessage: String?

    @State private var pendingDeleteBundle: ArchiveStore.Summary?
    @State private var showDeleteConfirmation = false

    @State private var exportingBundleIds: Set<String> = []
    @State private var exportSummary: String?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 190)
            Divider()
            detail
        }
        .task {
            await archives.reload()
        }
        .onReceive(NotificationCenter.default.publisher(for: DesktopSettings.didChangeNotification)) { _ in
            let newRoots = DesktopSettings.load().archiveRoots ?? [DesktopSettings.defaultArchiveRoot]
            guard newRoots != archives.roots else { return }
            archives.roots = newRoots
            Task { await archives.reload() }
        }
        .onChange(of: selectedBundleId) { _, newValue in
            guard let newValue, newValue != selectedBundle?.id,
                  let bundle = archives.archives.first(where: { $0.id == newValue })
            else { return }
            selectBundle(bundle)
        }
        .confirmationDialog(
            "Delete \"\(pendingDeleteBundle?.manifest.name ?? "")\"?",
            isPresented: $showDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Move to Trash", role: .destructive) {
                guard let bundle = pendingDeleteBundle else { return }
                pendingDeleteBundle = nil
                Task { await deleteBundle(bundle) }
            }
            Button("Cancel", role: .cancel) { pendingDeleteBundle = nil }
        } message: {
            Text("The archive bundle is moved to the Trash. Assets already restored to the gallery are unaffected.")
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Archives").font(.headline)
                Spacer()
                Button {
                    Task { await archives.reload() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Refresh")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)

            Divider()

            if archives.isLoading && archives.archives.isEmpty {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if archives.archives.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "archivebox")
                        .font(.largeTitle)
                        .foregroundStyle(.tertiary)
                    Text("No archives")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(archives.archives, selection: $selectedBundleId) { bundle in
                    archiveRow(bundle).tag(bundle.id)
                }
                .listStyle(.sidebar)
            }

            if let error = archives.error {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(3)
                    .padding(8)
            }
        }
    }

    private func archiveRow(_ bundle: ArchiveStore.Summary) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(bundle.manifest.name)
                .font(.callout)
                .lineLimit(1)
            Text(Date(timeIntervalSince1970: bundle.manifest.createdAt)
                .formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(ArchiveSheet.summaryLine(
                assetCount: bundle.manifest.assetCount,
                totalBytes: bundle.manifest.totalBytes,
                securedCount: 0
            ))
            .font(.caption2)
            .foregroundStyle(.secondary)
            if bundle.isIncomplete {
                Label("Incomplete", systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .padding(.vertical, 2)
        .contextMenu {
            Button("Restore All…") {
                // Select it first so progress/summary land somewhere visible —
                // this can be invoked on a bundle that isn't the current
                // selection.
                selectedBundleId = bundle.id
                selectBundle(bundle)
                restoreAll(bundle)
            }
            .disabled(bundle.isIncomplete)
            Button("Reveal in Finder") { revealInFinder(bundle) }
            Button("Export as Zip…") { exportAsZip(bundle) }
                .disabled(exportingBundleIds.contains(bundle.id))
            Divider()
            Button("Delete Archive…", role: .destructive) { requestDeleteBundle(bundle) }
        }
    }

    // MARK: - Detail (grid)

    @ViewBuilder
    private var detail: some View {
        if let bundle = selectedBundle {
            VStack(spacing: 0) {
                gridToolbar(bundle)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(nsColor: .windowBackgroundColor))

                Divider()

                if let errorMessage {
                    errorBanner(errorMessage)
                }

                if let restoreProgress {
                    restoreProgressStrip(restoreProgress)
                } else if let restoreSummary {
                    Text(restoreSummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 10)
                        .padding(.top, 6)
                } else if let exportSummary {
                    Text(exportSummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 10)
                        .padding(.top, 6)
                }

                if bundle.isIncomplete {
                    incompleteNotice
                } else if isLoadingEntries && entries.isEmpty {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Loading assets…").foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if entries.isEmpty {
                    emptyEntriesState
                } else {
                    assetGrid(bundle)
                }
            }
        } else {
            noSelectionState
        }
    }

    private func gridToolbar(_ bundle: ArchiveStore.Summary) -> some View {
        HStack(spacing: 14) {
            Toggle("Restore to original locations", isOn: $restoreToOriginalLocations)
                .toggleStyle(.checkbox)
            Toggle("Overwrite metadata for assets already in the gallery", isOn: $overwriteExistingMetadata)
                .toggleStyle(.checkbox)

            Spacer()

            if !selectedAssetIds.isEmpty {
                Text("\(selectedAssetIds.count) selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Button("Restore Selected") { restoreSelected(bundle) }
                .disabled(selectedAssetIds.isEmpty || archiver.isRunning || bundle.isIncomplete)
            Button("Restore All") { restoreAll(bundle) }
                .disabled(archiver.isRunning || bundle.isIncomplete)
        }
        .font(.callout)
    }

    private func restoreProgressStrip(_ progress: (done: Int, total: Int)) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Restoring \(progress.done) / \(progress.total)…")
                .font(.caption)
                .foregroundStyle(.secondary)
            ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.callout)
                .lineLimit(2)
            Spacer()
            Button(action: { errorMessage = nil }) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.orange.opacity(0.12))
    }

    private var incompleteNotice: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundStyle(.orange)
            Text("This archive never finished writing")
                .font(.title3)
                .foregroundStyle(.secondary)
            Text("Its assets were never removed from the gallery. Delete this bundle to discard it.")
                .font(.body)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyEntriesState: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("No assets in this archive")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var noSelectionState: some View {
        VStack(spacing: 12) {
            Image(systemName: "archivebox")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("Select an archive")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func assetGrid(_ bundle: ArchiveStore.Summary) -> some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 140, maximum: 200), spacing: 12)], spacing: 12) {
                ForEach(entries) { entry in
                    ArchivedAssetCell(
                        entry: entry,
                        bundlePath: bundle.bundlePath,
                        isSelected: selectedAssetIds.contains(entry.id)
                    )
                    .onTapGesture { toggleEntrySelection(entry.id) }
                }
            }
            .padding(12)

            if hasMoreEntries {
                Button {
                    Task { await loadNextPage() }
                } label: {
                    if isLoadingEntries {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Load \(Self.entriesPageSize) More")
                    }
                }
                .disabled(isLoadingEntries)
                .padding(.bottom, 16)
            }
        }
    }

    private func toggleEntrySelection(_ id: String) {
        if selectedAssetIds.contains(id) {
            selectedAssetIds.remove(id)
        } else {
            selectedAssetIds.insert(id)
        }
    }

    // MARK: - Bundle selection + entry paging

    private func selectBundle(_ bundle: ArchiveStore.Summary) {
        selectedBundle = bundle
        entries = []
        hasMoreEntries = false
        selectedAssetIds = []
        restoreSummary = nil
        exportSummary = nil
        errorMessage = nil
        guard !bundle.isIncomplete else { return }
        Task { await loadNextPage() }
    }

    /// Streams the next `entriesPageSize` rows of the selected bundle's
    /// `entries.jsonl`, appending to `entries`. Guards against a bundle
    /// switch racing the async read.
    private func loadNextPage() async {
        guard let bundle = selectedBundle else { return }
        isLoadingEntries = true
        let entriesPath = (bundle.bundlePath as NSString).appendingPathComponent("entries.jsonl")
        let skip = entries.count
        let pageSize = Self.entriesPageSize
        do {
            let page = try await Task.detached(priority: .utility) {
                try Self.loadEntriesPage(entriesPath: entriesPath, skip: skip, limit: pageSize)
            }.value
            guard selectedBundle?.id == bundle.id else { return }
            entries.append(contentsOf: page.entries)
            hasMoreEntries = !page.isLastPage
        } catch {
            guard selectedBundle?.id == bundle.id else { return }
            errorMessage = "Could not read this archive's assets: \(error.localizedDescription)"
        }
        isLoadingEntries = false
    }

    /// Pure entries.jsonl pager: skips the first `skip` decodable entries,
    /// then collects up to `limit` more, stopping the stream early rather
    /// than reading the whole file for every page. `isLastPage` is true only
    /// when the stream ran out before `limit` was reached — i.e. there is
    /// provably nothing left, not just "we stopped asking".
    struct EntriesPage: Sendable, Equatable {
        var entries: [ArchivedAsset]
        var isLastPage: Bool

        static func == (lhs: EntriesPage, rhs: EntriesPage) -> Bool {
            lhs.entries.map(\.id) == rhs.entries.map(\.id) && lhs.isLastPage == rhs.isLastPage
        }
    }

    private struct PageBoundaryReached: Error {}

    nonisolated static func loadEntriesPage(entriesPath: String, skip: Int, limit: Int) throws -> EntriesPage {
        guard limit > 0 else { return EntriesPage(entries: [], isLastPage: true) }
        var seen = 0
        var collected: [ArchivedAsset] = []
        do {
            _ = try ArchiveJSONL.read(at: entriesPath) { entry in
                guard seen >= skip else {
                    seen += 1
                    return
                }
                collected.append(entry)
                if collected.count > limit {
                    throw PageBoundaryReached()
                }
            }
        } catch is PageBoundaryReached {
            collected.removeLast()
            return EntriesPage(entries: collected, isLastPage: false)
        }
        return EntriesPage(entries: collected, isLastPage: true)
    }

    // MARK: - Restore

    private func restoreSelected(_ bundle: ArchiveStore.Summary) {
        guard !selectedAssetIds.isEmpty else { return }
        runRestore(bundlePath: bundle.bundlePath, assetIds: selectedAssetIds, total: selectedAssetIds.count)
    }

    private func restoreAll(_ bundle: ArchiveStore.Summary) {
        runRestore(bundlePath: bundle.bundlePath, assetIds: nil, total: bundle.manifest.assetCount)
    }

    private func runRestore(bundlePath: String, assetIds: Set<String>?, total: Int) {
        restoreSummary = nil
        errorMessage = nil
        restoreProgress = (0, max(total, 1))
        let request = GalleryArchiver.RestoreRequest(
            bundlePath: bundlePath,
            assetIds: assetIds,
            restoreToOriginalLocations: restoreToOriginalLocations,
            overwriteExistingMetadata: overwriteExistingMetadata
        )
        Task {
            do {
                let result = try await archiver.restore(request) { done, total in
                    restoreProgress = (done, total)
                }
                restoreProgress = nil
                restoreSummary = Self.restoreSummaryLine(result)
                selectedAssetIds = []
                await archives.reload()
            } catch {
                restoreProgress = nil
                errorMessage = "Restore failed: \(error.localizedDescription)"
            }
        }
    }

    /// "Restored N · M skipped · K re-identified · J renamed · F failed" —
    /// each clause is omitted when its count is zero, mirroring
    /// GalleryView.performArchive's summary construction.
    static func restoreSummaryLine(_ result: GalleryArchiver.RestoreResult) -> String {
        var parts = ["Restored \(result.restored)"]
        if result.skipped > 0 { parts.append("\(result.skipped) skipped") }
        if result.reIdentified > 0 { parts.append("\(result.reIdentified) re-identified") }
        if result.renamed > 0 { parts.append("\(result.renamed) renamed") }
        if !result.failed.isEmpty { parts.append("\(result.failed.count) failed") }
        return parts.joined(separator: " · ")
    }

    // MARK: - Export as Zip (T12)

    /// Prompts for a destination with an `NSSavePanel` (main actor), then
    /// shells `ditto` off the main actor via `ArchiveStore.exportAsZip`.
    /// Selects the bundle first (mirroring "Restore All…") so the summary/
    /// error lands somewhere visible even when invoked on a non-selected row.
    private func exportAsZip(_ bundle: ArchiveStore.Summary) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.zip]
        panel.nameFieldStringValue = "\(bundle.manifest.name).zip"
        panel.prompt = "Export"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        selectedBundleId = bundle.id
        selectBundle(bundle)
        exportingBundleIds.insert(bundle.id)
        Task {
            do {
                try await archives.exportAsZip(bundlePath: bundle.bundlePath, destination: url.path)
                exportSummary = "Exported to \(url.path)"
            } catch {
                errorMessage = "Export failed: \(error.localizedDescription)"
            }
            exportingBundleIds.remove(bundle.id)
        }
    }

    // MARK: - Reveal / delete bundle

    private func revealInFinder(_ bundle: ArchiveStore.Summary) {
        NSWorkspace.shared.selectFile(bundle.bundlePath, inFileViewerRootedAtPath: "")
    }

    private func requestDeleteBundle(_ bundle: ArchiveStore.Summary) {
        pendingDeleteBundle = bundle
        showDeleteConfirmation = true
    }

    private func deleteBundle(_ bundle: ArchiveStore.Summary) async {
        let path = bundle.bundlePath
        let success = await Task.detached(priority: .utility) {
            Self.trashOrRemoveBundle(atPath: path)
        }.value
        if !success {
            errorMessage = "Failed to delete \(bundle.manifest.name)."
        }
        if selectedBundle?.id == bundle.id {
            selectedBundle = nil
            selectedBundleId = nil
            entries = []
            selectedAssetIds = []
        }
        await archives.reload()
    }

    /// Trash the bundle directory, falling back to permanent removal where
    /// trashing is unavailable — same fallback AssetIngestor.trashOrRemove uses.
    private nonisolated static func trashOrRemoveBundle(atPath path: String) -> Bool {
        let fm = FileManager.default
        do {
            try fm.trashItem(at: URL(fileURLWithPath: path), resultingItemURL: nil)
            return true
        } catch {
            do {
                try fm.removeItem(atPath: path)
                return true
            } catch {
                return false
            }
        }
    }
}

// MARK: - Archived Asset Cell

/// A trimmed `GalleryCellView`: thumbnail + filename only, reading straight
/// off the bundle's `assets/<id>/` directory rather than through AssetIngestor
/// (these files aren't DAMStore-tracked while archived).
private struct ArchivedAssetCell: View {
    let entry: ArchivedAsset
    let bundlePath: String
    let isSelected: Bool

    @State private var thumbnail: NSImage?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            VStack(spacing: 4) {
                ZStack {
                    Color(nsColor: .controlBackgroundColor)
                    if let thumbnail {
                        Image(nsImage: thumbnail)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    } else {
                        Image(systemName: entry.kind == "video" ? "film" : "photo")
                            .font(.title)
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(height: 130)
                .clipShape(RoundedRectangle(cornerRadius: 8))

                Text(entry.filename)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(isSelected ? .white : .secondary, isSelected ? Color.accentColor : Color.clear)
                .font(.title3)
                .padding(6)
        }
        .padding(6)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
        )
        .task {
            await loadThumbnail()
        }
    }

    private func loadThumbnail() async {
        let thumbPath = (bundlePath as NSString).appendingPathComponent("assets/\(entry.id)/thumb.jpg")
        let fullPath = (bundlePath as NSString).appendingPathComponent("assets/\(entry.id)/\(entry.filename)")
        let image: NSImage? = await Task.detached {
            NSImage(contentsOfFile: thumbPath) ?? NSImage(contentsOfFile: fullPath)
        }.value
        await MainActor.run {
            thumbnail = image
        }
    }
}
