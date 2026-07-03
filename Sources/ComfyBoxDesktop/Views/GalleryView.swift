// GalleryView.swift — Asset gallery with grid, search, and filtering
//
// Displays DAMStore assets as a grid of thumbnails with sorting,
// filtering by favorite/content mode/character, and FTS5 search.
// Clicking a cell opens the AssetDetailView for full metadata
// display and editing. Phase 4: Added drag-and-drop, comparison
// selection, Quick Look via Space bar.

import SwiftUI
import AppKit

/// Sort options for the gallery.
enum GallerySortOrder: String, CaseIterable {
    case date = "Date"
    case rating = "Rating"
    case favorite = "Favorites First"
}

struct GalleryView: View {
    let store: DAMStore
    let ingestor: AssetIngestor
    var onCompare: (([DAMAsset]) -> Void)?
    /// Incremented by the app's Cmd+F command; consumed to focus search.
    @Binding var searchFocusRequests: Int

    @State private var assets: [DAMAsset] = []
    @State private var searchText: String = ""
    @State private var sortOrder: GallerySortOrder = .date
    @State private var filterFavorites: Bool = false
    @State private var filterContentMode: String?
    @State private var filterCharacter: String?
    @State private var selectedAsset: DAMAsset?
    @State private var lightboxIndex: Int? = nil
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    // Multi-selection (compare, bulk delete)
    @State private var selectedIds: Set<String> = []
    @State private var isSelectMode: Bool = false
    @State private var pendingDelete: [DAMAsset] = []
    @State private var showDeleteConfirmation: Bool = false

    // Available filter values extracted from assets.
    @State private var contentModes: [String] = []
    @State private var characters: [String] = []

    // Virtual folders
    @State private var folders: [DAMFolder] = []
    @State private var folderCounts: [String: Int] = [:]
    @State private var folderAssignments: [String: String] = [:]
    @State private var folderFilter: FolderFilter = .all
    @State private var showNewFolderPrompt: Bool = false
    @State private var newFolderName: String = ""
    /// Assets staged to be filed into the folder created by the prompt.
    @State private var pendingFolderAssets: [String] = []
    @State private var renamingFolder: DAMFolder?
    @State private var renameText: String = ""

    enum FolderFilter: Equatable, Hashable {
        case all
        case unfiled
        case folder(String)
    }

    // Search field focus
    @FocusState private var searchFieldFocused: Bool

    private let columns = [GridItem(.adaptive(minimum: 180, maximum: 240), spacing: 12)]

    var body: some View {
        HStack(spacing: 0) {
            folderSidebar
                .frame(width: 190)
            Divider()
            VStack(spacing: 0) {
                // Toolbar: search, sort, filter
                toolbarView
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(nsColor: .windowBackgroundColor))

                Divider()

                if let message = errorMessage {
                    errorBanner(message)
                }

                // Gallery grid
                if isLoading {
                    VStack(spacing: 12) {
                        ProgressView()
                            .controlSize(.large)
                        Text("Loading gallery...")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if filteredAssets.isEmpty {
                    emptyState
                } else {
                    galleryGrid
                }
            }
        }
        .alert("New Folder", isPresented: $showNewFolderPrompt) {
            TextField("Folder name", text: $newFolderName)
            Button("Create") {
                let name = newFolderName.trimmingCharacters(in: .whitespacesAndNewlines)
                let staged = pendingFolderAssets
                newFolderName = ""
                pendingFolderAssets = []
                guard !name.isEmpty else { return }
                Task { await createFolder(named: name, filing: staged) }
            }
            Button("Cancel", role: .cancel) {
                newFolderName = ""
                pendingFolderAssets = []
            }
        }
        .alert("Rename Folder", isPresented: Binding(
            get: { renamingFolder != nil },
            set: { if !$0 { renamingFolder = nil } }
        )) {
            TextField("Folder name", text: $renameText)
            Button("Rename") {
                if let folder = renamingFolder {
                    let name = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !name.isEmpty {
                        Task { await renameFolder(folder, to: name) }
                    }
                }
                renamingFolder = nil
            }
            Button("Cancel", role: .cancel) { renamingFolder = nil }
        }
        .sheet(item: $selectedAsset) { asset in
            AssetDetailView(
                asset: asset,
                thumbnailPath: ingestor.thumbnailPath(for: asset.id),
                onUpdate: { updated in
                    Task { await updateAsset(updated) }
                    selectedAsset = nil
                }
            )
            .frame(minWidth: 800, minHeight: 500)
        }
        .overlay {
            if let idx = lightboxIndex {
                GalleryLightbox(
                    assets: filteredAssets,
                    index: idx,
                    onIndexChange: { lightboxIndex = $0 },
                    onClose: { lightboxIndex = nil }
                )
                .transition(.opacity)
            }
        }
        .task(id: searchText) {
            // Debounce while the user is typing, then re-run the FTS query.
            if !searchText.isEmpty {
                try? await Task.sleep(for: .milliseconds(250))
                if Task.isCancelled { return }
            }
            await loadAssets()
        }
        .onChange(of: ingestor.ingestedCount) { _, _ in
            Task { await loadAssets() }
        }
        .onAppear { consumeSearchFocusRequest() }
        .onChange(of: searchFocusRequests) { _, _ in consumeSearchFocusRequest() }
        .onKeyPress(.space) {
            if let asset = selectedAsset {
                quickLookAsset(asset)
            }
            return .handled
        }
        .confirmationDialog(
            pendingDelete.count == 1
                ? "Delete \"\(pendingDelete.first?.filename ?? "")\"?"
                : "Delete \(pendingDelete.count) images?",
            isPresented: $showDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Move to Trash", role: .destructive) {
                let toDelete = pendingDelete
                pendingDelete = []
                Task { await deleteAssets(toDelete) }
            }
            Button("Cancel", role: .cancel) {
                pendingDelete = []
            }
        } message: {
            Text("The image files and their metadata are moved to the Trash.")
        }
    }

    // MARK: - Toolbar

    private var toolbarView: some View {
        HStack(spacing: 12) {
            // Search field
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search prompts...", text: $searchText)
                    .textFieldStyle(.plain)
                    .focused($searchFieldFocused)
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .frame(maxWidth: 300)

            Spacer()

            // Selection mode toggle
            Toggle(isOn: $isSelectMode) {
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.circle")
                    if isSelectMode && !selectedIds.isEmpty {
                        Text("\(selectedIds.count)")
                            .font(.caption2)
                    }
                }
            }
            .toggleStyle(.button)
            .help("Toggle selection mode for compare and bulk actions")
            .onChange(of: isSelectMode) { _, newValue in
                if !newValue { selectedIds.removeAll() }
            }

            if isSelectMode {
                Button("All") { selectedIds = Set(filteredAssets.map(\.id)) }
                    .controlSize(.small)
                    .help("Select all visible images")

                if !selectedIds.isEmpty {
                    Button("None") { selectedIds.removeAll() }
                        .controlSize(.small)
                        .help("Clear selection")
                }

                // Compare button (2-4 selected)
                if (2...4).contains(selectedIds.count) {
                    Button(action: { sendToComparison() }) {
                        Label("Compare \(selectedIds.count)", systemImage: "square.grid.2x2")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }

                // Bulk move to folder
                if !selectedIds.isEmpty {
                    Menu {
                        moveToFolderMenuItems(for: Array(selectedIds))
                    } label: {
                        Label("Move \(selectedIds.count)", systemImage: "folder")
                    }
                    .controlSize(.small)
                    .fixedSize()
                }

                // Bulk delete
                if !selectedIds.isEmpty {
                    Button(role: .destructive, action: { requestDelete(selectedAssetsList) }) {
                        Label("Delete \(selectedIds.count)", systemImage: "trash")
                    }
                    .controlSize(.small)
                }
            }

            // Favorite filter
            Toggle(isOn: $filterFavorites) {
                Image(systemName: filterFavorites ? "heart.fill" : "heart")
            }
            .toggleStyle(.button)
            .help("Show favorites only")

            // Content mode filter
            if !contentModes.isEmpty {
                Picker("Mode", selection: Binding(
                    get: { filterContentMode ?? "" },
                    set: { filterContentMode = $0.isEmpty ? nil : $0 }
                )) {
                    Text("All Modes").tag("")
                    ForEach(contentModes, id: \.self) { mode in
                        Text(mode.capitalized).tag(mode)
                    }
                }
                .frame(width: 120)
            }

            // Character filter
            if !characters.isEmpty {
                Picker("Character", selection: Binding(
                    get: { filterCharacter ?? "" },
                    set: { filterCharacter = $0.isEmpty ? nil : $0 }
                )) {
                    Text("All Characters").tag("")
                    ForEach(characters, id: \.self) { name in
                        Text(name).tag(name)
                    }
                }
                .frame(width: 120)
            }

            // Sort picker
            Picker("Sort", selection: $sortOrder) {
                ForEach(GallerySortOrder.allCases, id: \.self) { order in
                    Text(order.rawValue).tag(order)
                }
            }
            .frame(width: 140)

            // Refresh button
            Button(action: { Task { await loadAssets() } }) {
                Image(systemName: "arrow.clockwise")
            }
            .help("Refresh gallery")

            // Asset count
            Text("\(filteredAssets.count) images")
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    // MARK: - Gallery Grid

    private var galleryGrid: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(filteredAssets) { asset in
                    let isSelected = selectedIds.contains(asset.id)
                    GalleryCellView(
                        asset: asset,
                        thumbnailPath: ingestor.thumbnailPath(for: asset.id),
                        isComparisonSelected: isSelectMode ? isSelected : nil
                    )
                    .onTapGesture {
                        if isSelectMode {
                            toggleSelection(asset)
                        } else {
                            selectedAsset = asset
                        }
                    }
                    .contextMenu {
                        Button("Open in Lightbox") {
                            lightboxIndex = filteredAssets.firstIndex(where: { $0.id == asset.id })
                        }
                        Button("Reveal in Finder") {
                            revealInFinder(asset)
                        }
                        Button(asset.favorite ? "Unfavorite" : "Favorite") {
                            Task { await toggleFavorite(asset) }
                        }
                        Menu("Move to Folder") {
                            moveToFolderMenuItems(for: isSelectMode && selectedIds.contains(asset.id)
                                ? Array(selectedIds) : [asset.id])
                        }
                        if !isSelectMode {
                            Divider()
                            Button("Select") {
                                selectedIds.insert(asset.id)
                                isSelectMode = true
                            }
                        }
                        Divider()
                        Button("Delete…", role: .destructive) {
                            requestDelete([asset])
                        }
                    }
                    .draggable(DraggableAsset(path: asset.absolutePath))
                }
            }
            .padding(12)
        }
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

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)

            if !searchText.isEmpty {
                Text("No results for \"\(searchText)\"")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                Text("Try a different search term.")
                    .font(.body)
                    .foregroundStyle(.tertiary)
            } else if filterFavorites {
                Text("No favorites yet")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                Text("Heart an image to see it here.")
                    .font(.body)
                    .foregroundStyle(.tertiary)
            } else {
                Text("No images in gallery")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                Text("Generate an image or configure the watch directory.")
                    .font(.body)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Folder sidebar

    private var folderSidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            List(selection: $folderFilter) {
                folderRow(label: "All Images", icon: "photo.on.rectangle",
                          count: assets.count, tag: .all)
                folderRow(label: "Unfiled", icon: "tray",
                          count: max(0, assets.count - folderAssignments.count), tag: .unfiled)
                if !folders.isEmpty {
                    Section("Folders") {
                        ForEach(folders) { folder in
                            folderRow(label: folder.name, icon: "folder",
                                      count: folderCounts[folder.id] ?? 0, tag: .folder(folder.id))
                                .contextMenu {
                                    Button("Rename…") {
                                        renameText = folder.name
                                        renamingFolder = folder
                                    }
                                    Button("Delete Folder", role: .destructive) {
                                        Task { await deleteFolder(folder) }
                                    }
                                }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)

            Divider()
            Button {
                pendingFolderAssets = []
                showNewFolderPrompt = true
            } label: {
                Label("New Folder", systemImage: "folder.badge.plus")
                    .font(.callout)
            }
            .buttonStyle(.borderless)
            .padding(10)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func folderRow(label: String, icon: String, count: Int, tag: FolderFilter) -> some View {
        HStack {
            Label(label, systemImage: icon)
                .lineLimit(1)
            Spacer()
            Text("\(count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .tag(tag)
    }

    // MARK: - Folder actions

    private func createFolder(named name: String, filing assetIds: [String]) async {
        do {
            let folder = try await store.createFolder(name: name)
            if !assetIds.isEmpty {
                try await store.assignAssets(ids: assetIds, toFolder: folder.id)
                selectedIds.removeAll()
            }
            await loadAssets()
            folderFilter = .folder(folder.id)
        } catch {
            errorMessage = "Failed to create folder: \(error.localizedDescription)"
        }
    }

    private func renameFolder(_ folder: DAMFolder, to name: String) async {
        do {
            try await store.renameFolder(id: folder.id, name: name)
            await loadAssets()
        } catch {
            errorMessage = "Failed to rename folder: \(error.localizedDescription)"
        }
    }

    private func deleteFolder(_ folder: DAMFolder) async {
        do {
            try await store.deleteFolder(id: folder.id)
            if folderFilter == .folder(folder.id) { folderFilter = .all }
            await loadAssets()
        } catch {
            errorMessage = "Failed to delete folder: \(error.localizedDescription)"
        }
    }

    /// File assets into a folder (nil = unfile) and refresh.
    private func moveAssets(_ assetIds: [String], toFolder folderId: String?) async {
        guard !assetIds.isEmpty else { return }
        do {
            try await store.assignAssets(ids: assetIds, toFolder: folderId)
            selectedIds.removeAll()
            await loadAssets()
        } catch {
            errorMessage = "Failed to move: \(error.localizedDescription)"
        }
    }

    /// The shared "Move to Folder" menu body for context menus and the toolbar.
    @ViewBuilder
    private func moveToFolderMenuItems(for assetIds: [String]) -> some View {
        ForEach(folders) { folder in
            Button(folder.name) {
                Task { await moveAssets(assetIds, toFolder: folder.id) }
            }
        }
        if !folders.isEmpty { Divider() }
        Button("New Folder…") {
            pendingFolderAssets = assetIds
            showNewFolderPrompt = true
        }
        if assetIds.contains(where: { folderAssignments[$0] != nil }) {
            Divider()
            Button("Remove from Folder") {
                Task { await moveAssets(assetIds, toFolder: nil) }
            }
        }
    }

    // MARK: - Filtering and Sorting

    private var filteredAssets: [DAMAsset] {
        var results = assets

        // Apply folder filter.
        switch folderFilter {
        case .all:
            break
        case .unfiled:
            results = results.filter { folderAssignments[$0.id] == nil }
        case .folder(let folderId):
            results = results.filter { folderAssignments[$0.id] == folderId }
        }

        // Apply favorites filter.
        if filterFavorites {
            results = results.filter { $0.favorite }
        }

        // Apply content mode filter.
        if let mode = filterContentMode {
            results = results.filter { $0.contentMode == mode }
        }

        // Apply character filter.
        if let character = filterCharacter {
            results = results.filter { $0.characterName == character }
        }

        // Apply search — client-side for immediate feedback, FTS for big datasets.
        if !searchText.isEmpty {
            let query = searchText.lowercased()
            results = results.filter { asset in
                (asset.prompt?.lowercased().contains(query) ?? false)
                    || (asset.negativePrompt?.lowercased().contains(query) ?? false)
                    || asset.filename.lowercased().contains(query)
                    || (asset.characterName?.lowercased().contains(query) ?? false)
            }
        }

        // Apply sort.
        switch sortOrder {
        case .date:
            results.sort { $0.createdAt > $1.createdAt }
        case .rating:
            results.sort { $0.rating > $1.rating }
        case .favorite:
            results.sort {
                if $0.favorite != $1.favorite { return $0.favorite }
                return $0.createdAt > $1.createdAt
            }
        }

        return results
    }

    // MARK: - Data Loading

    private func loadAssets() async {
        isLoading = assets.isEmpty
        do {
            if !searchText.isEmpty {
                let ftsResults = try await store.searchPrompts(query: searchText, limit: 200)
                assets = ftsResults
            } else {
                assets = try await store.fetchAssets(limit: 500)
            }
            folders = try await store.listFolders()
            folderCounts = try await store.folderCounts()
            folderAssignments = try await store.folderAssignments()
            extractFilterValues()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    private func extractFilterValues() {
        contentModes = Array(Set(assets.compactMap { $0.contentMode })).sorted()
        characters = Array(Set(assets.compactMap { $0.characterName })).sorted()
    }

    // MARK: - Actions

    /// Focus the search field (called by Cmd+F keyboard shortcut).
    func focusSearch() {
        searchFieldFocused = true
    }

    /// Consume a pending Cmd+F focus request from the app commands.
    private func consumeSearchFocusRequest() {
        guard searchFocusRequests > 0 else { return }
        searchFocusRequests = 0
        // Defer a tick so the field is in the hierarchy before focusing.
        Task { focusSearch() }
    }

    private func updateAsset(_ asset: DAMAsset) async {
        do {
            try await store.insertAsset(asset)
            await loadAssets()
        } catch {
            errorMessage = "Failed to update: \(error.localizedDescription)"
        }
    }

    private func toggleFavorite(_ asset: DAMAsset) async {
        let updated = DAMAsset(
            id: asset.id,
            kind: asset.kind,
            filename: asset.filename,
            absolutePath: asset.absolutePath,
            fileSize: asset.fileSize,
            sha256: asset.sha256,
            width: asset.width,
            height: asset.height,
            createdAt: asset.createdAt,
            modifiedAt: Date(),
            ingestedAt: asset.ingestedAt,
            orphaned: asset.orphaned,
            prompt: asset.prompt,
            negativePrompt: asset.negativePrompt,
            seed: asset.seed,
            steps: asset.steps,
            guidance: asset.guidance,
            modelFamily: asset.modelFamily,
            rating: asset.rating,
            favorite: !asset.favorite,
            contentMode: asset.contentMode,
            characterName: asset.characterName
        )
        await updateAsset(updated)
    }

    private func revealInFinder(_ asset: DAMAsset) {
        NSWorkspace.shared.selectFile(
            asset.absolutePath,
            inFileViewerRootedAtPath: ""
        )
    }

    /// Assets for the current selection, in display order.
    private var selectedAssetsList: [DAMAsset] {
        filteredAssets.filter { selectedIds.contains($0.id) }
    }

    private func toggleSelection(_ asset: DAMAsset) {
        if selectedIds.contains(asset.id) {
            selectedIds.remove(asset.id)
        } else {
            selectedIds.insert(asset.id)
        }
    }

    private func sendToComparison() {
        onCompare?(selectedAssetsList)
    }

    /// Stage assets for deletion and show the confirmation dialog.
    private func requestDelete(_ toDelete: [DAMAsset]) {
        guard !toDelete.isEmpty else { return }
        pendingDelete = toDelete
        showDeleteConfirmation = true
    }

    /// Delete assets via the ingestor (trash file + sidecar, drop thumbnail
    /// and database row), then reload. Failures surface in the error banner
    /// but do not stop the remaining deletions.
    private func deleteAssets(_ toDelete: [DAMAsset]) async {
        var failures: [String] = []
        for asset in toDelete {
            do {
                try await ingestor.deleteAsset(asset)
                selectedIds.remove(asset.id)
            } catch {
                failures.append(asset.filename)
            }
        }
        if !failures.isEmpty {
            errorMessage = "Failed to delete: \(failures.joined(separator: ", "))"
        }
        lightboxIndex = nil
        await loadAssets()
    }

    /// Open Quick Look for an asset using macOS native preview.
    private func quickLookAsset(_ asset: DAMAsset) {
        let url = URL(fileURLWithPath: asset.absolutePath)
        NSWorkspace.shared.open(url)
    }
}

// MARK: - Draggable Asset (for drag-and-drop to Finder)

struct DraggableAsset: Transferable {
    let path: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .png) { asset in
            SentTransferredFile(URL(fileURLWithPath: asset.path))
        }
    }
}

// MARK: - Gallery Cell

struct GalleryCellView: View {
    let asset: DAMAsset
    let thumbnailPath: String
    var isComparisonSelected: Bool?

    @State private var thumbnail: NSImage?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            VStack(spacing: 4) {
                // Thumbnail
                ZStack {
                    Color(nsColor: .controlBackgroundColor)

                    if let thumb = thumbnail {
                        Image(nsImage: thumb)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    } else {
                        Image(systemName: "photo")
                            .font(.title)
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(height: 160)
                .clipShape(RoundedRectangle(cornerRadius: 8))

                // Prompt preview
                if let prompt = asset.prompt {
                    Text(prompt)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .truncationMode(.tail)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                // Rating + favorite
                HStack(spacing: 4) {
                    if asset.favorite {
                        Image(systemName: "heart.fill")
                            .font(.caption2)
                            .foregroundStyle(.red)
                    }

                    if asset.rating > 0 {
                        HStack(spacing: 1) {
                            ForEach(1...5, id: \.self) { star in
                                Image(systemName: star <= asset.rating ? "star.fill" : "star")
                                    .font(.system(size: 8))
                                    .foregroundStyle(star <= asset.rating ? .yellow : .gray.opacity(0.3))
                            }
                        }
                    }

                    Spacer()

                    if let mode = asset.contentMode {
                        Text(mode)
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                }
            }

            // Comparison selection overlay
            if let isSelected = isComparisonSelected {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? .white : .secondary, isSelected ? Color.accentColor : Color.clear)
                    .font(.title3)
                    .padding(6)
            }
        }
        .padding(6)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(
                    isComparisonSelected == true ? Color.accentColor : Color.clear,
                    lineWidth: 2
                )
        )
        .task {
            await loadThumbnail()
        }
    }

    private func loadThumbnail() async {
        let path = thumbnailPath
        let fullPath = asset.absolutePath
        let image: NSImage? = await Task.detached {
            NSImage(contentsOfFile: path) ?? NSImage(contentsOfFile: fullPath)
        }.value
        await MainActor.run {
            thumbnail = image
        }
    }
}

// MARK: - Lightbox

/// Full-screen zoomable image viewer with keyboard/on-screen prev-next navigation.
private struct GalleryLightbox: View {
    let assets: [DAMAsset]
    let index: Int
    let onIndexChange: (Int) -> Void
    let onClose: () -> Void

    @State private var image: NSImage?
    @State private var zoom: CGFloat = 1
    @State private var baseZoom: CGFloat = 1
    @State private var offset: CGSize = .zero
    @FocusState private var focused: Bool

    private var asset: DAMAsset? { assets.indices.contains(index) ? assets[index] : nil }

    var body: some View {
        ZStack {
            Color.black.opacity(0.93).ignoresSafeArea()
                .onTapGesture { onClose() }

            Group {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .scaleEffect(zoom)
                        .offset(offset)
                        .gesture(
                            MagnifyGesture()
                                .onChanged { zoom = min(6, max(1, baseZoom * $0.magnification)) }
                                .onEnded { _ in baseZoom = zoom }
                        )
                        .highPriorityGesture(
                            DragGesture()
                                .onChanged { if zoom > 1 { offset = $0.translation } }
                        )
                        .onTapGesture(count: 2) {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                zoom = zoom > 1 ? 1 : 2
                                baseZoom = zoom
                                offset = .zero
                            }
                        }
                } else {
                    ProgressView().controlSize(.large).tint(.white)
                }
            }
            .padding(40)

            // Prev / next
            HStack {
                navButton("chevron.left") { step(-1) }.opacity(index > 0 ? 1 : 0.25).disabled(index <= 0)
                Spacer()
                navButton("chevron.right") { step(1) }.opacity(index < assets.count - 1 ? 1 : 0.25).disabled(index >= assets.count - 1)
            }
            .padding(.horizontal, 20)

            // Close + caption chrome
            VStack {
                HStack {
                    Text("\(index + 1) / \(assets.count)")
                        .font(.callout.monospacedDigit()).foregroundStyle(.white.opacity(0.7))
                    Spacer()
                    Button { onClose() } label: {
                        Image(systemName: "xmark.circle.fill").font(.title)
                    }
                    .buttonStyle(.plain).foregroundStyle(.white.opacity(0.85))
                }
                .padding(16)
                Spacer()
                if let p = asset?.prompt, !p.isEmpty {
                    Text(p)
                        .font(.callout).foregroundStyle(.white.opacity(0.9))
                        .lineLimit(3).multilineTextAlignment(.center)
                        .padding(12)
                        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                        .frame(maxWidth: 760)
                        .padding(.bottom, 28)
                }
            }
        }
        .focusable()
        .focused($focused)
        .onKeyPress(.leftArrow) { step(-1); return .handled }
        .onKeyPress(.rightArrow) { step(1); return .handled }
        .onKeyPress(.escape) { onClose(); return .handled }
        .task(id: index) { await load() }
        .onAppear { focused = true }
    }

    private func navButton(_ icon: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(.white)
                .padding(14)
                .background(.black.opacity(0.4), in: Circle())
        }
        .buttonStyle(.plain)
    }

    private func step(_ delta: Int) {
        let next = index + delta
        guard assets.indices.contains(next) else { return }
        zoom = 1; baseZoom = 1; offset = .zero
        onIndexChange(next)
    }

    private func load() async {
        guard let path = asset?.absolutePath else { image = nil; return }
        image = await Task.detached { NSImage(contentsOfFile: path) }.value
    }
}
