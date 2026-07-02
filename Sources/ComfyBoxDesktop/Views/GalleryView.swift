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
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    // Comparison selection
    @State private var comparisonSelection: Set<String> = []
    @State private var isComparisonMode: Bool = false

    // Available filter values extracted from assets.
    @State private var contentModes: [String] = []
    @State private var characters: [String] = []

    // Search field focus
    @FocusState private var searchFieldFocused: Bool

    private let columns = [GridItem(.adaptive(minimum: 180, maximum: 240), spacing: 12)]

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar: search, sort, filter
            toolbarView
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(nsColor: .windowBackgroundColor))

            Divider()

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

            // Comparison mode toggle
            Toggle(isOn: $isComparisonMode) {
                HStack(spacing: 4) {
                    Image(systemName: "square.grid.2x2")
                    if isComparisonMode && !comparisonSelection.isEmpty {
                        Text("\(comparisonSelection.count)")
                            .font(.caption2)
                    }
                }
            }
            .toggleStyle(.button)
            .help("Toggle comparison selection mode")
            .onChange(of: isComparisonMode) { _, newValue in
                if !newValue { comparisonSelection.removeAll() }
            }

            // Compare button (visible when 2+ selected)
            if isComparisonMode && comparisonSelection.count >= 2 {
                Button(action: { sendToComparison() }) {
                    Label("Compare \(comparisonSelection.count)", systemImage: "arrow.right.circle")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
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
                    let isCompSelected = comparisonSelection.contains(asset.id)
                    GalleryCellView(
                        asset: asset,
                        thumbnailPath: ingestor.thumbnailPath(for: asset.id),
                        isComparisonSelected: isComparisonMode ? isCompSelected : nil
                    )
                    .onTapGesture {
                        if isComparisonMode {
                            toggleComparisonSelection(asset)
                        } else {
                            selectedAsset = asset
                        }
                    }
                    .contextMenu {
                        Button("Reveal in Finder") {
                            revealInFinder(asset)
                        }
                        Button(asset.favorite ? "Unfavorite" : "Favorite") {
                            Task { await toggleFavorite(asset) }
                        }
                        if !isComparisonMode {
                            Divider()
                            Button("Add to Comparison") {
                                if comparisonSelection.count < 4 {
                                    comparisonSelection.insert(asset.id)
                                    isComparisonMode = true
                                }
                            }
                        }
                    }
                    .draggable(DraggableAsset(path: asset.absolutePath))
                }
            }
            .padding(12)
        }
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

    // MARK: - Filtering and Sorting

    private var filteredAssets: [DAMAsset] {
        var results = assets

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

    private func toggleComparisonSelection(_ asset: DAMAsset) {
        if comparisonSelection.contains(asset.id) {
            comparisonSelection.remove(asset.id)
        } else if comparisonSelection.count < 4 {
            comparisonSelection.insert(asset.id)
        }
    }

    private func sendToComparison() {
        let selected = assets.filter { comparisonSelection.contains($0.id) }
        onCompare?(selected)
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
