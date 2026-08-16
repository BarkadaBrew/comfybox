// GalleryView.swift — Asset gallery with grid, search, and filtering
//
// Displays assets as a grid of thumbnails with sorting, filtering by
// favorite/content mode/character, and search. Clicking a cell opens the
// AssetDetailView for full metadata display and editing. Phase 4: Added
// drag-and-drop, comparison selection, Quick Look via Space bar.
//
// ONE GALLERY (2026-07-31). The rows now come from the CATALOG via
// `CatalogBrowser` — one query over every realm — instead of from a local-only
// DAMStore fetch, and a row's `asset_locations` decide whether its bytes open
// from disk or stream from the engine's /v1/gallery/file. That retires the
// split between this view and RemoteGalleryView, which read a bare directory
// listing with no metadata. DAMStore is unchanged and still owns ingest,
// ratings, favourites, folders and the secure vault; only browsing moved.
//
// The catalog and the DAM share ONE `assets` table (dam.sqlite3 was migrated in
// place), so ids are one id space and everything keyed by an id still matches.
// When the catalog cannot be opened this view falls back to the DAMStore fetch
// it always used, so a broken catalog degrades to the old gallery rather than
// to an empty one.

import SwiftUI
import AVKit
import AppKit
import LocalAuthentication
import ComfyBoxCatalog

/// Sort options for the gallery.
enum GallerySortOrder: String, CaseIterable {
    case date = "Date"
    case rating = "Rating"
    case favorite = "Favorites First"
}

struct GalleryView: View {
    let store: DAMStore
    let ingestor: AssetIngestor
    /// Archive engine (moves assets to a `.cbarchive` bundle). Optional so
    /// previews/tests can omit it.
    var archiver: GalleryArchiver?
    /// For server-backed actions (upscale). Optional so previews/tests can omit.
    var engine: EngineService?
    var onCompare: (([DAMAsset]) -> Void)?
    /// Send an image to Generate as an img2img reference.
    var onUseAsReference: ((DAMAsset) -> Void)?
    /// Send an image's full recipe (prompt, params, LoRAs, content mode) to Generate.
    var onSendToGenerate: ((DAMAsset) -> Void)?
    /// Send an image to the Motion tab as an image-to-video reference.
    var onAnimate: ((DAMAsset) -> Void)?
    /// Send an image to the Inpaint tab.
    var onInpaint: ((DAMAsset) -> Void)?
    /// Canvas projects images can be added to (Add to Canvas menu).
    var canvasStore: CanvasStore?
    /// Incremented by the app's Cmd+F command; consumed to focus search.
    @Binding var searchFocusRequests: Int
    /// Incremented by the "Gallery Health" palette command; consumed to open
    /// the maintenance sheet. Same counter-binding trick as `searchFocusRequests`.
    @Binding var maintenanceRequests: Int

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
    /// Separate from `errorMessage`, which a successful load clears: a refused
    /// orphan sweep is the one thing the user must still see afterwards.
    @State private var pruneWarning: String?

    // Multi-selection (compare, bulk delete)
    @State private var selectedIds: Set<String> = []
    @State private var isSelectMode: Bool = false
    // Grid page size. Select All raises it to the full scope for the rest of
    // the session so the selection is truthful, not capped at one page.
    @State private var loadLimit = 500
    private static let fullScopeLimit = 20_000
    @State private var mediaTools = MediaToolsService()
    @State private var sidecar = SidecarService()

    // The one reader. nil until the catalog opens (and permanently nil if it
    // cannot), in which case the DAMStore fetch below still serves the grid.
    @State private var browser: CatalogBrowser?
    @State private var selectedCollectionID: String?
    @State private var selectedLane: String?
    /// Engine URLs for rows whose bytes are not on this Mac, keyed by asset id.
    @State private var remoteURLs: [String: URL] = [:]
    @State private var pendingDelete: [DAMAsset] = []
    @State private var showDeleteConfirmation: Bool = false

    // Available filter values extracted from assets.
    @State private var contentModes: [String] = []
    @State private var characters: [String] = []

    // Secured (sensitive) assets — hidden until unlocked with Touch ID.
    @State private var securedIds: Set<String> = []
    @State private var revealSecured: Bool = false

    // App-wide "Rated G unless NSFW toggled" master gate (Todd 2026-07-17).
    // When hidden, the whole gallery is walled off regardless of the per-mode
    // filter below.
    @Environment(AppContentGate.self) private var contentGate

    // NSFW content gate (content-mode based, unlocked by a gallery password).
    @State private var nsfwMode: NSFWFilterMode = .blur
    @State private var nsfwUnlocked: Bool = false
    @State private var showNSFWPasswordSheet: Bool = false
    @State private var nsfwPasswordInput: String = ""
    @State private var nsfwPasswordError: Bool = false

    // Finder color labels (the file's own tags are the source of truth).
    @State private var colorLabels: [String: FinderColor] = [:]
    @State private var filterLabel: FinderColor?

    // Virtual folders
    @State private var folders: [DAMFolder] = []
    @State private var folderCounts: [String: Int] = [:]
    @State private var folderAssignments: [String: String] = [:]
    @State private var folderFilter: FolderFilter = .all
    /// Persona section filter: nil = main gallery (excludes agent/persona renders);
    /// a value shows only that persona's (source's) images.
    @State private var personaFilter: String?

    /// Sources that belong to the main gallery, not a persona section.
    ///
    /// "comfybox" is in here because it is the APP's own name, not a persona:
    /// the catalog backfill stamps it on 2,907 of the 2,994 rows in the live
    /// database, so leaving it out filed almost the whole library into a
    /// "Comfybox" persona section and left the main gallery showing 87 images.
    static let mainSources: Set<String> = ["", "desktop", "comfyui", "comfybox"]
    static func isMainSource(_ source: String?) -> Bool {
        mainSources.contains((source ?? "").lowercased())
    }
    /// Distinct persona (non-main) sources present in the library, with counts.
    private var personaSources: [(name: String, count: Int)] {
        var counts: [String: Int] = [:]
        for a in assets where !Self.isMainSource(a.source) {
            counts[(a.source ?? "").lowercased(), default: 0] += 1
        }
        return counts.sorted { $0.key < $1.key }.map { (name: $0.key, count: $0.value) }
    }
    @State private var showNewFolderPrompt: Bool = false
    @State private var newFolderName: String = ""
    /// Assets staged to be filed into the folder created by the prompt.
    @State private var pendingFolderAssets: [String] = []
    @State private var renamingFolder: DAMFolder?
    @State private var renameText: String = ""

    // Smart tabs — saved filter combinations.
    @State private var smartTabs: [SmartTab] = []
    @State private var showSaveTabPrompt: Bool = false
    @State private var newTabName: String = ""
    @State private var renamingSmartTab: SmartTab?
    @State private var renameSmartTabText: String = ""

    enum FolderFilter: Equatable, Hashable {
        case all
        case unfiled
        case folder(String)
    }

    // Folder import (Photo Mechanic-style "add existing folder")
    @State private var importProgress: (done: Int, total: Int)?
    @State private var importSummary: String?

    // Gallery Health / maintenance sheet (T11)
    @State private var showMaintenance: Bool = false

    // Archiving (moves assets to a .cbarchive bundle)
    @State private var showArchiveSheet: Bool = false
    @State private var archiveTargets: [DAMAsset] = []
    @State private var archiveFolder: DAMFolder?
    @State private var archiveProgress: (done: Int, total: Int)?
    @State private var archiveSummary: String?

    // Search field focus
    @FocusState private var searchFieldFocused: Bool

    /// Target masonry column width; the size control steps it.
    @State private var cellTargetWidth: CGFloat = 210

    /// Whether the collection / lane rail is on screen.
    ///
    /// The rail names bodies of work, and in the live catalog those names are
    /// "Adult", "Adult Scenes", "Erotic Portraiture", "Nightlife". They are
    /// exactly what the Rated-G-by-default gate exists to keep off the screen,
    /// so the rail is hidden with the grid rather than left standing beside the
    /// lock wall. A fresh gate is hidden, so this is false on every launch.
    static func showsCatalogRail(revealed: Bool, hasBrowser: Bool) -> Bool {
        revealed && hasBrowser
    }

    var body: some View {
        HStack(spacing: 0) {
            // NO TEST GUARDS THIS LINE. `showsCatalogRail` is unit-tested, but
            // nothing checks that the rail actually consults it (no ViewInspector
            // in this repo) — delete the `if` and every test still passes while
            // the gate is wide open. Change it by hand, carefully.
            if Self.showsCatalogRail(revealed: contentGate.revealed, hasBrowser: browser != nil) {
                catalogRail
                    .frame(width: 210)
                Divider()
            }
            folderSidebar
                .frame(width: 190)
            Divider()
            VStack(spacing: 0) {
                // Toolbar: search, sort, filter
                toolbarView
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(nsColor: .windowBackgroundColor))

                smartTabsRow
                    .background(Color(nsColor: .windowBackgroundColor))

                Divider()

                if let message = errorMessage {
                    errorBanner(message)
                }
                if let warning = pruneWarning {
                    errorBanner(warning) { pruneWarning = nil }
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
                } else if !contentGate.revealed {
                    ContentHiddenWall(note: "The gallery holds generated content.")
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
        .alert("Save Current View", isPresented: $showSaveTabPrompt) {
            TextField("Name", text: $newTabName)
            Button("Save") {
                let name = newTabName.trimmingCharacters(in: .whitespacesAndNewlines)
                newTabName = ""
                guard !name.isEmpty else { return }
                saveCurrentAsSmartTab(named: name)
            }
            Button("Cancel", role: .cancel) { newTabName = "" }
        } message: {
            Text("Saves the current search, filters, and sort order as a tab you can jump back to.")
        }
        .alert("Rename Tab", isPresented: Binding(
            get: { renamingSmartTab != nil },
            set: { if !$0 { renamingSmartTab = nil } }
        )) {
            TextField("Name", text: $renameSmartTabText)
            Button("Rename") {
                if let tab = renamingSmartTab {
                    let name = renameSmartTabText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !name.isEmpty { renameSmartTab(tab, to: name) }
                }
                renamingSmartTab = nil
            }
            Button("Cancel", role: .cancel) { renamingSmartTab = nil }
        }
        .sheet(item: $selectedAsset) { asset in
            AssetDetailView(
                assets: filteredAssets,
                index: filteredAssets.firstIndex(where: { $0.id == asset.id }) ?? 0,
                thumbnailProvider: { ingestor.thumbnailPath(for: $0.id) },
                mediaLocationProvider: { mediaLocation(for: $0) },
                onUpdate: { updated in
                    Task { await updateAsset(updated) }
                },
                onFullScreen: { target in
                    selectedAsset = nil
                    lightboxIndex = filteredAssets.firstIndex(where: { $0.id == target.id })
                },
                onSendToGenerate: onSendToGenerate
            )
            .frame(minWidth: 800, minHeight: 500)
        }
        .sheet(isPresented: $showNSFWPasswordSheet) {
            VStack(alignment: .leading, spacing: 12) {
                Label("Unlock NSFW content", systemImage: "eye.trianglebadge.exclamationmark")
                    .font(.headline)
                Text("Enter the gallery password to reveal NSFW content this session.")
                    .font(.caption).foregroundStyle(.secondary)
                SecureField("Password", text: $nsfwPasswordInput)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(submitNSFWPassword)
                if nsfwPasswordError {
                    Label("Incorrect password.", systemImage: "xmark.circle").font(.caption).foregroundStyle(.red)
                }
                HStack {
                    Spacer()
                    Button("Cancel") { showNSFWPasswordSheet = false }
                    Button("Unlock") { submitNSFWPassword() }.buttonStyle(.borderedProminent)
                }
            }
            .padding(18).frame(width: 360)
        }
        .sheet(isPresented: $showArchiveSheet) {
            ArchiveSheet(
                assets: archiveTargets,
                folder: archiveFolder,
                store: store,
                isPresented: $showArchiveSheet,
                onArchive: { name, destinationRoot in
                    performArchive(name: name, destinationRoot: destinationRoot,
                                   assets: archiveTargets, folder: archiveFolder)
                }
            )
        }
        .overlay {
            if let idx = lightboxIndex {
                GalleryLightbox(
                    assets: filteredAssets,
                    index: idx,
                    labelForAsset: { colorLabels[$0.id] },
                    onSetLabel: { asset, color in applyColorLabel(color, to: [asset]) },
                    onCopy: { copyAssets([$0]) },
                    onReveal: { revealInFinder($0) },
                    onSendToGenerate: onSendToGenerate,
                    mediaLocationProvider: { mediaLocation(for: $0) },
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
        .task {
            // Open the catalog once. A failure here is not an error banner: the
            // DAMStore path below still fills the grid, so the gallery degrades
            // to its old (Mac-only) self instead of to nothing.
            guard browser == nil else { return }
            guard let catalog = try? await CatalogStore.open() else { return }
            let b = CatalogBrowser(store: catalog, engineBaseURL: engineBaseURL)
            browser = b
            await loadAssets()
        }
        .onAppear {
            consumeSearchFocusRequest()
            consumeMaintenanceRequest()
            smartTabs = SmartTabStore.load()
        }
        .onChange(of: searchFocusRequests) { _, _ in consumeSearchFocusRequest() }
        .onChange(of: maintenanceRequests) { _, _ in consumeMaintenanceRequest() }
        .sheet(isPresented: $showMaintenance) {
            GalleryMaintenanceView(store: store, ingestor: ingestor, archiver: archiver)
        }
        .onKeyPress(.space) {
            if let asset = selectedAsset {
                quickLookAsset(asset)
            }
            return .handled
        }
        .onKeyPress(characters: CharacterSet(charactersIn: "01234567")) { press in
            // Photo Mechanic-style color classes — skip while typing a search.
            guard !searchFieldFocused, let character = press.characters.first else { return .ignored }
            return handleLabelKey(character) ? .handled : .ignored
        }
        .onKeyPress(keys: ["c"]) { press in
            // Cmd+C copies the lightbox image, the selection, or the hovered
            // detail — skip while typing in the search field.
            guard press.modifiers.contains(.command), !searchFieldFocused else { return .ignored }
            let targets = labelTargets.isEmpty ? (selectedAsset.map { [$0] } ?? []) : labelTargets
            guard !targets.isEmpty else { return .ignored }
            copyAssets(targets)
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
            Text(deleteConfirmationMessage)
        }
    }

    /// What the dialog promises, told truthfully for a mixed selection.
    ///
    /// The grid is catalog-backed now, so a selection can include rows whose
    /// bytes live on the server rather than on this Mac. Those cannot be moved
    /// to this Mac's Trash, and they will NOT be deleted permanently instead —
    /// the dialog has to say so before the button is pressed, not afterwards in
    /// an error banner.
    private var deleteConfirmationMessage: String {
        let base = "The image files and their metadata are moved to the Trash."
        let remote = pendingDelete.filter { !AssetIngestor.isOnThisMac($0.absolutePath) }
        guard !remote.isEmpty else { return base }
        let noun = remote.count == 1 ? "file is" : "files are"
        return base + "\n\n\(remote.count) selected \(noun) hosted on the server, not on this "
            + "Mac. macOS cannot move those to the Trash, so they are left in place and their "
            + "catalog entries are kept."
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
                Button("All") { Task { await selectAll() } }
                    .keyboardShortcut("a", modifiers: .command)
                    .controlSize(.small)
                    .help("Select every image in the current view (⌘A)")

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

                // Export selection to a video (2+ images)
                if selectedIds.count >= 2 && mediaTools.hasFFmpeg {
                    Menu {
                        ForEach([6, 8, 12, 24], id: \.self) { fps in
                            Button("\(fps) fps") { Task { await exportSelectionToVideo(fps: fps) } }
                        }
                    } label: {
                        Label("To Video", systemImage: "film")
                    }
                    .controlSize(.small)
                    .fixedSize()
                }

                // Archive selection
                if !selectedIds.isEmpty {
                    Button { requestArchive(selectedAssetsList, folder: nil) } label: {
                        Label("Archive \(selectedIds.count)", systemImage: "archivebox")
                    }
                    .controlSize(.small)
                }

                // Bulk delete
                if !selectedIds.isEmpty {
                    Button(role: .destructive, action: { requestDelete(selectedAssetsList) }) {
                        Label("Delete \(selectedIds.count)", systemImage: "trash")
                    }
                    .controlSize(.small)
                }
            }

            // Thumbnail size (masonry column width).
            Menu {
                Button("Small") { cellTargetWidth = 150 }
                Button("Medium") { cellTargetWidth = 210 }
                Button("Large") { cellTargetWidth = 300 }
            } label: {
                Image(systemName: "square.grid.2x2")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Thumbnail size")

            // Finder color-label filter.
            Menu {
                Button("Any Label") { filterLabel = nil }
                Divider()
                ForEach(FinderColor.keyboardOrder, id: \.self) { color in
                    Button(color.rawValue) { filterLabel = color }
                }
            } label: {
                Image(systemName: filterLabel == nil ? "circle.dashed" : "circle.fill")
                    .foregroundStyle(filterLabel?.displayColor ?? Color.secondary)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help(filterLabel.map { "Showing \($0.rawValue)-labeled images" } ?? "Filter by Finder color label")

            // Secured assets: locked by default, Touch ID / password to reveal.
            if !securedIds.isEmpty || revealSecured {
                Button {
                    if revealSecured {
                        revealSecured = false
                    } else {
                        Task { await unlockSecured() }
                    }
                } label: {
                    Image(systemName: revealSecured ? "lock.open" : "lock")
                        .foregroundStyle(revealSecured ? .orange : .secondary)
                }
                .buttonStyle(.borderless)
                .help(revealSecured
                      ? "Hide secured images"
                      : "Show secured images (\(securedIds.count)) — requires authentication")
            }

            // NSFW filter: Show / Blur / Hide, with a password-gated unlock.
            Menu {
                Picker("NSFW", selection: $nsfwMode) {
                    ForEach(NSFWFilterMode.allCases) { Label($0.rawValue, systemImage: $0.symbol).tag($0) }
                }
                Divider()
                if nsfwUnlocked {
                    Button("Lock NSFW") { nsfwUnlocked = false }
                } else {
                    Button("Unlock NSFW…") { requestNSFWUnlock() }
                }
            } label: {
                Image(systemName: nsfwUnlocked ? "eye" : nsfwMode.symbol)
                    .foregroundStyle(nsfwUnlocked ? .orange : .secondary)
            }
            .help("NSFW content: \(nsfwMode.rawValue)\(nsfwUnlocked ? " (unlocked)" : "")")

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

            // Gallery Health (orphan thumbnails, missing assets, regenerate,
            // incomplete archives)
            Button { showMaintenance = true } label: {
                Image(systemName: "stethoscope")
            }
            .help("Gallery health & maintenance")

            // Asset count
            Text("\(filteredAssets.count) images")
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    // MARK: - Smart Tabs

    /// A saved tab is "active" when the current filter state exactly matches
    /// it — computed rather than tracked, so it self-clears the moment any
    /// filter is changed by hand.
    private func matchesCurrentFilters(_ tab: SmartTab) -> Bool {
        guard tab.searchText == searchText else { return false }
        guard tab.filterFavorites == filterFavorites else { return false }
        guard tab.filterContentMode == filterContentMode else { return false }
        guard tab.filterCharacter == filterCharacter else { return false }
        guard tab.filterLabel == filterLabel?.rawValue else { return false }
        guard tab.sortOrder == sortOrder.rawValue else { return false }
        return true
    }

    private var activeSmartTabId: String? {
        smartTabs.first(where: matchesCurrentFilters)?.id
    }

    private var smartTabsRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(smartTabs) { tab in
                    smartTabButton(tab)
                }
                Button {
                    newTabName = ""
                    showSaveTabPrompt = true
                } label: {
                    Label("Save View", systemImage: "plus.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
    }

    @ViewBuilder
    private func smartTabButton(_ tab: SmartTab) -> some View {
        let isActive = activeSmartTabId == tab.id
        Button(tab.name) { applySmartTab(tab) }
            .buttonStyle(.bordered)
            .tint(isActive ? Color.accentColor : Color.secondary)
            .controlSize(.small)
            .contextMenu {
                Button("Update with Current Filters") { updateSmartTab(tab) }
                Button("Rename…") {
                    renameSmartTabText = tab.name
                    renamingSmartTab = tab
                }
                Divider()
                Button("Delete", role: .destructive) { deleteSmartTab(tab) }
            }
    }

    private func smartTab(from tab: SmartTab? = nil, id: String, name: String) -> SmartTab {
        SmartTab(
            id: id,
            name: name,
            searchText: searchText,
            filterFavorites: filterFavorites,
            filterContentMode: filterContentMode,
            filterCharacter: filterCharacter,
            filterLabel: filterLabel?.rawValue,
            sortOrder: sortOrder.rawValue
        )
    }

    private func applySmartTab(_ tab: SmartTab) {
        searchText = tab.searchText
        filterFavorites = tab.filterFavorites
        filterContentMode = tab.filterContentMode
        filterCharacter = tab.filterCharacter
        filterLabel = tab.filterLabel.flatMap { FinderColor(rawValue: $0) }
        sortOrder = GallerySortOrder(rawValue: tab.sortOrder) ?? .date
    }

    private func saveCurrentAsSmartTab(named name: String) {
        smartTabs.append(smartTab(id: UUID().uuidString, name: name))
        SmartTabStore.save(smartTabs)
    }

    private func updateSmartTab(_ tab: SmartTab) {
        guard let index = smartTabs.firstIndex(where: { $0.id == tab.id }) else { return }
        smartTabs[index] = smartTab(id: tab.id, name: tab.name)
        SmartTabStore.save(smartTabs)
    }

    private func renameSmartTab(_ tab: SmartTab, to name: String) {
        guard let index = smartTabs.firstIndex(where: { $0.id == tab.id }) else { return }
        smartTabs[index].name = name
        SmartTabStore.save(smartTabs)
    }

    private func deleteSmartTab(_ tab: SmartTab) {
        smartTabs.removeAll { $0.id == tab.id }
        SmartTabStore.save(smartTabs)
    }

    // MARK: - Gallery Grid

    private static let gridSpacing: CGFloat = 12

    private var galleryGrid: some View {
        GeometryReader { geo in
            // Masonry: pack the filtered assets into aspect-ratio-preserving
            // columns so portraits, landscapes, and squares all show in full.
            let available = geo.size.width - Self.gridSpacing * 2   // outer padding
            let columnCount = max(1, Int(available / (cellTargetWidth + Self.gridSpacing)))
            let cellWidth = (available - Self.gridSpacing * CGFloat(columnCount - 1)) / CGFloat(columnCount)
            let cols = masonryColumns(filteredAssets, columnCount: columnCount, cellWidth: cellWidth)

            ScrollView {
                HStack(alignment: .top, spacing: Self.gridSpacing) {
                    ForEach(Array(cols.enumerated()), id: \.offset) { _, column in
                        LazyVStack(spacing: Self.gridSpacing) {
                            ForEach(column) { asset in
                                decoratedCell(for: asset, width: cellWidth)
                            }
                        }
                    }
                }
                .padding(Self.gridSpacing)
            }
        }
    }

    /// Aspect ratio (w/h) from the stored dimensions, clamped so a freak
    /// panorama or sliver can't blow out a column. Falls back to square.
    private func aspectRatio(_ asset: DAMAsset) -> CGFloat {
        guard let w = asset.width, let h = asset.height, w > 0, h > 0 else { return 1 }
        return min(max(CGFloat(w) / CGFloat(h), 0.4), 2.5)
    }

    /// Distribute assets into `columnCount` columns, always appending to the
    /// currently-shortest column (classic masonry packing) using each image's
    /// aspect ratio to estimate cell height.
    private func masonryColumns(_ assets: [DAMAsset], columnCount: Int, cellWidth: CGFloat) -> [[DAMAsset]] {
        var columns = [[DAMAsset]](repeating: [], count: columnCount)
        var heights = [CGFloat](repeating: 0, count: columnCount)
        let captionAllowance: CGFloat = 40   // prompt + rating row + padding
        for asset in assets {
            let cellHeight = cellWidth / aspectRatio(asset) + captionAllowance
            let shortest = heights.enumerated().min(by: { $0.element < $1.element })?.offset ?? 0
            columns[shortest].append(asset)
            heights[shortest] += cellHeight + Self.gridSpacing
        }
        return columns
    }

    /// Blur NSFW cells when Blur mode is on and NSFW isn't unlocked.
    private func shouldBlurNSFW(_ asset: DAMAsset) -> Bool {
        nsfwMode == .blur && !nsfwUnlocked && asset.isNSFW
    }

    @ViewBuilder
    private func decoratedCell(for asset: DAMAsset, width: CGFloat) -> some View {
        let isSelected = selectedIds.contains(asset.id)
        let blurNSFW = shouldBlurNSFW(asset)
        GalleryCellView(
            asset: asset,
            thumbnailPath: ingestor.thumbnailPath(for: asset.id),
            remoteURL: mediaLocation(for: asset).remoteURL,
            cellWidth: width,
            aspectRatio: aspectRatio(asset),
            isComparisonSelected: isSelectMode ? isSelected : nil
        )
                    .blur(radius: blurNSFW ? 22 : 0)
                    .overlay {
                        if blurNSFW {
                            VStack(spacing: 4) {
                                Image(systemName: "eye.trianglebadge.exclamationmark").font(.title3)
                                Text("NSFW").font(.caption2.weight(.semibold))
                            }
                            .foregroundStyle(.white)
                            .padding(8)
                            .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                            .onTapGesture { requestNSFWUnlock() }
                        }
                    }
                    .overlay(alignment: .bottomLeading) {
                        if asset.kind == "video" {
                            Image(systemName: "play.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.white)
                                .shadow(radius: 2)
                                .padding(6)
                        }
                    }
                    .overlay(alignment: .topLeading) {
                        if securedIds.contains(asset.id) {
                            Image(systemName: "lock.fill")
                                .font(.caption)
                                .foregroundStyle(.white)
                                .padding(5)
                                .background(.orange.opacity(0.85), in: Circle())
                                .padding(6)
                        }
                    }
                    // Photo Mechanic-style color class: a colored frame + dot,
                    // mirroring the file's Finder tag.
                    .overlay {
                        if let label = colorLabels[asset.id] {
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(label.displayColor, lineWidth: 2.5)
                        }
                    }
                    .overlay(alignment: .topTrailing) {
                        if let label = colorLabels[asset.id] {
                            Circle()
                                .fill(label.displayColor)
                                .frame(width: 10, height: 10)
                                .overlay(Circle().stroke(.white.opacity(0.8), lineWidth: 1))
                                .padding(8)
                        }
                    }
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
                        Button("Copy") {
                            copyAssets(isSelectMode && selectedIds.contains(asset.id)
                                ? selectedAssetsList : [asset])
                        }
                        if onUseAsReference != nil {
                            Button("Use as Reference (img2img)") { onUseAsReference?(asset) }
                        }
                        if onSendToGenerate != nil {
                            Button("Send to Generate") { onSendToGenerate?(asset) }
                        }
                        if onAnimate != nil {
                            Button("Send to Motion (I2V)") { onAnimate?(asset) }
                        }
                        if onInpaint != nil {
                            Button("Edit / Inpaint") { onInpaint?(asset) }
                        }
                        if mediaTools.hasMagick {
                            Menu("Export As") {
                                ForEach(MediaToolsService.ImageFormat.allCases, id: \.self) { fmt in
                                    Button(fmt.display) { Task { await exportAsset(asset, to: fmt) } }
                                }
                            }
                            Menu("Transform & Export") {
                                ForEach(MediaToolsService.Transform.allCases.filter { $0 != .none }, id: \.self) { t in
                                    Button(t.display) { Task { await exportAsset(asset, to: .png, transform: t) } }
                                }
                            }
                        }
                        if sidecar.isAvailable {
                            Button("Write Finder Metadata") { Task { await embedFinderMetadata(asset) } }
                        }
                        if engine != nil {
                            Menu("Upscale") {
                                Button("to 2048px (long side)") { Task { await upscaleAsset(asset, to: 2048) } }
                                Button("to 4096px (long side)") { Task { await upscaleAsset(asset, to: 4096) } }
                            }
                        }
                        // Winner actions (spec 2026-08-10): the scheduler
                        // renders cheap 480p/4s — clips worth keeping get
                        // improved here instead.
                        if engine != nil, asset.kind == "video" {
                            Button("Re-render at 720p") { Task { await rerenderVideoAsset(asset) } }
                            Button("Extend +4s") { Task { await extendVideoAsset(asset) } }
                        }
                        if engine != nil {
                            Button("Auto-caption & Tag") {
                                let targets = isSelectMode && selectedIds.contains(asset.id)
                                    ? selectedAssetsList : [asset]
                                Task { await autoCaptionTag(targets) }
                            }
                        }
                        Button(asset.favorite ? "Unfavorite" : "Favorite") {
                            Task { await toggleFavorite(asset) }
                        }
                        Menu("Move to Folder") {
                            moveToFolderMenuItems(for: isSelectMode && selectedIds.contains(asset.id)
                                ? Array(selectedIds) : [asset.id])
                        }
                        if let canvasStore, !canvasStore.projects.isEmpty {
                            Menu("Add to Canvas") {
                                ForEach(canvasStore.projects) { project in
                                    Button(project.name) {
                                        addToCanvas(
                                            isSelectMode && selectedIds.contains(asset.id) ? selectedAssetsList : [asset],
                                            canvasId: project.id, store: canvasStore)
                                    }
                                }
                            }
                        }
                        Menu("Label") {
                            labelMenuItems(for: isSelectMode && selectedIds.contains(asset.id)
                                ? selectedAssetsList : [asset])
                        }
                        if securedIds.contains(asset.id) {
                            Button("Unsecure") {
                                Task { await unsecureAssets([asset]) }
                            }
                        } else {
                            Button("Secure…") {
                                let targets = isSelectMode && selectedIds.contains(asset.id)
                                    ? selectedAssetsList : [asset]
                                Task { await secureAssets(targets) }
                            }
                        }
                        if !isSelectMode {
                            Divider()
                            Button("Select") {
                                selectedIds.insert(asset.id)
                                isSelectMode = true
                            }
                        }
                        if remoteURLs[asset.id] != nil {
                            Divider()
                            Button("Save to this Mac") {
                                Task { await pullRemote(asset) }
                            }
                        }
                        Divider()
                        Button("Archive…") {
                            requestArchive(isSelectMode && selectedIds.contains(asset.id)
                                ? selectedAssetsList : [asset], folder: nil)
                        }
                        Divider()
                        Button("Delete…", role: .destructive) {
                            requestDelete([asset])
                        }
                    }
                    .draggable(DraggableAsset(path: asset.absolutePath))
    }

    /// A path in `directory` that no file occupies: `name.png`, then
    /// `name-1.png`, `name-2.png`, …
    ///
    /// A server file and a local file can share a basename, and the download
    /// below is an unattended write — without this it would silently overwrite
    /// the local original with the remote one. Uniquing rather than prompting:
    /// the answer is always "keep both".
    static func uniqueDestination(inDirectory directory: String, filename: String) -> String {
        let fm = FileManager.default
        let base = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        func candidate(_ suffix: String) -> String {
            let name = ext.isEmpty ? base + suffix : "\(base)\(suffix).\(ext)"
            return (directory as NSString).appendingPathComponent(name)
        }
        var path = candidate("")
        var n = 1
        while fm.fileExists(atPath: path) {
            path = candidate("-\(n)")
            n += 1
        }
        return path
    }

    /// Where one row's bytes are, as THIS surface decided it.
    ///
    /// The single accessor every surface goes through — grid cell, detail pane,
    /// lightbox — so they cannot disagree about whether a row is remote. The
    /// local half prefers the path `CatalogBrowser` resolved for the page (an
    /// asset's `mac` location can spell a different path than its primary
    /// `absolutePath`); the remote half is the stream URL the browser built for
    /// exactly the rows it found no local file for.
    func mediaLocation(for asset: DAMAsset) -> AssetMediaLocation {
        AssetMediaLocation(
            localPath: browser?.localPath(forID: asset.id) ?? asset.absolutePath,
            remoteURL: remoteURLs[asset.id])
    }

    /// Download a row whose bytes are on a server into the local output folder
    /// and ingest it — the one thing the retired Remote Gallery could do that
    /// browsing the catalog cannot.
    private func pullRemote(_ asset: DAMAsset) async {
        guard let url = remoteURLs[asset.id] else { return }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                errorMessage = "Server returned \(http.statusCode)"
                return
            }
            let dir = DesktopSettings.load().outputDirectory
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let dest = Self.uniqueDestination(inDirectory: dir, filename: asset.filename)
            try data.write(to: URL(fileURLWithPath: dest))
            _ = try? await ingestor.ingestFile(at: dest)
            errorMessage = nil
            await loadAssets()
        } catch {
            errorMessage = "Save failed: \(error.localizedDescription)"
        }
    }

    private func errorBanner(_ message: String,
                             dismiss: (() -> Void)? = nil) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.callout)
                .lineLimit(2)
            Spacer()
            Button(action: { if let dismiss { dismiss() } else { errorMessage = nil } }) {
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

    /// The engine this Mac's gallery streams server-side bytes from.
    private var engineBaseURL: String {
        guard let engine else { return "http://127.0.0.1:7870" }
        return "http://\(engine.serverHost):\(engine.serverPort)"
    }

    /// Collection + lane rail over the whole catalog. Collections come first
    /// because they are the bodies of work; lanes below them are the finer cut.
    ///
    /// Only rendered when the content gate is revealed — see `showsCatalogRail`.
    @ViewBuilder
    private var catalogRail: some View {
        if let browser {
            VStack(alignment: .leading, spacing: 0) {
                List {
                    Section {
                        railRow(title: "Everything",
                                icon: "square.grid.2x2",
                                count: browser.facets.kind.values.reduce(0, +),
                                isSelected: selectedCollectionID == nil && selectedLane == nil) {
                            selectedCollectionID = nil
                            selectedLane = nil
                            Task { await applyCatalogFilter() }
                        }
                    }
                    Section("Collections") {
                        ForEach(browser.rootCollections(), id: \.id) { root in
                            DisclosureGroup {
                                ForEach(browser.children(of: root), id: \.id) { child in
                                    railRow(title: child.name, icon: "folder",
                                            count: browser.count(of: child),
                                            isSelected: selectedCollectionID == child.id) {
                                        selectedCollectionID = child.id
                                        selectedLane = nil
                                        Task { await applyCatalogFilter() }
                                    }
                                }
                            } label: {
                                railRow(title: root.name, icon: "folder.fill",
                                        count: browser.count(of: root),
                                        isSelected: selectedCollectionID == root.id) {
                                    selectedCollectionID = root.id
                                    selectedLane = nil
                                    Task { await applyCatalogFilter() }
                                }
                            }
                        }
                    }
                    Section("Lane") {
                        ForEach(browser.facets.lane.sorted(by: { $0.key < $1.key }), id: \.key) { lane, count in
                            railRow(title: lane, icon: "line.3.horizontal.decrease",
                                    count: count, isSelected: selectedLane == lane) {
                                selectedLane = lane
                                selectedCollectionID = nil
                                Task { await applyCatalogFilter() }
                            }
                        }
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)

                if browser.isLoading {
                    ProgressView().controlSize(.small).padding(8)
                }
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
    }

    private func railRow(title: String, icon: String, count: Int,
                         isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                Text(title).lineLimit(1)
                Spacer()
                Text("\(count)").font(.caption).foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.plain)
        .listRowBackground(isSelected ? Color.accentColor.opacity(0.18) : nil)
    }

    /// The catalog query the rail currently describes. Unscoped and unclamped:
    /// this is the owner's own surface, and it is the content gate — not the
    /// mode ceiling — that governs what reaches the screen here.
    private func catalogQuery() -> CatalogQuery {
        CatalogQuery(text: searchText.isEmpty ? nil : searchText,
                     collectionID: selectedCollectionID,
                     lane: selectedLane,
                     limit: loadLimit)
    }

    private func applyCatalogFilter() async {
        guard browser != nil else { return }
        await loadAssets()
    }

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
                                    Button("Archive Folder…") {
                                        let ids = Set(folderAssignments.filter { $0.value == folder.id }.keys)
                                        Task { await requestArchiveFolder(ids: ids, folder: folder) }
                                    }
                                    Button("Delete Folder", role: .destructive) {
                                        Task { await deleteFolder(folder) }
                                    }
                                }
                        }
                    }
                }
                // Persona sections: agent-generated renders (Kira, Bree, …) live
                // in their own sections, kept out of the main gallery.
                if !personaSources.isEmpty {
                    Section("Personas") {
                        ForEach(personaSources, id: \.name) { p in
                            Button {
                                personaFilter = (personaFilter == p.name) ? nil : p.name
                            } label: {
                                HStack {
                                    Image(systemName: "person.crop.circle")
                                    Text(p.name.capitalized)
                                    Spacer()
                                    Text("\(p.count)").font(.caption).foregroundStyle(.secondary)
                                }
                            }
                            .buttonStyle(.plain)
                            .listRowBackground(personaFilter == p.name ? Color.accentColor.opacity(0.18) : nil)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)
            .onChange(of: folderFilter) { _, _ in personaFilter = nil }

            Divider()

            if let importProgress {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Importing \(importProgress.done) / \(importProgress.total)…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ProgressView(value: Double(importProgress.done),
                                 total: Double(max(importProgress.total, 1)))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
            } else if let importSummary {
                Text(importSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .padding(.horizontal, 10)
                    .padding(.top, 6)
            }

            if let archiveProgress {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Archiving \(archiveProgress.done) / \(archiveProgress.total)…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ProgressView(value: Double(archiveProgress.done),
                                 total: Double(max(archiveProgress.total, 1)))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
            } else if let archiveSummary {
                Text(archiveSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .padding(.horizontal, 10)
                    .padding(.top, 6)
            }

            HStack(spacing: 4) {
                Button {
                    pendingFolderAssets = []
                    showNewFolderPrompt = true
                } label: {
                    Label("New", systemImage: "folder.badge.plus")
                        .font(.callout)
                }
                .buttonStyle(.borderless)

                Spacer()

                Button {
                    chooseFolderToImport()
                } label: {
                    Label("Add Folder…", systemImage: "square.and.arrow.down.on.square")
                        .font(.callout)
                }
                .buttonStyle(.borderless)
                .disabled(importProgress != nil)
                .help("Import an existing folder of images into the gallery")
            }
            .padding(10)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: - Folder import

    /// Prompt for a folder and import it (Photo Mechanic style).
    private func chooseFolderToImport() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Add to Gallery"
        panel.message = "Choose a folder of images to add to the gallery."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await importFolder(url.path) }
    }

    private func importFolder(_ path: String) async {
        importSummary = nil
        importProgress = (0, 0)
        do {
            let summary = try await ingestor.importFolder(at: path) { done, total in
                importProgress = (done, total)
            }
            importProgress = nil
            var parts = ["Imported \(summary.imported)"]
            if summary.skipped > 0 { parts.append("\(summary.skipped) already added") }
            if summary.failed > 0 { parts.append("\(summary.failed) failed") }
            importSummary = parts.joined(separator: " · ")
            await loadAssets()
        } catch {
            importProgress = nil
            errorMessage = "Import failed: \(error.localizedDescription)"
        }
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

        // Persona sectioning: the main gallery excludes persona/agent renders
        // (Kira, Bree, …); selecting a persona shows only that source's images.
        if let personaFilter {
            results = results.filter { ($0.source ?? "").lowercased() == personaFilter }
        } else {
            results = results.filter { Self.isMainSource($0.source) }
        }

        // Secured assets stay hidden until unlocked.
        if !revealSecured {
            results = results.filter { !securedIds.contains($0.id) }
        }

        // NSFW: hide entirely when mode is .hide and not unlocked.
        if nsfwMode == .hide && !nsfwUnlocked {
            results = results.filter { !$0.isNSFW }
        }

        // Finder color-label filter.
        if let filterLabel {
            results = results.filter { colorLabels[$0.id] == filterLabel }
        }

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

        // Apply search. When the catalog is driving, the text has ALREADY been
        // matched against the whole catalog's FTS index (prompt, injected
        // prompt, caption, filename) — re-filtering here by substring would
        // throw away real hits on the fields this loop does not look at.
        if !searchText.isEmpty, browser == nil {
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
            // Self-heal: drop rows whose file was deleted out from under the
            // DAM before presenting (only worth it on a full, unfiltered load).
            // This is an unattended destructive sweep, so its refusal is SHOWN,
            // not swallowed: a circuit breaker trip means a volume is probably
            // unmounted, and silently skipping it would leave the gallery
            // looking half-empty with no explanation.
            if searchText.isEmpty {
                do { _ = try await ingestor.pruneOrphans() }
                catch let error as DAMStoreError {
                    if case .pruneRefused = error { pruneWarning = error.localizedDescription }
                } catch { /* a prune failure must never block browsing */ }
            }
            folders = try await store.listFolders()
            folderCounts = try await store.folderCounts()
            folderAssignments = try await store.folderAssignments()
            securedIds = try await store.securedAssetIds()

            if let browser {
                try await loadFromCatalog(browser)
            } else {
                try await loadFromLocalStore()
            }
            refreshColorLabels()
            extractFilterValues()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    /// THE converged read: one query over the whole catalog, both realms, every
    /// host. Rows keep their DAM record where the shared `assets` table has one
    /// (so ratings, favourites, folders and dates are the real ones) and only
    /// fall back to the catalog's own view for anything the DAM columns are
    /// empty for. Rows whose bytes are not on this Mac get an engine URL.
    private func loadFromCatalog(_ browser: CatalogBrowser) async throws {
        // The vault carve-out. A secured asset's row survives in the shared
        // table and very often still has a streamable twin on a server, so the
        // reader has to be told about it or converging on the catalog would
        // quietly undo every vault move. `filteredAssets` filters by the same
        // set again — this is deliberately belt AND braces.
        // NO TEST GUARDS THIS LINE. The withholding itself is tested on
        // CatalogBrowser, but nothing checks that this view actually feeds it the
        // secured set — delete this line and every test still passes while every
        // vaulted asset comes back. Change it by hand, carefully.
        browser.hiddenAssetIDs = revealSecured ? [] : securedIds
        await browser.apply(filter: catalogQuery())
        if let message = browser.error { errorMessage = message } else { errorMessage = nil }

        var damByID: [String: DAMAsset] = [:]
        for row in try await store.fetchAssets(limit: 20_000) { damByID[row.id] = row }

        var rows: [DAMAsset] = []
        var urls: [String: URL] = [:]
        rows.reserveCapacity(browser.items.count)
        for item in browser.items {
            rows.append(damByID[item.id] ?? browser.damAsset(for: item))
            if browser.localPath(forID: item.id) == nil,
               let url = browser.resolvedStreamURL(forID: item.id) {
                urls[item.id] = url
            }
        }
        assets = rows
        remoteURLs = urls
        // Only for rows whose bytes are actually here — regenerating a thumbnail
        // for a file that lives on another machine is a guaranteed miss.
        await ingestor.regenerateMissingThumbnails(for: rows.filter { urls[$0.id] == nil })
    }

    /// Fallback for when the catalog could not be opened: the Mac-only reader
    /// this view used before the two galleries converged.
    private func loadFromLocalStore() async throws {
        remoteURLs = [:]
        if !searchText.isEmpty {
            assets = try await store.searchPrompts(query: searchText,
                                                   limit: loadLimit == Self.fullScopeLimit ? loadLimit : 200)
        } else {
            assets = try await store.fetchAssets(limit: loadLimit)
            // Self-heal: backfill any thumbnail that's missing or was
            // left as a 0-byte file by a previously-interrupted write.
            await ingestor.regenerateMissingThumbnails(for: assets)
        }
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

    /// Consume a pending "Gallery Health" palette request by opening the
    /// maintenance sheet.
    private func consumeMaintenanceRequest() {
        guard maintenanceRequests > 0 else { return }
        maintenanceRequests = 0
        showMaintenance = true
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

    /// Export/transform one image via ImageMagick, then reveal it in Finder.
    private func exportAsset(_ asset: DAMAsset, to format: MediaToolsService.ImageFormat,
                             transform: MediaToolsService.Transform = .none) async {
        do {
            let out = try await mediaTools.export(source: asset.absolutePath, to: format, transform: transform)
            NSWorkspace.shared.selectFile(out, inFileViewerRootedAtPath: "")
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    /// Embed generation metadata into the image as standard EXIF/XMP/IPTC fields
    /// so Finder Get Info → More Info (Description + Keywords) and Spotlight read it.
    private func embedFinderMetadata(_ asset: DAMAsset) async {
        let tags = FinderTags.textTags(atPath: asset.absolutePath)
        var params: [String: Any] = [:]
        if let p = asset.prompt { params["prompt"] = p }
        if let n = asset.negativePrompt { params["negative_prompt"] = n }
        if let s = asset.seed { params["seed"] = s }
        if let st = asset.steps { params["steps"] = st }
        if let g = asset.guidance { params["guidance"] = g }
        if let m = asset.modelFamily { params["model"] = m }
        let json = (try? JSONSerialization.data(withJSONObject: params)).flatMap { String(data: $0, encoding: .utf8) }
        let meta = SidecarService.Metadata(
            description: asset.prompt ?? asset.filename,
            keywords: SidecarService.keywords(tags: tags, character: asset.characterName, contentMode: asset.contentMode),
            parametersJSON: json)
        do {
            try await sidecar.embed(meta, into: asset.absolutePath)
        } catch {
            errorMessage = "Metadata embed failed: \(error.localizedDescription)"
        }
    }

    /// Export the current multi-selection as a video (sorted by date), ingest it.
    private func exportSelectionToVideo(fps: Int = 8) async {
        let paths = filteredAssets
            .filter { selectedIds.contains($0.id) && $0.kind != "video" }
            .sorted { $0.createdAt < $1.createdAt }
            .map { $0.absolutePath }
        guard paths.count >= 2 else { errorMessage = "Select 2+ images to export a video."; return }
        let dir = DesktopSettings.load().outputDirectory
        let out = (dir as NSString).appendingPathComponent("sequence-\(Int(Date().timeIntervalSince1970)).mp4")
        do {
            _ = try await mediaTools.exportVideo(images: paths, fps: fps, output: out)
            try? await ingestor.ingestFile(at: out)
            NSWorkspace.shared.selectFile(out, inFileViewerRootedAtPath: "")
        } catch {
            errorMessage = "Video export failed: \(error.localizedDescription)"
        }
    }

    /// Add assets to a canvas, cascading their positions so they don't stack.
    private func addToCanvas(_ assets: [DAMAsset], canvasId: String, store canvasStore: CanvasStore) {
        for (offset, asset) in assets.enumerated() {
            var w = 320.0, h = 320.0
            if let w0 = asset.width, let h0 = asset.height, w0 > 0, h0 > 0 {
                let aspect = Double(w0) / Double(h0)
                if aspect >= 1 { h = 320 / aspect } else { w = 320 * aspect }
            }
            canvasStore.addItem(
                CanvasItem(imagePath: asset.absolutePath,
                           x: 60 + Double(offset) * 28, y: 60 + Double(offset) * 28,
                           width: w, height: h),
                toCanvas: canvasId)
        }
    }

    /// Caption + tag assets with the local vision model. Tags → Finder tags,
    /// caption → a Finder-aligned xattr (both visible in Finder + the gallery).
    private func autoCaptionTag(_ assets: [DAMAsset]) async {
        guard let engine else { return }
        let vision = VisionService(engine: engine)
        var done = 0, failed = 0
        for asset in assets {
            errorMessage = "Captioning \(asset.filename)… (\(done + 1)/\(assets.count))"
            do {
                let desc = try await vision.describe(imagePath: asset.absolutePath)
                if !desc.tags.isEmpty { try? FinderTags.addTextTags(desc.tags, atPath: asset.absolutePath) }
                if !desc.caption.isEmpty { FinderTags.setCaption(desc.caption, atPath: asset.absolutePath) }
                done += 1
            } catch {
                failed += 1
                if assets.count == 1 { errorMessage = "Caption failed: \(error.localizedDescription)"; return }
            }
        }
        errorMessage = failed == 0 ? nil : "Captioned \(done), \(failed) failed."
        await loadAssets()
    }

    /// Upscale an asset via SeedVR2 and ingest the result into the gallery.
    /// Winner action: replay this clip's exact render at 720p (same seed +
    /// prompt — the same clip, larger). Queued async on the engine; the
    /// result lands in the gallery like any render.
    private func rerenderVideoAsset(_ asset: DAMAsset) async {
        guard let engine else { return }
        do {
            let jobId = try await engine.rerenderVideo(path: asset.absolutePath)
            errorMessage = "720p re-render queued (job \(jobId.prefix(8))…) — takes ~4x the original render; watch the Queue tab."
        } catch {
            errorMessage = "Re-render failed: \(error.localizedDescription)"
        }
    }

    /// Winner action: chain a fresh 4s continuation from this clip's last frame.
    private func extendVideoAsset(_ asset: DAMAsset) async {
        guard let engine else { return }
        do {
            let jobId = try await engine.extendVideo(path: asset.absolutePath)
            errorMessage = "Extend queued (job \(jobId.prefix(8))…) — a new 4s clip chained from the last frame; watch the Queue tab."
        } catch {
            errorMessage = "Extend failed: \(error.localizedDescription)"
        }
    }

    private func upscaleAsset(_ asset: DAMAsset, to target: Int) async {
        guard let engine else { return }
        errorMessage = "Upscaling \(asset.filename) to \(target)px… (this runs on the server and may take a while)"
        do {
            let outputPath = try await engine.upscale(imagePath: asset.absolutePath, targetResolution: target)
            try? await ingestor.ingestFile(at: outputPath)
            errorMessage = nil
            await loadAssets()
        } catch {
            errorMessage = "Upscale failed: \(error.localizedDescription)"
        }
    }

    /// Copy assets to the clipboard as both file references (paste into Finder)
    /// and, for a single image, its bitmap (paste into image editors / docs).
    private func copyAssets(_ assets: [DAMAsset]) {
        let urls = assets
            .map { URL(fileURLWithPath: $0.absolutePath) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !urls.isEmpty else { return }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        var items: [NSPasteboardWriting] = urls.map { $0 as NSURL }
        // A single image also goes on as an image so ⌘V pastes the picture.
        if urls.count == 1, let image = NSImage(contentsOf: urls[0]) {
            items.append(image)
        }
        pasteboard.writeObjects(items)
    }

    /// Assets for the current selection, in display order.
    private var selectedAssetsList: [DAMAsset] {
        filteredAssets.filter { selectedIds.contains($0.id) }
    }

    /// Select every asset in the current view. The grid normally loads one
    /// page (`loadLimit`); if that page is full there may be more rows in
    /// scope, so raise the limit to the full scope and reload first — a
    /// "select all" that silently stops at the page boundary selects an
    /// arbitrary 500 of a 3,000-image gallery.
    private func selectAll() async {
        if loadLimit < Self.fullScopeLimit && assets.count >= loadLimit {
            loadLimit = Self.fullScopeLimit
            await loadAssets()
        }
        selectedIds = Set(filteredAssets.map(\.id))
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

    // MARK: - Finder color labels

    /// Read every visible asset's Finder tag (fast getxattr per file).
    private func refreshColorLabels() {
        var labels: [String: FinderColor] = [:]
        for asset in assets {
            if let color = FinderTags.colorLabel(atPath: asset.absolutePath) {
                labels[asset.id] = color
            }
        }
        colorLabels = labels
    }

    /// Photo Mechanic-style targets: the lightbox image when open, else the
    /// multi-selection, else nothing.
    private var labelTargets: [DAMAsset] {
        if let idx = lightboxIndex, filteredAssets.indices.contains(idx) {
            return [filteredAssets[idx]]
        }
        if isSelectMode { return selectedAssetsList }
        return []
    }

    /// Write the label onto the files (real Finder tags) and refresh.
    private func applyColorLabel(_ color: FinderColor?, to targets: [DAMAsset]) {
        guard !targets.isEmpty else { return }
        var failures: [String] = []
        for asset in targets {
            do {
                try FinderTags.setColorLabel(color, atPath: asset.absolutePath)
                if let color {
                    colorLabels[asset.id] = color
                } else {
                    colorLabels.removeValue(forKey: asset.id)
                }
            } catch {
                failures.append(asset.filename)
            }
        }
        if !failures.isEmpty {
            errorMessage = "Failed to tag: \(failures.joined(separator: ", "))"
        }
    }

    /// Handle Photo Mechanic-style digit shortcuts: 1–7 set colors, 0 clears.
    private func handleLabelKey(_ character: Character) -> Bool {
        guard let digit = character.wholeNumberValue, (0...7).contains(digit) else { return false }
        let targets = labelTargets
        guard !targets.isEmpty else { return false }
        if digit == 0 {
            applyColorLabel(nil, to: targets)
        } else {
            applyColorLabel(FinderColor.keyboardOrder[digit - 1], to: targets)
        }
        return true
    }

    /// The label picker shared by the context menu and toolbar.
    @ViewBuilder
    private func labelMenuItems(for targets: [DAMAsset]) -> some View {
        ForEach(Array(FinderColor.keyboardOrder.enumerated()), id: \.element) { index, color in
            Button("\(color.rawValue)  (\(index + 1))") {
                applyColorLabel(color, to: targets)
            }
        }
        Divider()
        Button("Clear Label  (0)") {
            applyColorLabel(nil, to: targets)
        }
    }

    // MARK: - Asset security

    /// Authenticate with Touch ID (or the login password) before revealing
    /// secured assets.
    /// Unlock NSFW content for the session. No password set → unlock directly;
    /// otherwise prompt for the gallery password.
    private func requestNSFWUnlock() {
        if !NSFWGate.isConfigured {
            nsfwUnlocked = true
        } else {
            nsfwPasswordInput = ""
            nsfwPasswordError = false
            showNSFWPasswordSheet = true
        }
    }

    private func submitNSFWPassword() {
        if NSFWGate.verify(nsfwPasswordInput) {
            nsfwUnlocked = true
            showNSFWPasswordSheet = false
        } else {
            nsfwPasswordError = true
        }
    }

    private func unlockSecured() async {
        let context = LAContext()
        context.localizedReason = "reveal secured images"
        var error: NSError?
        guard context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error) else {
            errorMessage = "Authentication unavailable: \(error?.localizedDescription ?? "unknown")"
            return
        }
        do {
            let ok = try await context.evaluatePolicy(
                .deviceOwnerAuthentication,
                localizedReason: "reveal secured images")
            if ok { revealSecured = true }
        } catch {
            // Cancelled or failed — stay locked, no error banner needed.
        }
    }

    /// Move assets into the secure vault (file + sidecar out of the output
    /// dir, thumbnail destroyed, hidden from gallery/search/history).
    private func secureAssets(_ toSecure: [DAMAsset]) async {
        var failures: [String] = []
        for asset in toSecure {
            do {
                _ = try await ingestor.secureAsset(asset)
                selectedIds.remove(asset.id)
            } catch {
                failures.append(asset.filename)
            }
        }
        if !failures.isEmpty {
            errorMessage = "Failed to secure: \(failures.joined(separator: ", "))"
        }
        lightboxIndex = nil
        await loadAssets()
    }

    /// Restore assets from the vault to their original location.
    private func unsecureAssets(_ toRestore: [DAMAsset]) async {
        var failures: [String] = []
        for asset in toRestore {
            do {
                _ = try await ingestor.unsecureAsset(asset)
            } catch {
                failures.append(asset.filename)
            }
        }
        if !failures.isEmpty {
            errorMessage = "Failed to unsecure: \(failures.joined(separator: ", "))"
        }
        await loadAssets()
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
        var refused: [String] = []
        for asset in toDelete {
            do {
                try await ingestor.deleteAsset(asset)
                selectedIds.remove(asset.id)
            } catch AssetIngestor.DeletionRefusal.notOnThisMac {
                // Not a failure to report as one: the file is on the server and
                // was deliberately left alone rather than permanently destroyed.
                refused.append(asset.filename)
            } catch {
                failures.append(asset.filename)
            }
        }
        var notes: [String] = []
        if !refused.isEmpty {
            notes.append("Left on the server (macOS cannot trash a network file): "
                + refused.joined(separator: ", "))
        }
        if !failures.isEmpty {
            notes.append("Failed to delete: \(failures.joined(separator: ", "))")
        }
        if !notes.isEmpty {
            errorMessage = notes.joined(separator: " — ")
        }
        lightboxIndex = nil
        await loadAssets()
    }

    // MARK: - Archiving

    /// Stage assets (and, for a folder archive, the folder itself) and show
    /// the archive sheet.
    private func requestArchive(_ toArchive: [DAMAsset], folder: DAMFolder?) {
        guard !toArchive.isEmpty else { return }
        archiveTargets = toArchive
        archiveFolder = folder
        showArchiveSheet = true
    }

    /// "Archive Folder…" must include every asset filed in the folder, not
    /// just whichever page happens to be sitting in the view's own `assets`
    /// array (capped at `fetchAssets(limit: 500)`) — fetch the full asset
    /// set from the store and filter by id instead, so folders larger than
    /// the page size still archive completely.
    private func requestArchiveFolder(ids: Set<String>, folder: DAMFolder) async {
        guard !ids.isEmpty else { return }
        do {
            let total = try await store.assetCount()
            let allAssets = try await store.fetchAssets(limit: total, offset: 0)
            requestArchive(Self.folderMembers(ids: ids, from: allAssets), folder: folder)
        } catch {
            errorMessage = "Failed to load folder assets: \(error.localizedDescription)"
        }
    }

    /// Given a folder's asset id set (from `folderAssignments()`) and the
    /// full, unpaged asset list, returns exactly the assets that belong to
    /// the folder. A pure, directly unit-testable helper isolating the
    /// membership filter from the store fetch above it.
    static func folderMembers(ids: Set<String>, from allAssets: [DAMAsset]) -> [DAMAsset] {
        allAssets.filter { ids.contains($0.id) }
    }

    /// Runs the archive via `GalleryArchiver`, mirroring the import strip's
    /// progress reporting. On completion: clear selection, refresh the grid
    /// (the same reload `deleteAssets` uses), show a one-line summary.
    /// Failures land in the existing `errorMessage` banner.
    private func performArchive(name: String, destinationRoot: String, assets toArchive: [DAMAsset], folder: DAMFolder?) {
        guard let archiver else { return }
        archiveSummary = nil
        archiveProgress = (0, toArchive.count)
        let request = GalleryArchiver.ArchiveRequest(
            name: name, destinationRoot: destinationRoot, assets: toArchive, folder: folder
        )
        Task {
            do {
                let result = try await archiver.archive(request) { done, total in
                    archiveProgress = (done, total)
                }
                archiveProgress = nil
                var parts = ["Archived \(result.archived)"]
                if result.skippedSecured > 0 { parts.append("\(result.skippedSecured) secured skipped") }
                if !result.failed.isEmpty { parts.append("\(result.failed.count) failed") }
                archiveSummary = parts.joined(separator: " · ")
                selectedIds.subtract(toArchive.map(\.id))
                await loadAssets()
            } catch {
                archiveProgress = nil
                errorMessage = "Archive failed: \(error.localizedDescription)"
            }
        }
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
    /// Engine URL for a row whose bytes are NOT on this Mac (the catalog knows
    /// the asset, `asset_locations` puts it on another host). nil for local rows.
    var remoteURL: URL?
    /// Column width in the masonry layout; nil = legacy fixed 160px height.
    var cellWidth: CGFloat?
    /// Image aspect ratio (w/h) used to size the cell to the true shape.
    var aspectRatio: CGFloat = 1
    var isComparisonSelected: Bool?

    @State private var thumbnail: NSImage?

    /// Height of the thumbnail area: aspect-derived in masonry, else fixed.
    private var thumbHeight: CGFloat {
        guard let cellWidth else { return 160 }
        return cellWidth / max(aspectRatio, 0.01)
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            VStack(spacing: 4) {
                // Thumbnail — shown in full at its aspect ratio (no crop).
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
                .frame(height: thumbHeight)
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
        let local: NSImage? = await Task.detached {
            NSImage(contentsOfFile: path) ?? NSImage(contentsOfFile: fullPath)
        }.value
        if let local {
            thumbnail = local
            return
        }
        // Nothing on disk. If the catalog placed this row on another host, its
        // bytes come from the engine's /v1/gallery/file — the same route the
        // Remote Gallery used, now reached through the one converged reader.
        guard let remoteURL else { return }
        guard let (data, response) = try? await URLSession.shared.data(from: remoteURL),
              (response as? HTTPURLResponse)?.statusCode ?? 200 == 200 else { return }
        thumbnail = NSImage(data: data)
    }
}

// MARK: - Lightbox

/// Full-screen zoomable image viewer with keyboard/on-screen prev-next navigation.
private struct GalleryLightbox: View {
    let assets: [DAMAsset]
    let index: Int
    /// Current Finder color label for an asset (nil = unlabeled).
    var labelForAsset: (DAMAsset) -> FinderColor? = { _ in nil }
    /// Set/clear the label on an asset (writes the Finder tag).
    var onSetLabel: (DAMAsset, FinderColor?) -> Void = { _, _ in }
    var onCopy: (DAMAsset) -> Void = { _ in }
    var onReveal: (DAMAsset) -> Void = { _ in }
    /// Send this asset's full recipe (prompt, params, LoRAs, content mode) to Generate.
    var onSendToGenerate: ((DAMAsset) -> Void)?
    /// Where the gallery decided each row's bytes are — the SAME answer the grid
    /// cell uses, so a cell that streamed its thumbnail never opens empty here.
    var mediaLocationProvider: (DAMAsset) -> AssetMediaLocation = {
        AssetMediaLocation(localPath: $0.absolutePath, remoteURL: nil)
    }
    let onIndexChange: (Int) -> Void
    let onClose: () -> Void

    @State private var image: NSImage?
    @State private var player: AVPlayer?
    @State private var zoom: CGFloat = 1
    @State private var baseZoom: CGFloat = 1
    @State private var offset: CGSize = .zero
    /// Why there is nothing on screen (server unreachable, file gone). Beats a
    /// ProgressView that spins forever.
    @State private var loadError: String?
    @FocusState private var focused: Bool
    @Environment(AppContentGate.self) private var contentGate

    private var asset: DAMAsset? { assets.indices.contains(index) ? assets[index] : nil }

    /// Where the current asset's bytes come from — gate first, then disk, then
    /// the engine.
    private var source: AssetMediaSource {
        guard let asset else { return .missing }
        return AssetMediaSource.resolve(mediaLocationProvider(asset), gateRevealed: contentGate.revealed)
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.93).ignoresSafeArea()
                .onTapGesture { onClose() }

            Group {
                if asset?.kind == "video", let player {
                    SafeVideoPlayer(player: player)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(24)
                } else if let image {
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
                } else if let loadError {
                    VStack(spacing: 10) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 34)).foregroundStyle(.white.opacity(0.7))
                        Text(loadError)
                            .font(.callout).foregroundStyle(.white.opacity(0.85))
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 460)
                    }
                } else {
                    VStack(spacing: 10) {
                        ProgressView().controlSize(.large).tint(.white)
                        if source.isRemote {
                            Text(asset?.kind == "video"
                                 ? "Fetching video from server..." : "Loading from server...")
                                .font(.caption).foregroundStyle(.white.opacity(0.7))
                        }
                    }
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

                    // Photo Mechanic-style color classes: click a swatch or
                    // press 1-7 (0 clears). Mirrors the file's Finder tag.
                    // Colour classes write a Finder tag to the file itself, so
                    // they only exist for a row whose file is on this Mac.
                    if let asset, source.isLocal {
                        let current = labelForAsset(asset)
                        HStack(spacing: 7) {
                            ForEach(Array(FinderColor.keyboardOrder.enumerated()), id: \.element) { i, color in
                                Button {
                                    onSetLabel(asset, current == color ? nil : color)
                                } label: {
                                    Circle()
                                        .fill(color.displayColor)
                                        .frame(width: 16, height: 16)
                                        .overlay(
                                            Circle().stroke(
                                                current == color ? .white : .white.opacity(0.25),
                                                lineWidth: current == color ? 2.5 : 1)
                                        )
                                }
                                .buttonStyle(.plain)
                                .help("\(color.rawValue) — press \(i + 1)")
                            }
                            Button {
                                onSetLabel(asset, nil)
                            } label: {
                                Image(systemName: "slash.circle")
                                    .font(.system(size: 14))
                                    .foregroundStyle(.white.opacity(current == nil ? 0.9 : 0.4))
                            }
                            .buttonStyle(.plain)
                            .help("Clear label — press 0")
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(.black.opacity(0.5), in: Capsule())
                    }

                    Spacer()

                    if let asset {
                        if source.isRemote {
                            Image(systemName: "externaldrive.connected.to.line.below")
                                .font(.title3).foregroundStyle(.white.opacity(0.6))
                                .help("Streamed from the engine — no copy of this file on this Mac.")
                        }
                        // Copy and Reveal need a file on this disk; for a
                        // server-side row they would quietly do nothing.
                        Button { onCopy(asset) } label: {
                            Image(systemName: "doc.on.doc").font(.title3)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.white.opacity(source.isLocal ? 0.85 : 0.3))
                        .disabled(!source.isLocal)
                        .help(source.localOnlyReason ?? "Copy image (⌘C)")
                        Button { onReveal(asset) } label: {
                            Image(systemName: "magnifyingglass.circle").font(.title3)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.white.opacity(source.isLocal ? 0.85 : 0.3))
                        .disabled(!source.isLocal)
                        .help(source.localOnlyReason ?? "Reveal in Finder")
                        if let onSendToGenerate {
                            Button { onSendToGenerate(asset) } label: {
                                Image(systemName: "wand.and.stars.inverse").font(.title3)
                            }
                            .buttonStyle(.plain).foregroundStyle(.white.opacity(0.85))
                            .help("Send to Generate")
                        }
                    }

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
        .onKeyPress(keys: ["c"]) { press in
            // Same local-file rule as the button: ⌘C on a server-side row would
            // put nothing on the pasteboard.
            guard press.modifiers.contains(.command), let asset, source.isLocal else { return .ignored }
            onCopy(asset)
            return .handled
        }
        .onKeyPress(characters: CharacterSet(charactersIn: "01234567")) { press in
            // The lightbox owns keyboard focus, so it must handle the color
            // classes itself — the gallery-level handler never sees these.
            // Tagging writes to the file, so it needs one on this Mac.
            guard let asset, source.isLocal, let character = press.characters.first,
                  let digit = character.wholeNumberValue else { return .ignored }
            onSetLabel(asset, digit == 0 ? nil : FinderColor.keyboardOrder[digit - 1])
            return .handled
        }
        // Keyed on the gate too: closing it mid-session must stop a playing
        // video and drop the loaded bytes, not just blur them.
        .task(id: "\(index)|\(contentGate.revealed)") { await load() }
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
        // Tear down any previous player before constructing a new one —
        // building AVPlayer/VideoPlayer inline in the view body (one per
        // re-render) is a known source of a Swift generic-metadata crash
        // inside _AVKit_SwiftUI; owning exactly one player per asset here
        // avoids the churn that triggers it.
        player?.pause()
        player = nil
        image = nil
        loadError = nil
        guard let asset else { return }
        let isVideo = asset.kind == "video"

        switch source {
        case .gated:
            // Rated G: read nothing, from disk or network. The lightbox can
            // only be opened from a revealed grid, but the load path itself
            // refuses rather than relying on that.
            loadError = "Locked."

        case .local(let path):
            if isVideo {
                player = AVPlayer(url: URL(fileURLWithPath: path))
            } else {
                image = await Task.detached { NSImage(contentsOfFile: path) }.value
                if image == nil { loadError = "Couldn't open \((path as NSString).lastPathComponent)." }
            }

        case .remote(let url):
            if isVideo {
                // The engine serves whole bodies with no Range support and
                // AVPlayer needs ranges over HTTP, so fetch the file once and
                // play the copy.
                do {
                    let file = try await RemoteMediaCache.localCopy(of: url, filename: asset.filename)
                    guard !Task.isCancelled else { return }
                    player = AVPlayer(url: file)
                } catch {
                    loadError = "Couldn't fetch this video from the server: \(error.localizedDescription)"
                }
            } else {
                guard let (data, response) = try? await URLSession.shared.data(from: url),
                      (response as? HTTPURLResponse)?.statusCode ?? 200 == 200,
                      let fetched = NSImage(data: data) else {
                    loadError = "Couldn't fetch this image from the server."
                    return
                }
                guard !Task.isCancelled else { return }
                image = fetched
            }

        case .missing:
            loadError = "This asset's file isn't on this Mac, and no server copy is known."
        }
    }
}
