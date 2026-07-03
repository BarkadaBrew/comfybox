// ComfyBoxDesktopApp.swift — SwiftUI app entry point
//
// Main application for ComfyBox Desktop. Creates the EngineService,
// DAMStore, AssetIngestor, and PresetManager on launch. Provides a
// tabbed interface for generation, gallery, comparison, and presets.
// Phase 4: Added preset view, comparison view, character library,
// and keyboard shortcut bindings.

import SwiftUI

@main
struct ComfyBoxDesktopApp: App {
    @State private var engine = EngineService()
    @State private var store: DAMStore?
    @State private var ingestor: AssetIngestor?
    @State private var presetManager = PresetManager()
    @State private var healthMonitor = HealthMonitor()
    @State private var promptLibrary = PromptLibraryStore()
    @State private var pendingPromptInsert: String?
    @State private var uiScale: String? = DesktopSettings.load().uiScale
    @State private var selectedTab: AppTab = .gallery
    @State private var initError: String?
    @State private var characters: [CharacterEntry] = []
    @State private var comparisonAssets: [DAMAsset]?
    @State private var pendingPreset: GenerationPreset?
    @State private var gallerySearchFocusRequests: Int = 0

    enum AppTab: String, CaseIterable {
        case dashboard = "Dashboard"
        case health = "Health"
        case generate = "Generate"
        case gallery = "Gallery"
        case compare = "Compare"
        case presets = "Presets"
        case prompts = "Prompts"
        case civitai = "CivitAI"
        case characters = "Characters"
        case server = "Server"

        var icon: String {
            switch self {
            case .dashboard: return "gauge.with.dots.needle.bottom.50percent"
            case .health: return "waveform.path.ecg"
            case .generate: return "wand.and.stars"
            case .gallery: return "photo.on.rectangle"
            case .compare: return "square.grid.2x2"
            case .presets: return "slider.horizontal.below.rectangle"
            case .prompts: return "text.book.closed"
            case .civitai: return "globe"
            case .characters: return "person.2.crop.square.stack"
            case .server: return "server.rack"
            }
        }

        var shortcutKey: KeyEquivalent {
            switch self {
            case .dashboard: return "1"
            case .generate: return "2"
            case .gallery: return "3"
            case .compare: return "4"
            case .presets: return "5"
            case .characters: return "6"
            case .server: return "7"
            case .health: return "8"
            case .prompts: return "9"
            case .civitai: return "0"
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            NavigationSplitView {
                List(AppTab.allCases, id: \.self, selection: $selectedTab) { tab in
                    Label(tab.rawValue, systemImage: tab.icon)
                }
                .navigationSplitViewColumnWidth(min: 140, ideal: 160, max: 200)
            } detail: {
                detailView
            }
            .navigationTitle("ComfyBox Desktop")
            .frame(minWidth: 900, minHeight: 600)
            .dynamicTypeSize(DesktopSettings.dynamicTypeSize(for: uiScale))
            .onReceive(NotificationCenter.default.publisher(for: DesktopSettings.didChangeNotification)) { _ in
                uiScale = DesktopSettings.load().uiScale
            }
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    connectionButton
                }

                if let ingestor = ingestor {
                    ToolbarItem(placement: .automatic) {
                        ingestorStatus(ingestor)
                    }
                }
            }
            .task {
                applySettings()
                await initializeDAM()
                await loadCharacters()
            }
        }
        .defaultSize(width: 1200, height: 800)
        .commands {
            // Tab switching shortcuts
            CommandGroup(after: .toolbar) {
                ForEach(AppTab.allCases, id: \.self) { tab in
                    Button("Show \(tab.rawValue)") {
                        selectedTab = tab
                    }
                    .keyboardShortcut(tab.shortcutKey, modifiers: .command)
                }
            }

            // Gallery search shortcut
            CommandGroup(after: .textEditing) {
                Button("Find in Gallery") {
                    selectedTab = .gallery
                    // GalleryView consumes the request and focuses its
                    // search field via focusSearch().
                    gallerySearchFocusRequests += 1
                }
                .keyboardShortcut("f", modifiers: .command)
            }
        }

        Settings {
            SettingsView(engine: engine)
        }
    }

    // MARK: - Detail View Router

    @ViewBuilder
    private var detailView: some View {
        switch selectedTab {
        case .dashboard:
            DashboardView(engine: engine, store: store, ingestor: ingestor)

        case .health:
            HealthBoardView(engine: engine, monitor: healthMonitor, store: store)

        case .server:
            ServerView(engine: engine)

        case .characters:
            CharactersView(engine: engine)

        case .generate:
            GenerationView(
                engine: engine,
                presetManager: presetManager,
                characters: characters,
                onGenerated: handleGenerated,
                pendingPreset: $pendingPreset,
                pendingPromptInsert: $pendingPromptInsert
            )

        case .prompts:
            PromptLibraryView(
                library: promptLibrary,
                store: store,
                onInsert: { text in
                    pendingPromptInsert = text
                    selectedTab = .generate
                }
            )

        case .civitai:
            CivitAIBrowserView(engine: engine, promptLibrary: promptLibrary)

        case .gallery:
            if let store = store, let ingestor = ingestor {
                GalleryView(
                    store: store,
                    ingestor: ingestor,
                    onCompare: { assets in
                        comparisonAssets = assets
                        selectedTab = .compare
                    },
                    searchFocusRequests: $gallerySearchFocusRequests
                )
            } else if let error = initError {
                errorView(error)
            } else {
                ProgressView("Initializing database...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

        case .compare:
            if let store = store, let ingestor = ingestor {
                ComparisonGridView(
                    store: store,
                    ingestor: ingestor,
                    pendingSelection: $comparisonAssets
                )
            } else if let error = initError {
                errorView(error)
            } else {
                ProgressView("Initializing database...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

        case .presets:
            PresetView(
                engine: engine,
                onApply: { preset in
                    applyPresetToGeneration(preset)
                }
            )
        }
    }

    // MARK: - Subviews

    private func errorView(_ error: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text("Database Error")
                .font(.title2)
            Text(error)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var connectionButton: some View {
        Button(action: {
            if engine.connectionState.isConnected {
                engine.disconnect()
            } else {
                engine.connect()
            }
        }) {
            HStack(spacing: 4) {
                Circle()
                    .fill(engine.connectionState.isConnected ? .green : .gray)
                    .frame(width: 8, height: 8)
                Text(engine.connectionState.isConnected ? "Connected" : "Connect")
                    .font(.caption)
            }
        }
    }

    private func ingestorStatus(_ ingestor: AssetIngestor) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(ingestor.isWatching ? .blue : .gray)
                .frame(width: 6, height: 6)
            Text("\(ingestor.ingestedCount) ingested")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Settings

    /// Apply persisted settings to the engine on launch.
    private func applySettings() {
        let settings = DesktopSettings.load()
        engine.serverHost = settings.serverHost
        engine.serverPort = settings.serverPort
        engine.outputDirectory = settings.outputDirectory

        if settings.autoConnect {
            engine.connect()
        }

        healthMonitor.watchedServices =
            settings.watchedServices ?? DesktopSettings.defaultWatchedServices
        healthMonitor.startMonitoring()
    }

    // MARK: - Initialization

    private func initializeDAM() async {
        do {
            let damStore = try await DAMStore.open()
            let assetIngestor = AssetIngestor(
                store: damStore,
                watchDirectory: engine.outputDirectory
            )
            store = damStore
            ingestor = assetIngestor
            await assetIngestor.startWatching()
        } catch {
            initError = error.localizedDescription
        }
    }

    private func loadCharacters() async {
        characters = await engine.fetchCharacters()
    }

    // MARK: - Generation -> Ingestion Bridge

    /// Called when GenerationView successfully generates an image.
    /// Writes a metadata sidecar and ingests the output into DAMStore
    /// so it appears in the gallery with its generation parameters.
    private func handleGenerated(_ path: String, _ request: GenerationRequest) {
        guard let ingestor = ingestor else { return }
        Task {
            writeSidecarIfMissing(for: path, request: request)
            do {
                try await ingestor.ingestFile(at: path)
            } catch {
                // Ingestion failure is non-fatal; the file is still on disk.
                print("[ComfyBoxDesktop] Auto-ingest failed for \(path): \(error)")
            }
        }
    }

    /// Write a `{basename}.json` sidecar next to the generated image so
    /// AssetIngestor picks up prompt/seed/steps/size at ingest. Skipped if
    /// a sidecar already exists (e.g. written by the server).
    private func writeSidecarIfMissing(for imagePath: String, request: GenerationRequest) {
        let jsonPath = ((imagePath as NSString).deletingPathExtension) + ".json"
        guard !FileManager.default.fileExists(atPath: jsonPath) else { return }

        var metadata: [String: Any] = [
            "prompt": request.prompt,
            "steps": request.steps,
            "guidance": Double(request.guidance),
            "width": request.width,
            "height": request.height,
        ]
        if request.seed > 0 {
            metadata["seed"] = request.seed
        }
        if let model = request.modelId ?? engine.currentModel {
            metadata["model"] = model
        }

        if let data = try? JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys]) {
            try? data.write(to: URL(fileURLWithPath: jsonPath))
        }
    }

    // MARK: - Preset Application

    /// Queue a preset for the GenerationView and switch to the generate tab.
    /// GenerationView consumes `pendingPreset` onAppear/onChange and applies
    /// it via its applyPreset method.
    private func applyPresetToGeneration(_ preset: GenerationPreset) {
        pendingPreset = preset
        selectedTab = .generate
    }
}
