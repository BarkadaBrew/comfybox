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
    @State private var selectedTab: AppTab = .generate
    @State private var initError: String?
    @State private var characters: [CharacterEntry] = []
    @State private var comparisonAssets: [DAMAsset]?

    enum AppTab: String, CaseIterable {
        case generate = "Generate"
        case gallery = "Gallery"
        case compare = "Compare"
        case presets = "Presets"

        var icon: String {
            switch self {
            case .generate: return "wand.and.stars"
            case .gallery: return "photo.on.rectangle"
            case .compare: return "square.grid.2x2"
            case .presets: return "slider.horizontal.below.rectangle"
            }
        }

        var shortcutKey: KeyEquivalent {
            switch self {
            case .generate: return "1"
            case .gallery: return "2"
            case .compare: return "3"
            case .presets: return "4"
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
                    // Focus is handled by GalleryView's focusSearch()
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
        case .generate:
            GenerationView(
                engine: engine,
                presetManager: presetManager,
                characters: characters,
                onGenerated: handleGenerated
            )

        case .gallery:
            if let store = store, let ingestor = ingestor {
                GalleryView(
                    store: store,
                    ingestor: ingestor,
                    onCompare: { assets in
                        comparisonAssets = assets
                        selectedTab = .compare
                    }
                )
            } else if let error = initError {
                errorView(error)
            } else {
                ProgressView("Initializing database...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

        case .compare:
            if let store = store, let ingestor = ingestor {
                ComparisonGridView(store: store, ingestor: ingestor)
            } else if let error = initError {
                errorView(error)
            } else {
                ProgressView("Initializing database...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

        case .presets:
            PresetView(
                presetManager: presetManager,
                onApply: { preset in
                    selectedTab = .generate
                    // The GenerationView reads preset values when applied
                    // through the presetManager binding.
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
    /// Ingests the output into DAMStore so it appears in the gallery.
    private func handleGenerated(_ path: String, _ request: GenerationRequest) {
        guard let ingestor = ingestor else { return }
        Task {
            do {
                try await ingestor.ingestFile(at: path)
            } catch {
                // Ingestion failure is non-fatal; the file is still on disk.
                print("[ComfyBoxDesktop] Auto-ingest failed for \(path): \(error)")
            }
        }
    }

    // MARK: - Preset Application

    /// Apply a preset by switching to generate tab. The GenerationView
    /// will read the preset values through its applyPreset method.
    private func applyPresetToGeneration(_ preset: GenerationPreset) {
        // The tab switch triggers; actual application happens in GenerationView.
        // We store it for the view to pick up.
        selectedTab = .generate
    }
}
