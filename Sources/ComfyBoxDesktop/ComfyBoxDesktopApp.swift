// ComfyBoxDesktopApp.swift — SwiftUI app entry point
//
// Main application for ComfyBox Desktop. Creates the EngineService,
// DAMStore, and AssetIngestor on launch. Provides a tabbed interface
// for generation and gallery views. Generation output is automatically
// ingested into the DAM. Settings persist via DesktopSettings.

import SwiftUI

@main
struct ComfyBoxDesktopApp: App {
    @State private var engine = EngineService()
    @State private var store: DAMStore?
    @State private var ingestor: AssetIngestor?
    @State private var selectedTab: AppTab = .generate
    @State private var initError: String?

    enum AppTab: String, CaseIterable {
        case generate = "Generate"
        case gallery = "Gallery"

        var icon: String {
            switch self {
            case .generate: return "wand.and.stars"
            case .gallery: return "photo.on.rectangle"
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
                switch selectedTab {
                case .generate:
                    GenerationView(engine: engine, onGenerated: handleGenerated)
                case .gallery:
                    if let store = store, let ingestor = ingestor {
                        GalleryView(store: store, ingestor: ingestor)
                    } else if let error = initError {
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
                    } else {
                        ProgressView("Initializing database...")
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
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
            }
        }
        .defaultSize(width: 1200, height: 800)

        Settings {
            SettingsView(engine: engine)
        }
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

    // MARK: - Generation → Ingestion Bridge

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
}
