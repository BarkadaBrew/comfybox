// ComfyBoxDesktopApp.swift — SwiftUI app entry point
//
// Main application for ComfyBox Desktop. Creates the EngineService
// on launch and provides the primary window with a tabbed interface
// for generation and gallery views.

import SwiftUI

@main
struct ComfyBoxDesktopApp: App {
    @State private var engine = EngineService()
    @State private var selectedTab: AppTab = .generate

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
                    GenerationView(engine: engine)
                case .gallery:
                    GalleryView()
                }
            }
            .navigationTitle("ComfyBox Desktop")
            .frame(minWidth: 900, minHeight: 600)
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    connectionButton
                }
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
}

// MARK: - Settings View

struct SettingsView: View {
    @Bindable var engine: EngineService

    var body: some View {
        Form {
            Section("Server Connection") {
                TextField("Host", text: $engine.serverHost)
                TextField("Port", value: $engine.serverPort, format: .number)
            }

            Section("Output") {
                TextField("Output Directory", text: $engine.outputDirectory)
            }
        }
        .formStyle(.grouped)
        .frame(width: 400, height: 200)
    }
}
