// SettingsView.swift — Application settings with persistence
//
// Provides configuration for server connection, output directory,
// default generation parameters, and gallery settings. Settings
// persist to ~/.comfybox/desktop-config.json.

import SwiftUI

// MARK: - Settings Storage

/// Persisted settings stored at ~/.comfybox/desktop-config.json.
struct DesktopSettings: Codable {
    var serverHost: String
    var serverPort: UInt16
    var autoConnect: Bool
    var outputDirectory: String
    var defaultSteps: Int
    var defaultGuidance: Float
    var defaultWidth: Int
    var defaultHeight: Int
    var thumbnailSize: Int
    var gallerySortDefault: String

    static let defaultSettings = DesktopSettings(
        serverHost: "127.0.0.1",
        serverPort: 7870,
        autoConnect: true,
        outputDirectory: NSString(string: "~/Pictures/ComfyBox").expandingTildeInPath,
        defaultSteps: 9,
        defaultGuidance: 3.5,
        defaultWidth: 1024,
        defaultHeight: 1024,
        thumbnailSize: 180,
        gallerySortDefault: "date"
    )

    static var configPath: String {
        let dir = NSString(string: "~/.comfybox").expandingTildeInPath
        return (dir as NSString).appendingPathComponent("desktop-config.json")
    }

    /// Load settings from disk, returning defaults on any failure.
    static func load() -> DesktopSettings {
        let path = configPath
        guard FileManager.default.fileExists(atPath: path),
              let data = FileManager.default.contents(atPath: path) else {
            return defaultSettings
        }

        do {
            let decoder = JSONDecoder()
            return try decoder.decode(DesktopSettings.self, from: data)
        } catch {
            return defaultSettings
        }
    }

    /// Save settings to disk.
    func save() {
        let path = Self.configPath
        let dir = (path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true
        )

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(self)
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            // Save failure is non-fatal — settings revert to defaults on next launch.
        }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @Bindable var engine: EngineService

    @State private var settings: DesktopSettings
    @State private var hasUnsavedChanges: Bool = false
    @State private var showingSaveConfirmation: Bool = false

    init(engine: EngineService) {
        self.engine = engine
        self._settings = State(initialValue: DesktopSettings.load())
    }

    var body: some View {
        TabView {
            serverTab
                .tabItem {
                    Label("Server", systemImage: "server.rack")
                }

            generationTab
                .tabItem {
                    Label("Generation", systemImage: "wand.and.stars")
                }

            galleryTab
                .tabItem {
                    Label("Gallery", systemImage: "photo.on.rectangle")
                }
        }
        .frame(width: 480, height: 340)
        .onDisappear {
            if hasUnsavedChanges {
                applyAndSave()
            }
        }
    }

    // MARK: - Server Tab

    private var serverTab: some View {
        Form {
            Section("Connection") {
                TextField("Host", text: $settings.serverHost)
                    .onChange(of: settings.serverHost) { _, _ in hasUnsavedChanges = true }

                TextField("Port", value: $settings.serverPort, format: .number)
                    .onChange(of: settings.serverPort) { _, _ in hasUnsavedChanges = true }

                Toggle("Auto-connect on launch", isOn: $settings.autoConnect)
                    .onChange(of: settings.autoConnect) { _, _ in hasUnsavedChanges = true }
            }

            Section("Status") {
                HStack {
                    Text("Connection")
                    Spacer()
                    HStack(spacing: 4) {
                        Circle()
                            .fill(engine.connectionState.isConnected ? .green : .gray)
                            .frame(width: 8, height: 8)
                        Text(engine.connectionState.label)
                            .foregroundStyle(.secondary)
                    }
                }

                if let model = engine.currentModel {
                    HStack {
                        Text("Active Model")
                        Spacer()
                        Text(model)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }

            Section {
                HStack {
                    Spacer()
                    Button("Apply & Save") {
                        applyAndSave()
                    }
                    .disabled(!hasUnsavedChanges)
                    .buttonStyle(.borderedProminent)

                    Button("Reset to Defaults") {
                        settings = .defaultSettings
                        hasUnsavedChanges = true
                    }
                    Spacer()
                }
            }
        }
        .formStyle(.grouped)
    }

    // MARK: - Generation Tab

    private var generationTab: some View {
        Form {
            Section("Default Parameters") {
                HStack {
                    Text("Steps")
                    Spacer()
                    TextField("Steps", value: $settings.defaultSteps, format: .number)
                        .frame(width: 60)
                        .multilineTextAlignment(.trailing)
                        .onChange(of: settings.defaultSteps) { _, _ in hasUnsavedChanges = true }
                }

                HStack {
                    Text("Guidance")
                    Spacer()
                    TextField("Guidance", value: $settings.defaultGuidance, format: .number.precision(.fractionLength(1)))
                        .frame(width: 60)
                        .multilineTextAlignment(.trailing)
                        .onChange(of: settings.defaultGuidance) { _, _ in hasUnsavedChanges = true }
                }
            }

            Section("Default Resolution") {
                HStack {
                    Text("Width")
                    Spacer()
                    TextField("Width", value: $settings.defaultWidth, format: .number)
                        .frame(width: 80)
                        .multilineTextAlignment(.trailing)
                        .onChange(of: settings.defaultWidth) { _, _ in hasUnsavedChanges = true }
                }

                HStack {
                    Text("Height")
                    Spacer()
                    TextField("Height", value: $settings.defaultHeight, format: .number)
                        .frame(width: 80)
                        .multilineTextAlignment(.trailing)
                        .onChange(of: settings.defaultHeight) { _, _ in hasUnsavedChanges = true }
                }
            }

            Section("Output") {
                TextField("Output Directory", text: $settings.outputDirectory)
                    .onChange(of: settings.outputDirectory) { _, _ in hasUnsavedChanges = true }
            }

            Section {
                HStack {
                    Spacer()
                    Button("Apply & Save") {
                        applyAndSave()
                    }
                    .disabled(!hasUnsavedChanges)
                    .buttonStyle(.borderedProminent)
                    Spacer()
                }
            }
        }
        .formStyle(.grouped)
    }

    // MARK: - Gallery Tab

    private var galleryTab: some View {
        Form {
            Section("Display") {
                HStack {
                    Text("Thumbnail Size")
                    Spacer()
                    Picker("", selection: $settings.thumbnailSize) {
                        Text("Small (140)").tag(140)
                        Text("Medium (180)").tag(180)
                        Text("Large (240)").tag(240)
                    }
                    .frame(width: 160)
                    .onChange(of: settings.thumbnailSize) { _, _ in hasUnsavedChanges = true }
                }

                HStack {
                    Text("Default Sort")
                    Spacer()
                    Picker("", selection: $settings.gallerySortDefault) {
                        Text("Date").tag("date")
                        Text("Rating").tag("rating")
                        Text("Favorites First").tag("favorite")
                    }
                    .frame(width: 160)
                    .onChange(of: settings.gallerySortDefault) { _, _ in hasUnsavedChanges = true }
                }
            }

            Section {
                HStack {
                    Spacer()
                    Button("Apply & Save") {
                        applyAndSave()
                    }
                    .disabled(!hasUnsavedChanges)
                    .buttonStyle(.borderedProminent)
                    Spacer()
                }
            }
        }
        .formStyle(.grouped)
    }

    // MARK: - Actions

    private func applyAndSave() {
        // Apply to engine
        engine.serverHost = settings.serverHost
        engine.serverPort = settings.serverPort
        engine.outputDirectory = settings.outputDirectory

        // Persist to disk
        settings.save()
        hasUnsavedChanges = false
    }
}
