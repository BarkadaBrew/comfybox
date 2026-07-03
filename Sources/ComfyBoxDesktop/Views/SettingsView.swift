// SettingsView.swift — Application settings with persistence
//
// Provides configuration for server connection, output directory,
// default generation parameters, and gallery settings. Settings
// persist to ~/.comfybox/desktop-config.json.

import SwiftUI
import ZImage

// MARK: - AI Provider form models

/// Editable string fields for one AI-provider endpoint, mapped to/from `AIProviderEndpoint`.
struct EndpointForm {
    var baseUrl: String = ""
    var model: String = ""
    var apiKey: String = ""

    init() {}
    init(_ endpoint: AIProviderEndpoint?) {
        baseUrl = endpoint?.baseUrl ?? ""
        model = endpoint?.model ?? ""
        apiKey = endpoint?.apiKey ?? ""
    }

    /// A configured endpoint requires both a base URL and a model; otherwise it's absent.
    func toEndpoint() -> AIProviderEndpoint? {
        let b = baseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
        let m = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !b.isEmpty, !m.isEmpty else { return nil }
        let k = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return AIProviderEndpoint(baseUrl: b, model: m, apiKey: k.isEmpty ? nil : k)
    }
}

/// The full editable provider registry for the Settings form.
struct ProviderFormBundle {
    var prompt = EndpointForm()
    var vision = EndpointForm()
    var captioning = EndpointForm()

    init() {}
    init(_ registry: AIProviderRegistry) {
        prompt = EndpointForm(registry.promptOptimization)
        vision = EndpointForm(registry.vision)
        captioning = EndpointForm(registry.captioning)
    }

    func toRegistry() -> AIProviderRegistry {
        AIProviderRegistry(
            promptOptimization: prompt.toEndpoint(),
            vision: vision.toEndpoint(),
            captioning: captioning.toEndpoint()
        )
    }
}

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
    /// Health-board endpoints. Optional so configs written before the
    /// health board still decode; nil means "use defaults".
    var watchedServices: [WatchedService]?
    /// CivitAI API key (optional; unlocks auth-gated listings + downloads).
    var civitaiApiKey: String?

    /// Starter set for the health board when nothing is configured yet:
    /// the coffeeshop stack (Bree's server web UI, the legacy image service)
    /// plus the local LM Studio prompt-enhancement endpoint.
    static let defaultWatchedServices: [WatchedService] = [
        WatchedService(name: "LM Studio", urlString: "http://127.0.0.1:1234/v1/models"),
        WatchedService(name: "Image Service", urlString: "http://127.0.0.1:7861/health"),
        WatchedService(name: "Bree Server", urlString: "http://10.0.100.232:3000/health"),
    ]

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
        gallerySortDefault: "date",
        watchedServices: nil,
        civitaiApiKey: nil
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

    // AI provider registry (server config, /v1/config)
    @State private var providerForm = ProviderFormBundle()
    @State private var loadedServerConfig: ComfyBoxServerConfig?
    @State private var providerStatus: String?
    @State private var providerIsError: Bool = false

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

            providersTab
                .tabItem {
                    Label("AI Providers", systemImage: "brain")
                }
        }
        .frame(width: 480, height: 460)
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

            Section("CivitAI") {
                SecureField("API key (optional — unlocks gated listings & downloads)", text: Binding(
                    get: { settings.civitaiApiKey ?? "" },
                    set: { settings.civitaiApiKey = $0.isEmpty ? nil : $0 }
                ))
                .onChange(of: settings.civitaiApiKey) { _, _ in hasUnsavedChanges = true }
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

    // MARK: - AI Providers Tab

    private var providersTab: some View {
        Form {
            Section {
                Text("Local AI endpoints (OpenAI-compatible, e.g. LM Studio). Stored server-side in ~/.comfybox/config.json.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            endpointSection("Prompt Optimization", form: $providerForm.prompt)
            endpointSection("Vision (optional)", form: $providerForm.vision)
            endpointSection("Captioning (optional)", form: $providerForm.captioning)

            if let status = providerStatus {
                Section {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(providerIsError ? .red : .secondary)
                }
            }

            Section {
                HStack {
                    Spacer()
                    Button("Reload") { Task { await loadProviders() } }
                    Button("Save") { Task { await saveProviders() } }
                        .buttonStyle(.borderedProminent)
                        .disabled(!engine.connectionState.isConnected)
                    Spacer()
                }
            }
        }
        .formStyle(.grouped)
        .task { await loadProviders() }
    }

    @ViewBuilder
    private func endpointSection(_ title: String, form: Binding<EndpointForm>) -> some View {
        Section(title) {
            TextField("Base URL", text: form.baseUrl)
                .textContentType(.URL)
                .autocorrectionDisabled()
            TextField("Model", text: form.model)
                .autocorrectionDisabled()
            SecureField("API Key (optional)", text: form.apiKey)
        }
    }

    private func loadProviders() async {
        guard engine.connectionState.isConnected else {
            providerStatus = "Connect to the server to edit AI providers."
            providerIsError = true
            return
        }
        do {
            let config = try await engine.fetchServerConfig()
            loadedServerConfig = config
            providerForm = ProviderFormBundle(config.providers)
            providerStatus = nil
            providerIsError = false
        } catch {
            providerStatus = "Failed to load: \(error.localizedDescription)"
            providerIsError = true
        }
    }

    private func saveProviders() async {
        do {
            // Fetch-mutate-save so we preserve server-owned fields PUT would otherwise reset.
            var config: ComfyBoxServerConfig
            if let loaded = loadedServerConfig {
                config = loaded
            } else {
                config = try await engine.fetchServerConfig()
            }
            config.providers = providerForm.toRegistry()
            try await engine.saveServerConfig(config)
            loadedServerConfig = config
            providerStatus = "Saved to ~/.comfybox/config.json"
            providerIsError = false
        } catch {
            providerStatus = "Failed to save: \(error.localizedDescription)"
            providerIsError = true
        }
    }

    // MARK: - Actions

    private func applyAndSave() {
        let endpointChanged = engine.serverHost != settings.serverHost
            || engine.serverPort != settings.serverPort
        let wasConnected = engine.connectionState.isConnected

        // Apply to engine
        engine.serverHost = settings.serverHost
        engine.serverPort = settings.serverPort
        engine.outputDirectory = settings.outputDirectory

        // Reconnect so a new host/port takes effect immediately.
        if endpointChanged && wasConnected {
            engine.disconnect()
            engine.connect()
        }

        // Persist to disk
        settings.save()
        hasUnsavedChanges = false
    }
}
