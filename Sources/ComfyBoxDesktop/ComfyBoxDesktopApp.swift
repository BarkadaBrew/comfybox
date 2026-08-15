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
    @State private var engine: EngineService
    @State private var store: DAMStore?
    @State private var ingestor: AssetIngestor?
    @State private var archiver: GalleryArchiver?
    @State private var presetManager = PresetManager()
    @State private var healthMonitor = HealthMonitor()
    @State private var promptLibrary = PromptLibraryStore()
    @State private var canvasStore = CanvasStore()
    @State private var mfluxService = MfluxService()
    @State private var breeService = BreeService()
    @State private var kiraClient = KiraClient()
    /// App-wide "Rated G unless NSFW toggled" gate (Todd 2026-07-17). Starts
    /// hidden every launch by design.
    @State private var contentGate = AppContentGate()
    @State private var showNSFWReveal = false
    @State private var nsfwPasswordInput = ""
    @State private var nsfwPasswordError = false
    @State private var decoupageService = DecoupageService()
    @State private var faceSwapService = FaceSwapService()
    @State private var activityLog = ActivityLog()
    @State private var agentService: AgentService
    @State private var pendingPromptInsert: String?
    @State private var pendingReferenceImage: String?
    @State private var pendingMotionReference: String?
    @State private var pendingInpaintImage: String?
    @State private var pendingContentMode: ContentMode?

    init() {
        let engine = EngineService()
        _engine = State(initialValue: engine)
        _agentService = State(initialValue: AgentService(engine: engine))
    }
    @State private var uiScale: String? = DesktopSettings.load().uiScale
    @State private var selectedTab: AppTab = .gallery
    @State private var initError: String?
    @State private var characters: [CharacterEntry] = []
    @State private var comparisonAssets: [DAMAsset]?
    @State private var pendingPreset: GenerationPreset?
    @State private var gallerySearchFocusRequests: Int = 0
    @State private var showCommandPalette = false
    @State private var showSplash = true

    enum AppTab: String, CaseIterable {
        case dashboard = "Dashboard"
        case generate = "Generate"
        case gallery = "Gallery"
        case compare = "Compare"
        case presets = "Presets"
        case prompts = "Prompts"
        case assistant = "Assistant"
        case motion = "Motion"
        case mflux = "mflux"
        case decoupage = "Découpage"
        case face = "Face"
        case inpaint = "Inpaint"
        case bree = "Bree"
        case kira = "Kira"
        case canvas = "Canvas"
        case civitai = "CivitAI"
        case models = "Models"
        case characters = "Characters"
        case applications = "Applications"
        case queue = "Queue"
        case remoteGallery = "Remote Gallery"

        /// Sidebar grouping for the hub.
        enum Section: String, CaseIterable, Identifiable {
            case create = "Create"
            case library = "Library"
            case operate = "Operate"
            case suite = "Suite"
            var id: String { rawValue }
        }

        var section: Section {
            switch self {
            case .generate, .motion, .mflux, .decoupage, .face, .inpaint, .canvas, .assistant: return .create
            case .gallery, .compare, .presets, .prompts, .characters, .civitai, .models, .remoteGallery: return .library
            case .dashboard, .applications, .queue: return .operate
            case .bree, .kira: return .suite
            }
        }

        static func tabs(in section: Section) -> [AppTab] {
            allCases.filter { $0.section == section }
        }

        var icon: String {
            switch self {
            case .dashboard: return "gauge.with.dots.needle.bottom.50percent"
            case .generate: return "wand.and.stars"
            case .gallery: return "photo.on.rectangle"
            case .compare: return "square.grid.2x2"
            case .presets: return "slider.horizontal.below.rectangle"
            case .prompts: return "text.book.closed"
            case .assistant: return "sparkles"
            case .motion: return "film.stack"
            case .mflux: return "cube.transparent"
            case .decoupage: return "square.3.layers.3d"
            case .face: return "person.crop.circle.badge.checkmark"
            case .inpaint: return "paintbrush.pointed"
            case .bree: return "brain.head.profile"
            case .kira: return "moon.stars"
            case .canvas: return "rectangle.on.rectangle.angled"
            case .civitai: return "globe"
            case .models: return "square.stack.3d.up.fill"
            case .characters: return "person.2.crop.square.stack"
            case .applications: return "square.grid.3x3.square"
            case .queue: return "list.bullet.rectangle"
            case .remoteGallery: return "photo.stack"
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
            case .prompts: return "9"
            case .civitai: return "0"
            case .canvas: return "y"
            case .assistant: return "i"
            case .motion: return "m"
            case .mflux: return "x"
            case .bree: return "b"
            case .kira: return "k"
            case .models: return "l"
            case .decoupage: return "d"
            case .applications: return "a"
            case .queue: return "q"
            case .remoteGallery: return "r"
            case .face: return "f"
            case .inpaint: return "e"
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            NavigationSplitView {
                List(selection: $selectedTab) {
                    ForEach(AppTab.Section.allCases) { section in
                        Section(section.rawValue) {
                            ForEach(AppTab.tabs(in: section), id: \.self) { tab in
                                Label(tab.rawValue, systemImage: tab.icon).tag(tab)
                            }
                        }
                    }
                }
                .navigationSplitViewColumnWidth(min: 150, ideal: 170, max: 210)
            } detail: {
                detailView
            }
            .environment(contentGate)
            .navigationTitle("CoffeeShop Desktop")
            .frame(minWidth: 900, minHeight: 600)
            .overlay {
                if showSplash {
                    SplashView(isPresented: $showSplash)
                        .transition(.opacity)
                        .zIndex(10)
                }
            }
            .overlay {
                if showCommandPalette {
                    ZStack(alignment: .top) {
                        Color.black.opacity(0.15).ignoresSafeArea()
                            .onTapGesture { showCommandPalette = false }
                        CommandPaletteView(isPresented: $showCommandPalette, commands: paletteCommands)
                            .padding(.top, 80)
                            .shadow(radius: 30)
                    }
                    .transition(.opacity)
                }
            }
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
            // Hidden reveal trigger — NO visible button anywhere (a visible
            // "Show NSFW" control would advertise hidden content to anyone who
            // opens the app). Reveal is a keyboard shortcut only: ⌃⌥⌘U toggles
            // the gate (prompting for the gallery password if one is set). The
            // gate re-hides on every launch regardless. (Todd 2026-07-18)
            .background {
                Button("", action: toggleNSFWReveal)
                    .keyboardShortcut("u", modifiers: [.command, .option, .control])
                    .hidden()
            }
            .sheet(isPresented: $showNSFWReveal) {
                VStack(alignment: .leading, spacing: 12) {
                    Label("Unlock content", systemImage: "lock.open.fill")
                        .font(.headline)
                    Text("Enter the gallery password. Content re-locks on relaunch.")
                        .font(.callout).foregroundStyle(.secondary)
                    SecureField("Password", text: $nsfwPasswordInput)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(submitNSFWReveal)
                    if nsfwPasswordError {
                        Text("Incorrect password").font(.caption).foregroundStyle(.red)
                    }
                    HStack {
                        Spacer()
                        Button("Cancel") { showNSFWReveal = false; nsfwPasswordInput = "" }
                        Button("Unlock") { submitNSFWReveal() }.buttonStyle(.borderedProminent)
                    }
                }
                .padding(20)
                .frame(width: 380)
            }
            .task {
                applySettings()
                await initializeDAM()
                await loadCharacters()
            }
        }
        .defaultSize(width: 1200, height: 800)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About \(Branding.appName)") { showAboutPanel() }
            }
            CommandGroup(after: .toolbar) {
                Button("Command Palette") { showCommandPalette.toggle() }
                    .keyboardShortcut("k", modifiers: .command)
            }

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

        MenuBarExtra("CoffeeShop", systemImage: menuBarSymbol) {
            Text("CoffeeShop Suite").font(.headline)
            Divider()
            Button(engine.connectionState.isConnected
                   ? "Connected · \(engine.serverHost):\(engine.serverPort)"
                   : "Disconnected") {
                engine.connectionState.isConnected ? engine.disconnect() : engine.connect()
            }
            if let down = downServices, !down.isEmpty {
                Text("⚠ Down: \(down.joined(separator: ", "))")
            }
            Divider()
            Button("Restart ComfyBox Daemon") {
                Task {
                    try? await ServiceController().perform(.restart, on: WatchedService(
                        name: "ComfyBox Server", urlString: "",
                        control: ServiceControl(launchdLabel: "com.barkadabrew.comfybox")))
                }
            }
            Button("New Render") { selectedTab = .generate; activate() }
            Button("Open CoffeeShop Desktop") { activate() }
            if !activityLog.recent.isEmpty {
                Divider()
                Text("Recent")
                ForEach(activityLog.recent.prefix(6)) { entry in
                    Label(entry.message, systemImage: entry.icon)
                }
            }
            Divider()
            Button("Quit CoffeeShop Desktop") { NSApp.terminate(nil) }
        }
        .menuBarExtraStyle(.menu)
    }

    /// Menu-bar glyph reflecting suite status (template-rendered, so vary the
    /// symbol rather than color): coffee cup when healthy, warning otherwise.
    private var menuBarSymbol: String {
        if let down = downServices, !down.isEmpty { return "exclamationmark.triangle.fill" }
        return engine.connectionState.isConnected ? "cup.and.saucer.fill" : "cup.and.saucer"
    }

    private var downServices: [String]? {
        let down = healthMonitor.services.filter { $0.state == .down }.map { $0.service.name }
        return down.isEmpty ? nil : down
    }

    private func activate() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.windows.first?.makeKeyAndOrderFront(nil)
    }

    /// Native About panel — macOS auto-includes the app icon as the logo.
    private func showAboutPanel() {
        let credits = NSAttributedString(
            string: "\(Branding.tagline)\n\nThe hub for the entire Coffeeshop suite — images, video, and voice; models & LoRAs; mflux, Découpage, and face identity; stack monitoring; and Bree.\n\nPowered by the ComfyBox engine (Z-Image / MLX).",
            attributes: [.foregroundColor: NSColor.secondaryLabelColor,
                         .font: NSFont.systemFont(ofSize: 11)])
        NSApplication.shared.orderFrontStandardAboutPanel(options: [
            .applicationName: Branding.appName,
            .applicationVersion: Branding.version,
            .credits: credits,
        ])
        NSApp.activate(ignoringOtherApps: true)
    }

    // MARK: - Command Palette

    /// Commands offered by ⌘K: navigate to any tab + a handful of actions.
    private var paletteCommands: [PaletteCommand] {
        var commands: [PaletteCommand] = AppTab.allCases.map { tab in
            PaletteCommand(
                title: "Go to \(tab.rawValue)",
                subtitle: "Tab",
                systemImage: tab.icon,
                keywords: [tab.rawValue],
                action: { selectedTab = tab }
            )
        }
        commands.append(contentsOf: [
            PaletteCommand(
                title: engine.connectionState.isConnected ? "Disconnect Server" : "Connect Server",
                subtitle: "ComfyBox", systemImage: "bolt.horizontal",
                keywords: ["server", "connection"],
                action: { engine.connectionState.isConnected ? engine.disconnect() : engine.connect() }),
            PaletteCommand(
                title: "Restart ComfyBox Daemon", subtitle: "launchctl kickstart",
                systemImage: "arrow.clockwise", keywords: ["daemon", "server", "restart", "launchctl"],
                action: {
                    Task {
                        try? await ServiceController().perform(.restart, on: WatchedService(
                            name: "ComfyBox Server", urlString: "",
                            control: ServiceControl(launchdLabel: "com.barkadabrew.comfybox")))
                    }
                }),
            PaletteCommand(
                title: "Reload Bree Handoff", subtitle: "Bree", systemImage: "brain.head.profile",
                keywords: ["companion", "vault"], action: { breeService.reload(); selectedTab = .bree }),
            PaletteCommand(
                title: "New Render", subtitle: "Generate", systemImage: "wand.and.stars",
                keywords: ["generate", "image"], action: { selectedTab = .generate }),
            PaletteCommand(
                title: "Find in Gallery", subtitle: "Gallery", systemImage: "magnifyingglass",
                keywords: ["search"], action: { selectedTab = .gallery; gallerySearchFocusRequests += 1 }),
            PaletteCommand(
                title: "Open Settings", subtitle: "Preferences", systemImage: "gearshape",
                keywords: ["preferences", "config", "keys"],
                action: {
                    if #available(macOS 14, *) {
                        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
                    }
                }),
        ])
        return commands
    }

    // MARK: - Detail View Router

    @ViewBuilder
    private var detailView: some View {
        switch selectedTab {
        case .dashboard:
            DashboardView(engine: engine, monitor: healthMonitor, store: store, ingestor: ingestor)

        case .applications:
            ApplicationsView(engine: engine)

        case .queue:
            QueueView(engine: engine)

        case .remoteGallery:
            RemoteGalleryView(engine: engine, ingestor: ingestor)

        case .characters:
            CharactersView(engine: engine)

        case .generate:
            GenerationView(
                engine: engine,
                presetManager: presetManager,
                characters: characters,
                onGenerated: handleGenerated,
                onBatchComplete: handleBatchComplete,
                pendingPreset: $pendingPreset,
                pendingPromptInsert: $pendingPromptInsert,
                pendingReferenceImage: $pendingReferenceImage,
                pendingContentMode: $pendingContentMode,
                agent: agentService
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

        case .assistant:
            AgentView(
                agent: agentService,
                onUsePrompt: { prompt in
                    pendingPromptInsert = prompt
                    selectedTab = .generate
                }
            )

        case .motion:
            MotionView(engine: engine, pendingMotionReference: $pendingMotionReference)

        case .mflux:
            MfluxView(mflux: mfluxService, ingestor: ingestor)

        case .decoupage:
            DecoupageView(decoupage: decoupageService, ingestor: ingestor)

        case .face:
            FaceView(mflux: mfluxService, faceSwap: faceSwapService, ingestor: ingestor)

        case .inpaint:
            InpaintView(engine: engine, ingestor: ingestor, pendingImage: $pendingInpaintImage)

        case .bree:
            BreeView(bree: breeService)

        case .kira:
            KiraView(client: kiraClient)

        case .canvas:
            CanvasView(
                store: canvasStore,
                onSendToGenerate: { path in
                    Task { await sendCanvasImageToGenerate(path) }
                },
                onUseAsReference: { path in
                    pendingReferenceImage = path
                    selectedTab = .generate
                }
            )

        case .civitai:
            CivitAIBrowserView(engine: engine, promptLibrary: promptLibrary)

        case .models:
            ModelsView(engine: engine)

        case .gallery:
            if let store = store, let ingestor = ingestor {
                GalleryView(
                    store: store,
                    ingestor: ingestor,
                    engine: engine,
                    onCompare: { assets in
                        comparisonAssets = assets
                        selectedTab = .compare
                    },
                    onUseAsReference: { asset in
                        pendingReferenceImage = asset.absolutePath
                        selectedTab = .generate
                    },
                    onSendToGenerate: { asset in
                        guard let recipe = ImageRecipe.read(fromImageAt: asset.absolutePath, fallback: asset) else { return }
                        pendingPreset = recipe.preset
                        pendingContentMode = recipe.contentMode
                        selectedTab = .generate
                    },
                    onAnimate: { asset in
                        pendingMotionReference = asset.absolutePath
                        selectedTab = .motion
                    },
                    onInpaint: { asset in
                        pendingInpaintImage = asset.absolutePath
                        selectedTab = .inpaint
                    },
                    canvasStore: canvasStore,
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

    /// Reveal/hide toggle, invoked ONLY by the hidden ⌃⌥⌘U keyboard shortcut —
    /// there is deliberately no visible control (Todd 2026-07-18). Revealing
    /// prompts for the gallery password when one is configured; the gate
    /// re-hides on every launch regardless.
    private func toggleNSFWReveal() {
        if contentGate.revealed {
            contentGate.hide()
        } else if contentGate.requiresPassword {
            nsfwPasswordInput = ""
            nsfwPasswordError = false
            showNSFWReveal = true
        } else {
            contentGate.reveal()
        }
    }

    private func submitNSFWReveal() {
        if contentGate.reveal(withPassword: nsfwPasswordInput) {
            showNSFWReveal = false
            nsfwPasswordInput = ""
            nsfwPasswordError = false
        } else {
            nsfwPasswordError = true
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
        // One-time: move any plaintext API keys into the Keychain.
        AppSecrets.migrateFromSettingsIfNeeded()
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
            let galleryArchiver = GalleryArchiver(store: damStore, ingestor: assetIngestor)
            store = damStore
            ingestor = assetIngestor
            archiver = galleryArchiver
            await assetIngestor.startWatching()

            // Crash recovery (T7): finish any archive whose source removal
            // was interrupted by a crash between commit and cleanup.
            // Fire-and-forget — must not block app launch.
            let archiveRoots = DesktopSettings.load().archiveRoots ?? [DesktopSettings.defaultArchiveRoot]
            Task {
                await galleryArchiver.resumePendingRemovals(in: archiveRoots)
            }
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
    /// Re-render a canvas image: load its original prompt (from the DAM asset
    /// at that path, else the JSON sidecar) into Generate and switch tabs.
    private func sendCanvasImageToGenerate(_ path: String) async {
        var prompt: String?
        if let store, let asset = try? await store.fetchAsset(byPath: path) {
            prompt = asset.prompt
        }
        if prompt == nil {
            // Fall back to a {name}.json sidecar next to the file.
            let sidecar = ((path as NSString).deletingPathExtension) + ".json"
            if let data = FileManager.default.contents(atPath: sidecar),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                prompt = json["prompt"] as? String
            }
        }
        if let prompt, !prompt.isEmpty {
            pendingPromptInsert = prompt
        }
        selectedTab = .generate
    }

    private func handleGenerated(_ path: String, _ request: GenerationRequest) {
        activityLog.log("photo", "Rendered \((path as NSString).lastPathComponent)")
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

    /// After a multi-image batch (Studio Packs FR-4 / #202), hand the whole
    /// set off to the Compare tab — same pendingSelection mechanism Gallery's
    /// "Compare" action already uses. ingestFile is upsert-safe (INSERT OR
    /// REPLACE keyed by id), so re-ingesting paths handleGenerated already
    /// queued independently is safe, not a duplicate.
    private func handleBatchComplete(_ paths: [String], _ request: GenerationRequest) {
        guard let ingestor = ingestor else { return }
        Task {
            var assets: [DAMAsset] = []
            for path in paths {
                writeSidecarIfMissing(for: path, request: request)
                if let asset = try? await ingestor.ingestFile(at: path) {
                    assets.append(asset)
                }
            }
            guard !assets.isEmpty else { return }
            await MainActor.run {
                comparisonAssets = assets
                selectedTab = .compare
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
        if !request.loras.isEmpty {
            metadata["loras"] = request.loras.map { lora in
                ["name": lora.filename.replacingOccurrences(of: ".safetensors", with: ""),
                 "scale": Double(lora.scale)]
            }
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
