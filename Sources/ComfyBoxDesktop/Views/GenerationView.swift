// GenerationView.swift — Main image generation interface
//
// Provides prompt entry, parameter controls, model selection,
// LoRA picker, queue status, and image preview. Communicates
// with the WarmServer through EngineService. On successful
// generation, calls onGenerated to trigger DAM ingestion.
// Phase 4: Added preset save, prompt enhancement, keyboard shortcuts.

import AppKit
import SwiftUI
import ZImage

/// Common resolution presets.
struct ResolutionPreset: Identifiable, Hashable {
    let id: String
    let width: Int
    let height: Int
    /// Aspect-ratio hint shown next to the pixel size in the picker.
    let hint: String

    init(id: String, width: Int, height: Int, hint: String = "") {
        self.id = id
        self.width = width
        self.height = height
        self.hint = hint
    }

    var label: String {
        hint.isEmpty ? "\(width) × \(height)" : "\(width) × \(height)  (\(hint))"
    }

    /// Sentinel for user-entered dimensions; width/height come from the
    /// custom fields, not this entry.
    static let custom = ResolutionPreset(id: "custom", width: 0, height: 0, hint: "custom")

    static let presets: [ResolutionPreset] = [
        ResolutionPreset(id: "512sq", width: 512, height: 512, hint: "1:1 draft"),
        ResolutionPreset(id: "768sq", width: 768, height: 768, hint: "1:1"),
        ResolutionPreset(id: "1024sq", width: 1024, height: 1024, hint: "1:1"),
        ResolutionPreset(id: "1280sq", width: 1280, height: 1280, hint: "1:1 headshot"),
        ResolutionPreset(id: "768x1024", width: 768, height: 1024, hint: "3:4 portrait"),
        ResolutionPreset(id: "1024x768", width: 1024, height: 768, hint: "4:3 landscape"),
        ResolutionPreset(id: "1024x1536", width: 1024, height: 1536, hint: "2:3 full body"),
        ResolutionPreset(id: "1536x1024", width: 1536, height: 1024, hint: "3:2 landscape"),
        ResolutionPreset(id: "768x1344", width: 768, height: 1344, hint: "9:16 tall"),
        ResolutionPreset(id: "1344x768", width: 1344, height: 768, hint: "16:9 wide"),
        ResolutionPreset(id: "1536sq", width: 1536, height: 1536, hint: "1:1 hi-res"),
    ]
}

struct GenerationView: View {
    @Bindable var engine: EngineService
    var presetManager: PresetManager?
    var characters: [CharacterEntry]
    var onGenerated: ((String, GenerationRequest) -> Void)?
    /// Fired once after a multi-image batch (batchCount > 1) finishes, with
    /// every output path in generation order — lets the app hand the set off
    /// to the Compare tab (Studio Packs FR-4 / #202: batch variation board).
    var onBatchComplete: (([String], GenerationRequest) -> Void)?
    /// Preset queued by the Presets tab; consumed on appear / on change.
    @Binding var pendingPreset: GenerationPreset?
    /// Prompt queued by the Prompt Library; replaces the prompt field once.
    @Binding var pendingPromptInsert: String?
    /// Image queued as an img2img reference (from Gallery/Canvas "Use as
    /// Reference"); consumed once.
    @Binding var pendingReferenceImage: String?
    /// Content mode queued by "Send to Generate" (Gallery/detail); consumed once.
    @Binding var pendingContentMode: ContentMode?
    /// Shared image assistant (Dan's v1.3) that can drive these controls.
    var agent: AgentService?

    // Generation parameters
    // Persisted across tab switches (and app relaunch) so leaving Generate and
    // coming back doesn't wipe your work. @SceneStorage is a drop-in for @State.
    @SceneStorage("gen.prompt") private var prompt: String = ""
    @SceneStorage("gen.negativePrompt") private var negativePrompt: String = ""
    @SceneStorage("gen.resolutionId") private var resolutionId: String = ResolutionPreset.presets[2].id
    @State private var selectedResolution: ResolutionPreset = ResolutionPreset.presets[2]
    @SceneStorage("gen.customWidth") private var customWidth: Int = 1024
    @SceneStorage("gen.customHeight") private var customHeight: Int = 1024
    @SceneStorage("gen.steps") private var steps: Double = 9
    @SceneStorage("gen.guidance") private var guidance: Double = 3.5
    @SceneStorage("gen.seedText") private var seedText: String = ""
    @State private var displayedImage: NSImage?

    // img2img reference
    @State private var referenceImagePath: String?
    @State private var referenceThumbnail: NSImage?
    @State private var imageStrength: Double = 0.6
    @State private var showReference: Bool = false

    /// Output dimensions: the picked preset, or the custom fields.
    private var effectiveWidth: Int {
        selectedResolution.id == ResolutionPreset.custom.id ? customWidth : selectedResolution.width
    }
    private var effectiveHeight: Int {
        selectedResolution.id == ResolutionPreset.custom.id ? customHeight : selectedResolution.height
    }

    private var seedWalkHint: String {
        guard let base = UInt64(seedText), base > 0 else {
            return seedWalkDirection == .random
                ? "\(batchCount) fresh random seeds."
                : "No fixed seed set — every image is already random regardless of direction."
        }
        switch seedWalkDirection {
        case .up: return "Sweeps \(base) → \(base + UInt64(batchCount - 1))."
        case .down: return "Sweeps \(base) → \(max(1, Int(base) - batchCount + 1))."
        case .random: return "\(batchCount) fresh random seeds (ignores the seed field)."
        }
    }

    // LoRA selections
    @State private var selectedLoras: [LoRASelection] = []
    /// Persisted LoRA stack (JSON) so it survives leaving/returning to the tab.
    @SceneStorage("gen.lorasJSON") private var lorasJSON: String = ""

    // Sidebar sections
    @State private var showModelSelector: Bool = true
    @State private var showLoraPicker: Bool = false
    @State private var showQueuePanel: Bool = false
    @State private var showCharacters: Bool = false
    @State private var showCamera: Bool = false
    @State private var showLighting: Bool = false
    @State private var showStudioPacks: Bool = false
    @State private var studioPacks: [StudioPack] = []
    @State private var studioPackWarning: String?
    /// Selected template id per pack id — absent means "no template chosen,
    /// use the raw prompt-field subject" for that pack's Apply button.
    @State private var selectedTemplateByPackId: [String: String] = [:]
    /// Current slot values for whichever template is selected. Reset to the
    /// new template's defaults each time a different template is chosen.
    @State private var templateSlotValues: [String: String] = [:]
    /// The recipe most recently applied via a Studio Pack — drives
    /// vector-first SVG export and metadata recording on the next render.
    @State private var activeStudioPackRecipe: StudioPackRecipe?
    @State private var svgOutputPath: String?
    @State private var svgExportError: String?
    /// QA lint results (FR-8 / #201) — prompt lint runs at apply time,
    /// output lint runs after generation.
    @State private var promptQAResults: [StudioPackQAResult] = []
    @State private var outputQAResults: [StudioPackQAResult] = []
    /// DyPE high-resolution scaling: "none" | "ntk" | "yarn".
    @State private var dype: String = "none"
    /// Number of images to generate in one batch (seed sweep).
    @State private var batchCount: Int = 1
    /// Direction the batch's seed walks across iterations when a fixed seed
    /// is set — Up/Down sweep from it, Random ignores it entirely.
    @State private var seedWalkDirection: SeedWalkDirection = .up
    /// Number of variants queued via "Add to Queue" this session (display only).
    @State private var queuedVariantCount: Int = 0
    @State private var batchProgress: String?
    /// Set when a LoRA swap fails at generate time, so it's visible instead of
    /// silently rendering with no LoRAs.
    @State private var loraSwapWarning: String?
    /// SeedVR2 upscale of the render (0 = off, else target long-side px).
    @State private var seedvrUpscale: Int = 0
    @State private var showAssistant: Bool = true
    // Cloud backend selection (Local / Replicate / Fal)
    @State private var backend: CloudProvider = .local
    @State private var cloudModel: String = ""
    @State private var isCloudGenerating: Bool = false
    @State private var shotTemplates = ShotTemplateStore()
    @State private var lastAppliedActionSummary: String?
    /// Unresolvable pack/model/LoRA references from the assistant's last
    /// action (FR-6 / #199) — flagged, never silently dropped.
    @State private var agentActionWarning: String?

    // Preset save sheet
    @State private var showingSavePreset: Bool = false
    /// Server presets available to load from the Generate tab.
    @State private var serverPresets: [ServerPreset] = []
    /// Name of the currently-loaded preset (nil = none / custom).
    @State private var activePresetName: String?

    /// Fruit mode steering the optimizer + negative prompt. View state only →
    /// resets to Neutral each launch (never silently persists 🥑).
    @State private var contentMode: ContentMode = .neutral
    /// Content mode → preset id (Settings → Server → Content Mode Defaults).
    @State private var contentModeDefaultPresets: [ContentMode: String] = [:]

    // Prompt enhancement
    @State private var isEnhancing: Bool = false
    @State private var enhanceAvailable: Bool = true

    var body: some View {
        HSplitView {
            // Left panel: Controls
            controlPanel
                .frame(minWidth: 360, idealWidth: 460, maxWidth: 760)

            // Right panel: Image preview
            previewPanel
                .frame(minWidth: 400)
        }
        .onAppear {
            // Restore persisted resolution + LoRA stack first, then let any pending
            // preset/prompt (from another tab) override them.
            if resolutionId == ResolutionPreset.custom.id {
                selectedResolution = .custom
            } else if let match = ResolutionPreset.presets.first(where: { $0.id == resolutionId }) {
                selectedResolution = match
            }
            if selectedLoras.isEmpty, let data = lorasJSON.data(using: .utf8),
               let decoded = try? JSONDecoder().decode([LoRASelection].self, from: data) {
                selectedLoras = decoded
            }
            consumePendingPreset(); consumePendingPrompt(); consumePendingReference(); consumePendingContentMode()
        }
        .onChange(of: selectedResolution.id) { _, id in resolutionId = id }
        .onChange(of: selectedLoras) { _, loras in
            if let data = try? JSONEncoder().encode(loras) { lorasJSON = String(decoding: data, as: UTF8.self) }
        }
        .task { await loadServerPresets() }
        .task {
            studioPacks = StudioPackLibrary.loadAll().packs
        }
        .onChange(of: pendingPreset?.id) { _, _ in consumePendingPreset() }
        .onChange(of: pendingPromptInsert) { _, _ in consumePendingPrompt() }
        .onChange(of: pendingReferenceImage) { _, _ in consumePendingReference() }
        .onChange(of: pendingContentMode) { _, _ in consumePendingContentMode() }
        .sheet(isPresented: $showingSavePreset) {
            SavePresetSheet(
                promptTemplate: prompt,
                modelId: engine.currentModel,
                loras: selectedLoras,
                steps: Int(steps),
                guidance: Float(guidance),
                width: effectiveWidth,
                height: effectiveHeight,
                onSave: { name in
                    // Save to the canonical server preset store (shared with
                    // Bree/Telegram), not the old device-local list.
                    let preset = ServerPreset(
                        name: name,
                        prompt: prompt.isEmpty ? nil : prompt,
                        steps: Int(steps),
                        guidance: guidance,
                        width: effectiveWidth,
                        height: effectiveHeight,
                        loras: selectedLoras.map {
                            ServerPresetLora(filename: $0.filename, scale: Double($0.scale))
                        }
                    )
                    var withModel = preset
                    if let model = engine.currentModel {
                        if model.hasPrefix("/") { withModel.customModelPath = model }
                        else { withModel.model = model }
                    }
                    // Capture the seed so the preset reproduces exactly (0/empty = random).
                    if let s = UInt64(seedText), s > 0 { withModel.seed = Int(truncatingIfNeeded: s) }
                    let toSave = withModel
                    Task {
                        try? await engine.savePreset(toSave)
                        await loadServerPresets()
                    }
                    activePresetName = name
                    showingSavePreset = false
                },
                onCancel: { showingSavePreset = false }
            )
        }
    }

    // MARK: - Control Panel

    private var controlPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Server status bar
                serverStatusBar

                Divider()

                // Model selector (collapsible)
                DisclosureGroup(isExpanded: $showModelSelector) {
                    ModelSelector(engine: engine)
                        .padding(.top, 4)
                } label: {
                    Label("Model", systemImage: "cpu")
                        .font(.headline)
                }

                // Image assistant (Dan's v1.3) — can populate the controls below.
                if let agent {
                    DisclosureGroup(isExpanded: $showAssistant) {
                        GenerateAssistantPanel(
                            agent: agent,
                            onApply: { action in applyAgentAction(action) }
                        )
                        .padding(.top, 4)
                    } label: {
                        HStack(spacing: 6) {
                            Label("Assistant", systemImage: "sparkles").font(.headline)
                            if let summary = lastAppliedActionSummary {
                                Text("applied: \(summary)")
                                    .font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                            }
                        }
                    }
                    if let warning = agentActionWarning {
                        Text(warning)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                    Divider()
                }

                // Preset load / save
                presetBar

                Divider()

                // Backend (Local / Replicate / Fal)
                backendSection

                Divider()

                // Prompt
                promptSection

                Divider()

                // Reference image (img2img)
                referenceSection

                Divider()

                // Parameters
                parameterSection

                Divider()

                // LoRA picker (collapsible)
                DisclosureGroup(isExpanded: $showLoraPicker) {
                    LoRAPicker(engine: engine, selectedLoras: $selectedLoras)
                        .padding(.top, 4)
                    selectedLoraKeywordsRow
                } label: {
                    HStack {
                        Label("LoRA Adapters", systemImage: "slider.horizontal.3")
                            .font(.headline)
                        if !selectedLoras.isEmpty {
                            Text("\(selectedLoras.count)")
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.2))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }

                // Studio Packs (collapsible)
                Divider()
                DisclosureGroup(isExpanded: $showStudioPacks) {
                    studioPacksSection
                        .padding(.top, 4)
                } label: {
                    Label("Studio Packs", systemImage: "square.stack.3d.up")
                        .font(.headline)
                }

                // Camera / shot directives (collapsible)
                Divider()
                DisclosureGroup(isExpanded: $showCamera) {
                    CameraPanel(shotTemplates: shotTemplates) { directive in
                        prompt = directive.appended(to: prompt)
                    } onInsertTemplate: { template in
                        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                        prompt = trimmed.isEmpty ? template.directive : "\(trimmed), \(template.directive)"
                    }
                    .padding(.top, 4)
                } label: {
                    Label("Camera & Shot", systemImage: "camera")
                        .font(.headline)
                }

                // Lighting direction (collapsible)
                Divider()
                DisclosureGroup(isExpanded: $showLighting) {
                    LightingPanel { directive in
                        prompt = directive.appended(to: prompt)
                    }
                    .padding(.top, 4)
                } label: {
                    Label("Lighting", systemImage: "lightbulb")
                        .font(.headline)
                }

                // Characters (collapsible)
                if !characters.isEmpty {
                    Divider()
                    DisclosureGroup(isExpanded: $showCharacters) {
                        CharacterLibraryView(
                            characters: characters,
                            onInsert: { entry in
                                insertCharacterPrompt(entry)
                            }
                        )
                        .frame(maxHeight: 300)
                        .padding(.top, 4)
                    } label: {
                        Label("Characters", systemImage: "person.2")
                            .font(.headline)
                    }
                }

                Divider()

                // Queue panel (collapsible)
                DisclosureGroup(isExpanded: $showQueuePanel) {
                    QueuePanel(engine: engine)
                        .padding(.top, 4)
                } label: {
                    HStack {
                        Label("Queue", systemImage: "list.bullet.rectangle")
                            .font(.headline)
                        if engine.queueCount > 0 {
                            Text("\(engine.queueCount)")
                                .font(.caption2)
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.orange)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }

                Divider()

                // Action buttons
                actionButtons

                // Error display
                if let error = engine.lastError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                Spacer()
            }
            .padding()
        }
    }

    private var serverStatusBar: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)

            Text(engine.connectionState.label)
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            if let model = engine.currentModel {
                Text(model)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            if engine.queueCount > 0 {
                Text("Queue: \(engine.queueCount)")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private var statusColor: Color {
        switch engine.connectionState {
        case .connected: return .green
        case .connecting: return .yellow
        case .disconnected: return .gray
        case .error: return .red
        }
    }

    /// Choose Local (MLX server) or a cloud provider (Replicate / Fal).
    private var backendSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Backend", systemImage: "cpu").font(.headline)
                Spacer()
                Picker("", selection: $backend) {
                    ForEach(CloudProvider.allCases) { Text($0.rawValue).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
                .onChange(of: backend) { _, newValue in
                    cloudModel = newValue.defaultModel
                }
            }
            if backend != .local {
                TextField("Model (e.g. \(backend.defaultModel))", text: $cloudModel)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                if cloudBackendKey.isEmpty {
                    Label("Add a \(backend.rawValue) key in Settings → AI Providers.", systemImage: "key")
                        .font(.caption2).foregroundStyle(.orange)
                } else {
                    Text("Renders on \(backend.rawValue). LoRAs and the local model are ignored; steps map to the provider's schema.")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
    }

    private var cloudBackendKey: String {
        switch backend {
        case .replicate: return AppSecrets.replicate ?? ""
        case .fal: return AppSecrets.fal ?? ""
        case .local: return ""
        }
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Prompt")
                    .font(.headline)
                Spacer()
                // Fruit mode — steers the optimizer + negative prompt (text only).
                Picker("Mode", selection: $contentMode) {
                    ForEach(ContentMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .help("Content mode: steers prompt optimization and negative prompt (not guidance)")
                .onChange(of: contentMode) { _, mode in
                    if let presetId = contentModeDefaultPresets[mode],
                       let preset = serverPresets.first(where: { $0.id == presetId }) {
                        applyPreset(preset.toGenerationPreset())
                        activePresetName = preset.name
                    }
                }
                // Enhance button
                Button(action: { enhancePrompt() }) {
                    HStack(spacing: 4) {
                        if isEnhancing {
                            ProgressView()
                                .controlSize(.mini)
                        } else {
                            Image(systemName: "sparkles")
                        }
                        Text("Enhance")
                    }
                }
                .controlSize(.small)
                .buttonStyle(.bordered)
                .disabled(!canEnhance)
                .help("Send prompt to LLM for enhancement")
            }

            TextEditor(text: $prompt)
                .font(.body)
                .frame(minHeight: 80, maxHeight: 160)
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                )

            Text("Negative prompt").font(.caption).foregroundStyle(.secondary)
            TextField("things to avoid (optional)", text: $negativePrompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...3)
        }
    }

    /// img2img reference image + strength. Nil path = pure text-to-image.
    private var referenceSection: some View {
        DisclosureGroup(isExpanded: $showReference) {
            VStack(alignment: .leading, spacing: 8) {
                if let path = referenceImagePath {
                    HStack(alignment: .top, spacing: 10) {
                        Group {
                            if let thumb = referenceThumbnail {
                                Image(nsImage: thumb).resizable().aspectRatio(contentMode: .fill)
                            } else {
                                Image(systemName: "photo").foregroundStyle(.tertiary)
                            }
                        }
                        .frame(width: 72, height: 72)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        VStack(alignment: .leading, spacing: 4) {
                            Text((path as NSString).lastPathComponent)
                                .font(.caption).lineLimit(1).truncationMode(.middle)
                            Button("Remove", role: .destructive) { clearReference() }
                                .controlSize(.small)
                        }
                        Spacer()
                    }
                    NumericSliderField(label: "Strength", value: $imageStrength,
                                       range: 0...1, step: 0.05, fractionDigits: 2)
                    Text("Higher strength follows the reference more closely; lower lets the prompt reinvent it.")
                        .font(.caption2).foregroundStyle(.tertiary)
                } else {
                    Button {
                        chooseReferenceImage()
                    } label: {
                        Label("Choose Reference Image…", systemImage: "photo.badge.plus")
                    }
                    .controlSize(.small)
                    Text("Add an image to guide generation (image-to-image).")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Label("Reference (img2img)", systemImage: "photo.on.rectangle").font(.headline)
                if referenceImagePath != nil {
                    Text(String(format: "%.0f%%", imageStrength * 100))
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
    }

    private func chooseReferenceImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .image]
        panel.prompt = "Use as Reference"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        setReference(path: url.path)
    }

    /// Set the reference image (called by the picker or an external "use as
    /// reference" hook) and load its thumbnail.
    func setReference(path: String) {
        referenceImagePath = path
        showReference = true
        Task {
            let image = await Task.detached { NSImage(contentsOfFile: path) }.value
            await MainActor.run { referenceThumbnail = image }
        }
    }

    private func clearReference() {
        referenceImagePath = nil
        referenceThumbnail = nil
    }

    /// Consume an img2img reference queued by the Gallery/Canvas.
    private func consumePendingReference() {
        guard let path = pendingReferenceImage, !path.isEmpty else { return }
        pendingReferenceImage = nil
        setReference(path: path)
    }

    /// Trigger-word chips for currently-selected LoRAs — tap to insert into the prompt.
    @ViewBuilder
    private var selectedLoraKeywordsRow: some View {
        let words: [String] = selectedLoras.reduce(into: []) { acc, selection in
            guard let info = engine.availableLoras.first(where: { $0.id == selection.id }) else { return }
            acc.append(contentsOf: info.triggerwords)
        }
        let uniqueWords = Array(NSOrderedSet(array: words)) as? [String] ?? words

        if !uniqueWords.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                Text("Trigger words — tap to insert into prompt")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                FlowLayout(spacing: 4) {
                    ForEach(uniqueWords, id: \.self) { word in
                        Button(action: { insertKeyword(word) }) {
                            Text(word)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.12))
                                .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(.top, 4)
        }
    }

    private func insertKeyword(_ word: String) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        prompt = trimmed.isEmpty ? word : "\(trimmed), \(word)"
    }

    private var parameterSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Parameters")
                .font(.headline)

            // Resolution picker
            VStack(alignment: .leading, spacing: 4) {
                Text("Resolution")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Picker("Resolution", selection: $selectedResolution) {
                    ForEach(ResolutionPreset.presets) { preset in
                        Text(preset.label).tag(preset)
                    }
                    Divider()
                    Text("Custom…").tag(ResolutionPreset.custom)
                }
                .pickerStyle(.menu)
                .labelsHidden()

                if selectedResolution.id == ResolutionPreset.custom.id {
                    HStack(spacing: 6) {
                        TextField("Width", value: $customWidth, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder)
                            .font(.subheadline.monospacedDigit())
                            .frame(width: 70)
                        Text("×").foregroundStyle(.secondary)
                        TextField("Height", value: $customHeight, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder)
                            .font(.subheadline.monospacedDigit())
                            .frame(width: 70)
                        Text("px")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                        Spacer()
                    }
                }
            }

            // Steps
            NumericSliderField(label: "Steps", value: $steps, range: 1...50, step: 1)

            // Guidance
            NumericSliderField(label: "Guidance", value: $guidance, range: 0...20, step: 0.5, fractionDigits: 1)

            // Seed field
            VStack(alignment: .leading, spacing: 4) {
                Text("Seed (empty = random)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                TextField("Random", text: $seedText)
                    .textFieldStyle(.roundedBorder)
            }

            // Batch count + Seed Walk direction
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Batch").font(.subheadline).foregroundStyle(.secondary)
                    Stepper(value: $batchCount, in: 1...16) {
                        Text("\(batchCount) image\(batchCount == 1 ? "" : "s")").font(.subheadline.monospacedDigit())
                    }
                }
                if batchCount > 1 {
                    Text("Seed Walk").font(.subheadline).foregroundStyle(.secondary)
                    Picker("", selection: $seedWalkDirection) {
                        ForEach(SeedWalkDirection.allCases) { direction in
                            Text(direction.label).tag(direction)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    Text(seedWalkHint)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            // SeedVR2 upscale (post-render creative scaling)
            VStack(alignment: .leading, spacing: 4) {
                Text("Upscale (SeedVR2)").font(.subheadline).foregroundStyle(.secondary)
                Picker("", selection: $seedvrUpscale) {
                    Text("Off").tag(0)
                    Text("2048px").tag(2048)
                    Text("4096px").tag(4096)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                if seedvrUpscale > 0 {
                    Text("Each render is upscaled to \(seedvrUpscale)px long-side after generation.")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }

            // DyPE high-resolution scaling
            VStack(alignment: .leading, spacing: 4) {
                Text("High-res scaling (DyPE)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Picker("", selection: $dype) {
                    Text("Off").tag("none")
                    Text("NTK (fast)").tag("ntk")
                    Text("YaRN (quality)").tag("yarn")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text("Dynamic Position Extrapolation renders natively above the model's base resolution. Use for large sizes.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    /// Compact "what will render" summary shown above Generate so the config is
    /// confirmable at a glance — notably the LoRA stack (with scales) that used
    /// to be silently dropped.
    private var configSummary: some View {
        VStack(alignment: .leading, spacing: 3) {
            summaryRow("Model", (engine.currentModel as NSString?)?.lastPathComponent ?? "—")
            summaryRow("LoRAs", selectedLoras.isEmpty
                ? "none"
                : selectedLoras.map {
                    "\($0.filename.replacingOccurrences(of: ".safetensors", with: "")) @\(String(format: "%g", $0.scale))"
                  }.joined(separator: ", "))
            summaryRow("Params", "\(Int(steps)) steps · g\(String(format: "%g", guidance)) · \(effectiveWidth)×\(effectiveHeight) · seed \(seedText.isEmpty ? "random" : seedText) · \(contentMode.rawValue)")
            if let warn = loraSwapWarning {
                Text(warn).foregroundStyle(.orange)
            }
        }
        .font(.caption2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func summaryRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text(label).foregroundStyle(.secondary).frame(width: 46, alignment: .leading)
            Text(value).textSelection(.enabled).lineLimit(3)
            Spacer(minLength: 0)
        }
    }

    private var actionButtons: some View {
        VStack(spacing: 8) {
            configSummary
            // Generate button
            Button(action: { submitGeneration() }) {
                HStack {
                    if engine.isGenerating || isCloudGenerating {
                        ProgressView()
                            .controlSize(.small)
                            .padding(.trailing, 4)
                        Text(batchProgress ?? (isCloudGenerating ? "Generating on \(backend.rawValue)…" : "Generating..."))
                    } else {
                        Image(systemName: "wand.and.stars")
                        Text(backend == .local
                             ? (batchCount > 1 ? "Generate \(batchCount)" : "Generate")
                             : "Generate on \(backend.rawValue)")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canGenerate)
            .keyboardShortcut(.return, modifiers: .command)

            // Add to Queue — submits without taking over the preview, so you
            // can queue several variants (e.g. via Seed Walk) back to back.
            if backend == .local {
                Button(action: { queueVariant() }) {
                    HStack {
                        Image(systemName: "text.badge.plus")
                        Text(queuedVariantCount > 0 ? "Add to Queue (\(queuedVariantCount) pending)" : "Add to Queue")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(!canQueue)
                .help("Submit the current settings as a queued render without taking over the preview pane.")
            }

            // Save / Clear row
            HStack(spacing: 8) {
                Button(action: { showingSavePreset = true }) {
                    HStack {
                        Image(systemName: "square.and.arrow.down")
                        Text("Save Preset")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(presetManager == nil)
                .keyboardShortcut("s", modifiers: .command)

                Button(action: { clearPrompt() }) {
                    HStack {
                        Image(systemName: "doc")
                        Text("New")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .keyboardShortcut("n", modifiers: .command)
            }
        }
    }

    private var canGenerate: Bool {
        let hasPrompt = !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if backend == .local {
            return engine.connectionState.isConnected && !engine.isGenerating && hasPrompt
        }
        // Cloud backends don't need the local server, just a key.
        return !isCloudGenerating && hasPrompt && !cloudBackendKey.isEmpty
    }

    /// Unlike canGenerate, does NOT require the server to be idle — the
    /// whole point of Add to Queue is stacking variants while one runs.
    private var canQueue: Bool {
        backend == .local && engine.connectionState.isConnected
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var canEnhance: Bool {
        engine.connectionState.isConnected
            && !isEnhancing
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && enhanceAvailable
    }

    // MARK: - Preview Panel

    private var previewPanel: some View {
        VStack {
            if engine.isGenerating {
                ZStack(alignment: .bottom) {
                    if let liveFrame = engine.livePreviewImage {
                        Image(nsImage: liveFrame)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .padding()
                            .blur(radius: 1.5)
                            .overlay(alignment: .topLeading) {
                                Label("Live Preview", systemImage: "eye")
                                    .font(.caption2.weight(.semibold))
                                    .padding(.horizontal, 8).padding(.vertical, 4)
                                    .background(.black.opacity(0.5), in: Capsule())
                                    .foregroundStyle(.white)
                                    .padding(10)
                            }
                    }
                    VStack(spacing: 12) {
                        if let pct = engine.queueInfo?.progressPercent, pct > 0 {
                            ProgressView(value: Double(pct), total: 100)
                                .frame(width: 220)
                            Text("\(batchProgress.map { $0 + " · " } ?? "")\(pct)%")
                                .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                        } else {
                            ProgressView().controlSize(.large)
                            Text(batchProgress ?? "Generating image…").foregroundStyle(.secondary)
                        }
                    }
                    .padding(12)
                    .background {
                        if engine.livePreviewImage != nil {
                            RoundedRectangle(cornerRadius: 10).fill(.thinMaterial)
                        }
                    }
                    .padding(.bottom, engine.livePreviewImage != nil ? 20 : 0)
                    .frame(maxWidth: .infinity, maxHeight: engine.livePreviewImage != nil ? nil : .infinity, alignment: .center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let image = displayedImage {
                VStack {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding()

                    if let duration = engine.lastDurationMs {
                        Text("Rendered in \(duration)ms")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.bottom, 8)
                    }
                    if let svgPath = svgOutputPath {
                        HStack(spacing: 4) {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                            Text("SVG exported")
                            Button("Reveal") {
                                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: svgPath)])
                            }
                            .buttonStyle(.link)
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 8)
                    } else if let svgError = svgExportError {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                            Text(svgError)
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 8)
                    }
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "photo")
                        .font(.system(size: 48))
                        .foregroundStyle(.tertiary)
                    Text("Generated images will appear here")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Actions

    private func submitGeneration() {
        let seed: UInt64
        if let parsed = UInt64(seedText), parsed > 0 {
            seed = parsed
        } else {
            seed = 0
        }

        let request = GenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt,
            width: effectiveWidth,
            height: effectiveHeight,
            steps: Int(steps),
            guidance: Float(guidance),
            seed: seed,
            modelId: engine.currentModel,
            loras: selectedLoras,
            initImagePath: referenceImagePath,
            imageStrength: referenceImagePath != nil ? Float(imageStrength) : nil,
            dype: dype == "none" ? nil : dype
        )

        // Cloud backend: route to Replicate / Fal instead of the local server.
        if backend != .local {
            submitCloudGeneration(request: request, seed: seed)
            return
        }

        let count = max(1, batchCount)
        Task {
            await runGenerationBatch(request: request, seed: seed, count: count, updatePreview: true)
        }
    }

    /// Add the current settings as a queued variant instead of taking over
    /// the preview pane — lets you stack up several variants (e.g. via Seed
    /// Walk) without watching each one finish before queuing the next. Runs
    /// through the same server queue as Generate; still results land in the
    /// Gallery/Compare tab via onGenerated/onBatchComplete.
    private func queueVariant() {
        let seed: UInt64
        if let parsed = UInt64(seedText), parsed > 0 {
            seed = parsed
        } else {
            seed = 0
        }

        let request = GenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt,
            width: effectiveWidth,
            height: effectiveHeight,
            steps: Int(steps),
            guidance: Float(guidance),
            seed: seed,
            modelId: engine.currentModel,
            loras: selectedLoras,
            initImagePath: referenceImagePath,
            imageStrength: referenceImagePath != nil ? Float(imageStrength) : nil,
            dype: dype == "none" ? nil : dype
        )

        guard backend == .local else { return }  // cloud queueing isn't wired — local server queue only.

        let count = max(1, batchCount)
        queuedVariantCount += count
        Task {
            await runGenerationBatch(request: request, seed: seed, count: count, updatePreview: false)
        }
    }

    /// Shared batch-generation core for both Generate and Add to Queue.
    /// `updatePreview` controls whether the result takes over the main
    /// preview pane — Add to Queue leaves whatever's currently shown alone.
    private func runGenerationBatch(request: GenerationRequest, seed: UInt64, count: Int, updatePreview: Bool) async {
        // Swap LoRAs if any selected (before generation). Preselected LoRAs
        // left over from a different model (e.g. Send-to-Generate on a
        // Z-Image render, then switching to Krea2) are shown in the picker
        // but skipped here rather than attempted and failing the whole
        // swap — only what's actually compatible with the active model
        // goes to the server.
        let activeModel = engine.currentModelFamily ?? engine.currentModel
        let (compatibleLoras, skippedLoras) = selectedLoras.reduce(into: ([LoRASelection](), [LoRASelection]())) { acc, sel in
            let lora = engine.availableLoras.first { $0.id == sel.id }
            let compat = lora?.modelCompatibility ?? ""
            if case .incompatible = LoRACompatibility.status(loraCompatibility: compat, modelIdentifier: activeModel) {
                acc.1.append(sel)
            } else {
                acc.0.append(sel)
            }
        }

        if !compatibleLoras.isEmpty {
            do {
                try await engine.swapLoras(compatibleLoras)
                await MainActor.run {
                    loraSwapWarning = skippedLoras.isEmpty ? nil
                        : "Skipped \(skippedLoras.count) LoRA(s) not compatible with the active model: \(skippedLoras.map { $0.filename }.joined(separator: ", "))"
                }
            } catch {
                await MainActor.run {
                    loraSwapWarning = "⚠ LoRA load failed — rendering without them: \(error.localizedDescription)"
                }
            }
        } else {
            await MainActor.run {
                loraSwapWarning = skippedLoras.isEmpty ? nil
                    : "Skipped \(skippedLoras.count) LoRA(s) not compatible with the active model: \(skippedLoras.map { $0.filename }.joined(separator: ", "))"
            }
        }

        // Batch: generate `count` images. A fixed seed walks per
        // seedWalkDirection; seed 0 (random) yields a fresh random each time.
        var batchPaths: [String] = []
        for i in 0..<count {
            if updatePreview, count > 1 {
                await MainActor.run { batchProgress = "Generating \(i + 1) of \(count)…" }
            }
            var req = request
            req.seed = BatchSeedSweep.seed(baseSeed: seed, index: i, direction: seedWalkDirection)
            do {
                let outputPath = try await engine.generate(req, contentMode: contentMode)
                // Optional SeedVR2 upscale of the render.
                var finalPath = outputPath
                if seedvrUpscale > 0 {
                    if updatePreview { await MainActor.run { batchProgress = "Upscaling to \(seedvrUpscale)px…" } }
                    if let up = try? await engine.upscale(imagePath: outputPath, targetResolution: seedvrUpscale) {
                        finalPath = up
                    } else {
                        await MainActor.run { engine.lastError = "SeedVR2 upscale unavailable (server needs --seedvr2-weights); kept the base render." }
                    }
                }
                if updatePreview {
                    if let image = NSImage(contentsOfFile: finalPath) {
                        await MainActor.run { displayedImage = image }
                    }
                    if let recipe = activeStudioPackRecipe, let svg = recipe.svgDefaults, svg.enabled {
                        await exportVectorSVG(from: finalPath, recipe: recipe, preset: svg.preset ?? "default")
                    } else {
                        await MainActor.run { svgOutputPath = nil; svgExportError = nil }
                    }
                    if let recipe = activeStudioPackRecipe {
                        let svgWanted = recipe.svgDefaults?.enabled ?? false
                        let results = StudioPackQALinter.lintOutput(
                            rules: recipe.qaRules, packId: recipe.packId,
                            svgWanted: svgWanted, svgExported: svgOutputPath != nil
                        )
                        await MainActor.run { outputQAResults = results }
                    }
                }
                batchPaths.append(finalPath)
                onGenerated?(finalPath, req)
            } catch {
                await MainActor.run { engine.lastError = error.localizedDescription }
                break
            }
        }
        if updatePreview { await MainActor.run { batchProgress = nil } }
        await MainActor.run { queuedVariantCount = max(0, queuedVariantCount - (updatePreview ? 0 : count)) }
        if batchPaths.count > 1 {
            onBatchComplete?(batchPaths, request)
        }
    }

    /// Export a vector-first render's SVG alongside its PNG. SVG failure is
    /// surfaced as a warning and never hides the already-successful PNG.
    private func exportVectorSVG(from pngPath: String, recipe: StudioPackRecipe, preset: String) async {
        let pngURL = URL(fileURLWithPath: pngPath)
        let svgURL = pngURL.deletingPathExtension().appendingPathExtension("svg")
        do {
            try await Task.detached(priority: .utility) {
                try SVGExporter.convert(input: pngURL, output: svgURL, preset: preset)
            }.value
            await MainActor.run {
                svgOutputPath = svgURL.path
                svgExportError = nil
            }
            await recordVectorMetadata(pngPath: pngPath, recipe: recipe, preset: preset)
        } catch {
            await MainActor.run {
                svgOutputPath = nil
                svgExportError = "SVG export failed: \(error.localizedDescription)"
            }
        }
    }

    /// Best-effort: record pack/template/vector-mode provenance as searchable
    /// Finder keywords on the rendered PNG. Never blocks or fails the render.
    private func recordVectorMetadata(pngPath: String, recipe: StudioPackRecipe, preset: String) async {
        let sidecar = SidecarService()
        guard sidecar.isAvailable else { return }
        var keywords = ["studio-pack:\(recipe.packId)", "vector-mode", "svg-preset:\(preset)"]
        if let templateId = recipe.templateId { keywords.append("template:\(templateId)") }
        let existing = await sidecar.read(from: pngPath)
        let metadata = SidecarService.Metadata(
            description: existing.description ?? recipe.prompt,
            keywords: existing.keywords + keywords
        )
        try? await sidecar.embed(metadata, into: pngPath)
    }

    /// Generate via a cloud provider (Replicate / Fal), download the result
    /// into the output directory, and ingest it like a local render.
    private func submitCloudGeneration(request: GenerationRequest, seed: UInt64) {
        let key = cloudBackendKey
        guard !key.isEmpty else {
            engine.lastError = "\(backend.rawValue) API key not set — add it in Settings → AI Providers."
            return
        }
        let client = CloudImageClient(provider: backend, model: cloudModel, apiKey: key)
        let params = CloudImageParams(
            prompt: request.prompt,
            width: request.width, height: request.height,
            steps: request.steps, seed: seed,
            initImagePath: referenceImagePath
        )
        let outputDir = URL(fileURLWithPath: DesktopSettings.load().outputDirectory)

        isCloudGenerating = true
        engine.lastError = nil
        Task {
            defer { isCloudGenerating = false }
            do {
                let fileURL = try await client.generate(params, downloadTo: outputDir)
                if let image = NSImage(contentsOfFile: fileURL.path) {
                    await MainActor.run { displayedImage = image }
                }
                onGenerated?(fileURL.path, request)
            } catch {
                await MainActor.run { engine.lastError = error.localizedDescription }
            }
        }
    }

    private func clearPrompt() {
        prompt = ""
        seedText = ""
        displayedImage = nil
        engine.lastError = nil
    }

    /// Consume a preset queued by the Presets tab, if any.
    private func consumePendingPreset() {
        guard let preset = pendingPreset else { return }
        pendingPreset = nil
        applyPreset(preset)
    }

    /// Consume a prompt queued by the Prompt Library, if any. Replaces the
    /// prompt text; other parameters are untouched.
    private func consumePendingPrompt() {
        guard let text = pendingPromptInsert, !text.isEmpty else { return }
        pendingPromptInsert = nil
        prompt = text
    }

    /// Consume a content mode queued by "Send to Generate" (Gallery/detail), if any.
    private func consumePendingContentMode() {
        guard let mode = pendingContentMode else { return }
        pendingContentMode = nil
        contentMode = mode
    }

    /// Apply an assistant action to the generation controls. Only fields the
    /// action set are changed; a `generate` flag kicks off a render.
    private func applyAgentAction(_ action: AgentAction) {
        let warnings = action.validationWarnings(
            availablePackIds: Set(studioPacks.map { $0.id }),
            availableModelIds: Set([engine.currentModel].compactMap { $0 }),
            availableLoRAFilenames: Set(engine.availableLoras.map { $0.filename })
        )
        agentActionWarning = warnings.isEmpty ? nil : warnings.joined(separator: " ")

        // Apply a named Studio Pack (and template, if also named) first —
        // explicit fields below still override anything it sets.
        if let packId = action.studioPackId, let pack = studioPacks.first(where: { $0.id == packId }) {
            if let templateId = action.templateId, let template = pack.templates.first(where: { $0.id == templateId }) {
                applyStudioPackTemplate(pack, template: template)
            } else {
                applyStudioPack(pack)
            }
        }

        if let p = action.prompt { prompt = p }
        if let s = action.steps { steps = Double(min(max(s, 1), 50)) }
        if let g = action.guidance { guidance = min(max(g, 0), 20) }
        if let w = action.width, let h = action.height {
            if let match = ResolutionPreset.presets.first(where: { $0.width == w && $0.height == h }) {
                selectedResolution = match
            } else {
                customWidth = w
                customHeight = h
                selectedResolution = .custom
            }
        }
        if let seed = action.seed { seedText = seed > 0 ? String(seed) : "" }
        lastAppliedActionSummary = action.summary
        if action.generate == true, !engine.isGenerating {
            submitGeneration()
        }
    }

    // MARK: - Preset bar (load / save presets from the Generate tab)

    @ViewBuilder private var presetBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "slider.horizontal.below.rectangle").foregroundStyle(.secondary)
            Menu {
                if serverPresets.isEmpty {
                    Text("No saved presets")
                } else {
                    ForEach(serverPresets) { preset in
                        Button {
                            applyServerPreset(preset)
                        } label: {
                            if activePresetName == preset.name {
                                Label(preset.name, systemImage: "checkmark")
                            } else {
                                Text(preset.name)
                            }
                        }
                    }
                }
                Divider()
                Button { Task { await loadServerPresets() } } label: { Label("Refresh", systemImage: "arrow.clockwise") }
            } label: {
                HStack(spacing: 4) {
                    Text(activePresetName ?? "Load Preset").lineLimit(1)
                    Image(systemName: "chevron.down").font(.caption2)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .menuStyle(.borderlessButton)
            .fixedSize(horizontal: false, vertical: true)

            Button { showingSavePreset = true } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .controlSize(.small)
            .help("Save the current prompt, LoRAs and settings as a preset")
        }
        .padding(8)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    private func loadServerPresets() async {
        serverPresets = await engine.fetchPresets()
        if let config = try? await engine.fetchServerConfig() {
            contentModeDefaultPresets = Dictionary(uniqueKeysWithValues: config.contentModeDefaultPresets.compactMap { key, value in
                ContentMode(rawValue: key).map { ($0, value) }
            })
        }
    }

    /// Switch to a different server preset's model/LoRAs/settings while
    /// keeping the current prompt and seed — so you can rerender the same
    /// thing with a different model+LoRA combo (e.g. a Krea2 preset in
    /// place of a Z-Image one) without retyping anything.
    private func applyServerPreset(_ preset: ServerPreset) {
        applyPreset(preset.toGenerationPreset(), preserveContent: true)
        activePresetName = preset.name
    }

    // MARK: - Studio Packs

    @ViewBuilder
    private var studioPacksSection: some View {
        if studioPacks.isEmpty {
            Text("No Studio Packs available")
                .font(.caption)
                .foregroundStyle(.tertiary)
        } else {
            VStack(alignment: .leading, spacing: 8) {
                Text("Applies prompt, negative prompt, settings, and LoRA stack. Pick a template and fill its slots, or use the prompt field's text as the subject directly.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(studioPacks) { pack in
                    studioPackCard(pack)
                }
                if let warning = studioPackWarning {
                    Text(warning)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
                if !promptQAResults.isEmpty {
                    qaResultsView(title: "Prompt QA", results: promptQAResults)
                }
                if !outputQAResults.isEmpty {
                    qaResultsView(title: "Output QA", results: outputQAResults)
                }
            }
        }
    }

    @ViewBuilder
    private func qaResultsView(title: String, results: [StudioPackQAResult]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            ForEach(results) { result in
                HStack(spacing: 4) {
                    Image(systemName: result.passed ? "checkmark.circle.fill" : (result.blocks ? "xmark.octagon.fill" : "exclamationmark.triangle.fill"))
                        .foregroundStyle(result.passed ? .green : (result.blocks ? .red : .orange))
                        .font(.caption2)
                    Text(result.description)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func studioPackCard(_ pack: StudioPack) -> some View {
        let selectedTemplate = selectedTemplateByPackId[pack.id].flatMap { id in
            pack.templates.first { $0.id == id }
        }

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(pack.name).font(.caption).fontWeight(.semibold)
                Spacer()
                Button("Apply") { applyStudioPack(pack) }
                    .controlSize(.small)
                    .help("Apply using the current prompt field as the subject")
            }
            Text(pack.description)
                .font(.caption2)
                .foregroundStyle(.secondary)
            if let svg = pack.svgDefaults, svg.enabled {
                Text("SVG export recommended (\(svg.preset ?? "default") preset) — export via CLI --svg for now.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            if !pack.templates.isEmpty {
                Menu {
                    ForEach(pack.templates) { template in
                        Button(template.name) { selectTemplate(template, in: pack) }
                    }
                } label: {
                    Text(selectedTemplate?.name ?? "Choose a template…")
                        .font(.caption2)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                if let template = selectedTemplate {
                    ForEach(template.slots) { slot in
                        HStack(spacing: 4) {
                            Text(slot.label)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .frame(width: 80, alignment: .leading)
                            TextField(
                                slot.placeholder,
                                text: Binding(
                                    get: { templateSlotValues[slot.id] ?? slot.defaultValue },
                                    set: { templateSlotValues[slot.id] = $0 }
                                )
                            )
                            .textFieldStyle(.roundedBorder)
                            .font(.caption2)
                        }
                    }
                    Button("Apply Template") { applyStudioPackTemplate(pack, template: template) }
                        .controlSize(.small)
                }
            }
        }
        .padding(6)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func selectTemplate(_ template: StudioPackTemplate, in pack: StudioPack) {
        selectedTemplateByPackId[pack.id] = template.id
        var values: [String: String] = [:]
        for slot in template.slots { values[slot.id] = slot.defaultValue }
        templateSlotValues = values
    }

    /// Resolve a Studio Pack against the local LoRA library and apply it to
    /// the current generation controls, using the current prompt text as the
    /// subject. Does not touch saved presets.
    private func applyStudioPack(_ pack: StudioPack) {
        let availableIds = Set(engine.availableLoras.map { $0.id })
        let recipe = StudioPackResolver.resolve(pack: pack, subject: prompt, availableLoraIds: availableIds)
        applyResolvedRecipe(recipe, packName: pack.name)
    }

    /// Resolve a Studio Pack template with its filled slot values and apply
    /// the result. Does not touch saved presets.
    private func applyStudioPackTemplate(_ pack: StudioPack, template: StudioPackTemplate) {
        let availableIds = Set(engine.availableLoras.map { $0.id })
        let recipe = StudioPackResolver.resolve(
            pack: pack, template: template, slotValues: templateSlotValues, availableLoraIds: availableIds
        )
        applyResolvedRecipe(recipe, packName: "\(pack.name) — \(template.name)")
    }

    private func applyResolvedRecipe(_ recipe: StudioPackRecipe, packName: String) {
        prompt = recipe.prompt
        if let neg = recipe.negativePrompt { negativePrompt = neg }
        if let s = recipe.steps { steps = Double(s) }
        if let g = recipe.guidance { guidance = Double(g) }
        if let w = recipe.width, let h = recipe.height {
            if let match = ResolutionPreset.presets.first(where: { $0.width == w && $0.height == h }) {
                selectedResolution = match
            } else {
                customWidth = w
                customHeight = h
                selectedResolution = .custom
            }
        }
        selectedLoras = recipe.loras.compactMap { ref in
            guard let info = engine.availableLoras.first(where: { $0.id == ref.loraId }) else { return nil }
            return LoRASelection(id: info.id, filename: info.filename, scale: ref.scale)
        }
        if let modelId = recipe.model, modelId != engine.currentModel {
            Task {
                do {
                    try await engine.activateModel(id: modelId)
                } catch {
                    do {
                        try await engine.loadModel(id: modelId)
                    } catch {
                        engine.lastError = "Failed to activate pack model \(modelId): \(error.localizedDescription)"
                    }
                }
            }
        }
        activePresetName = nil
        studioPackWarning = recipe.warnings.isEmpty ? nil : recipe.warnings.joined(separator: " ")
        lastAppliedActionSummary = "Applied Studio Pack: \(packName)"
        // Carried into the next render so it knows whether to also export
        // SVG and which pack/template to record in metadata.
        activeStudioPackRecipe = recipe
        outputQAResults = []
        promptQAResults = StudioPackQALinter.lintPrompt(
            rules: recipe.qaRules, prompt: recipe.prompt, negativePrompt: recipe.negativePrompt
        )
    }

    /// Apply a preset to the current generation parameters.
    /// Apply a preset's model/LoRAs/settings. `preserveContent` keeps the
    /// current prompt and seed untouched — for switching presets mid-session
    /// (e.g. rerendering the same prompt/seed with a different model+LoRA
    /// combo) rather than loading a preset fresh (Send-to-Generate, the
    /// Presets tab's Apply, both of which should load the preset's own
    /// prompt/seed).
    func applyPreset(_ preset: GenerationPreset, preserveContent: Bool = false) {
        if !preserveContent {
            prompt = preset.promptTemplate
            // Restore a saved seed (nil/0 = random).
            seedText = (preset.seed ?? 0) > 0 ? String(preset.seed!) : ""
        }
        negativePrompt = preset.negativePrompt ?? ""
        steps = Double(preset.steps)
        guidance = Double(preset.guidance)

        // Find a matching resolution preset, else carry the preset's exact
        // dimensions through the custom fields.
        if let match = ResolutionPreset.presets.first(where: {
            $0.width == preset.width && $0.height == preset.height
        }) {
            selectedResolution = match
        } else if preset.width > 0, preset.height > 0 {
            customWidth = preset.width
            customHeight = preset.height
            selectedResolution = .custom
        }

        // Convert preset LoRAs to LoRASelections. Presets reconstructed from
        // embedded image metadata only ever carry a display name (never the
        // real library id — PNG metadata doesn't store it), so look up the
        // actual library id by filename; the picker's "selected" check keys
        // on id, and a mismatched id silently fails to show the checkmark
        // even though generation itself would still use the right file.
        selectedLoras = preset.loras.map { presetLora in
            if let match = engine.availableLoras.first(where: { $0.filename == presetLora.filename }) {
                return LoRASelection(id: match.id, filename: match.filename, scale: presetLora.scale)
            }
            return LoRASelection(id: presetLora.id, filename: presetLora.filename, scale: presetLora.scale)
        }

        // Activate the preset's model via the model-pool API if it differs
        // from the currently active model. Embedded/preset model ids are
        // short display names (e.g. "krea-2-turbo", "cyberrealisticZImage_v50")
        // never the server's full path/spec, so resolve before comparing —
        // a raw string mismatch here used to trigger a doomed activate call
        // on nearly every image sent back to Generate.
        if let modelId = preset.modelId {
            switch ModelReferenceResolver.resolve(
                modelId, currentModel: engine.currentModel,
                availableModels: engine.availableModels.map { ($0.id, $0.displayName) }
            ) {
            case .alreadyActive:
                break
            case .resolved(let resolvedId):
                Task {
                    do {
                        try await engine.activateModel(id: resolvedId)
                    } catch {
                        // Not in the pool yet — try loading (and activating) it.
                        do {
                            try await engine.loadModel(id: resolvedId)
                        } catch {
                            engine.lastError = "Failed to activate preset model \(resolvedId): \(error.localizedDescription)"
                        }
                    }
                }
            case .unresolved:
                engine.lastError = "Preset model '\(modelId)' isn't in the local catalog — keeping the active model."
            }
        }
    }

    /// Insert a character's prompt snippet into the current prompt.
    private func insertCharacterPrompt(_ entry: CharacterEntry) {
        if prompt.isEmpty {
            prompt = entry.promptSnippet
        } else {
            prompt += ", \(entry.promptSnippet)"
        }
    }

    /// Send the current prompt to the LLM enhancement endpoint.
    private func enhancePrompt() {
        guard canEnhance else { return }
        isEnhancing = true

        Task {
            do {
                let enhanced = try await engine.enhancePrompt(prompt, contentMode: contentMode)
                await MainActor.run {
                    prompt = enhanced
                    isEnhancing = false
                }
            } catch {
                await MainActor.run {
                    isEnhancing = false
                    // If endpoint not found, disable the button.
                    if case EngineServiceError.serverError(404, _) = error {
                        enhanceAvailable = false
                    }
                }
            }
        }
    }
}
