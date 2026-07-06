// GenerationView.swift — Main image generation interface
//
// Provides prompt entry, parameter controls, model selection,
// LoRA picker, queue status, and image preview. Communicates
// with the WarmServer through EngineService. On successful
// generation, calls onGenerated to trigger DAM ingestion.
// Phase 4: Added preset save, prompt enhancement, keyboard shortcuts.

import SwiftUI

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
    /// Preset queued by the Presets tab; consumed on appear / on change.
    @Binding var pendingPreset: GenerationPreset?
    /// Prompt queued by the Prompt Library; replaces the prompt field once.
    @Binding var pendingPromptInsert: String?
    /// Image queued as an img2img reference (from Gallery/Canvas "Use as
    /// Reference"); consumed once.
    @Binding var pendingReferenceImage: String?
    /// Shared image assistant (Dan's v1.3) that can drive these controls.
    var agent: AgentService?

    // Generation parameters
    @State private var prompt: String = ""
    @State private var selectedResolution: ResolutionPreset = ResolutionPreset.presets[2]
    @State private var customWidth: Int = 1024
    @State private var customHeight: Int = 1024
    @State private var steps: Double = 9
    @State private var guidance: Double = 3.5
    @State private var seedText: String = ""
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

    // LoRA selections
    @State private var selectedLoras: [LoRASelection] = []

    // Sidebar sections
    @State private var showModelSelector: Bool = true
    @State private var showLoraPicker: Bool = false
    @State private var showQueuePanel: Bool = false
    @State private var showCharacters: Bool = false
    @State private var showCamera: Bool = false
    @State private var showLighting: Bool = false
    /// DyPE high-resolution scaling: "none" | "ntk" | "yarn".
    @State private var dype: String = "none"
    /// Number of images to generate in one batch (seed sweep).
    @State private var batchCount: Int = 1
    @State private var batchProgress: String?
    @State private var showAssistant: Bool = true
    // Cloud backend selection (Local / Replicate / Fal)
    @State private var backend: CloudProvider = .local
    @State private var cloudModel: String = ""
    @State private var isCloudGenerating: Bool = false
    @State private var shotTemplates = ShotTemplateStore()
    @State private var lastAppliedActionSummary: String?

    // Preset save sheet
    @State private var showingSavePreset: Bool = false

    // Prompt enhancement
    @State private var isEnhancing: Bool = false
    @State private var enhanceAvailable: Bool = true

    var body: some View {
        HSplitView {
            // Left panel: Controls
            controlPanel
                .frame(minWidth: 340, maxWidth: 420)

            // Right panel: Image preview
            previewPanel
                .frame(minWidth: 400)
        }
        .onAppear { consumePendingPreset(); consumePendingPrompt(); consumePendingReference() }
        .onChange(of: pendingPreset?.id) { _, _ in consumePendingPreset() }
        .onChange(of: pendingPromptInsert) { _, _ in consumePendingPrompt() }
        .onChange(of: pendingReferenceImage) { _, _ in consumePendingReference() }
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
                    Task { try? await engine.savePreset(toSave) }
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
                    Divider()
                }

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

            // Batch count (seed sweep)
            HStack {
                Text("Batch").font(.subheadline).foregroundStyle(.secondary)
                Stepper(value: $batchCount, in: 1...16) {
                    Text("\(batchCount) image\(batchCount == 1 ? "" : "s")").font(.subheadline.monospacedDigit())
                }
                if batchCount > 1 {
                    Text("seed sweep").font(.caption2).foregroundStyle(.tertiary)
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

    private var actionButtons: some View {
        VStack(spacing: 8) {
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
            // Swap LoRAs if any selected (before generation).
            if !selectedLoras.isEmpty {
                do {
                    try await engine.swapLoras(selectedLoras)
                } catch {
                    // LoRA swap failure — still attempt generation with
                    // whatever LoRAs are currently loaded.
                }
            }

            // Batch: generate `count` images. A fixed seed sweeps seed, seed+1…;
            // seed 0 (random) yields a fresh random each time.
            for i in 0..<count {
                if count > 1 { await MainActor.run { batchProgress = "Generating \(i + 1) of \(count)…" } }
                var req = request
                if seed > 0 { req.seed = seed + UInt64(i) }
                do {
                    let outputPath = try await engine.generate(req)
                    if let image = NSImage(contentsOfFile: outputPath) {
                        await MainActor.run { displayedImage = image }
                    }
                    onGenerated?(outputPath, req)
                } catch {
                    await MainActor.run { engine.lastError = error.localizedDescription }
                    break
                }
            }
            await MainActor.run { batchProgress = nil }
        }
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

    /// Apply an assistant action to the generation controls. Only fields the
    /// action set are changed; a `generate` flag kicks off a render.
    private func applyAgentAction(_ action: AgentAction) {
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

    /// Apply a preset to the current generation parameters.
    func applyPreset(_ preset: GenerationPreset) {
        prompt = preset.promptTemplate
        steps = Double(preset.steps)
        guidance = Double(preset.guidance)
        // Restore a saved seed (nil/0 = random).
        seedText = (preset.seed ?? 0) > 0 ? String(preset.seed!) : ""

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

        // Convert preset LoRAs to LoRASelections
        selectedLoras = preset.loras.map {
            LoRASelection(id: $0.id, filename: $0.filename, scale: $0.scale)
        }

        // Activate the preset's model via the model-pool API if it differs
        // from the currently active model.
        if let modelId = preset.modelId, modelId != engine.currentModel {
            Task {
                do {
                    try await engine.activateModel(id: modelId)
                } catch {
                    // Not in the pool yet — try loading (and activating) it.
                    do {
                        try await engine.loadModel(id: modelId)
                    } catch {
                        engine.lastError = "Failed to activate preset model \(modelId): \(error.localizedDescription)"
                    }
                }
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
                let enhanced = try await engine.enhancePrompt(prompt)
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
