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

    // Generation parameters
    @State private var prompt: String = ""
    @State private var selectedResolution: ResolutionPreset = ResolutionPreset.presets[2]
    @State private var customWidth: Int = 1024
    @State private var customHeight: Int = 1024
    @State private var steps: Double = 9
    @State private var guidance: Double = 3.5
    @State private var seedText: String = ""
    @State private var displayedImage: NSImage?

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
        .onAppear { consumePendingPreset() }
        .onChange(of: pendingPreset?.id) { _, _ in consumePendingPreset() }
        .sheet(isPresented: $showingSavePreset) {
            if let pm = presetManager {
                SavePresetSheet(
                    promptTemplate: prompt,
                    modelId: engine.currentModel,
                    loras: selectedLoras,
                    steps: Int(steps),
                    guidance: Float(guidance),
                    width: effectiveWidth,
                    height: effectiveHeight,
                    onSave: { name in
                        _ = pm.create(
                            name: name,
                            promptTemplate: prompt,
                            modelId: engine.currentModel,
                            loras: selectedLoras,
                            steps: Int(steps),
                            guidance: Float(guidance),
                            width: effectiveWidth,
                            height: effectiveHeight
                        )
                        showingSavePreset = false
                    },
                    onCancel: { showingSavePreset = false }
                )
            }
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

                Divider()

                // Prompt
                promptSection

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
        }
    }

    private var actionButtons: some View {
        VStack(spacing: 8) {
            // Generate button
            Button(action: { submitGeneration() }) {
                HStack {
                    if engine.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                            .padding(.trailing, 4)
                        Text("Generating...")
                    } else {
                        Image(systemName: "wand.and.stars")
                        Text("Generate")
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
        engine.connectionState.isConnected
            && !engine.isGenerating
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
                VStack(spacing: 12) {
                    ProgressView()
                        .controlSize(.large)
                    Text("Generating image...")
                        .foregroundStyle(.secondary)
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
            loras: selectedLoras
        )

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

            do {
                let outputPath = try await engine.generate(request)
                // Load the generated image for display.
                if let image = NSImage(contentsOfFile: outputPath) {
                    await MainActor.run {
                        displayedImage = image
                    }
                }
                // Notify app to ingest the generated file into DAM.
                onGenerated?(outputPath, request)
            } catch {
                // Surface the error to the UI. EngineService sets lastError
                // for server errors, but not for every failure mode (e.g.
                // connection loss), so record it here as well.
                await MainActor.run {
                    engine.lastError = error.localizedDescription
                }
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

    /// Apply a preset to the current generation parameters.
    func applyPreset(_ preset: GenerationPreset) {
        prompt = preset.promptTemplate
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
