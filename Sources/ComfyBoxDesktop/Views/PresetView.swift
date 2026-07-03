// PresetView.swift — Server-backed preset management
//
// Full CRUD against the canonical /v1/presets store (shared with Bree and
// the Telegram bot), replacing the old device-local preset list. Apply maps
// a server preset onto the Generate tab; legacy image-service routing fields
// (engine/provider/mode) are shown as chips and preserved verbatim on save.

import SwiftUI

struct PresetView: View {
    @Bindable var engine: EngineService
    var onApply: ((GenerationPreset) -> Void)?

    @State private var presets: [ServerPreset] = []
    @State private var editing: ServerPreset?
    @State private var isNew: Bool = false
    @State private var isLoading = false
    @State private var loadError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            if let loadError {
                Text(loadError).font(.caption).foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12).padding(.vertical, 6)
            }
            if presets.isEmpty {
                emptyState
            } else {
                presetList
            }
        }
        .navigationTitle("Presets")
        .task { await reload() }
        .onChange(of: engine.connectionState.isConnected) { _, connected in
            if connected { Task { await reload() } }
        }
        .sheet(item: $editing) { preset in
            ServerPresetEditor(
                original: preset,
                isNew: isNew,
                onSave: { updated in Task { await save(updated) } },
                onCancel: { editing = nil }
            )
        }
    }

    private var header: some View {
        HStack {
            Text("Presets")
                .font(.headline)
            Text("\(presets.count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Spacer()
            if isLoading { ProgressView().controlSize(.small) }
            Button { Task { await reload() } } label: { Image(systemName: "arrow.clockwise") }
                .buttonStyle(.borderless)
            Button {
                isNew = true
                editing = ServerPreset(name: "")
            } label: { Label("New Preset", systemImage: "plus") }
                .buttonStyle(.borderedProminent)
                .disabled(!engine.connectionState.isConnected)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var presetList: some View {
        ScrollView {
            LazyVStack(spacing: 8) {
                ForEach(presets) { preset in
                    ServerPresetRow(
                        preset: preset,
                        onApply: { onApply?(preset.toGenerationPreset()) },
                        onEdit: { isNew = false; editing = preset },
                        onDuplicate: { Task { await duplicate(preset) } },
                        onDelete: { Task { await delete(preset) } }
                    )
                }
            }
            .padding(12)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "slider.horizontal.below.rectangle")
                .font(.system(size: 32)).foregroundStyle(.tertiary)
            Text(engine.connectionState.isConnected ? "No presets yet" : "Connect to the server to manage presets")
                .font(.subheadline).foregroundStyle(.secondary)
            if engine.connectionState.isConnected {
                Text("Create one here, or use “Save as Preset” in Generate.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding()
    }

    // MARK: - Actions

    private func reload() async {
        guard engine.connectionState.isConnected else { return }
        isLoading = true; defer { isLoading = false }
        loadError = nil
        presets = await engine.fetchPresets()
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private func save(_ preset: ServerPreset) async {
        do {
            try await engine.savePreset(preset)
            editing = nil
            await reload()
        } catch {
            loadError = "Save failed: \(error.localizedDescription)"
        }
    }

    private func duplicate(_ preset: ServerPreset) async {
        var copy = preset
        copy.id = UUID().uuidString
        copy.name = "\(preset.name) Copy"
        await save(copy)
    }

    private func delete(_ preset: ServerPreset) async {
        do {
            try await engine.deletePreset(id: preset.id)
            await reload()
        } catch {
            loadError = "Delete failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Row

private struct ServerPresetRow: View {
    let preset: ServerPreset
    var onApply: () -> Void
    var onEdit: () -> Void
    var onDuplicate: () -> Void
    var onDelete: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: preset.mediaKind == "video" ? "film" : "photo")
                .font(.title3).foregroundStyle(.secondary).frame(width: 24)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(preset.name).font(.headline)
                    if let engineName = preset.engine {
                        chip(engineName)
                    }
                    if let provider = preset.provider, provider != "local" {
                        chip(provider)
                    }
                }
                if !preset.description.isEmpty {
                    Text(preset.description)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                Text(summaryLine)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            Button("Apply", action: onApply)
                .controlSize(.small)
                .buttonStyle(.borderedProminent)
            Button { onEdit() } label: { Image(systemName: "pencil") }
                .buttonStyle(.borderless)
            Menu {
                Button("Duplicate", action: onDuplicate)
                Button("Delete", role: .destructive, action: onDelete)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .buttonStyle(.borderless)
            .fixedSize()
        }
        .padding(12)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
    }

    private var summaryLine: String {
        var parts: [String] = []
        if let model = preset.model ?? preset.customModelPath {
            parts.append((model as NSString).lastPathComponent)
        }
        if let w = preset.width, let h = preset.height { parts.append("\(w)×\(h)") }
        if let steps = preset.steps { parts.append("\(steps) steps") }
        if let guidance = preset.guidance { parts.append(String(format: "g %.1f", guidance)) }
        if !preset.loras.isEmpty { parts.append("\(preset.loras.count) LoRA\(preset.loras.count == 1 ? "" : "s")") }
        if let scheduler = preset.scheduler { parts.append(scheduler) }
        return parts.joined(separator: " · ")
    }

    private func chip(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
            .foregroundStyle(.secondary)
    }
}

// MARK: - Editor

private struct ServerPresetEditor: View {
    let original: ServerPreset
    let isNew: Bool
    let onSave: (ServerPreset) -> Void
    let onCancel: () -> Void

    @State private var name: String
    @State private var descriptionText: String
    @State private var prompt: String
    @State private var negativePrompt: String
    @State private var model: String
    @State private var widthText: String
    @State private var heightText: String
    @State private var stepsText: String
    @State private var guidanceText: String
    @State private var lorasText: String
    @State private var scheduler: String

    init(original: ServerPreset, isNew: Bool,
         onSave: @escaping (ServerPreset) -> Void, onCancel: @escaping () -> Void) {
        self.original = original
        self.isNew = isNew
        self.onSave = onSave
        self.onCancel = onCancel
        _name = State(initialValue: original.name)
        _descriptionText = State(initialValue: original.description)
        _prompt = State(initialValue: original.prompt ?? "")
        _negativePrompt = State(initialValue: original.negativePrompt ?? "")
        _model = State(initialValue: original.customModelPath ?? original.model ?? "")
        _widthText = State(initialValue: original.width.map(String.init) ?? "")
        _heightText = State(initialValue: original.height.map(String.init) ?? "")
        _stepsText = State(initialValue: original.steps.map(String.init) ?? "")
        _guidanceText = State(initialValue: original.guidance.map { String(format: "%g", $0) } ?? "")
        _lorasText = State(initialValue: original.loras
            .map { $0.scale == 1.0 ? $0.filename : "\($0.filename)=\($0.scale)" }
            .joined(separator: ", "))
        _scheduler = State(initialValue: original.scheduler ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(isNew ? "New Preset" : "Edit Preset").font(.headline).padding()
            Divider()
            Form {
                TextField("Name", text: $name)
                TextField("Description", text: $descriptionText)
                Section("Prompt") {
                    TextField("Prompt template", text: $prompt, axis: .vertical).lineLimit(2...6)
                    TextField("Negative prompt", text: $negativePrompt, axis: .vertical).lineLimit(1...3)
                }
                Section("Model & Parameters") {
                    TextField("Model (name or path)", text: $model)
                    HStack {
                        TextField("Width", text: $widthText).frame(width: 90)
                        Text("×").foregroundStyle(.secondary)
                        TextField("Height", text: $heightText).frame(width: 90)
                        Spacer()
                    }
                    HStack {
                        TextField("Steps", text: $stepsText).frame(width: 90)
                        TextField("Guidance", text: $guidanceText).frame(width: 90)
                        TextField("Scheduler", text: $scheduler)
                    }
                    TextField("LoRAs (file=scale, comma-separated)", text: $lorasText)
                }
            }
            .formStyle(.grouped)
            Divider()
            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                Button("Save") { onSave(buildPreset()) }
                    .buttonStyle(.borderedProminent)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
        }
        .frame(minWidth: 540, idealWidth: 600, minHeight: 540, idealHeight: 620)
    }

    /// Apply the edited fields onto the original so unedited (legacy routing)
    /// fields pass through untouched.
    private func buildPreset() -> ServerPreset {
        var p = original
        if p.id.isEmpty { p.id = UUID().uuidString }
        p.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        p.description = descriptionText.trimmingCharacters(in: .whitespacesAndNewlines)
        p.prompt = prompt.isEmpty ? nil : prompt
        p.negativePrompt = negativePrompt.isEmpty ? nil : negativePrompt
        let modelValue = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if modelValue.hasPrefix("/") || modelValue.hasPrefix("~") {
            p.customModelPath = modelValue
        } else {
            p.model = modelValue.isEmpty ? nil : modelValue
            p.customModelPath = nil
        }
        p.width = Int(widthText)
        p.height = Int(heightText)
        p.steps = Int(stepsText)
        p.guidance = Double(guidanceText.replacingOccurrences(of: ",", with: "."))
        p.scheduler = scheduler.isEmpty ? nil : scheduler
        p.loras = lorasText.split(separator: ",").compactMap { part in
            let token = part.trimmingCharacters(in: .whitespaces)
            guard !token.isEmpty else { return nil }
            let pieces = token.split(separator: "=", maxSplits: 1)
            let filename = String(pieces[0]).trimmingCharacters(in: .whitespaces)
            let scale = pieces.count > 1 ? (Double(pieces[1]) ?? 1.0) : 1.0
            return ServerPresetLora(filename: filename, scale: scale)
        }
        return p
    }
}

// MARK: - Save Preset Sheet (called from GenerationView)

struct SavePresetSheet: View {
    var promptTemplate: String
    var modelId: String?
    var loras: [LoRASelection]
    var steps: Int
    var guidance: Float
    var width: Int
    var height: Int
    var onSave: (String) -> Void
    var onCancel: () -> Void

    @State private var presetName: String = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Save as Preset")
                    .font(.headline)
                Spacer()
            }
            .padding()

            Divider()

            Form {
                Section("Preset Name") {
                    TextField("My Preset", text: $presetName)
                }

                Section("Settings to Save") {
                    LabeledContent("Steps", value: "\(steps)")
                    LabeledContent("Guidance", value: String(format: "%.1f", guidance))
                    LabeledContent("Resolution", value: "\(width) x \(height)")
                    if let model = modelId {
                        LabeledContent("Model", value: model)
                    }
                    if !loras.isEmpty {
                        LabeledContent("LoRAs", value: "\(loras.count) selected")
                    }
                    if !promptTemplate.isEmpty {
                        LabeledContent("Prompt") {
                            Text(promptTemplate)
                                .lineLimit(3)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .formStyle(.grouped)

            Divider()

            HStack {
                Button("Cancel") { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save Preset") { onSave(presetName) }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(presetName.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
        }
        .frame(width: 400, height: 360)
    }
}
