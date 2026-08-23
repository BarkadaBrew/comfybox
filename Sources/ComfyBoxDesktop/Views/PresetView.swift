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
    /// The server's configured warm-start model (ComfyBoxServerConfig.modelSpec),
    /// so a preset whose model matches can show a "Warm" badge instead of the
    /// user having to hand-edit ~/.comfybox/config.json or a launchd plist arg.
    @State private var warmModelSpec: String?

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
        .task {
            await reload()
            warmModelSpec = (try? await engine.fetchServerConfig())?.modelSpec
            // The editor's Add-LoRA picker needs the server's library list;
            // Generate populates it lazily, so make sure it's fresh here too.
            await engine.refreshLoras()
        }
        .onChange(of: engine.connectionState.isConnected) { _, connected in
            if connected { Task { await reload() } }
        }
        .sheet(item: $editing) { preset in
            ServerPresetEditor(
                original: preset,
                isNew: isNew,
                availableLoras: engine.availableLoras,
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
            Menu {
                Button("Import from Image Service") { Task { await importLegacy() } }
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .disabled(!engine.connectionState.isConnected)
            .help("Import presets from the old image-service")
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
                        isWarm: presetModelSpec(preset) != nil && presetModelSpec(preset) == warmModelSpec,
                        onApply: { onApply?(preset.toGenerationPreset()) },
                        onEdit: { isNew = false; editing = preset },
                        onDuplicate: { Task { await duplicate(preset) } },
                        onDelete: { Task { await delete(preset) } },
                        onSetWarm: presetModelSpec(preset) != nil ? { Task { await setAsWarm(preset) } } : nil
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

    private func importLegacy() async {
        do {
            let count = try await engine.importLegacyPresets()
            loadError = count > 0
                ? nil
                : "No new presets to import (already imported, or none found)."
            await reload()
        } catch {
            loadError = "Import failed: \(error.localizedDescription)"
        }
    }

    private func duplicate(_ preset: ServerPreset) async {
        var copy = preset
        copy.id = UUID().uuidString
        copy.name = "\(preset.name) copy"   // lowercase — matches Save-as-New's " copy" suffix
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

    /// A preset's effective model spec, matching how Apply/applyPreset already
    /// resolve it (custom path takes precedence over a catalog/CivitAI id).
    private func presetModelSpec(_ preset: ServerPreset) -> String? {
        let path = preset.customModelPath?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let path, !path.isEmpty { return path }
        let model = preset.model?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (model?.isEmpty == false) ? model : nil
    }

    /// Make a preset's model the server's warm-start default: load + activate
    /// it now (so the change is visible immediately) and persist modelSpec to
    /// ~/.comfybox/config.json (so it survives the next server restart) —
    /// replaces having to hand-edit the config file or a launchd plist arg.
    private func setAsWarm(_ preset: ServerPreset) async {
        guard let spec = presetModelSpec(preset) else { return }
        do {
            do {
                try await engine.activateModel(id: spec)
            } catch {
                try await engine.loadModel(id: spec)
            }
            var config = try await engine.fetchServerConfig()
            config.modelSpec = spec
            try await engine.saveServerConfig(config)
            warmModelSpec = spec
        } catch {
            loadError = "Set Warm failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Row

private struct ServerPresetRow: View {
    let preset: ServerPreset
    var isWarm: Bool = false
    var onApply: () -> Void
    var onEdit: () -> Void
    var onDuplicate: () -> Void
    var onDelete: () -> Void
    var onSetWarm: (() -> Void)?

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
                    if isWarm {
                        Label("Warm", systemImage: "flame.fill")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(.orange.opacity(0.15), in: Capsule())
                            .help("This preset's model loads by default on server startup.")
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
                if let onSetWarm, !isWarm {
                    Button("Set as Warm", action: onSetWarm)
                }
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
    let availableLoras: [LoRAInfo]
    let onSave: (ServerPreset) -> Void
    let onCancel: () -> Void

    /// Editable LoRA row — stable identity for ForEach even when the same
    /// file appears twice while the user is rearranging.
    private struct EditableLora: Identifiable, Equatable {
        let id = UUID()
        var filename: String
        var scale: Double
    }

    @State private var name: String
    @State private var descriptionText: String
    @State private var prompt: String
    @State private var negativePrompt: String
    @State private var model: String
    @State private var widthText: String
    @State private var heightText: String
    @State private var stepsText: String
    @State private var guidanceText: String
    /// Kroma is a first-class recipe field, never a generic `loras[]` row.
    /// Keep it visible beside the additional LoRAs while saving edits back to
    /// `kroma.strength`, which is the engine's single source of truth.
    @State private var kromaStrength: Double
    @State private var editableLoras: [EditableLora]
    @State private var scheduler: String
    @State private var saveAsName: String = ""
    @State private var showingSaveAs = false

    init(original: ServerPreset, isNew: Bool, availableLoras: [LoRAInfo],
         onSave: @escaping (ServerPreset) -> Void, onCancel: @escaping () -> Void) {
        self.original = original
        self.isNew = isNew
        self.availableLoras = availableLoras
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
        _kromaStrength = State(initialValue: original.kroma?.strength ?? 0)
        _editableLoras = State(initialValue: original.loras
            .filter { lora in
                guard let kromaFile = original.kroma?.file else { return true }
                return lora.filename != kromaFile
            }
            .map { EditableLora(filename: $0.filename, scale: $0.scale) })
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
                }
                Section("LoRAs") {
                    if original.kroma != nil {
                        structuredKromaRow
                    }
                    loraRows
                    addLoraMenu
                    presetLoraKeywordsRow
                }
            }
            .formStyle(.grouped)
            Divider()
            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                if !isNew {
                    Button("Save as New…") {
                        saveAsName = name.trimmingCharacters(in: .whitespaces) + " copy"
                        showingSaveAs = true
                    }
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                Button("Save") { onSave(buildPreset()) }
                    .buttonStyle(.borderedProminent)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
        }
        .frame(minWidth: 540, idealWidth: 600, minHeight: 540, idealHeight: 620)
        .alert("Save as New Preset", isPresented: $showingSaveAs) {
            TextField("New preset name", text: $saveAsName)
            Button("Cancel", role: .cancel) { }
            Button("Save Copy") {
                var copy = buildPreset()
                copy.id = UUID().uuidString
                copy.name = saveAsName.trimmingCharacters(in: .whitespaces)
                onSave(copy)
            }
            .disabled(saveAsName.trimmingCharacters(in: .whitespaces).isEmpty)
        } message: {
            Text("Creates a separate preset with these settings; “\(original.name)” is left unchanged.")
        }
    }

    // MARK: - LoRA editing

    /// Kroma looks and behaves like an adjustable LoRA in the editor, but it
    /// writes the structured recipe field. Showing it here avoids the tempting
    /// (and invalid) workaround of also adding its file to `loras[]`.
    @ViewBuilder
    private var structuredKromaRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                HStack(spacing: 5) {
                    Text("Kroma")
                    Text("Recipe")
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.accentColor.opacity(0.15), in: Capsule())
                }
                .frame(minWidth: 120, maxWidth: 180, alignment: .leading)
                Slider(value: $kromaStrength, in: 0...1.5, step: 0.05)
                TextField("", value: Binding(
                    get: { kromaStrength },
                    set: { kromaStrength = min(max($0, 0), 1.5) }
                ), format: .number.precision(.fractionLength(0...2)))
                    .font(.system(.caption, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 52)
            }
            Text(original.kroma?.file ?? "Engine-default Kroma file")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.middle)
                .help("Saved as kroma.strength; this file is excluded from the generic LoRA stack.")
        }
    }

    /// One row per selected LoRA: name, scale slider, numeric value, remove.
    @ViewBuilder
    private var loraRows: some View {
        if editableLoras.isEmpty {
            Text(original.kroma == nil ? "No LoRAs — add one below." : "No additional LoRAs — add one below.")
                .font(.caption).foregroundStyle(.secondary)
        }
        ForEach($editableLoras) { $lora in
            HStack(spacing: 8) {
                Text(displayName(for: lora.filename))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(minWidth: 120, maxWidth: 180, alignment: .leading)
                    .help(lora.filename)
                // Slider range matches the field's clamp (-3...3) so a typed
                // value is not silently snapped back on the next slider nudge.
                // NEGATIVE weights are meaningful — bidirectional LoRAs (e.g.
                // the age slider) use the sign as direction (Todd 2026-08-10).
                // The engine itself accepts ±10; ±3 is the sane editing range.
                Slider(value: $lora.scale, in: -3...3, step: 0.05)
                TextField("", value: Binding(
                    get: { lora.scale },
                    // Manual entry may exceed the everyday range on purpose
                    // (e.g. 2.0 overdrive) — clamp only the absurd.
                    set: { lora.scale = min(max($0, -3.0), 3.0) }
                ), format: .number.precision(.fractionLength(0...2)))
                    .font(.system(.caption, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 52)
                Button {
                    editableLoras.removeAll { $0.id == lora.id }
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
                .help("Remove this LoRA")
            }
        }
    }

    /// Picker for adding a LoRA from the server's library. Quarantined and
    /// already-added files are excluded; selection seeds the recommended scale.
    @ViewBuilder
    private var addLoraMenu: some View {
        let added = Set(editableLoras.map(\.filename))
        let structuredKromaFile = original.kroma?.file
        let candidates = availableLoras
            .filter {
                !$0.quarantined
                    && !added.contains($0.filename)
                    && $0.filename != structuredKromaFile
            }
            .sorted { $0.filename.localizedCaseInsensitiveCompare($1.filename) == .orderedAscending }
        Menu {
            if candidates.isEmpty {
                Text(availableLoras.isEmpty ? "No LoRAs on the server (is it connected?)" : "All available LoRAs added")
            }
            ForEach(candidates) { info in
                Button {
                    editableLoras.append(EditableLora(
                        filename: info.filename,
                        scale: Double(info.recommendedScale)
                    ))
                } label: {
                    if info.category.isEmpty {
                        Text(displayName(for: info.filename))
                    } else {
                        Text("\(displayName(for: info.filename))  —  \(info.category)")
                    }
                }
            }
        } label: {
            Label("Add LoRA", systemImage: "plus.circle")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    private func displayName(for filename: String) -> String {
        filename
            .replacingOccurrences(of: ".safetensors", with: "")
            .replacingOccurrences(of: "_", with: " ")
    }

    /// Trigger words for the currently selected LoRAs — tap to insert into
    /// the preset's prompt template.
    @ViewBuilder
    private var presetLoraKeywordsRow: some View {
        let filenames = Set(editableLoras.map(\.filename))
        let words: [String] = availableLoras
            .filter { filenames.contains($0.filename) }
            .reduce(into: []) { $0.append(contentsOf: $1.triggerwords) }
        let uniqueWords = Array(NSOrderedSet(array: words)) as? [String] ?? words

        if !uniqueWords.isEmpty {
            FlowLayout(spacing: 4) {
                ForEach(uniqueWords, id: \.self) { word in
                    Button(action: {
                        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                        prompt = trimmed.isEmpty ? word : "\(trimmed), \(word)"
                    }) {
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
        if var kroma = p.kroma {
            kroma.strength = min(max(kromaStrength, 0), 1.5)
            p.kroma = kroma
        }
        let structuredKromaFile = p.kroma?.file
        p.loras = editableLoras
            .filter { !$0.filename.isEmpty && $0.filename != structuredKromaFile }
            .map { ServerPresetLora(filename: $0.filename, scale: $0.scale) }
        return p
    }
}

// MARK: - Save Preset Sheet (called from GenerationView)

struct SavePresetSheet: View {
    var promptTemplate: String
    var negativePrompt: String = ""
    var modelId: String?
    var loras: [LoRASelection]
    var steps: Int
    var guidance: Float
    var width: Int
    var height: Int
    /// (name, negativePrompt) — the sheet lets the user edit the negative
    /// prompt before saving, so the callback returns the edited value.
    var onSave: (String, String) -> Void
    var onCancel: () -> Void

    @State private var presetName: String = ""
    @State private var editedNegative: String = ""
    @State private var didSeedNegative = false

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
                    TextField("Negative prompt (saved with the preset)",
                              text: $editedNegative, axis: .vertical)
                        .lineLimit(1...3)
                        .font(.caption)
                }
            }
            .formStyle(.grouped)

            Divider()

            HStack {
                Button("Cancel") { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save Preset") { onSave(presetName, editedNegative) }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(presetName.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
        }
        .frame(width: 400, height: 400)
        .onAppear {
            if !didSeedNegative {
                editedNegative = negativePrompt
                didSeedNegative = true
            }
        }
    }
}
