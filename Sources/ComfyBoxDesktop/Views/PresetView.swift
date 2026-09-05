// PresetView.swift — Server-backed preset management
//
// Full CRUD against the canonical /v1/presets store (shared with Bree and
// the Telegram bot), replacing the old device-local preset list. Apply maps
// a server preset onto the Generate tab; legacy image-service routing fields
// (engine/provider/mode) are shown as chips and preserved verbatim on save.

import SwiftUI
import ZImage

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
            // Refresh the picker inventory before the preset list. The editor
            // receives a value snapshot of `availableLoras`; opening it while
            // this fetch is still pending would freeze the old catalog into
            // that sheet until the user closed and reopened it.
            await engine.refreshLoras()
            await reload()
            warmModelSpec = (try? await engine.fetchServerConfig())?.modelSpec
        }
        .onChange(of: engine.connectionState.isConnected) { _, connected in
            if connected {
                Task {
                    await engine.refreshLoras()
                    await reload()
                }
            }
        }
        .sheet(item: $editing) { preset in
            ServerPresetEditor(
                original: preset,
                isNew: isNew,
                availableLoras: engine.availableLoras,
                engine: engine,
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
            Button {
                Task {
                    await engine.refreshLoras()
                    await reload()
                }
            } label: { Image(systemName: "arrow.clockwise") }
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
                Task { await beginEditing(ServerPreset(name: ""), asNew: true) }
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
                        onEdit: { Task { await beginEditing(preset, asNew: false) } },
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

    /// A sheet captures `availableLoras` by value, so make the inventory
    /// current before constructing it. This is also what makes LoRAs imported
    /// while the app is running immediately visible in Presets.
    private func beginEditing(_ preset: ServerPreset, asNew: Bool) async {
        await engine.refreshLoras()
        isNew = asNew
        editing = preset
    }

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
        if let sampler = preset.sampler ?? preset.scheduler {
            let schedule = preset.sigmaSchedule.map { " / \($0)" } ?? ""
            parts.append("\(sampler)\(schedule)")
        } else if let schedule = preset.sigmaSchedule {
            parts.append("default / \(schedule)")
        }
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
    /// #277: lets the panel cross-check its local effective-recipe
    /// computation against `POST /v1/presets/resolve` for an already-saved
    /// preset. Optional so previews/tests can construct the editor without a
    /// live server; the panel still shows the local computation.
    let engine: EngineService?
    let onSave: (ServerPreset) -> Void
    let onCancel: () -> Void
    /// Set once per sheet appearance if the live engine disagrees with (or
    /// rejects) this preset — e.g. flagged invalid at load (WP-E20, AC-44c).
    /// nil means either "matches" or "not checked yet".
    @State private var serverResolveError: String?
    /// Review r2 (I5): the `.task` cross-check must actually COMPARE the
    /// engine's resolved stack against the local computation, not merely
    /// confirm the request succeeded. Set when they disagree (names, scales,
    /// roles, or order) for the saved (as-loaded) preset.
    @State private var serverMismatch: String?

    /// Editable LoRA row — stable identity for ForEach even when the same
    /// file appears twice while the user is rearranging.
    private struct EditableLora: Identifiable, Equatable {
        let id = UUID()
        var filename: String
        var scale: Double
        var role: String?
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
    @State private var editableLoras: [EditableLora]
    @State private var sampler: String
    @State private var sigmaSchedule: String
    @State private var saveAsName: String = ""
    @State private var showingSaveAs = false

    init(original: ServerPreset, isNew: Bool, availableLoras: [LoRAInfo], engine: EngineService? = nil,
         onSave: @escaping (ServerPreset) -> Void, onCancel: @escaping () -> Void) {
        self.original = original
        self.isNew = isNew
        self.availableLoras = availableLoras
        self.engine = engine
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
        // Todd 2026-09-04: kroma is a regular LoRA — `loras[]` is the single
        // source the editor shows, verbatim, no special-casing.
        _editableLoras = State(initialValue: original.loras
            .map { EditableLora(filename: $0.filename, scale: $0.scale, role: $0.role) })
        _sampler = State(initialValue: original.sampler ?? original.scheduler ?? "")
        _sigmaSchedule = State(initialValue: original.sigmaSchedule ?? "")
    }

    private var samplingModelFamily: String? {
        let edited = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !edited.isEmpty { return edited }
        return original.customModelPath ?? original.model
    }

    private var samplingValidationError: String? {
        guard !sampler.isEmpty || !sigmaSchedule.isEmpty else { return nil }
        guard !SamplingRecipeCatalog.supports(
            sampler: sampler.isEmpty ? nil : sampler,
            sigmaSchedule: sigmaSchedule.isEmpty ? nil : sigmaSchedule,
            forModelFamily: samplingModelFamily
        ) else { return nil }
        let family = SamplingRecipeCatalog.canonicalFamily(samplingModelFamily) ?? "this model"
        return "The selected sampler/scheduler pair is not supported by \(family)."
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
                        Spacer()
                    }
                    SamplingRecipePicker(
                        sampler: $sampler,
                        sigmaSchedule: $sigmaSchedule,
                        modelFamily: samplingModelFamily,
                        showsExplanation: true
                    )
                }
                Section("LoRAs") {
                    loraRows
                    addLoraMenu
                    presetLoraKeywordsRow
                }
                Section("Effective recipe") {
                    effectiveRecipeView
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
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty || samplingValidationError != nil)
            }
            .padding()
        }
        .frame(minWidth: 560, idealWidth: 620, minHeight: 590, idealHeight: 680)
        .task {
            // #277 / review r2 (I5): cross-check against the live engine
            // once per sheet appearance, by actually COMPARING its resolved
            // stack (names, scales, roles, order) to the local computation
            // for the SAME (as-loaded, unedited) preset — not merely
            // confirming the request succeeded. Only meaningful for an
            // already-saved preset — the endpoint resolves by id, so it
            // cannot see unsaved edits (the panel above is the live preview
            // for those, and is not what this checks).
            guard !isNew, !original.id.isEmpty, let engine else { return }
            do {
                let serverResolved = try await engine.resolvePreset(id: original.id)
                serverResolveError = nil
                let declared = original.toImagePreset()
                let serverStack = PresetEffectiveRecipePresenter.compute(
                    resolved: serverResolved, declared: declared).loraStack
                let localStack = PresetEffectiveRecipePresenter.compute(declared: declared).loraStack
                serverMismatch = serverStack == localStack
                    ? nil
                    : "Local preview disagrees with the live engine's resolved stack for this preset."
            } catch {
                serverResolveError = error.localizedDescription
                serverMismatch = nil
            }
        }
        .alert("Save as New Preset", isPresented: $showingSaveAs) {
            TextField("New preset name", text: $saveAsName)
            Button("Cancel", role: .cancel) { }
            Button("Save Copy") {
                var copy = buildPreset()
                copy.id = UUID().uuidString
                copy.name = saveAsName.trimmingCharacters(in: .whitespaces)
                onSave(copy)
            }
            .disabled(saveAsName.trimmingCharacters(in: .whitespaces).isEmpty || samplingValidationError != nil)
        } message: {
            Text("Creates a separate preset with these settings; “\(original.name)” is left unchanged.")
        }
    }

    // MARK: - Effective recipe (#277)

    /// What `POST /v1/generate {"preset": id}` would actually run for the
    /// CURRENT field values — recomputed on every render, so it updates as
    /// the user edits. See ``PresetEffectiveRecipePresenter``.
    @ViewBuilder
    private var effectiveRecipeView: some View {
        let recipe = effectiveRecipe
        VStack(alignment: .leading, spacing: 6) {
            if let error = serverResolveError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.caption2).foregroundStyle(.red)
            }
            if let mismatch = serverMismatch {
                Label(mismatch, systemImage: "arrow.triangle.2.circlepath.circle")
                    .font(.caption2).foregroundStyle(.orange)
            }
            if let unresolved = recipe.unresolved {
                VStack(alignment: .leading, spacing: 3) {
                    Label("Label-only — the engine will not expand this preset", systemImage: "tag")
                        .font(.caption).foregroundStyle(.orange)
                    Text(unresolved.message)
                        .font(.caption2).foregroundStyle(.secondary)
                    if let hint = unresolved.hint {
                        Text(hint)
                            .font(.caption2).foregroundStyle(.tertiary)
                    }
                }
            } else {
                LabeledContent("Model", value: recipe.model?.isEmpty == false ? recipe.model! : "Model default")
                if let family = recipe.checkpointFamily, !family.isEmpty {
                    LabeledContent("Checkpoint family", value: family)
                }
                LabeledContent("Steps", value: "\(recipe.steps)")
                LabeledContent("Guidance", value: recipe.guidance.map { String(format: "%.2g", $0) } ?? "Model default")
                let recipeLine = [recipe.sampler, recipe.sigmaSchedule].compactMap { $0 }.joined(separator: " / ")
                if !recipeLine.isEmpty {
                    LabeledContent("Sampler / schedule", value: recipeLine)
                }
                if recipe.loraStack.isEmpty {
                    Text("No LoRAs applied").font(.caption2).foregroundStyle(.secondary)
                } else {
                    ForEach(recipe.loraStack) { lora in
                        HStack(spacing: 6) {
                            Text(lora.filename)
                                .font(.caption2).lineLimit(1).truncationMode(.middle)
                            if let role = lora.role {
                                Text(role)
                                    .font(.caption2)
                                    .padding(.horizontal, 5).padding(.vertical, 1)
                                    .background(.quaternary, in: Capsule())
                            }
                            Spacer()
                            Text(String(format: "%.2f", lora.scale))
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    // MARK: - LoRA editing

    /// One row per selected LoRA: name, scale slider, numeric value, remove.
    /// Todd 2026-09-04: kroma is a regular LoRA — it shows here like any
    /// other row, with whatever role (or none) it was declared with.
    @ViewBuilder
    private var loraRows: some View {
        if editableLoras.isEmpty {
            Text("No LoRAs — add one below.")
                .font(.caption).foregroundStyle(.secondary)
        }
        ForEach($editableLoras) { $lora in
            HStack(spacing: 8) {
                Text(displayName(for: lora.filename))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(minWidth: 120, maxWidth: 180, alignment: .leading)
                    .help(lora.filename)
                Menu {
                    Button("Style / unassigned") { lora.role = nil }
                    Divider()
                    Button("Accelerator") { lora.role = "accel" }
                    Button("Kroma") { lora.role = "kroma" }
                    Button("Bypass") { lora.role = "bypass" }
                    Button("Control") { lora.role = "control" }
                } label: {
                    Text(roleLabel(for: lora.role))
                        .font(.caption2)
                        .lineLimit(1)
                        .frame(width: 72, alignment: .leading)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Declare what this LoRA does. Krea-2 distill files must be Accelerator; the role is not inferred from the filename.")
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
        let candidates = availableLoras
            .filter {
                !$0.quarantined
                    && !added.contains($0.filename)
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
                        scale: Double(info.recommendedScale),
                        role: nil
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

    private func roleLabel(for role: String?) -> String {
        switch role {
        case "accel": return "Accelerator"
        case "kroma": return "Kroma"
        case "bypass": return "Bypass"
        case "control": return "Control"
        case .some(let role): return role.capitalized
        case nil: return "Style"
        }
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
        p.sampler = sampler.isEmpty ? nil : sampler
        p.sigmaSchedule = sigmaSchedule.isEmpty ? nil : sigmaSchedule
        // Keep the legacy sampler spelling synchronized for older preset
        // consumers; modern engine validation and Generate use `sampler`.
        p.scheduler = p.sampler
        // Todd 2026-09-04: kroma is a regular LoRA — `loras[]` (editableLoras)
        // is the single source. Review r2, C1 (Critical): `p.kroma` is a
        // DEPRECATED, derived, read-only echo — carrying `original.kroma`
        // through unedited (as `bypass`/`upscale` legitimately do) resurrects
        // a row the user just deleted, because the server's compatibility
        // shim folds a non-nil `kroma` back into `loras[]` on the next save.
        // The desktop must NEVER send it; `ServerPreset.encode` also never
        // emits it, belt and braces.
        p.kroma = nil
        p.loras = editableLoras
            .filter { !$0.filename.isEmpty }
            .map { ServerPresetLora(filename: $0.filename, scale: $0.scale, role: $0.role) }
        return p
    }

    /// #277: what the engine would actually run for this preset's CURRENT
    /// (possibly unsaved) field values — recomputed on every render, so
    /// editing a field updates it live.
    private var effectiveRecipe: EffectiveRecipe {
        PresetEffectiveRecipePresenter.compute(declared: buildPreset().toImagePreset())
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
    var sampler: String = ""
    var sigmaSchedule: String = ""
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
                    LabeledContent("Sampler", value: sampler.isEmpty ? "Model Default" : sampler)
                    LabeledContent("Scheduler", value: sigmaSchedule.isEmpty ? "Model Default" : sigmaSchedule)
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
        .frame(width: 420, height: 440)
        .onAppear {
            if !didSeedNegative {
                editedNegative = negativePrompt
                didSeedNegative = true
            }
        }
    }
}
