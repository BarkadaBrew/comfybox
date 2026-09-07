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
    /// #359: "Make expandable" (one preset) / "Backfill all" (below).
    @State private var backfillViewModel = PresetBackfillViewModel()
    /// Minted ONCE, when a run finishes — computing `id = UUID()` inside the
    /// `.sheet(item:)` getter gave the box a new identity on every SwiftUI
    /// evaluation, which re-presents the sheet.
    @State private var backfillResults: BackfillResultsBox?

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
        .sheet(item: $backfillResults) { results in
            BackfillResultsView(results: results.items) { backfillResults = nil }
        }
    }

    /// `.sheet(item:)` needs an `Identifiable` — wraps the array so an empty
    /// (but non-nil) result set still shows "nothing to backfill" instead of
    /// silently not presenting. The id is fixed at construction, not
    /// recomputed per render.
    struct BackfillResultsBox: Identifiable {
        let id = UUID()
        let items: [PresetBackfillViewModel.Outcome]
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
                Divider()
                // #359: batch version of a row's "Make Expandable" — every
                // preset currently label-only for `no_model` (declares no
                // `model` the engine can expand), not just the 26 this
                // shipped against.
                Button("Make All Expandable (\(backfillCandidateCount))…") {
                    Task { await runBackfillAll() }
                }
                .disabled(backfillCandidateCount == 0 || backfillViewModel.isRunning)
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .disabled(!engine.connectionState.isConnected)
            .help("Import presets, or fill in the model spec on label-only presets so /v1/generate can expand them")
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
                        isExpandable: PresetBackfillViewModel.isExpandable(preset),
                        isBackfillable: PresetBackfillViewModel.isBackfillable(preset),
                        onApply: { onApply?(preset.toGenerationPreset()) },
                        onEdit: { Task { await beginEditing(preset, asNew: false) } },
                        onDuplicate: { Task { await duplicate(preset) } },
                        onDelete: { Task { await delete(preset) } },
                        onSetWarm: presetModelSpec(preset) != nil ? { Task { await setAsWarm(preset) } } : nil,
                        onMakeExpandable: { Task { await makeExpandable(preset) } }
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
        preset.effectiveModelSpec
    }

    // MARK: - #359: checkpoint_family backfill

    private var backfillCandidateCount: Int {
        PresetBackfillViewModel.backfillCandidates(presets).count
    }

    /// One row's "Make Expandable".
    private func makeExpandable(_ preset: ServerPreset) async {
        let outcome = await backfillViewModel.backfill(
            preset,
            detectFamily: { spec in await engine.fetchModelFamily(forSpec: spec) },
            save: { updated in try await engine.savePreset(updated) }
        )
        switch outcome.status {
        case .failed(let message):
            loadError = "Could not make '\(preset.name)' expandable: \(message)"
        case .updated(_, nil, let note):
            // Expandable now; only the checkpoint_family label is outstanding.
            loadError = note.map { "'\(preset.name)' is expandable — \($0)" }
        case .updated, .skipped:
            loadError = nil
        }
        await reload()
    }

    /// The header menu's "Make All Expandable (N)…" — every candidate in the
    /// current list, reported per-preset.
    private func runBackfillAll() async {
        let results = await backfillViewModel.backfillAll(
            presets,
            detectFamily: { spec in await engine.fetchModelFamily(forSpec: spec) },
            save: { updated in try await engine.savePreset(updated) }
        )
        backfillResults = BackfillResultsBox(items: results)
        await reload()
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
    /// #359: would `/v1/generate {"preset": id}` expand this preset's model +
    /// LoRA stack, or does it stay a provenance label? Same decision the
    /// editor's Effective-recipe panel runs (`PresetLoRAStack.decide`).
    var isExpandable: Bool = true
    /// Label-only specifically for a model-family reason ("Make Expandable"
    /// can plausibly fix it) — other unresolved reasons (video, bypass,
    /// unknown engine/provider) are not shown the action.
    var isBackfillable: Bool = false
    var onApply: () -> Void
    var onEdit: () -> Void
    var onDuplicate: () -> Void
    var onDelete: () -> Void
    var onSetWarm: (() -> Void)?
    var onMakeExpandable: (() -> Void)?

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
                    expandableBadge
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
                if let onMakeExpandable, isBackfillable {
                    Button("Make Expandable", action: onMakeExpandable)
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

    /// #359: "Expandable" (green) when `/v1/generate {"preset": id}` would
    /// apply this preset's model + LoRA stack as a whole; "Label only" (the
    /// state all 26 desktop-saved presets shipped in) when the engine has
    /// nothing to expand it with and it renders on whatever's already
    /// resident — same signal as the editor's Effective-recipe panel.
    @ViewBuilder
    private var expandableBadge: some View {
        if isExpandable {
            Label("Expandable", systemImage: "checkmark.seal.fill")
                .font(.caption2)
                .foregroundStyle(.green)
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(.green.opacity(0.12), in: Capsule())
                .help("POST /v1/generate {\"preset\": \"\(preset.id)\"} applies this preset's model + LoRA stack.")
        } else {
            Label("Label only", systemImage: "tag")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(.quaternary, in: Capsule())
                .help(isBackfillable
                    ? "This preset names no model the engine can expand, so /v1/generate renders it on whatever is already resident — use \u{201C}Make Expandable\u{201D}."
                    : "This preset will not expand on POST /v1/generate {preset} — see the editor's Effective recipe panel for why.")
        }
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
    /// #359: `GET /v1/model/family`'s answer for the CURRENT `model` field —
    /// refreshed by `.task(id: model)` below, so it tracks edits without
    /// blocking `buildPreset()` (a plain, synchronous function) on a network
    /// call. nil while unresolved or when the field is empty.
    @State private var detectedModelFamily: ModelFamilyInfo?

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
    // #419: the rest of the recipe — one value type (`PresetSamplingEditorState`)
    // seeded from `original` and written back verbatim by `buildPreset()`,
    // so the seed → edit → write cycle is unit-testable without a view.
    // Neutral sentinels mean "model default" and are written as nil.
    @State private var sampling: PresetSamplingEditorState
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
        _sampling = State(initialValue: PresetSamplingEditorState(original: original))
    }

    /// The family the sampling gates key on (#419, review B2): the ENGINE's
    /// answer for the current `model` text (`GET /v1/model/family`, kept
    /// fresh by `.task(id: model)`) and nothing else. A family inferred from
    /// the text can be WRONG — "/Models/zeta-chroma.safetensors" reads as
    /// chroma but is Z-Image-based (#154) — and a wrong guess would grey and
    /// block what the engine actually accepts. nil (pending, offline, or the
    /// engine has no answer) is permissive: loaded values stay untouched,
    /// the option lists show the union, and the engine validates on save.
    private var samplingModelFamily: String? {
        guard let family = detectedModelFamily?.family, !family.isEmpty else { return nil }
        return family
    }

    /// #419: the whole recipe as the editor currently holds it, with the
    /// neutral sentinels already collapsed to nil. `buildPreset()` writes
    /// exactly this; `samplingValidationError` validates exactly this.
    private var samplingDraft: PresetSamplingDraft {
        sampling.draft(modelFamily: samplingModelFamily, sampler: sampler, sigmaSchedule: sigmaSchedule)
    }

    /// nil = the engine would accept this recipe on the preset's family.
    /// Non-nil blocks Save (and Save as New) with the engine's own wording.
    /// This — not a silent reset — is what happens to a loaded value a
    /// closed gate refuses: the control is greyed, Clear sits beside it, and
    /// Save waits for the user to decide.
    private var samplingValidationError: String? {
        PresetSamplingValidator.validationError(samplingDraft)
    }

    private var stage2EtaStatus: SamplingGate.Status {
        SamplingGate.stage2Eta(
            modelFamily: samplingModelFamily, stage2Sampler: sampling.stage2Sampler, stage1Sampler: sampler)
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
                }
                Section("Sampling") {
                    samplingSection
                }
                Section("Detail pass (stage 2)") {
                    stage2Section
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
        .frame(minWidth: 560, idealWidth: 640, minHeight: 720, idealHeight: 860)
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
        .task(id: model) {
            // #359: re-resolve whenever the model field changes — `.task(id:)`
            // cancels the previous in-flight lookup, so fast typing does not
            // pile up requests. Runs even for a brand-new preset (`isNew`):
            // the "engine-known alias" / `checkpoint_family` decision belongs
            // to `buildPreset()` on every save, not just an edit of one that
            // already exists.
            //
            // Round 3: drop the previous answer BEFORE awaiting. It described
            // the path the field used to hold, and a Save landing in this
            // window would otherwise pair the new `custom_model_path` with the
            // old `model`. `PresetModelFieldBuilder` also checks
            // `ModelFamilyInfo.answers(_:)` — belt and braces, because this
            // clear alone cannot cover an answer that arrives for a spec the
            // user has already typed past.
            detectedModelFamily = nil
            guard let engine else { return }
            let spec = model.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !spec.isEmpty else { return }
            let answer = await engine.fetchModelFamily(forSpec: spec)
            // Only adopt an answer that is still about what the field holds.
            guard spec == model.trimmingCharacters(in: .whitespacesAndNewlines) else { return }
            detectedModelFamily = answer
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

    // MARK: - Sampling (#419)

    /// Sampler + schedule (the existing family-aware picker), then the shared
    /// RES4LYF knobs — the SAME `SamplingAdvancedControls` the Generate panel
    /// shows, so the eta/bongmath gate exists once — plus the Krea 2 VAE
    /// override. A refused value is greyed with Clear beside it and blocks
    /// Save (`samplingValidationError`); nothing is reset except on the
    /// user's own sampler change.
    @ViewBuilder
    private var samplingSection: some View {
        SamplingRecipePicker(
            sampler: $sampler,
            sigmaSchedule: $sigmaSchedule,
            modelFamily: samplingModelFamily,
            showsExplanation: true
        )
        .onChange(of: sampler) { _, newSampler in
            // User-initiated (the picker is the only writer): the one reset.
            sampling.samplerDidChange(to: newSampler, modelFamily: samplingModelFamily)
        }
        SamplingAdvancedControls(
            shift: $sampling.shift,
            projectorScale: $sampling.projectorScale,
            eta: $sampling.eta,
            bongmath: $sampling.bongmath,
            noiseType: $sampling.noiseType,
            noiseAlpha: $sampling.noiseAlpha,
            implicitSteps: $sampling.implicitSteps,
            c2: $sampling.c2,
            sampler: sampler,
            sigmaSchedule: sigmaSchedule,
            modelFamily: samplingModelFamily
        )
        vaeRow
        // The pair error is the picker's own to show (above); this label
        // covers every OTHER rule so a refusal is never reported twice.
        if let error = samplingValidationError, pairIsSupported {
            Label(error, systemImage: "exclamationmark.triangle.fill")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    /// Review B3: rendered whenever the family honours a VAE override OR the
    /// preset already carries one — a value the family refuses is greyed
    /// with Clear, never hidden while it blocks Save.
    @ViewBuilder
    private var vaeRow: some View {
        let status = SamplingGate.vae(modelFamily: samplingModelFamily)
        let hasValue = !sampling.vae.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if status.isHonoured || hasValue {
            HStack(spacing: 6) {
                TextField("VAE (path; empty = model directory's VAE)", text: $sampling.vae)
                    .disabled(status.isRefused)
                    .help("Krea 2 decode VAE override — e.g. the Wan 2.1 VAE. Empty = the model directory's own VAE.")
                if status.isRefused, hasValue {
                    Button("Clear VAE") { sampling.vae = "" }.controlSize(.small)
                }
            }
            if status.isRefused, hasValue, let note = status.note {
                Label(note, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    private var pairIsSupported: Bool {
        SamplingRecipeCatalog.supports(
            sampler: sampler.isEmpty ? nil : sampler,
            sigmaSchedule: sigmaSchedule.isEmpty ? nil : sigmaSchedule,
            forModelFamily: samplingModelFamily)
    }

    /// The optional second stage (WP-E17, Krea 2 only): re-noises the latent
    /// to the stretched tail and solves again. Needs steps AND denoise; its
    /// sampler/schedule default to the render's; `stage2.eta` follows the
    /// RES4LYF rule on whichever sampler the stage actually runs, with
    /// `stage2.eta ?? eta` as the effective value (the engine's rule).
    /// `stage2.bongmath` has no control — the engine 400s it as
    /// unimplemented — but a stored value is shown with Clear, not dropped.
    @ViewBuilder
    private var stage2Section: some View {
        let status = SamplingGate.stage2(modelFamily: samplingModelFamily)
        Toggle("Run a detail pass after the main render", isOn: $sampling.stage2Enabled)
            // Turning OFF is always possible (it is the Clear for a refused
            // stage); turning ON is what the family gate withholds.
            .disabled(status.isRefused && !sampling.stage2Enabled)
        if status.isRefused, let note = status.note {
            if sampling.stage2Enabled {
                Label(note, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2).foregroundStyle(.orange)
            } else {
                Text(note).font(.caption2).foregroundStyle(.tertiary)
            }
        }
        if sampling.stage2Enabled {
            Group {
                HStack {
                    TextField("Steps", text: $sampling.stage2StepsText).frame(width: 90)
                    TextField("Denoise (0–1]", text: $sampling.stage2DenoiseText).frame(width: 110)
                    Spacer()
                }
                Text("Steps and denoise are both required. Denoise is the fraction of the schedule the stage re-runs (e.g. 0.2).")
                    .font(.caption2).foregroundStyle(.tertiary)
                SamplingRecipePicker(
                    sampler: $sampling.stage2Sampler,
                    sigmaSchedule: $sampling.stage2SigmaSchedule,
                    modelFamily: samplingModelFamily,
                    showsExplanation: false
                )
                .onChange(of: sampling.stage2Sampler) { _, newSampler in
                    sampling.stage2SamplerDidChange(
                        to: newSampler, stage1Sampler: sampler, modelFamily: samplingModelFamily)
                }
                Text("Model Default here means the main render's sampler / scheduler; an empty eta inherits the main render's eta.")
                    .font(.caption2).foregroundStyle(.tertiary)
                NumericSliderField(label: "Eta (SDE)", value: $sampling.stage2Eta, range: 0...1, step: 0.05, fractionDigits: 2)
                    .disabled(stage2EtaStatus.isRefused)
                if let note = stage2EtaStatus.note {
                    HStack(alignment: .top, spacing: 6) {
                        if stage2EtaStatus.isRefused, (sampling.stage2Eta != 0 || sampling.eta != 0) {
                            Label(note, systemImage: "exclamationmark.triangle.fill")
                                .font(.caption2).foregroundStyle(.orange)
                            Spacer(minLength: 0)
                            if sampling.stage2Eta != 0 {
                                Button("Clear stage 2 eta") { sampling.stage2Eta = 0 }.controlSize(.small)
                            }
                        } else {
                            Text(note).font(.caption2).foregroundStyle(.tertiary)
                        }
                    }
                }
                if sampling.stage2Bongmath == true {
                    HStack(alignment: .top, spacing: 6) {
                        Label("This preset declares stage2.bongmath — the engine refuses it (parity tier T3, not implemented yet).",
                              systemImage: "exclamationmark.triangle.fill")
                            .font(.caption2).foregroundStyle(.orange)
                        Spacer(minLength: 0)
                        Button("Clear") { sampling.stage2Bongmath = nil }.controlSize(.small)
                    }
                }
            }
            .disabled(status.isRefused)
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
                // #419: the rest of the recipe `ResolvedPreset` carries.
                if let shift = recipe.shift {
                    LabeledContent(SamplingRecipeCatalog.shiftLabel(forModelFamily: samplingModelFamily),
                                   value: String(format: "%g", shift))
                }
                if let eta = recipe.eta, eta != 0 {
                    LabeledContent("Eta (SDE)", value: String(format: "%g", eta))
                }
                if recipe.bongmath == true {
                    LabeledContent("Bongmath", value: "on")
                }
                if let stage2 = recipe.stage2 {
                    LabeledContent("Detail pass", value: PresetEffectiveRecipePresenter.stage2Summary(stage2))
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
        let loras = editableLoras
            .filter { !$0.filename.isEmpty }
            .map { ServerPresetLora(filename: $0.filename, scale: $0.scale, role: $0.role) }
        // #359: `model` — the field `PresetLoRAStack.decide` reads FIRST —
        // plus `checkpoint_family`, so `/v1/generate {preset}` can expand
        // this preset instead of leaving it a `no_model` label. For a typed
        // PATH the canonical engine spec goes in `model` and the path itself
        // stays in `custom_model_path`; see `PresetModelFieldBuilder`.
        // `detectedModelFamily` is the engine's answer (`GET
        // /v1/model/family`) for the CURRENT `model` text, kept fresh by
        // `.task(id: model)`. On a MISS nothing is erased: `model` keeps its
        // existing value as long as the typed path is unchanged (round 2
        // ruling 1 — writing nil here reverted a just-backfilled preset to
        // `no_model`), and `checkpoint_family` falls back to whatever this
        // preset already declared.
        let modelFields = PresetModelFieldBuilder.build(
            modelText: model,
            loras: loras,
            detection: detectedModelFamily,
            fallbackModel: original.model,
            fallbackCustomModelPath: original.customModelPath,
            fallbackCheckpointFamily: original.checkpointFamily
        )
        p.model = modelFields.model
        p.customModelPath = modelFields.customModelPath
        p.checkpointFamily = modelFields.checkpointFamily
        p.width = Int(widthText)
        p.height = Int(heightText)
        p.steps = Int(stepsText)
        p.guidance = Double(guidanceText.replacingOccurrences(of: ",", with: "."))
        p.sampler = sampler.isEmpty ? nil : sampler
        p.sigmaSchedule = sigmaSchedule.isEmpty ? nil : sigmaSchedule
        // Keep the legacy sampler spelling synchronized for older preset
        // consumers; modern engine validation and Generate use `sampler`.
        p.scheduler = p.sampler
        // #419: the rest of the recipe — what the user sees is what is
        // saved, no family involved (neutral sentinels are written as nil;
        // a stored `stage2.bongmath` is preserved until the user clears it).
        sampling.write(into: &p)
        // Todd 2026-09-04: kroma is a regular LoRA — `loras[]` (editableLoras)
        // is the single source. Review r2, C1 (Critical): `p.kroma` is a
        // DEPRECATED, derived, read-only echo — carrying `original.kroma`
        // through unedited (as `bypass`/`upscale` legitimately do) resurrects
        // a row the user just deleted, because the server's compatibility
        // shim folds a non-nil `kroma` back into `loras[]` on the next save.
        // The desktop must NEVER send it; `ServerPreset.encode` also never
        // emits it, belt and braces.
        p.kroma = nil
        p.loras = loras
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
    /// #419: the RES4LYF knobs the Generate panel already holds — shown so
    /// the user sees they are part of what gets saved. 0 / false / nil =
    /// model default (not shown).
    var eta: Double = 0
    var bongmath: Bool = false
    var shift: Double? = nil
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
                    if let shift {
                        LabeledContent("Shift", value: String(format: "%g", shift))
                    }
                    if eta != 0 {
                        LabeledContent("Eta (SDE)", value: String(format: "%g", eta))
                    }
                    if bongmath {
                        LabeledContent("Bongmath", value: "on")
                    }
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
        .frame(width: 420, height: 500)
        .onAppear {
            if !didSeedNegative {
                editedNegative = negativePrompt
                didSeedNegative = true
            }
        }
    }
}

// MARK: - #359: backfill results

/// One-shot summary sheet for "Make Expandable" (a single-item list) and
/// "Make All Expandable (N)…" (the whole batch) — per-preset
/// Updated / Updated (label pending) / Failed, so a batch run is auditable
/// rather than a single toast. "Label pending" means the preset IS expandable
/// now — `model` was written — and only the `checkpoint_family` label is
/// outstanding, because it turns on an accelerator LoRA whose role nobody has
/// declared (or on a variant the engine refuses to guess).
private struct BackfillResultsView: View {
    let results: [PresetBackfillViewModel.Outcome]
    let onDismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Make Presets Expandable").font(.headline)
                Spacer()
                Button("Done", action: onDismiss).keyboardShortcut(.defaultAction)
            }
            .padding()
            Divider()
            if results.isEmpty {
                Text("No label-only presets needed a model backfill.")
                    .font(.subheadline).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(results) { outcome in
                    HStack(alignment: .top, spacing: 8) {
                        icon(for: outcome.status)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(outcome.name).font(.subheadline)
                            Text(detail(for: outcome.status))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .frame(minWidth: 420, idealWidth: 460, minHeight: 320, idealHeight: 420)
    }

    @ViewBuilder
    private func icon(for status: PresetBackfillViewModel.Outcome.Status) -> some View {
        switch status {
        case .updated(_, .some, _):
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .updated(_, nil, _):
            // Written and expandable — the checkpoint_family LABEL is still
            // outstanding (an accelerator LoRA with no role, or a variant the
            // engine will not guess). Not a failure.
            Image(systemName: "checkmark.circle").foregroundStyle(.orange)
        case .skipped:
            Image(systemName: "minus.circle").foregroundStyle(.secondary)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        }
    }

    private func detail(for status: PresetBackfillViewModel.Outcome.Status) -> String {
        switch status {
        case .updated(let model, let label, let note):
            let head = "model set to \"\(model)\" — now expands"
            if let label { return head + ", checkpoint_family \"\(label)\"" }
            return note.map { "\(head). \($0)" } ?? head
        case .skipped(let reason): return reason
        case .failed(let reason): return reason
        }
    }
}
