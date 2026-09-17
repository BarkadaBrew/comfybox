// DirectorSidebar.swift — the Director tab's settings column (WP3): preset,
// resolution, length, seed, LoRAs, audio mode, global prompt + enhance,
// speech readout, effective config, validation issues and the document /
// render buttons. Binds through DirectorDocumentModel; no timeline rules here.

import SwiftUI
import ZImage

/// Director output sizes (multiples of 32). Local on purpose — Motion's
/// VideoResolution is left untouched.
enum DirectorResolution: String, CaseIterable, Identifiable {
    case portrait = "Portrait 576×896"
    case landscape = "Landscape 896×576"
    case wide = "Landscape 704×448"
    case tall = "Portrait 448×704"
    case square = "Square 512×512"

    var id: String { rawValue }

    var size: (width: Int, height: Int) {
        switch self {
        case .portrait: return (576, 896)
        case .landscape: return (896, 576)
        case .wide: return (704, 448)
        case .tall: return (448, 704)
        case .square: return (512, 512)
        }
    }

    static func matching(width: Int, height: Int) -> DirectorResolution? {
        allCases.first { $0.size == (width, height) }
    }
}

struct DirectorSidebar: View {
    @Bindable var engine: EngineService
    let model: DirectorDocumentModel
    let isGenerating: Bool
    let isValidating: Bool
    let onNew: () -> Void
    let onOpen: () -> Void
    let onSave: () -> Void
    let onSaveAs: () -> Void
    let onValidate: () -> Void
    let onGenerate: () -> Void

    @State private var presets: [ServerPreset] = []
    @State private var pendingSeconds: Double?
    @State private var seedText: String = ""
    @State private var globalAttemptId: String?
    @State private var showAdvanced = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                documentButtons

                labeled("Preset") {
                    Picker("Preset", selection: Binding(
                        get: { model.timeline.settings.preset ?? "" },
                        set: { value in model.updateSettings { $0.preset = value.isEmpty ? nil : value } })
                    ) {
                        Text("None (server defaults)").tag("")
                        ForEach(presets, id: \.id) { preset in
                            Text(preset.name).tag(preset.id)
                        }
                        if let current = model.timeline.settings.preset, !presets.contains(where: { $0.id == current }) {
                            Text(current).tag(current)
                        }
                    }
                    .labelsHidden()
                }

                labeled("Resolution") {
                    Picker("Resolution", selection: Binding(
                        get: { DirectorResolution.matching(width: model.timeline.settings.width, height: model.timeline.settings.height) },
                        set: { value in
                            guard let value else { return }
                            model.updateSettings { $0.width = value.size.width; $0.height = value.size.height }
                        })
                    ) {
                        ForEach(DirectorResolution.allCases) { r in Text(r.rawValue).tag(Optional(r)) }
                        if DirectorResolution.matching(width: model.timeline.settings.width, height: model.timeline.settings.height) == nil {
                            Text("Custom \(model.timeline.settings.width)×\(model.timeline.settings.height)").tag(DirectorResolution?.none)
                        }
                    }
                    .labelsHidden()
                }

                labeled("Length") {
                    NumericSliderField(
                        label: "Seconds",
                        value: Binding(
                            get: { pendingSeconds ?? model.timelineSeconds },
                            set: { pendingSeconds = $0 }),
                        range: 4...Double(DirectorMath.maxTimelineFrames / max(1, model.fps)),
                        step: 0.5, fractionDigits: 2,
                        onEditingEnded: commitSeconds)
                    Text(String(format: "%d f = %.2f s · %d chunk(s)", model.lengthFrames, model.timelineSeconds,
                                DirectorMath.chunkLayout(lengthFrames: model.lengthFrames).count))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }

                labeled("Seed") {
                    TextField("Seed", text: $seedText)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(commitSeed)
                        .onChange(of: seedText) { _, _ in commitSeed() }
                    Text("Chunk k renders with seed + k").font(.caption2).foregroundStyle(.tertiary)
                }

                labeled("Audio") {
                    Picker("Audio", selection: Binding(
                        get: { model.timeline.audio.mode == .imported ? DirectorTimeline.AudioMode.imported : .generated },
                        set: { model.setAudioMode($0) })
                    ) {
                        Text("Generated (per chunk)").tag(DirectorTimeline.AudioMode.generated)
                        Text("Imported clips").tag(DirectorTimeline.AudioMode.imported)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    Text("Inpaint (generate around imported audio) — Phase 2")
                        .font(.caption2).foregroundStyle(.tertiary)
                }

                labeled("Global prompt") {
                    TextEditor(text: Binding(get: { model.timeline.globalPrompt }, set: { model.setGlobalPrompt($0) }))
                        .font(.body).scrollContentBackground(.hidden)
                        .frame(minHeight: 90)
                        .padding(6)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor), lineWidth: 1))
                    OptimizeBar(
                        engine: engine,
                        prompt: Binding(get: { model.timeline.globalPrompt }, set: { model.setGlobalPrompt($0) }),
                        optimizationAttemptId: $globalAttemptId)
                    Text(String(format: "speech %.1f s of %.2f s", model.speechSeconds, model.timelineSeconds))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(model.speechSeconds > model.timelineSeconds ? Color.red : Color.secondary)
                        .help("Quoted dialogue at 2.5 words/s across the global prompt and every segment")
                }

                labeled("LoRAs") {
                    LoRAPicker(
                        engine: engine,
                        selectedLoras: Binding(
                            get: { model.timeline.settings.loras.map { LoRASelection(id: $0.path, filename: $0.path, scale: $0.scale, role: $0.role) } },
                            set: { selections in
                                model.updateSettings { $0.loras = selections.map { .init(path: $0.filename, scale: $0.scale, role: $0.role) } }
                            }),
                        familyOverride: "ltx",
                        strictFamilyFilter: true)
                        .frame(minHeight: 180, maxHeight: 260)
                    if model.timeline.settings.loras.isEmpty {
                        Text("None selected — the preset's LoRAs apply").font(.caption2).foregroundStyle(.tertiary)
                    }
                }

                DisclosureGroup("Advanced", isExpanded: $showAdvanced) { advanced }

                EffectiveConfigCard(engine: engine)

                issuesList

                actionButtons
            }
            .padding(14)
        }
        .task {
            seedText = String(model.timeline.settings.seed)
            presets = await engine.fetchPresets().filter { $0.mediaKind == "video" }
        }
        .onChange(of: model.timeline.settings.seed) { _, seed in
            if UInt64(seedText) != seed { seedText = String(seed) }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "timeline.selection").foregroundStyle(.purple)
            Text("LTX-2 Director").font(.headline)
            Spacer()
            if !engine.connectionState.isConnected {
                Text("offline").font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    private var documentButtons: some View {
        HStack(spacing: 6) {
            Button("New", action: onNew)
            Button("Open…", action: onOpen)
            Button("Save", action: onSave)
            Button("Save As…", action: onSaveAs)
        }
        .controlSize(.small)
    }

    private var advanced: some View {
        VStack(alignment: .leading, spacing: 10) {
            Stepper(value: Binding(
                get: { model.timeline.settings.fps },
                set: { value in model.updateSettings { $0.fps = value } }), in: 1...120) {
                Text("FPS \(model.timeline.settings.fps)")
            }
            labeled("Steps (blank = preset default)") {
                TextField("default", text: Binding(
                    get: { model.timeline.settings.steps.map(String.init) ?? "" },
                    set: { text in model.updateSettings(coalesce: "steps") { $0.steps = Int(text) } }))
                    .textFieldStyle(.roundedBorder)
            }
            labeled("Negative prompt (blank = preset)") {
                TextField("preset negative", text: Binding(
                    get: { model.timeline.settings.negativePrompt ?? "" },
                    set: { text in model.updateSettings(coalesce: "negative") { $0.negativePrompt = text.isEmpty ? nil : text } }))
                    .textFieldStyle(.roundedBorder)
            }
            labeled("Character (blank = none, no injection)") {
                TextField("none", text: Binding(
                    get: { model.timeline.settings.character ?? "" },
                    set: { text in model.updateSettings(coalesce: "character") { $0.character = text.isEmpty ? nil : text } }))
                    .textFieldStyle(.roundedBorder)
            }
        }
        .padding(.top, 6)
    }

    private var issuesList: some View {
        labeled("Issues") {
            if model.issues.isEmpty {
                Text(model.plan == nil ? "Not validated yet" : "No issues")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(model.issues.enumerated()), id: \.offset) { _, issue in
                        Button {
                            model.select(issue: issue)
                        } label: {
                            HStack(alignment: .top, spacing: 6) {
                                Image(systemName: issue.severity == .error ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                                    .foregroundStyle(issue.severity == .error ? Color.red : Color.orange)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(issue.code).font(.caption.monospaced())
                                    Text(issue.message).font(.caption2).foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                                Spacer(minLength: 0)
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(issue.ids.isEmpty)
                    }
                }
            }
            if let plan = model.plan {
                Text("Plan: \(plan.chunks.count) chunk(s) · boundaries \(plan.boundaryFrames.map(String.init).joined(separator: ", "))")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    private var actionButtons: some View {
        HStack {
            Button(action: onValidate) {
                HStack {
                    if isValidating { ProgressView().controlSize(.small) }
                    Text("Validate")
                }
            }
            .disabled(isValidating)
            Spacer()
            Button(action: onGenerate) {
                HStack {
                    if isGenerating { ProgressView().controlSize(.small).padding(.trailing, 2) }
                    Text(isGenerating ? "Rendering…" : "Generate")
                }
                .frame(minWidth: 110)
            }
            .buttonStyle(.borderedProminent)
            // Warnings never disable Generate; only errors, offline, or a
            // render already in flight.
            .disabled(!engine.connectionState.isConnected || !model.canGenerate || isGenerating)
        }
    }

    // MARK: - Commits

    private func commitSeconds() {
        guard let seconds = pendingSeconds else { return }
        pendingSeconds = nil
        model.setLengthFrames(DirectorMath.frames(seconds: seconds, fps: model.fps))
    }

    private func commitSeed() {
        guard let seed = UInt64(seedText.trimmingCharacters(in: .whitespaces)) else { return }
        model.updateSettings(coalesce: "seed") { $0.seed = seed }
    }

    private func labeled<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }
}
