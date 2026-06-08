// PresetView.swift — Preset list with CRUD actions
//
// Displays saved generation presets in a list. Supports apply,
// edit, duplicate, and delete actions. The "Save as Preset" flow
// is triggered from GenerationView and routes through PresetManager.

import SwiftUI

struct PresetView: View {
    @Bindable var presetManager: PresetManager
    var onApply: ((GenerationPreset) -> Void)?

    @State private var editingPreset: GenerationPreset?
    @State private var showingEditor: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("Presets")
                    .font(.headline)
                Spacer()
                Text("\(presetManager.presets.count) saved")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            if presetManager.presets.isEmpty {
                emptyState
            } else {
                presetList
            }
        }
        .sheet(isPresented: $showingEditor) {
            if let preset = editingPreset {
                PresetEditorSheet(
                    preset: preset,
                    onSave: { updated in
                        presetManager.update(updated)
                        showingEditor = false
                        editingPreset = nil
                    },
                    onCancel: {
                        showingEditor = false
                        editingPreset = nil
                    }
                )
                .frame(minWidth: 420, minHeight: 400)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "slider.horizontal.below.rectangle")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)
            Text("No presets saved")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text("Use Cmd+S or the Save button to create one.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var presetList: some View {
        List {
            ForEach(presetManager.presets) { preset in
                PresetRow(preset: preset)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        onApply?(preset)
                    }
                    .contextMenu {
                        Button("Apply") { onApply?(preset) }
                        Button("Edit...") {
                            editingPreset = preset
                            showingEditor = true
                        }
                        Button("Duplicate") {
                            _ = presetManager.duplicate(preset)
                        }
                        Divider()
                        Button("Delete", role: .destructive) {
                            presetManager.delete(id: preset.id)
                        }
                    }
            }
        }
        .listStyle(.inset)
    }
}

// MARK: - Preset Row

private struct PresetRow: View {
    let preset: GenerationPreset

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(preset.name)
                    .font(.subheadline.weight(.medium))
                Spacer()
                if let model = preset.modelId {
                    Text(model)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            if !preset.promptTemplate.isEmpty {
                Text(preset.promptTemplate)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.tail)
            }

            HStack(spacing: 8) {
                Label("\(preset.steps) steps", systemImage: "slider.horizontal.3")
                Label(String(format: "%.1f cfg", preset.guidance), systemImage: "tuningfork")
                Label("\(preset.width)x\(preset.height)", systemImage: "aspectratio")
                if !preset.loras.isEmpty {
                    Label("\(preset.loras.count) LoRA", systemImage: "link")
                }
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Preset Editor Sheet

struct PresetEditorSheet: View {
    @State var preset: GenerationPreset
    var onSave: (GenerationPreset) -> Void
    var onCancel: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Title bar
            HStack {
                Text("Edit Preset")
                    .font(.headline)
                Spacer()
            }
            .padding()

            Divider()

            Form {
                Section("Basics") {
                    TextField("Name", text: $preset.name)
                    TextField("Model ID", text: Binding(
                        get: { preset.modelId ?? "" },
                        set: { preset.modelId = $0.isEmpty ? nil : $0 }
                    ))
                }

                Section("Prompt Template") {
                    TextEditor(text: $preset.promptTemplate)
                        .frame(minHeight: 60)
                        .font(.body)
                }

                Section("Parameters") {
                    HStack {
                        Text("Steps")
                        Spacer()
                        TextField("Steps", value: $preset.steps, format: .number)
                            .frame(width: 60)
                            .multilineTextAlignment(.trailing)
                    }
                    HStack {
                        Text("Guidance")
                        Spacer()
                        TextField("Guidance", value: $preset.guidance, format: .number.precision(.fractionLength(1)))
                            .frame(width: 60)
                            .multilineTextAlignment(.trailing)
                    }
                    HStack {
                        Text("Width")
                        Spacer()
                        TextField("Width", value: $preset.width, format: .number)
                            .frame(width: 80)
                            .multilineTextAlignment(.trailing)
                    }
                    HStack {
                        Text("Height")
                        Spacer()
                        TextField("Height", value: $preset.height, format: .number)
                            .frame(width: 80)
                            .multilineTextAlignment(.trailing)
                    }
                }
            }
            .formStyle(.grouped)

            Divider()

            // Action buttons
            HStack {
                Button("Cancel") { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save") { onSave(preset) }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(preset.name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
        }
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
