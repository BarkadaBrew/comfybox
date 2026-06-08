// GenerationView.swift — Main image generation interface
//
// Provides prompt entry, parameter controls, generation button,
// and image preview. Communicates with the WarmServer through
// EngineService. On successful generation, calls onGenerated
// to trigger DAM ingestion.

import SwiftUI

/// Common resolution presets.
struct ResolutionPreset: Identifiable, Hashable {
    let id: String
    let width: Int
    let height: Int
    var label: String { "\(width) x \(height)" }

    static let presets: [ResolutionPreset] = [
        ResolutionPreset(id: "512sq", width: 512, height: 512),
        ResolutionPreset(id: "768p", width: 768, height: 1024),
        ResolutionPreset(id: "1024sq", width: 1024, height: 1024),
        ResolutionPreset(id: "1024l", width: 1024, height: 768),
    ]
}

struct GenerationView: View {
    @Bindable var engine: EngineService
    var onGenerated: ((String, GenerationRequest) -> Void)?

    // Generation parameters
    @State private var prompt: String = ""
    @State private var selectedResolution: ResolutionPreset = ResolutionPreset.presets[2]
    @State private var steps: Double = 9
    @State private var guidance: Double = 3.5
    @State private var seedText: String = ""
    @State private var displayedImage: NSImage?

    var body: some View {
        HSplitView {
            // Left panel: Controls
            controlPanel
                .frame(minWidth: 320, maxWidth: 400)

            // Right panel: Image preview
            previewPanel
                .frame(minWidth: 400)
        }
    }

    // MARK: - Control Panel

    private var controlPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Server status bar
                serverStatusBar

                Divider()

                // Prompt
                promptSection

                Divider()

                // Parameters
                parameterSection

                Divider()

                // Generate button
                generateButton

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
            Text("Prompt")
                .font(.headline)

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
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            // Steps slider
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Steps")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(steps))")
                        .font(.subheadline)
                        .monospacedDigit()
                }

                Slider(value: $steps, in: 1...50, step: 1)
            }

            // Guidance slider
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Guidance")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.1f", guidance))
                        .font(.subheadline)
                        .monospacedDigit()
                }

                Slider(value: $guidance, in: 0...20, step: 0.5)
            }

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

    private var generateButton: some View {
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
    }

    private var canGenerate: Bool {
        engine.connectionState.isConnected
            && !engine.isGenerating
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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
            width: selectedResolution.width,
            height: selectedResolution.height,
            steps: Int(steps),
            guidance: Float(guidance),
            seed: seed
        )

        Task {
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
                // Error is already stored in engine.lastError
            }
        }
    }
}
