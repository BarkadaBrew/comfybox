// ModelSelector.swift — Model pool management and selection
//
// Displays available models from the server registry and the active model pool.
// Allows loading, activating, and unloading models. Shows current active model
// prominently with pool VRAM usage.

import SwiftUI

struct ModelSelector: View {
    @Bindable var engine: EngineService

    @State private var selectedRegistryModel: String?
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Active model header
            activeModelHeader

            // Loaded models (pool)
            if !engine.poolModels.isEmpty {
                poolSection
            }

            // Available models from registry
            if !engine.availableModels.isEmpty {
                registrySection
            }

            // Error display
            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal, 4)
            }
        }
    }

    // MARK: - Active Model

    private var activeModelHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Active Model")
                .font(.subheadline)
                .fontWeight(.semibold)

            if let model = engine.currentModel {
                HStack(spacing: 8) {
                    Circle()
                        .fill(.green)
                        .frame(width: 8, height: 8)
                    Text(model)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let family = engine.currentModelFamily {
                        Text(family)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                }
            } else {
                HStack(spacing: 8) {
                    Circle()
                        .fill(.gray)
                        .frame(width: 8, height: 8)
                    Text("No model loaded")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Pool Section

    private var poolSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Model Pool")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                Button(action: { Task { await engine.refreshPool() } }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Refresh pool status")
            }

            ForEach(engine.poolModels) { poolModel in
                poolModelRow(poolModel)
            }
        }
    }

    private func poolModelRow(_ model: PoolModelInfo) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(model.active ? .green : .orange)
                .frame(width: 6, height: 6)

            VStack(alignment: .leading, spacing: 1) {
                Text(model.model)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: 6) {
                    Text(model.family)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("\(model.vramMB) MB")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            if model.active {
                Text("Active")
                    .font(.caption2)
                    .foregroundStyle(.green)
            } else {
                HStack(spacing: 4) {
                    Button("Activate") {
                        Task { await activatePoolModel(model.id) }
                    }
                    .font(.caption2)
                    .buttonStyle(.bordered)
                    .controlSize(.mini)

                    Button(action: { Task { await unloadPoolModel(model.id) } }) {
                        Image(systemName: "xmark.circle")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                    .help("Unload model")
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(model.active ? Color.green.opacity(0.05) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Registry Section

    private var registrySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Available Models")
                .font(.subheadline)
                .fontWeight(.semibold)

            // Group by family
            let families = Dictionary(grouping: engine.availableModels) { $0.family }
            let sortedFamilies = families.keys.sorted()

            ForEach(sortedFamilies, id: \.self) { family in
                if let models = families[family] {
                    DisclosureGroup(family) {
                        ForEach(models) { model in
                            registryModelRow(model)
                        }
                    }
                    .font(.caption)
                }
            }
        }
    }

    private func registryModelRow(_ model: ModelInfo) -> some View {
        let isLoaded = engine.poolModels.contains { $0.model == model.huggingFaceId }

        return HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(.caption)
                HStack(spacing: 6) {
                    Text("\(String(format: "%.1f", model.parametersBillions))B")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(model.quantization)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("\(String(format: "%.1f", model.estimatedVRAM_GB)) GB VRAM")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            if isLoaded {
                Text("Loaded")
                    .font(.caption2)
                    .foregroundStyle(.green)
            } else {
                Button("Load") {
                    Task { await loadRegistryModel(model) }
                }
                .font(.caption2)
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(engine.isLoadingModel)
            }
        }
        .padding(.vertical, 2)
        .help(model.description)
    }

    // MARK: - Actions

    private func loadRegistryModel(_ model: ModelInfo) async {
        errorMessage = nil
        do {
            try await engine.loadModel(
                id: model.huggingFaceId,
                quantization: model.quantization == "bf16" ? nil : model.quantization,
                activate: true
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func activatePoolModel(_ id: String) async {
        errorMessage = nil
        do {
            try await engine.activateModel(id: id)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func unloadPoolModel(_ id: String) async {
        errorMessage = nil
        do {
            try await engine.unloadModel(id: id)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
