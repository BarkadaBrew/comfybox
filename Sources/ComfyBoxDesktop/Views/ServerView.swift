// ServerView.swift — Server status and model-pool control: view loaded models,
// activate/unload, and load available models. Ports the Electron Server view.

import SwiftUI

struct ServerView: View {
    @Bindable var engine: EngineService

    @State private var busy: String?          // id of a model with an op in flight
    @State private var actionError: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                statusSection
                loadedPoolSection
                availableModelsSection
            }
            .padding(20)
        }
        .navigationTitle("Server")
        .task { await refreshAll() }
    }

    // MARK: - Status

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle()
                    .fill(engine.connectionState.isConnected ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
                Text(engine.connectionState.label).font(.headline)
                Spacer()
                Button {
                    engine.connectionState.isConnected ? engine.disconnect() : engine.connect()
                } label: {
                    Text(engine.connectionState.isConnected ? "Disconnect" : "Connect")
                }
            }
            if let q = engine.queueInfo {
                HStack(spacing: 12) {
                    StatTile(title: "Status", value: q.isRendering ? "Rendering" : "Idle",
                             systemImage: "bolt.fill", tint: q.isRendering ? .orange : .green)
                    StatTile(title: "Pending", value: "\(q.pendingCount)", systemImage: "tray.full")
                    StatTile(title: "Memory",
                             value: q.memoryUsageMB >= 1024 ? String(format: "%.1f GB", Double(q.memoryUsageMB) / 1024) : "\(q.memoryUsageMB) MB",
                             systemImage: "memorychip")
                }
            }
            if let err = actionError {
                Text(err).font(.caption).foregroundStyle(.red)
            }
        }
    }

    // MARK: - Loaded pool

    private var loadedPoolSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Model Pool").font(.headline)
                Spacer()
                Button { Task { await engine.refreshPool() } } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .disabled(!engine.connectionState.isConnected)
            }

            if engine.poolModels.isEmpty {
                emptyRow("No models loaded.")
            } else {
                ForEach(engine.poolModels) { m in
                    HStack(spacing: 10) {
                        Circle().fill(m.active ? Color.green : Color.secondary.opacity(0.4))
                            .frame(width: 8, height: 8)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.model).lineLimit(1).truncationMode(.middle)
                            Text("\(m.family) · \(m.vramMB >= 1024 ? String(format: "%.1f GB", Double(m.vramMB) / 1024) : "\(m.vramMB) MB")")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if m.active {
                            Text("Active").font(.caption).foregroundStyle(.green)
                        } else {
                            Button("Activate") { Task { await run(m.id) { try await engine.activateModel(id: m.id) } } }
                                .buttonStyle(.bordered)
                        }
                        Button {
                            Task { await run(m.id) { try await engine.unloadModel(id: m.id) } }
                        } label: {
                            Image(systemName: "eject")
                        }
                        .buttonStyle(.borderless)
                        .help("Unload")
                        if busy == m.id { ProgressView().controlSize(.small) }
                    }
                    .padding(10)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    // MARK: - Available models

    private var availableModelsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Available Models").font(.headline)
                Spacer()
                Button { Task { await engine.refreshModels() } } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .disabled(!engine.connectionState.isConnected)
            }

            if engine.availableModels.isEmpty {
                emptyRow("No catalog models reported.")
            } else {
                ForEach(engine.availableModels) { m in
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.displayName).lineLimit(1)
                            Text("\(m.family) · \(m.quantization) · \(String(format: "%.1f", m.estimatedVRAM_GB)) GB")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Load") { Task { await run(m.id) { try await engine.loadModel(id: m.id) } } }
                            .buttonStyle(.bordered)
                            .disabled(!engine.connectionState.isConnected)
                        if busy == m.id { ProgressView().controlSize(.small) }
                    }
                    .padding(10)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    private func emptyRow(_ text: String) -> some View {
        Text(text).font(.callout).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 4)
    }

    // MARK: - Actions

    private func refreshAll() async {
        guard engine.connectionState.isConnected else { return }
        await engine.refreshModels()
        await engine.refreshPool()
    }

    private func run(_ id: String, _ op: @escaping () async throws -> Void) async {
        busy = id
        actionError = nil
        defer { busy = nil }
        do {
            try await op()
            await engine.refreshPool()
        } catch {
            actionError = error.localizedDescription
        }
    }
}
