// ServerView.swift — Server status: connection, active model, and queue.
// Model / LoRA / storage management lives in the Models tab.

import SwiftUI

struct ServerView: View {
    @Bindable var engine: EngineService

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                statusSection
                activeModelSection
            }
            .padding(20)
        }
        .navigationTitle("Server")
        .task { await engine.refreshPool() }
    }

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
            Text("\(engine.serverHost):\(engine.serverPort)")
                .font(.caption).foregroundStyle(.secondary)

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
        }
    }

    private var activeModelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Active Model").font(.headline)
            if let active = engine.poolModels.first(where: { $0.active }) {
                HStack(spacing: 10) {
                    Circle().fill(Color.green).frame(width: 8, height: 8)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(active.model).lineLimit(1).truncationMode(.middle)
                        Text("\(active.family) · \(active.vramMB >= 1024 ? String(format: "%.1f GB", Double(active.vramMB) / 1024) : "\(active.vramMB) MB")")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .padding(10)
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
            } else {
                Text(engine.connectionState.isConnected ? "No model active." : "Connect to see the active model.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Text("Load, activate, and manage models & LoRAs in the Models tab.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }
}
