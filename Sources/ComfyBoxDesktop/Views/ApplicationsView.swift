// ApplicationsView.swift — Surfaces the Coffeeshop suite is exposed on
//
// The hub view for external integrations: which apps/clients consume ComfyBox,
// their live status, endpoints, and what to fix when something's off.

import SwiftUI
import AppKit

struct ApplicationsView: View {
    @Bindable var engine: EngineService
    @State private var surfaces = SurfacesService()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                ForEach(surfaces.surfaces) { surface in
                    surfaceCard(surface)
                }
            }
            .padding(20)
        }
        .navigationTitle("Applications")
        .onAppear { reload() }
        .onChange(of: engine.connectionState.isConnected) { _, _ in reload() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Surfaces").font(.title2.bold())
            Text("Where the Coffeeshop suite is exposed. ComfyBox is the hub; these are the clients that consume it.")
                .font(.callout).foregroundStyle(.secondary)
            Button { reload() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                .controlSize(.small).padding(.top, 2)
        }
    }

    private func surfaceCard(_ s: Surface) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Circle().fill(color(s.health)).frame(width: 10, height: 10).padding(.top, 5)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(s.name).font(.headline)
                    Text(s.health.rawValue.uppercased()).font(.caption2.bold())
                        .foregroundStyle(color(s.health))
                }
                Text(s.detail).font(.callout).foregroundStyle(.secondary)
                if let ep = s.endpoint {
                    HStack(spacing: 6) {
                        Image(systemName: "link").font(.caption2).foregroundStyle(.tertiary)
                        Text(ep).font(.caption.monospaced()).textSelection(.enabled)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(ep, forType: .string)
                        } label: { Image(systemName: "doc.on.doc") }
                            .buttonStyle(.borderless).controlSize(.small).help("Copy")
                    }
                }
                if let hint = s.hint {
                    Label(hint, systemImage: "wrench.and.screwdriver")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
        }
        .padding(14)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
    }

    private func color(_ h: Surface.Health) -> Color {
        switch h {
        case .ok: return .green
        case .degraded: return .orange
        case .off: return .red
        case .unknown: return .gray
        }
    }

    private func reload() {
        surfaces.serverPort = Int(engine.serverPort)
        surfaces.refresh(bridgeReachable: engine.connectionState.isConnected)
    }
}
