// DashboardView.swift — At-a-glance status: connection, model, queue, live
// progress, and recent renders. Reads EngineService state (polled every 3s)
// and the DAM for recent assets. Ports the Electron Dashboard view (parity).

import SwiftUI

struct DashboardView: View {
    @Bindable var engine: EngineService
    let store: DAMStore?
    let ingestor: AssetIngestor?

    @State private var recent: [DAMAsset] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                statTiles
                if let q = engine.queueInfo, q.isRendering, let pct = q.progressPercent {
                    renderProgress(pct)
                }
                recentRenders
            }
            .padding(20)
        }
        .navigationTitle("Dashboard")
        .task(id: ingestor?.ingestedCount) { await loadRecent() }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(engine.connectionState.isConnected ? Color.green : Color.gray)
                .frame(width: 10, height: 10)
            Text(engine.connectionState.isConnected ? "Connected" : "Disconnected")
                .font(.headline)
            Spacer()
            if let model = engine.currentModel {
                Label(model, systemImage: "cube")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }

    // MARK: - Stat tiles

    private var statTiles: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
            let q = engine.queueInfo
            StatTile(title: "Status",
                     value: (q?.isRendering ?? false) ? "Rendering" : "Idle",
                     systemImage: "bolt.fill",
                     tint: (q?.isRendering ?? false) ? .orange : .green)
            StatTile(title: "Pending", value: "\(q?.pendingCount ?? engine.queueCount)", systemImage: "tray.full")
            StatTile(title: "Renders", value: "\(q?.renderCount ?? 0)", systemImage: "photo.stack")
            StatTile(title: "Uptime", value: formatUptime(q?.uptimeSeconds ?? 0), systemImage: "clock")
            StatTile(title: "Memory", value: formatMemory(q?.memoryUsageMB ?? 0), systemImage: "memorychip")
            if let ms = q?.lastRenderDurationMs {
                StatTile(title: "Last render", value: String(format: "%.1fs", Double(ms) / 1000), systemImage: "timer")
            }
        }
    }

    private func renderProgress(_ pct: Double) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Current render")
                    .font(.subheadline).bold()
                Spacer()
                Text("\(Int(pct))%")   // server reports progress_percent on a 0–100 scale
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: min(max(pct / 100, 0), 1))
        }
        .padding(12)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Recent renders

    private var recentRenders: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent renders")
                .font(.headline)
            if recent.isEmpty {
                Text("No renders yet.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(recent) { asset in
                            DashboardThumbnail(
                                path: ingestor?.thumbnailPath(for: asset.id) ?? asset.absolutePath,
                                caption: asset.prompt
                            )
                        }
                    }
                }
            }
        }
    }

    // MARK: - Data

    private func loadRecent() async {
        guard let store else { return }
        recent = (try? await store.fetchAssets(limit: 12)) ?? []
    }

    // MARK: - Formatting

    private func formatUptime(_ seconds: Int) -> String {
        if seconds <= 0 { return "—" }
        let h = seconds / 3600, m = (seconds % 3600) / 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m" }
        return "\(seconds)s"
    }

    private func formatMemory(_ mb: UInt64) -> String {
        if mb == 0 { return "—" }
        if mb >= 1024 { return String(format: "%.1f GB", Double(mb) / 1024) }
        return "\(mb) MB"
    }
}

// MARK: - Reusable tile

struct StatTile: View {
    let title: String
    let value: String
    var systemImage: String = "circle"
    var tint: Color = .accentColor

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: systemImage)
                .font(.caption)
                .foregroundStyle(.secondary)
                .labelStyle(.titleAndIcon)
            Text(value)
                .font(.title3).bold()
                .foregroundStyle(tint)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
        .padding(12)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Thumbnail

private struct DashboardThumbnail: View {
    let path: String
    let caption: String?

    @State private var image: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Group {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    RoundedRectangle(cornerRadius: 8).fill(.quaternary)
                }
            }
            .frame(width: 120, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .frame(width: 120)
        .help(caption ?? "")
        .task {
            let p = path
            image = await Task.detached { NSImage(contentsOfFile: p) }.value
        }
    }
}
