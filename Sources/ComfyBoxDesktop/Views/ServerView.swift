// ServerView.swift — Server status: connection, active model, and queue.
// Model / LoRA / storage management lives in the Models tab.

import SwiftUI

struct ServerView: View {
    @Bindable var engine: EngineService
    var store: DAMStore?

    enum StatRange: String, CaseIterable, Identifiable {
        case daily = "Daily", weekly = "Weekly", all = "All Time"
        var id: String { rawValue }
        /// Trailing window in days; nil = all history.
        var days: Int? { self == .daily ? 1 : self == .weekly ? 7 : nil }
        var heatmapDays: Int { self == .daily ? 30 : self == .weekly ? 84 : 365 }
    }

    /// Asset kinds to break out. "voice" appears once voice memos are tracked.
    enum AssetType: String, CaseIterable, Identifiable {
        case all = "All", image = "Images", video = "Videos", voice = "Voice Memos"
        var id: String { rawValue }
        var kind: String? {
            switch self { case .all: return nil; case .image: return "image"
            case .video: return "video"; case .voice: return "voice" }
        }
        var symbol: String {
            switch self { case .all: return "square.grid.2x2"; case .image: return "photo"
            case .video: return "film"; case .voice: return "waveform" }
        }
    }

    @State private var statRange: StatRange = .weekly
    @State private var heatmapType: AssetType = .all
    @State private var timestampsByKind: [String: [Date]] = [:]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                statusSection
                activeModelSection
                externalClientsSection
                serverStatsSection
                heatmapSection
            }
            .padding(20)
        }
        .navigationTitle("Server")
        .task { await engine.refreshPool(); await loadStats() }
    }

    // MARK: - External clients (Krita / ComfyUI bridge)

    private var externalClientsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("External Clients (Krita / ComfyUI)").font(.headline)
            Text("ComfyBox exposes a ComfyUI-compatible bridge. In Krita AI Diffusion → Custom ComfyUI Server, use one of these:")
                .font(.caption).foregroundStyle(.secondary)
            endpointRow(label: "This Mac", url: "http://127.0.0.1:\(engine.serverPort)")
            if let ip = Self.lanIPv4() {
                endpointRow(label: "LAN", url: "http://\(ip):\(engine.serverPort)")
            }
            Text("Supports text-to-image, inpaint/outpaint, and DyPE (auto-enabled above 1024px). Results stream back over WebSocket.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private func endpointRow(label: String, url: String) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.caption).foregroundStyle(.secondary).frame(width: 56, alignment: .leading)
            Text(url).font(.callout.monospaced()).textSelection(.enabled)
            Spacer()
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(url, forType: .string)
            } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless).help("Copy")
        }
        .padding(8)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }

    /// Primary LAN IPv4 (en0/en1), for pointing external clients at this Mac.
    static func lanIPv4() -> String? {
        var address: String?
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return nil }
        defer { freeifaddrs(ifaddr) }
        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let flags = Int32(ptr.pointee.ifa_flags)
            guard let sa = ptr.pointee.ifa_addr else { continue }
            let family = sa.pointee.sa_family
            let name = String(cString: ptr.pointee.ifa_name)
            if (flags & (IFF_UP | IFF_RUNNING)) == (IFF_UP | IFF_RUNNING),
               family == UInt8(AF_INET), name == "en0" || name == "en1" {
                var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                if getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count),
                               nil, 0, NI_NUMERICHOST) == 0 {
                    address = String(cString: host)
                }
            }
        }
        return address
    }

    // MARK: - Server Stats (Daily / Weekly / All-time, by asset type)

    private func loadStats() async {
        if let store { timestampsByKind = (try? await store.assetCreationTimestampsByKind()) ?? [:] }
    }

    /// Timestamps for a type within the selected range.
    private func timestamps(for type: AssetType, range: StatRange) -> [Date] {
        let all: [Date] = type.kind.map { timestampsByKind[$0] ?? [] }
            ?? timestampsByKind.values.flatMap { $0 }
        guard let days = range.days else { return all.sorted() }
        let cutoff = Date().addingTimeInterval(-Double(days) * 86_400)
        return all.filter { $0 >= cutoff }.sorted()
    }

    private var serverStatsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Server Stats").font(.headline)
                Spacer()
                Picker("", selection: $statRange) {
                    ForEach(StatRange.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented).fixedSize().labelsHidden()
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 12)], spacing: 12) {
                ForEach(AssetType.allCases) { type in
                    StatTile(title: type.rawValue,
                             value: "\(timestamps(for: type, range: statRange).count)",
                             systemImage: type.symbol)
                }
                if let q = engine.queueInfo {
                    StatTile(title: "Renders (total)", value: "\(q.renderCount)", systemImage: "bolt.fill")
                }
            }
            Text("Generated in the \(statRange == .all ? "app's history" : statRange.rawValue.lowercased() + " window").")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    // MARK: - Generation heatmap (by asset type)

    private var heatmapSection: some View {
        let series = timestamps(for: heatmapType, range: .all)
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Generation Heatmap").font(.headline)
                Spacer()
                Picker("", selection: $heatmapType) {
                    ForEach(AssetType.allCases) { Label($0.rawValue, systemImage: $0.symbol).tag($0) }
                }.fixedSize().labelsHidden()
            }
            if series.isEmpty {
                Text("No \(heatmapType == .all ? "assets" : heatmapType.rawValue.lowercased()) generated yet.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ActivityHeatmapView(days: ActivityStats.dayCounts(timestamps: series, days: statRange.heatmapDays))
                Text("\(series.count) \(heatmapType == .all ? "total" : heatmapType.rawValue.lowercased()) over the last \(statRange.heatmapDays) days.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
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
