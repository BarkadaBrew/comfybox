import SwiftUI

/// One effective LTX-2 video parameter with provenance
/// (GET /v1/video/config/effective — task #9 Phase 1).
public struct EffectiveVideoParam: Codable, Sendable, Equatable, Identifiable {
    public let name: String
    public let envKey: String?
    public let tier: String
    public let value: String
    public let source: String   // request | preset | configFile | env | builtin
    public let valid: Bool
    public let note: String?
    public var id: String { name }
}

/// Read-only readout of the video config the NEXT render would use, with a
/// source badge per parameter. The missing-rescale detector: a value the user
/// expects to be set shows a gray `builtin` badge; rejected values show red.
struct EffectiveConfigCard: View {
    @Bindable var engine: EngineService
    @State private var params: [EffectiveVideoParam] = []
    @State private var loaded = false

    var body: some View {
        DisclosureGroup {
            if params.isEmpty {
                Text(loaded ? "Server did not report a config." : "Loading…")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(params) { p in
                        HStack(spacing: 6) {
                            Text(p.name)
                                .font(.system(.caption, design: .monospaced))
                                .frame(width: 150, alignment: .leading)
                            Text(p.value.isEmpty ? "—" : p.value)
                                .font(.system(.caption, design: .monospaced))
                                .lineLimit(1)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            badge(for: p)
                        }
                        if let note = p.note {
                            Text(note)
                                .font(.caption2).foregroundStyle(.red)
                                .padding(.leading, 156)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        } label: {
            HStack(spacing: 6) {
                Label("Effective Config", systemImage: "list.bullet.rectangle")
                    .font(.caption.weight(.medium))
                if params.contains(where: { !$0.valid }) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red).font(.caption2)
                }
            }
        }
        .task { await refresh() }
        .onChange(of: engine.connectionState.isConnected) { _, connected in
            if connected { Task { await refresh() } }
        }
    }

    private func refresh() async {
        params = await engine.fetchEffectiveVideoConfig()
        loaded = true
    }

    @ViewBuilder
    private func badge(for p: EffectiveVideoParam) -> some View {
        Text(p.valid ? p.source : "rejected")
            .font(.caption2)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(badgeColor(p).opacity(0.18), in: Capsule())
            .foregroundStyle(badgeColor(p))
    }

    private func badgeColor(_ p: EffectiveVideoParam) -> Color {
        if !p.valid { return .red }
        switch p.source {
        case "env": return .orange
        case "configFile": return .blue
        case "builtin": return .gray
        default: return .green
        }
    }
}
