// DaemonRenderQueuePanel.swift — the DAEMON-side render queue, shared by the
// Kira tab and the Queue tab (Todd 2026-08-30: the engine queue only ever
// holds the one dispatched job; this is where everything else waits).
//
// Verbose booking description per row (kind/mode/clip/quality/loras/theme),
// age + watchdog, move up/down within lane, expedite, selective delete.

import SwiftUI

struct DaemonRenderQueuePanel: View {
    @Bindable var client: KiraClient

    // ── Daemon render queue (Todd 2026-08-30) ────────────────────────────

    private func queueAge(_ row: KiraQueueRow) -> String {
        let secs = max(0, Date().timeIntervalSince1970 - row.enqueuedAtMs / 1000)
        if secs < 90 { return "\(Int(secs))s" }
        if secs < 5400 { return "\(Int(secs / 60))m" }
        return String(format: "%.1fh", secs / 3600)
    }

    @ViewBuilder var body: some View {
        DisclosureGroup(isExpanded: .constant(true)) {
            if client.queueRows.isEmpty {
                Text("queue empty").font(.caption2).foregroundStyle(.tertiary)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(client.queueRows) { row in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(row.state == "running" ? "▶" : "\(row.pos)")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(row.state == "running" ? .green : .secondary)
                                .frame(width: 22, alignment: .trailing)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(row.detail ?? row.label)
                                    .font(.caption)
                                    .lineLimit(2)
                                HStack(spacing: 6) {
                                    Text("#\(row.id) \(row.kind) · \(row.owner) · waiting \(queueAge(row))")
                                        .font(.caption2).foregroundStyle(.tertiary)
                                    if let wd = row.watchdogMs {
                                        Text("watchdog \(Int(wd / 60000))m")
                                            .font(.caption2).foregroundStyle(.tertiary)
                                    }
                                }
                            }
                            Spacer()
                            if row.state == "pending" {
                                Button { Task { await client.moveQueueJob(row.id, delta: -1) } } label: { Text("↑") }
                                    .buttonStyle(.plain).help("Move up (within its lane)")
                                Button { Task { await client.moveQueueJob(row.id, delta: 1) } } label: { Text("↓") }
                                    .buttonStyle(.plain).help("Move down")
                                Button { Task { await client.bumpQueueJob(row.id) } } label: { Text("⤒") }
                                    .buttonStyle(.plain).help("Expedite — front of its lane + next-pick priority")
                                Button { Task { await client.removeQueueJob(row.id) } } label: { Text("✕").foregroundStyle(.red) }
                                    .buttonStyle(.plain).help("Remove this job from the queue (the render it would have produced is skipped)")
                            }
                        }
                        .padding(.vertical, 1)
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                Text("Render queue").font(.caption)
                let pending = client.queueRows.filter { $0.state == "pending" }.count
                Text(client.queueRows.isEmpty ? "idle" : "\(pending) pending")
                    .font(.caption2).foregroundStyle(.secondary)
                Button { Task { await client.refreshRenderQueue() } } label: {
                    Image(systemName: "arrow.clockwise").font(.caption2)
                }
                .buttonStyle(.plain).help("Refresh")
            }
        }
        .onAppear { Task { await client.refreshRenderQueue() } }
        .task {
            // Passive refresh while the tab is visible — the queue moves on
            // render timescales (minutes), 15s is plenty.
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 15_000_000_000)
                await client.refreshRenderQueue()
            }
        }
    }

}
