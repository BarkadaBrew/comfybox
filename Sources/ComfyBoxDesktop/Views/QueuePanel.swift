// QueuePanel.swift — Render queue status panel
//
// Shows the current render queue status from the WarmServer health endpoint.
// Displays active render progress, pending queue count, completed render
// statistics, and server resource usage. Auto-refreshes via EngineService
// health polling (every 3 seconds).

import SwiftUI

struct QueuePanel: View {
    @Bindable var engine: EngineService

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Text("Render Queue")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                if engine.connectionState.isConnected {
                    statusIndicator
                }
            }

            if let info = engine.queueInfo {
                queueContent(info)
            } else if engine.connectionState.isConnected {
                Text("Loading queue status...")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                Text("Connect to server to view queue")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Status Indicator

    private var statusIndicator: some View {
        HStack(spacing: 4) {
            if engine.isGenerating || (engine.queueInfo?.isRendering ?? false) {
                Circle()
                    .fill(.orange)
                    .frame(width: 6, height: 6)
                Text("Rendering")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            } else {
                Circle()
                    .fill(.green)
                    .frame(width: 6, height: 6)
                Text("Idle")
                    .font(.caption2)
                    .foregroundStyle(.green)
            }
        }
    }

    // MARK: - Queue Content

    private func queueContent(_ info: QueueInfo) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // Active render
            activeRenderSection(info)

            Divider()

            // Statistics
            statsSection(info)

            // Server resources
            resourceSection(info)

            // Last error
            if let lastError = info.lastError, !lastError.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Last Error")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(lastError)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .lineLimit(3)
                }
            }
        }
    }

    private func activeRenderSection(_ info: QueueInfo) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Status")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if info.isRendering {
                    if let percent = info.progressPercent {
                        // Determinate progress when the server reports it.
                        HStack(spacing: 6) {
                            ProgressView(value: min(max(percent, 0), 100), total: 100)
                                .controlSize(.small)
                                .frame(width: 80)
                            Text("\(Int(percent.rounded()))%")
                                .font(.caption)
                                .monospacedDigit()
                                .foregroundStyle(.orange)
                        }
                    } else {
                        HStack(spacing: 4) {
                            ProgressView()
                                .controlSize(.mini)
                            Text("Rendering...")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }
                } else {
                    Text("Ready")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }

            // Current job id
            if info.isRendering, let jobId = info.currentJobId {
                HStack {
                    Text("Job")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(jobId)
                        .font(.caption)
                        .monospaced()
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            // Pending count
            if info.pendingCount > 0 {
                HStack {
                    Text("Pending")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(info.pendingCount) job\(info.pendingCount == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    private func statsSection(_ info: QueueInfo) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Statistics")
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack {
                Text("Completed renders")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(info.renderCount)")
                    .font(.caption)
                    .monospacedDigit()
            }

            if let lastMs = info.lastRenderDurationMs, lastMs > 0 {
                HStack {
                    Text("Last render")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(formatDuration(lastMs))
                        .font(.caption)
                        .monospacedDigit()
                }
            }

            HStack {
                Text("Server uptime")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(formatUptime(info.uptimeSeconds))
                    .font(.caption)
                    .monospacedDigit()
            }
        }
    }

    private func resourceSection(_ info: QueueInfo) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Resources")
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack {
                Text("Memory usage")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(formatMemory(info.memoryUsageMB))
                    .font(.caption)
                    .monospacedDigit()
            }

            if !engine.poolModels.isEmpty {
                HStack {
                    Text("Models loaded")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(engine.poolModels.count)")
                        .font(.caption)
                        .monospacedDigit()
                }
            }
        }
    }

    // MARK: - Formatting

    private func formatDuration(_ ms: Int) -> String {
        if ms < 1000 { return "\(ms) ms" }
        let seconds = Double(ms) / 1000.0
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return "\(minutes)m \(remainingSeconds)s"
    }

    private func formatUptime(_ seconds: Int) -> String {
        if seconds < 60 { return "\(seconds)s" }
        if seconds < 3600 {
            let m = seconds / 60
            let s = seconds % 60
            return "\(m)m \(s)s"
        }
        let h = seconds / 3600
        let m = (seconds % 3600) / 60
        return "\(h)h \(m)m"
    }

    private func formatMemory(_ mb: UInt64) -> String {
        if mb < 1024 { return "\(mb) MB" }
        let gb = Double(mb) / 1024.0
        return String(format: "%.1f GB", gb)
    }
}
