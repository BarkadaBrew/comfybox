// QueuePanel.swift — Render queue status panel
//
// Shows the current render queue status from the WarmServer health endpoint.
// Displays active render progress, pending queue count, completed render
// statistics, and server resource usage. Auto-refreshes via EngineService
// health polling (every 3 seconds).

import SwiftUI

struct QueuePanel: View {
    @Bindable var engine: EngineService

    @State private var jobs: EngineService.QueueJobList?
    @State private var actionError: String?

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
                pendingJobsSection
            } else if engine.connectionState.isConnected {
                Text("Loading queue status...")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                Text("Connect to server to view queue")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            if let actionError {
                Text(actionError)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }
        }
        // The health poll updates queueInfo every few seconds; piggyback on it
        // to refresh the detailed job list only while something is queued or
        // rendering (avoids constant polling when idle).
        .task(id: pollKey) { await reloadJobs() }
    }

    /// Changes whenever the coarse queue state moves, retriggering the job fetch.
    private var pollKey: String {
        let info = engine.queueInfo
        return "\(info?.pendingCount ?? 0)-\(info?.isRendering ?? false)-\(info?.currentJobId ?? "")-\(Int(info?.progressPercent ?? 0))"
    }

    private func reloadJobs() async {
        guard engine.connectionState.isConnected else { jobs = nil; return }
        jobs = await engine.fetchQueueJobs()
    }

    // MARK: - Pending jobs (management)

    @ViewBuilder
    private var pendingJobsSection: some View {
        if let jobs, !jobs.pending.isEmpty || jobs.isRendering {
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Jobs")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if jobs.isRendering {
                        Button {
                            Task {
                                do { try await engine.interruptRender(); await reloadJobs() }
                                catch { actionError = error.localizedDescription }
                            }
                        } label: {
                            Label("Interrupt", systemImage: "stop.circle")
                                .font(.caption2)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.orange)
                        .help("Cancel the in-flight render")
                    }
                    if !jobs.pending.isEmpty {
                        Button {
                            Task {
                                do { try await engine.clearQueue(); await reloadJobs() }
                                catch { actionError = error.localizedDescription }
                            }
                        } label: {
                            Label("Clear", systemImage: "xmark.bin")
                                .font(.caption2)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.red)
                        .help("Cancel all pending jobs")
                    }
                }

                if let summary = jobs.activeSummary, jobs.isRendering {
                    HStack(spacing: 4) {
                        Image(systemName: "play.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                        Text(summary)
                            .font(.caption2)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }

                ForEach(jobs.pending) { job in
                    HStack(spacing: 4) {
                        Image(systemName: jobIcon(job.kind))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text(job.summary)
                            .font(.caption2)
                            .lineLimit(1)
                            .truncationMode(.tail)
                        Spacer()
                        Button {
                            Task {
                                do { try await engine.cancelQueueJob(id: job.id); await reloadJobs() }
                                catch { actionError = error.localizedDescription }
                            }
                        } label: {
                            Image(systemName: "xmark.circle")
                                .font(.caption2)
                        }
                        .buttonStyle(.borderless)
                        .help("Cancel this job")
                    }
                }
            }
        }
    }

    private func jobIcon(_ kind: String) -> String {
        switch kind {
        case "generate": return "photo"
        case "controlnet": return "square.3.layers.3d"
        case "lora_swap": return "square.stack.3d.up"
        case "model_switch": return "arrow.triangle.2.circlepath"
        case "shutdown": return "power"
        default: return "clock"
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
