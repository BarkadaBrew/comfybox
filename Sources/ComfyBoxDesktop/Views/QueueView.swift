// QueueView.swift — Comprehensive ComfyBox queue management
//
// Live view of the server render queue: the active job (prompt, source app,
// progress, elapsed) plus every pending job, with pause/resume, clear, per-job
// cancel, and reorder (priority). Sources are labelled by requesting app
// (Desktop, Krita/ComfyUI, Bree, API).

import SwiftUI

struct QueueView: View {
    @Bindable var engine: EngineService
    /// Daemon-side queue (Kira cycles) — the engine list below only ever
    /// shows the ONE dispatched job; the rest wait in the daemon.
    @Bindable var kira: KiraClient

    @State private var queue: EngineService.QueueJobList?
    @State private var refreshTick = 0
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                if let error { Label(error, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange) }
                activeSection
                pendingSection
                // Daemon-side queue: everything the scheduler has booked that
                // has NOT yet been dispatched to the engine (Todd 2026-08-30
                // "the queue doesnt show all queued or pending jobs").
                DaemonRenderQueuePanel(client: kira)
                countsFooter
            }
            .padding(20)
        }
        .navigationTitle("Queue")
        .task(id: refreshTick) {
            queue = await engine.fetchQueueJobs()
            try? await Task.sleep(for: .seconds(1.5))
            if !Task.isCancelled { refreshTick += 1 }
        }
    }

    // MARK: - Header (pause / clear)

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Render Queue").font(.title2.bold())
                Text(engine.connectionState.isConnected ? "Live from ComfyBox" : "Server offline")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if queue?.isPaused == true {
                Label("Paused", systemImage: "pause.circle.fill").font(.caption.bold()).foregroundStyle(.orange)
            }
            Button { Task { await toggle(pause: !(queue?.isPaused ?? false)) } } label: {
                Label(queue?.isPaused == true ? "Resume" : "Pause",
                      systemImage: queue?.isPaused == true ? "play.fill" : "pause.fill")
            }
            .disabled(busy || !engine.connectionState.isConnected)
            Button(role: .destructive) { Task { await clearAll() } } label: {
                Label("Clear pending", systemImage: "trash")
            }
            .disabled(busy || (queue?.pending.isEmpty ?? true))
        }
    }

    // MARK: - Active job

    /// Map the engine's LTX-2 render phase to a user-facing stage label. The
    /// base vs refine distinction is what makes a 2-pass render legible: you
    /// see "Rendering — base pass" then "Refining — 2nd pass" instead of the
    /// progress bar silently restarting.
    static func stageLabel(_ phase: String?) -> String? {
        switch phase {
        case "modelLoad":    return "Loading models"
        case "textEncode":   return "Encoding prompt"
        case "baseDenoise":  return "Rendering — base pass"
        case "refineDenoise": return "Refining — 2nd pass"
        case "vaeDecode":    return "Decoding video"
        case "vocoder":      return "Generating audio"
        case "postProcess":  return "Finishing"
        case .some(let p):   return p
        case .none:          return nil
        }
    }

    @ViewBuilder private var activeSection: some View {
        if let q = queue, q.isRendering {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "circle.fill").font(.system(size: 8)).foregroundStyle(.green)
                    Text("Rendering").font(.headline)
                    sourceBadge(q.activeSource ?? "api")
                    Spacer()
                    // PR #384 review, item 2b: name the row's own job as the
                    // interrupt `target` (comfybox#362). Without it, "stop this
                    // render" stopped whatever happened to be active by the time
                    // the request landed — which, on a queue shared with Bree and
                    // Kira, can be someone else's render.
                    Button(role: .destructive) { Task { await interrupt(jobId: q.activeJobId) } } label: {
                        Label("Interrupt", systemImage: "stop.fill")
                    }.controlSize(.small).disabled(busy)
                        .help(q.activeJobId.map { "Cancel this render (job \($0.prefix(8)))" }
                              ?? "Cancel whichever render is active")
                }
                Text(q.activeSummary ?? "—").font(.callout).foregroundStyle(.secondary)
                    .lineLimit(3).fixedSize(horizontal: false, vertical: true)
                if let stage = Self.stageLabel(q.phase) {
                    HStack(spacing: 6) {
                        Image(systemName: "wand.and.rays").font(.caption2).foregroundStyle(.secondary)
                        Text(stage).font(.caption.weight(.semibold)).foregroundStyle(.primary)
                    }
                }
                if let pct = q.progressPercent {
                    HStack(spacing: 8) {
                        ProgressView(value: Double(pct), total: 100)
                        Text("\(pct)%").font(.caption.monospacedDigit()).frame(width: 38)
                    }
                }
            }
            .padding(14)
            .background(.green.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        } else if queue != nil {
            Label("Idle — no active render", systemImage: "moon.zzz")
                .font(.callout).foregroundStyle(.secondary)
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: - Pending jobs (reorderable)

    @ViewBuilder private var pendingSection: some View {
        let jobs = queue?.pending ?? []
        VStack(alignment: .leading, spacing: 8) {
            Text("Pending (\(jobs.count))").font(.headline)
            if jobs.isEmpty {
                Text("Nothing queued.").font(.callout).foregroundStyle(.tertiary).padding(.vertical, 4)
            } else {
                ForEach(Array(jobs.enumerated()), id: \.element.id) { idx, job in
                    pendingRow(job, index: idx, count: jobs.count)
                }
            }
        }
    }

    private func pendingRow(_ job: EngineService.QueueJob, index: Int, count: Int) -> some View {
        HStack(spacing: 10) {
            Text("\(index + 1)").font(.caption.monospacedDigit().bold())
                .foregroundStyle(.secondary).frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    sourceBadge(job.source)
                    Text(job.kind).font(.caption2).foregroundStyle(.tertiary)
                    if let at = job.enqueuedAt {
                        Text(at, style: .relative).font(.caption2).foregroundStyle(.tertiary)
                    }
                }
                Text(job.summary).font(.callout).lineLimit(2).fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            // Reorder + cancel
            HStack(spacing: 2) {
                iconButton("chevron.up.2", help: "To top", disabled: index == 0) { await move(job, "top") }
                iconButton("chevron.up", help: "Up", disabled: index == 0) { await move(job, "up") }
                iconButton("chevron.down", help: "Down", disabled: index == count - 1) { await move(job, "down") }
                iconButton("xmark.circle.fill", help: "Cancel", disabled: false, tint: .red) { await cancel(job) }
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
    }

    private func iconButton(_ system: String, help: String, disabled: Bool, tint: Color = .secondary, _ action: @escaping () async -> Void) -> some View {
        Button { Task { await action() } } label: {
            Image(systemName: system).foregroundStyle(tint)
        }
        .buttonStyle(.borderless).controlSize(.small).help(help).disabled(disabled || busy)
    }

    private var countsFooter: some View {
        HStack(spacing: 16) {
            if let q = queue {
                Label("\(q.renderCount) done", systemImage: "checkmark.circle").foregroundStyle(.green)
                if q.failedCount > 0 { Label("\(q.failedCount) failed", systemImage: "xmark.octagon").foregroundStyle(.red) }
            }
            Spacer()
            Button { refreshTick += 1 } label: { Image(systemName: "arrow.clockwise") }
                .buttonStyle(.borderless).controlSize(.small).help("Refresh now")
        }
        .font(.caption)
        .padding(.top, 4)
    }

    // MARK: - Source badge

    private func sourceBadge(_ source: String) -> some View {
        let (label, color): (String, Color) = {
            switch source.lowercased() {
            case "desktop": return ("Desktop", .blue)
            case "comfyui", "krita": return ("Krita / ComfyUI", .purple)
            case "bree": return ("Bree", .pink)
            default: return (source.capitalized, .gray)
            }
        }()
        return Text(label).font(.caption2.bold())
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.2), in: Capsule())
            .foregroundStyle(color)
    }

    // MARK: - Actions

    private func run(_ op: @escaping () async throws -> Void) async {
        busy = true; error = nil
        do { try await op() } catch { self.error = error.localizedDescription }
        busy = false
        refreshTick += 1
    }
    private func toggle(pause: Bool) async { await run { try await engine.setQueuePaused(pause) } }
    private func clearAll() async { await run { try await engine.clearQueue() } }
    /// `jobId` nil = the legacy default target ("whatever /health shows as
    /// active"); non-nil = that job specifically, so we cannot stop someone
    /// else's render by racing the queue.
    private func interrupt(jobId: String?) async { await run { try await engine.interruptRender(target: jobId) } }
    private func cancel(_ job: EngineService.QueueJob) async { await run { try await engine.cancelQueueJob(id: job.id) } }
    private func move(_ job: EngineService.QueueJob, _ dir: String) async { await run { try await engine.moveQueueJob(id: job.id, direction: dir) } }
}
