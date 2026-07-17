// KiraView.swift — Kira tab in the Suite (comfybox#240 D1)
//
// D1 scope: health strip (always truthful, fast-polled while visible) +
// host-agnostic (host, port, token) binding editor. The dashboard cards
// (state/world/agenda/mode/scheduler/compute/gallery) and the embedded comms
// pane bind to the Workstream A service contract (/v1/kira/*) and land as
// D2–D7 — they render here as explicit placeholders, not empty lies.

import SwiftUI

struct KiraView: View {
    @Bindable var client: KiraClient
    @State private var tokenDraft: String = ""
    @State private var showBindingEditor = false

    var body: some View {
        VStack(spacing: 0) {
            healthStrip
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if client.lastError != nil {
                        unreachableCard
                    }
                    if let actionError = client.actionError {
                        Label(actionError, systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                    herNowCard
                    agendaCard
                    controlsCard
                    computeCard
                    recentOutputCard
                    bindingCard
                    placeholderCard(
                        title: "Conversation",
                        detail: "Embedded comms over /ws/backroom with shared Telegram memory — Workstream A6 (#1317) + D6 (#245), the one service-contract piece not yet landed.")
                }
                .padding(16)
                .frame(maxWidth: 700, alignment: .leading)
            }
        }
        .navigationTitle("Kira")
        .onAppear {
            tokenDraft = client.token
            client.startPolling()
        }
        .onDisappear {
            client.stopPolling()
        }
    }

    // MARK: - Health strip (F1)

    private var healthStrip: some View {
        HStack(spacing: 14) {
            statusDot(
                label: "daemon",
                ok: client.isReachable,
                detailWhenBad: client.lastError)
            statusDot(
                label: "running",
                ok: client.health?.isRunning == true,
                detailWhenBad: client.health == nil ? nil : "daemon reports not running")
            statusDot(
                label: client.health?.isPaused == true ? "paused" : "active",
                ok: client.health?.isPaused == false,
                detailWhenBad: client.health?.isPaused == true ? "daemon is paused" : nil)
            statusDot(
                label: "auto-render",
                ok: client.health?.autonomousRenderEnabled == true,
                detailWhenBad: client.health?.autonomousRenderEnabled == false ? "autonomous rendering off" : nil)

            Spacer()

            if let health = client.health {
                Text("\(health.name) · \(health.toolCount.map { "\($0) tools" } ?? "")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(health.fetchedAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .help("Last successful health poll")
            }

            Button {
                showBindingEditor.toggle()
            } label: {
                Label("\(client.binding.host):\(client.binding.port)", systemImage: "network")
                    .font(.caption)
            }
            .buttonStyle(.borderless)
            .help("Kira core binding — host-agnostic: the Linux daemon today, 127.0.0.1 after the Mac migration")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private func statusDot(label: String, ok: Bool, detailWhenBad: String?) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(client.health == nil && label != "daemon" ? Color.gray : (ok ? Color.green : Color.red))
                .frame(width: 8, height: 8)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .help(detailWhenBad ?? "\(label): ok")
    }

    // MARK: - Cards

    private var unreachableCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Kira's core is unreachable", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(.orange)
            Text(client.lastError ?? "")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("The daemon API listens on localhost on the server — until the Mac migration (Workstream B) or a LAN bind, reach it via an SSH tunnel: ssh -N -L \(client.binding.port):127.0.0.1:\(client.binding.port) todd@10.0.100.232, then bind to 127.0.0.1.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.08)))
    }

    private var bindingCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Core binding", systemImage: "network")
                .font(.headline)
            if showBindingEditor || client.lastError != nil {
                HStack {
                    TextField("Host", text: Binding(
                        get: { client.binding.host },
                        set: { client.binding.host = $0 }))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 180)
                    TextField("Port", value: Binding(
                        get: { client.binding.port },
                        set: { client.binding.port = $0 }), format: .number.grouping(.never))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 80)
                    SecureField("Daemon API token", text: $tokenDraft)
                        .textFieldStyle(.roundedBorder)
                        .frame(minWidth: 180)
                        .onSubmit { client.token = tokenDraft }
                    Button("Apply") {
                        client.token = tokenDraft
                        Task { await client.refreshHealth() }
                    }
                }
                Text("Token is stored in the macOS Keychain. One switch re-points the whole tab (host-agnostic) — no per-card configuration.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            } else {
                Text("\(client.binding.host):\(client.binding.port) · token \(client.token.isEmpty ? "not set" : "set")")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4)))
    }

    // MARK: - Live dashboard cards (D2–D4)

    private var herNowCard: some View {
        card {
            if let state = client.state {
                HStack(spacing: 10) {
                    Label("Her Now", systemImage: "sparkles").font(.headline)
                    chip(state.mood, tint: .pink)
                    chip(state.energy, tint: .orange)
                    chip(state.arcPhase, tint: .purple)
                    Spacer()
                    Text(state.fetchedAt, style: .relative)
                        .font(.caption2).foregroundStyle(.tertiary)
                }
                if let dynamicLine = state.dynamicLine {
                    Text(dynamicLine).font(.caption).foregroundStyle(.secondary)
                }
                Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 4) {
                    ForEach(state.scores, id: \.name) { score in
                        GridRow {
                            Text(score.name).font(.caption).foregroundStyle(.secondary)
                                .gridColumnAlignment(.trailing)
                            ProgressView(value: min(max(score.value, 0), 100), total: 100)
                                .frame(minWidth: 140)
                            Text("\(Int(score.value))").font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                if !state.worldPresent {
                    Text("world slice not served yet (A3 gap — barkada/Ube land with it)")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            } else {
                Label("Her Now", systemImage: "sparkles").font(.headline).foregroundStyle(.secondary)
                Text(client.stateError ?? "loading…").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private var agendaCard: some View {
        card {
            Label("Agenda", systemImage: "list.bullet.rectangle").font(.headline)
            if let beat = client.state?.campaignBeat {
                Text(beat).font(.caption).foregroundStyle(.secondary)
            }
            if let agenda = client.state?.agenda, !agenda.isEmpty {
                ForEach(Array(agenda.prefix(6).enumerated()), id: \.offset) { _, item in
                    Text("• \(item)").font(.caption).lineLimit(1)
                }
            } else {
                Text("nothing queued").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private var controlsCard: some View {
        card {
            HStack {
                Label("Creation", systemImage: "wand.and.rays").font(.headline)
                Spacer()
                if client.actionInFlight { ProgressView().controlSize(.small) }
            }
            if let scheduler = client.scheduler {
                HStack(spacing: 12) {
                    Toggle("24/7 creation", isOn: Binding(
                        get: { !scheduler.paused },
                        set: { on in Task { await client.setSchedulerPaused(!on) } }))
                        .toggleStyle(.switch)
                        .disabled(client.actionInFlight)
                    Text(cadenceLine(scheduler))
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else {
                Text("scheduler status unavailable").font(.caption).foregroundStyle(.tertiary)
            }
            if !client.allowedModes.isEmpty {
                Picker("Mode", selection: Binding(
                    get: { client.contentMode ?? "" },
                    set: { mode in Task { await client.setContentMode(mode) } })) {
                    ForEach(client.allowedModes, id: \.self) { mode in
                        Text(modeEmoji(mode)).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 260)
                .disabled(client.actionInFlight)
                .help("Content mode — allowlisted server-side; reflected across surfaces")
            }
        }
    }

    private var computeCard: some View {
        card {
            HStack {
                Label("Compute", systemImage: "memorychip").font(.headline)
                Spacer()
                if client.computeLoading {
                    ProgressView().controlSize(.small)
                } else {
                    Button("Refresh") { Task { await client.refreshCompute() } }
                        .controlSize(.small)
                }
            }
            if let error = client.computeError {
                Text(error).font(.caption).foregroundStyle(.orange)
                Text("The compute read proxies ComfyBox over the interim SSH/MCP bridge — slow or flaky until the Mac migration (FDD R-6).")
                    .font(.caption2).foregroundStyle(.tertiary)
            } else if let summary = client.computeSummary {
                Text(summary)
                    .font(.system(.caption2, design: .monospaced))
                    .lineLimit(14)
                    .textSelection(.enabled)
            } else {
                Text("on-demand — tap Refresh (free-compute and bridge-reconnect are 501 stubs server-side until real flows are wired)")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private var recentOutputCard: some View {
        card {
            Label("Recent output", systemImage: "photo.stack").font(.headline)
            if client.recentMedia.isEmpty {
                Text("no recent media reported").font(.caption).foregroundStyle(.tertiary)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(client.recentMedia) { item in
                            KiraMediaThumb(client: client, item: item)
                        }
                    }
                }
                .frame(height: 96)
            }
        }
    }

    private func cadenceLine(_ scheduler: KiraSchedulerStatus) -> String {
        var parts: [String] = []
        if let interval = scheduler.intervalMinutes { parts.append("every \(interval)m") }
        if let images = scheduler.imageCount { parts.append("\(images) img") }
        if let videos = scheduler.videoCount { parts.append("\(videos) video (\(scheduler.videoMode ?? "?"))") }
        return parts.joined(separator: " · ")
    }

    private func chip(_ text: String?, tint: Color) -> some View {
        Group {
            if let text {
                Text(text)
                    .font(.caption)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill(tint.opacity(0.15)))
                    .foregroundStyle(tint)
            }
        }
    }

    private func modeEmoji(_ mode: String) -> String {
        switch mode {
        case "apple": return "🍏"
        case "neutral": return "⚪️"
        case "banana": return "🍌"
        case "avocado": return "🥑"
        default: return mode
        }
    }

    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8, content: content)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4)))
    }

    private func placeholderCard(title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.secondary)
            Text(detail)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary, style: StrokeStyle(lineWidth: 1, dash: [4])))
        .opacity(0.8)
    }
}

/// Auth-aware media thumbnail — loads bytes through the daemon's guarded
/// file serve (AsyncImage can't attach the Bearer header).
private struct KiraMediaThumb: View {
    let client: KiraClient
    let item: KiraMediaItem
    @State private var image: NSImage?
    @State private var failed = false

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                RoundedRectangle(cornerRadius: 6)
                    .fill(.quaternary)
                    .overlay {
                        if failed {
                            Image(systemName: "eye.slash").foregroundStyle(.tertiary)
                        } else if item.kind == "video" {
                            Image(systemName: "film").foregroundStyle(.tertiary)
                        } else {
                            ProgressView().controlSize(.small)
                        }
                    }
            }
        }
        .frame(width: 84, height: 96)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .help((item.path as NSString).lastPathComponent)
        .task {
            guard image == nil, item.kind == "image" else { return }
            if let data = await client.loadMediaData(for: item), let loaded = NSImage(data: data) {
                image = loaded
            } else {
                failed = true
            }
        }
    }
}
