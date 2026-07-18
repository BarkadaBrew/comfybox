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
    @State private var suggestionDraft: String = ""
    @State private var suggestionKind: String = "image"
    /// Which cards are expanded. Empty by default → every card starts collapsed
    /// on launch (Todd 2026-07-17). Expansion is per-session, not persisted.
    @State private var expandedCards: Set<String> = []

    var body: some View {
        VStack(spacing: 0) {
            healthStrip
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if client.lastError != nil {
                        unreachableCard
                    }
                    if let actionError = client.actionError {
                        Label(actionError, systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                    foldable("Her Now", systemImage: "sparkles", "herNow") { herNowBody }
                    foldable("Agenda", systemImage: "list.bullet.rectangle", "agenda") { agendaBody }
                    foldable("Creation", systemImage: "wand.and.rays", "controls") { controlsBody }
                    foldable("Suggestion box", systemImage: "lightbulb", "suggest") { suggestionBody }
                    foldable("Compute", systemImage: "memorychip", "compute") { computeBody }
                    foldable("Recent output", systemImage: "photo.stack", "recent") { recentOutputBody }
                    foldable("Core binding", systemImage: "network", "binding",
                             forceExpanded: client.lastError != nil) { bindingBody }
                    foldable("Conversation", systemImage: "bubble.left.and.bubble.right", "convo") {
                        Text("Embedded comms over /ws/backroom with shared Telegram memory — Workstream A6 (#1317) + D6 (#245), the one service-contract piece not yet landed.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
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

    // MARK: - Foldable card container (collapsed by default)

    private func cardExpansion(_ key: String) -> Binding<Bool> {
        Binding(
            get: { expandedCards.contains(key) },
            set: { isOn in
                if isOn { expandedCards.insert(key) } else { expandedCards.remove(key) }
            })
    }

    /// A collapsible card. `forceExpanded` pins it open regardless of user
    /// state (used to surface the binding editor when the daemon is unreachable).
    private func foldable<Content: View>(
        _ title: String, systemImage: String, _ key: String,
        forceExpanded: Bool = false,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        let expansion = forceExpanded ? .constant(true) : cardExpansion(key)
        return DisclosureGroup(isExpanded: expansion) {
            VStack(alignment: .leading, spacing: 8, content: content)
                .padding(.top, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label(title, systemImage: systemImage).font(.headline)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4)))
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
                if expandedCards.contains("binding") {
                    expandedCards.remove("binding")
                } else {
                    expandedCards.insert("binding")
                }
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

    @ViewBuilder private var bindingBody: some View {
        Text("\(client.binding.host):\(client.binding.port) · token \(client.token.isEmpty ? "not set" : "set")")
            .font(.callout)
            .foregroundStyle(.secondary)
        HStack {
            TextField("Host", text: Binding(
                get: { client.binding.host },
                set: { client.binding.host = $0 }))
                .textFieldStyle(.roundedBorder)
                .frame(width: 160)
            TextField("Port", value: Binding(
                get: { client.binding.port },
                set: { client.binding.port = $0 }), format: .number.grouping(.never))
                .textFieldStyle(.roundedBorder)
                .frame(width: 70)
            SecureField("Daemon API token", text: $tokenDraft)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 160)
                .onSubmit { client.token = tokenDraft }
            Button("Apply") {
                client.token = tokenDraft
                Task { await client.refreshHealth() }
            }
        }
        Text("Token is stored in the macOS Keychain. One switch re-points the whole tab (host-agnostic) — no per-card configuration.")
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }

    // MARK: - Live dashboard cards (D2–D4)

    @ViewBuilder private var herNowBody: some View {
        if let state = client.state {
            HStack(spacing: 10) {
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
            Text(client.stateError ?? "loading…").font(.caption).foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder private var agendaBody: some View {
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

    @ViewBuilder private var controlsBody: some View {
        if client.actionInFlight {
            ProgressView().controlSize(.small)
        }
        Group {
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

    @ViewBuilder private var suggestionBody: some View {
        Text("Ideas Kira picks up on her content runs — image/video seed one render, a session themes one cycle, an arc stays active until you remove it.")
            .font(.caption2).foregroundStyle(.tertiary)
        HStack(spacing: 8) {
            Picker("", selection: $suggestionKind) {
                ForEach(client.suggestionKinds, id: \.self) { kind in
                    Text(kind).tag(kind)
                }
            }
            .labelsHidden()
            .frame(width: 110)
            TextField("e.g. golden hour on the balcony in the red dress", text: $suggestionDraft)
                .textFieldStyle(.roundedBorder)
                .onSubmit { submitSuggestion() }
            Button("Add") { submitSuggestion() }
                .disabled(client.actionInFlight
                          || suggestionDraft.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        if !client.suggestions.isEmpty {
            ForEach(client.suggestions.prefix(8)) { suggestion in
                HStack(spacing: 8) {
                    chip(suggestion.kind, tint: kindTint(suggestion.kind))
                    Text(suggestion.text).font(.caption).lineLimit(1)
                    Spacer()
                    Text(suggestion.status == "pending"
                         ? (suggestion.kind == "arc" ? "active" : "pending")
                         : "picked up")
                        .font(.caption2)
                        .foregroundStyle(suggestion.status == "pending" ? Color.blue : .secondary)
                    Button {
                        Task { await client.deleteSuggestion(suggestion.id) }
                    } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                    }
                    .buttonStyle(.borderless)
                    .help(suggestion.status == "pending" ? "Remove before pickup" : "Clear")
                }
            }
        }
    }

    private func submitSuggestion() {
        let text = suggestionDraft
        suggestionDraft = ""
        Task { await client.addSuggestion(kind: suggestionKind, text: text) }
    }

    private func kindTint(_ kind: String) -> Color {
        switch kind {
        case "image": return .teal
        case "video": return .indigo
        case "arc": return .purple
        case "session": return .orange
        default: return .gray
        }
    }

    @ViewBuilder private var computeBody: some View {
        HStack {
            if client.computeLoading {
                ProgressView().controlSize(.small)
            } else {
                Button("Refresh") { Task { await client.refreshCompute() } }
                    .controlSize(.small)
            }
            Spacer()
        }
        if let error = client.computeError {
            Text(error).font(.caption).foregroundStyle(.orange)
            Text("The compute read proxies the ComfyBox engine — slow or flaky if the bridge is remote (FDD R-6).")
                .font(.caption2).foregroundStyle(.tertiary)
        } else if let compute = client.computeSnapshot {
            computeMeter(compute)
        } else {
            Text("on-demand — tap Refresh (free-compute and bridge-reconnect are 501 stubs server-side until real flows are wired)")
                .font(.caption).foregroundStyle(.tertiary)
        }
    }

    /// VRAM meter (FDD C1): resident model VRAM against the admission budget, so
    /// the video-render headroom is visible at a glance.
    @ViewBuilder
    private func computeMeter(_ compute: KiraComputeSnapshot) -> some View {
        let resident = compute.residentVramMB ?? 0
        let budget = compute.budgetVramMB ?? 0
        if budget > 0 {
            let fraction = min(max(resident / budget, 0), 1)
            let headroom = max(budget - resident, 0)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Model VRAM").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Text("\(gb(resident)) / \(gb(budget)) GB")
                        .font(.caption.monospacedDigit())
                }
                // Filled bar with the admission budget as the full track.
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4).fill(.quaternary)
                        RoundedRectangle(cornerRadius: 4)
                            .fill(fraction > 0.85 ? Color.orange : Color.green)
                            .frame(width: geo.size.width * fraction)
                    }
                }
                .frame(height: 10)
                Text("Budget is the quantized LTX-2 admission floor. Headroom \(gb(headroom)) GB — a video render evicts resident image models first, then needs ~\(gb(budget)) GB.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }

        Grid(alignment: .leading, horizontalSpacing: 8, verticalSpacing: 3) {
            if let device = compute.deviceName {
                statRow("Device", "\(device)\(compute.deviceTotalBytes.map { " · \(gb($0 / (1024*1024))) GB unified" } ?? "")")
            }
            if let rss = compute.processMemoryMB {
                statRow("Warm server", "\(gb(rss)) GB resident")
            }
            if let model = compute.activeModel {
                statRow("Active model", "\(model)\(compute.modelFamily.map { " · \($0)" } ?? "")")
            }
            statRow("Status", statusText(compute))
        }

        DisclosureGroup("Raw") {
            Text(client.computeSummary ?? "")
                .font(.system(.caption2, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.caption2)
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).font(.caption2).foregroundStyle(.tertiary)
                .gridColumnAlignment(.trailing)
            Text(value).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
    }

    private func statusText(_ compute: KiraComputeSnapshot) -> String {
        var parts: [String] = []
        if let loaded = compute.loaded { parts.append(loaded ? "loaded" : "not loaded") }
        if let rendering = compute.isRendering { parts.append(rendering ? "rendering" : "idle") }
        return parts.isEmpty ? "—" : parts.joined(separator: " · ")
    }

    /// MB → GB, one decimal.
    private func gb(_ mb: Double) -> String {
        String(format: "%.1f", mb / 1024)
    }

    @ViewBuilder private var recentOutputBody: some View {
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
            .contentGated()   // G-rated by default (Todd 2026-07-17)
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
