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
    @State private var hostDraft: String = ""
    @State private var portDraft: String = ""
    @State private var suggestionDraft: String = ""
    @State private var suggestionKind: String = "image"
    @State private var suggestionTier: String = "any"
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
                    foldable("Character", systemImage: "person.text.rectangle", "character") {
                        KiraCharacterCard(client: client)
                    }
                    foldable("Lorebook", systemImage: "book.closed", "lorebook") {
                        KiraLorebookCard(client: client)
                    }
                    foldable("World map", systemImage: "map", "world") {
                        KiraWorldMapCard(client: client)
                    }
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
            hostDraft = client.binding.host
            portDraft = String(client.binding.port)
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
            // Drafts, not live bindings (Kimi review 2026-07-27): binding.didSet
            // persists to UserDefaults and restarts polling, so typing directly
            // into client.binding did both on EVERY keystroke. Apply commits once.
            TextField("Host", text: $hostDraft)
                .textFieldStyle(.roundedBorder)
                .frame(width: 160)
            TextField("Port", text: $portDraft)
                .textFieldStyle(.roundedBorder)
                .frame(width: 70)
            SecureField("Daemon API token", text: $tokenDraft)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 160)
                .onSubmit { applyBinding() }
            Button("Apply") { applyBinding() }
        }
        Text("Token is stored in the macOS Keychain. One switch re-points the whole tab (host-agnostic) — no per-card configuration.")
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }

    /// Commit the binding drafts in one shot (persist + poll restart happen once).
    private func applyBinding() {
        client.token = tokenDraft
        var binding = client.binding
        binding.host = hostDraft.trimmingCharacters(in: .whitespaces)
        if let port = Int(portDraft.trimmingCharacters(in: .whitespaces)), (1...65535).contains(port) {
            binding.port = port
        } else {
            portDraft = String(binding.port)   // reject garbage, restore the live value
        }
        client.binding = binding   // no-op if unchanged (didSet guards equality)
        Task { await client.refreshHealth() }
    }

    // MARK: - Live dashboard cards (D2–D4)

    @ViewBuilder private var herNowBody: some View {
        if let state = client.state {
            if let staleNote = client.stateError {
                // Set only when a refresh failed while a snapshot is on screen.
                Text(staleNote).font(.caption2).foregroundStyle(.orange)
            }
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
            // Editable attributes (Todd 2026-07-27): pin mood, override arc phase.
            HerNowEditor(client: client)
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
                // Stream steering (tiered scheduler v2, Todd 2026-07-27):
                // precedence = your override > Kira's choice > schedule.
                if let stream = client.streamStatus {
                    HStack(spacing: 10) {
                        Text("Stream:").font(.caption).foregroundStyle(.secondary)
                        Text(streamSummary(stream)).font(.caption)
                        Picker("", selection: Binding(
                            get: { stream.todd ?? "auto" },
                            set: { v in Task { await client.setStreamOverride(v == "auto" ? nil : v) } })) {
                            Text("Auto").tag("auto")
                            ForEach(Self.tierOrder, id: \.self) { Text(modeEmoji($0)).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 240)
                        .disabled(client.actionInFlight)
                        .help("Your sticky override — pins every cycle to a tier, all hours, until Auto. Kira's own choice and the schedule take over when cleared.")
                    }
                }
                // Per-tier schedule + pacing. Editing writes the FULL tiers
                // map (server replaces; an unchecked tier is scheduled off).
                ForEach(Self.tierOrder, id: \.self) { mode in
                    tierRow(mode, scheduler: scheduler)
                }
                Text("Overlapping windows are fine — the most explicit open tier wins. Kira can pick any tier herself; /stream overrides from Telegram.")
                    .font(.caption2).foregroundStyle(.tertiary)
            } else {
                Text("scheduler status unavailable").font(.caption).foregroundStyle(.tertiary)
            }
            if !client.allowedModes.isEmpty {
                Picker("Chat mode", selection: Binding(
                    get: { client.contentMode ?? "" },
                    set: { mode in Task { await client.setContentMode(mode) } })) {
                    ForEach(client.allowedModes, id: \.self) { mode in
                        Text(modeEmoji(mode)).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 300)
                .disabled(client.actionInFlight)
                .help("Conversation register only — the 24/7 stream is driven by the tier schedule above, not this picker.")
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
            Picker("", selection: $suggestionTier) {
                Text("any").tag("any")
                ForEach(Self.tierOrder, id: \.self) { Text(modeEmoji($0)).tag($0) }
            }
            .labelsHidden()
            .frame(width: 76)
            .help("Optional tier tag — the idea waits for a cycle of that tier (an apple-tagged arc themes only apple cycles). \"any\" = next cycle regardless.")
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
        // "any" = untagged (consumed by whatever tier's cycle runs next);
        // a tier tag holds the idea for that tier's window.
        let tier = suggestionTier == "any" ? nil : suggestionTier
        Task { await client.addSuggestion(kind: suggestionKind, text: text, tier: tier) }
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

    /// Display order for tier rows — mildest first (the scheduler's own
    /// overlap precedence is the reverse: most explicit wins).
    static let tierOrder = ["neutral", "apple", "banana", "avocado"]

    private func cadenceLine(_ scheduler: KiraSchedulerStatus) -> String {
        var parts: [String] = []
        if let interval = scheduler.intervalMinutes { parts.append("every \(interval)m") }
        if scheduler.tiers.isEmpty {
            // Pre-v2 daemon: legacy flat summary.
            if scheduler.unlimitedImages {
                parts.append("∞ img (fit cycle)")
            } else if let images = scheduler.imageCount {
                parts.append("\(images) img")
            }
            if let videos = scheduler.videoCount { parts.append("\(videos) video") }
            if let start = scheduler.activeHoursStart, let end = scheduler.activeHoursEnd {
                parts.append("\(Self.windowLabel(start))–\(Self.windowLabel(end))")
            }
        } else {
            for mode in Self.tierOrder {
                guard let tier = scheduler.tiers[mode] else { continue }
                var bit = modeEmoji(mode)
                if let start = tier.activeHoursStart, let end = tier.activeHoursEnd {
                    bit += " \(Self.windowLabel(start))–\(Self.windowLabel(end))"
                } else {
                    bit += " 24/7"
                }
                bit += tier.unlimitedImages ? " ∞" : " \(tier.imageCount)img"
                if mode != "neutral" && tier.videoCount > 0 { bit += " \(tier.videoCount)v" }
                parts.append(bit)
            }
        }
        return parts.joined(separator: " · ")
    }

    private func streamSummary(_ stream: KiraStreamStatus) -> String {
        let mode = stream.effectiveMode.map { "\(modeEmoji($0)) \($0)" } ?? "idle (no window open)"
        switch stream.source {
        case "override-todd": return "\(mode) — your override"
        case "override-kira": return "\(mode) — Kira's choice"
        case "none-open": return mode
        default: return "\(mode) — schedule"
        }
    }

    // ── Tier rows (tiered scheduler v2) ──

    /// Mutate-and-PUT: the server treats the tiers map as a full replacement.
    private func putTiers(_ mutate: (inout [String: KiraTierConfig]) -> Void) {
        guard var tiers = client.scheduler?.tiers else { return }
        mutate(&tiers)
        Task { await client.updateSchedulerTiers(tiers) }
    }

    @ViewBuilder private func tierRow(_ mode: String, scheduler: KiraSchedulerStatus) -> some View {
        HStack(spacing: 10) {
            Toggle(isOn: Binding(
                get: { scheduler.tiers[mode] != nil },
                set: { on in
                    putTiers { tiers in
                        if on {
                            tiers[mode] = KiraTierConfig(
                                activeHoursStart: nil, activeHoursEnd: nil,
                                imageCount: 2, unlimitedImages: false,
                                videoCount: mode == "neutral" ? 0 : 1)
                        } else {
                            tiers.removeValue(forKey: mode)
                        }
                    }
                })) {
                Text("\(modeEmoji(mode)) \(mode)")
                    .font(.caption)
                    .frame(width: 78, alignment: .leading)
            }
            .toggleStyle(.checkbox)
            .disabled(client.actionInFlight)

            if let tier = scheduler.tiers[mode] {
                Toggle("window", isOn: Binding(
                    get: { tier.activeHoursStart != nil },
                    set: { on in
                        putTiers { tiers in
                            tiers[mode]?.activeHoursStart = on ? "20:00" : nil
                            tiers[mode]?.activeHoursEnd = on ? "08:00" : nil
                        }
                    }))
                    .toggleStyle(.checkbox).font(.caption)
                    .disabled(client.actionInFlight)
                    .help("Off = this tier is eligible 24/7. Overnight windows wrap midnight; the most explicit open tier wins overlaps.")
                if tier.activeHoursStart != nil {
                    Picker("", selection: tierHourBinding(mode, isStart: true)) {
                        ForEach(0..<24, id: \.self) { Text(Self.hourLabel($0)).tag($0) }
                    }
                    .labelsHidden().frame(width: 86)
                    .disabled(client.actionInFlight)
                    Text("–").font(.caption).foregroundStyle(.tertiary)
                    Picker("", selection: tierHourBinding(mode, isStart: false)) {
                        ForEach(0..<24, id: \.self) { Text(Self.hourLabel($0)).tag($0) }
                    }
                    .labelsHidden().frame(width: 86)
                    .disabled(client.actionInFlight)
                }
                Stepper("img: \(tier.unlimitedImages ? "∞" : String(tier.imageCount))",
                        onIncrement: { putTiers { $0[mode]?.imageCount = tier.imageCount + 1 } },
                        onDecrement: { putTiers { $0[mode]?.imageCount = max(0, tier.imageCount - 1) } })
                    .font(.caption)
                    .disabled(client.actionInFlight || tier.unlimitedImages)
                Toggle("∞", isOn: Binding(
                    get: { tier.unlimitedImages },
                    set: { on in putTiers { $0[mode]?.unlimitedImages = on } }))
                    .toggleStyle(.checkbox).font(.caption)
                    .disabled(client.actionInFlight)
                    .help("Unlimited-within-cycle: image renders chain until the cycle (or this tier's window) closes.")
                if mode != "neutral" {
                    Stepper("vid: \(tier.videoCount)",
                            onIncrement: { putTiers { $0[mode]?.videoCount = tier.videoCount + 1 } },
                            onDecrement: { putTiers { $0[mode]?.videoCount = max(0, tier.videoCount - 1) } })
                        .font(.caption)
                        .disabled(client.actionInFlight)
                } else {
                    Text("stills only").font(.caption2).foregroundStyle(.tertiary)
                        .help("Neutral is the Autocord film stream — no video.")
                }
                Spacer(minLength: 0)
            } else {
                Text("off").font(.caption2).foregroundStyle(.tertiary)
                Spacer(minLength: 0)
            }
        }
    }

    /// Hour-grain binding for one end of a tier's window (writes the pair).
    private func tierHourBinding(_ mode: String, isStart: Bool) -> Binding<Int> {
        Binding(
            get: {
                let tier = client.scheduler?.tiers[mode]
                return Self.hour(from: isStart ? tier?.activeHoursStart : tier?.activeHoursEnd) ?? (isStart ? 20 : 8)
            },
            set: { h in
                putTiers { tiers in
                    guard var tier = tiers[mode] else { return }
                    var start = Self.hour(from: tier.activeHoursStart) ?? 20
                    var end = Self.hour(from: tier.activeHoursEnd) ?? 8
                    if isStart { start = h } else { end = h }
                    guard start != end else { return }   // zero-length → server 400; use the window toggle for 24/7
                    tier.activeHoursStart = String(format: "%02d:00", start)
                    tier.activeHoursEnd = String(format: "%02d:00", end)
                    tiers[mode] = tier
                }
            })
    }

    // ── Content-window helpers ──

    /// "HH:MM" → hour Int (minutes dropped; the pickers edit at hour grain).
    private static func hour(from hm: String?) -> Int? {
        guard let first = hm?.split(separator: ":").first, let h = Int(first) else { return nil }
        return (0...23).contains(h) ? h : nil
    }

    private static func hourLabel(_ h: Int) -> String {
        switch h {
        case 0: return "12 AM"
        case 12: return "12 PM"
        case 1...11: return "\(h) AM"
        default: return "\(h - 12) PM"
        }
    }

    /// Display label for a stored "HH:MM" (keeps minutes when non-zero).
    private static func windowLabel(_ hm: String) -> String {
        let bits = hm.split(separator: ":")
        guard let h = bits.first.flatMap({ Int($0) }) else { return hm }
        let minutes = bits.count > 1 ? String(bits[1]) : "00"
        let base = hourLabel(h)
        return minutes == "00" ? base : base.replacingOccurrences(of: " ", with: ":\(minutes) ")
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
        .overlay(alignment: .bottomTrailing) {
            // Taste verdicts (Inner Loop F3): ❤️/😐 feed her draw-weighting
            // store via POST /v1/kira/taste — she renders more of what lands.
            if let sent = client.tasteSent[item.path] {
                Text(sent == "up" ? "❤️" : "😐")
                    .font(.caption2)
                    .padding(3)
                    .background(.black.opacity(0.45), in: Capsule())
                    .padding(3)
            } else {
                HStack(spacing: 2) {
                    Button { Task { await client.sendTaste(path: item.path, verdict: "up") } }
                        label: { Text("❤️").font(.caption2) }
                    Button { Task { await client.sendTaste(path: item.path, verdict: "down") } }
                        label: { Text("😐").font(.caption2) }
                }
                .buttonStyle(.borderless)
                .padding(2)
                .background(.black.opacity(0.45), in: Capsule())
                .padding(3)
            }
        }
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
