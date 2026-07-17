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
                    bindingCard
                    placeholderCard(
                        title: "Her Now · Her World · Agenda",
                        detail: "Mood, energy, arc, relationship scores, barkada, agenda — binds to GET /v1/kira/state (Workstream A, #1312).")
                    placeholderCard(
                        title: "Creation controls",
                        detail: "24/7 scheduler toggle, cadence, make-now, content mode — binds to /v1/kira/content-scheduler/* and /v1/kira/content-mode (#1313, #1314).")
                    placeholderCard(
                        title: "Compute",
                        detail: "Pool headroom, LTX-2 admission threshold, free-compute, bridge reconnect — binds to /v1/kira/compute (#1315).")
                    placeholderCard(
                        title: "Recent output",
                        detail: "Kira-scoped gallery strip with vision verdicts + deliver-to-Telegram — binds to /v1/kira/media/* (#1316).")
                    placeholderCard(
                        title: "Conversation",
                        detail: "Embedded comms over /ws/backroom with shared Telegram memory — Workstream A6 (#1317) + D6 (#245).")
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
