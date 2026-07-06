// ServiceControlEditor.swift — Configure start/stop/restart for a service
//
// A watched service can be driven either by a local launchd label (the common
// case for local agents like com.barkadabrew.comfybox) or by explicit shell
// commands run locally or over SSH (for the home server / littleroundbox).

import SwiftUI

struct ServiceControlEditor: View {
    let service: WatchedService
    var onSave: (WatchedService) -> Void
    var onCancel: () -> Void

    enum Mode: String, CaseIterable, Identifiable {
        case none = "Monitor only"
        case launchd = "launchd (local)"
        case commands = "Commands / SSH"
        var id: String { rawValue }
    }

    @State private var mode: Mode
    @State private var launchdLabel: String
    @State private var sshHost: String
    @State private var startCommand: String
    @State private var stopCommand: String
    @State private var restartCommand: String

    init(service: WatchedService, onSave: @escaping (WatchedService) -> Void, onCancel: @escaping () -> Void) {
        self.service = service
        self.onSave = onSave
        self.onCancel = onCancel
        let c = service.control
        _launchdLabel = State(initialValue: c?.launchdLabel ?? "")
        _sshHost = State(initialValue: c?.sshHost ?? "")
        _startCommand = State(initialValue: c?.startCommand ?? "")
        _stopCommand = State(initialValue: c?.stopCommand ?? "")
        _restartCommand = State(initialValue: c?.restartCommand ?? "")
        if let c, c.isActionable {
            _mode = State(initialValue: c.launchdLabel != nil ? .launchd : .commands)
        } else {
            _mode = State(initialValue: .none)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Control · \(service.name)").font(.headline)

            Picker("Method", selection: $mode) {
                ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            switch mode {
            case .none:
                Text("No lifecycle control — this service is monitored only.")
                    .font(.caption).foregroundStyle(.secondary)
            case .launchd:
                field("launchd label") {
                    TextField("com.barkadabrew.comfybox", text: $launchdLabel).textFieldStyle(.roundedBorder)
                }
                Text("Restart → launchctl kickstart -k, Start → kickstart, Stop → kill SIGTERM (gui domain).")
                    .font(.caption2).foregroundStyle(.tertiary)
            case .commands:
                field("SSH host (optional, e.g. todd@10.0.100.232)") {
                    TextField("", text: $sshHost).textFieldStyle(.roundedBorder).autocorrectionDisabled()
                }
                field("Start command") { TextField("", text: $startCommand).textFieldStyle(.roundedBorder) }
                field("Stop command") { TextField("", text: $stopCommand).textFieldStyle(.roundedBorder) }
                field("Restart command (optional; else stop && start)") {
                    TextField("", text: $restartCommand).textFieldStyle(.roundedBorder)
                }
                Text("Commands run in a login shell; with an SSH host they run there.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }

            HStack {
                Spacer()
                Button("Cancel") { onCancel() }
                Button("Save") { save() }.buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
        .frame(width: 420)
    }

    @ViewBuilder private func field<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }

    private func save() {
        var updated = service
        switch mode {
        case .none:
            updated.control = nil
        case .launchd:
            let label = launchdLabel.trimmingCharacters(in: .whitespaces)
            updated.control = label.isEmpty ? nil : ServiceControl(launchdLabel: label)
        case .commands:
            func opt(_ s: String) -> String? {
                let t = s.trimmingCharacters(in: .whitespaces); return t.isEmpty ? nil : t
            }
            let control = ServiceControl(
                sshHost: opt(sshHost),
                startCommand: opt(startCommand),
                stopCommand: opt(stopCommand),
                restartCommand: opt(restartCommand))
            updated.control = control.isActionable ? control : nil
        }
        onSave(updated)
    }
}
