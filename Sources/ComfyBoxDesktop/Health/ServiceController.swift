// ServiceController.swift — Start/stop/restart Coffeeshop suite services
//
// The Health board monitors; this makes it act. Local launchd agents are
// driven with `launchctl` on the current GUI domain; remote or custom services
// run shell commands (optionally over SSH). Command construction is pure and
// unit-tested; execution runs a short-lived process and returns combined
// output.

import Foundation

public enum ServiceAction: String, Sendable {
    case start, stop, restart
}

public struct ServiceController: Sendable {
    public init() {}

    public enum ControlError: Error, LocalizedError {
        case noControl
        case unsupported(ServiceAction)
        case failed(Int32, String)

        public var errorDescription: String? {
            switch self {
            case .noControl: return "No control is configured for this service."
            case .unsupported(let a): return "This service has no \(a.rawValue) command."
            case .failed(let code, let out): return "Exit \(code): \(out.isEmpty ? "(no output)" : out)"
            }
        }
    }

    // MARK: - Pure command construction (tested)

    /// The argv for a launchctl action on the current GUI domain.
    /// `uid` is normally the current user's; injected for testability.
    public static func launchctlArgs(_ action: ServiceAction, label: String, uid: uid_t) -> [String] {
        let target = "gui/\(uid)/\(label)"
        switch action {
        case .restart: return ["kickstart", "-k", target]
        case .start:   return ["kickstart", target]
        case .stop:    return ["kill", "SIGTERM", target]
        }
    }

    /// The shell command string for a custom (non-launchd) control action,
    /// or nil if that action isn't defined. Wrapped for SSH when `sshHost` set.
    public static func customCommand(_ action: ServiceAction, control: ServiceControl) -> String? {
        let raw: String?
        switch action {
        case .start: raw = control.startCommand
        case .stop: raw = control.stopCommand
        case .restart:
            // Fall back to stop && start when no explicit restart is given.
            if let r = control.restartCommand { raw = r }
            else if let stop = control.stopCommand, let start = control.startCommand {
                raw = "\(stop) && \(start)"
            } else { raw = control.startCommand ?? control.stopCommand }
        }
        guard let command = raw, !command.isEmpty else { return nil }
        if let host = control.sshHost, !host.isEmpty {
            return "ssh \(host) \(shellQuote(command))"
        }
        return command
    }

    /// Single-quote a string for safe embedding in a shell command.
    public static func shellQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    // MARK: - Execution

    /// Perform an action on a service. Returns combined stdout+stderr.
    @discardableResult
    public func perform(_ action: ServiceAction, on service: WatchedService) async throws -> String {
        guard let control = service.control, control.isActionable else {
            throw ControlError.noControl
        }
        if let label = control.launchdLabel, !label.isEmpty {
            let args = Self.launchctlArgs(action, label: label, uid: getuid())
            return try await runProcess("/bin/launchctl", args)
        }
        guard let command = Self.customCommand(action, control: control) else {
            throw ControlError.unsupported(action)
        }
        return try await runProcess("/bin/zsh", ["-lc", command])
    }

    private func runProcess(_ launchPath: String, _ args: [String]) async throws -> String {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: launchPath)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        let output = String(data: data, encoding: .utf8) ?? ""
        if proc.terminationStatus != 0 {
            throw ControlError.failed(proc.terminationStatus, output.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
