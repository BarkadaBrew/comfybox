// HealthModels.swift — Models for the local health board
//
// Data types behind the "Datadog, but local" health board (see
// coffeeshop-dashboard-prd). A WatchedService is a user-configured
// HTTP endpoint; HealthMonitor probes each one and derives a
// HealthState with latency and detail. UI-free so the logic is
// testable without a view harness.

import Foundation

/// Health of a single service (or the whole stack — worst-of).
public enum HealthState: String, Sendable, CaseIterable {
    case healthy
    case degraded
    case down
    case unknown

    /// Severity ordering for worst-of aggregation (higher is worse).
    var severityRank: Int {
        switch self {
        case .healthy: return 0
        case .unknown: return 1
        case .degraded: return 2
        case .down: return 3
        }
    }

    public var label: String {
        switch self {
        case .healthy: return "Healthy"
        case .degraded: return "Degraded"
        case .down: return "Down"
        case .unknown: return "Unknown"
        }
    }

    public var systemImage: String {
        switch self {
        case .healthy: return "checkmark.circle.fill"
        case .degraded: return "exclamationmark.triangle.fill"
        case .down: return "xmark.octagon.fill"
        case .unknown: return "questionmark.circle"
        }
    }

    /// Latency above this is considered degraded even on a 2xx response.
    public static let slowLatencyMs = 2000

    /// Derive a state from an HTTP response. Connection failures never get
    /// here — the probe reports those as .down directly.
    public static func fromHTTP(statusCode: Int, latencyMs: Int) -> HealthState {
        guard (200..<300).contains(statusCode) else { return .degraded }
        return latencyMs >= slowLatencyMs ? .degraded : .healthy
    }
}

/// A user-configured endpoint to watch (persisted in desktop-config.json).
public struct WatchedService: Codable, Identifiable, Sendable, Equatable, Hashable {
    public var id: String
    public var name: String
    public var urlString: String
    /// Optional lifecycle control (start/stop/restart). nil = monitor-only.
    public var control: ServiceControl?

    public init(id: String = UUID().uuidString, name: String, urlString: String,
                control: ServiceControl? = nil) {
        self.id = id
        self.name = name
        self.urlString = urlString
        self.control = control
    }
}

/// How to start/stop/restart a watched service. Either a local launchd agent
/// (by label) or shell commands run locally or over SSH.
public struct ServiceControl: Codable, Sendable, Equatable, Hashable {
    /// Local launchd label, e.g. "com.barkadabrew.comfybox". When set, the
    /// controller uses `launchctl` on the current GUI domain.
    public var launchdLabel: String?
    /// SSH target "user@host" for remote services. When set, commands run there.
    public var sshHost: String?
    /// Explicit shell commands (used when launchdLabel is nil).
    public var startCommand: String?
    public var stopCommand: String?
    public var restartCommand: String?

    public init(launchdLabel: String? = nil, sshHost: String? = nil,
                startCommand: String? = nil, stopCommand: String? = nil,
                restartCommand: String? = nil) {
        self.launchdLabel = launchdLabel
        self.sshHost = sshHost
        self.startCommand = startCommand
        self.stopCommand = stopCommand
        self.restartCommand = restartCommand
    }

    /// True if any control action is possible.
    public var isActionable: Bool {
        launchdLabel != nil || startCommand != nil || stopCommand != nil || restartCommand != nil
    }
}

/// Latest probe outcome for one watched service.
public struct ServiceHealth: Identifiable, Sendable, Equatable {
    public var service: WatchedService
    public var state: HealthState
    public var latencyMs: Int?
    public var detail: String?
    public var lastChecked: Date?

    public var id: String { service.id }
}

/// One entry in the health event timeline (state transitions only).
public struct HealthEvent: Identifiable, Sendable, Equatable {
    public enum Severity: Sendable {
        case info
        case warning
        case error
    }

    public let id: String
    public let timestamp: Date
    public let severity: Severity
    public let source: String
    public let message: String

    public init(
        id: String = UUID().uuidString,
        timestamp: Date = Date(),
        severity: Severity,
        source: String,
        message: String
    ) {
        self.id = id
        self.timestamp = timestamp
        self.severity = severity
        self.source = source
        self.message = message
    }
}

/// Raw outcome of probing one URL.
public struct ProbeResult: Sendable, Equatable {
    public let state: HealthState
    public let latencyMs: Int?
    public let detail: String?

    public init(state: HealthState, latencyMs: Int?, detail: String?) {
        self.state = state
        self.latencyMs = latencyMs
        self.detail = detail
    }
}

/// Abstraction over the HTTP check so HealthMonitor is testable.
public protocol HealthProbe: Sendable {
    func probe(_ url: URL) async -> ProbeResult
}

/// Real probe: GET the URL with a short timeout; map response/failure to a state.
public struct URLHealthProbe: HealthProbe {
    private let timeout: TimeInterval

    public init(timeout: TimeInterval = 3.0) {
        self.timeout = timeout
    }

    public func probe(_ url: URL) async -> ProbeResult {
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.httpMethod = "GET"

        let start = Date()
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let latencyMs = Int(Date().timeIntervalSince(start) * 1000)
            guard let http = response as? HTTPURLResponse else {
                return ProbeResult(state: .degraded, latencyMs: latencyMs, detail: "non-HTTP response")
            }
            let state = HealthState.fromHTTP(statusCode: http.statusCode, latencyMs: latencyMs)
            let detail = state == .healthy ? nil : "HTTP \(http.statusCode)"
            return ProbeResult(state: state, latencyMs: latencyMs, detail: detail)
        } catch {
            return ProbeResult(state: .down, latencyMs: nil, detail: error.localizedDescription)
        }
    }
}
