// ServerHealth.swift — Consumer of coffeeshop-server's get_server_health
//
// Surfaces Littleroundbox (home server) health in the Health board. Per the
// 2026-07-06 handoff, the contract is one MCP tool `get_server_health`; this is
// the desktop consumer. The tool is being built on the server side, so the
// model is tolerant (every section optional) and the service degrades to a
// clear "not yet available" state. The one hard rule from the handoff:
// suppressed alerts are ALWAYS surfaced (a stopped container silently muted for
// 7 weeks is exactly what this must never let recur).

import Foundation

public struct ServerHealth: Decodable, Sendable, Equatable {
    public var summary: String? = nil
    public var system: ServerHealthSystem? = nil
    public var services: [ServerHealthUnit]? = nil
    public var containers: [ServerHealthContainer]? = nil
    public var macPipeline: [ServerHealthUnit]? = nil
    public var suppressedAlerts: [ServerHealthSuppressed]? = nil
    public var checkedAt: String? = nil

    enum CodingKeys: String, CodingKey {
        case summary, system, services, containers
        case macPipeline = "mac_pipeline"
        case suppressedAlerts = "suppressed_alerts"
        case checkedAt = "checked_at"
    }

    /// Suppressed alerts are always presented — even the empty case is a signal
    /// ("nothing is being hidden from you").
    public var suppressed: [ServerHealthSuppressed] { suppressedAlerts ?? [] }

    /// Containers that are stopped/unhealthy — the actionable set.
    public var problemContainers: [ServerHealthContainer] {
        (containers ?? []).filter { !$0.isRunning }
    }
}

public struct ServerHealthSystem: Decodable, Sendable, Equatable {
    public var disk: String?
    public var memory: String?
    public var load: String?
    public var uptime: String?
}

public struct ServerHealthUnit: Decodable, Sendable, Equatable, Identifiable {
    public var name: String
    public var state: String?
    public var detail: String?
    public var id: String { name }

    enum CodingKeys: String, CodingKey { case name, state, status, detail }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = (try? c.decode(String.self, forKey: .name)) ?? "unknown"
        state = (try? c.decodeIfPresent(String.self, forKey: .state))
            ?? (try? c.decodeIfPresent(String.self, forKey: .status)) ?? nil
        detail = try? c.decodeIfPresent(String.self, forKey: .detail)
    }

    public var isHealthy: Bool {
        guard let s = state?.lowercased() else { return false }
        return ["active", "running", "up", "healthy", "ok"].contains(s)
    }
}

public struct ServerHealthContainer: Decodable, Sendable, Equatable, Identifiable {
    public var name: String
    public var state: String?
    public var restartPolicy: String?
    public var suppressed: Bool?
    public var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, state, status
        case restartPolicy = "restart_policy"
        case suppressed
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = (try? c.decode(String.self, forKey: .name)) ?? "unknown"
        state = (try? c.decodeIfPresent(String.self, forKey: .state))
            ?? (try? c.decodeIfPresent(String.self, forKey: .status)) ?? nil
        restartPolicy = try? c.decodeIfPresent(String.self, forKey: .restartPolicy)
        suppressed = try? c.decodeIfPresent(Bool.self, forKey: .suppressed)
    }

    public var isRunning: Bool { (state?.lowercased() ?? "").contains("run") || (state?.lowercased() ?? "") == "up" }
    public var isSuppressed: Bool { suppressed == true }
    /// A stopped container without a restart policy is the 7-week-silent trap.
    public var lacksRestartPolicy: Bool {
        guard let p = restartPolicy?.lowercased() else { return false }
        return p.isEmpty || p == "no" || p == "none"
    }
}

public struct ServerHealthSuppressed: Decodable, Sendable, Equatable, Identifiable {
    public var alert: String
    public var reason: String?
    public var since: String?
    public var id: String { alert }

    enum CodingKeys: String, CodingKey { case alert, reason, since }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        alert = (try? c.decode(String.self, forKey: .alert)) ?? "alert"
        reason = try? c.decodeIfPresent(String.self, forKey: .reason)
        since = try? c.decodeIfPresent(String.self, forKey: .since)
    }
}
