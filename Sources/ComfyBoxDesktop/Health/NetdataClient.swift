// NetdataClient.swift — littleroundbox server metrics via Netdata
//
// Fetches a filtered /api/v1/allmetrics snapshot from the Netdata agent on
// the home server (LAN-scoped ufw rule opened 2026-07-03) and reduces it to
// the handful of headline numbers the Health board shows. Parsing is a pure
// static function so the wire format is testable without a network.

import Foundation

/// Headline metrics for the littleroundbox server card.
public struct ServerMetrics: Sendable, Equatable {
    public var cpuPercent: Double?        // busy = 100 - idle
    public var load1: Double?
    public var ramUsedMiB: Double?
    public var ramTotalMiB: Double?
    public var diskUsedGiB: Double?
    public var diskAvailGiB: Double?
    public var netInKbps: Double?
    public var netOutKbps: Double?        // magnitude (netdata reports egress negative)
    public var uptimeSeconds: Double?

    public var ramUsedFraction: Double? {
        guard let used = ramUsedMiB, let total = ramTotalMiB, total > 0 else { return nil }
        return used / total
    }

    public var diskUsedFraction: Double? {
        guard let used = diskUsedGiB, let avail = diskAvailGiB, used + avail > 0 else { return nil }
        return used / (used + avail)
    }
}

public struct NetdataClient: Sendable {
    public let baseURL: URL
    private let timeout: TimeInterval

    public init(baseURL: URL, timeout: TimeInterval = 5.0) {
        self.baseURL = baseURL
        self.timeout = timeout
    }

    /// Fetch the filtered snapshot, or nil when the agent is unreachable.
    public func fetchMetrics() async -> ServerMetrics? {
        var components = URLComponents(url: baseURL.appendingPathComponent("api/v1/allmetrics"),
                                       resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "format", value: "json"),
            // Netdata "simple patterns" are SPACE-separated (a pipe silently
            // matches nothing and returns {}).
            URLQueryItem(name: "filter", value: "system.* disk_space.*"),
        ]
        guard let url = components?.url else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200
        else { return nil }
        return Self.parse(allMetrics: data)
    }

    /// Reduce a Netdata allmetrics JSON document to headline numbers.
    public static func parse(allMetrics data: Data) -> ServerMetrics? {
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return nil
        }

        func dimensions(_ chart: String) -> [String: Double] {
            guard let chartDict = root[chart] as? [String: Any],
                  let dims = chartDict["dimensions"] as? [String: Any] else { return [:] }
            var values: [String: Double] = [:]
            for (name, raw) in dims {
                if let dim = raw as? [String: Any], let value = dim["value"] as? Double {
                    values[name] = value
                }
            }
            return values
        }

        var metrics = ServerMetrics()

        let cpu = dimensions("system.cpu")
        if let idle = cpu["idle"] {
            metrics.cpuPercent = max(0, min(100, 100 - idle))
        }

        let load = dimensions("system.load")
        metrics.load1 = load["load1"]

        let ram = dimensions("system.ram")
        if !ram.isEmpty {
            let used = ram["used"] ?? 0
            let total = ram.values.reduce(0, +)  // free + used + cached + buffers
            metrics.ramUsedMiB = used
            metrics.ramTotalMiB = total > 0 ? total : nil
        }

        let disk = dimensions("disk_space./")
        metrics.diskUsedGiB = disk["used"]
        metrics.diskAvailGiB = disk["avail"]

        let net = dimensions("system.net")
        if let inbound = net["InOctets"] { metrics.netInKbps = abs(inbound) }
        if let outbound = net["OutOctets"] { metrics.netOutKbps = abs(outbound) }

        metrics.uptimeSeconds = dimensions("system.uptime")["uptime"]

        // A snapshot with nothing recognizable means the payload wasn't Netdata.
        let recognized: [Any?] = [
            metrics.cpuPercent, metrics.ramUsedMiB, metrics.diskUsedGiB, metrics.uptimeSeconds,
        ]
        return recognized.contains(where: { $0 != nil }) ? metrics : nil
    }
}
