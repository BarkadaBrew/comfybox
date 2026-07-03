// UptimeTracker.swift — Persistent per-service uptime & interruption history
//
// Every HealthMonitor check records one outcome per service into a daily
// bucket (ok / total / interruptions). Buckets are keyed by day string and
// persisted as a small JSON file, so uptime history survives app restarts
// and is honest about gaps: time the app wasn't watching simply isn't
// counted, rather than being guessed at.
//
// "Interruption" = a transition INTO .down from a non-down state — a service
// that stays down for fifty checks is one interruption, not fifty.

import Foundation

/// Aggregated stats for one service over a trailing window.
public struct UptimeStats: Equatable {
    public var okChecks: Int = 0
    public var totalChecks: Int = 0
    public var interruptions: Int = 0

    /// Percent of counted checks that were up (healthy or degraded).
    public var uptimePercent: Double {
        totalChecks > 0 ? Double(okChecks) / Double(totalChecks) * 100.0 : 0
    }
}

@MainActor
public final class UptimeTracker {
    /// One day's outcomes for one service.
    private struct DayBucket: Codable {
        var ok: Int = 0
        var total: Int = 0
        var interruptions: Int = 0
    }

    // serviceId -> dayKey ("2026-07-02") -> bucket
    private var buckets: [String: [String: DayBucket]] = [:]
    /// Last non-unknown state per service, for interruption edge detection.
    private var lastStates: [String: HealthState] = [:]

    private let path: URL
    private let calendar = Calendar.current

    /// `~/.comfybox/health-uptime.json`.
    public nonisolated static func defaultPath() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".comfybox/health-uptime.json")
    }

    public nonisolated init(path: URL = UptimeTracker.defaultPath()) {
        self.path = path
        if let data = try? Data(contentsOf: path),
           let loaded = try? JSONDecoder().decode([String: [String: DayBucket]].self, from: data) {
            buckets = loaded
        }
    }

    /// Record one check outcome. `.unknown` (invalid URL, no probe) is not
    /// counted toward uptime — it says nothing about the service.
    public func record(serviceId: String, state: HealthState, date: Date = Date()) {
        guard state != .unknown else { return }

        let key = Self.dayKey(for: date)
        var bucket = buckets[serviceId]?[key] ?? DayBucket()
        bucket.total += 1
        if state != .down {
            bucket.ok += 1
        } else if lastStates[serviceId] != .down {
            bucket.interruptions += 1
        }
        lastStates[serviceId] = state
        buckets[serviceId, default: [:]][key] = bucket
    }

    /// Stats over the trailing `days` days (including today).
    public func stats(serviceId: String, days: Int, now: Date = Date()) -> UptimeStats {
        guard let serviceBuckets = buckets[serviceId] else { return UptimeStats() }
        let keys = Set((0..<max(days, 0)).compactMap { offset -> String? in
            guard let day = calendar.date(byAdding: .day, value: -offset, to: now) else { return nil }
            return Self.dayKey(for: day)
        })
        var stats = UptimeStats()
        for (key, bucket) in serviceBuckets where keys.contains(key) {
            _ = key
            stats.okChecks += bucket.ok
            stats.totalChecks += bucket.total
            stats.interruptions += bucket.interruptions
        }
        return stats
    }

    /// Persist to disk, pruning history older than ~90 days to keep the file small.
    public func save(now: Date = Date()) {
        if let cutoffDate = calendar.date(byAdding: .day, value: -90, to: now) {
            let cutoff = Self.dayKey(for: cutoffDate)
            for (service, days) in buckets {
                buckets[service] = days.filter { $0.key >= cutoff }
            }
        }
        let dir = path.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try? encoder.encode(buckets) {
            try? data.write(to: path, options: .atomic)
        }
    }

    /// "2026-07-02" — sorts lexicographically in date order, which pruning relies on.
    private static func dayKey(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.timeZone = .current
        return formatter.string(from: date)
    }
}
