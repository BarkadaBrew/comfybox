// MetricsHistory.swift — Time-series health samples for in-app trends
//
// "Datadog, but local, with history." HealthMonitor appends a sample each
// check; the Health board charts them (CPU, disk, service availability) over a
// selectable range. Persisted to ~/.comfybox/metrics-history.json, capped.
// Downsampling is pure and unit-tested.

import Foundation

public struct MetricsSample: Codable, Sendable, Equatable, Identifiable {
    public var date: Date
    public var cpuPercent: Double?
    public var diskUsedPercent: Double?
    public var servicesUp: Int
    public var servicesTotal: Int
    public var id: Date { date }

    public init(date: Date, cpuPercent: Double? = nil, diskUsedPercent: Double? = nil,
                servicesUp: Int = 0, servicesTotal: Int = 0) {
        self.date = date
        self.cpuPercent = cpuPercent
        self.diskUsedPercent = diskUsedPercent
        self.servicesUp = servicesUp
        self.servicesTotal = servicesTotal
    }

    public var serviceUpPercent: Double {
        servicesTotal > 0 ? Double(servicesUp) / Double(servicesTotal) * 100 : 0
    }
}

@Observable
@MainActor
public final class MetricsHistory {
    public private(set) var samples: [MetricsSample] = []
    private let cap = 4000

    public init() { load() }

    public func record(_ sample: MetricsSample) {
        samples.append(sample)
        if samples.count > cap { samples.removeFirst(samples.count - cap) }
        save()
    }

    /// Samples within the trailing window (nil days = all).
    public func samples(days: Int?) -> [MetricsSample] {
        guard let days else { return samples }
        let cutoff = Date().addingTimeInterval(-Double(days) * 86_400)
        return samples.filter { $0.date >= cutoff }
    }

    // MARK: - Pure downsampling (tested)

    /// Average samples into at most `maxPoints` time-ordered buckets, preserving
    /// the overall shape. Fewer than maxPoints samples pass through unchanged.
    public nonisolated static func downsample(_ input: [MetricsSample], maxPoints: Int) -> [MetricsSample] {
        guard maxPoints > 0, input.count > maxPoints else { return input }
        let sorted = input.sorted { $0.date < $1.date }
        guard let first = sorted.first?.date, let last = sorted.last?.date, last > first else { return sorted }
        let span = last.timeIntervalSince(first)
        let bucketSize = span / Double(maxPoints)

        var buckets: [Int: [MetricsSample]] = [:]
        for s in sorted {
            let idx = min(maxPoints - 1, Int(s.date.timeIntervalSince(first) / bucketSize))
            buckets[idx, default: []].append(s)
        }
        return buckets.keys.sorted().map { idx -> MetricsSample in
            let group = buckets[idx]!
            let mid = group[group.count / 2].date
            return MetricsSample(
                date: mid,
                cpuPercent: average(group.compactMap { $0.cpuPercent }),
                diskUsedPercent: average(group.compactMap { $0.diskUsedPercent }),
                servicesUp: Int(average(group.map { Double($0.servicesUp) }) ?? 0),
                servicesTotal: group.map { $0.servicesTotal }.max() ?? 0)
        }
    }

    private nonisolated static func average(_ xs: [Double]) -> Double? {
        xs.isEmpty ? nil : xs.reduce(0, +) / Double(xs.count)
    }

    // MARK: - Persistence

    private static var path: String {
        let dir = NSString(string: "~/.comfybox").expandingTildeInPath
        return (dir as NSString).appendingPathComponent("metrics-history.json")
    }

    private func load() {
        guard let data = FileManager.default.contents(atPath: Self.path) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        samples = (try? decoder.decode([MetricsSample].self, from: data)) ?? []
    }

    private func save() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(samples) else { return }
        let dir = (Self.path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try? data.write(to: URL(fileURLWithPath: Self.path))
    }
}
