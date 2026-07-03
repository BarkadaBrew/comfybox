// HostMetrics.swift — Local Mac metrics for the Health board
//
// CPU busy % (from host_statistics tick deltas between polls), 1-minute
// load, root-volume disk space, and system uptime. The tick-delta math is
// a pure function so it's testable; the samplers are thin Mach/Foundation
// calls.

import Foundation

/// Cumulative CPU ticks since boot (host_statistics HOST_CPU_LOAD_INFO).
public struct CPUTicks: Sendable, Equatable {
    public var user: UInt64
    public var system: UInt64
    public var idle: UInt64
    public var nice: UInt64

    public init(user: UInt64, system: UInt64, idle: UInt64, nice: UInt64) {
        self.user = user
        self.system = system
        self.idle = idle
        self.nice = nice
    }

    /// Busy percentage over the interval between two cumulative samples.
    /// Returns nil when the interval is empty (same sample, or counter reset).
    public static func busyPercent(from older: CPUTicks, to newer: CPUTicks) -> Double? {
        let busy = (newer.user &- older.user) &+ (newer.system &- older.system) &+ (newer.nice &- older.nice)
        let idle = newer.idle &- older.idle
        let total = busy &+ idle
        guard total > 0, total < UInt64(Int64.max) else { return nil }
        return Double(busy) / Double(total) * 100.0
    }
}

/// Snapshot of local host metrics beyond memory (which HealthMonitor
/// already samples).
public struct HostMetrics: Sendable, Equatable {
    public var cpuPercent: Double?
    public var load1: Double?
    public var diskFreeGB: Double?
    public var diskTotalGB: Double?
    public var uptimeSeconds: Double?

    public var diskUsedFraction: Double? {
        guard let free = diskFreeGB, let total = diskTotalGB, total > 0 else { return nil }
        return (total - free) / total
    }
}

/// Samples local metrics; holds the previous CPU tick sample so successive
/// polls yield an interval percentage.
@MainActor
public final class HostMetricsSampler {
    private var previousTicks: CPUTicks?

    public init() {}

    public func sample() -> HostMetrics {
        var metrics = HostMetrics()

        if let ticks = Self.readCPUTicks() {
            if let previous = previousTicks {
                metrics.cpuPercent = CPUTicks.busyPercent(from: previous, to: ticks)
            }
            previousTicks = ticks
        }

        var loads = [Double](repeating: 0, count: 3)
        if getloadavg(&loads, 3) >= 1 {
            metrics.load1 = loads[0]
        }

        if let attrs = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()),
           let free = attrs[.systemFreeSize] as? NSNumber,
           let total = attrs[.systemSize] as? NSNumber {
            metrics.diskFreeGB = free.doubleValue / 1_000_000_000
            metrics.diskTotalGB = total.doubleValue / 1_000_000_000
        }

        metrics.uptimeSeconds = ProcessInfo.processInfo.systemUptime

        return metrics
    }

    private static func readCPUTicks() -> CPUTicks? {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return CPUTicks(
            user: UInt64(info.cpu_ticks.0),
            system: UInt64(info.cpu_ticks.1),
            idle: UInt64(info.cpu_ticks.2),
            nice: UInt64(info.cpu_ticks.3)
        )
    }
}
