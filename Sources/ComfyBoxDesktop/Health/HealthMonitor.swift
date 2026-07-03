// HealthMonitor.swift — Polls watched services and host memory
//
// The single health feed behind the Health board. Probes each
// WatchedService, keeps the latest ServiceHealth per service, and
// records an event ONLY on state transitions so a self-healing retry
// does not spam the timeline (PRD: transient retry must not page).
// Also samples host unified-memory pressure each check.

import Foundation

@Observable
@MainActor
public final class HealthMonitor {
    // MARK: - Published State

    public var services: [ServiceHealth] = []
    public var events: [HealthEvent] = []
    public var isMonitoring: Bool = false
    public var lastRefresh: Date?

    /// Host unified memory: fraction used (0-1) and totals in GB.
    public var memoryUsedFraction: Double?
    public var memoryUsedGB: Double?
    public var memoryTotalGB: Double?

    /// Wider local-Mac metrics (CPU, load, disk, uptime), sampled each check.
    public var hostMetrics: HostMetrics?
    private let hostSampler = HostMetricsSampler()

    /// Endpoints to watch. Changing this takes effect on the next check.
    public var watchedServices: [WatchedService] = []

    /// Persistent uptime / interruption history, one record per check.
    let uptime: UptimeTracker

    // MARK: - Configuration

    public static let maxEvents = 100
    private let pollInterval: TimeInterval
    private let probe: HealthProbe
    private var pollTask: Task<Void, Never>?
    private var previousStates: [String: HealthState] = [:]

    public init(
        probe: HealthProbe = URLHealthProbe(),
        pollInterval: TimeInterval = 10.0,
        uptime: UptimeTracker = UptimeTracker()
    ) {
        self.probe = probe
        self.pollInterval = pollInterval
        self.uptime = uptime
    }

    /// Worst state across all watched services.
    public var overallState: HealthState {
        services.map(\.state).max(by: { $0.severityRank < $1.severityRank }) ?? .unknown
    }

    // MARK: - Monitoring

    public func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.checkNow()
                let interval = self?.pollInterval ?? 10
                try? await Task.sleep(for: .seconds(interval))
            }
        }
    }

    public func stopMonitoring() {
        pollTask?.cancel()
        pollTask = nil
        isMonitoring = false
    }

    /// Probe every watched service once and refresh host memory stats.
    public func checkNow() async {
        let targets = watchedServices
        let checkedAt = Date()

        var results: [String: ProbeResult] = [:]
        await withTaskGroup(of: (String, ProbeResult).self) { group in
            for service in targets {
                guard let url = URL(string: service.urlString) else {
                    results[service.id] = ProbeResult(
                        state: .unknown, latencyMs: nil, detail: "invalid URL"
                    )
                    continue
                }
                let probe = self.probe
                group.addTask {
                    (service.id, await probe.probe(url))
                }
            }
            for await (id, result) in group {
                results[id] = result
            }
        }

        var updated: [ServiceHealth] = []
        for service in targets {
            let result = results[service.id]
                ?? ProbeResult(state: .unknown, latencyMs: nil, detail: nil)
            updated.append(ServiceHealth(
                service: service,
                state: result.state,
                latencyMs: result.latencyMs,
                detail: result.detail,
                lastChecked: checkedAt
            ))
            recordTransition(for: service, to: result.state, detail: result.detail)
            uptime.record(serviceId: service.id, state: result.state, date: checkedAt)
        }
        if !targets.isEmpty { uptime.save() }

        // Drop transition memory for services no longer watched.
        let watchedIds = Set(targets.map(\.id))
        previousStates = previousStates.filter { watchedIds.contains($0.key) }

        services = updated
        sampleHostMemory()
        hostMetrics = hostSampler.sample()
        lastRefresh = checkedAt
    }

    // MARK: - Events

    /// Append an event only when the service's state actually changed.
    private func recordTransition(for service: WatchedService, to state: HealthState, detail: String?) {
        let previous = previousStates[service.id]
        previousStates[service.id] = state
        guard previous != state else { return }

        let severity: HealthEvent.Severity
        switch state {
        case .healthy: severity = .info
        case .degraded: severity = .warning
        case .down: severity = .error
        case .unknown: severity = .warning
        }

        var message: String
        if previous == nil {
            message = "\(service.name) is \(state.label.lowercased())"
        } else {
            message = "\(service.name) went \(state.label.lowercased())"
        }
        if let detail, !detail.isEmpty {
            message += " (\(detail))"
        }

        events.insert(
            HealthEvent(severity: severity, source: service.name, message: message),
            at: 0
        )
        if events.count > Self.maxEvents {
            events.removeLast(events.count - Self.maxEvents)
        }
    }

    // MARK: - Host Memory

    /// Sample host memory usage via Mach host statistics.
    private func sampleHostMemory() {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        guard total > 0, let free = Self.freeMemoryBytes() else {
            memoryUsedFraction = nil
            memoryUsedGB = nil
            memoryTotalGB = nil
            return
        }
        let used = max(0, total - free)
        memoryUsedFraction = used / total
        memoryUsedGB = used / 1_073_741_824
        memoryTotalGB = total / 1_073_741_824
    }

    /// Free + inactive + purgeable pages, in bytes, or nil on failure.
    private static func freeMemoryBytes() -> Double? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pageSize = Double(vm_kernel_page_size)
        let reclaimable = Double(stats.free_count)
            + Double(stats.inactive_count)
            + Double(stats.purgeable_count)
        return reclaimable * pageSize
    }
}
