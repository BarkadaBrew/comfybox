// HostMetricsTests.swift — Host CPU tick math and sampler sanity

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("HostMetrics")
struct HostMetricsTests {
    @Test("busy percent from tick deltas")
    func busyPercent() {
        let older = CPUTicks(user: 100, system: 50, idle: 850, nice: 0)
        let newer = CPUTicks(user: 130, system: 60, idle: 900, nice: 10)
        // busy delta = 30+10+10 = 50, idle delta = 50, total = 100 → 50%
        let pct = CPUTicks.busyPercent(from: older, to: newer)
        #expect(pct != nil)
        #expect(abs((pct ?? 0) - 50.0) < 0.001)
    }

    @Test("identical samples yield nil (no interval)")
    func emptyInterval() {
        let ticks = CPUTicks(user: 1, system: 1, idle: 1, nice: 0)
        #expect(CPUTicks.busyPercent(from: ticks, to: ticks) == nil)
    }

    @Test("disk used fraction")
    func diskFraction() {
        var metrics = HostMetrics()
        metrics.diskFreeGB = 250
        metrics.diskTotalGB = 1000
        #expect(abs((metrics.diskUsedFraction ?? 0) - 0.75) < 0.001)
    }

    @Test("sampler returns plausible live values, CPU after two samples")
    @MainActor
    func liveSampler() {
        let sampler = HostMetricsSampler()
        let first = sampler.sample()
        #expect(first.cpuPercent == nil)          // needs a prior sample
        #expect((first.load1 ?? -1) >= 0)
        #expect((first.diskTotalGB ?? 0) > 100)   // this Mac is > 100 GB
        #expect((first.uptimeSeconds ?? 0) > 60)

        let second = sampler.sample()
        if let cpu = second.cpuPercent {          // may be nil if ticks unchanged
            #expect(cpu >= 0 && cpu <= 100)
        }
    }
}
