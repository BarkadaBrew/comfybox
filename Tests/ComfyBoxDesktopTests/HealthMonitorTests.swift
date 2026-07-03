// HealthMonitorTests.swift — Tests for the health board's monitor core

import Testing
import Foundation
@testable import ComfyBoxDesktop

// MARK: - Fake probe

/// A HealthProbe returning canned results per URL, switchable between checks.
private final class FakeProbe: HealthProbe, @unchecked Sendable {
    var results: [String: ProbeResult] = [:]

    func probe(_ url: URL) async -> ProbeResult {
        results[url.absoluteString]
            ?? ProbeResult(state: .down, latencyMs: nil, detail: "no canned result")
    }
}

@Suite("HealthState")
struct HealthStateTests {
    @Test("2xx with fast latency is healthy")
    func fastOK() {
        #expect(HealthState.fromHTTP(statusCode: 200, latencyMs: 42) == .healthy)
    }

    @Test("2xx with slow latency is degraded")
    func slowOK() {
        #expect(HealthState.fromHTTP(statusCode: 200, latencyMs: 2500) == .degraded)
    }

    @Test("non-2xx status is degraded")
    func errorStatus() {
        #expect(HealthState.fromHTTP(statusCode: 500, latencyMs: 10) == .degraded)
        #expect(HealthState.fromHTTP(statusCode: 404, latencyMs: 10) == .degraded)
    }
}

@Suite("HealthMonitor")
struct HealthMonitorTests {
    private static let serviceA = WatchedService(name: "Service A", urlString: "http://localhost:1111/health")
    private static let serviceB = WatchedService(name: "Service B", urlString: "http://localhost:2222/health")

    @Test("checkNow populates service health from the probe")
    @MainActor
    func populatesServices() async {
        let probe = FakeProbe()
        probe.results[Self.serviceA.urlString] = ProbeResult(state: .healthy, latencyMs: 12, detail: nil)
        probe.results[Self.serviceB.urlString] = ProbeResult(state: .down, latencyMs: nil, detail: "connection refused")

        let monitor = HealthMonitor(probe: probe, uptime: UptimeTracker(path: FileManager.default.temporaryDirectory.appendingPathComponent("uptime-\(UUID().uuidString).json")))
        monitor.watchedServices = [Self.serviceA, Self.serviceB]
        await monitor.checkNow()

        #expect(monitor.services.count == 2)
        let a = monitor.services.first { $0.service.id == Self.serviceA.id }
        let b = monitor.services.first { $0.service.id == Self.serviceB.id }
        #expect(a?.state == .healthy)
        #expect(a?.latencyMs == 12)
        #expect(b?.state == .down)
        #expect(b?.detail == "connection refused")
    }

    @Test("a state transition records exactly one event")
    @MainActor
    func transitionEvent() async {
        let probe = FakeProbe()
        probe.results[Self.serviceA.urlString] = ProbeResult(state: .healthy, latencyMs: 10, detail: nil)
        let monitor = HealthMonitor(probe: probe, uptime: UptimeTracker(path: FileManager.default.temporaryDirectory.appendingPathComponent("uptime-\(UUID().uuidString).json")))
        monitor.watchedServices = [Self.serviceA]

        await monitor.checkNow()   // unknown -> healthy: one event
        #expect(monitor.events.count == 1)
        #expect(monitor.events.first?.severity == .info)

        // Same state again: no new event (flap suppression).
        await monitor.checkNow()
        await monitor.checkNow()
        #expect(monitor.events.count == 1)

        // Down transition: one error event.
        probe.results[Self.serviceA.urlString] = ProbeResult(state: .down, latencyMs: nil, detail: "timeout")
        await monitor.checkNow()
        #expect(monitor.events.count == 2)
        #expect(monitor.events.first?.severity == .error)

        // Recovery: one info event.
        probe.results[Self.serviceA.urlString] = ProbeResult(state: .healthy, latencyMs: 9, detail: nil)
        await monitor.checkNow()
        #expect(monitor.events.count == 3)
        #expect(monitor.events.first?.severity == .info)
        #expect(monitor.events.first?.message.contains("healthy") == true)
    }

    @Test("event log is capped")
    @MainActor
    func eventCap() async {
        let probe = FakeProbe()
        let monitor = HealthMonitor(probe: probe, uptime: UptimeTracker(path: FileManager.default.temporaryDirectory.appendingPathComponent("uptime-\(UUID().uuidString).json")))
        monitor.watchedServices = [Self.serviceA]

        // Flip state every check to force an event per check.
        for i in 0..<(HealthMonitor.maxEvents + 20) {
            probe.results[Self.serviceA.urlString] = i.isMultiple(of: 2)
                ? ProbeResult(state: .healthy, latencyMs: 5, detail: nil)
                : ProbeResult(state: .down, latencyMs: nil, detail: "flap")
            await monitor.checkNow()
        }
        #expect(monitor.events.count == HealthMonitor.maxEvents)
    }

    @Test("overall state is the worst of all services")
    @MainActor
    func overallState() async {
        let probe = FakeProbe()
        probe.results[Self.serviceA.urlString] = ProbeResult(state: .healthy, latencyMs: 5, detail: nil)
        probe.results[Self.serviceB.urlString] = ProbeResult(state: .degraded, latencyMs: 3000, detail: nil)
        let monitor = HealthMonitor(probe: probe, uptime: UptimeTracker(path: FileManager.default.temporaryDirectory.appendingPathComponent("uptime-\(UUID().uuidString).json")))
        monitor.watchedServices = [Self.serviceA, Self.serviceB]
        await monitor.checkNow()
        #expect(monitor.overallState == .degraded)

        probe.results[Self.serviceB.urlString] = ProbeResult(state: .down, latencyMs: nil, detail: nil)
        await monitor.checkNow()
        #expect(monitor.overallState == .down)
    }

    @Test("removing a watched service drops its health entry")
    @MainActor
    func removeService() async {
        let probe = FakeProbe()
        probe.results[Self.serviceA.urlString] = ProbeResult(state: .healthy, latencyMs: 5, detail: nil)
        probe.results[Self.serviceB.urlString] = ProbeResult(state: .healthy, latencyMs: 5, detail: nil)
        let monitor = HealthMonitor(probe: probe, uptime: UptimeTracker(path: FileManager.default.temporaryDirectory.appendingPathComponent("uptime-\(UUID().uuidString).json")))
        monitor.watchedServices = [Self.serviceA, Self.serviceB]
        await monitor.checkNow()
        #expect(monitor.services.count == 2)

        monitor.watchedServices = [Self.serviceA]
        await monitor.checkNow()
        #expect(monitor.services.count == 1)
        #expect(monitor.services.first?.service.id == Self.serviceA.id)
    }

    @Test("WatchedService round-trips through Codable")
    func watchedServiceCodable() throws {
        let service = WatchedService(name: "LM Studio", urlString: "http://127.0.0.1:1234/v1/models")
        let data = try JSONEncoder().encode(service)
        let decoded = try JSONDecoder().decode(WatchedService.self, from: data)
        #expect(decoded == service)
    }
}
