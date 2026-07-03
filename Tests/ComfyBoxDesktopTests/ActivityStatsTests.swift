// ActivityStatsTests.swift — Tests for render-activity summary logic

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("ActivityStats")
struct ActivityStatsTests {
    private let calendar = Calendar.current

    /// A fixed noon anchor so day math is stable regardless of run time.
    private var today: Date {
        calendar.date(bySettingHour: 12, minute: 0, second: 0, of: Date())!
    }

    private func daysAgo(_ n: Int, hour: Int = 12) -> Date {
        let day = calendar.date(byAdding: .day, value: -n, to: today)!
        return calendar.date(bySettingHour: hour, minute: 0, second: 0, of: day)!
    }

    @Test("empty timestamps produce an empty summary")
    func emptySummary() {
        let s = ActivityStats.summarize(timestamps: [], now: today)
        #expect(s.totalCount == 0)
        #expect(s.activeDays == 0)
        #expect(s.currentStreak == 0)
        #expect(s.longestStreak == 0)
        #expect(s.peakHour == nil)
    }

    @Test("counts, active days, and peak hour")
    func countsAndPeakHour() {
        let stamps = [
            daysAgo(0, hour: 9), daysAgo(0, hour: 14), daysAgo(0, hour: 14),
            daysAgo(2, hour: 14), daysAgo(2, hour: 20),
        ]
        let s = ActivityStats.summarize(timestamps: stamps, now: today)
        #expect(s.totalCount == 5)
        #expect(s.activeDays == 2)
        #expect(s.peakHour == 14)
    }

    @Test("current streak counts consecutive days ending today or yesterday")
    func currentStreak() {
        // today, -1, -2 active -> streak 3
        let s1 = ActivityStats.summarize(
            timestamps: [daysAgo(0), daysAgo(1), daysAgo(2)], now: today)
        #expect(s1.currentStreak == 3)

        // yesterday + day before, nothing today -> streak still alive at 2
        let s2 = ActivityStats.summarize(
            timestamps: [daysAgo(1), daysAgo(2)], now: today)
        #expect(s2.currentStreak == 2)

        // gap two days ago -> no current streak
        let s3 = ActivityStats.summarize(timestamps: [daysAgo(3)], now: today)
        #expect(s3.currentStreak == 0)
    }

    @Test("longest streak spans the full history")
    func longestStreak() {
        // 5-day run ending 10 days ago, plus a 2-day run now.
        var stamps: [Date] = []
        for n in 10...14 { stamps.append(daysAgo(n)) }
        stamps.append(daysAgo(0))
        stamps.append(daysAgo(1))
        let s = ActivityStats.summarize(timestamps: stamps, now: today)
        #expect(s.longestStreak == 5)
        #expect(s.currentStreak == 2)
    }

    @Test("dayCounts buckets by calendar day, newest last, dense over the range")
    func dayCounts() {
        let stamps = [daysAgo(0), daysAgo(0), daysAgo(3)]
        let cells = ActivityStats.dayCounts(timestamps: stamps, days: 5, now: today)
        #expect(cells.count == 5)
        #expect(cells.last?.count == 2)      // today
        #expect(cells[1].count == 1)         // 3 days ago
        #expect(cells[0].count == 0)         // 4 days ago, empty filler
    }
}

@Suite("UptimeTracker")
@MainActor
struct UptimeTrackerTests {
    private func tempPath() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("uptime-\(UUID().uuidString).json")
    }

    @Test("records checks and computes uptime percent")
    func uptimePercent() {
        let tracker = UptimeTracker(path: tempPath())
        for _ in 0..<9 { tracker.record(serviceId: "svc", state: .healthy) }
        tracker.record(serviceId: "svc", state: .down)

        let stats = tracker.stats(serviceId: "svc", days: 7)
        #expect(stats.totalChecks == 10)
        #expect(stats.okChecks == 9)
        #expect(abs(stats.uptimePercent - 90.0) < 0.01)
    }

    @Test("interruptions count transitions into down, not down checks")
    func interruptions() {
        let tracker = UptimeTracker(path: tempPath())
        tracker.record(serviceId: "svc", state: .healthy)
        tracker.record(serviceId: "svc", state: .down)   // 1st interruption
        tracker.record(serviceId: "svc", state: .down)   // still down, not a new one
        tracker.record(serviceId: "svc", state: .healthy)
        tracker.record(serviceId: "svc", state: .down)   // 2nd interruption

        let stats = tracker.stats(serviceId: "svc", days: 7)
        #expect(stats.interruptions == 2)
    }

    @Test("degraded counts as up for uptime, unknown is excluded")
    func degradedAndUnknown() {
        let tracker = UptimeTracker(path: tempPath())
        tracker.record(serviceId: "svc", state: .healthy)
        tracker.record(serviceId: "svc", state: .degraded)
        tracker.record(serviceId: "svc", state: .unknown)

        let stats = tracker.stats(serviceId: "svc", days: 7)
        #expect(stats.totalChecks == 2)   // unknown not counted
        #expect(stats.okChecks == 2)
        #expect(abs(stats.uptimePercent - 100.0) < 0.01)
    }

    @Test("persists across instances")
    func persistence() {
        let path = tempPath()
        let t1 = UptimeTracker(path: path)
        t1.record(serviceId: "svc", state: .healthy)
        t1.record(serviceId: "svc", state: .down)
        t1.save()

        let t2 = UptimeTracker(path: path)
        let stats = t2.stats(serviceId: "svc", days: 7)
        #expect(stats.totalChecks == 2)
        #expect(stats.interruptions == 1)
    }

    @Test("stats for an unknown service are zeroed")
    func unknownService() {
        let tracker = UptimeTracker(path: tempPath())
        let stats = tracker.stats(serviceId: "nope", days: 7)
        #expect(stats.totalChecks == 0)
        #expect(stats.uptimePercent == 0)
        #expect(stats.interruptions == 0)
    }
}
