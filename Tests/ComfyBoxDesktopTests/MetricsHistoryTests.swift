import XCTest
@testable import ComfyBoxDesktop

final class MetricsHistoryTests: XCTestCase {

    private func series(_ n: Int, start: Date, step: TimeInterval = 60) -> [MetricsSample] {
        (0..<n).map { i in
            MetricsSample(date: start.addingTimeInterval(Double(i) * step),
                          cpuPercent: Double(i % 100), diskUsedPercent: 50,
                          servicesUp: 2, servicesTotal: 3)
        }
    }

    func testDownsampleNoOpWhenUnderLimit() {
        let base = Date(timeIntervalSince1970: 1_000_000)
        let s = series(5, start: base)
        XCTAssertEqual(MetricsHistory.downsample(s, maxPoints: 10).count, 5)
    }

    func testDownsampleReducesToBuckets() {
        let base = Date(timeIntervalSince1970: 1_000_000)
        let s = series(1000, start: base)
        let out = MetricsHistory.downsample(s, maxPoints: 50)
        XCTAssertLessThanOrEqual(out.count, 50)
        XCTAssertGreaterThan(out.count, 0)
        // Time-ordered.
        XCTAssertEqual(out, out.sorted { $0.date < $1.date })
    }

    func testDownsampleAveragesWithinBucket() {
        let base = Date(timeIntervalSince1970: 0)
        // Two samples 0s and 60s, cpu 10 and 30 → one bucket avg 20.
        let s = [
            MetricsSample(date: base, cpuPercent: 10),
            MetricsSample(date: base.addingTimeInterval(60), cpuPercent: 30),
        ]
        let out = MetricsHistory.downsample(s, maxPoints: 1)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out.first?.cpuPercent ?? 0, 20, accuracy: 0.001)
    }

    func testServiceUpPercent() {
        let d = Date(timeIntervalSince1970: 0)
        XCTAssertEqual(MetricsSample(date: d, servicesUp: 3, servicesTotal: 4).serviceUpPercent, 75, accuracy: 0.001)
        XCTAssertEqual(MetricsSample(date: d, servicesUp: 0, servicesTotal: 0).serviceUpPercent, 0)
    }

    func testDownsamplePreservesTotalAsMax() {
        let base = Date(timeIntervalSince1970: 0)
        let s = [
            MetricsSample(date: base, servicesUp: 1, servicesTotal: 2),
            MetricsSample(date: base.addingTimeInterval(30), servicesUp: 3, servicesTotal: 4),
        ]
        let out = MetricsHistory.downsample(s, maxPoints: 1)
        XCTAssertEqual(out.first?.servicesTotal, 4)
    }
}
