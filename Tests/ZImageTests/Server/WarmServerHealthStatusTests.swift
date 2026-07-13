import XCTest
@testable import ZImage

/// Unit tests for the pure /health status derivation used by the non-blocking
/// health endpoint (#217).
final class WarmServerHealthStatusTests: XCTestCase {
  private let stale = 300_000

  func testOkWhenIdle() {
    XCTAssertEqual(
      WarmServerHealthStatus.derive(shuttingDown: false, activeRenderAgeMs: nil, staleThresholdMs: stale),
      "ok")
  }

  func testOkDuringNormalRender() {
    XCTAssertEqual(
      WarmServerHealthStatus.derive(shuttingDown: false, activeRenderAgeMs: 120_000, staleThresholdMs: stale),
      "ok")
  }

  func testRenderStalePastThreshold() {
    XCTAssertEqual(
      WarmServerHealthStatus.derive(shuttingDown: false, activeRenderAgeMs: 300_001, staleThresholdMs: stale),
      "render_stale")
  }

  func testAtThresholdIsNotYetStale() {
    XCTAssertEqual(
      WarmServerHealthStatus.derive(shuttingDown: false, activeRenderAgeMs: 300_000, staleThresholdMs: stale),
      "ok")
  }

  func testShuttingDownWinsOverStale() {
    XCTAssertEqual(
      WarmServerHealthStatus.derive(shuttingDown: true, activeRenderAgeMs: 900_000, staleThresholdMs: stale),
      "shutting_down")
  }
}
