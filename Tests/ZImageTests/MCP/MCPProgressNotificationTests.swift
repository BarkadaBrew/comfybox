import XCTest

@testable import ZImage

/// comfybox#292 — MCP `notifications/progress` during renders.
///
/// The cadence and termination rules live in a PURE scheduler so they are
/// testable without a render, a server, or a clock. The polling wrapper is
/// tested for the one invariant intent.md makes non-negotiable: nothing it
/// starts outlives the request.
final class MCPProgressNotificationTests: XCTestCase {

  private typealias Snapshot = MCPProgressScheduler.Snapshot

  // MARK: - Cadence

  func testDefaultPollIntervalIsTwoSeconds() {
    XCTAssertEqual(MCPProgressScheduler.defaultInterval, 2.0)
  }

  func testFirstTickEmitsEvenAtZeroPercent() {
    let decision = MCPProgressScheduler.decide(
      elapsed: 2, lastEmitted: nil,
      snapshot: Snapshot(isRendering: true, progressPercent: 0, pending: 0, activeJobId: "J-1"))
    guard case .emit(let progress, let total, _) = decision else {
      return XCTFail("expected an emit, got \(decision)")
    }
    XCTAssertEqual(progress, 0)
    XCTAssertEqual(total, 100)
  }

  /// MCP requires the progress value to INCREASE with every notification for a
  /// given token. A percent that has not moved emits nothing rather than
  /// repeating (or, worse, going backwards).
  func testUnchangedPercentEmitsNothing() {
    let decision = MCPProgressScheduler.decide(
      elapsed: 6, lastEmitted: 40,
      snapshot: Snapshot(isRendering: true, progressPercent: 40, pending: 0, activeJobId: "J-1"))
    XCTAssertEqual(decision, .skip)
  }

  func testIncreasingPercentEmits() {
    let decision = MCPProgressScheduler.decide(
      elapsed: 8, lastEmitted: 40,
      snapshot: Snapshot(isRendering: true, progressPercent: 55, pending: 0, activeJobId: "J-1"))
    guard case .emit(let progress, _, _) = decision else {
      return XCTFail("expected an emit, got \(decision)")
    }
    XCTAssertEqual(progress, 55)
  }

  /// The queue snapshot reports the ACTIVE render, which can roll over to a
  /// different job mid-poll. A backwards percent must never be emitted.
  func testDecreasingPercentIsSuppressed() {
    let decision = MCPProgressScheduler.decide(
      elapsed: 10, lastEmitted: 80,
      snapshot: Snapshot(isRendering: true, progressPercent: 3, pending: 0, activeJobId: "J-2"))
    XCTAssertEqual(decision, .skip)
  }

  func testQueuedRenderReportsQueueDepthInTheMessage() {
    let decision = MCPProgressScheduler.decide(
      elapsed: 2, lastEmitted: nil,
      snapshot: Snapshot(isRendering: false, progressPercent: nil, pending: 3, activeJobId: nil))
    guard case .emit(let progress, _, let message) = decision else {
      return XCTFail("expected an emit, got \(decision)")
    }
    XCTAssertEqual(progress, 0)
    XCTAssertTrue(message.lowercased().contains("queue"), message)
    XCTAssertTrue(message.contains("3"), message)
  }

  func testRenderingMessageNamesThePercent() {
    let decision = MCPProgressScheduler.decide(
      elapsed: 4, lastEmitted: nil,
      snapshot: Snapshot(isRendering: true, progressPercent: 63, pending: 1, activeJobId: "J-1"))
    guard case .emit(_, _, let message) = decision else {
      return XCTFail("expected an emit, got \(decision)")
    }
    XCTAssertTrue(message.contains("63"), message)
  }

  // MARK: - Termination

  func testFailedPollSkipsRatherThanStops() {
    XCTAssertEqual(
      MCPProgressScheduler.decide(elapsed: 4, lastEmitted: 10, snapshot: nil), .skip)
  }

  func testPollingStopsAtTheDeadline() {
    let decision = MCPProgressScheduler.decide(
      elapsed: MCPProgressScheduler.maxDuration + 1, lastEmitted: 10,
      snapshot: Snapshot(isRendering: true, progressPercent: 90, pending: 0, activeJobId: "J-1"))
    guard case .stop = decision else { return XCTFail("expected a stop, got \(decision)") }
  }

  func testPercentIsClampedToTheZeroHundredRange() {
    let over = MCPProgressScheduler.decide(
      elapsed: 2, lastEmitted: nil,
      snapshot: Snapshot(isRendering: true, progressPercent: 140, pending: 0, activeJobId: "J-1"))
    guard case .emit(let progress, _, _) = over else {
      return XCTFail("expected an emit, got \(over)")
    }
    XCTAssertEqual(progress, 100)
  }

  // MARK: - Queue snapshot parsing

  func testSnapshotParsesTheLockBasedQueuePayload() throws {
    let data = try JSONSerialization.data(withJSONObject: [
      "is_rendering": true, "progress_percent": 45, "active_job_id": "J-1",
      "pending": [["id": "a"], ["id": "b"]],
    ] as [String: Any])
    let snapshot = try XCTUnwrap(MCPProgressScheduler.Snapshot(queuePayload: data))
    XCTAssertTrue(snapshot.isRendering)
    XCTAssertEqual(snapshot.progressPercent, 45)
    XCTAssertEqual(snapshot.pending, 2)
    XCTAssertEqual(snapshot.activeJobId, "J-1")
  }

  func testSnapshotOfGarbageIsNil() {
    XCTAssertNil(MCPProgressScheduler.Snapshot(queuePayload: Data("not json".utf8)))
  }

  // MARK: - The wrapper: no token, no polling; no task outlives the call

  private final class RecordingReporter: MCPProgressReporter, @unchecked Sendable {
    private let lock = NSLock()
    private var events: [Double] = []
    func report(progress: Double, total: Double?, message: String?) async {
      lock.lock()
      events.append(progress)
      lock.unlock()
    }
    var emitted: [Double] {
      lock.lock()
      defer { lock.unlock() }
      return events
    }
  }

  private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func bump() {
      lock.lock()
      n += 1
      lock.unlock()
    }
    var value: Int {
      lock.lock()
      defer { lock.unlock() }
      return n
    }
  }

  /// No `progressToken` in the request `_meta` → not one extra HTTP call.
  /// (The engine is shared with LM Studio; a poll nobody asked for is waste.)
  func testWithoutAReporterTheEngineIsNeverPolled() async throws {
    let polls = Counter()
    let value = try await MCPToolExecutor.withProgressNotifications(
      reporter: nil, interval: 0.001,
      poll: { polls.bump(); return nil }
    ) {
      try await Task.sleep(nanoseconds: 30_000_000)
      return 7
    }
    XCTAssertEqual(value, 7)
    XCTAssertEqual(polls.value, 0)
  }

  /// With a reporter the poller runs DURING the work and is cancelled and
  /// awaited before the call returns — intent.md: no orphaned work.
  func testPollerEmitsDuringWorkAndStopsWhenTheWorkReturns() async throws {
    let reporter = RecordingReporter()
    let percent = Counter()
    let value = try await MCPToolExecutor.withProgressNotifications(
      reporter: reporter, interval: 0.005,
      poll: {
        percent.bump()
        return Snapshot(
          isRendering: true, progressPercent: min(99, percent.value * 5), pending: 0,
          activeJobId: "J-1")
      }
    ) {
      try await Task.sleep(nanoseconds: 150_000_000)
      return "done"
    }
    XCTAssertEqual(value, "done")
    let duringWork = reporter.emitted
    XCTAssertGreaterThanOrEqual(duringWork.count, 2, "expected several progress notifications")
    XCTAssertEqual(duringWork, duringWork.sorted(), "progress must never go backwards")

    // Nothing kept running: after the call returned, no further emissions.
    try await Task.sleep(nanoseconds: 100_000_000)
    XCTAssertEqual(reporter.emitted.count, duringWork.count, "poller outlived the request")
  }

  func testWorkErrorStillStopsThePoller() async {
    struct Boom: Error {}
    let reporter = RecordingReporter()
    do {
      _ = try await MCPToolExecutor.withProgressNotifications(
        reporter: reporter, interval: 0.005,
        poll: { Snapshot(isRendering: true, progressPercent: 10, pending: 0, activeJobId: "J-1") }
      ) { () -> Int in
        try await Task.sleep(nanoseconds: 30_000_000)
        throw Boom()
      }
      XCTFail("expected the work error to propagate")
    } catch {
      XCTAssertTrue(error is Boom)
    }
    let after = reporter.emitted.count
    try? await Task.sleep(nanoseconds: 60_000_000)
    XCTAssertEqual(reporter.emitted.count, after, "poller outlived a failed request")
  }

  // MARK: - Wire format

  func testProgressNotificationMatchesTheMCPWireShape() throws {
    let json = MCPProgressNotification.json(
      token: AnyCodable("tok-1"), progress: 42, total: 100, message: "rendering — 42%")
    let params = try XCTUnwrap(json["params"] as? [String: Any])
    XCTAssertEqual(json["jsonrpc"] as? String, "2.0")
    XCTAssertEqual(json["method"] as? String, "notifications/progress")
    XCTAssertNil(json["id"], "a notification has no id")
    XCTAssertEqual(params["progressToken"] as? String, "tok-1")
    XCTAssertEqual(params["progress"] as? Double, 42)
    XCTAssertEqual(params["total"] as? Double, 100)
    XCTAssertEqual(params["message"] as? String, "rendering — 42%")
  }

  /// The token is echoed VERBATIM — MCP allows a string or an integer, and a
  /// client that sent 7 must not get "7" back.
  func testIntegerProgressTokenIsEchoedAsAnInteger() throws {
    let json = MCPProgressNotification.json(
      token: AnyCodable(7), progress: 1, total: nil, message: nil)
    let params = try XCTUnwrap(json["params"] as? [String: Any])
    XCTAssertEqual(params["progressToken"] as? Int, 7)
    XCTAssertNil(params["total"])
    XCTAssertNil(params["message"])
  }
}
