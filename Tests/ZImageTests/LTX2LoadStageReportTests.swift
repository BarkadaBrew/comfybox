// LTX2LoadStageReportTests.swift — comfybox#340 load-path instrumentation.
//
// The #340 wedge was undiagnosable from the logs: `ensureLoaded` emitted
// "LTX-2: loading text encoder (Gemma 3 12B)…" and then nothing until the
// watchdog's SIGTERM ~15 minutes later, so a genuinely slow stage and a
// deadlock looked identical. `LTX2VideoGenerator.loadStageReport` is the pure
// core of the per-stage timing that replaced that single span; it is tested
// here so the wording and the slow/normal threshold are exercised without
// needing a live hang (or any model weights) to reproduce.

import XCTest

@testable import ZImage

final class LTX2LoadStageReportTests: XCTestCase {

  func testHealthyStageReportsElapsedAndIsNotFlaggedSlow() {
    let report = LTX2VideoGenerator.loadStageReport(
      stage: "text encoder materialize parameters (MLX.eval)", seconds: 4.2)

    XCTAssertFalse(report.isSlow, "a 4.2s stage is a normal warm load, not an incident")
    XCTAssertTrue(report.message.contains("4.20s"), report.message)
    XCTAssertTrue(
      report.message.contains("text encoder materialize parameters"),
      "the stage must be named so the log points at it: \(report.message)")
    XCTAssertFalse(report.message.contains("ABNORMALLY SLOW"), report.message)
  }

  func testStageAtOrOverThresholdIsFlaggedSlowAndSaysWhy() {
    let report = LTX2VideoGenerator.loadStageReport(
      stage: "text encoder bind weights", seconds: 870)

    XCTAssertTrue(report.isSlow, "870s (the observed 2026-09-01 wedge) must be flagged")
    XCTAssertTrue(report.message.contains("870.00s"), report.message)
    XCTAssertTrue(report.message.contains("ABNORMALLY SLOW"), report.message)
    XCTAssertTrue(
      report.message.contains("#340"),
      "the warning must carry the ticket so the next occurrence is greppable: \(report.message)")
  }

  /// Exactly at the threshold counts as slow — a stage that sits on the line is
  /// already far outside the 3–17s a healthy text-encoder load takes.
  func testThresholdIsInclusive() {
    let warnAfter = LTX2VideoGenerator.slowLoadStageWarnSeconds
    XCTAssertTrue(
      LTX2VideoGenerator.loadStageReport(stage: "s", seconds: warnAfter).isSlow,
      "a stage exactly at the threshold is slow")
    XCTAssertFalse(
      LTX2VideoGenerator.loadStageReport(stage: "s", seconds: warnAfter - 0.01).isSlow,
      "just under the threshold is still healthy")
  }

  func testThresholdIsWellClearOfHealthyLoadsAndWellUnderTheWatchdog() {
    // Production text-encoder loads ran 3–17s end to end; the watchdog that
    // killed the wedged renders fired at ~15 minutes.
    XCTAssertGreaterThan(
      LTX2VideoGenerator.slowLoadStageWarnSeconds, 17,
      "must not cry wolf on an ordinary cold start")
    XCTAssertLessThan(
      LTX2VideoGenerator.slowLoadStageWarnSeconds, 15 * 60,
      "must fire long before the watchdog kills the process and takes the log with it")
  }
}
