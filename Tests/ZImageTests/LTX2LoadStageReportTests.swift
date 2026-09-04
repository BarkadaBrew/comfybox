// LTX2LoadStageReportTests.swift — comfybox#340 load-path instrumentation.
//
// The #340 wedge was undiagnosable from the logs: `LTX2VideoGenerator.load`
// emitted "LTX-2: loading text encoder (Gemma 3 12B)…" and then nothing until
// the watchdog's SIGTERM ~15 minutes later, so a genuinely slow stage and a
// deadlock looked identical.
//
// Codex review r1 caught that timing a stage and reporting it AFTERWARDS still
// leaves a killed load with no stage name — the report never runs. So the
// instrumentation is three messages, not one: an ENTRY line before the body
// (the line that survives a SIGTERM), a repeating STILL RUNNING heartbeat while
// the body is in flight, and a completion or failure line after. All four
// formatters are pure and tested here; no model weights, no live hang.

import XCTest

@testable import ZImage

final class LTX2LoadStageReportTests: XCTestCase {

  private let stage = "text encoder materialize parameters (MLX.eval)"

  // MARK: entry — the line that must survive a watchdog kill

  func testEntryMessageNamesTheStageBeforeItRuns() {
    let message = LTX2VideoGenerator.loadStageEntryMessage(stage: stage)

    XCTAssertTrue(
      message.contains(stage),
      "the entry line is the last thing a SIGTERM'd load leaves behind, so it must name the "
        + "active stage: \(message)")
    XCTAssertTrue(message.contains("started"), message)
  }

  // MARK: heartbeat — visible BEFORE the kill, not only in hindsight

  func testStillRunningMessageNamesStageAndElapsed() {
    let message = LTX2VideoGenerator.loadStageStillRunningMessage(stage: stage, seconds: 60)

    XCTAssertTrue(message.contains(stage), message)
    XCTAssertTrue(message.contains("60"), "the heartbeat must say how long it has been: \(message)")
    XCTAssertTrue(message.contains("STILL RUNNING"), message)
    XCTAssertTrue(
      message.contains("#340"),
      "the heartbeat must be greppable by ticket: \(message)")
  }

  /// Codex r1: the earlier warning claimed `/health` stalls with the load. It
  /// does not — `/health` reads a lock-backed snapshot precisely so it stays
  /// answerable during a blocking render (WarmServer, #217). Saying otherwise
  /// would send the next person to look at the wrong signal, and the whole
  /// point of these strings is to be read during an incident.
  func testStalledSubjectIsTheRenderQueueAndHealthIsCalledOutAsUnaffected() {
    let heartbeat = LTX2VideoGenerator.loadStageStillRunningMessage(stage: stage, seconds: 60)
    let slow = LTX2VideoGenerator.loadStageReport(stage: stage, seconds: 870).message

    for message in [heartbeat, slow] {
      XCTAssertTrue(
        message.contains("render queue"),
        "the stalled thing is the render queue: \(message)")
      XCTAssertTrue(
        message.contains("/health"),
        "must say /health stays up, so nobody trusts a green health check here: \(message)")
      XCTAssertFalse(
        message.contains("/health and every queued job stall"),
        "the retracted claim must not survive anywhere: \(message)")
    }
  }

  // MARK: completion

  func testHealthyStageReportsElapsedAndIsNotFlaggedSlow() {
    let report = LTX2VideoGenerator.loadStageReport(stage: stage, seconds: 4.2)

    XCTAssertFalse(report.isSlow, "a 4.2s stage is a normal warm load, not an incident")
    XCTAssertTrue(report.message.contains("4.20s"), report.message)
    XCTAssertTrue(
      report.message.contains("text encoder materialize parameters"),
      "the stage must be named so the log points at it: \(report.message)")
    XCTAssertFalse(report.message.contains("ABNORMALLY SLOW"), report.message)
  }

  func testStageAtOrOverThresholdIsFlaggedSlowAndSaysWhy() {
    let report = LTX2VideoGenerator.loadStageReport(stage: "text encoder bind weights", seconds: 870)

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

  /// The threshold governs the heartbeat interval, which is what actually fires
  /// before the watchdog. Production text-encoder loads ran 3–17s end to end;
  /// the watchdog that killed the wedged renders fired at ~15 minutes.
  func testThresholdIsWellClearOfHealthyLoadsAndWellUnderTheWatchdog() {
    XCTAssertGreaterThan(
      LTX2VideoGenerator.slowLoadStageWarnSeconds, 17,
      "must not cry wolf on an ordinary cold start")
    XCTAssertLessThan(
      LTX2VideoGenerator.slowLoadStageWarnSeconds, 15 * 60,
      "the heartbeat must fire many times before the watchdog kills the process")
  }

  // MARK: failure

  func testFailureMessageNamesStageElapsedErrorAndTheDiscard() {
    struct Boom: Error, CustomStringConvertible { var description: String { "tokenizer.json unreadable" } }

    let message = LTX2VideoGenerator.loadStageFailureMessage(
      stage: "tokenizer load (tokenizer.json parse)", seconds: 1.5, error: Boom())

    XCTAssertTrue(message.contains("tokenizer load"), message)
    XCTAssertTrue(message.contains("1.50s"), message)
    XCTAssertTrue(message.contains("tokenizer.json unreadable"), "the cause must be logged: \(message)")
    XCTAssertTrue(
      message.contains("discard"),
      "must say the partial load was dropped, so a retry is understood to start clean: \(message)")
  }
}
