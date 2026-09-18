import Foundation
import XCTest

@testable import ZImage

/// Priority lanes, hold-until and the overnight window (FDD §4.10.9, WP23).
final class QueueLaneTests: XCTestCase {

  private struct Job {
    let name: String
    var schedule: QueueSchedule
    var runsWhilePaused = false
  }

  private func next(_ jobs: [Job], now: Date = Date(), paused: Bool = false) -> String? {
    QueueLaneScheduler.nextIndex(
      in: jobs, now: now, paused: paused,
      schedule: { $0.schedule }, runsWhilePaused: { $0.runsWhilePaused }
    ).map { jobs[$0].name }
  }

  private let window = BatchWindow(start: "23:00", end: "07:00", timeZone: "America/New_York")

  private func date(_ text: String) -> Date {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd HH:mm"
    formatter.timeZone = TimeZone(identifier: "America/New_York")
    return formatter.date(from: text)!
  }

  // MARK: selection

  func testInteractiveCutsTheLineAheadOfBatch() {
    let jobs = [
      Job(name: "sequence", schedule: QueueSchedule(lane: .batch)),
      Job(name: "image", schedule: QueueSchedule(lane: .interactive)),
    ]
    XCTAssertEqual(next(jobs), "image")
  }

  func testEachLaneKeepsSubmissionOrderWithinItself() {
    let jobs = [
      Job(name: "batch1", schedule: QueueSchedule(lane: .batch)),
      Job(name: "image1", schedule: QueueSchedule(lane: .interactive)),
      Job(name: "image2", schedule: QueueSchedule(lane: .interactive)),
      Job(name: "batch2", schedule: QueueSchedule(lane: .batch)),
    ]
    XCTAssertEqual(next(jobs), "image1")
    XCTAssertEqual(next(Array(jobs.dropFirst(2))), "image2")
    XCTAssertEqual(next([jobs[0], jobs[3]]), "batch1")
  }

  func testAHeldJobIsInvisibleUntilItsTime() {
    let now = date("2026-09-17 14:00")
    let jobs = [
      Job(name: "tonight", schedule: QueueSchedule(lane: .batch, holdUntil: date("2026-09-17 23:00")))
    ]
    XCTAssertNil(next(jobs, now: now), "an idle GPU does not release a held job")
    XCTAssertEqual(next(jobs, now: date("2026-09-17 23:00")), "tonight")
    XCTAssertEqual(next(jobs, now: date("2026-09-18 02:00")), "tonight")
  }

  func testAHeldBatchJobDoesNotBlockUnheldWork() {
    let now = date("2026-09-17 14:00")
    let jobs = [
      Job(name: "tonight", schedule: QueueSchedule(lane: .batch, holdUntil: date("2026-09-17 23:00"))),
      Job(name: "now", schedule: QueueSchedule(lane: .batch)),
    ]
    XCTAssertEqual(next(jobs, now: now), "now")
  }

  func testPauseStillOnlyRunsWhatMayRunWhilePaused() {
    let jobs = [
      Job(name: "image", schedule: QueueSchedule(lane: .interactive)),
      Job(name: "model-switch", schedule: QueueSchedule(lane: .interactive), runsWhilePaused: true),
    ]
    XCTAssertEqual(next(jobs, paused: true), "model-switch", "lanes do not override the pause carve-out")
    XCTAssertNil(next([jobs[0]], paused: true))
  }

  func testAnEmptyOrFullyHeldQueueHasNothingToRun() {
    XCTAssertNil(next([]))
    let held = [Job(name: "x", schedule: QueueSchedule(lane: .batch, holdUntil: .distantFuture))]
    XCTAssertNil(next(held))
  }

  // MARK: the window

  func testAWindowThatCrossesMidnightIsOpenOnBothSides() {
    XCTAssertTrue(window.isOpen(at: date("2026-09-17 23:30")))
    XCTAssertTrue(window.isOpen(at: date("2026-09-18 03:00")))
    XCTAssertTrue(window.isOpen(at: date("2026-09-18 06:59")))
    XCTAssertFalse(window.isOpen(at: date("2026-09-18 07:00")), "the end is exclusive")
    XCTAssertFalse(window.isOpen(at: date("2026-09-17 14:00")))
  }

  func testADaytimeWindowIsAlsoSupported() {
    let daytime = BatchWindow(start: "09:00", end: "17:00", timeZone: "America/New_York")
    XCTAssertTrue(daytime.isOpen(at: date("2026-09-17 12:00")))
    XCTAssertFalse(daytime.isOpen(at: date("2026-09-17 22:00")))
  }

  func testNextOpening() {
    XCTAssertEqual(window.nextOpening(after: date("2026-09-17 14:00")), date("2026-09-17 23:00"))
    // Already open: now.
    let inside = date("2026-09-18 02:00")
    XCTAssertEqual(window.nextOpening(after: inside), inside)
    // After the window closed this morning, tonight is still today.
    XCTAssertEqual(window.nextOpening(after: date("2026-09-18 08:00")), date("2026-09-18 23:00"))
  }

  func testNextClosingOnlyExistsWhileOpen() {
    XCTAssertEqual(window.nextClosing(after: date("2026-09-17 23:30")), date("2026-09-18 07:00"))
    XCTAssertNil(window.nextClosing(after: date("2026-09-17 14:00")))
  }

  func testAMalformedWindowIsInvalidRatherThanAlwaysOpen() {
    let bad = BatchWindow(start: "25:00", end: "07:00")
    XCTAssertFalse(bad.isValid)
    XCTAssertFalse(bad.isOpen(at: Date()))
    XCTAssertFalse(BatchWindow(start: "23:00", end: "23:00").isValid)
  }

  // MARK: dispositions

  func testResolveTurnsADispositionIntoAHoldTime() {
    let afternoon = date("2026-09-17 14:00")
    XCTAssertNil(QueueSchedule.resolve(lane: .batch, disposition: .now, window: window, now: afternoon))
    XCTAssertEqual(
      QueueSchedule.resolve(lane: .batch, disposition: .tonight, window: window, now: afternoon),
      date("2026-09-17 23:00"))
    XCTAssertEqual(
      QueueSchedule.resolve(
        lane: .batch, disposition: .at(date("2026-09-20 05:00")), window: window, now: afternoon),
      date("2026-09-20 05:00"))
    XCTAssertNil(
      QueueSchedule.resolve(
        lane: .batch, disposition: .at(date("2026-09-01 05:00")), window: window, now: afternoon),
      "a time in the past is eligible now, not held forever")
  }

  func testInteractiveWorkIsNeverHeld() {
    XCTAssertNil(
      QueueSchedule.resolve(
        lane: .interactive, disposition: .tonight, window: window, now: date("2026-09-17 14:00")))
  }

  func testDispositionParsing() {
    XCTAssertEqual(QueueDisposition(raw: "now", holdUntil: nil), .now)
    XCTAssertEqual(QueueDisposition(raw: "TONIGHT", holdUntil: nil), .tonight)
    let when = date("2026-09-20 05:00")
    XCTAssertEqual(QueueDisposition(raw: "at", holdUntil: when), .at(when))
    XCTAssertNil(QueueDisposition(raw: "at", holdUntil: nil), "'at' without a time is not a disposition")
    XCTAssertNil(QueueDisposition(raw: "whenever", holdUntil: nil))
  }

  // MARK: spill

  func testSpillDecidesWhatHappensWhenTheWindowCloses() {
    let morning = date("2026-09-18 07:00")
    XCTAssertNil(
      QueueLaneScheduler.holdAfterWindow(spill: .idle, window: window, now: morning),
      "idle keeps going, still below interactive work")
    XCTAssertEqual(
      QueueLaneScheduler.holdAfterWindow(spill: .wait, window: window, now: morning),
      date("2026-09-18 23:00"))
    XCTAssertEqual(
      QueueLaneScheduler.holdAfterWindow(spill: .strict, window: window, now: morning),
      .distantFuture, "strict stands down until someone changes its disposition")
  }

  // MARK: honest estimates

  func testANightHoldsAboutFourMinutesOfVideo() {
    // 23:00–07:00 is 8 h; at 25 min per chunk that is 19 chunks.
    XCTAssertEqual(QueueLaneScheduler.chunksPerNight(window: window), 19)
    // A 12-minute programme is ~60 chunks: four nights, not one.
    XCTAssertEqual(QueueLaneScheduler.nightsNeeded(chunks: 60, window: window), 4)
    XCTAssertEqual(QueueLaneScheduler.nightsNeeded(chunks: 3, window: window), 1)
    XCTAssertEqual(QueueLaneScheduler.nightsNeeded(chunks: 0, window: window), 0)
  }

  func testEstimatesDoNotDivideByZero() {
    let broken = BatchWindow(start: "bad", end: "07:00")
    XCTAssertEqual(QueueLaneScheduler.chunksPerNight(window: broken), 0)
    XCTAssertEqual(QueueLaneScheduler.nightsNeeded(chunks: 10, window: broken), 0)
  }

  // MARK: what a Director submit asks for

  private func payload(_ object: [String: Any]) throws -> WarmServer.DirectorPayload {
    let timeline: [String: Any] = [
      "version": 1,
      "settings": ["width": 576, "height": 896, "length_frames": 289],
      "global_prompt": "a barista", "keyframes": [], "prompt_segments": [], "audio_clips": [],
    ]
    var body = object
    body["timeline"] = timeline
    // The same decoder every route uses: snake_case in, camelCase properties.
    return try WarmServer.decode(
      WarmServer.DirectorPayload.self, from: try JSONSerialization.data(withJSONObject: body))
  }

  func testAMultiChunkSequenceIsBatchWorkByDefault() throws {
    let now = date("2026-09-17 14:00")
    let single = WarmServer.directorSchedule(
      payload: try payload([:]), chunkCount: 1, window: window, now: now)
    XCTAssertEqual(single.lane, .interactive, "a single clip is something a person waits for")
    XCTAssertNil(single.holdUntil)

    let sequence = WarmServer.directorSchedule(
      payload: try payload([:]), chunkCount: 3, window: window, now: now)
    XCTAssertEqual(sequence.lane, .batch, "a sequence fills idle GPU instead of blocking an image")
    XCTAssertNil(sequence.holdUntil, "batch still runs today unless it was asked to wait")
  }

  func testTonightHoldsUntilTheWindowOpens() throws {
    let schedule = WarmServer.directorSchedule(
      payload: try payload(["disposition": "tonight"]), chunkCount: 3, window: window,
      now: date("2026-09-17 14:00"))
    XCTAssertEqual(schedule.holdUntil, date("2026-09-17 23:00"))
    XCTAssertFalse(schedule.isEligible(at: date("2026-09-17 22:59")))
    XCTAssertTrue(schedule.isEligible(at: date("2026-09-17 23:00")))
  }

  func testAnExplicitTimeAndSpillAreHonoured() throws {
    let iso = ISO8601DateFormatter()
    let when = date("2026-09-20 05:00")
    let schedule = WarmServer.directorSchedule(
      payload: try payload([
        "disposition": "at", "hold_until": iso.string(from: when), "spill": "idle",
      ]),
      chunkCount: 3, window: window, now: date("2026-09-17 14:00"))
    XCTAssertEqual(schedule.holdUntil, when)
    XCTAssertEqual(schedule.spill, .idle)
  }

  func testAnExplicitLaneOverridesTheDefault() throws {
    let schedule = WarmServer.directorSchedule(
      payload: try payload(["lane": "interactive"]), chunkCount: 6, window: window,
      now: date("2026-09-17 14:00"))
    XCTAssertEqual(schedule.lane, .interactive, "the caller can say it is waiting")
  }

  func testNonsenseFallsBackRatherThanFailingTheSubmit() throws {
    let schedule = WarmServer.directorSchedule(
      payload: try payload(["lane": "urgent", "disposition": "eventually", "spill": "maybe"]),
      chunkCount: 3, window: window, now: date("2026-09-17 14:00"))
    XCTAssertEqual(schedule.lane, .batch)
    XCTAssertNil(schedule.holdUntil)
    XCTAssertEqual(schedule.spill, .wait)
  }

  // MARK: wire shape

  func testScheduleRoundTripsAsSnakeCase() throws {
    let schedule = QueueSchedule(lane: .batch, holdUntil: date("2026-09-17 23:00"), spill: .wait)
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let raw = try XCTUnwrap(String(data: try encoder.encode(schedule), encoding: .utf8))
    XCTAssertTrue(raw.contains("\"hold_until\""))
    XCTAssertTrue(raw.contains("\"batch\""))
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    XCTAssertEqual(try decoder.decode(QueueSchedule.self, from: try encoder.encode(schedule)), schedule)
  }
}
