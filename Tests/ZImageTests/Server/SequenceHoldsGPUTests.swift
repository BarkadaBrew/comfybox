import Foundation
import XCTest

@testable import ZImage

/// A sequence holds the GPU for HOURS, and says so (Todd 2026-09-19:
/// "sequences lease the GPU for the entire time and should suspend the
/// preselected ticks").
///
/// `/health.video.active_jobs` cannot carry this: a clip is minutes and a
/// sequence is hours (the 30 s demo ran 38-42 min PER CHUNK), and a periodic
/// producer needs to tell them apart. Kira's 24/7 cycle suspends its ticks on
/// `sequence_active`; without that it keeps booking work behind a two-hour job
/// and the queue fills faster than it drains — the 2026-08 backlog, 20/hr in
/// against 3.9/hr out.
final class SequenceHoldsGPUTests: XCTestCase {

  private func plan(chunks: Int) -> DirectorPlan {
    DirectorPlan(
      lengthFrames: 737, fps: 24, width: 576, height: 896, audioMode: "imported",
      chunks: (0..<chunks).map {
        .init(index: $0, startFrame: $0 * 248, endFrame: $0 * 248 + 248, frames: 249,
              seed: nil, carryOver: $0 > 0, audio: "imported")
      },
      keyframeTicks: [], boundaryFrames: [], warnings: [])
  }

  func testAMultiChunkJobCountsAsASequence() {
    let tracker = VideoJobTracker()
    XCTAssertEqual(tracker.activeSequenceCount, 0)
    _ = tracker.register(source: "test", mode: .i2v, plan: plan(chunks: 3), stageCount: 3)
    XCTAssertEqual(tracker.activeSequenceCount, 1, "three chunks is a sequence")
    XCTAssertEqual(tracker.activeSequenceProgress?.of, 3)
  }

  func testASINGLECHUNKClipIsNotASequence() {
    // The distinction that matters: a one-chunk render is minutes, and the
    // 24/7 cycle must NOT suspend itself for it.
    let tracker = VideoJobTracker()
    _ = tracker.register(source: "test", mode: .i2v, plan: plan(chunks: 1), stageCount: 1)
    XCTAssertEqual(tracker.activeSequenceCount, 0)
    XCTAssertEqual(tracker.activeJobCount, 1, "it is still an active job, just not a sequence")
  }

  func testAPlAINVideoJobIsNotASequence() {
    let tracker = VideoJobTracker()
    _ = tracker.register(source: "test", mode: .i2v)
    XCTAssertEqual(tracker.activeSequenceCount, 0, "no plan at all is not a sequence")
    XCTAssertEqual(tracker.activeJobCount, 1)
  }

  func testAFINISHEDSequenceReleasesTheSignal() throws {
    // The suspend must end when the render does, not when the job ages out of
    // the tracker's TTL — otherwise one sequence silences the cycle for an
    // hour after it finished.
    let tracker = VideoJobTracker()
    let (jobId, _) = tracker.register(
      source: "test", mode: .i2v, plan: plan(chunks: 3), stageCount: 3)
    XCTAssertEqual(tracker.activeSequenceCount, 1)
    tracker.markFailed(jobId, error: WarmServerError.invalidRequest(message: "stop"))
    XCTAssertEqual(
      tracker.activeSequenceCount, 0,
      "a completed sequence stops holding the cycle down, even while still queryable")
  }

  func testProgressReportsTheOLDESTSequenceNotWhicheverTheDictionaryYields() throws {
    // `jobs` is a dictionary. Reporting `.first` would name a different
    // sequence run to run once two overlap, so a watcher's "2 of 3" could jump
    // backwards. Oldest wins — it is the one closest to done.
    let tracker = VideoJobTracker()
    let (first, _) = tracker.register(
      source: "test", mode: .i2v, plan: plan(chunks: 5), stageCount: 5)
    tracker.setStage(first, index: 3, count: 5)
    // A later sequence with a different chunk count, so the two are tellable apart.
    let (second, _) = tracker.register(
      source: "test", mode: .i2v, plan: plan(chunks: 2), stageCount: 2)
    tracker.setStage(second, index: 0, count: 2)

    XCTAssertEqual(tracker.activeSequenceCount, 2)
    let progress = try XCTUnwrap(tracker.activeSequenceProgress)
    XCTAssertEqual(progress.of, 5, "the OLDER sequence is reported")
    XCTAssertEqual(progress.stage, 3)
  }
}
