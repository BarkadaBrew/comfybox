import Foundation
import XCTest

@testable import ZImage

/// Task #19 (specs/motion-tab-prompt-lab.md rev 2, findings #1–3): render
/// traces as append-only lifecycle events keyed by a stable render_id,
/// carrying schema_version + task_kind from day one. Terminal-only writes
/// can't record crashes; events + recovery can.
final class RenderTraceStoreTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("traces-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  func testLifecycleEventsAppendAndReadBack() throws {
    let store = RenderTraceStore(directory: tempDir)
    let renderId = "r-123"

    store.append(RenderTraceEvent(
      renderId: renderId, event: .submitted, taskKind: .videoRender,
      payload: ["prompt": "a cat", "source": "test"]))
    store.append(RenderTraceEvent(
      renderId: renderId, event: .started, taskKind: .videoRender, payload: [:]))
    store.append(RenderTraceEvent(
      renderId: renderId, event: .terminal, taskKind: .videoRender,
      payload: ["status": "succeeded", "output_path": "/tmp/x.mp4"]))
    store.flush()

    let events = store.events(renderId: renderId)
    XCTAssertEqual(events.map(\.event), [.submitted, .started, .terminal])
    XCTAssertEqual(events[0].schemaVersion, 1)
    XCTAssertEqual(events[0].taskKind, .videoRender)
    XCTAssertEqual(events[0].payload["prompt"], "a cat")
  }

  func testUnfinishedTracesMarkedAbandonedOnRecovery() throws {
    let store = RenderTraceStore(directory: tempDir)
    store.append(RenderTraceEvent(
      renderId: "crashed-render", event: .submitted, taskKind: .videoRender, payload: [:]))
    store.append(RenderTraceEvent(
      renderId: "crashed-render", event: .started, taskKind: .videoRender, payload: [:]))
    store.append(RenderTraceEvent(
      renderId: "finished-render", event: .submitted, taskKind: .videoRender, payload: [:]))
    store.append(RenderTraceEvent(
      renderId: "finished-render", event: .terminal, taskKind: .videoRender,
      payload: ["status": "succeeded"]))
    store.flush()

    // A new store over the same directory = process restart.
    let recovered = RenderTraceStore(directory: tempDir)
    let abandonedCount = recovered.markAbandonedOpenTraces()
    XCTAssertEqual(abandonedCount, 1, "only the trace with no terminal event")

    let events = recovered.events(renderId: "crashed-render")
    XCTAssertEqual(events.last?.event, .abandoned)
    XCTAssertEqual(recovered.events(renderId: "finished-render").count, 2, "finished trace untouched")
  }

  func testConcurrentAppendsAllSurvive() throws {
    let store = RenderTraceStore(directory: tempDir)
    DispatchQueue.concurrentPerform(iterations: 50) { i in
      store.append(RenderTraceEvent(
        renderId: "burst", event: .submitted, taskKind: .imageRender,
        payload: ["i": "\(i)"]))
    }
    store.flush()
    XCTAssertEqual(store.events(renderId: "burst").count, 50, "single serialized writer loses nothing")
  }

  /// comfybox#328 (Codex round 1, finding 2): `GET /v1/video/traces` returns
  /// `TraceSummary`, not the raw submitted payload — a field WarmServer only
  /// stuffs into the payload dict never actually reaches that endpoint's
  /// response unless `TraceSummary` also declares and copies it. Pins that
  /// `enhancement_skipped`/`beat_schedule_ignored` survive the trip.
  func testRecentSummariesSurfacesEnhancementAndBeatScheduleMarkers() throws {
    let store = RenderTraceStore(directory: tempDir)
    store.append(RenderTraceEvent(
      renderId: "r-beats", event: .submitted, taskKind: .videoRender,
      payload: ["prompt": "she walks closer", "enhancement_skipped": "beat_schedule"]))
    store.append(RenderTraceEvent(
      renderId: "r-i2v-beats", event: .submitted, taskKind: .videoRender,
      payload: ["prompt": "a portrait", "beat_schedule_ignored": "i2v_unsupported"]))
    store.append(RenderTraceEvent(
      renderId: "r-plain", event: .submitted, taskKind: .videoRender,
      payload: ["prompt": "no beats here"]))
    store.flush()

    let summaries = Dictionary(uniqueKeysWithValues: store.recentSummaries(limit: 10).map { ($0.renderId, $0) })
    XCTAssertEqual(summaries["r-beats"]?.enhancementSkipped, "beat_schedule")
    XCTAssertNil(summaries["r-beats"]?.beatScheduleIgnored)
    XCTAssertEqual(summaries["r-i2v-beats"]?.beatScheduleIgnored, "i2v_unsupported")
    XCTAssertNil(summaries["r-i2v-beats"]?.enhancementSkipped)
    XCTAssertNil(summaries["r-plain"]?.enhancementSkipped)
    XCTAssertNil(summaries["r-plain"]?.beatScheduleIgnored)
  }

  /// comfybox#307 (review r2, item 1): the exact bug class #328 fixed for
  /// `enhancement_skipped` — `refine_skipped` is written into a trace
  /// event's payload (`VideoJobTracker.markSucceeded`), but `GET
  /// /v1/video/traces` returns `TraceSummary`, which whitelists fields.
  /// Unlike enhancement/beat markers (known at submit time), the refine
  /// outcome is known only at completion, so this reads from the TERMINAL
  /// event, not `submitted`.
  func testRecentSummariesSurfacesRefineSkippedFromTheTerminalEvent() throws {
    let store = RenderTraceStore(directory: tempDir)
    store.append(RenderTraceEvent(
      renderId: "r-refine-skip", event: .submitted, taskKind: .videoRender,
      payload: ["prompt": "12s 480p two-stage"]))
    store.append(RenderTraceEvent(
      renderId: "r-refine-skip", event: .terminal, taskKind: .videoRender,
      payload: [
        "status": "succeeded",
        "refine_skipped": "volume_gate (pre-refine volume 30000 > refine_max_vol 26000)",
      ]))
    store.append(RenderTraceEvent(
      renderId: "r-refine-ran", event: .submitted, taskKind: .videoRender,
      payload: ["prompt": "fits comfortably"]))
    store.append(RenderTraceEvent(
      renderId: "r-refine-ran", event: .terminal, taskKind: .videoRender,
      payload: ["status": "succeeded"]))
    store.flush()

    let summaries = Dictionary(uniqueKeysWithValues: store.recentSummaries(limit: 10).map { ($0.renderId, $0) })
    XCTAssertEqual(
      summaries["r-refine-skip"]?.refineSkipped,
      "volume_gate (pre-refine volume 30000 > refine_max_vol 26000)")
    XCTAssertNil(summaries["r-refine-ran"]?.refineSkipped, "refine ran — omitted, not a string \"null\"")
  }

  /// A `refine_skipped` key on the SUBMITTED event (not terminal) must be
  /// ignored — it can only be known once the render finishes, so a stray key
  /// there (a caller echoing request shape, say) must not leak through.
  func testRefineSkippedOnSubmittedEventIsIgnored() throws {
    let store = RenderTraceStore(directory: tempDir)
    store.append(RenderTraceEvent(
      renderId: "r-premature", event: .submitted, taskKind: .videoRender,
      payload: ["prompt": "x", "refine_skipped": "should not surface"]))
    store.append(RenderTraceEvent(
      renderId: "r-premature", event: .terminal, taskKind: .videoRender,
      payload: ["status": "succeeded"]))
    store.flush()

    let summaries = Dictionary(uniqueKeysWithValues: store.recentSummaries(limit: 10).map { ($0.renderId, $0) })
    XCTAssertNil(summaries["r-premature"]?.refineSkipped)
  }

  func testJSONLLinesAreSelfDescribing() throws {
    let store = RenderTraceStore(directory: tempDir)
    store.append(RenderTraceEvent(
      renderId: "r-9", event: .submitted, taskKind: .videoRender, payload: [:]))
    store.flush()

    let files = try FileManager.default.contentsOfDirectory(atPath: tempDir.path)
      .filter { $0.hasSuffix(".jsonl") }
    XCTAssertEqual(files.count, 1)
    let line = try String(contentsOf: tempDir.appendingPathComponent(files[0]), encoding: .utf8)
      .split(separator: "\n").first.map(String.init) ?? ""
    let obj = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
    XCTAssertEqual(obj?["schema_version"] as? Int, 1)
    XCTAssertEqual(obj?["task_kind"] as? String, "video_render")
    XCTAssertEqual(obj?["render_id"] as? String, "r-9")
    XCTAssertNotNil(obj?["ts"], "timestamped")
  }
}
