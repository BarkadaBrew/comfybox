// VideoTracePayloadTests.swift — comfybox#405 review round 2, item 3
//
// The sync `/v1/video/generate` route used to write NO trace at all: a sync
// render was invisible to `GET /v1/video/traces` and to every winner action.
// Ruling 5 fixed that but shipped with zero tests, which is how the async
// route ended up with `resolved_width` and the sync route without it in the
// first place.
//
// These drive the EXACT statics both routes call — `videoTracePayload` for the
// submitted event and `videoMeasuredOutputPayload` for the terminal one, plus
// the three `recordSyncVideo*` seams the sync route uses — against a real
// `RenderTraceStore` pointed at a temp directory, and assert the submitted +
// terminal pair on the success AND the failure path.

import XCTest
@testable import ZImage

final class VideoTracePayloadTests: XCTestCase {

  private var tempDir: URL!
  private var store: RenderTraceStore!

  override func setUpWithError() throws {
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("comfybox-405-traces-\(UUID().uuidString)")
    store = RenderTraceStore(directory: tempDir)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  // MARK: - Fixtures

  /// A `PreparedLocalVideo` with no weights involved: `LTX2VideoGenerator.init`
  /// only stores its configuration (loading happens inside `generate`), so this
  /// is inert.
  private func makePrep(
    width: Int = 512, height: Int = 320,
    predicted: (Int, Int) = (768, 480),
    stage1: (Int, Int)? = (512, 320),
    reason: VideoDimensionReason = .sourceAspect,
    initImage: String? = "/tmp/kira-405.png"
  ) -> WarmServer.PreparedLocalVideo {
    let request = LTX2VideoRequest(
      prompt: "a test clip",
      initImagePath: initImage,
      width: width, height: height,
      outputPath: "/tmp/comfybox-405.mp4")
    let generator = LTX2VideoGenerator(
      config: .init(weightsDir: "/nonexistent", gemmaPath: "/nonexistent"))
    return WarmServer.PreparedLocalVideo(
      generator: generator,
      request: request,
      mode: initImage == nil ? .t2v : .i2v,
      source: "test",
      optimizationAttemptId: nil,
      enhancementSkippedReason: nil,
      beatScheduleIgnoredReason: nil,
      resolvedDimensions: ResolvedVideoDimensions(
        width: predicted.0, height: predicted.1, reason: reason,
        budgetWidth: 832, budgetHeight: 480,
        sourceWidth: 576, sourceHeight: 1024,
        stage1Width: stage1?.0, stage1Height: stage1?.1))
  }

  private func makeResult(
    outputWidth: Int = 768, outputHeight: Int = 480,
    refineApplied: Bool = true, refineScale: Float = 1.5,
    refineSkippedReason: String? = nil
  ) -> LTX2VideoResult {
    LTX2VideoResult(
      outputPath: "/tmp/comfybox-405.mp4", frameCount: 49,
      durationSeconds: 2.04, elapsedSeconds: 51,
      refineSkippedReason: refineSkippedReason,
      outputWidth: outputWidth, outputHeight: outputHeight,
      refineApplied: refineApplied, refineScale: refineScale)
  }

  // MARK: - The submitted payload (ONE builder, both routes)

  func testSubmittedPayloadCarriesTheFullDimensionRecord() {
    let payload = WarmServer.videoTracePayload(prep: makePrep(), body: Data("{}".utf8))
    XCTAssertEqual(payload["predicted_width"], "768")
    XCTAssertEqual(payload["predicted_height"], "480")
    XCTAssertEqual(payload["dimension_reason"], "source_aspect")
    XCTAssertEqual(payload["dimension_budget"], "832x480")
    XCTAssertEqual(payload["source_size"], "576x1024")
    XCTAssertEqual(payload["stage1_size"], "512x320")
    // The generator dims stay too — they are what the pipeline paints.
    XCTAssertEqual(payload["width"], "512")
    XCTAssertEqual(payload["height"], "320")
    XCTAssertEqual(payload["image_path"], "/tmp/kira-405.png")
  }

  func testBothRoutesUseTheSameSubmittedBuilderSoTheyCannotDrift() {
    // The async route calls `videoTracePayload` inline; the sync route calls it
    // through `recordSyncVideoSubmitted`. Same prep must give the same keys.
    let prep = makePrep()
    let body = Data("{}".utf8)
    let asyncPayload = WarmServer.videoTracePayload(prep: prep, body: body)

    WarmServer.recordSyncVideoSubmitted(store, renderId: "sync-1", prep: prep, body: body)
    store.flush()
    guard let syncPayload = submittedPayload(forRenderId: "sync-1") else {
      return XCTFail("sync route wrote no submitted event")
    }
    XCTAssertEqual(
      Set(asyncPayload.keys), Set(syncPayload.keys),
      "the sync and async submitted events must carry the same field set")
    for (k, v) in asyncPayload {
      XCTAssertEqual(syncPayload[k], v, "field '\(k)' differs between the routes")
    }
  }

  func testT2VSubmittedPayloadOmitsTheI2VOnlyFields() {
    let payload = WarmServer.videoTracePayload(
      prep: makePrep(stage1: nil, reason: .default, initImage: nil),
      body: Data("{}".utf8))
    XCTAssertNil(payload["stage1_size"], "a single-scale render has no stage 1")
    XCTAssertNil(payload["image_path"])
    XCTAssertEqual(payload["dimension_reason"], "default")
  }

  // MARK: - The measured terminal payload

  func testMeasuredPayloadReportsTheEncodedDimsAndTheRefineFacts() {
    let applied = WarmServer.videoMeasuredOutputPayload(result: makeResult())
    XCTAssertEqual(applied["output_width"], "768")
    XCTAssertEqual(applied["output_height"], "480")
    XCTAssertEqual(applied["refine_applied"], "true")
    XCTAssertEqual(applied["refine_scale"], "1.500")

    let skipped = WarmServer.videoMeasuredOutputPayload(
      result: makeResult(
        outputWidth: 512, outputHeight: 320,
        refineApplied: false, refineSkippedReason: "volume_gate"))
    XCTAssertEqual(skipped["output_width"], "512", "a skipped refine outputs the generator dims")
    XCTAssertEqual(skipped["refine_applied"], "false")

    // A result from a path that measured nothing must not invent a size.
    let unmeasured = WarmServer.videoMeasuredOutputPayload(
      result: LTX2VideoResult(
        outputPath: "/tmp/x.mp4", frameCount: 0, durationSeconds: 0, elapsedSeconds: 0))
    XCTAssertNil(unmeasured["output_width"])
    XCTAssertNil(unmeasured["output_height"])
  }

  // MARK: - The sync route writes submitted + terminal on BOTH paths

  func testSyncRouteWritesSubmittedAndTerminalOnSuccess() {
    WarmServer.recordSyncVideoSubmitted(
      store, renderId: "sync-ok", prep: makePrep(), body: Data("{}".utf8))
    WarmServer.recordSyncVideoSucceeded(store, renderId: "sync-ok", result: makeResult())
    store.flush()

    guard let summary = store.recentSummaries().first(where: { $0.renderId == "sync-ok" }) else {
      return XCTFail("the sync render is missing from GET /v1/video/traces")
    }
    XCTAssertTrue(summary.events.contains("submitted"))
    XCTAssertTrue(summary.events.contains("terminal"))
    XCTAssertEqual(summary.status, "succeeded")
    XCTAssertEqual(summary.outputPath, "/tmp/comfybox-405.mp4")
    // The MEASURED pair wins over the prediction once the render has finished.
    XCTAssertEqual(summary.outputWidth, "768")
    XCTAssertEqual(summary.outputHeight, "480")
    XCTAssertEqual(summary.dimensionSource, "measured")
    XCTAssertEqual(summary.predictedWidth, "768")
    XCTAssertEqual(summary.refineApplied, "true")
    XCTAssertEqual(summary.refineScale, "1.500")
    XCTAssertEqual(summary.dimensionReason, "source_aspect")
    XCTAssertEqual(summary.stage1Size, "512x320")
  }

  func testSyncRouteWritesSubmittedAndTerminalOnFailure() {
    struct Boom: Error {}
    WarmServer.recordSyncVideoSubmitted(
      store, renderId: "sync-fail", prep: makePrep(), body: Data("{}".utf8))
    WarmServer.recordSyncVideoFailed(store, renderId: "sync-fail", error: Boom())
    store.flush()

    guard let summary = store.recentSummaries().first(where: { $0.renderId == "sync-fail" }) else {
      return XCTFail("a failed sync render must still leave a trace")
    }
    XCTAssertTrue(summary.events.contains("submitted"))
    XCTAssertTrue(summary.events.contains("terminal"))
    XCTAssertEqual(summary.status, "failed")
    XCTAssertNotNil(summary.error)
    // No measurement exists, so the summary falls back to the prediction and
    // says so rather than claiming a size the file never had.
    XCTAssertEqual(summary.dimensionSource, "predicted")
    XCTAssertEqual(summary.outputWidth, "768")
  }

  func testAnInFlightRenderReportsThePredictionAndLabelsItAsSuch() {
    WarmServer.recordSyncVideoSubmitted(
      store, renderId: "sync-running", prep: makePrep(), body: Data("{}".utf8))
    store.flush()
    guard let summary = store.recentSummaries().first(where: { $0.renderId == "sync-running" })
    else { return XCTFail("missing trace") }
    XCTAssertEqual(summary.status, "running")
    XCTAssertEqual(summary.dimensionSource, "predicted")
    XCTAssertEqual(summary.outputWidth, "768")
    XCTAssertNil(summary.refineApplied, "there is no refine verdict before the render finishes")
  }

  // MARK: - Helpers

  private func submittedPayload(forRenderId id: String) -> [String: String]? {
    // Read the JSONL back the way the store itself does, so the assertion is
    // against what actually landed on disk.
    guard let files = try? FileManager.default.contentsOfDirectory(
      at: tempDir, includingPropertiesForKeys: nil) else { return nil }
    for file in files {
      guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
      for line in text.split(separator: "\n") {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              obj["render_id"] as? String == id,
              obj["event"] as? String == "submitted",
              let payload = obj["payload"] as? [String: String]
        else { continue }
        return payload
      }
    }
    return nil
  }
}
