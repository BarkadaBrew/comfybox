import XCTest
@testable import ZImage

/// Repro for the lost request-level tuning (2026-08-11): the wire body
/// carries tuning.stage1_sigmas (verified in the render trace) but the
/// render resolves (env). Decode the EXACT failing body shape.
final class LocalVideoRequestDecodeTests: XCTestCase {
  func testTuningSurvivesFullBodyDecode() throws {
    let body = """
    {"prompt":"x","preset":"kira-video-avocado","width":480,"height":832,
     "frames":97,"seed":4242,"fps":24,"audio":true,"enhance":false,
     "source":"tuning",
     "tuning":{"stage1_sigmas":[1.0,0.955,0.893,0.812,0.715,0.603,0.482,0.241,0.121,0.0]},
     "output_path":"/tmp/x.mp4"}
    """
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let req = try decoder.decode(WarmServer.LocalVideoRequest.self, from: Data(body.utf8))
    XCTAssertNotNil(req.tuning, "tuning must decode")
    XCTAssertEqual(req.tuning?.stage1Sigmas?.count, 10, "sigmas must survive")
  }

  // MARK: - comfybox#307: top-level `two_pass` convenience field

  private func decodeLocalVideoRequest(_ json: String) throws -> WarmServer.LocalVideoRequest {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(WarmServer.LocalVideoRequest.self, from: Data(json.utf8))
  }

  func testTwoPassTrueDecodes() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":true}"#)
    XCTAssertEqual(req.twoPass, true)
  }

  func testTwoPassFalseDecodes() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":false}"#)
    XCTAssertEqual(req.twoPass, false)
  }

  func testTwoPassExplicitNullDecodesAsNil() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":null}"#)
    XCTAssertNil(req.twoPass)
  }

  func testTwoPassAbsentDecodesAsNil() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x"}"#)
    XCTAssertNil(req.twoPass, "no two_pass key, no tuning key — additive field must not be required")
  }

  /// `two_pass` is additive: it must not disturb the pre-existing
  /// `tuning.two_stage` decode path exercised above.
  func testTwoPassAlongsideExistingTuningBlockBothDecode() throws {
    let req = try decodeLocalVideoRequest(#"""
      {"prompt":"x","two_pass":true,"tuning":{"refine_scale":1.35}}
      """#)
    XCTAssertEqual(req.twoPass, true)
    XCTAssertEqual(req.tuning?.refineScale, 1.35)
  }

  // MARK: - comfybox#307 (review r1, item 3a): the REAL request-preparation
  // wiring — `WarmServer.effectiveVideoTuning(for:)` is the exact function
  // `prepareLocalVideo` calls, not a re-implementation of the merge. These
  // decode a real wire body and assert on ITS output, pinning the actual
  // path a `two_pass` request travels before it ever reaches
  // `LTX2ConfigResolver.resolveTyped`.

  func testEffectiveTuningFollowsTopLevelTwoPassWithNoTuningBlock() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":true}"#)
    XCTAssertEqual(WarmServer.effectiveVideoTuning(for: req)?.twoStage, true)
  }

  func testEffectiveTuningTwoPassFalseWithNoTuningBlock() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":false}"#)
    XCTAssertEqual(WarmServer.effectiveVideoTuning(for: req)?.twoStage, false)
  }

  func testEffectiveTuningNilTwoPassLeavesTuningNil() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x"}"#)
    XCTAssertNil(WarmServer.effectiveVideoTuning(for: req))
  }

  /// The nested field is the more specific one — it wins when the wire body
  /// carries both, decoded exactly as a real caller would send it.
  func testEffectiveTuningNestedTwoStageWinsOverConflictingTopLevelTwoPass() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":true,"tuning":{"two_stage":false}}"#)
    XCTAssertEqual(WarmServer.effectiveVideoTuning(for: req)?.twoStage, false)
  }

  /// `two_pass` fills in `tuning.two_stage` when the tuning block is present
  /// but doesn't itself set `two_stage` — other tuning fields on the request
  /// survive the merge untouched.
  func testEffectiveTuningTwoPassFillsInAlongsideOtherTuningFields() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":true,"tuning":{"refine_scale":1.35}}"#)
    let merged = WarmServer.effectiveVideoTuning(for: req)
    XCTAssertEqual(merged?.twoStage, true)
    XCTAssertEqual(merged?.refineScale, 1.35)
  }

  // MARK: - comfybox#307 (review r2, item 2a): the merge alone proves the
  // MERGE function is correct, but nothing above proves its result is what
  // actually reaches the `LTX2VideoRequest` the generator renders — a
  // one-line revert (`tuning: effectiveTuning` → `tuning: req.tuning`) at
  // the construction site would pass every test above. These call
  // `WarmServer.buildLocalVideoRequest`, the ACTUAL construction
  // `prepareLocalVideo` runs, with a real decoded request.

  private func buildRequest(_ json: String) throws -> LTX2VideoRequest {
    let req = try decodeLocalVideoRequest(json)
    return WarmServer.buildLocalVideoRequest(
      req: req, effectiveTuning: WarmServer.effectiveVideoTuning(for: req), videoPreset: nil,
      effectivePrompt: req.prompt, effectiveInitImage: nil,
      renderWidth: 704, renderHeight: 448,
      foldedFramesPerChunk: 97, foldedExtendSeconds: 0,
      resolvedLoRAs: [], effectiveBeatSchedule: nil,
      resolvedOutput: "/tmp/o.mp4")
  }

  func testBuildLocalVideoRequestCarriesTopLevelTwoPassIntoTuning() throws {
    let request = try buildRequest(#"{"prompt":"x","two_pass":true}"#)
    XCTAssertEqual(request.tuning?.twoStage, true, "the constructed LTX2VideoRequest must carry the merged tuning")
  }

  func testBuildLocalVideoRequestNoTwoPassLeavesTuningNil() throws {
    let request = try buildRequest(#"{"prompt":"x"}"#)
    XCTAssertNil(request.tuning)
  }

  func testBuildLocalVideoRequestNestedTuningStillWinsOnConflict() throws {
    let request = try buildRequest(#"{"prompt":"x","two_pass":true,"tuning":{"two_stage":false}}"#)
    XCTAssertEqual(request.tuning?.twoStage, false)
  }
}
