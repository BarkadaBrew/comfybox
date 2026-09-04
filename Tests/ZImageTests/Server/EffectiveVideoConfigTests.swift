import XCTest
@testable import ZImage

/// comfybox#307 (review r1, item 2): `POST /v1/video/config/effective` is the
/// preflight callers use to check what a render would actually resolve to —
/// it must accept the same `two_pass` convenience the real generate routes
/// do, and reflect it in the derived plan (specifically the
/// `two_stage_halving` step), not just in the raw resolved param readout.
final class EffectiveVideoConfigTests: XCTestCase {

  private func decodeQuery(_ json: String) throws -> WarmServer.EffectiveVideoConfigQuery {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(WarmServer.EffectiveVideoConfigQuery.self, from: Data(json.utf8))
  }

  // MARK: - Query decode

  func testTwoPassTrueDecodesOnTheQuery() throws {
    let q = try decodeQuery(#"{"two_pass":true}"#)
    XCTAssertEqual(q.twoPass, true)
  }

  func testTwoPassAbsentDecodesAsNilOnTheQuery() throws {
    let q = try decodeQuery(#"{"width":704,"height":448}"#)
    XCTAssertNil(q.twoPass)
  }

  // MARK: - effectiveVideoTuning(for: EffectiveVideoConfigQuery) — same merge as the request path

  func testEffectiveTuningFollowsTopLevelTwoPassOnTheQuery() throws {
    let q = try decodeQuery(#"{"two_pass":true}"#)
    XCTAssertEqual(WarmServer.effectiveVideoTuning(for: q)?.twoStage, true)
  }

  func testEffectiveTuningNestedTwoStageWinsOnTheQuery() throws {
    let q = try decodeQuery(#"{"two_pass":true,"tuning":{"two_stage":false}}"#)
    XCTAssertEqual(WarmServer.effectiveVideoTuning(for: q)?.twoStage, false)
  }

  // MARK: - Plan output: two_pass must reach two_stage_halving / stage1_floor

  private func plan(width: Int, height: Int, resolvedTwoStage: Bool) -> [[String: String]] {
    WarmServer.effectiveVideoRenderPlan(
      width: width, height: height, frames: nil, duration: nil, fps: nil,
      presetWidth: nil, presetHeight: nil,
      videoConfigDefaults: VideoDefaultValues(),
      resolvedTwoStage: resolvedTwoStage)
  }

  func testTwoStageHalvingStepAppearsWhenTwoStageResolvesTrue() {
    let steps = plan(width: 1024, height: 1024, resolvedTwoStage: true).map { $0["step"] }
    XCTAssertTrue(steps.contains("two_stage_halving"), "\(steps)")
  }

  func testNoHalvingOrFloorStepWhenTwoStageResolvesFalse() {
    let steps = plan(width: 1024, height: 1024, resolvedTwoStage: false).map { $0["step"] }
    XCTAssertFalse(steps.contains("two_stage_halving"), "\(steps)")
    XCTAssertFalse(steps.contains("stage1_floor"), "\(steps)")
  }

  func testStage1FloorStepWhenDimsTooSmallToHalve() {
    // Small enough that stageOneDims' halved-area floor (512*320) rejects halving.
    let steps = plan(width: 320, height: 320, resolvedTwoStage: true).map { $0["step"] }
    XCTAssertTrue(steps.contains("stage1_floor"), "\(steps)")
    XCTAssertFalse(steps.contains("two_stage_halving"), "\(steps)")
  }

  /// End-to-end through the ACTUAL two functions the route calls, in the
  /// same order: decode → merge (`two_pass` alone, no `tuning` block) →
  /// resolve → plan. This is the wiring the review flagged as untested —
  /// `two_pass` reaching the plan's `two_stage_halving` step with nothing
  /// else in the request naming `two_stage` anywhere.
  func testTwoPassAloneDrivesTwoStageHalvingThroughTheRealPath() throws {
    let q = try decodeQuery(#"{"width":1024,"height":1024,"two_pass":true}"#)
    let effectiveTuning = WarmServer.effectiveVideoTuning(for: q)
    let resolved = LTX2ConfigResolver.resolveTyped(
      request: effectiveTuning, preset: nil, environment: ["LTX2_TWO_STAGE": "0"], configFile: [:])
    XCTAssertTrue(resolved.twoStage, "two_pass must win over the env-global default")
    let steps = WarmServer.effectiveVideoRenderPlan(
      width: q.width, height: q.height, frames: q.frames, duration: q.duration, fps: q.fps,
      presetWidth: nil, presetHeight: nil,
      videoConfigDefaults: VideoDefaultValues(),
      resolvedTwoStage: resolved.twoStage
    ).map { $0["step"] }
    XCTAssertTrue(steps.contains("two_stage_halving"), "\(steps)")
  }

  /// Same path with `two_pass` absent — the plan must NOT halve, matching
  /// pre-existing (env/builtin `two_stage=false`) behavior exactly.
  func testAbsentTwoPassLeavesTwoStageHalvingOutThroughTheRealPath() throws {
    let q = try decodeQuery(#"{"width":1024,"height":1024}"#)
    let effectiveTuning = WarmServer.effectiveVideoTuning(for: q)
    let resolved = LTX2ConfigResolver.resolveTyped(
      request: effectiveTuning, preset: nil, environment: [:], configFile: [:])
    XCTAssertFalse(resolved.twoStage)
    let steps = WarmServer.effectiveVideoRenderPlan(
      width: q.width, height: q.height, frames: q.frames, duration: q.duration, fps: q.fps,
      presetWidth: nil, presetHeight: nil,
      videoConfigDefaults: VideoDefaultValues(),
      resolvedTwoStage: resolved.twoStage
    ).map { $0["step"] }
    XCTAssertFalse(steps.contains("two_stage_halving"), "\(steps)")
  }
}
