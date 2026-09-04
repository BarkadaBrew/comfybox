import XCTest
import MLX
@testable import ZImage

/// comfybox#307 (review r1, item 1): the skip reason must NOT live only on
/// `LTX2Pipeline.lastRefineSkipReason` — a cold preemption resume can rebuild
/// the pipeline/generator from scratch (`VideoGeneratorHolder.release()`
/// deallocates them), which would silently drop a reason recorded on an
/// earlier chunk before the eviction. It has to live on `LTX2RenderContext`,
/// which travels WITH the checkpoint (`LTX2ResumeState.context`) — the whole
/// reason that type exists (see its own doc comment).
///
/// This pins the context as a plain, testable value: default nil, settable,
/// and — the actual regression scenario — surviving the exact round trip
/// production code takes it through (`LTX2RenderContext` → `LTX2ResumeState
/// .context` → downcast back), simulating a checkpoint handed to a rebuilt
/// generator that has no memory of the pipeline that recorded the reason.
final class LTX2RenderContextRefineSkipTests: XCTestCase {

  override func setUpWithError() throws {
    do {
      try MLX.withError {
        let probe = MLXArray([1 as Float, 2], [2]) + MLXArray([3 as Float, 4], [2])
        MLX.eval(probe)
      }
    } catch {
      throw XCTSkip("MLX evaluation is unavailable in this test runner: \(error)")
    }
  }

  private func request() -> LTX2VideoRequest {
    LTX2VideoRequest(prompt: "x", width: 704, height: 448, framesPerChunk: 97, outputPath: "/tmp/o.mp4")
  }

  func testDefaultsToNil() {
    let ctx = LTX2RenderContext(request: request())
    XCTAssertNil(ctx.refineSkippedReason)
  }

  func testIsSettable() {
    let ctx = LTX2RenderContext(request: request())
    ctx.refineSkippedReason = "volume_gate (pre-refine volume 30000 > refine_max_vol 26000)"
    XCTAssertEqual(ctx.refineSkippedReason, "volume_gate (pre-refine volume 30000 > refine_max_vol 26000)")
  }

  /// The regression this task fixes, expressed as a pure round trip: a
  /// context carrying a recorded skip reason, checkpointed into an
  /// `LTX2ResumeState` (exactly as `LTX2VideoGenerator.render`'s `checkpoint`
  /// closure does), handed to a SIMULATED "cold" resume that only has the
  /// state — never the original pipeline — and downcasts `.context` back.
  /// The reason must still be there.
  func testSkipReasonSurvivesACheckpointResumeRoundTrip() {
    let ctx = LTX2RenderContext(request: request())
    ctx.refineSkippedReason = "upsampler_unavailable (two_stage requested but no upsampler loaded — check LTX2_UPSAMPLER_PATH)"

    // Simulate `checkpoint()`'s snapshot: a NEW context object, fields copied
    // across — this is what actually crosses the eviction boundary.
    let snapshot = LTX2RenderContext(request: ctx.request)
    snapshot.chunkIndex = ctx.chunkIndex
    snapshot.refineSkippedReason = ctx.refineSkippedReason

    let state = LTX2ResumeState(
      videoLatents: MLXArray([0 as Float]), stepIndex: 0, sigmas: [1.0, 0.0],
      phase: .baseDenoise, chunkIndex: 0, seed: 42,
      audioLatents: nil, audioNoiseKey: nil, configFingerprint: "fp",
      context: snapshot)

    // "Cold resume": nothing but `state` — the pipeline/generator that
    // recorded the reason is gone (deallocated by eviction). `render()`
    // rebuilds `ctx` from exactly this downcast.
    let resumedCtx = state.context as? LTX2RenderContext
    XCTAssertEqual(
      resumedCtx?.refineSkippedReason,
      "upsampler_unavailable (two_stage requested but no upsampler loaded — check LTX2_UPSAMPLER_PATH)")
  }

  /// A checkpoint taken with NO skip recorded yet must not manufacture one —
  /// the field survives the round trip as nil too.
  func testNoSkipReasonSurvivesAsNil() {
    let ctx = LTX2RenderContext(request: request())
    let state = LTX2ResumeState(
      videoLatents: MLXArray([0 as Float]), stepIndex: 0, sigmas: [1.0, 0.0],
      phase: .baseDenoise, chunkIndex: 0, seed: 42,
      audioLatents: nil, audioNoiseKey: nil, configFingerprint: "fp",
      context: ctx)
    XCTAssertNil((state.context as? LTX2RenderContext)?.refineSkippedReason)
  }
}
