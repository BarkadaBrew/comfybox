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

  /// The regression this task fixes, expressed as a round trip through the
  /// REAL snapshot builder (`LTX2RenderContext.checkpointSnapshot` — comfybox
  /// #307 review r2, item 2c: not a hand re-implementation of what
  /// `LTX2VideoGenerator.render`'s `checkpoint()` closure does; this test
  /// calls the exact function that closure calls, so deleting the
  /// `snapshot.refineSkippedReason = ...` line inside it fails HERE), boxed
  /// into an `LTX2ResumeState` and handed to a SIMULATED "cold" resume that
  /// only has the state — never the original pipeline — downcasting
  /// `.context` back. The reason must still be there.
  func testSkipReasonSurvivesACheckpointResumeRoundTrip() {
    let ctx = LTX2RenderContext(request: request())
    ctx.refineSkippedReason = "upsampler_unavailable (two_stage requested but no upsampler loaded — check LTX2_UPSAMPLER_PATH)"

    let snapshot = LTX2RenderContext.checkpointSnapshot(
      from: ctx, chunk: 0, frames: [], audio: nil, seedImage: nil,
      refineSkippedReason: ctx.refineSkippedReason, elapsedThisSegment: 1.5)

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
  /// the field survives the round trip (through the real builder) as nil too.
  func testNoSkipReasonSurvivesAsNil() {
    let ctx = LTX2RenderContext(request: request())
    let snapshot = LTX2RenderContext.checkpointSnapshot(
      from: ctx, chunk: 0, frames: [], audio: nil, seedImage: nil,
      refineSkippedReason: nil, elapsedThisSegment: 1.5)
    let state = LTX2ResumeState(
      videoLatents: MLXArray([0 as Float]), stepIndex: 0, sigmas: [1.0, 0.0],
      phase: .baseDenoise, chunkIndex: 0, seed: 42,
      audioLatents: nil, audioNoiseKey: nil, configFingerprint: "fp",
      context: snapshot)
    XCTAssertNil((state.context as? LTX2RenderContext)?.refineSkippedReason)
  }

  /// `checkpointSnapshot` also carries `accumulatedSeconds` forward
  /// correctly (prior segments + this one) — pinned here since this test is
  /// now the one place exercising the real builder end to end.
  func testCheckpointSnapshotAccumulatesSeconds() {
    let ctx = LTX2RenderContext(request: request())
    ctx.accumulatedSeconds = 10.0
    let snapshot = LTX2RenderContext.checkpointSnapshot(
      from: ctx, chunk: 1, frames: [], audio: nil, seedImage: nil,
      refineSkippedReason: nil, elapsedThisSegment: 2.5)
    XCTAssertEqual(snapshot.accumulatedSeconds, 12.5)
    XCTAssertEqual(snapshot.chunkIndex, 1)
  }

  /// A negative segment delta (clock skew) must never subtract from the
  /// accumulated total — matches the original inline `max(0, ...)`.
  func testCheckpointSnapshotClampsNegativeElapsed() {
    let ctx = LTX2RenderContext(request: request())
    ctx.accumulatedSeconds = 10.0
    let snapshot = LTX2RenderContext.checkpointSnapshot(
      from: ctx, chunk: 1, frames: [], audio: nil, seedImage: nil,
      refineSkippedReason: nil, elapsedThisSegment: -3.0)
    XCTAssertEqual(snapshot.accumulatedSeconds, 10.0)
  }
}
