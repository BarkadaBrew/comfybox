import XCTest
import MLX
import MLXRandom
@testable import ZImage

final class LTX2PreemptionTests: XCTestCase {
  func testSignalRaiseClear() {
    let s = PreemptionSignal()
    XCTAssertFalse(s.isRaised)
    s.raise(); XCTAssertTrue(s.isRaised)
    s.clear(); XCTAssertFalse(s.isRaised)
  }

  func testSeededNoiseIsDeterministicPerStep() {
    let a = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 7)
    let b = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 7)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self), "same seed+step must be identical")
    let c = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 8)
    XCTAssertFalse(MLX.allClose(a, c).item(Bool.self), "different step must differ")
    let d = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 43, step: 7)
    XCTAssertFalse(MLX.allClose(a, d).item(Bool.self), "different seed must differ")
  }

  func testSeededNoiseIgnoresGlobalStreamPosition() {
    MLXRandom.seed(1)
    let a = ancestralVideoNoise(shape: [2, 2], seed: 9, step: 3)
    MLXRandom.seed(999)
    _ = MLXRandom.normal([16])   // scramble the global stream
    let b = ancestralVideoNoise(shape: [2, 2], seed: 9, step: 3)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self),
      "seeded draw must be independent of global stream position — this IS the resume guarantee")
  }

  func testUnseededNoiseUsesGlobalStream() {
    MLXRandom.seed(7)
    let a = ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0)
    MLXRandom.seed(7)
    let b = ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self),
      "unseeded path must remain the plain global stream (unchanged behaviour)")
  }

  func testNoiseDtypeIsFloat32() {
    XCTAssertEqual(ancestralVideoNoise(shape: [2, 2], seed: 1, step: 0).dtype, .float32)
    XCTAssertEqual(ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0).dtype, .float32)
  }

  /// Regression for the codex-review finding (2026-08-15): the production
  /// chunk scheduler derives each chunk's seed as `request.seed + chunk`
  /// (`LTX2VideoGenerator.swift:911`), so a bare `seed + step` key formula
  /// makes chunk `c` step `i` collide with chunk `c+1` step `i-1` —
  /// `(seed+1) + i == seed + (i+1)`. The step multiplier must make that
  /// class of collision unreachable.
  func testChunkSeedDoesNotAliasStepKey() {
    let s: UInt64 = 42
    let i = 7
    let a = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: s &+ 1, step: i)
    let b = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: s, step: i &+ 1)
    XCTAssertFalse(MLX.allClose(a, b).item(Bool.self),
      "chunk-seed(+1)/step(i) must not collide with chunk-seed(+0)/step(i+1)")
  }

  // MARK: - Task 4: resume admissibility (pure logic only; the real resume
  // behaviour is Task 6's integration test).

  private func state(
    fingerprint: String = "704x480f97s8-euler_a-cfg3.5",
    sigmas: [Float] = [1.0, 0.5, 0.0],
    step: Int = 1,
    phase: LTX2Phase = .baseDenoise,
    refineClean: MLXArray? = nil
  ) -> LTX2ResumeState {
    LTX2ResumeState(
      videoLatents: MLXArray.zeros([1]), stepIndex: step, sigmas: sigmas,
      phase: phase, chunkIndex: 0, seed: 1,
      audioLatents: nil, audioNoiseKey: nil,
      configFingerprint: fingerprint, refineCleanLatents: refineClean)
  }

  private func validate(_ s: LTX2ResumeState, fingerprint: String, sigmas: [Float]) throws {
    try LTX2ResumeValidator.validate(
      checkpointFingerprint: s.configFingerprint, currentFingerprint: fingerprint,
      checkpointSigmas: s.sigmas, currentSigmas: sigmas, stepIndex: s.stepIndex)
  }

  func testResumeAcceptsMatchingConfig() {
    let s = state()
    XCTAssertNoThrow(try validate(s, fingerprint: s.configFingerprint, sigmas: s.sigmas))
  }

  /// Controller ruling: resume THROWS on a fingerprint mismatch. Never a
  /// silent restart from step 0.
  func testResumeThrowsOnFingerprintMismatch() {
    let s = state(fingerprint: "704x480f97s8-euler_a-cfg3.5")
    XCTAssertThrowsError(
      try validate(s, fingerprint: "704x480f97s8-euler_a-cfg2.0", sigmas: s.sigmas)
    ) { error in
      guard case LTX2ResumeError.configFingerprintMismatch(let checkpoint, let current) = error else {
        return XCTFail("expected configFingerprintMismatch, got \(error)")
      }
      XCTAssertEqual(checkpoint, "704x480f97s8-euler_a-cfg3.5")
      XCTAssertEqual(current, "704x480f97s8-euler_a-cfg2.0")
      // Both fingerprints must be named in the message — a mismatch is a bug
      // report, not a shrug.
      let described = (error as? LTX2ResumeError)?.errorDescription ?? ""
      XCTAssertTrue(described.contains("cfg3.5") && described.contains("cfg2.0"), described)
    }
  }

  /// The fingerprint covers dims/steps/sampler/cfg but NOT the sigma VALUES,
  /// so a re-tuned schedule (e.g. 46217ef's tarn1 sigmas) would slip through
  /// fingerprint-only validation and silently re-time every remaining step.
  func testResumeThrowsOnSigmaScheduleChangeWithIdenticalFingerprint() {
    let s = state(sigmas: [1.0, 0.5, 0.0])
    XCTAssertThrowsError(
      try validate(s, fingerprint: s.configFingerprint, sigmas: [1.0, 0.6, 0.0])
    ) { error in
      guard case LTX2ResumeError.sigmaScheduleMismatch(_, _, let idx) = error else {
        return XCTFail("expected sigmaScheduleMismatch, got \(error)")
      }
      XCTAssertEqual(idx, 1, "must name the first differing index")
    }
  }

  func testResumeThrowsOnSigmaCountChange() {
    let s = state(sigmas: [1.0, 0.5, 0.0])
    XCTAssertThrowsError(
      try validate(s, fingerprint: s.configFingerprint, sigmas: [1.0, 0.75, 0.5, 0.0]))
  }

  func testResumeThrowsWhenStepIsOutsideTheSchedule() {
    let s = state(sigmas: [1.0, 0.5, 0.0], step: 5)   // schedule has 2 steps
    XCTAssertThrowsError(
      try validate(s, fingerprint: s.configFingerprint, sigmas: s.sigmas)
    ) { error in
      guard case LTX2ResumeError.stepOutOfRange = error else {
        return XCTFail("expected stepOutOfRange, got \(error)")
      }
    }
  }

  /// A checkpoint taken at a chunk/pre-load boundary has no trajectory to
  /// protect — it restarts the phase from its deterministic beginning, so it
  /// carries the not-started sentinel instead of a real fingerprint.
  func testNotStartedSentinelIsDistinctFromAnyRealFingerprint() {
    let s = state(fingerprint: ltx2NotStartedFingerprint, sigmas: [], step: 0)
    XCTAssertEqual(s.configFingerprint, ltx2NotStartedFingerprint)
    XCTAssertNotEqual(ltx2NotStartedFingerprint, "704x480f97s8-euler_a-cfg3.5")
  }

  /// The refine loop's checkpoint is distinguished from the base→refine
  /// boundary purely by carrying the clean base latent; the dispatch in
  /// LTX2Pipeline keys off exactly this.
  func testRefineCheckpointIsIdentifiedByItsCleanBaseLatent() {
    let boundary = state(phase: .refineDenoise, refineClean: nil)
    let midRefine = state(phase: .refineDenoise, refineClean: MLXArray.zeros([1, 2]))
    XCTAssertNil(boundary.refineCleanLatents)
    XCTAssertNotNil(midRefine.refineCleanLatents)
  }

  // MARK: - Forward-only unwind points (review fix, Critical 1)

  /// A boundary the resumed checkpoint has already passed must NOT
  /// re-checkpoint: doing so regresses a refined render to "refine not
  /// started" (next resume double-refines) or relabels partially refined
  /// latents as finished base latents (next resume refines from scratch).
  func testUnwindPointRefusesToRegressAResumedRender() {
    // base→refine while resuming a mid-refine or finished-refine checkpoint.
    XCTAssertFalse(LTX2UnwindGuard.mayCheckpoint(at: .refineDenoise, whileResuming: .refineDenoise))
    XCTAssertFalse(LTX2UnwindGuard.mayCheckpoint(at: .refineDenoise, whileResuming: .vaeDecode))
    // refine→decode while resuming a checkpoint that already reached decode.
    XCTAssertFalse(LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: .vaeDecode))
  }

  func testUnwindPointAllowsForwardProgress() {
    // Fresh render: every boundary is fair game.
    XCTAssertTrue(LTX2UnwindGuard.mayCheckpoint(at: .refineDenoise, whileResuming: nil))
    XCTAssertTrue(LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: nil))
    // Resumed mid-base: both later boundaries are real progress.
    XCTAssertTrue(LTX2UnwindGuard.mayCheckpoint(at: .refineDenoise, whileResuming: .baseDenoise))
    XCTAssertTrue(LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: .baseDenoise))
    // Resumed mid-refine: the refine actually ran, so refine→decode is forward.
    XCTAssertTrue(LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: .refineDenoise))
  }

  func testUnwindRankIsStrictlyOrderedAlongTheRenderPipeline() {
    let pipelineOrder: [LTX2Phase] = [
      .modelLoad, .textEncode, .baseDenoise, .refineDenoise, .vaeDecode, .vocoder, .postProcess,
    ]
    let ranks = pipelineOrder.map(LTX2UnwindGuard.rank)
    XCTAssertEqual(ranks, ranks.sorted(), "rank must follow the render's phase order")
    XCTAssertEqual(Set(ranks).count, ranks.count, "ranks must be distinct or the guard cannot compare")
  }

  // MARK: - Refine machinery drift (review fix, Important 2)

  /// Two-stage/upsampler config is re-resolved every render. If it drifted off
  /// while a MID-REFINE checkpoint was parked, continuing would send a
  /// partially denoised REFINE-resolution tensor to the decoder as finished
  /// work — refuse.
  func testMidRefineResumeIsRefusedWhenTheRefineMachineryIsGone() {
    let midRefine = state(phase: .refineDenoise, refineClean: MLXArray.zeros([1, 2]))
    XCTAssertTrue(LTX2RefineAvailability.mustRefuse(
      resume: midRefine, twoStage: false, upsamplerLoaded: true),
      "two-stage turned off under a mid-refine checkpoint is real drift")
    XCTAssertTrue(LTX2RefineAvailability.mustRefuse(
      resume: midRefine, twoStage: true, upsamplerLoaded: false),
      "a missing upsampler under a mid-refine checkpoint is real drift")
    XCTAssertFalse(LTX2RefineAvailability.mustRefuse(
      resume: midRefine, twoStage: true, upsamplerLoaded: true),
      "machinery present — nothing to refuse")
  }

  /// Regression for the round-2 review: a base→refine BOUNDARY checkpoint
  /// holds clean, finished, BASE-resolution latents, so skipping the refine
  /// gives exactly what a fresh render under the current config gives. The
  /// built-in default is `two_stage=false`, so refusing here would make EVERY
  /// preempted render unresumable on an unremarkable config.
  func testBoundaryRefineCheckpointResumesWithTwoStageOff() {
    let boundary = state(phase: .refineDenoise, refineClean: nil)
    XCTAssertFalse(LTX2RefineAvailability.mustRefuse(
      resume: boundary, twoStage: false, upsamplerLoaded: false),
      "a boundary checkpoint must resume under two_stage=false — the default")
    XCTAssertFalse(LTX2RefineAvailability.mustRefuse(
      resume: boundary, twoStage: true, upsamplerLoaded: false),
      "a boundary checkpoint must resume with the upsampler absent")
  }

  func testNonRefinePhasesAndFreshRendersAreNeverRefused() {
    XCTAssertFalse(LTX2RefineAvailability.mustRefuse(
      resume: nil, twoStage: false, upsamplerLoaded: false), "a fresh render is never refused")
    XCTAssertFalse(LTX2RefineAvailability.mustRefuse(
      resume: state(phase: .baseDenoise), twoStage: false, upsamplerLoaded: false))
    XCTAssertFalse(LTX2RefineAvailability.mustRefuse(
      resume: state(phase: .vaeDecode, refineClean: nil), twoStage: false, upsamplerLoaded: false),
      "a .vaeDecode checkpoint skips the refine by design")
  }

  func testRefineUnavailableOnResumeDescribesTheDrift() {
    let error = LTX2ResumeError.refineUnavailableOnResume(twoStage: false, upsamplerLoaded: true)
    let described = error.errorDescription ?? ""
    XCTAssertTrue(described.contains("two_stage=false"), described)
    XCTAssertTrue(described.contains("upsampler_loaded=true"), described)
    XCTAssertNotEqual(
      error,
      LTX2ResumeError.refineUnavailableOnResume(twoStage: true, upsamplerLoaded: false),
      "the two drift modes must be distinguishable")
  }

  /// The checkpoint materializes every array field it stores (Task 2's
  /// invariant) — including the field added for the refine pass.
  func testRefineCleanLatentsAreMaterializedOnCapture() throws {
    let live = MLXArray.zeros([2, 2]) + MLXArray(Float(3))
    let s = state(refineClean: live)
    let stored = try XCTUnwrap(s.refineCleanLatents)
    XCTAssertEqual(stored.mean().item(Float.self), 3)
  }
}
