import XCTest
import MLX
import MLXRandom
@testable import ZImage

/// AC-74 — the Z-Image `res_2s` change is intended, measured and recorded
/// (FDD-krea2-raw-recipe D2, §3.2).
///
/// Weight-free half. `ZImagePipeline` feeds its schedulers
/// `velocity = -guidedNoise` (the transformer emits `x₀ − ε`, so the flow
/// velocity `dx/dσ = ε − x₀` is its negation) and, since WP-E2, converts that
/// velocity through `scheduler.modelInput(velocity:sample:sigma:)` once per
/// evaluation. This file exercises exactly that expression on Z-Image's own
/// dynamically-shifted flow grid:
///   * `res_2s` post-fix reconstructs x₀ (AC-10's assertion on the Z-Image path);
///   * the pre-fix feed (velocity straight into `res_2s`) measurably does not —
///     the render *must* differ, and this is the measurement;
///   * euler/flow — the default path — is byte-identical with the conversion
///     in the loop (AC-6).
/// The live-render pre/post comparison needs weights and lives with the
/// integration phase; see the CHANGELOG entry this AC also requires.
final class ZImageRES2sCorrectionTests: XCTestCase {

  /// Z-Image Turbo's scheduler config (dynamic shifting, 256→4096 tokens,
  /// 0.5→1.15) and the mu the pipeline derives for a 1024² render.
  static func zImageConfig() -> ZImageSchedulerConfig {
    FlowMatchSchedulerTests.makeConfig(
      numTrainTimesteps: 1000, shift: 3.0, useDynamicShifting: true,
      baseShift: 0.5, maxShift: 1.15, baseImageSeqLen: 256, maxImageSeqLen: 4096)
  }

  static func zImageMu() -> Float {
    PipelineUtilities.calculateShift(
      imageSeqLen: 4096, baseSeqLen: 256, maxSeqLen: 4096, baseShift: 0.5, maxShift: 1.15)
  }

  /// The pipeline's loop body, with the transformer replaced by the exact
  /// Z-Image-sign prediction `noisePred = x₀ − ε` of a fixed (x₀, ε).
  /// `convert == false` is the pre-E2 body (velocity fed raw).
  static func runZImageLoop(
    scheduler input: any ZImageScheduler,
    field: ModelOutputConventionTests.RectifiedFlowField,
    convert: Bool
  ) -> MLXArray {
    var scheduler = input
    scheduler.reset()
    let sigmas = scheduler.sigmas.asArray(Float.self)
    var latents = field.sample(at: sigmas[0])
    // Z-Image sign: the transformer emits x₀ − ε; the pipeline negates it.
    func noisePred(_ x: MLXArray, _ sigma: Float) -> MLXArray { -field.velocity(x, sigma) }

    for stepIndex in 0..<scheduler.numInferenceSteps {
      let sigma = sigmas[stepIndex]
      let guidedNoise = noisePred(latents, sigma)
      let velocity = -guidedNoise
      let modelOutput = convert
        ? scheduler.modelInput(velocity: velocity, sample: latents, sigma: sigma)
        : velocity
      if scheduler.requiresIntermediateEvaluation,
         let intermediateSample = scheduler.intermediateStep(
           modelOutput: modelOutput, timestepIndex: stepIndex, sample: latents
         ) {
        let intermediateSigma = scheduler.intermediateSigma(timestepIndex: stepIndex)
          ?? sigmas[stepIndex + 1]
        let intermediateVelocity = -noisePred(intermediateSample, intermediateSigma)
        let intermediateOutput = convert
          ? scheduler.modelInput(
              velocity: intermediateVelocity, sample: intermediateSample, sigma: intermediateSigma)
          : intermediateVelocity
        latents = scheduler.finalizeStep(
          originalOutput: modelOutput,
          intermediateOutput: intermediateOutput,
          timestepIndex: stepIndex,
          sample: latents
        )
      } else {
        latents = scheduler.step(modelOutput: modelOutput, timestepIndex: stepIndex, sample: latents)
      }
      MLX.eval(latents)
    }
    return latents
  }

  func testRES2sOnZImagePathReconstructsX0() throws {
    let scheduler = try SchedulerFactory.create(
      kind: .res2s, sigmaSchedule: .flow, numInferenceSteps: 9,
      config: Self.zImageConfig(), mu: Self.zImageMu())
    let field = ModelOutputConventionTests.makeField(seed: 21)

    let post = Self.runZImageLoop(scheduler: scheduler, field: field, convert: true)
    let postErr = ModelOutputConventionTests.relativeError(post, field.x0)
    XCTAssertLessThanOrEqual(postErr, 1e-5, "post-fix res_2s must reconstruct x₀; got \(postErr)")

    let pre = Self.runZImageLoop(scheduler: scheduler, field: field, convert: false)
    let preErr = ModelOutputConventionTests.relativeError(pre, field.x0)
    XCTAssertGreaterThan(preErr, 1.0, "pre-fix res_2s should miss x₀ by >1.0; got \(preErr)")

    // The change is real and measured: pre and post are different latents.
    let delta = ModelOutputConventionTests.relativeError(pre, post)
    XCTAssertGreaterThan(delta, 1.0, "pre/post res_2s must differ; relative delta \(delta)")
  }

  func testRES2sOnZImagePathPreFixWasVelocityInExponentialFrame() throws {
    // Documents the pre-change failure mode precisely: fed velocity,
    // res_2s converges toward v = ε − x₀ rather than x₀.
    let scheduler = try SchedulerFactory.create(
      kind: .res2s, sigmaSchedule: .flow, numInferenceSteps: 20,
      config: Self.zImageConfig(), mu: Self.zImageMu())
    let field = ModelOutputConventionTests.makeField(seed: 22)
    let pre = Self.runZImageLoop(scheduler: scheduler, field: field, convert: false)
    let v = field.eps - field.x0
    let towardV = ModelOutputConventionTests.relativeError(pre, v)
    let towardX0 = ModelOutputConventionTests.relativeError(pre, field.x0)
    XCTAssertLessThan(towardV, towardX0)
  }

  /// AC-6: the default euler/flow path is untouched by the conversion —
  /// same bytes with the helper in the loop and without it.
  func testEulerFlowByteIdenticalWithConversionInLoop() throws {
    let scheduler = try SchedulerFactory.create(
      kind: .euler, sigmaSchedule: .flow, numInferenceSteps: 9,
      config: Self.zImageConfig(), mu: Self.zImageMu())
    XCTAssertEqual(scheduler.modelOutputConvention, .velocity)
    let field = ModelOutputConventionTests.makeField(seed: 23)
    let with = Self.runZImageLoop(scheduler: scheduler, field: field, convert: true)
    let without = Self.runZImageLoop(scheduler: scheduler, field: field, convert: false)
    XCTAssertEqual(with.asArray(Float.self), without.asArray(Float.self))
    // And with the exact field, euler lands on x₀ to float32 precision.
    let err = ModelOutputConventionTests.relativeError(with, field.x0)
    XCTAssertLessThanOrEqual(err, 1e-5)
  }
}
