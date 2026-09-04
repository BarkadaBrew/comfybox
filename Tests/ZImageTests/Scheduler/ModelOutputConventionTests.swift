import XCTest
import MLX
import MLXRandom
@testable import ZImage

/// WP-E2 — `ModelOutputConvention` and the `res_2s` correction
/// (FDD-krea2-raw-recipe §3.2, D2).
///
/// Weight-free. The model is replaced by a closed-form rectified-flow velocity
/// field, so the x₀ reconstruction runs in milliseconds with no GPU work of
/// consequence. Pins:
///   * AC-10 `res_2s` fed the data prediction x₀ = x − σ·v reconstructs x₀ to
///           ≤1e-5 relative over the krea2 schedule; fed velocity (the
///           pre-change behaviour) it misses by >1.0 relative.
///   * AC-11 every `SchedulerKind` reports a convention; `res_2s`, `res_3s`
///           and the WP-E13 tableau conformers (`ralston_2s/3s/4s` — RES4LYF
///           anchors their rows at the step's x₀ too) are `.dataPrediction`,
///           everything else is `.velocity`.
final class ModelOutputConventionTests: XCTestCase {

  // MARK: - AC-11

  func testConventionTable() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let expected: [SchedulerKind: ModelOutputConvention] = [
      .euler: .velocity,
      .heun: .velocity,
      .dpmplusplus2m: .velocity,
      .dpmplusplus2sa: .velocity,
      .deis: .velocity,
      .ddim: .velocity,
      .res2s: .dataPrediction,
      // WP-E13: RES4LYF anchors every tableau row at the step's x₀ and sigma,
      // so the linear-frame ralstons consume the data prediction too.
      .ralston2s: .dataPrediction,
      .ralston3s: .dataPrediction,
      .ralston4s: .dataPrediction,
      .res3s: .dataPrediction,
      // RES4LYF heun family (claude/krea2-spatial-noise): same x0 anchoring.
      .heun2s: .dataPrediction,
      .heun3s: .dataPrediction,
      // WP-E14: the DEIS ramp is anchored in BOTH halves — the ralston warm-up
      // and the multistep, whose recycled history is re-anchored at the
      // current step's x₀ and σ.
      .deis2m: .dataPrediction,
      .deis3m: .dataPrediction,
      .deis4m: .dataPrediction,
    ]
    // Every kind must be in the table — a new kind without a declared
    // convention is the silent-default hazard D2 exists to close.
    XCTAssertEqual(Set(expected.keys), Set(SchedulerKind.allCases))

    for kind in SchedulerKind.allCases {
      let scheduler = try SchedulerFactory.create(
        kind: kind, sigmaSchedule: .flow, numInferenceSteps: 9, config: config, seed: 1)
      XCTAssertEqual(
        scheduler.modelOutputConvention, expected[kind]!,
        "\(kind.rawValue) reports \(scheduler.modelOutputConvention)")
    }

    // The RES family and the WP-E13 tableau conformers take x₀; everything
    // else takes the velocity.
    let dataPrediction = SchedulerKind.allCases.filter { kind in
      (try? SchedulerFactory.create(
        kind: kind, sigmaSchedule: .flow, numInferenceSteps: 9, config: config, seed: 1))?
        .modelOutputConvention == .dataPrediction
    }
    XCTAssertEqual(
      dataPrediction,
      [.res2s, .ralston2s, .ralston3s, .ralston4s, .res3s, .heun2s, .heun3s, .deis2m, .deis3m, .deis4m])
  }

  /// The conversion helper the pipelines call: identity for velocity
  /// schedulers (so the euler/flow path stays byte-identical — AC-6), and
  /// x − σ·v for data-prediction schedulers, in the sample's dtype.
  func testModelInputConversion() {
    let config = FlowMatchSchedulerTests.makeConfig()
    let x = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float], [1, 1, 2, 2])
    let v = MLXArray([0.5, -0.5, 1.0, -1.0] as [Float], [1, 1, 2, 2])

    let euler = try! SchedulerFactory.create(kind: .euler, numInferenceSteps: 4, config: config)
    let asVelocity = euler.modelInput(velocity: v, sample: x, sigma: 0.75)
    XCTAssertEqual(asVelocity.asArray(Float.self), v.asArray(Float.self))

    let res2s = try! SchedulerFactory.create(kind: .res2s, numInferenceSteps: 4, config: config)
    let asX0 = res2s.modelInput(velocity: v, sample: x, sigma: 0.75)
    let expected: [Float] = [1.0 - 0.75 * 0.5, 2.0 + 0.75 * 0.5, 3.0 - 0.75, 4.0 + 0.75]
    let got = asX0.asArray(Float.self)
    for i in 0..<4 {
      XCTAssertEqual(got[i], expected[i], accuracy: 1e-6, "i=\(i)")
    }

    // dtype follows the sample, not the scalar (the Float * MLXArray rule).
    let xHalf = x.asType(.bfloat16)
    let vHalf = v.asType(.bfloat16)
    XCTAssertEqual(res2s.modelInput(velocity: vHalf, sample: xHalf, sigma: 0.75).dtype, .bfloat16)
    XCTAssertEqual(euler.modelInput(velocity: vHalf, sample: xHalf, sigma: 0.75).dtype, .bfloat16)
  }

  // MARK: - AC-10

  /// Rectified flow: x_σ = σ·ε + (1−σ)·x₀, model emits v = dx/dσ = ε − x₀.
  /// `evaluate` is the exact velocity of a fixed (x₀, ε) — independent of x.
  struct RectifiedFlowField {
    let x0: MLXArray
    let eps: MLXArray
    func velocity(_ x: MLXArray, _ sigma: Float) -> MLXArray { eps - x0 }
    func sample(at sigma: Float) -> MLXArray { sigma * eps + (1.0 - sigma) * x0 }
  }

  static func makeField(seed: UInt64 = 7, shape: [Int] = [1, 16, 8, 8]) -> RectifiedFlowField {
    let keys = MLXRandom.split(key: MLXRandom.key(seed))
    let x0 = MLXRandom.normal(shape, key: keys.0)
    let eps = MLXRandom.normal(shape, key: keys.1)
    MLX.eval(x0, eps)
    return RectifiedFlowField(x0: x0, eps: eps)
  }

  static func relativeError(_ got: MLXArray, _ want: MLXArray) -> Float {
    let diff = MLX.sqrt(MLX.sum(MLX.square(got - want))).item(Float.self)
    let norm = MLX.sqrt(MLX.sum(MLX.square(want))).item(Float.self)
    return diff / norm
  }

  /// Drive a 2-row scheduler over its whole grid with the pipeline's
  /// conversion discipline: one conversion per evaluation, after CFG.
  /// `convert == false` reproduces the pre-change behaviour (velocity fed raw).
  static func run(
    scheduler input: any ZImageScheduler,
    field: RectifiedFlowField,
    convert: Bool
  ) -> MLXArray {
    var scheduler = input
    scheduler.reset()
    let sigmas = scheduler.sigmas.asArray(Float.self)
    var x = field.sample(at: sigmas[0])
    for i in 0..<scheduler.numInferenceSteps {
      let sigma = sigmas[i]
      let v1 = field.velocity(x, sigma)
      let m1 = convert ? scheduler.modelInput(velocity: v1, sample: x, sigma: sigma) : v1
      if scheduler.requiresIntermediateEvaluation,
         let mid = scheduler.intermediateStep(modelOutput: m1, timestepIndex: i, sample: x) {
        let midSigma = scheduler.intermediateSigma(timestepIndex: i) ?? sigmas[i + 1]
        let v2 = field.velocity(mid, midSigma)
        let m2 = convert ? scheduler.modelInput(velocity: v2, sample: mid, sigma: midSigma) : v2
        x = scheduler.finalizeStep(
          originalOutput: m1, intermediateOutput: m2, timestepIndex: i, sample: x)
      } else {
        x = scheduler.step(modelOutput: m1, timestepIndex: i, sample: x)
      }
      MLX.eval(x)
    }
    return x
  }

  func testRES2sReconstructsX0() throws {
    // The krea2 schedule at 1024² (4096 tokens), 9 steps — the production grid.
    let mu = Krea2Sampling.mu(seqLen: 4096, align: 16)
    let scheduler = try SchedulerFactory.create(
      kind: .res2s, sigmaSchedule: .krea2, numInferenceSteps: 9,
      config: Krea2Sampling.schedulerConfig(), mu: mu)
    XCTAssertEqual(scheduler.modelOutputConvention, .dataPrediction)
    let field = Self.makeField()

    // Post-change: x₀ in, x₀ out.
    let reconstructed = Self.run(scheduler: scheduler, field: field, convert: true)
    let err = Self.relativeError(reconstructed, field.x0)
    XCTAssertLessThanOrEqual(err, 1e-5, "res_2s given x₀ must reconstruct x₀; relative error \(err)")

    // Pre-change: velocity in — dimensionally wrong, and measurably so.
    let wrong = Self.run(scheduler: scheduler, field: field, convert: false)
    let wrongErr = Self.relativeError(wrong, field.x0)
    XCTAssertGreaterThan(wrongErr, 1.0, "res_2s given velocity should miss x₀ by >1.0; got \(wrongErr)")
  }

  /// The same reconstruction holds on the other grids res_2s can be asked
  /// for, so the convention is a property of the solver, not of one schedule.
  func testRES2sReconstructsX0AcrossSchedules() throws {
    let field = Self.makeField(seed: 11)
    let config = Krea2Sampling.schedulerConfig()
    let mu = Krea2Sampling.mu(seqLen: 4096, align: 16)
    for steps in [4, 9, 20] {
      for schedule in SigmaScheduleKind.allCases {
        let scheduler = try SchedulerFactory.create(
          kind: .res2s, sigmaSchedule: schedule, numInferenceSteps: steps, config: config, mu: mu)
        let got = Self.run(scheduler: scheduler, field: field, convert: true)
        let err = Self.relativeError(got, field.x0)
        XCTAssertLessThanOrEqual(err, 1e-5, "\(schedule.rawValue) steps=\(steps): relative error \(err)")
      }
    }
  }

  /// Velocity schedulers are untouched by the conversion: with the exact
  /// field, euler/flow lands on x₀ with or without the helper in the loop,
  /// and both runs are the same bytes.
  func testVelocitySchedulersUnchangedByConversion() throws {
    let field = Self.makeField(seed: 3)
    let config = FlowMatchSchedulerTests.makeConfig()
    for kind in SchedulerKind.allCases
    where (try? SchedulerFactory.create(
      kind: kind, sigmaSchedule: .flow, numInferenceSteps: 9, config: config, seed: 5))?
      .modelOutputConvention != .dataPrediction {
      let scheduler = try SchedulerFactory.create(
        kind: kind, sigmaSchedule: .flow, numInferenceSteps: 9, config: config, seed: 5)
      let a = Self.run(scheduler: scheduler, field: field, convert: true)
      let b = Self.run(scheduler: scheduler, field: field, convert: false)
      XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self), "\(kind.rawValue) moved")
    }
  }
}
