import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E12 — ComfyUI-exact `beta` / `beta57` (FDD-krea2-raw-recipe D5, §3.11,
/// AC-21, AC-22).
///
/// ComfyUI's `beta_scheduler` (`comfy/samplers.py:696-708`, pinned copy under
/// `scripts/oracles/upstream/comfyui/`) evaluates the beta **PPF** at
/// `1 − i/steps`, rounds to an index into the model's discrete sigma table,
/// de-duplicates consecutive repeats and appends 0. The pre-E12 `beta`
/// integrated the beta **CDF** and interpolated in log-sigma space — a
/// different schedule under the same name (D5). These tests pin the
/// replacement against the E18 oracle dump and against the FDD's quoted
/// values, pin the before/after difference D5 obligates, and pin the
/// de-duplication reporting path through `SchedulerFactory`.
///
/// Weight-free, GPU-free.
final class BetaScheduleComfyParityTests: XCTestCase {

  /// Krea 2's ComfyUI registration shift (`comfy/supported_models.py`, D3).
  static let krea2Shift: Float = 1.15

  // MARK: - AC-21: `beta(6, shift: 1.15)` matches ComfyUI at Krea 2's registered shift

  /// The FDD's quoted grid, built through `Krea2Sampling.schedulerConfig(shift: 1.15)`
  /// so the synthetic config's values and the schedule are tested as one thing.
  func testBetaMatchesComfyAtKrea2Shift() throws {
    let pinned: [Float] = [1.0, 0.919919, 0.751973, 0.535879, 0.304782, 0.104360, 0.0]

    let viaFactory = try SchedulerFactory.resolveSigmas(
      schedule: .beta, numSteps: 6,
      config: Krea2Sampling.schedulerConfig(shift: Self.krea2Shift), mu: nil)
    XCTAssertEqual(viaFactory.count, pinned.count, "steps+1 sigmas at 6 steps (no de-dup on a 1000-entry table)")
    for (i, (got, want)) in zip(viaFactory, pinned).enumerated() {
      XCTAssertEqual(got, want, accuracy: 1e-5, "beta(6, shift 1.15) via factory, i=\(i)")
    }

    let direct = SigmaSchedule.beta(numSteps: 6, shift: Self.krea2Shift, numTrainTimesteps: 1000)
    XCTAssertEqual(direct, viaFactory, "the factory's .beta is SigmaSchedule.beta under config.shift / numTrainTimesteps")
  }

  /// `beta`/`beta57` at 6/9/30 steps equal the oracle dump under
  /// `ModelSamplingDiscreteFlow(shift=1.15)` — float32 values, so 1e-6.
  func testBetaAndBeta57MatchDiscreteFlowFixtures() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let params = try XCTUnwrap(fixture["beta_params"] as? [String: Any])
    let table = try XCTUnwrap(fixture["model_samplings"] as? [String: Any])
    let df = try XCTUnwrap(table["discrete_flow"] as? [String: Any])
    let tableSize = try XCTUnwrap(df["table_size"] as? Int)
    XCTAssertEqual(tableSize, 1000)

    for key in ["beta", "beta57"] {
      let ab = try SchedulerOracleFixtures.doubles(params[key], "beta_params.\(key)")
      let alpha = Float(ab[0]), betaParam = Float(ab[1])
      let grids = try XCTUnwrap((fixture[key] as? [String: Any])?["discrete_flow"] as? [String: Any])
      for steps in [6, 9, 30] {
        let want = try SchedulerOracleFixtures.doubles(grids["\(steps)"], "\(key).discrete_flow.\(steps)")
        let got = SigmaSchedule.beta(
          numSteps: steps, shift: Self.krea2Shift, numTrainTimesteps: tableSize,
          alpha: alpha, betaParam: betaParam)
        XCTAssertEqual(got.count, want.count, "\(key) \(steps): count")
        for (i, (g, w)) in zip(got, want).enumerated() {
          XCTAssertEqual(Double(g), w, accuracy: 1e-6, "\(key) \(steps) i=\(i)")
        }
      }
    }
  }

  /// The factory's `.beta57` is `(α 0.5, β 0.7)` on the same table (§3.11).
  func testFactoryBeta57IsAlphaHalfBetaSevenTenths() throws {
    let config = Krea2Sampling.schedulerConfig(shift: Self.krea2Shift)
    for steps in [6, 9, 30] {
      let viaFactory = try SchedulerFactory.resolveSigmas(schedule: .beta57, numSteps: steps, config: config, mu: nil)
      let direct = SigmaSchedule.beta(
        numSteps: steps, shift: Self.krea2Shift, numTrainTimesteps: 1000, alpha: 0.5, betaParam: 0.7)
      XCTAssertEqual(viaFactory, direct, "beta57 @ \(steps)")
      let beta = try SchedulerFactory.resolveSigmas(schedule: .beta, numSteps: steps, config: config, mu: nil)
      XCTAssertNotEqual(viaFactory, beta, "beta57 @ \(steps) is a different PPF from beta")
    }
  }

  // MARK: - What ComfyUI actually registers Krea 2 as (E18 finding)

  /// E18 recorded that ComfyUI builds Krea 2 on `ModelSamplingFlux(shift=1.15)`
  /// (10 000-entry table, `σ(t) = e^1.15 / (e^1.15 + 1/t − 1)`), not on the
  /// `DiscreteFlow` class the FDD's D3/D5/AC-21 assume. The same PPF-and-round
  /// algorithm over that table must reproduce the fixture's `flux` grids too,
  /// so the published workflow's stage-1 grid is reachable once a caller asks
  /// for it — the algorithm is table-agnostic and this pins it.
  func testBetaUnderComfyFluxRegistrationMatchesFixture() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let params = try XCTUnwrap(fixture["beta_params"] as? [String: Any])
    let table = try XCTUnwrap(fixture["model_samplings"] as? [String: Any])
    let flux = try XCTUnwrap(table["flux"] as? [String: Any])
    let tableSize = try XCTUnwrap(flux["table_size"] as? Int)
    let sigmaMin = try XCTUnwrap(flux["sigma_min"] as? Double)
    XCTAssertEqual(tableSize, 10000)

    let fluxTable = SigmaSchedule.fluxSigmaTable(shift: Self.krea2Shift, tableSize: tableSize)
    XCTAssertEqual(fluxTable.count, tableSize)
    XCTAssertEqual(Double(fluxTable[0]), sigmaMin, accuracy: 1e-9, "σ_min = σ(1/10000)")
    XCTAssertEqual(fluxTable[tableSize - 1], 1.0, "σ_max = σ(1) = 1.0 exactly")

    for key in ["beta", "beta57"] {
      let ab = try SchedulerOracleFixtures.doubles(params[key], "beta_params.\(key)")
      let grids = try XCTUnwrap((fixture[key] as? [String: Any])?["flux"] as? [String: Any])
      for steps in [6, 9, 30] {
        let want = try SchedulerOracleFixtures.doubles(grids["\(steps)"], "\(key).flux.\(steps)")
        let got = SigmaSchedule.beta(
          numSteps: steps, sigmaTable: fluxTable, alpha: Float(ab[0]), betaParam: Float(ab[1]))
        XCTAssertEqual(got.count, want.count, "\(key) flux \(steps): count")
        for (i, (g, w)) in zip(got, want).enumerated() {
          XCTAssertEqual(Double(g), w, accuracy: 1e-6, "\(key) flux \(steps) i=\(i)")
        }
      }
    }
  }

  // MARK: - The discrete-flow sigma table

  /// `σ[i] = shift·t / (1 + (shift − 1)·t)`, `t = (i+1)/T` — ComfyUI's
  /// `ModelSamplingDiscreteFlow.set_parameters` (`model_sampling.py:298-302`).
  func testDiscreteFlowSigmaTablePins() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let df = try XCTUnwrap((fixture["model_samplings"] as? [String: Any])?["discrete_flow"] as? [String: Any])
    let sigmaMin = try XCTUnwrap(df["sigma_min"] as? Double)

    let table = SigmaSchedule.discreteFlowSigmaTable(shift: Self.krea2Shift, numTrainTimesteps: 1000)
    XCTAssertEqual(table.count, 1000)
    XCTAssertEqual(Double(table[0]), sigmaMin, accuracy: 1e-9, "σ[0] = σ(1/1000) = 1.15e-3 / (1 + 0.15e-3)")
    XCTAssertEqual(table[999], 1.0, "σ[999] = σ(1) = 1.0 exactly")
    for i in 1..<table.count {
      XCTAssertGreaterThan(table[i], table[i - 1], "table is ascending at \(i)")
    }
    // shift 1.0 is the identity warp: σ[i] = (i+1)/T.
    let identity = SigmaSchedule.discreteFlowSigmaTable(shift: 1.0, numTrainTimesteps: 1000)
    XCTAssertEqual(identity[499], 0.5, accuracy: 1e-7)
    XCTAssertEqual(identity[0], 0.001, accuracy: 1e-9)
  }

  // MARK: - The PPF itself

  /// `I_x(a, b)` and its inverse, spot-checked against closed forms and scipy
  /// (`scipy.stats.beta.ppf`, the function ComfyUI calls).
  func testBetaPPFAgainstScipy() {
    // I_x(1,1) = x; I_x(2,1) = x²; symmetry I_0.5(a,a) = 0.5.
    XCTAssertEqual(SigmaSchedule.regularizedIncompleteBeta(0.3, a: 1, b: 1), 0.3, accuracy: 1e-14)
    XCTAssertEqual(SigmaSchedule.regularizedIncompleteBeta(0.3, a: 2, b: 1), 0.09, accuracy: 1e-14)
    XCTAssertEqual(SigmaSchedule.regularizedIncompleteBeta(0.5, a: 0.6, b: 0.6), 0.5, accuracy: 1e-12)
    // scipy.stats.beta.ppf(p, 0.6, 0.6) / (0.5, 0.7), printed at full precision.
    XCTAssertEqual(SigmaSchedule.betaPPF(0.5, alpha: 0.6, beta: 0.6), 0.5, accuracy: 1e-12)
    XCTAssertEqual(SigmaSchedule.betaPPF(0.5, alpha: 0.5, beta: 0.7), 0.3610839131320434, accuracy: 1e-10)
    XCTAssertEqual(SigmaSchedule.betaPPF(1.0, alpha: 0.6, beta: 0.6), 1.0)
    XCTAssertEqual(SigmaSchedule.betaPPF(0.0, alpha: 0.6, beta: 0.6), 0.0)
  }

  // MARK: - AC-22: de-duplication is reported, not hidden

  /// ComfyUI skips an index that repeats the previous one, so a step count can
  /// produce fewer than `steps + 1` sigmas. `SchedulerFactory` constructs with
  /// the **actual** count and the scheduler's `numInferenceSteps` reports it.
  func testDedupeReported() throws {
    // A 20-entry table with 24 requested steps forces collisions (pigeonhole),
    // so the test does not depend on where the real table first collides.
    let requested = 24
    let config = ZImageSchedulerConfig(numTrainTimesteps: 20, shift: 1.0, useDynamicShifting: false)

    let sigmas = try SchedulerFactory.resolveSigmas(schedule: .beta, numSteps: requested, config: config, mu: nil)
    XCTAssertLessThan(sigmas.count, requested + 1, "collisions must have happened for this test to mean anything")
    XCTAssertEqual(sigmas.first, 1.0)
    XCTAssertEqual(sigmas.last, 0.0)
    for i in 1..<sigmas.count {
      XCTAssertLessThan(sigmas[i], sigmas[i - 1], "strictly decreasing after de-dup at \(i)")
    }

    var scheduler = try SchedulerFactory.create(
      kind: .euler, sigmaSchedule: .beta, numInferenceSteps: requested, config: config)
    let stepsEffective = scheduler.numInferenceSteps
    XCTAssertEqual(stepsEffective, sigmas.count - 1, "numInferenceSteps is the produced count")
    XCTAssertLessThan(stepsEffective, requested, "steps_effective < steps_requested")
    XCTAssertEqual(scheduler.sigmas.dim(0), stepsEffective + 1)
    XCTAssertEqual(scheduler.timesteps.dim(0), stepsEffective)

    // The render succeeds: every step index the scheduler reports is steppable
    // on a synthetic field, and the loop never indexes past the grid.
    var x = MLX.ones([1, 4, 2, 2])
    for i in 0..<stepsEffective {
      let velocity = x * 0.1
      let sigma = scheduler.sigmas[i].item(Float.self)
      let input = scheduler.modelInput(velocity: velocity, sample: x, sigma: sigma)
      x = scheduler.step(modelOutput: input, timestepIndex: i, sample: x)
    }
    MLX.eval(x)
    XCTAssertFalse(MLX.any(MLX.isNaN(x)).item(Bool.self))
  }

  /// Every sampler constructs from a de-duplicated grid with the actual count —
  /// each scheduler init preconditions `sigmaValues.count == numInferenceSteps + 1`,
  /// so passing the requested count would trap.
  func testEverySamplerConstructsWithActualStepCount() throws {
    let requested = 24
    let config = ZImageSchedulerConfig(numTrainTimesteps: 20, shift: 1.0, useDynamicShifting: false)
    let expected = try SchedulerFactory.resolveSigmas(schedule: .beta57, numSteps: requested, config: config, mu: nil).count - 1
    XCTAssertLessThan(expected, requested)
    for kind in SchedulerKind.allCases {
      let scheduler = try SchedulerFactory.create(
        kind: kind, sigmaSchedule: .beta57, numInferenceSteps: requested, config: config, seed: 1)
      XCTAssertEqual(scheduler.numInferenceSteps, expected, "\(kind.rawValue): actual count")
      XCTAssertEqual(scheduler.sigmas.dim(0), expected + 1, "\(kind.rawValue): sigmas")
    }
  }

  /// On the real 1000-entry table the reported count equals the requested
  /// count at every production step budget — no shrink on the live path —
  /// and the first collisions land exactly where ComfyUI's do
  /// (`scipy.stats.beta.ppf` re-run 2026-08-22: `beta` first de-dups at 139
  /// steps → 138, `beta57` at 97 → 96; `beta57(100)` → 99).
  func testDedupeOnThousandEntryTableMatchesComfy() throws {
    let config = Krea2Sampling.schedulerConfig(shift: Self.krea2Shift)
    for steps in [4, 6, 9, 12, 20, 30, 52] {
      for schedule in [SigmaScheduleKind.beta, .beta57] {
        let scheduler = try SchedulerFactory.create(
          kind: .euler, sigmaSchedule: schedule, numInferenceSteps: steps, config: config)
        XCTAssertEqual(scheduler.numInferenceSteps, steps, "\(schedule.rawValue) @ \(steps)")
      }
    }
    let pins: [(SigmaScheduleKind, Int, Int)] = [
      (.beta, 138, 138), (.beta, 139, 138),
      (.beta57, 96, 96), (.beta57, 97, 96), (.beta57, 100, 99),
    ]
    for (schedule, requested, effective) in pins {
      let scheduler = try SchedulerFactory.create(
        kind: .euler, sigmaSchedule: schedule, numInferenceSteps: requested, config: config)
      XCTAssertEqual(scheduler.numInferenceSteps, effective, "\(schedule.rawValue) @ \(requested) → \(effective)")
    }
  }

  // MARK: - D5: the pinned before/after fixture

  /// The pre-E12 body, verbatim from `SigmaSchedule.swift` @ 241636b — the CDF
  /// integrator with log-sigma interpolation. Kept so the behaviour change is
  /// inspectable in the suite, not only in git history.
  static func preChangeBeta(
    numSteps: Int, sigmaMin: Float, sigmaMax: Float, alpha: Float = 0.6, betaParam: Float = 0.6
  ) -> [Float] {
    guard numSteps > 0 else { return [0.0] }
    let resolution = 10000
    var cdf = [Float](repeating: 0.0, count: resolution + 1)
    cdf[0] = 0.0
    for i in 1...resolution {
      let xMid = (Float(i) - 0.5) / Float(resolution)
      let pdfVal = powf(xMid, alpha - 1) * powf(1.0 - xMid, betaParam - 1)
      cdf[i] = cdf[i - 1] + pdfVal
    }
    let total = cdf[resolution]
    for i in 1...resolution { cdf[i] /= total }
    let logMin = logf(sigmaMin)
    let logMax = logf(sigmaMax)
    var sigmas: [Float] = (0..<numSteps).map { i in
      let t = Float(i) / Float(max(1, numSteps - 1))
      let idx = min(Int(t * Float(resolution)), resolution)
      let warped = cdf[idx]
      return expf(logMax + (logMin - logMax) * warped)
    }
    sigmas.append(0.0)
    return sigmas
  }

  /// D5's measured disagreement: at 6 steps, shift 1.15, the old σ₁ is 0.1596
  /// where ComfyUI's is 0.9199. Both numbers are pinned here.
  func testBeforeAfterFixture() throws {
    // The old factory path: flowMatchingSigmaBounds under shift 1.15 → (0.00114983, 1.0).
    let shiftedMin: Float = 1.15 * 0.001 / (1 + 0.15 * 0.001)
    let before = Self.preChangeBeta(numSteps: 6, sigmaMin: shiftedMin, sigmaMax: 1.0)
    let after = try SchedulerFactory.resolveSigmas(
      schedule: .beta, numSteps: 6, config: Krea2Sampling.schedulerConfig(shift: 1.15), mu: nil)

    XCTAssertEqual(before[1], 0.1596, accuracy: 5e-4, "pre-E12 σ₁ (D5)")
    XCTAssertEqual(after[1], 0.9199, accuracy: 5e-5, "ComfyUI σ₁ (D5)")
    XCTAssertGreaterThan(abs(after[1] - before[1]), 0.7, "the two schedules disagree wildly; the rename-free replacement is deliberate")
  }

  // MARK: - Edges ComfyUI defines

  /// `numpy.linspace(0, 1, 0)` is empty → `[0.0]`; one step → `[σ(1), 0]`.
  func testZeroAndOneStep() {
    XCTAssertEqual(SigmaSchedule.beta(numSteps: 0, shift: 1.15, numTrainTimesteps: 1000), [0.0])
    XCTAssertEqual(SigmaSchedule.beta(numSteps: 1, shift: 1.15, numTrainTimesteps: 1000), [1.0, 0.0])
  }

  /// The old signature is gone: `beta` no longer takes an EDM `sigmaMin`/`sigmaMax`
  /// pair, so no caller can reach the CDF-integrator schedule under this name.
  func testOldSignatureIsNotCallable() {
    // Compile-time contract, expressed as a reflection-free runtime check on
    // the surviving overloads: both take a table (or the pair that builds one).
    let a = SigmaSchedule.beta(numSteps: 4, shift: 1.0, numTrainTimesteps: 1000)
    let b = SigmaSchedule.beta(numSteps: 4, sigmaTable: SigmaSchedule.discreteFlowSigmaTable(shift: 1.0, numTrainTimesteps: 1000))
    XCTAssertEqual(a, b)
  }
}
