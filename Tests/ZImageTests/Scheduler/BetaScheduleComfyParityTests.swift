import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E12 / WP-E12b — ComfyUI-exact `beta` / `beta57` (FDD-krea2-raw-recipe D5,
/// §3.11, AC-21 as re-pinned by Addendum A.1, AC-22).
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
/// Which **table** the PPF indexes is the model family's (A.1): ComfyUI
/// registers Krea 2 as `ModelSamplingFlux(shift=mu)` — a 10 000-entry table
/// `σ(t) = e^mu / (e^mu + 1/t − 1)` — so under `Krea2Sampling.schedulerConfig()`
/// the factory builds `SigmaSchedule.fluxSigmaTable(shift: mu, tableSize: 10000)`
/// and **requires `mu`**. Families that decode a `scheduler_config.json`
/// (Z-Image) stay on the 1000-entry DiscreteFlow table built from
/// `shift`/`num_train_timesteps`, unchanged and `mu`-free.
///
/// Weight-free, GPU-free.
final class BetaScheduleComfyParityTests: XCTestCase {

  /// Krea 2's ComfyUI registration shift (`comfy/supported_models.py`, D3).
  static let krea2Shift: Float = 1.15

  // MARK: - AC-21 (A.1): `beta(6)` at shift 1.15 matches ComfyUI's ModelSamplingFlux grid

  /// The A.1 grid, built the way a render builds it: `resolveShift(explicit: 1.15)`
  /// → `(config, mu)` → `SchedulerFactory.resolveSigmas(.beta)`. `shift` is mu,
  /// the table is the Flux table, and the values are `comfy_sigmas.json`
  /// `beta.flux.6` (scipy re-run 2026-08-22 agrees to 1e-6).
  func testBetaMatchesComfyAtKrea2Shift() throws {
    let pinned: [Float] = [1.0, 0.969095, 0.892582, 0.759584, 0.545649, 0.241540, 0.0]
    let resolved = try Krea2Sampling.resolveShift(explicit: Self.krea2Shift, seqLen: 4096, align: 16)

    let viaFactory = try SchedulerFactory.resolveSigmas(
      schedule: .beta, numSteps: 6, config: resolved.config, mu: resolved.mu)
    XCTAssertEqual(viaFactory.count, pinned.count, "steps+1 sigmas at 6 steps (no de-dup on the 10 000-entry table)")
    for (i, (got, want)) in zip(viaFactory, pinned).enumerated() {
      XCTAssertEqual(got, want, accuracy: 1e-5, "beta(6, shift 1.15) via factory, i=\(i)")
    }

    let direct = SigmaSchedule.beta(
      numSteps: 6, sigmaTable: SigmaSchedule.fluxSigmaTable(shift: Self.krea2Shift, tableSize: 10000))
    XCTAssertEqual(direct, viaFactory, "the factory's .beta under the Krea 2 family is SigmaSchedule.beta over fluxSigmaTable(shift: mu)")

    // The withdrawn grid (DiscreteFlow, linear shift 1.15) is measurably elsewhere.
    XCTAssertNotEqual(viaFactory[1], 0.919919, accuracy: 1e-3, "the 1000-entry DiscreteFlow σ₁ must not come back")
  }

  /// `beta57(6)` at shift 1.15 — the second A.1 pin.
  func testBeta57MatchesComfyAtKrea2Shift() throws {
    let pinned: [Float] = [1.0, 0.941558, 0.823777, 0.640931, 0.390072, 0.125063, 0.0]
    let resolved = try Krea2Sampling.resolveShift(explicit: Self.krea2Shift, seqLen: 4096, align: 16)
    let viaFactory = try SchedulerFactory.resolveSigmas(
      schedule: .beta57, numSteps: 6, config: resolved.config, mu: resolved.mu)
    XCTAssertEqual(viaFactory.count, pinned.count)
    for (i, (got, want)) in zip(viaFactory, pinned).enumerated() {
      XCTAssertEqual(got, want, accuracy: 1e-5, "beta57(6, shift 1.15) via factory, i=\(i)")
    }
  }

  /// `beta`/`beta57` at 6/9/30 steps equal the oracle dump under
  /// `ModelSamplingFlux(shift=1.15)` **through the factory** with the Krea 2
  /// family config — float32 values, so 1e-6. This is the grid the published
  /// workflow's stage 1 ran on.
  func testKrea2FamilyBetaAndBeta57MatchFluxFixtures() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let params = try XCTUnwrap(fixture["beta_params"] as? [String: Any])
    let flux = try XCTUnwrap((fixture["model_samplings"] as? [String: Any])?["flux"] as? [String: Any])
    XCTAssertEqual(try XCTUnwrap(flux["table_size"] as? Int), 10000)
    let resolved = try Krea2Sampling.resolveShift(explicit: Self.krea2Shift, seqLen: 4096, align: 16)

    for (key, schedule) in [("beta", SigmaScheduleKind.beta), ("beta57", .beta57)] {
      XCTAssertNotNil(params[key])
      let grids = try XCTUnwrap((fixture[key] as? [String: Any])?["flux"] as? [String: Any])
      for steps in [6, 9, 30] {
        let want = try SchedulerOracleFixtures.doubles(grids["\(steps)"], "\(key).flux.\(steps)")
        let got = try SchedulerFactory.resolveSigmas(
          schedule: schedule, numSteps: steps, config: resolved.config, mu: resolved.mu)
        XCTAssertEqual(got.count, want.count, "\(key) flux \(steps): count")
        for (i, (g, w)) in zip(got, want).enumerated() {
          XCTAssertEqual(Double(g), w, accuracy: 1e-6, "\(key) flux \(steps) i=\(i)")
        }
      }
    }
  }

  /// The dynamic case (`shift` omitted) indexes the same Flux table at the
  /// resolution-derived mu — ComfyUI's `ModelSamplingFlux` node with
  /// base/max shift does exactly this. At 1024² mu is exactly 0.90625;
  /// values from a scipy re-run of `beta_scheduler` on 2026-08-22.
  func testDynamicMuBetaIsTheFluxGridAtResolutionMu() throws {
    let resolved = try Krea2Sampling.resolveShift(explicit: nil, seqLen: 4096, align: 16)
    XCTAssertEqual(resolved.mu, 0.90625)
    let beta = try SchedulerFactory.resolveSigmas(
      schedule: .beta, numSteps: 6, config: resolved.config, mu: resolved.mu)
    let beta57 = try SchedulerFactory.resolveSigmas(
      schedule: .beta57, numSteps: 6, config: resolved.config, mu: resolved.mu)
    let wantBeta: [Float] = [1.0, 0.960898, 0.866880, 0.712314, 0.484844, 0.199727, 0.0]
    let wantBeta57: [Float] = [1.0, 0.926610, 0.785565, 0.583135, 0.333864, 0.100735, 0.0]
    XCTAssertEqual(beta.count, 7)
    XCTAssertEqual(beta57.count, 7)
    for i in 0..<7 {
      XCTAssertEqual(beta[i], wantBeta[i], accuracy: 1e-5, "beta dynamic i=\(i)")
      XCTAssertEqual(beta57[i], wantBeta57[i], accuracy: 1e-5, "beta57 dynamic i=\(i)")
    }
  }

  /// `beta`/`beta57` at 6/9/30 steps equal the oracle dump under
  /// `ModelSamplingDiscreteFlow(shift=1.15)` — the 1000-entry table families
  /// with a `scheduler_config.json` (Z-Image) index, and the D5 before/after
  /// record. Reached directly and through the factory with a decoded-style
  /// config; **no `mu`** is involved on this path.
  func testBetaAndBeta57MatchDiscreteFlowFixtures() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let params = try XCTUnwrap(fixture["beta_params"] as? [String: Any])
    let table = try XCTUnwrap(fixture["model_samplings"] as? [String: Any])
    let df = try XCTUnwrap(table["discrete_flow"] as? [String: Any])
    let tableSize = try XCTUnwrap(df["table_size"] as? Int)
    XCTAssertEqual(tableSize, 1000)
    let config = ZImageSchedulerConfig(numTrainTimesteps: tableSize, shift: Self.krea2Shift, useDynamicShifting: false)
    XCTAssertEqual(config.modelSampling, .discreteFlow, "a decoded-style config is DiscreteFlow by default")

    for (key, schedule) in [("beta", SigmaScheduleKind.beta), ("beta57", .beta57)] {
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
        let viaFactory = try SchedulerFactory.resolveSigmas(schedule: schedule, numSteps: steps, config: config, mu: nil)
        XCTAssertEqual(viaFactory, got, "\(key) \(steps): DiscreteFlow families reach the same grid through the factory, mu-free")
      }
    }
  }

  /// The factory's `.beta57` is `(α 0.5, β 0.7)` on the same table (§3.11),
  /// under the Krea 2 family's Flux table.
  func testFactoryBeta57IsAlphaHalfBetaSevenTenths() throws {
    let config = Krea2Sampling.schedulerConfig()
    let mu = Self.krea2Shift
    let fluxTable = SigmaSchedule.fluxSigmaTable(shift: mu, tableSize: 10000)
    for steps in [6, 9, 30] {
      let viaFactory = try SchedulerFactory.resolveSigmas(schedule: .beta57, numSteps: steps, config: config, mu: mu)
      let direct = SigmaSchedule.beta(numSteps: steps, sigmaTable: fluxTable, alpha: 0.5, betaParam: 0.7)
      XCTAssertEqual(viaFactory, direct, "beta57 @ \(steps)")
      let beta = try SchedulerFactory.resolveSigmas(schedule: .beta, numSteps: steps, config: config, mu: mu)
      XCTAssertNotEqual(viaFactory, beta, "beta57 @ \(steps) is a different PPF from beta")
    }
  }

  // MARK: - The table is the family's; mu is required on the Flux table (fail loud)

  /// Under the Krea 2 family every table-backed schedule needs `mu` — there is
  /// no shift to build the Flux table from otherwise, and a default would be a
  /// silent grid.
  func testKrea2FamilyTableSchedulesRequireMu() {
    let config = Krea2Sampling.schedulerConfig()
    for schedule in [SigmaScheduleKind.beta, .beta57, .karras, .exponential] {
      XCTAssertThrowsError(
        try SchedulerFactory.resolveSigmas(schedule: schedule, numSteps: 9, config: config, mu: nil),
        "\(schedule.rawValue) with mu: nil must throw"
      ) { error in
        XCTAssertEqual(error as? SchedulerFactoryError, .missingMu(schedule))
        XCTAssertTrue("\(error)".contains(schedule.rawValue), "the error names the schedule: \(error)")
      }
      XCTAssertThrowsError(
        try SchedulerFactory.create(
          kind: .euler, sigmaSchedule: schedule, numInferenceSteps: 9, config: config, mu: nil))
    }
  }

  /// DiscreteFlow families (Z-Image) are untouched: `mu` is neither required
  /// nor consulted by the table-backed schedules, so a `mu` passed for `.flow`'s
  /// dynamic warp cannot move `beta`/`karras`.
  func testDiscreteFlowFamilyIgnoresMuForTableSchedules() throws {
    let config = FlowMatchSchedulerTests.makeConfig()  // the Z-Image Turbo scheduler_config.json shape
    XCTAssertEqual(config.modelSampling, .discreteFlow)
    for schedule in [SigmaScheduleKind.beta, .beta57, .karras, .exponential] {
      let withoutMu = try SchedulerFactory.resolveSigmas(schedule: schedule, numSteps: 9, config: config, mu: nil)
      let withMu = try SchedulerFactory.resolveSigmas(schedule: schedule, numSteps: 9, config: config, mu: 1.15)
      XCTAssertEqual(withoutMu, withMu, "\(schedule.rawValue): DiscreteFlow table does not read mu")
      XCTAssertEqual(withoutMu.count, 10)
    }
  }

  // MARK: - The Flux sigma table (what ComfyUI registers Krea 2 as; E18 finding, A.1)

  /// `σ[i] = e^shift / (e^shift + 1/t − 1)`, `t = (i+1)/10000` — ComfyUI's
  /// `ModelSamplingFlux.set_parameters`. `σ_min` and `σ_max` equal the fixture's.
  func testFluxSigmaTablePins() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let flux = try XCTUnwrap((fixture["model_samplings"] as? [String: Any])?["flux"] as? [String: Any])
    let tableSize = try XCTUnwrap(flux["table_size"] as? Int)
    let sigmaMin = try XCTUnwrap(flux["sigma_min"] as? Double)
    XCTAssertEqual(tableSize, 10000)

    let fluxTable = SigmaSchedule.fluxSigmaTable(shift: Self.krea2Shift, tableSize: tableSize)
    XCTAssertEqual(fluxTable.count, tableSize)
    XCTAssertEqual(Double(fluxTable[0]), sigmaMin, accuracy: 1e-9, "σ_min = σ(1/10000)")
    XCTAssertEqual(fluxTable[tableSize - 1], 1.0, "σ_max = σ(1) = 1.0 exactly")
    for i in 1..<fluxTable.count {
      XCTAssertGreaterThan(fluxTable[i], fluxTable[i - 1], "table is ascending at \(i)")
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

  /// On the 1000-entry DiscreteFlow table (the families that decode a
  /// `scheduler_config.json`) the reported count equals the requested count at
  /// every production step budget — no shrink on the live path — and the first
  /// collisions land exactly where ComfyUI's do (`scipy.stats.beta.ppf` re-run
  /// 2026-08-22: `beta` first de-dups at 139 steps → 138, `beta57` at 97 → 96;
  /// `beta57(100)` → 99).
  func testDedupeOnThousandEntryTableMatchesComfy() throws {
    let config = ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: Self.krea2Shift, useDynamicShifting: false)
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

  /// On the Krea 2 family's 10 000-entry Flux table no production budget
  /// shrinks either, and the first collisions are far out (scipy re-run
  /// 2026-08-22 at shift 1.15 and at mu 0.90625 alike: `beta` 552 → 551,
  /// `beta57` 307 → 306).
  func testDedupeOnFluxTableMatchesComfy() throws {
    let config = Krea2Sampling.schedulerConfig()
    for mu: Float in [Self.krea2Shift, 0.90625] {
      for steps in [4, 6, 9, 12, 20, 30, 52] {
        for schedule in [SigmaScheduleKind.beta, .beta57] {
          let scheduler = try SchedulerFactory.create(
            kind: .euler, sigmaSchedule: schedule, numInferenceSteps: steps, config: config, mu: mu)
          XCTAssertEqual(scheduler.numInferenceSteps, steps, "\(schedule.rawValue) @ \(steps), mu \(mu)")
        }
      }
      let pins: [(SigmaScheduleKind, Int, Int)] = [
        (.beta, 551, 551), (.beta, 552, 551),
        (.beta57, 306, 306), (.beta57, 307, 306),
      ]
      for (schedule, requested, effective) in pins {
        let sigmas = try SchedulerFactory.resolveSigmas(
          schedule: schedule, numSteps: requested, config: config, mu: mu)
        XCTAssertEqual(sigmas.count - 1, effective, "\(schedule.rawValue) @ \(requested) → \(effective), mu \(mu)")
      }
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
  /// where ComfyUI's is 0.9199. Both numbers are pinned here — on the
  /// 1000-entry DiscreteFlow table D5 measured them on, which A.1 keeps "only
  /// as the D5 before/after record". The Krea 2 family itself (Flux table,
  /// `shift` = mu 1.15) lands at 0.969095 (AC-21 as re-pinned) — also pinned,
  /// so the three grids can never be confused for one another.
  func testBeforeAfterFixture() throws {
    // The old factory path: flowMatchingSigmaBounds under shift 1.15 → (0.00114983, 1.0).
    let shiftedMin: Float = 1.15 * 0.001 / (1 + 0.15 * 0.001)
    let before = Self.preChangeBeta(numSteps: 6, sigmaMin: shiftedMin, sigmaMax: 1.0)
    let discreteFlowConfig = ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: 1.15, useDynamicShifting: false)
    let after = try SchedulerFactory.resolveSigmas(
      schedule: .beta, numSteps: 6, config: discreteFlowConfig, mu: nil)
    let krea2 = try SchedulerFactory.resolveSigmas(
      schedule: .beta, numSteps: 6, config: Krea2Sampling.schedulerConfig(), mu: 1.15)

    XCTAssertEqual(before[1], 0.1596, accuracy: 5e-4, "pre-E12 σ₁ (D5)")
    XCTAssertEqual(after[1], 0.9199, accuracy: 5e-5, "ComfyUI DiscreteFlow σ₁ (D5 record)")
    XCTAssertEqual(krea2[1], 0.969095, accuracy: 1e-5, "ComfyUI ModelSamplingFlux σ₁ — what Krea 2 renders on (A.1)")
    XCTAssertGreaterThan(abs(after[1] - before[1]), 0.7, "the two schedules disagree wildly; the rename-free replacement is deliberate")
    XCTAssertGreaterThan(abs(krea2[1] - after[1]), 0.04, "and the two ComfyUI tables are not interchangeable either")
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
