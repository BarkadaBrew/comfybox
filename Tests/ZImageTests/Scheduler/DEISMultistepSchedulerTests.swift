import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E14 — `deis_2m` / `deis_3m` / `deis_4m` (FDD-krea2-raw-recipe §3.12,
/// AC-23/24/25/26, Addendum A.1).
///
/// Three behaviours, from three different sources, and the distinction matters:
///
///  1. **The `rhoab` coefficients** are RES4LYF's closed form
///     (`beta/deis_coefficients.py:86-121`, `deis_mode="rhoab"`). Pinned against
///     E18's `res4lyf_deis_coeffs.json` oracle — never re-derived here.
///  2. **The order ramp with a ralston warm-up** is
///     `beta/rk_coefficients_beta.py:1374-1393`: while
///     `step < order + multistep_extra_initial_steps` (the latter defaults to
///     **1**, `:1343`), `deis_Nm` runs `ralston_Ns` instead. So `deis_3m` warms
///     up for **4** steps, `deis_2m` for 3 and `deis_4m` for 5 (AC-24 as
///     corrected by Addendum A.1).
///  3. **The multistep assembly** is `rk_sampler_beta.py`: `b = coeff/h`,
///     one model call per step (`rows − multistep_stages − row_offset + 1`),
///     and the previous steps' DATA predictions re-anchored at the CURRENT
///     step's `x₀` and `σ` (`:925`, `get_epsilon_anchored`). The history is
///     `data_prev_`, rolled at `:2126-2128`. That anchoring is the linear
///     frame's, and the frame — not the recycling — is what caps this whole
///     family at order 2; see `testMeasuredOrderIsTwoForEveryVariant`.
///
/// The AC-26 T1 gate is the published stage-2 recipe, and it is — by AC-24's
/// own arithmetic — entirely warm-up: `deis_3m` at 2 steps never reaches the
/// DEIS coefficients. That is exactly what the oracle recorded
/// (`rk_type == "ralston_3s"` on both steps), so the trace gate here proves the
/// RAMP, and the coefficients are gated by the oracle fixture plus an
/// independent double-precision re-run of upstream's own algorithm.
final class DEISMultistepSchedulerTests: XCTestCase {

  // MARK: - AC-23: the rhoab coefficients are RES4LYF's

  /// Every case in E18's oracle, every step index, including the `order == 1`
  /// empty entry and the `min(i+1, max_order)` ramp. Computed in `Double`.
  func testRhoabCoefficients() throws {
    let fixture = try SchedulerOracleFixtures.json("res4lyf_deis_coeffs.json")
    XCTAssertEqual(fixture["deis_mode"] as? String, "rhoab")
    let cases = try XCTUnwrap(fixture["cases"] as? [[String: Any]])
    XCTAssertEqual(cases.count, 12, "3 orders × 4 grids")

    for c in cases {
      let name = try XCTUnwrap(c["name"] as? String)
      let maxOrder = try XCTUnwrap(c["max_order"] as? Int)
      let sigmas = try SchedulerOracleFixtures.doubles(c["sigmas"], "\(name).sigmas")
      let want = try XCTUnwrap(c["coeffs"] as? [[Any]])

      let got = DEISCoefficients.rhoabList(sigmas: sigmas, maxOrder: maxOrder)
      XCTAssertEqual(got.count, sigmas.count - 1, "\(name): one entry per step")
      XCTAssertEqual(got.count, want.count, "\(name)")

      for (i, (g, w)) in zip(got, want).enumerated() {
        let wanted = try SchedulerOracleFixtures.doubles(w, "\(name).coeffs[\(i)]")
        // The ramp: order 1 contributes no coefficients at all.
        XCTAssertEqual(
          g.count, min(i + 1, maxOrder) == 1 ? 0 : min(i + 1, maxOrder),
          "\(name)[\(i)]: min(i+1, max_order) ramp")
        XCTAssertEqual(g.count, wanted.count, "\(name)[\(i)]")
        for (j, (a, b)) in zip(g, wanted).enumerated() {
          XCTAssertEqual(a, b, accuracy: 1e-6 * max(abs(b), 1.0), "\(name)[\(i)][\(j)]")
        }
        // Structural identity: the Lagrange weights integrate the constant 1
        // over [σᵢ, σᵢ₊₁], so they sum to `h`. This is what makes a constant
        // data prediction integrate EXACTLY (see `testExactOnAConstant…`).
        if !g.isEmpty {
          let h = sigmas[i + 1] - sigmas[i]
          XCTAssertEqual(g.reduce(0, +), h, accuracy: 1e-9 * max(abs(h), 1e-6), "\(name)[\(i)]: Σcoeff = h")
        }
      }
    }
  }

  /// The single-index accessor the scheduler actually calls agrees with the
  /// whole-list form, so the coefficient the solver uses is the fixture's.
  func testSingleIndexMatchesTheList() throws {
    let sigmas: [Double] = [1.0, 0.82, 0.61, 0.44, 0.29, 0.17, 0.08, 0.021, 0.0]
    for maxOrder in [2, 3, 4] {
      let list = DEISCoefficients.rhoabList(sigmas: sigmas, maxOrder: maxOrder)
      for i in 0..<(sigmas.count - 1) {
        XCTAssertEqual(
          DEISCoefficients.rhoab(sigmas: sigmas, index: i, maxOrder: maxOrder), list[i],
          "order \(maxOrder) index \(i)")
      }
    }
  }

  // MARK: - AC-24: the order ramp and the ralston warm-up

  /// `multistep_extra_initial_steps = 1`, so the warm-up is `order + 1` steps
  /// long and runs `ralston_{order}s` — derived from
  /// `rk_coefficients_beta.py:1343,1376-1390`, one variant at a time.
  func testWarmUpLengthAndSamplerPerVariant() {
    XCTAssertEqual(DEISMultistepScheduler.multistepExtraInitialSteps, 1)
    let expected: [(DEISMultistepScheduler.Order, Int, String)] = [
      (.two, 3, "ralston_2s"), (.three, 4, "ralston_3s"), (.four, 5, "ralston_4s"),
    ]
    for (order, steps, sampler) in expected {
      XCTAssertEqual(order.warmUpStepCount, steps, order.name)
      XCTAssertEqual(order.warmUpSampler, sampler, order.name)
      XCTAssertEqual(order.warmUpStages.rawValue, order.rawValue, order.name)
    }
  }

  /// AC-24's first half: the published stage-2 settings. `deis_3m` at 2 steps
  /// is ENTIRELY `ralston_3s` — 6 model evaluations, DEIS coefficients never
  /// engaging — and the record says so instead of leaving it to be
  /// rediscovered.
  func testOrderRampAtThePublishedStage2Settings() throws {
    let (base, startIndex, _, steps) = try Self.stage2Scheduler(order: .three)
    XCTAssertEqual(steps, 2)
    var scheduler = base
    let (_, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: MLXArray([Float(0.4)], [1]), startIndex: startIndex
    ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

    let deis = try XCTUnwrap(scheduler as? DEISMultistepScheduler)
    XCTAssertEqual(deis.order, .three)
    XCTAssertEqual(deis.order.rawValue, 3, "maxOrder 3")
    XCTAssertEqual(deis.warmUpSampler, "ralston_3s")
    XCTAssertEqual(deis.warmUpSteps, 2, "both steps warmed up")
    XCTAssertEqual(stats.evaluateCalls, 6, "2 steps × 3 ralston rows")
    XCTAssertEqual(stats.modelEvals, 6)
    XCTAssertEqual(stats.rowsAtStart, 3)
  }

  /// AC-24's second half: at 8 steps the ramp actually fires. Steps 0…3 are
  /// `ralston_3s` (3 rows each) and steps 4…7 are order-3 multistep (1 row
  /// each) — 16 evaluations, not `8 × 3`. `Stats.modelEvals` is the counted
  /// truth and `rowsAtStart` is only a label (WP-E3).
  func testOrderRampAtEightSteps() throws {
    let config = Krea2Sampling.schedulerConfig()
    var scheduler = try SchedulerFactory.create(
      kind: .deis3m, sigmaSchedule: .bongTangent, numInferenceSteps: 8, config: config,
      mu: Self.tracedMu, res4lyfSigmaPreparation: true)
    XCTAssertEqual(scheduler.numInferenceSteps, 8)

    let grid = scheduler.sigmas.asArray(Float.self).map(Double.init)
    let (_, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: MLXArray([Float(0.6)], [1])
    ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

    let deis = try XCTUnwrap(scheduler as? DEISMultistepScheduler)
    XCTAssertEqual(deis.warmUpSteps, 4, "step < order + multistep_extra_initial_steps")
    XCTAssertEqual(deis.warmUpSampler, "ralston_3s")
    XCTAssertEqual(deis.multistepSteps, 4, "steps 4…7 ran the DEIS coefficients")
    XCTAssertEqual(deis.rows, 1, "the row count fell to 1 once multistep engaged")
    XCTAssertEqual(stats.stepsRun, 8)
    XCTAssertEqual(stats.rowsAtStart, 3, "a label — the FIRST step's rows")
    XCTAssertEqual(stats.evaluateCalls, 4 * 3 + 4 * 1, "counted, not 8 × rowsAtStart")
    XCTAssertEqual(stats.modelEvals, 16)
    XCTAssertNotEqual(
      stats.modelEvals, stats.stepsRun * stats.rowsAtStart,
      "this is the sampler the product identity does NOT hold for")

    // "…and steps 4–7 use order-3 coefficients": the ramp
    // `order = min(i + 1, max_order)` is already saturated everywhere the
    // multistep half runs, so each of those steps interpolates through THREE
    // nodes — `Dᵢ`, `Dᵢ₋₁`, `Dᵢ₋₂`.
    for i in 4..<8 {
      XCTAssertEqual(
        DEISCoefficients.rhoab(sigmas: grid, index: i, maxOrder: 3).count, 3,
        "step \(i): order-3 coefficients")
    }
  }

  /// The same ramp for the other two variants, measured through the driver.
  func testOrderRampForEveryVariant() throws {
    let config = Krea2Sampling.schedulerConfig()
    for (kind, order) in [
      (SchedulerKind.deis2m, DEISMultistepScheduler.Order.two),
      (.deis3m, .three), (.deis4m, .four),
    ] {
      let steps = 12
      var scheduler = try SchedulerFactory.create(
        kind: kind, sigmaSchedule: .bongTangent, numInferenceSteps: steps, config: config,
        mu: Self.tracedMu, res4lyfSigmaPreparation: true)
      let effective = scheduler.numInferenceSteps
      let (_, stats) = Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: MLXArray([Float(0.5)], [1])
      ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

      let deis = try XCTUnwrap(scheduler as? DEISMultistepScheduler, kind.rawValue)
      let warm = order.warmUpStepCount
      XCTAssertEqual(deis.warmUpSteps, warm, kind.rawValue)
      XCTAssertEqual(deis.multistepSteps, effective - warm, kind.rawValue)
      XCTAssertEqual(deis.warmUpSampler, order.warmUpSampler, kind.rawValue)
      XCTAssertEqual(
        stats.evaluateCalls, warm * order.rawValue + (effective - warm),
        "\(kind.rawValue): warm-up rows + one call per multistep step")
    }
  }

  /// The ramp is a property of the RUN, not of the absolute grid index: a
  /// stage-2 render starts at `startIndex 8` of a 10-step grid and still warms
  /// up from its own step 0, exactly as RES4LYF does on the sliced schedule its
  /// node hands the sampler.
  func testWarmUpCountsFromTheRunStartNotTheGridIndex() throws {
    let (base, startIndex, totalSteps, _) = try Self.stage2Scheduler(order: .three)
    XCTAssertEqual(startIndex, 8)
    XCTAssertEqual(totalSteps, 10)
    var scheduler = base
    let (_, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: MLXArray([Float(0.4)], [1]), startIndex: startIndex
    ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }
    XCTAssertEqual(stats.evaluateCalls, 6, "3 rows on both steps — not multistep at grid index 8")
  }

  // MARK: - AC-26 T1: the production path against RES4LYF's own trace

  static let tracedMu: Float = 1.15

  static func fluxSigmaMin() -> Float {
    SigmaSchedule.fluxSigmaTable(shift: tracedMu, tableSize: Krea2Sampling.fluxTableSize)[0]
  }

  /// The stage-2 scheduler exactly as `Krea2Pipeline.makeScheduler` builds it:
  /// `SchedulerFactory`, Krea 2's config, the traced `mu`, RES4LYF sigma
  /// preparation on. `denoise 0.2` is a partial start on the full 10-step
  /// `bong_tangent` grid, never a re-generated one, and nothing is read from
  /// the trace to construct it (Addendum A.1).
  static func stage2Scheduler(order: DEISMultistepScheduler.Order) throws -> (
    scheduler: any ZImageScheduler, startIndex: Int, totalSteps: Int, steps: Int
  ) {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let stage2 = try XCTUnwrap(
      fixture["stage2_bong_tangent_denoise"] as? [String: Any],
      "comfy_sigmas.json: stage2_bong_tangent_denoise")
    let steps = try XCTUnwrap(stage2["steps"] as? Int)
    let totalSteps = try XCTUnwrap(stage2["total_steps"] as? Int)
    let kind: SchedulerKind = order == .two ? .deis2m : (order == .three ? .deis3m : .deis4m)
    let scheduler = try SchedulerFactory.create(
      kind: kind,
      sigmaSchedule: .bongTangent,
      numInferenceSteps: totalSteps,
      config: Krea2Sampling.schedulerConfig(),
      mu: tracedMu,
      res4lyfSigmaPreparation: true)
    return (scheduler, totalSteps - steps, totalSteps, steps)
  }

  /// The grid the solver walks is OURS — `bong_tangent` from our producer plus
  /// `RES4LYFSigmaPreparation`'s `ModelSamplingFlux` σ_min — and it lands on
  /// the oracle's `sigmas_run` sigma for sigma.
  func testStage2GridComesFromOurOwnProducers() throws {
    let (scheduler, startIndex, totalSteps, _) = try Self.stage2Scheduler(order: .three)
    let m = try RES4LYFTraceFixture.load("deis3m_bong2_T1").manifest
    XCTAssertEqual(m.recipe.sampler, "deis_3m")

    let grid = scheduler.sigmas.asArray(Float.self)
    XCTAssertEqual(grid.count, totalSteps + 1)
    XCTAssertEqual(scheduler.numInferenceSteps, totalSteps)
    XCTAssertEqual(Double(Self.fluxSigmaMin()), m.sigmaMin, accuracy: 1e-9)
    XCTAssertEqual(scheduler.finalConversionSigma, Self.fluxSigmaMin())

    let window = Array(grid[startIndex...])
    XCTAssertEqual(window.count, m.sigmasRun.count - 1)
    for (i, (g, w)) in zip(window, m.sigmasRun).enumerated() {
      XCTAssertEqual(Double(g), w, accuracy: 1e-6, "sigmas_run[\(i)]")
    }
  }

  /// **The E14 gate.** A FACTORY-created `deis_3m`, driven by the PRODUCTION
  /// `Krea2DenoiseLoop` against the scripted denoiser, reproduces the T1
  /// trace's FINAL tensor — after the model-free `σ_min → 0` conversion, which
  /// the loop applies and this test does not. Nothing is read from the fixture
  /// but the initial latent and the answer; no grid surgery, no hand-applied
  /// tail.
  ///
  /// It is the ramp that is under test here: the oracle ran `ralston_3s` on
  /// both steps (`rk_type`), so a `deis_3m` that engaged its multistep
  /// coefficients early — or warmed up for 3 steps instead of 4 — misses this
  /// tensor by orders of magnitude.
  ///
  /// `floor: 1.0` makes this an ABSOLUTE 1e-4, for the reason S-FIX-1 recorded
  /// on the `res_2s` gate: the final latent is the data prediction at σ_min and
  /// its magnitude has collapsed to ~5e-3, so a tolerance relative to it would
  /// measure float32 ulps of the trajectory rather than agreement.
  ///
  /// Measured margins (2026-08-22), so the threshold is not taken on faith:
  /// this passes at **1.34e-7**, and with `.deis3m` removed from
  /// `isRES4LYFFamily` — i.e. the σ_min preparation and the model-free tail
  /// gone — the same difference is **7.4e-4**, 7.4× over the gate.
  func testDEIS3mThroughTheDenoiseLoopReproducesTheT1TraceFinalTensor() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    let (base, startIndex, _, _) = try Self.stage2Scheduler(order: .three)
    var scheduler = base

    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit), startIndex: startIndex
    ) { latent, sigma in
      RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    XCTAssertEqual(stats.stepsRun, 2)
    XCTAssertEqual(stats.evaluateCalls, m.modelCallsTotal, "the oracle made 6 calls; so did we")
    XCTAssertEqual(stats.modelEvals, 6, "the σ_min conversion is model-free")
    XCTAssertEqual(Double(try XCTUnwrap(stats.finalConversionSigma)), m.sigmaMin, accuracy: 1e-9)

    XCTAssertTraceClose(
      x, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }

  /// …and step by step against the oracle's recorded row samples, so a
  /// divergence is located rather than merely detected.
  func testDEIS3mMatchesTheT1TraceRowByRow() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    let (base, startIndex, _, _) = try Self.stage2Scheduler(order: .three)
    var scheduler = try XCTUnwrap(base as? DEISMultistepScheduler)
    scheduler.reset()
    XCTAssertEqual(scheduler.modelOutputConvention, .dataPrediction)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      let i = startIndex + step.index
      XCTAssertEqual(step.rkType, "ralston_3s", "the oracle warmed up on this step")
      XCTAssertEqual(scheduler.rows, 3, "step \(step.index): still warming up")
      XCTAssertTraceClose(x, try trace.tensor(step.x0), rtol: 1e-4, "step \(step.index): x entering")

      var k: [MLXArray] = []
      for r in 0..<scheduler.rows {
        let call = step.modelCalls[r]
        let xr = r == 0 ? x : scheduler.rowSample(timestepIndex: i, row: r, x0: x, k: k)
        if r > 0 {
          XCTAssertEqual(
            Double(scheduler.rowSigma(timestepIndex: i, row: r)), call.sTmp, accuracy: 1e-6,
            "step \(step.index) row \(r): substep sigma")
        }
        XCTAssertTraceClose(
          xr, try trace.tensor(call.xIn), rtol: 1e-4, "step \(step.index) row \(r): row sample")
        k.append(try trace.tensor(call.denoised))
      }
      x = scheduler.commit(timestepIndex: i, x0: x, k: k)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")
    }
    XCTAssertEqual(scheduler.warmUpSteps, 2)
  }

  // MARK: - The multistep assembly, cross-checked independently

  /// An independent re-run of upstream's own algorithm — RES4LYF's `rhoab`
  /// closed form plus `rk_sampler_beta.py`'s multistep assembly and its
  /// `data_prev_` roll — transcribed into double-precision Python from the
  /// vendored source (`scripts/oracles/upstream/res4lyf/beta/`) and evaluated
  /// on the scripted denoiser at `x₀ = 0.7` over `σ: 1.0 → 0.05`.
  ///
  /// These are the terminal values that reference produced (2026-08-22). They
  /// are NOT re-derived by this file: they exist so that the Swift multistep —
  /// the half the T1 trace cannot reach, because the published stage never
  /// leaves the warm-up — has an oracle of its own.
  func testMultistepMatchesAnIndependentReferenceRun() {
    let expected: [(DEISMultistepScheduler.Order, Int, Double)] = [
      (.two, 8, 0.198466466219), (.two, 12, 0.188801524265),
      (.three, 8, 0.194737552942), (.three, 12, 0.186273317320),
      (.four, 8, 0.193411474227), (.four, 12, 0.185469506187),
    ]
    for (order, n, want) in expected {
      let grid = ExplicitRKSchedulerTests.linearGrid(steps: n)
      let scheduler = DEISMultistepScheduler(
        order: order, numInferenceSteps: n, sigmaValues: grid)
      let got = ExplicitRKSchedulerTests.runThroughDriver(scheduler)
      XCTAssertEqual(got, want, accuracy: 2e-5, "\(order.name) at \(n) steps")
    }
  }

  /// The one trajectory the anchoring integrates EXACTLY, for the same reason
  /// the ralstons do: with a constant data prediction every epsilon is
  /// `(x₀ − D)/σ` and `Σⱼ coeffⱼ = h`, so the step is the closed-form
  /// `x(σ) = D + (σ/σ₀)(x₀ − D)`. It exercises the multistep branch (the
  /// history is used) without depending on the field's curvature.
  func testExactOnAConstantDataPrediction() {
    let dConst: Float = -0.35
    let grid: [Float] = [1.0, 0.82, 0.61, 0.44, 0.29, 0.17, 0.08, 0.03, 0.008]
    for order in DEISMultistepScheduler.Order.allCases {
      var scheduler: any ZImageScheduler = DEISMultistepScheduler(
        order: order, numInferenceSteps: grid.count - 1, sigmaValues: grid)
      let (out, stats) = Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: MLXArray([Float(0.7)], [1])
      ) { latent, sigma in (latent - dConst) / sigma }
      let want = dConst + Float(grid[grid.count - 1] / grid[0]) * (0.7 - dConst)
      XCTAssertEqual(out.item(Float.self), want, accuracy: 3e-6, order.name)
      let warm = order.warmUpStepCount
      XCTAssertEqual(
        stats.evaluateCalls, warm * order.rawValue + (grid.count - 1 - warm), order.name)
    }
  }

  // MARK: - AC-13: run state does not leak between runs

  /// `deis_Nm` is the FIRST conformer whose run state is not idempotent: a step
  /// counter that drives the order ramp, an origin index that pins the run's
  /// contiguity, and a latent history that the multistep half integrates
  /// through. Every one of them has to be cleared, or a second render on the
  /// same instance would start already-ramped — skipping its warm-up and
  /// interpolating through the PREVIOUS image's data predictions.
  ///
  /// Asserted as bit-identity between two runs of one instance, which is the
  /// only formulation that catches all three at once (a stale counter changes
  /// the evaluation count; a stale history changes the pixels; a stale origin
  /// index trips the contiguity precondition).
  func testResetBetweenRunsOnOneInstance() throws {
    let config = Krea2Sampling.schedulerConfig()
    for kind in [SchedulerKind.deis2m, .deis3m, .deis4m] {
      var scheduler = try SchedulerFactory.create(
        kind: kind, sigmaSchedule: .bongTangent, numInferenceSteps: 10, config: config,
        mu: Self.tracedMu, res4lyfSigmaPreparation: true)
      let initial = MLXArray((0..<64).map { Float($0 % 7) * 0.11 - 0.3 }, [1, 1, 8, 8])

      let (first, firstStats) = Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: initial
      ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

      // Dirty it deliberately between runs, the way a partial/abandoned render
      // would: a second run must not care what the first one left behind.
      let mid = try XCTUnwrap(scheduler as? DEISMultistepScheduler)
      XCTAssertGreaterThan(mid.multistepSteps, 0, "\(kind.rawValue): the ramp must have fired")

      let (second, secondStats) = Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: initial
      ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

      XCTAssertEqual(
        first.asArray(Float.self), second.asArray(Float.self),
        "\(kind.rawValue): the second run saw state from the first")
      XCTAssertEqual(firstStats.evaluateCalls, secondStats.evaluateCalls, kind.rawValue)
      XCTAssertEqual(firstStats.modelEvals, secondStats.modelEvals, kind.rawValue)
      XCTAssertEqual(firstStats.rowsAtStart, secondStats.rowsAtStart, kind.rawValue)

      let after = try XCTUnwrap(scheduler as? DEISMultistepScheduler)
      XCTAssertEqual(after.warmUpSteps, mid.warmUpSteps, "\(kind.rawValue): warm-up count reran")
      XCTAssertEqual(after.multistepSteps, mid.multistepSteps, kind.rawValue)
    }
  }

  /// A run that starts partway down the grid resets to ITS own origin, so a
  /// stage-2 render after a full one still warms up (the origin index is run
  /// state too, not a constant).
  func testResetRestoresTheRunOriginForAPartialSecondRun() throws {
    let (base, startIndex, _, _) = try Self.stage2Scheduler(order: .three)
    var scheduler = base
    let initial = MLXArray([Float(0.4)], [1])
    let (fresh, freshStats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: initial, startIndex: startIndex
    ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

    let (again, againStats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: initial, startIndex: startIndex
    ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

    XCTAssertEqual(fresh.asArray(Float.self), again.asArray(Float.self))
    XCTAssertEqual(freshStats.evaluateCalls, 6)
    XCTAssertEqual(againStats.evaluateCalls, 6, "the second run warmed up from its own step 0")
  }

  // MARK: - AC-25: the order of accuracy, measured

  /// AC-25 for `deis_Nm`, **measured and pinned two-sided** — the controller's
  /// ruling, and the same discipline `ExplicitRKSchedulerTests` applies to the
  /// ralston family.
  ///
  /// Every variant lands at **2** — measured 1.97 / 2.05 / 2.01 for 2m / 3m /
  /// 4m at the finest pair.
  ///
  /// The cause is the FRAME, not the history. RES4LYF's anchored linear frame
  /// freezes the `1/σ` kernel at the step's own sigma, so whatever the
  /// interpolation produces the step collapses to `x' = x₀ + (h/σ)(x₀ − D_eff)`.
  /// That integrates a constant `D` exactly, but the exact solution weights
  /// `D(s)` by `σ'/s²` across the step where the scheme weights it by `1/σ`;
  /// the difference integrates to `−D′h³/(6σ²)`, a local `O(h³)` and therefore
  /// global order 2 — independent of the interpolation degree and of whether
  /// any history was recycled. That is why `ralston_3s`/`ralston_4s` cap at 2
  /// with no history at all (`ExplicitRKSchedulerTests`,
  /// `RES4LYFTableau.swift`), and why `deis_2m`, whose nominal order IS 2,
  /// loses nothing here.
  ///
  /// The lower bound proves the solver is consistent and correctly wired; the
  /// upper bound proves nobody "fixed" the frame into a textbook
  /// Adams–Bashforth and diverged from the oracle.
  ///
  /// Precision: the coefficients and the scripted field are `Double`; the state
  /// the driver carries is float32 MLX, as in production. At these step counts
  /// the truncation error (≥ 2e-5) is orders of magnitude above float32
  /// round-off, so the measured slopes are the scheme's, not the arithmetic's —
  /// which the monotonicity assertion below would catch if it stopped being
  /// true.
  func testMeasuredOrderIsTwoForEveryVariant() {
    let reference = ExplicitRKSchedulerTests.referenceSolution()
    for order in DEISMultistepScheduler.Order.allCases {
      var errors: [Double] = []
      for n in [16, 32, 64, 128] {
        let scheduler = DEISMultistepScheduler(
          order: order, numInferenceSteps: n,
          sigmaValues: ExplicitRKSchedulerTests.linearGrid(steps: n))
        errors.append(abs(ExplicitRKSchedulerTests.runThroughDriver(scheduler) - reference))
      }
      let orders = ExplicitRKSchedulerTests.observedOrders(errors)
      let context = "\(order.name) orders \(orders) from errors \(errors)"
      for (i, p) in orders.enumerated() {
        let floorAt = i == orders.count - 1 ? 1.85 : 1.5
        XCTAssertGreaterThan(p, floorAt, "pair \(i): \(context)")
        XCTAssertLessThan(
          p, 2.6,
          "pair \(i) reached order \(p): RES4LYF's anchored linear frame — the frozen `1/σ` "
            + "kernel that caps this whole family at 2 — is gone. That is a divergence from "
            + "the oracle, not an improvement. \(context)")
      }
      XCTAssertTrue(
        zip(errors, errors.dropFirst()).allSatisfy { $0 > $1 },
        "error must fall monotonically as steps double: \(context)")
    }
  }

  // MARK: - Wiring: accepted, advertised, krea2-only, RES4LYF-prepared

  func testWireNamesResolveAndAreAdvertised() throws {
    let config = Krea2Sampling.schedulerConfig()
    let expected: [(String, SchedulerKind, DEISMultistepScheduler.Order)] = [
      ("deis_2m", .deis2m, .two), ("deis_3m", .deis3m, .three), ("deis_4m", .deis4m, .four),
    ]
    for (name, kind, order) in expected {
      XCTAssertEqual(SchedulerKind(rawValue: name), kind)
      XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind(name), kind)
      // RES4LYF's UI groups these under `multistep/`; a workflow value pastes verbatim.
      XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("multistep/\(name)"), kind)
      XCTAssertTrue(
        RecipeNameResolver.advertisedSamplerNames.contains(name),
        "\(name) is accepted but not advertised — the E4 reconciliation AC")
      XCTAssertTrue(RecipeNameResolver.validSamplerNames.contains(name))

      XCTAssertTrue(kind.isNRowTableau, name)
      XCTAssertTrue(kind.isRES4LYFFamily, "\(name) obeys prepare_sigmas and the model-free tail")

      let scheduler = try SchedulerFactory.create(
        kind: kind, sigmaSchedule: .krea2, numInferenceSteps: 9, config: config, mu: 1.15,
        res4lyfSigmaPreparation: true)
      let deis = try XCTUnwrap(scheduler as? DEISMultistepScheduler, name)
      XCTAssertEqual(deis.order, order, name)
      XCTAssertEqual(deis.rows, order.rawValue, "\(name): the warm-up's rows at step 0")
      XCTAssertEqual(deis.modelOutputConvention, .dataPrediction, name)
      XCTAssertEqual(deis.finalConversionSigma, Self.fluxSigmaMin(), "\(name): the σ_min tail")
    }
  }

  /// The tableau gate is ONE predicate: `deis_*` is refused off Krea 2 for the
  /// same reason `ralston_*` is — only `Krea2DenoiseLoop` dispatches rows.
  func testDEISSamplersAreRefusedOffKrea2() throws {
    for kind in [SchedulerKind.deis2m, .deis3m, .deis4m] {
      for family in WarmModelFamily.allCases {
        let names = ResolvedRecipeNames(
          scheduler: kind, schedulerRequested: kind.rawValue,
          sigmaSchedule: nil, sigmaScheduleRequested: nil)
        let error = GeneratePayload.validateFamilyRecipe(names, family: family)
        if family == .krea2 {
          XCTAssertNil(error, "\(kind.rawValue) must run on krea2")
        } else {
          let message = try XCTUnwrap(try XCTUnwrap(error).errorDescription)
          XCTAssertTrue(message.contains(kind.rawValue), message)
        }
      }
    }
  }

  /// The legacy `deis` name is a DIFFERENT sampler and must not be disturbed:
  /// it stays a 1-row velocity scheduler on every family, so a Z-Image request
  /// asking for `deis` still renders (WP-E14 self-review, recorded deviation).
  func testLegacyDEISNameIsUntouched() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("deis"), .deis)
    XCTAssertFalse(SchedulerKind.deis.isNRowTableau)
    XCTAssertFalse(SchedulerKind.deis.isRES4LYFFamily)
    let scheduler = try SchedulerFactory.create(kind: .deis, numInferenceSteps: 9, config: config)
    XCTAssertTrue(scheduler is DEISScheduler)
    XCTAssertNil(
      GeneratePayload.validateFamilyRecipe(
        ResolvedRecipeNames(
          scheduler: .deis, schedulerRequested: "deis", sigmaSchedule: nil,
          sigmaScheduleRequested: nil),
        family: .flux1))
  }

  // MARK: - Provenance (AC-24's record half)

  /// `Krea2RunTrace` carries what actually ran, so nobody rediscovers that a
  /// 2-step `deis_3m` never reached its coefficients.
  func testTraceCarriesTheWarmUpProvenance() throws {
    let (base, startIndex, _, _) = try Self.stage2Scheduler(order: .three)
    var scheduler = base
    let (_, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: MLXArray([Float(0.4)], [1]), startIndex: startIndex
    ) { latent, sigma in RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma) }

    let shift = try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: 4096, align: 16)
    let trace = Krea2RunTrace(
      sampler: .deis3m, sigmaSchedule: .bongTangent, sigmaScheduleRequested: nil,
      mu: shift.mu, shift: shift.shift, shiftSource: shift.source.rawValue,
      sigmas: scheduler.sigmas.asArray(Float.self) + [0.0],
      finalConversionSigma: stats.finalConversionSigma,
      warmupSampler: Krea2RunTrace.warmUpSampler(of: scheduler),
      warmupSteps: Krea2RunTrace.warmUpSteps(of: scheduler),
      stepsRequested: 2, stepsEffective: scheduler.numInferenceSteps, stepsRun: stats.stepsRun,
      modelEvals: stats.modelEvals, startIndex: startIndex, denoise: 0.2,
      guidance: 1.0, eta: 0, bongmath: false, seed: 1, width: 1024, height: 1024)

    XCTAssertEqual(trace.warmupSampler, "ralston_3s")
    XCTAssertEqual(trace.warmupSteps, 2)
    XCTAssertEqual(trace.modelEvals, 6)
  }

  /// A sampler with no warm-up reports none, rather than an invented one.
  func testTraceReportsNoWarmUpForNonRampingSamplers() throws {
    let config = Krea2Sampling.schedulerConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .ralston3s, sigmaSchedule: .krea2, numInferenceSteps: 6, config: config, mu: 1.15)
    XCTAssertNil(Krea2RunTrace.warmUpSampler(of: scheduler))
    XCTAssertEqual(Krea2RunTrace.warmUpSteps(of: scheduler), 0)
  }
}
