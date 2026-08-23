import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E16 — parity tier **T3**: RES4LYF's `bongmath` fixed point
/// (FDD-krea2-raw-recipe §3.13, §4 AC-26, D18).
///
/// The load-bearing gates are the two T3 step traces E18 exported from
/// `sample_rk_beta` — `res2s_beta6_T3` and `deis3m_bong2_T3` — at
/// `eta = eta_substep = 0.5` **and** `bongmath = true`. T3 composes with T2:
/// the same recorded noise, the same SDE split, plus the fixed point. They are
/// driven through the **factory-built** scheduler and the **production**
/// `Krea2DenoiseLoop`, on OUR grid; nothing is read from the fixture but the
/// initial latent, the recorded noise tensors, and the answer.
///
/// What `bongmath` actually is (`rk_method_beta.py:695-767`, called from
/// `rk_sampler_beta.py:1967-1974`): after a non-final row's sample has been
/// built AND re-noised, upstream re-derives the STEP's `x₀` so that the
/// re-noised row sample is again exactly what the tableau would have produced
/// from it — a 100-iteration fixed point over the row's own epsilon history.
/// It makes **no model call** (`model_calls_total` is 12 / 6 at every tier),
/// and the re-derived `x₀` is what the rest of the step then uses: the
/// remaining rows, the commit, the T2 step swap and the model-free tail.
final class RES4LYFBongMathParityTests: XCTestCase {

  // MARK: - Fixtures and wiring

  /// The T2 machinery, reused verbatim: T3 is T2 plus the fixed point, so the
  /// injector, its recorded streams and their exhaustion checks are the same.
  private func recordedInjector(_ trace: RES4LYFTraceFixture) throws -> (
    injector: RES4LYFSDENoiseInjector,
    step: RES4LYFEtaSDEParityTests.RecordedNoiseStream,
    substep: RES4LYFEtaSDEParityTests.RecordedNoiseStream
  ) {
    let m = trace.manifest
    let stepNoise = try m.steps.compactMap { $0.noiseStep }.map { try trace.tensor($0) }
    let substepNoise = try m.steps.flatMap { $0.substeps }.compactMap { $0.noise }
      .map { try trace.tensor($0) }
    let step = RES4LYFEtaSDEParityTests.RecordedNoiseStream(
      name: "\(trace.name) step noise", tensors: stepNoise)
    let substep = RES4LYFEtaSDEParityTests.RecordedNoiseStream(
      name: "\(trace.name) substep noise", tensors: substepNoise)
    let injector = RES4LYFSDENoiseInjector(
      eta: m.recipe.eta, etaSubstep: m.recipe.etaSubstep,
      sNoise: m.recipe.sNoise, sNoiseSubstep: m.recipe.sNoise,
      sigmaMax: m.sigmaMax, stepNoise: step, substepNoise: substep)
    return (injector, step, substep)
  }

  /// The production `bongmath`, built for the trace's own model sampling.
  private func bongMath(_ m: RES4LYFTrace) -> RES4LYFBongMath {
    RES4LYFBongMath(sigmaMin: m.sigmaMin, sigmaMax: m.sigmaMax)
  }

  /// The traced shift — `ModelSamplingFlux(mu = 1.15)`, which is what the
  /// pipeline factory derives `σ_min` / `σ_max` from.
  private static func tracedShift() throws -> Krea2Sampling.ScheduleShift {
    try Krea2Sampling.resolveShift(explicit: 1.15, seqLen: 4096, align: 16)
  }

  // MARK: - AC-26 T3, res_2s: the PRODUCTION loop against the oracle

  /// The T3 gate for stage 1. Factory-built `res_2s + beta` at 6 steps under
  /// `ModelSamplingFlux(1.15)`, driven by `Krea2DenoiseLoop` with BOTH hooks
  /// attached, reproduces the trace's final tensor — after the model-free
  /// `σ_min → 0` conversion, whose epsilon is taken from the bongmath-rebased
  /// `x₀` and not from the sample the step started on.
  ///
  /// `floor: 1.0` makes this an ABSOLUTE 1e-4, for the reason S-FIX-1 recorded
  /// on the T1 gate: the final latent's magnitude has collapsed to ~1e-2.
  func testRES2sT3ThroughTheDenoiseLoopReproducesTheTraceFinalTensor() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T3")
    let m = trace.manifest
    XCTAssertTrue(m.recipe.bongmath, "res2s_beta6_T3 must be the bongmath run")
    var scheduler = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let (injector, stepNoise, substepNoise) = try recordedInjector(trace)
    let bong = bongMath(m)

    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit),
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: injector, bongmath: bong)

    // The fixed point is ALGEBRAIC: it costs no model evaluation and takes no
    // extra step, and `modelEvals` says so because it is counted.
    XCTAssertEqual(stats.stepsRun, 6)
    XCTAssertEqual(stats.rowsAtStart, 2)
    XCTAssertEqual(stats.evaluateCalls, 12, "2 rows × 6 steps — unchanged by bongmath")
    XCTAssertEqual(stats.modelEvals, m.modelCallsTotal, "the oracle made the same 12 calls")
    XCTAssertEqual(bong.extraModelEvals, 0, "RES4LYF's bong_iter never calls the model")
    XCTAssertEqual(Double(try XCTUnwrap(stats.finalConversionSigma)), m.sigmaMin, accuracy: 1e-9)

    // T2's draws are unchanged by T3 — same streams, same order, all consumed.
    XCTAssertEqual(stepNoise.drawn, 6)
    XCTAssertEqual(substepNoise.drawn, 6)
    XCTAssertTrue(stepNoise.exhausted && substepNoise.exhausted)

    XCTAssertTraceClose(
      x, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }

  /// …step by step, so a divergence is located rather than merely detected.
  /// The tensor that matters is `step/x_0`: E18 exported it AFTER the step's
  /// bongmath, so it is the oracle's own statement that the anchor moved.
  func testRES2sT3MatchesTheTraceStepByStep() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T3")
    let m = trace.manifest
    var scheduler = try XCTUnwrap(
      try RES4LYFTraceParityTests.productionRES2sScheduler() as? RES2sScheduler)
    scheduler.reset()
    let (injector, _, _) = try recordedInjector(trace)
    let bong = bongMath(m)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      let entering = x
      var x0 = x

      let k1 = RES4LYFScriptedDenoiser.denoised(x, sigma: Float(step.sigma))
      var mid = try XCTUnwrap(
        scheduler.intermediateStep(modelOutput: k1, timestepIndex: step.index, sample: x0))
      let sub = try XCTUnwrap(step.substeps.first)
      let erased: any ZImageScheduler = scheduler
      injector.injectSubstep(
        sample: &mid, x0: x0, timestepIndex: step.index, row: 1, scheduler: erased)
      XCTAssertTraceClose(
        mid, try trace.tensor(try XCTUnwrap(sub.xPost)), rtol: 1e-4,
        "step \(step.index): x after the substep re-noise")

      // The fixed point, at the position upstream runs it: after the substep
      // re-noise, before the second row's model call.
      let snapshot = scheduler
      let extra = bong.iterate(
        x0: &x0, rowSample: mid, timestepIndex: step.index, row: 0, scheduler: erased,
        buildRowSample: { anchor in
          var s = snapshot
          return s.intermediateStep(
            modelOutput: k1, timestepIndex: step.index, sample: anchor)!
        },
        evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) })
      XCTAssertEqual(extra, 0, "step \(step.index): the fixed point is algebraic")

      if let bongRecord = step.bongmath.first {
        XCTAssertEqual(bongRecord.row, 0)
        XCTAssertTraceClose(
          x0, try trace.tensor(bongRecord.x0), rtol: 1e-4,
          "step \(step.index): the re-derived x₀")
        XCTAssertTraceClose(
          x0, try trace.tensor(bongRecord.xRows[0]), rtol: 1e-4,
          "step \(step.index): x_[0] IS the re-derived x₀ (zum(0) = 0)")
      } else {
        // Upstream's guards refused this step — `h ≥ σ_max/2` at the tail —
        // so the anchor must be exactly the sample the step started on.
        XCTAssertEqual(
          x0.asType(.float32).asArray(Float.self), entering.asType(.float32).asArray(Float.self),
          "step \(step.index): a refused fixed point must not move x₀ at all")
      }
      XCTAssertTraceClose(
        x0, try trace.tensor(step.x0), rtol: 1e-4,
        "step \(step.index): x₀ as the step's update saw it")

      let midSigma = try XCTUnwrap(scheduler.intermediateSigma(timestepIndex: step.index))
      let k2 = RES4LYFScriptedDenoiser.denoised(mid, sigma: midSigma)
      x = scheduler.finalizeStep(
        originalOutput: k1, intermediateOutput: k2, timestepIndex: step.index, sample: x0)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")

      let erasedAfter: any ZImageScheduler = scheduler
      injector.inject(sample: &x, x0: x0, timestepIndex: step.index, scheduler: erasedAfter)
      XCTAssertTraceClose(x, try trace.tensor(step.xOut), rtol: 1e-4, "step \(step.index): x_out")
    }
  }

  // MARK: - AC-26 T3, deis_3m: three rows, TWO fixed points per step

  /// The T3 gate for stage 2 — `deis_3m`, 2 steps at denoise 0.2, every step
  /// the `ralston_3s` warm-up (AC-24). Three rows means bongmath runs TWICE
  /// per step (after rows 0 and 1, never after the committing row), in the
  /// LINEAR frame where `ε = (x₀ − denoised)/σ`.
  func testDEIS3mT3ThroughTheDenoiseLoopReproducesTheTraceFinalTensor() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T3")
    let m = trace.manifest
    let (base, startIndex, _, _) = try DEISMultistepSchedulerTests.stage2Scheduler(order: .three)
    var scheduler = base
    let (injector, stepNoise, substepNoise) = try recordedInjector(trace)
    let bong = bongMath(m)

    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit), startIndex: startIndex,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: injector, bongmath: bong)

    XCTAssertEqual(stats.stepsRun, 2)
    XCTAssertEqual(stats.evaluateCalls, m.modelCallsTotal, "6 calls, unchanged by bongmath")
    XCTAssertEqual(stats.modelEvals, 6)
    XCTAssertEqual(bong.extraModelEvals, 0)
    XCTAssertEqual(bong.rebases, 4, "rows 0 and 1 of 3, on both steps")
    XCTAssertEqual(bong.refusals, 0, "σ > 0.03 and h < σ_max/2 hold on both steps")
    XCTAssertEqual(stepNoise.drawn, 2)
    XCTAssertEqual(substepNoise.drawn, 4)
    XCTAssertTrue(stepNoise.exhausted && substepNoise.exhausted)

    XCTAssertTraceClose(
      x, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }

  /// …row by row, against every `bong_iter` the oracle recorded.
  func testDEIS3mT3MatchesTheTraceRowByRow() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T3")
    let m = trace.manifest
    let (base, startIndex, _, _) = try DEISMultistepSchedulerTests.stage2Scheduler(order: .three)
    var scheduler = try XCTUnwrap(base as? DEISMultistepScheduler)
    scheduler.reset()
    let (injector, _, _) = try recordedInjector(trace)
    let bong = bongMath(m)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      let i = startIndex + step.index
      XCTAssertEqual(scheduler.rows, 3, "step \(step.index): still warming up")
      var x0 = x

      var k: [MLXArray] = []
      for r in 0..<scheduler.rows {
        let call = step.modelCalls[r]
        var xr = r == 0 ? x0 : scheduler.rowSample(timestepIndex: i, row: r, x0: x0, k: k)
        if r > 0 {
          let sub = step.substeps[r - 1]
          let erased: any ZImageScheduler = scheduler
          injector.injectSubstep(sample: &xr, x0: x0, timestepIndex: i, row: r, scheduler: erased)
          XCTAssertTraceClose(
            xr, try trace.tensor(try XCTUnwrap(sub.xPost)), rtol: 1e-4,
            "step \(step.index) row \(r): after the substep re-noise")

          let snapshot = scheduler
          let rowIndex = r
          let history = k
          _ = bong.iterate(
            x0: &x0, rowSample: xr, timestepIndex: i, row: r - 1, scheduler: erased,
            buildRowSample: { anchor in
              var s = snapshot
              return s.rowSample(timestepIndex: i, row: rowIndex, x0: anchor, k: history)
            },
            evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) })

          let record = step.bongmath[r - 1]
          XCTAssertEqual(record.row, r - 1)
          XCTAssertTraceClose(
            x0, try trace.tensor(record.x0), rtol: 1e-4,
            "step \(step.index): x₀ re-derived after row \(r - 1)")
        }
        // The row the model is evaluated on is NOT rewritten by the fixed
        // point — only the anchor behind it is.
        XCTAssertTraceClose(
          xr, try trace.tensor(call.xIn), rtol: 1e-4, "step \(step.index) row \(r): row sample")
        let sigmaR = Float(call.sTmp)
        k.append(
          scheduler.modelInput(
            velocity: RES4LYFScriptedDenoiser.velocity(xr, sigma: sigmaR), sample: xr,
            sigma: sigmaR))
      }

      XCTAssertTraceClose(
        x0, try trace.tensor(step.x0), rtol: 1e-4,
        "step \(step.index): the commit's anchor is the LAST re-derived x₀")
      x = scheduler.commit(timestepIndex: i, x0: x0, k: k)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")
      let erased: any ZImageScheduler = scheduler
      injector.inject(sample: &x, x0: x0, timestepIndex: i, scheduler: erased)
      XCTAssertTraceClose(x, try trace.tensor(step.xOut), rtol: 1e-4, "step \(step.index): x_out")
    }
  }

  // MARK: - The guards are upstream's, step for step

  /// `bong_iter` is refused where upstream refuses it, and nowhere else.
  ///
  /// `rk_sampler_beta.py:1967,1971`: `σ > 0.03`, `s_[row] > σ_min`, and
  /// `h < σ_max/2`. On `res_2s + beta` at 6 steps the last condition fails at
  /// steps 4 and 5 (`h` = 0.815 and 6.64 against a bound of 0.5) — and E18's
  /// trace records an EMPTY `bongmath` array on exactly those two steps. The
  /// count our hook reports must be that array's shape, not a rounder number.
  func testTheGuardsRefuseExactlyTheStepsUpstreamRefused() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T3")
    let m = trace.manifest
    let stepsUpstreamRan = m.steps.filter { !$0.bongmath.isEmpty }.map { $0.index }
    XCTAssertEqual(stepsUpstreamRan, [0, 1, 2, 3], "the fixture's own record of the guards")

    var scheduler = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let (injector, _, _) = try recordedInjector(trace)
    let bong = bongMath(m)
    _ = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit),
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: injector, bongmath: bong)

    XCTAssertEqual(bong.rebasedSteps, stepsUpstreamRan, "the steps the fixed point ran on")
    XCTAssertEqual(bong.rebases, 4, "one rebasing row per step on a 2-row tableau")
    XCTAssertEqual(bong.refusals, 2, "steps 4 and 5, refused by h ≥ σ_max/2")
  }

  /// The fixed point actually reaches its fixed point: after it returns, the
  /// row sample the tableau builds from the re-derived `x₀` IS the re-noised
  /// row sample the fixed point was handed. That is the whole invariant, and
  /// it is checked on the production grid rather than assumed from 100
  /// iterations.
  func testTheFixedPointIsAFixedPoint() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T3")
    let m = trace.manifest
    var scheduler = try XCTUnwrap(
      try RES4LYFTraceParityTests.productionRES2sScheduler() as? RES2sScheduler)
    let (injector, _, _) = try recordedInjector(trace)
    let bong = bongMath(m)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      var x0 = x
      let k1 = RES4LYFScriptedDenoiser.denoised(x, sigma: Float(step.sigma))
      var mid = try XCTUnwrap(
        scheduler.intermediateStep(modelOutput: k1, timestepIndex: step.index, sample: x0))
      let erased: any ZImageScheduler = scheduler
      injector.injectSubstep(
        sample: &mid, x0: x0, timestepIndex: step.index, row: 1, scheduler: erased)
      let snapshot = scheduler
      let build: (MLXArray) -> MLXArray = { anchor in
        var s = snapshot
        return s.intermediateStep(modelOutput: k1, timestepIndex: step.index, sample: anchor)!
      }
      _ = bong.iterate(
        x0: &x0, rowSample: mid, timestepIndex: step.index, row: 0, scheduler: erased,
        buildRowSample: build,
        evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) })

      if !step.bongmath.isEmpty {
        let residual = MLX.abs(build(x0).asType(.float32) - mid.asType(.float32))
          .max().item(Float.self)
        XCTAssertLessThan(
          residual, 1e-5, "step \(step.index): the tableau must reproduce the re-noised row")
      }

      let midSigma = try XCTUnwrap(scheduler.intermediateSigma(timestepIndex: step.index))
      let k2 = RES4LYFScriptedDenoiser.denoised(mid, sigma: midSigma)
      x = scheduler.finalizeStep(
        originalOutput: k1, intermediateOutput: k2, timestepIndex: step.index, sample: x0)
      let after: any ZImageScheduler = scheduler
      injector.inject(sample: &x, x0: x0, timestepIndex: step.index, scheduler: after)
    }
  }

  /// **Evidence for the one deviation from upstream** (`RES4LYFBongMath`'s
  /// dtype note): the hook hands the anchor back in the SAMPLE's dtype, so on
  /// a 3+-row tableau a later rebase starts from a bfloat16-rounded anchor,
  /// where upstream would have kept float32.
  ///
  /// That is harmless because the fixed point's LIMIT does not depend on the
  /// starting iterate — it is determined by the target, the row data
  /// predictions, `h` and the tableau, and 100 rounds of a contraction erase
  /// the start. Measured here rather than argued: the same rebase is run from
  /// the true anchor, from a bfloat16 round-trip of it, and from a badly
  /// corrupted one, and all three land on the same answer.
  func testTheFixedPointsLimitDoesNotDependOnTheStartingAnchor() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T3")
    let m = trace.manifest
    var scheduler = try XCTUnwrap(
      try RES4LYFTraceParityTests.productionRES2sScheduler() as? RES2sScheduler)
    let step = m.steps[0]
    let x = try trace.tensor(m.xInit)
    let k1 = RES4LYFScriptedDenoiser.denoised(x, sigma: Float(step.sigma))
    let mid = try trace.tensor(try XCTUnwrap(step.substeps.first?.xPost))
    let snapshot = scheduler

    func rebase(from start: MLXArray) -> MLXArray {
      var anchor = start
      _ = bongMath(m).iterate(
        x0: &anchor, rowSample: mid, timestepIndex: step.index, row: 0,
        scheduler: snapshot as any ZImageScheduler,
        buildRowSample: { candidate in
          var s = snapshot
          return s.intermediateStep(
            modelOutput: k1, timestepIndex: step.index, sample: candidate)!
        },
        evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) })
      return anchor.asType(.float32)
    }

    let truth = try trace.tensor(try XCTUnwrap(step.bongmath.first).x0).asType(.float32)
    let fromTrue = rebase(from: x)
    // The bfloat16 round-trip a later rebase would actually receive…
    let fromRounded = rebase(from: x.asType(.bfloat16).asType(.float32))
    // …and a start with no relationship to the answer at all.
    let fromGarbage = rebase(from: MLXArray.zeros(like: x.asType(.float32)))

    for (name, got) in [("true", fromTrue), ("bf16 round-trip", fromRounded),
                        ("zeros", fromGarbage)] {
      XCTAssertTraceClose(got, truth, rtol: 1e-6, "started from \(name)")
      XCTAssertEqual(
        MLX.abs(got - fromTrue).max().item(Float.self), 0, accuracy: 1e-6,
        "the limit must not depend on the start (\(name))")
    }
  }

  // MARK: - The gate has power

  /// **`bongmath` is load-bearing, and this is the measurement that says so.**
  /// Each T3 trace is re-run with the fixed point switched off — everything
  /// else identical, including the recorded noise — and the distance to
  /// `final/x` recorded. The "off" row IS what this loop produced before
  /// WP-E16, so it is the RED these gates went green from.
  ///
  /// Measured 2026-08-22, max-abs against a gate of 1e-4 absolute:
  ///
  /// | run                  | res_2s + beta 6 | deis_3m + bong_tangent 2 |
  /// |----------------------|-----------------|--------------------------|
  /// | T3 (the gate)        | **5.96e-7**     | **6.71e-8**              |
  /// | fixed point off (T2) | 3.26e-3  (33x)  | 4.18e-2  (418x)          |
  func testBongmathIsLoadBearing() throws {
    for name in ["res2s_beta6_T3", "deis3m_bong2_T3"] {
      let withBong = try distanceToFinal(name, bongmath: true)
      XCTAssertLessThan(withBong, 1e-4, "\(name): the T3 gate itself")

      let without = try distanceToFinal(name, bongmath: false)
      XCTAssertGreaterThan(
        without, 10 * 1e-4,
        "\(name): T2 alone must MISS the T3 trace — otherwise this gate proves nothing")
    }
  }

  /// Max-abs distance from the production loop's final latent to the trace's,
  /// with the fixed point switchable.
  private func distanceToFinal(_ name: String, bongmath: Bool) throws -> Float {
    let trace = try RES4LYFTraceFixture.load(name)
    let m = trace.manifest
    let (injector, _, _) = try recordedInjector(trace)
    var scheduler: any ZImageScheduler
    var startIndex = 0
    if name.hasPrefix("res2s") {
      scheduler = try RES4LYFTraceParityTests.productionRES2sScheduler()
    } else {
      let (base, start, _, _) = try DEISMultistepSchedulerTests.stage2Scheduler(order: .three)
      scheduler = base
      startIndex = start
    }
    let (x, _) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit), startIndex: startIndex,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: injector, bongmath: bongmath ? bongMath(m) : nil)
    return MLX.abs(x.asType(.float32) - (try trace.tensor(m.final.x)).asType(.float32))
      .max().item(Float.self)
  }

  // MARK: - `bongmath = false` is a provable no-op

  /// **The `bongmath = false` proof**, in the two halves it actually has.
  ///
  /// 1. The pipeline builds NO hook at all for a false request, on every
  ///    sampler — so `bongmath: false` and "no T3 argument" are the same call
  ///    into `Krea2DenoiseLoop`. That is the load-bearing half.
  /// 2. Driving the loop both ways is then bit-identical at equal `Stats` and
  ///    zero extra evaluations. **This half witnesses determinism, not
  ///    equivalence to pre-E16** — both runs pass `nil` for the hook, so it
  ///    cannot see a change to the driver itself.
  ///
  /// What witnesses equivalence to pre-E16 is the T1 and T2 trace gates, which
  /// are exact oracle fixtures and are unchanged by this WP
  /// (`RES4LYFTraceParityTests`, `RES4LYFEtaSDEParityTests`,
  /// `RalstonTraceParityTests`). The last assertion here ties this test to
  /// them: the answer is still the **T2** trace, so it is not two identical
  /// wrong answers.
  func testBongmathFalseBuildsNoHookAndIsBitIdenticalAtEqualCost() throws {
    for sampler in SchedulerKind.allCases {
      XCTAssertNil(
        try Krea2Pipeline.makeBongMath(
          bongmath: false, sampler: sampler, sigmaSchedule: .beta,
          shift: try Self.tracedShift()),
        "bongmath false must build no hook, whatever the sampler (\(sampler.rawValue))")
    }

    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T2")
    let m = trace.manifest
    let x0 = try trace.tensor(m.xInit)

    var omitted = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let (omittedInjector, _, _) = try recordedInjector(trace)
    let (baseline, baseStats) = Krea2DenoiseLoop.run(
      scheduler: &omitted, initialSample: x0,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: omittedInjector)

    var explicit = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let (explicitInjector, _, _) = try recordedInjector(trace)
    let (offed, offStats) = Krea2DenoiseLoop.run(
      scheduler: &explicit, initialSample: x0,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: explicitInjector,
      bongmath: try Krea2Pipeline.makeBongMath(
        bongmath: false, sampler: .res2s, sigmaSchedule: .beta, shift: try Self.tracedShift()))

    XCTAssertEqual(baseStats, offStats, "bongmath false changes no count")
    XCTAssertEqual(baseStats.evaluateCalls, 12, "and costs no extra evaluation")
    XCTAssertEqual(
      baseline.asType(.float32).asArray(Float.self),
      offed.asType(.float32).asArray(Float.self),
      "bongmath false must be bit-identical to no hook")
    XCTAssertTraceClose(offed, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0)
  }

  // MARK: - The tier gate (D18): a 400 that names the sampler

  /// `bongmath: true` with a sampler RES4LYF's fixed point is not defined
  /// against is refused BY SAMPLER, exactly as `eta` is — never ignored, never
  /// silently downgraded.
  func testBongmathOnANonRES4LYFSamplerIsRefusedByName() throws {
    for sampler in SchedulerKind.allCases where !sampler.isRES4LYFFamily {
      XCTAssertThrowsError(
        try Krea2Pipeline.makeBongMath(
          bongmath: true, sampler: sampler, sigmaSchedule: .beta, shift: try Self.tracedShift()),
        "bongmath must be refused on \(sampler.rawValue)"
      ) { error in
        guard case Krea2ScheduleError.bongmathUnsupportedSampler(let named) = error else {
          return XCTFail("\(sampler.rawValue): wrong error \(error)")
        }
        XCTAssertEqual(named, sampler.rawValue, "the refusal must name the sampler")
        XCTAssertTrue(
          "\(Krea2ScheduleError.bongmathUnsupportedSampler(sampler: named))"
            .contains(sampler.rawValue))
      }
    }
    for sampler in SchedulerKind.allCases where sampler.isRES4LYFFamily {
      XCTAssertNotNil(
        try Krea2Pipeline.makeBongMath(
          bongmath: true, sampler: sampler, sigmaSchedule: .beta, shift: try Self.tracedShift()))
    }
  }

  /// The tier gate no longer refuses `bongmath` as unimplemented: T3 has
  /// landed, so the only refusal left is the sampler one above.
  func testTierGateNoLongerRefusesBongmath() {
    XCTAssertNoThrow(try Krea2Pipeline.validateTiers(eta: 0.5, bongmath: true))
    XCTAssertNoThrow(try Krea2Pipeline.validateTiers(eta: 0, bongmath: true))
  }
}
