import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E15 — parity tier **T2**: RES4LYF's `eta` SDE
/// (FDD-krea2-raw-recipe §3.13, D18, AC-26/AC-27/AC-28).
///
/// The load-bearing gates here are the two T2 step traces E18 exported from
/// `sample_rk_beta` against the scripted denoiser — `res2s_beta6_T2` and
/// `deis3m_bong2_T2`, both at `eta = eta_substep = 0.5`, `noise_mode_sde =
/// "hard"`, `s_noise = 1.0`. They are driven through the **factory-built**
/// scheduler and the **production** `Krea2DenoiseLoop`, on OUR grid, exactly
/// as S-FIX-1 and E14 drive their T1 gates: nothing is read from the fixture
/// but the initial latent, the recorded noise tensors and the answer.
///
/// The noise stream is the one thing that cannot be re-derived — MLX's RNG is
/// not torch's — so the gate feeds the trace's **recorded** noise through the
/// injector's noise-stream seam (`RES4LYFNoiseStream`). That is what makes
/// this an exact reproduction of the oracle's trajectory rather than a
/// statistical check: every other number in the SDE (`σ_up`, `σ_down`,
/// `alpha_ratio`, the epsilon the swap is written in, the substep the swap
/// happens at, and the ORDER of the draws) comes from our own code.
final class RES4LYFEtaSDEParityTests: XCTestCase {

  // MARK: - Noise streams the gate drives the injector with

  /// Hands back the trace's recorded (already z-scored) noise tensors in the
  /// order upstream drew them, and refuses an unexpected draw.
  ///
  /// Order is itself under test: a port that re-noised the final row, or
  /// skipped a substep, or drew for a `σ' == 0` target would desynchronise
  /// here and fail on the very next tensor rather than merely drifting.
  final class RecordedNoiseStream: RES4LYFNoiseStream {
    private let name: String
    private let tensors: [MLXArray]
    private(set) var drawn = 0

    init(name: String, tensors: [MLXArray]) {
      self.name = name
      self.tensors = tensors
    }

    func next(like sample: MLXArray) -> MLXArray {
      guard drawn < tensors.count else {
        XCTFail("\(name): the injector asked for draw \(drawn + 1) but the trace recorded \(tensors.count)")
        return MLXArray.zeros(like: sample)
      }
      let t = tensors[drawn]
      drawn += 1
      XCTAssertEqual(t.shape, sample.shape, "\(name): draw \(drawn) shape")
      // float32, not `sample.dtype`: the stream's contract is that `swap` owns
      // the single cast, and the recorded tensors are upstream's work_dtype.
      return t.asType(.float32)
    }

    var exhausted: Bool { drawn == tensors.count }
  }

  /// Hands back the SAME tensor every draw, so a test can vary one thing about
  /// it (its dtype) and hold everything else fixed.
  final class FixedNoiseStream: RES4LYFNoiseStream {
    private let tensor: MLXArray
    init(_ tensor: MLXArray) { self.tensor = tensor }
    func next(like sample: MLXArray) -> MLXArray { tensor }
  }

  /// Counts draws and returns zeros: proves eta 0 draws nothing at all.
  final class CountingNoiseStream: RES4LYFNoiseStream {
    private(set) var drawn = 0
    func next(like sample: MLXArray) -> MLXArray {
      drawn += 1
      return MLXArray.zeros(sample.shape, dtype: .float32)
    }
  }

  /// The injector, wired to a trace's recorded noise.
  private func recordedInjector(_ trace: RES4LYFTraceFixture) throws
    -> (injector: RES4LYFSDENoiseInjector, step: RecordedNoiseStream, substep: RecordedNoiseStream)
  {
    let m = trace.manifest
    let stepNoise = try m.steps.compactMap { $0.noiseStep }.map { try trace.tensor($0) }
    let substepNoise = try m.steps.flatMap { $0.substeps }.compactMap { $0.noise }
      .map { try trace.tensor($0) }
    let step = RecordedNoiseStream(name: "\(trace.name) step noise", tensors: stepNoise)
    let substep = RecordedNoiseStream(name: "\(trace.name) substep noise", tensors: substepNoise)
    let injector = RES4LYFSDENoiseInjector(
      eta: Double(m.recipe.eta),
      etaSubstep: Double(m.recipe.etaSubstep),
      sNoise: Double(m.recipe.sNoise),
      sNoiseSubstep: Double(m.recipe.sNoise),
      sigmaMax: m.sigmaMax,
      stepNoise: step,
      substepNoise: substep)
    return (injector, step, substep)
  }

  // MARK: - The split is ours, and it is the oracle's

  /// The PRODUCTION split (`RES4LYFSDESplit.hardVP`) equals the fixture's
  /// recorded `σ_up` / `σ_down` / `alpha_ratio` at every step and every
  /// re-noised substep of both T2 traces, and equals the harness's
  /// independently written closed form. Two implementations, one oracle.
  func testProductionHardVPSplitMatchesTheOracleAndTheHarness() throws {
    for name in ["res2s_beta6_T2", "deis3m_bong2_T2"] {
      let m = try RES4LYFTraceFixture.load(name).manifest
      let eta = m.recipe.eta
      for step in m.steps {
        let got = RES4LYFSDESplit.hardVP(sigmaNext: step.sigmaNext, eta: eta, sigmaMax: m.sigmaMax)
        let harness = RES4LYFHardVPSplit.split(sigmaNext: step.sigmaNext, eta: eta, sigmaMax: m.sigmaMax)
        XCTAssertEqual(got.up, step.sigmaUpEta, accuracy: 1e-12, "\(name) step \(step.index) σ_up")
        XCTAssertEqual(got.down, step.sigmaDownEta, accuracy: 1e-12, "\(name) step \(step.index) σ_down")
        XCTAssertEqual(got.alpha, step.alphaRatioEta, accuracy: 1e-12, "\(name) step \(step.index) α")
        XCTAssertEqual(got.up, harness.up, accuracy: 1e-15, "\(name) step \(step.index): two derivations")
        XCTAssertEqual(got.down, harness.down, accuracy: 1e-15)
        XCTAssertEqual(got.alpha, harness.alpha, accuracy: 1e-15)

        for sub in step.substeps where sub.noise != nil {
          let g = RES4LYFSDESplit.hardVP(
            sigmaNext: sub.subSigmaNext, eta: m.recipe.etaSubstep, sigmaMax: m.sigmaMax)
          XCTAssertEqual(g.up, sub.subSigmaUpEta, accuracy: 1e-12, "\(name) step \(step.index) row \(sub.row) σ_up")
          XCTAssertEqual(g.down, sub.subSigmaDownEta, accuracy: 1e-12)
          XCTAssertEqual(g.alpha, sub.subAlphaRatioEta, accuracy: 1e-12)
        }
      }
    }
  }

  /// `eta = 0` is the identity split — `σ_up = 0`, `σ_down = σ'`, `α = 1` — so
  /// the injector's early return is the same statement as the arithmetic's.
  func testZeroEtaSplitIsTheIdentity() {
    for sigmaNext in [1.0, 0.5, 3.1575e-4] {
      let s = RES4LYFSDESplit.hardVP(sigmaNext: sigmaNext, eta: 0, sigmaMax: 1.0)
      XCTAssertEqual(s.up, 0)
      XCTAssertEqual(s.down, sigmaNext)
      XCTAssertEqual(s.alpha, 1)
    }
    // …and a zero target is never re-noised whatever eta says.
    let z = RES4LYFSDESplit.hardVP(sigmaNext: 0, eta: 0.5, sigmaMax: 1.0)
    XCTAssertEqual(z.up, 0)
  }

  /// `eta ≥ 1` would put `σ_up` at or above `σ'` and take the square root of a
  /// negative number; upstream clamps to `σ'·0.9999` first
  /// (`rk_noise_sampler_beta.py:279-284`).
  func testEtaAtOrAboveOneTakesUpstreamsClamp() {
    let s = RES4LYFSDESplit.hardVP(sigmaNext: 0.5, eta: 1.0, sigmaMax: 1.0)
    XCTAssertEqual(s.up, 0.5 * 0.9999, accuracy: 1e-15)
    XCTAssertTrue(s.down.isFinite && s.down > 0, "σ_down \(s.down)")
    XCTAssertTrue(s.alpha.isFinite && s.alpha > 0, "α \(s.alpha)")
  }

  // MARK: - AC-26 T2, res_2s: the PRODUCTION loop against the oracle

  /// The T2 gate for stage 1. Factory-built `res_2s + beta` at 6 steps under
  /// `ModelSamplingFlux(1.15)`, driven by `Krea2DenoiseLoop` with the SDE
  /// injector attached, reproduces the trace's FINAL tensor — after the
  /// model-free `σ_min → 0` conversion, which the loop applies.
  ///
  /// `floor: 1.0` makes this an ABSOLUTE 1e-4, for the reason S-FIX-1 recorded
  /// on the T1 gate: the final latent is the data prediction at σ_min and its
  /// magnitude has collapsed to ~1e-2, so a tolerance relative to it would
  /// measure float32 ulps rather than agreement.
  func testRES2sT2ThroughTheDenoiseLoopReproducesTheTraceFinalTensor() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T2")
    let m = trace.manifest
    var scheduler = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let (injector, stepNoise, substepNoise) = try recordedInjector(trace)

    let (x, stats) = try Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit),
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: injector)

    // The SDE costs no model evaluation and takes no extra step (§3.13 scope 4).
    XCTAssertEqual(stats.stepsRun, 6)
    XCTAssertEqual(stats.rowsAtStart, 2)
    XCTAssertEqual(stats.evaluateCalls, 12, "2 rows × 6 steps — unchanged by eta")
    XCTAssertEqual(stats.modelEvals, m.modelCallsTotal)
    XCTAssertEqual(Double(try XCTUnwrap(stats.finalConversionSigma)), m.sigmaMin, accuracy: 1e-9)

    // Every recorded draw was consumed, and no more were asked for.
    XCTAssertEqual(stepNoise.drawn, 6, "one step re-noise per step")
    XCTAssertEqual(substepNoise.drawn, 6, "one substep re-noise per step (row 0 of 2)")
    XCTAssertTrue(stepNoise.exhausted && substepNoise.exhausted)

    XCTAssertTraceClose(
      x, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }

  /// …and step by step, so a divergence is located rather than merely
  /// detected: the substep sample the second row is evaluated on (`x_post`,
  /// i.e. AFTER the substep re-noise) and the step result (`x_out`, AFTER the
  /// step re-noise) both come from our scheduler plus our injector.
  func testRES2sT2MatchesTheTraceStepByStep() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T2")
    let m = trace.manifest
    var scheduler = try XCTUnwrap(
      try RES4LYFTraceParityTests.productionRES2sScheduler() as? RES2sScheduler)
    scheduler.reset()
    let (injector, _, _) = try recordedInjector(trace)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      XCTAssertTraceClose(x, try trace.tensor(step.x0), rtol: 1e-4, "step \(step.index): x entering")
      let x0 = x

      let k1 = RES4LYFScriptedDenoiser.denoised(x, sigma: Float(step.sigma))
      var mid = try XCTUnwrap(
        scheduler.intermediateStep(modelOutput: k1, timestepIndex: step.index, sample: x))
      let sub = try XCTUnwrap(step.substeps.first)
      XCTAssertTraceClose(
        mid, try trace.tensor(try XCTUnwrap(sub.xPre)), rtol: 1e-4,
        "step \(step.index): x before the substep re-noise")

      var erased: any ZImageScheduler = scheduler
      injector.injectSubstep(
        sample: &mid, x0: x0, timestepIndex: step.index, row: 1, scheduler: erased)
      XCTAssertTraceClose(
        mid, try trace.tensor(try XCTUnwrap(sub.xPost)), rtol: 1e-4,
        "step \(step.index): x after the substep re-noise")
      XCTAssertTraceClose(
        mid, try trace.tensor(step.modelCalls[1].xIn), rtol: 1e-4,
        "step \(step.index): the second row is evaluated on the RE-NOISED sample")

      let midSigma = try XCTUnwrap(scheduler.intermediateSigma(timestepIndex: step.index))
      let k2 = RES4LYFScriptedDenoiser.denoised(mid, sigma: midSigma)
      x = scheduler.finalizeStep(
        originalOutput: k1, intermediateOutput: k2, timestepIndex: step.index, sample: x0)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")

      erased = scheduler
      injector.inject(sample: &x, x0: x0, timestepIndex: step.index, scheduler: erased)
      XCTAssertTraceClose(x, try trace.tensor(step.xOut), rtol: 1e-4, "step \(step.index): x_out")
    }
  }

  // MARK: - AC-26 T2, deis_3m: three rows, two substep re-noises

  /// The T2 gate for stage 2 — the published `deis_3m`, 2 steps at denoise
  /// 0.2, whose every step is the `ralston_3s` warm-up (AC-24). Three rows
  /// means TWO substep re-noises per step plus the step re-noise: an injector
  /// that only implemented the step-level swap misses this tensor.
  func testDEIS3mT2ThroughTheDenoiseLoopReproducesTheTraceFinalTensor() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T2")
    let m = trace.manifest
    let (base, startIndex, _, _) = try DEISMultistepSchedulerTests.stage2Scheduler(order: .three)
    var scheduler = base
    let (injector, stepNoise, substepNoise) = try recordedInjector(trace)

    let (x, stats) = try Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit), startIndex: startIndex,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: injector)

    XCTAssertEqual(stats.stepsRun, 2)
    XCTAssertEqual(stats.evaluateCalls, m.modelCallsTotal, "6 calls, unchanged by eta")
    XCTAssertEqual(stats.modelEvals, 6)
    XCTAssertEqual(stepNoise.drawn, 2)
    XCTAssertEqual(substepNoise.drawn, 4, "rows 1 and 2 of 3, on both steps")
    XCTAssertTrue(stepNoise.exhausted && substepNoise.exhausted)

    XCTAssertTraceClose(
      x, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }

  /// …row by row: each row's sample is the tableau's, re-noised, and the model
  /// call the oracle made at that row was made on exactly that tensor.
  func testDEIS3mT2MatchesTheTraceRowByRow() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T2")
    let m = trace.manifest
    let (base, startIndex, _, _) = try DEISMultistepSchedulerTests.stage2Scheduler(order: .three)
    var scheduler = try XCTUnwrap(base as? DEISMultistepScheduler)
    scheduler.reset()
    let (injector, _, _) = try recordedInjector(trace)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      let i = startIndex + step.index
      XCTAssertEqual(scheduler.rows, 3, "step \(step.index): still warming up")
      XCTAssertTraceClose(x, try trace.tensor(step.x0), rtol: 1e-4, "step \(step.index): x entering")
      let x0 = x

      var k: [MLXArray] = []
      for r in 0..<scheduler.rows {
        let call = step.modelCalls[r]
        var xr = r == 0 ? x : scheduler.rowSample(timestepIndex: i, row: r, x0: x0, k: k)
        if r > 0 {
          let sub = step.substeps[r - 1]
          XCTAssertEqual(sub.row, r - 1, "substep record for row \(r)")
          XCTAssertTraceClose(
            xr, try trace.tensor(try XCTUnwrap(sub.xPre)), rtol: 1e-4,
            "step \(step.index) row \(r): before the substep re-noise")
          let erased: any ZImageScheduler = scheduler
          injector.injectSubstep(
            sample: &xr, x0: x0, timestepIndex: i, row: r, scheduler: erased)
          XCTAssertTraceClose(
            xr, try trace.tensor(try XCTUnwrap(sub.xPost)), rtol: 1e-4,
            "step \(step.index) row \(r): after the substep re-noise")
        }
        XCTAssertTraceClose(
          xr, try trace.tensor(call.xIn), rtol: 1e-4, "step \(step.index) row \(r): row sample")
        let sigmaR = Float(call.sTmp)
        k.append(
          scheduler.modelInput(
            velocity: RES4LYFScriptedDenoiser.velocity(xr, sigma: sigmaR), sample: xr, sigma: sigmaR))
      }

      x = scheduler.commit(timestepIndex: i, x0: x0, k: k)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")
      let erased: any ZImageScheduler = scheduler
      injector.inject(sample: &x, x0: x0, timestepIndex: i, scheduler: erased)
      XCTAssertTraceClose(x, try trace.tensor(step.xOut), rtol: 1e-4, "step \(step.index): x_out")
    }
  }

  // MARK: - The gate has power

  /// **Both halves of the swap are load-bearing, and this is the measurement
  /// that says so** — otherwise the T2 gates above would only prove that some
  /// code ran.
  ///
  /// Each trace is re-run with one half of the SDE switched off and the
  /// distance to `final/x` recorded. Measured 2026-08-22, against a gate of
  /// 1e-4 absolute:
  ///
  /// | run                    | res_2s + beta 6 | deis_3m + bong_tangent 2 |
  /// |------------------------|-----------------|--------------------------|
  /// | full T2 (the gate)     | **5.37e-7**     | **9.69e-8**              |
  /// | no SUBSTEP swap        | 1.05e-2 (105x)  | 1.32e-2 (132x)           |
  /// | no STEP swap           | 4.16e-3 (42x)   | 2.04e-2 (204x)           |
  /// | eta 0 entirely (T1)    | 1.16e-2 (116x)  | 3.26e-2 (326x)           |
  ///
  /// The "no substep swap" row is the one worth staring at: it is the port
  /// mistake FDD 3.13's step-level formula invites, and it misses the answer by
  /// two orders of magnitude on both recipes.
  func testBothHalvesOfTheSwapAreLoadBearing() throws {
    for name in ["res2s_beta6_T2", "deis3m_bong2_T2"] {
      let full = try distanceToFinal(name, eta: nil, etaSubstep: nil)
      XCTAssertLessThan(full, 1e-4, "\(name): the T2 gate itself")

      let noSubstep = try distanceToFinal(name, eta: nil, etaSubstep: 0)
      XCTAssertGreaterThan(
        noSubstep, 100 * 1e-4,
        "\(name): a step-only injector must MISS the trace by two orders of magnitude")

      let noStep = try distanceToFinal(name, eta: 0, etaSubstep: nil)
      XCTAssertGreaterThan(
        noStep, 10 * 1e-4, "\(name): a substep-only injector must miss the trace too")

      let neither = try distanceToFinal(name, eta: 0, etaSubstep: 0)
      XCTAssertGreaterThan(neither, 100 * 1e-4, "\(name): T1 is not T2")
    }
  }

  /// Max-abs distance from the production loop's final latent to the trace's,
  /// with the SDE's two etas overridable so half of it can be switched off.
  private func distanceToFinal(_ name: String, eta: Double?, etaSubstep: Double?) throws -> Float {
    let trace = try RES4LYFTraceFixture.load(name)
    let m = trace.manifest
    let stepTensors = try m.steps.compactMap { $0.noiseStep }.map { try trace.tensor($0) }
    let substepTensors = try m.steps.flatMap { $0.substeps }.compactMap { $0.noise }
      .map { try trace.tensor($0) }
    // The disabled half draws nothing, so its stream is never asked; the other
    // half's draw order is unchanged.
    let injector = RES4LYFSDENoiseInjector(
      eta: eta ?? m.recipe.eta,
      etaSubstep: etaSubstep ?? m.recipe.etaSubstep,
      sNoise: m.recipe.sNoise, sNoiseSubstep: m.recipe.sNoise,
      sigmaMax: m.sigmaMax,
      stepNoise: RecordedNoiseStream(name: "\(name) step", tensors: stepTensors),
      substepNoise: RecordedNoiseStream(name: "\(name) substep", tensors: substepTensors))

    var scheduler: any ZImageScheduler
    var startIndex = 0
    if name.hasPrefix("res2s") {
      scheduler = try RES4LYFTraceParityTests.productionRES2sScheduler()
    } else {
      let (base, start, _, _) = try DEISMultistepSchedulerTests.stage2Scheduler(order: .three)
      scheduler = base
      startIndex = start
    }
    let (x, _) = try Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit), startIndex: startIndex,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: injector)
    return MLX.abs(x.asType(.float32) - (try trace.tensor(m.final.x)).asType(.float32))
      .max().item(Float.self)
  }

  // MARK: - AC-26 T1 still holds: eta 0 is a provable no-op

  /// **The eta = 0 proof.** The same factory scheduler and the same denoiser,
  /// run twice through the production loop — once with `noise: nil` (the
  /// pre-E15 path), once with an injector at `eta = 0` — produce
  /// BIT-IDENTICAL latents, and the injector draws no noise at all.
  ///
  /// Bit-identical, not close: the injector's early return means the loop
  /// executes the same ops on the same values, so anything short of exact
  /// equality would be a change to the default path.
  func testEtaZeroIsBitIdenticalToNoInjector() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T1")
    let m = trace.manifest
    let x0 = try trace.tensor(m.xInit)

    var withoutInjector = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let (baseline, baseStats) = try Krea2DenoiseLoop.run(
      scheduler: &withoutInjector, initialSample: x0,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) })

    let stepNoise = CountingNoiseStream()
    let substepNoise = CountingNoiseStream()
    var withInjector = try RES4LYFTraceParityTests.productionRES2sScheduler()
    let (injected, injectedStats) = try Krea2DenoiseLoop.run(
      scheduler: &withInjector, initialSample: x0,
      evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
      noise: RES4LYFSDENoiseInjector(
        eta: 0, etaSubstep: 0, sNoise: 1.0, sNoiseSubstep: 1.0, sigmaMax: 1.0,
        stepNoise: stepNoise, substepNoise: substepNoise))

    XCTAssertEqual(stepNoise.drawn, 0, "eta 0 draws no step noise")
    XCTAssertEqual(substepNoise.drawn, 0, "eta 0 draws no substep noise")
    XCTAssertEqual(baseStats, injectedStats, "eta 0 changes no count")
    XCTAssertEqual(
      baseline.asType(.float32).asArray(Float.self),
      injected.asType(.float32).asArray(Float.self),
      "eta 0 must be bit-identical to no injector")
    // …and it is still the T1 trace, so this is not two identical wrong answers.
    XCTAssertTraceClose(injected, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0)
  }

  // MARK: - The production noise stream

  /// The seeded gaussian stream is deterministic in the stage's seed and
  /// independent of MLX's global RNG (AC-27): identical seeds give identical
  /// draws, a different seed gives different ones, and drawing does not move
  /// the global stream the initial latent is sampled from.
  func testSeededGaussianStreamIsDeterministicAndGlobalRNGFree() {
    let shape = [1, 16, 8, 8]
    func draws(seed: UInt64) -> [[Float]] {
      let s = RES4LYFGaussianNoiseStream(seed: seed, layout: .channelsAtAxis1)
      let probe = MLXArray.zeros(shape, dtype: .float32)
      return (0..<3).map { _ in s.next(like: probe).asArray(Float.self) }
    }
    let a = draws(seed: 4243)
    let b = draws(seed: 4243)
    let c = draws(seed: 4244)
    XCTAssertEqual(a, b, "same seed, same stream")
    XCTAssertNotEqual(a, c, "a different stage seed is a different stream")
    XCTAssertNotEqual(a[0], a[1], "successive draws are independent")

    // Global-RNG independence: seeding globally, drawing from the stream, and
    // then sampling globally gives the same global sample as not drawing.
    MLXRandom.seed(99)
    let untouched = MLXRandom.normal(shape).asArray(Float.self)
    MLXRandom.seed(99)
    _ = draws(seed: 4243)
    let after = MLXRandom.normal(shape).asArray(Float.self)
    XCTAssertEqual(untouched, after, "the injector must not consume the global RNG stream")
  }

  /// Upstream z-scores every draw per channel (`normalize_zscore(channelwise:
  /// True)`) before the `σ_up` multiply, and the recorded tensors carry it:
  /// per-channel mean 0 and unbiased std 1. Our stream must produce noise with
  /// the same normalisation, in both latent layouts Krea 2 uses.
  func testSeededGaussianStreamIsChannelwiseZScored() {
    func check(_ noise: MLXArray, groups: [MLXArray], _ what: String) {
      for (c, g) in groups.enumerated() {
        XCTAssertEqual(g.mean().item(Float.self), 0, accuracy: 2e-5, "\(what) channel \(c) mean")
        XCTAssertEqual(
          MLX.std(g, axes: Array(0..<g.ndim), ddof: 1).item(Float.self), 1, accuracy: 2e-5,
          "\(what) channel \(c) std")
      }
    }

    let nchw = RES4LYFGaussianNoiseStream(seed: 7, layout: .channelsAtAxis1)
      .next(like: MLXArray.zeros([1, 16, 8, 8], dtype: .float32))
    check(nchw, groups: (0..<16).map { nchw[0, $0] }, "NCHW")

    // Krea 2's working latent is patchified `(1, tokens, C·p·p)`, channel `c`
    // owning the contiguous slice `c·p² ..< (c+1)·p²` of the last axis.
    let patched = RES4LYFGaussianNoiseStream(seed: 7, layout: .patchifiedTrailing(channels: 16))
      .next(like: MLXArray.zeros([1, 64, 64], dtype: .float32))
    check(patched, groups: (0..<16).map { patched[0..., 0..., ($0 * 4)..<(($0 + 1) * 4)] }, "patchified")
  }

  /// …and the patchified grouping IS the latent's channel grouping, checked
  /// where it matters rather than where it is convenient: un-patchify the
  /// noise the production layout produced and the result is z-scored per
  /// LATENT channel, which is the statement upstream's
  /// `normalize_zscore(channelwise: True)` makes about `(1, C, H, W)`.
  ///
  /// Without this the patchified case would only be self-consistent — it would
  /// pass just as happily if `Krea2Sampling.patchify` interleaved channels
  /// instead of grouping them.
  func testPatchifiedZScoreIsPerLatentChannelAfterUnpatchify() {
    let channels = Krea2VAE.latentChannels  // 16
    let patch = 2
    let hTok = 6, wTok = 5
    let noise = RES4LYFGaussianNoiseStream(
      seed: 31, layout: .patchifiedTrailing(channels: channels)
    ).next(like: MLXArray.zeros([1, hTok * wTok, channels * patch * patch], dtype: .float32))

    let nchw = Krea2Sampling.unpatchify(noise, patch: patch, h: hTok, w: wTok, c: channels)
    XCTAssertEqual(nchw.shape, [1, channels, hTok * patch, wTok * patch])
    for c in 0..<channels {
      let plane = nchw[0, c]
      XCTAssertEqual(plane.mean().item(Float.self), 0, accuracy: 2e-5, "channel \(c) mean")
      XCTAssertEqual(
        MLX.std(plane, axes: [0, 1], ddof: 1).item(Float.self), 1, accuracy: 2e-5,
        "channel \(c) std")
    }
    // And it is genuinely per channel, not a whole-tensor normalisation that
    // happens to look per-channel: the GLOBAL unbiased std of a per-channel
    // z-score is not 1 (the fixtures' recorded noise reads 0.9926 the same way).
    XCTAssertNotEqual(
      MLX.std(nchw, axes: Array(0..<nchw.ndim), ddof: 1).item(Float.self), 1, accuracy: 1e-6)
  }

  // MARK: - Who owns the cast (E15 review, item 1)

  /// **The stream must NOT pre-round to the sample's dtype.** Upstream keeps
  /// noise in `work_dtype` right through the `σ_up · noise` multiply and
  /// rounds once, at the end of the swap.
  ///
  /// Every other test here probes with a float32 sample, where the two
  /// spellings are indistinguishable — so this is the ONLY thing standing
  /// between the production bfloat16 path and a silent revert to
  /// `.asType(sample.dtype)`.
  func testTheStreamDrawsFloat32WhateverTheSampleDtype() {
    let stream = RES4LYFGaussianNoiseStream(seed: 11, layout: .channelsAtAxis1)
    let shape = [1, 16, 8, 8]

    let fromBF16 = stream.next(like: MLXArray.zeros(shape, dtype: .bfloat16))
    XCTAssertEqual(
      fromBF16.dtype, .float32,
      "a bfloat16 sample must not narrow the draw — swap owns the single cast")
    XCTAssertEqual(fromBF16.shape, shape)

    let fromF32 = stream.next(like: MLXArray.zeros(shape, dtype: .float32))
    XCTAssertEqual(fromF32.dtype, .float32)

    // …and the narrowing this prevents is not cosmetic: bfloat16 keeps 8
    // mantissa bits, so rounding the draw first genuinely loses the value.
    let narrowed = fromBF16.asType(.bfloat16).asType(.float32)
    XCTAssertGreaterThan(
      MLX.abs(fromBF16 - narrowed).max().item(Float.self), 1e-4,
      "if this is 0 the draw was already bfloat16 and the test proves nothing")
  }

  /// …and the sibling half: `swap` casts back to the SAMPLE's dtype, once, so
  /// a bfloat16 latent stays bfloat16 and the float32 noise never widens it.
  func testTheSwapCastsBackToTheSampleDtypeExactlyOnce() {
    let sigmas: [Float] = [1.0, 0.6, 0.3, 3.157_511_5e-4]
    let scheduler: any ZImageScheduler = RES2sScheduler(
      numInferenceSteps: 3, sigmaValues: sigmas)
    let shape = [1, 16, 8, 8]
    let noise = RES4LYFGaussianNoiseStream(seed: 5, layout: .channelsAtAxis1)
      .next(like: MLXArray.zeros(shape, dtype: .float32))
    let x0 = MLXRandom.normal(shape).asType(.bfloat16)
    let xNext = (x0 * MLXArray(Float(0.6)).asType(.bfloat16)).asType(.bfloat16)

    func swapped(with draw: MLXArray) -> MLXArray {
      var x = xNext
      let injector = RES4LYFSDENoiseInjector(
        eta: 0.5, etaSubstep: 0.5, sNoise: 1.0, sNoiseSubstep: 1.0, sigmaMax: 1.0,
        stepNoise: FixedNoiseStream(draw), substepNoise: FixedNoiseStream(draw))
      injector.inject(sample: &x, x0: x0, timestepIndex: 0, scheduler: scheduler)
      return x
    }

    let fromFloat32 = swapped(with: noise)
    XCTAssertEqual(
      fromFloat32.dtype, .bfloat16, "the swap must hand the loop back its own width")
    XCTAssertEqual(x0.dtype, .bfloat16, "…without having widened the sample it read")

    // The cast's PLACEMENT is observable at the output, which is what makes
    // this a regression gate rather than a type annotation.
    let fromPreRounded = swapped(with: noise.asType(.bfloat16))
    XCTAssertNotEqual(
      fromFloat32.asType(.float32).asArray(Float.self),
      fromPreRounded.asType(.float32).asArray(Float.self),
      "rounding the draw before the σ_up multiply changes the result")
  }

  /// The production stream seeds mirror upstream's: `SharkSampler` hands
  /// `sample_rk_beta` `noise_seed = seed + 1` (the fixtures record exactly
  /// that — `seed 4242`, `noise_seed_sde 4243`), and `rk_sampler_beta.py:364`
  /// derives the substep stream as `noise_seed + MAX_STEPS` with
  /// `MAX_STEPS = 10000`.
  func testStreamSeedsAreDerivedFromTheStageSeedAsUpstreamDoes() throws {
    let m = try RES4LYFTraceFixture.load("res2s_beta6_T2").manifest
    XCTAssertEqual(
      RES4LYFSDENoiseInjector.stepNoiseSeed(stageSeed: UInt64(m.recipe.seed)),
      UInt64(m.recipe.noiseSeedSde), "noise_seed_sde = seed + 1")
    XCTAssertEqual(
      RES4LYFSDENoiseInjector.substepNoiseSeed(stageSeed: UInt64(m.recipe.seed)),
      UInt64(m.recipe.noiseSeedSde) + 10_000, "noise_seed_substep = noise_seed + MAX_STEPS")
    // Per stage, not per render: a different stage seed is a different stream.
    XCTAssertNotEqual(
      RES4LYFSDENoiseInjector.stepNoiseSeed(stageSeed: 1),
      RES4LYFSDENoiseInjector.stepNoiseSeed(stageSeed: 2))
    // …and the two streams never coincide, which is why upstream offsets at all.
    XCTAssertNotEqual(
      RES4LYFSDENoiseInjector.stepNoiseSeed(stageSeed: 4242),
      RES4LYFSDENoiseInjector.substepNoiseSeed(stageSeed: 4242))
  }

  /// AC-27's determinism, at the loop: two runs of the SAME payload — same
  /// seed, same grid, same sampler — are byte-identical, and changing only the
  /// seed changes the latent.
  func testSeededInjectorRunsAreReproducibleAndSeedSensitive() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T2")
    let x0 = try trace.tensor(trace.manifest.xInit)

    func run(seed: UInt64) throws -> [Float] {
      var scheduler = try RES4LYFTraceParityTests.productionRES2sScheduler()
      let (x, _) = try Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: x0,
        evaluate: { RES4LYFScriptedDenoiser.velocity($0, sigma: $1) },
        noise: RES4LYFSDENoiseInjector(
          eta: 0.5, stageSeed: seed, layout: .channelsAtAxis1))
      return x.asType(.float32).asArray(Float.self)
    }

    XCTAssertEqual(try run(seed: 4242), try run(seed: 4242), "identical payloads, identical output")
    XCTAssertNotEqual(try run(seed: 4242), try run(seed: 4243), "the stage seed moves the stream")
  }
}
