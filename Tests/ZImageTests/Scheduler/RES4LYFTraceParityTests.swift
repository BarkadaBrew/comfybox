import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E18 — the RES4LYF step-trace harness, exercised against its own fixtures
/// (FDD-krea2-raw-recipe §5.2, AC-26's instrument).
///
/// Six traces, `{res_2s + beta, 6} × {deis_3m + bong_tangent, 2 @ denoise 0.2}`
/// at tiers T1 (eta 0, bongmath off), T2 (eta 0.5), T3 (eta 0.5 + bongmath),
/// were exported from RES4LYF `sample_rk_beta` against the scripted denoiser
/// `0.5·tanh(x) + 0.25·σ − 0.1·x` on a 1×16×8×8 latent. Nothing here needs
/// weights or a GPU.
///
/// What this file proves about the *harness* (E18's deliverable):
///   * every trace loads and is shaped as pinned;
///   * the Swift scripted denoiser reproduces every recorded model call, so a
///     Swift sampler driven by it sees the same field the oracle saw;
///   * the recorded epsilons follow RES4LYF's anchoring convention;
///   * the recorded eta split equals the closed-form hard-mode VP split, and
///     the recorded re-noise is reconstructible from x₀, x_next, the split and
///     the exported noise tensor — the Swift side never reproduces torch's RNG;
///   * the final linear tail from σ_min to 0 is what RES4LYF does.
/// And one genuine AC-26 T1 gate already met by existing code:
///   * `RES2sScheduler` (post-E2, data-prediction input) reproduces the
///     `res_2s + beta` T1 trace row by row to 1e-4 relative.
/// The T2/T3 and DEIS gates land with WP-E14/E15/E16 on top of this harness.
final class RES4LYFTraceParityTests: XCTestCase {

  private func loadAll() throws -> [RES4LYFTraceFixture] {
    try RES4LYFTraceFixture.names.map { try RES4LYFTraceFixture.load($0) }
  }

  // MARK: - Fixture shape

  func testTraceFixturesPresentAndShapedAsPinned() throws {
    for trace in try loadAll() {
      let m = trace.manifest
      XCTAssertEqual(m.latentShape, [1, 16, 8, 8], "\(trace.name): latent shape")
      XCTAssertEqual(m.denoiser, "0.5*tanh(x) + 0.25*sigma - 0.1*x", "\(trace.name): denoiser")
      XCTAssertEqual(m.recipe.modelSampling, "ModelSamplingFlux", "\(trace.name): traced under ComfyUI's Krea 2 registration")
      XCTAssertEqual(m.recipe.shift, 1.15)
      XCTAssertEqual(m.recipe.noiseModeSde, "hard")
      XCTAssertEqual(m.recipe.noiseAnchor, 1.0)
      XCTAssertEqual(m.sigmaMax, 1.0)

      switch m.recipe.tier {
      case "T1": XCTAssertEqual(m.recipe.eta, 0.0); XCTAssertFalse(m.recipe.bongmath)
      case "T2": XCTAssertEqual(m.recipe.eta, 0.5); XCTAssertFalse(m.recipe.bongmath)
      case "T3": XCTAssertEqual(m.recipe.eta, 0.5); XCTAssertTrue(m.recipe.bongmath)
      default: XCTFail("\(trace.name): unknown tier \(m.recipe.tier)")
      }
      XCTAssertEqual(m.recipe.etaSubstep, m.recipe.eta, "\(trace.name): ClownsharKSampler sets eta_substep = eta")

      if trace.name.hasPrefix("res2s") {
        XCTAssertEqual(m.recipe.sampler, "res_2s")
        XCTAssertEqual(m.recipe.scheduler, "beta")
        XCTAssertEqual(m.recipe.steps, 6)
        XCTAssertEqual(m.recipe.denoise, 1.0)
        XCTAssertEqual(m.sigmasSchedule.count, 7)
      } else {
        XCTAssertEqual(m.recipe.sampler, "deis_3m")
        XCTAssertEqual(m.recipe.scheduler, "bong_tangent")
        XCTAssertEqual(m.recipe.steps, 2)
        XCTAssertEqual(m.recipe.denoise, 0.2)
        XCTAssertEqual(m.sigmasSchedule.count, 3)
      }

      // RES4LYF's prepare_sigmas inserts σ_min before the trailing 0, so the
      // loop runs `schedule steps` real steps, the last one landing on σ_min.
      XCTAssertEqual(m.sigmasRun.count, m.sigmasSchedule.count + 1, "\(trace.name): σ_min inserted")
      XCTAssertEqual(m.sigmasRun[m.sigmasRun.count - 2], m.sigmaMin, accuracy: 1e-12)
      XCTAssertEqual(m.sigmasRun.last, 0.0)
      XCTAssertEqual(m.steps.count, m.sigmasRun.count - 2, "\(trace.name): one record per loop step")
      XCTAssertTrue(m.final.linearTailFromSigmaMin)

      for (i, step) in m.steps.enumerated() {
        XCTAssertEqual(step.index, i)
        XCTAssertEqual(step.sigma, m.sigmasRun[i], accuracy: 1e-12)
        XCTAssertEqual(step.sigmaNext, m.sigmasRun[i + 1], accuracy: 1e-12)
        XCTAssertEqual(step.substepSigmas.count, step.cNodes.count)
        XCTAssertEqual(step.aMatrix.count, step.rows)
        XCTAssertEqual(step.bWeights.first?.count, step.rows)
        XCTAssertFalse(step.modelCalls.isEmpty, "\(trace.name) step \(i): no model calls")
        _ = try trace.tensor(step.x0)
        _ = try trace.tensor(step.xNext)
        _ = try trace.tensor(step.xOut)
        for call in step.modelCalls {
          XCTAssertEqual(try trace.tensor(call.xIn).shape, [1, 16, 8, 8])
          XCTAssertEqual(try trace.tensor(call.denoised).shape, [1, 16, 8, 8])
        }
      }
      XCTAssertEqual(
        m.modelCallsTotal, m.steps.reduce(0) { $0 + $1.modelCalls.count },
        "\(trace.name): model_calls_total equals the recorded calls")
      _ = try trace.tensor(m.xInit)
      _ = try trace.tensor(m.final.x)
    }
  }

  /// The published stage-2 settings (`deis_3m`, 2 steps): both loop steps run
  /// `ralston_3s` — the warm-up — so true DEIS coefficients never engage and the
  /// stage costs exactly 6 model evaluations (FDD §3.12).
  func testDEIS3mTwoStepStageIsAllRalstonWarmup() throws {
    for tier in ["T1", "T2", "T3"] {
      let trace = try RES4LYFTraceFixture.load("deis3m_bong2_\(tier)")
      let m = trace.manifest
      XCTAssertEqual(m.steps.count, 2, "\(trace.name)")
      for step in m.steps {
        XCTAssertEqual(step.rkTypeRequested, "deis_3m", "\(trace.name) step \(step.index)")
        XCTAssertEqual(step.rkType, "ralston_3s", "\(trace.name) step \(step.index)")
        XCTAssertFalse(step.exponential)
        XCTAssertEqual(step.multistepStages, 0)
        XCTAssertEqual(step.rows, 3)
        XCTAssertEqual(step.rowOffset, 1)
        XCTAssertEqual(step.modelCalls.count, 3)
        // ralston_3s tableau, verbatim from r4_rk_coeff.py.
        // (RES4LYF appends the final-row node 1 to every tableau's c.)
        XCTAssertEqual(step.cNodes, [0.0, 0.5, 0.75, 1.0])
        XCTAssertEqual(step.aMatrix, [[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]])
        XCTAssertEqual(step.bWeights, [[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0]])
      }
      XCTAssertEqual(m.modelCallsTotal, 6, "\(trace.name)")
    }
  }

  func testRES2sStepsAreTwoRowExponential() throws {
    for tier in ["T1", "T2", "T3"] {
      let trace = try RES4LYFTraceFixture.load("res2s_beta6_\(tier)")
      for step in trace.manifest.steps {
        XCTAssertEqual(step.rkType, "res_2s", "\(trace.name) step \(step.index)")
        XCTAssertTrue(step.exponential)
        XCTAssertEqual(step.rows, 2)
        XCTAssertEqual(step.rowOffset, 1)
        XCTAssertEqual(step.modelCalls.count, 2)
        XCTAssertEqual(step.cNodes, [0.0, 0.5, 1.0])
        XCTAssertEqual(step.h, -log(step.sigmaNext / step.sigma), accuracy: 1e-9)
      }
      XCTAssertEqual(trace.manifest.modelCallsTotal, 12, "\(trace.name): 6 steps × 2 rows")
    }
  }

  // MARK: - The scripted denoiser is shared

  func testScriptedDenoiserReproducesEveryModelCall() throws {
    for trace in try loadAll() {
      for step in trace.manifest.steps {
        for call in step.modelCalls {
          let x = try trace.tensor(call.xIn)
          let want = try trace.tensor(call.denoised)
          let got = RES4LYFScriptedDenoiser.denoised(x, sigma: Float(call.sTmp))
          XCTAssertTraceClose(got, want, rtol: 1e-5, "\(trace.name) step \(step.index) row \(call.row)")
        }
      }
    }
  }

  /// RES4LYF's model call returns an epsilon anchored at x₀ and the step sigma
  /// (`noise_anchor = 1.0`): `denoised − x_0` in the exponential frame,
  /// `(x_0 − denoised)/σ` in the linear frame.
  func testRecordedEpsilonFollowsRES4LYFAnchoring() throws {
    for trace in try loadAll() {
      for step in trace.manifest.steps {
        for call in step.modelCalls {
          let x0 = try trace.tensor(call.x0)
          let denoised = try trace.tensor(call.denoised)
          let want = try trace.tensor(call.eps)
          let got = step.exponential ? denoised - x0 : (x0 - denoised) / Float(call.sigma)
          XCTAssertTraceClose(got, want, rtol: 1e-5, "\(trace.name) step \(step.index) row \(call.row)")
        }
      }
    }
  }

  // MARK: - The eta split and the injected noise

  func testHardModeVPSplitMatchesRecordedValues() throws {
    for trace in try loadAll() {
      let eta = trace.manifest.recipe.eta
      for step in trace.manifest.steps {
        let want = RES4LYFHardVPSplit.split(sigmaNext: step.sigmaNext, eta: eta, sigmaMax: trace.manifest.sigmaMax)
        XCTAssertEqual(step.sigmaUpEta, want.up, accuracy: 1e-12, "\(trace.name) step \(step.index) σ_up")
        XCTAssertEqual(step.sigmaDownEta, want.down, accuracy: 1e-12, "\(trace.name) step \(step.index) σ_down")
        XCTAssertEqual(step.alphaRatioEta, want.alpha, accuracy: 1e-12, "\(trace.name) step \(step.index) α")
        if eta == 0 {
          XCTAssertNil(step.noiseStep, "\(trace.name) step \(step.index): T1 injects no noise")
        }
        for sub in step.substeps {
          let w = RES4LYFHardVPSplit.split(
            sigmaNext: sub.subSigmaNext, eta: trace.manifest.recipe.etaSubstep, sigmaMax: trace.manifest.sigmaMax)
          // Rows whose substep target is the final row (row == rows − 1) are
          // not re-noised by RES4LYF; their split is recorded as the identity.
          if sub.row < step.rows - step.rowOffset {
            XCTAssertEqual(sub.subSigmaUpEta, w.up, accuracy: 1e-12, "\(trace.name) step \(step.index) row \(sub.row) sub σ_up")
            XCTAssertEqual(sub.subSigmaDownEta, w.down, accuracy: 1e-12, "\(trace.name) step \(step.index) row \(sub.row) sub σ_down")
            XCTAssertEqual(sub.subAlphaRatioEta, w.alpha, accuracy: 1e-12, "\(trace.name) step \(step.index) row \(sub.row) sub α")
          } else {
            XCTAssertEqual(sub.subSigmaUpEta, 0.0, "\(trace.name) step \(step.index) row \(sub.row): final row is not re-noised")
          }
        }
      }
    }
  }

  /// `swap_noise_step`: `eps_next = (x_0 − x_next)/(σ − σ')`,
  /// `denoised_next = x_0 − σ·eps_next`,
  /// `x = α·(denoised_next + σ_down·eps_next) + σ_up·noise·s_noise`.
  /// With the exported (z-scored) noise tensor the Swift side can reproduce
  /// the re-noise bit-for-purpose without torch's RNG.
  func testStepReNoiseIsReconstructibleFromExportedNoise() throws {
    var checked = 0
    for trace in try loadAll() where trace.manifest.recipe.eta > 0 {
      let sNoise = Float(trace.manifest.recipe.sNoise)
      for step in trace.manifest.steps {
        guard let noiseKey = step.noiseStep else {
          XCTAssertEqual(step.sigmaNext, 0, "\(trace.name) step \(step.index): only a 0 target skips the re-noise")
          continue
        }
        let x0 = try trace.tensor(step.x0)
        let xNext = try trace.tensor(step.xNext)
        let noise = try trace.tensor(noiseKey)
        let want = try trace.tensor(step.xOut)
        let sigma = Float(step.sigma), sigmaNext = Float(step.sigmaNext)
        let epsNext = (x0 - xNext) / (sigma - sigmaNext)
        let denoisedNext = x0 - sigma * epsNext
        let got = Float(step.alphaRatioEta) * (denoisedNext + Float(step.sigmaDownEta) * epsNext)
          + Float(step.sigmaUpEta) * noise * sNoise
        XCTAssertTraceClose(got, want, rtol: 1e-4, "\(trace.name) step \(step.index) re-noise")
        checked += 1

        // The same identity per substep, against the sub-split and sub-noise.
        for sub in step.substeps {
          guard let nKey = sub.noise, let preKey = sub.xPre, let postKey = sub.xPost,
            let subX0Key = sub.x0 else { continue }
          let x0 = try trace.tensor(subX0Key)
          let xPre = try trace.tensor(preKey)
          let subNoise = try trace.tensor(nKey)
          let subWant = try trace.tensor(postKey)
          let subNext = Float(sub.subSigmaNext)
          let e = (x0 - xPre) / (sigma - subNext)
          let d = x0 - sigma * e
          let g = Float(sub.subAlphaRatioEta) * (d + Float(sub.subSigmaDownEta) * e)
            + Float(sub.subSigmaUpEta) * subNoise * sNoise
          XCTAssertTraceClose(g, subWant, rtol: 1e-4, "\(trace.name) step \(step.index) row \(sub.row) sub re-noise")
          checked += 1
        }
      }
    }
    XCTAssertGreaterThan(checked, 0, "T2/T3 traces must carry re-noise events")
  }

  /// After the loop RES4LYF lands on σ_min and takes a model-free linear tail
  /// to 0: `x_final = x_out − σ_min · eps_last`, with `eps_last` the last
  /// step's `(x_0 − x_next)/(σ − σ_next)` computed *before* that step's re-noise.
  func testFinalLinearTailFromSigmaMin() throws {
    for trace in try loadAll() {
      let m = trace.manifest
      let last = try XCTUnwrap(m.steps.last)
      let x0 = try trace.tensor(last.x0)
      let xNext = try trace.tensor(last.xNext)
      let xOut = try trace.tensor(last.xOut)
      let epsLast = (x0 - xNext) / Float(last.sigma - last.sigmaNext)
      XCTAssertTraceClose(epsLast, try trace.tensor(m.final.epsLast), rtol: 1e-5, "\(trace.name) eps_last")
      let got = xOut - Float(m.sigmaMin) * epsLast
      XCTAssertTraceClose(got, try trace.tensor(m.final.x), rtol: 1e-5, "\(trace.name) final x")
    }
  }

  // MARK: - AC-26, T1, res_2s: the PRODUCTION scheduler against the oracle

  /// The published stage-1 recipe as `Krea2Pipeline.makeScheduler` builds it:
  /// `res_2s` + `beta` at 6 steps under Krea 2's `ModelSamplingFlux(1.15)`,
  /// with RES4LYF sigma preparation on. Nothing is read from the trace to
  /// construct it (Addendum A.1: the grid must be ours).
  static let tracedMu: Float = 1.15

  static func productionRES2sScheduler(steps: Int = 6) throws -> any ZImageScheduler {
    try SchedulerFactory.create(
      kind: .res2s,
      sigmaSchedule: .beta,
      numInferenceSteps: steps,
      config: Krea2Sampling.schedulerConfig(),
      mu: tracedMu,
      c2: 0.5,
      res4lyfSigmaPreparation: true)
  }

  /// The factory's own grid IS `sigmas_run` without its zero sentinel: `beta`
  /// at 6 steps over the Flux table, then `prepare_sigmas` inserting σ_min.
  func testProductionRES2sGridIsTheOraclePreparedGrid() throws {
    let m = try RES4LYFTraceFixture.load("res2s_beta6_T1").manifest
    let scheduler = try Self.productionRES2sScheduler()
    let grid = scheduler.sigmas.asArray(Float.self)

    XCTAssertEqual(scheduler.numInferenceSteps, m.steps.count, "6 solver steps, as upstream runs")
    XCTAssertEqual(grid.count, m.sigmasRun.count - 1, "the trailing 0 is the conversion's, not a step's")
    for (i, (g, w)) in zip(grid, m.sigmasRun).enumerated() {
      XCTAssertEqual(Double(g), w, accuracy: 1e-6, "sigmas_run[\(i)]")
    }
    XCTAssertEqual(Double(try XCTUnwrap(scheduler.finalConversionSigma)), m.sigmaMin, accuracy: 1e-9)
  }

  /// Drives the PRODUCTION `RES2sScheduler` — factory-built, prepared grid, no
  /// surgery here — with the trace's recorded data predictions, and checks the
  /// intermediate sample, the intermediate sigma and the step result against
  /// RES4LYF after every step.
  func testRES2sSchedulerMatchesT1TraceStepByStep() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T1")
    let m = trace.manifest
    var scheduler = try XCTUnwrap(try Self.productionRES2sScheduler() as? RES2sScheduler)
    XCTAssertEqual(scheduler.modelOutputConvention, .dataPrediction)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      let x0Recorded = try trace.tensor(step.x0)
      XCTAssertTraceClose(x, x0Recorded, rtol: 1e-4, "step \(step.index): x entering the step")

      let k1 = try trace.tensor(step.modelCalls[0].denoised)
      let intermediate = try XCTUnwrap(
        scheduler.intermediateStep(modelOutput: k1, timestepIndex: step.index, sample: x))
      XCTAssertTraceClose(
        intermediate, try trace.tensor(step.modelCalls[1].xIn), rtol: 1e-4,
        "step \(step.index): intermediate sample")
      XCTAssertEqual(
        scheduler.intermediateSigma(timestepIndex: step.index)!, Float(step.modelCalls[1].sTmp),
        accuracy: 1e-6, "step \(step.index): intermediate sigma")

      let k2 = try trace.tensor(step.modelCalls[1].denoised)
      x = scheduler.finalizeStep(
        originalOutput: k1, intermediateOutput: k2, timestepIndex: step.index, sample: x)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")
      XCTAssertTraceClose(x, try trace.tensor(step.xOut), rtol: 1e-4, "step \(step.index): T1 has no re-noise")
    }

    // Then RES4LYF's linear tail from σ_min to 0.
    let last = try XCTUnwrap(m.steps.last)
    let epsLast = (try trace.tensor(last.x0) - x) / Float(last.sigma - last.sigmaNext)
    let final = x - Float(m.sigmaMin) * epsLast
    XCTAssertTraceClose(final, try trace.tensor(m.final.x), rtol: 1e-4, "final x after the σ_min tail")
  }

  /// S-FIX-1's gate for `res_2s`: a FACTORY-created scheduler driven by the
  /// PRODUCTION `Krea2DenoiseLoop` against the scripted denoiser reproduces the
  /// trace's FINAL tensor — after the σ_min → 0 conversion, which the loop
  /// applies and this test does not. Fails on `final/x` if either the σ_min
  /// preparation or the model-free tail is missing from production.
  func testRES2sThroughTheDenoiseLoopReproducesTheTraceFinalTensor() throws {
    let trace = try RES4LYFTraceFixture.load("res2s_beta6_T1")
    let m = trace.manifest
    var scheduler = try Self.productionRES2sScheduler()

    let (x, stats) = try Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit)
    ) { latent, sigma in
      RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    XCTAssertEqual(stats.stepsRun, 6)
    XCTAssertEqual(stats.rowsAtStart, 2)
    XCTAssertEqual(stats.evaluateCalls, 12, "2 rows × 6 steps")
    XCTAssertEqual(stats.modelEvals, m.modelCallsTotal, "the σ_min conversion costs no model call")
    XCTAssertEqual(Double(try XCTUnwrap(stats.finalConversionSigma)), m.sigmaMin, accuracy: 1e-9)

    // `floor: 1.0` makes this an ABSOLUTE 1e-4 on a latent whose trajectory
    // runs at |x| ≈ 3. The final latent is the data prediction at σ_min, so its
    // own magnitude has collapsed to ~5e-3; measured relative to THAT, the
    // float32 round-off this driver accumulates — a flat ~6e-7 absolute at
    // every step, never growing — would read as 1e-4 "relative". The gate keeps
    // all of its power: without the σ_min preparation and the model-free tail
    // the same difference is 5.7e-2, 570× over this threshold.
    XCTAssertTraceClose(
      x, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }
}
