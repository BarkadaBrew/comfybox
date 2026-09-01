import Foundation
import MLX
import XCTest

@testable import ZImage

/// Implicit-RK refinement (RES4LYF `full_iter`) for `res_2s` — the EXPONENTIAL-
/// frame 2-row sampler re-iterated `implicitStepsFull` times as a fixed point,
/// against RES4LYF's own step trace, driven by the PRODUCTION path.
///
/// Sibling of ``RES4LYFImplicitTests`` (heun_2s), and it reuses that file's
/// `ImplicitTrace` / `Fixture` decoders. Ground truth is
/// `scripts/oracles/gen_implicit_fixture.py`
/// (`res4lyf_trace_{explicit,implicit}_res2s`), which runs upstream
/// `sample_rk_beta(rk_type="res_2s", implicit_steps_full=…, eta=0, bongmath off,
/// guides off)` on the scripted denoiser and exports, per step AND per full_iter
/// pass, every model call's `x_in` / `s_eval` / `denoised` / eps and the pass's
/// `x_next`.
///
/// The essential difference from heun_2s: heun is an N-row `TableauScheduler`,
/// so its `full_iter` loop lives in the driver's N-row branch. res_2s is NOT
/// N-row — it runs through the driver's 2-row branch
/// (`RES2sScheduler.intermediateStep` / `intermediateSigma` / `finalizeStep`),
/// which is where this test's `full_iter` re-iteration was added.
///
/// res_2s is exponential-frame (data-prediction, φ-functions): `h = −log(σ'/σ)`.
/// But the re-anchor needs no special exponential arithmetic in the driver —
/// with `noise_anchor = 1` upstream's `RK.__call__` returns the raw denoised
/// (`rk_method_beta.py:954-957`), so pass > 0 re-anchoring row 0 on the previous
/// pass's `x_next` at σ_next is the exact analogue of the linear heun case,
/// while every row's data prediction stays anchored at the step's x₀ and σ.
final class RES4LYFImplicitRes2sTests: XCTestCase {

  typealias ImplicitTrace = RES4LYFImplicitTests.ImplicitTrace
  typealias Fixture = RES4LYFImplicitTests.Fixture

  /// The `res_2s` scheduler on the fixture's own grid: `sigmas_run` minus the
  /// trailing sentinel zero (RES4LYF's prepared grid lands on σ_min), with the
  /// model-free σ_min → 0 conversion the driver applies after the last step.
  /// `c2 = 0.5` mirrors the fixture's `sample_rk_beta(c2=0.5)`.
  func makeScheduler(_ m: ImplicitTrace) -> RES2sScheduler {
    let grid = m.sigmasRun.dropLast().map { Float($0) }   // drop the trailing 0
    XCTAssertEqual(grid.count, m.steps.count + 1, "grid is steps+1 solver sigmas")
    return RES2sScheduler(
      numInferenceSteps: m.steps.count,
      sigmaValues: grid,
      c2: 0.5,
      finalConversionSigma: Float(m.sigmaMin))
  }

  // MARK: - implicitStepsFull = 1 reproduces the implicit trace, end to end

  func testRes2sImplicitThroughDenoiseLoopReproducesFinal() throws {
    let fx = try Fixture("implicit_res2s")
    let m = fx.manifest
    XCTAssertEqual(m.recipe.sampler, "res_2s")
    XCTAssertEqual(m.recipe.implicitStepsFull, 1)
    XCTAssertEqual(m.recipe.implicitStepsDiag, 0)

    var scheduler: any ZImageScheduler = makeScheduler(m)
    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler,
      initialSample: try fx.tensor(m.xInit),
      startIndex: 0,
      modelEvalsPerEvaluate: 1,
      implicitStepsFull: 1
    ) { latent, sigma in
      RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    XCTAssertEqual(stats.stepsRun, m.steps.count)
    XCTAssertEqual(stats.rowsAtStart, 2)
    // 4 steps × 2 passes × 2 rows = 16 model calls; the σ_min tail adds none.
    XCTAssertEqual(stats.evaluateCalls, 16, "4 steps × 2 full_iter passes × 2 rows")
    XCTAssertEqual(stats.finalConversionSigma, Float(m.sigmaMin))

    XCTAssertTraceClose(
      x, try fx.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }

  // MARK: - Pass-by-pass: the re-anchoring is exactly upstream's

  func testRes2sImplicitPassByPass() throws {
    let fx = try Fixture("implicit_res2s")
    let m = fx.manifest
    let base = makeScheduler(m)
    let grid = base.sigmas.asArray(Float.self)

    // Each step is anchored on the ORACLE's own x₀ (not a self-carried
    // trajectory), so the per-pass anchoring is checked independently of
    // cross-step float32 drift. The full carried trajectory is validated
    // end-to-end by `testRes2sImplicitThroughDenoiseLoopReproducesFinal`.
    for step in m.steps {
      let i = step.index
      let sigma = grid[i]
      let sigmaNext = grid[i + 1]
      XCTAssertEqual(step.h, -Foundation.log(Double(sigmaNext) / Double(sigma)), accuracy: 1e-5,
        "step \(i): exponential-frame h = −log(σ'/σ)")

      let x0 = try fx.tensor(step.fullIters[0].x0)  // the step's anchor (all passes)
      var committed = x0
      for pass in step.fullIters {
        // Feed the ORACLE's own data predictions (`denoised`) into the 2-row
        // scheduler API, so the check is OUR intermediateStep / intermediateSigma
        // / finalizeStep / re-anchoring against upstream — not a re-derivation of
        // the model output.
        //
        // rtol 1e-4 holds for EVERY row here, including the re-anchored pass>0
        // row 0 at the σ≈3e-4 schedule tail — tighter than the heun_2s port,
        // which needs 2e-4 there. The reason is the exponential frame: RES4LYF's
        // `__call__` forms `eps = eps_unmoored + noise_anchor·(eps_anchored −
        // eps_unmoored)` and, at noise_anchor 1, returns `denoised' = x₀ − σ·eps
        // = the raw denoised (rk_method_beta.py:954-957) — so the data
        // prediction is carried through directly. heun's linear frame instead
        // keeps `eps_unmoored = (x_tmp − denoised)/sub_sigma`, which divides by
        // the tiny tail σ and loses ~3e-5 to float32 cancellation; res_2s has no
        // such division in the anchored data it commits.
        var s = makeScheduler(m)

        // Row 0. Pass 0 samples the model at (x₀, σ); pass > 0 re-anchors on the
        // previous pass's committed x_next at σ_next. The DATA prediction stays
        // x₀/σ anchored either way (upstream's `sigma` arg).
        let call0 = pass.modelCalls[0]
        let row0Sample = pass.passIndex == 0 ? x0 : committed
        let row0Sigma = pass.passIndex == 0 ? sigma : sigmaNext
        XCTAssertEqual(Double(row0Sigma), call0.sEval, accuracy: 1e-6,
          "step \(i) pass \(pass.passIndex) row 0: model-eval sigma")
        XCTAssertEqual(call0.sigmaAnchor, Double(sigma), accuracy: 1e-6,
          "step \(i) pass \(pass.passIndex) row 0: data prediction stays x₀/σ anchored")
        XCTAssertTraceClose(row0Sample, try fx.tensor(call0.xIn), rtol: 1e-4,
          "step \(i) pass \(pass.passIndex) row 0: the sample the model saw")
        let out0 = try fx.tensor(call0.denoised)

        // Row 1: the exponential substep at σ·e^{−c₂h}, built from x₀ and row 0's
        // (re-anchored) data prediction — the substep sigma never moves.
        let call1 = pass.modelCalls[1]
        guard let mid = s.intermediateStep(modelOutput: out0, timestepIndex: i, sample: x0) else {
          XCTFail("step \(i): RES2sScheduler.intermediateStep returned nil")
          return
        }
        guard let midSigma = s.intermediateSigma(timestepIndex: i) else {
          XCTFail("step \(i): RES2sScheduler.intermediateSigma returned nil")
          return
        }
        XCTAssertEqual(Double(midSigma), call1.sEval, accuracy: 1e-6,
          "step \(i) pass \(pass.passIndex) row 1: substep model-eval sigma")
        XCTAssertEqual(call1.sigmaAnchor, Double(sigma), accuracy: 1e-6,
          "step \(i) pass \(pass.passIndex) row 1: data prediction stays x₀/σ anchored")
        XCTAssertTraceClose(mid, try fx.tensor(call1.xIn), rtol: 1e-4,
          "step \(i) pass \(pass.passIndex) row 1: the substep sample the model saw")
        let out1 = try fx.tensor(call1.denoised)

        committed = s.finalizeStep(
          originalOutput: out0, intermediateOutput: out1, timestepIndex: i, sample: x0)
        XCTAssertTraceClose(committed, try fx.tensor(pass.xNext), rtol: 1e-4,
          "step \(i) pass \(pass.passIndex): x_next")
      }
    }
  }

  // MARK: - implicitStepsFull = 0 reproduces the plain res_2s trace

  func testImplicitStepsFullZeroReproducesPlainRes2s() throws {
    let fx = try Fixture("explicit_res2s")
    let m = fx.manifest
    XCTAssertEqual(m.recipe.sampler, "res_2s")
    XCTAssertEqual(m.recipe.implicitStepsFull, 0)
    for step in m.steps {
      XCTAssertEqual(step.fullIters.count, 1, "explicit fixture is one pass/step")
    }

    var scheduler: any ZImageScheduler = makeScheduler(m)
    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler,
      initialSample: try fx.tensor(m.xInit),
      startIndex: 0,
      modelEvalsPerEvaluate: 1,
      implicitStepsFull: 0
    ) { latent, sigma in
      RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    // 4 steps × 1 pass × 2 rows = 8: the opt-in default takes no extra passes.
    XCTAssertEqual(stats.evaluateCalls, 8, "implicitStepsFull 0 ⇒ plain res_2s")
    XCTAssertTraceClose(
      x, try fx.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x, plain res_2s")

    // And the per-step substep + x_next match the oracle's single-pass trace,
    // anchored on the oracle's own x₀ each step (independent of cross-step drift).
    for step in m.steps {
      let i = step.index
      let x0 = try fx.tensor(step.fullIters[0].x0)
      var s = makeScheduler(m)
      let call0 = step.fullIters[0].modelCalls[0]
      let out0 = try fx.tensor(call0.denoised)
      guard let mid = s.intermediateStep(modelOutput: out0, timestepIndex: i, sample: x0) else {
        XCTFail("step \(i): intermediateStep returned nil")
        return
      }
      let call1 = step.fullIters[0].modelCalls[1]
      XCTAssertTraceClose(mid, try fx.tensor(call1.xIn), rtol: 1e-4,
        "step \(i) row 1: plain-explicit substep sample")
      let out1 = try fx.tensor(call1.denoised)
      let committed = s.finalizeStep(
        originalOutput: out0, intermediateOutput: out1, timestepIndex: i, sample: x0)
      XCTAssertTraceClose(committed, try fx.tensor(step.fullIters[0].xNext), rtol: 1e-4,
        "step \(i): plain-explicit x_next")
    }
  }
}
