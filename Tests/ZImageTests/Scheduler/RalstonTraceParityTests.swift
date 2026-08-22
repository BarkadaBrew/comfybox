import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E13, AC-26 — `ralston_3s` against RES4LYF's own step trace.
///
/// The published stage-2 recipe is `deis_3m` at 2 steps, and RES4LYF's order
/// ramp (`rk_coefficients_beta.py:1343,1376`, `multistep_extra_initial_steps = 1`)
/// swaps `ralston_3s` in for `step < order + 1` — so **both** steps of the
/// stage run `ralston_3s` and E18's `deis3m_bong2_*` traces are, verbatim,
/// `ralston_3s` traces. That makes them this WP's parity gate, not E14's.
///
/// Addendum A.1 is honoured: the sigma grid is built by **our** producers
/// (`bong_tangent` at the denoise-expanded step count, plus the `ModelSamplingFlux`
/// σ_min tail) and only then checked against the fixture. The trace's own
/// `sigmas_run` is never fed back into the sampler.
final class RalstonTraceParityTests: XCTestCase {

  /// Krea 2's registered `ModelSamplingFlux(shift = 1.15)` σ_min — the value
  /// RES4LYF's `prepare_sigmas` inserts before the trailing zero (A.1).
  static func fluxSigmaMin() -> Float {
    SigmaSchedule.fluxSigmaTable(shift: 1.15, tableSize: 10000)[0]
  }

  /// The stage-2 grid, from our own schedule code: `bong_tangent` at
  /// `total_steps = int(steps / denoise)`, tail-sliced to `steps + 1`, then
  /// σ_min appended in place of the trailing zero.
  static func stage2Grid() throws -> (schedule: [Float], loop: [Float], totalSteps: Int, steps: Int) {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let stage2 = try XCTUnwrap(
      fixture["stage2_bong_tangent_denoise"] as? [String: Any],
      "comfy_sigmas.json: stage2_bong_tangent_denoise")
    let steps = try XCTUnwrap(stage2["steps"] as? Int)
    let totalSteps = try XCTUnwrap(stage2["total_steps"] as? Int)

    let full = try SchedulerFactory.resolveSigmas(
      schedule: .bongTangent, numSteps: totalSteps,
      config: Krea2Sampling.schedulerConfig(), mu: nil)
    let schedule = Array(full.suffix(steps + 1))
    let loop = schedule.dropLast() + [fluxSigmaMin()]
    return (schedule, Array(loop), totalSteps, steps)
  }

  // MARK: - The grid is ours, and it lands on the oracle's

  func testStage2GridComesFromOurOwnProducers() throws {
    let (schedule, loop, totalSteps, steps) = try Self.stage2Grid()
    XCTAssertEqual(steps, 2)
    XCTAssertEqual(totalSteps, 10, "int(steps / denoise) at 2 steps / denoise 0.2")

    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    XCTAssertEqual(schedule.count, m.sigmasSchedule.count)
    for (i, (g, w)) in zip(schedule, m.sigmasSchedule).enumerated() {
      XCTAssertEqual(Double(g), w, accuracy: 1e-6, "sigmas_schedule[\(i)]")
    }
    XCTAssertEqual(Double(Self.fluxSigmaMin()), m.sigmaMin, accuracy: 1e-9, "ModelSamplingFlux σ_min")
    // The loop grid is `sigmas_run` without its trailing zero: the σ_min → 0
    // tail is RES4LYF's model-free linear step, not a sampler step.
    XCTAssertEqual(loop.count, m.sigmasRun.count - 1)
    for (i, (g, w)) in zip(loop, m.sigmasRun).enumerated() {
      XCTAssertEqual(Double(g), w, accuracy: 1e-6, "sigmas_run[\(i)]")
    }
  }

  // MARK: - AC-26 T1: row by row against the oracle

  func testRalston3sMatchesTheT1TraceRowByRow() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    let (_, grid, _, _) = try Self.stage2Grid()
    XCTAssertEqual(grid.count, m.steps.count + 1)

    var scheduler = RalstonScheduler(
      stages: .three, numInferenceSteps: m.steps.count, sigmaValues: grid)
    XCTAssertEqual(scheduler.modelOutputConvention, .dataPrediction)
    XCTAssertEqual(scheduler.rows, 3)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      XCTAssertTraceClose(x, try trace.tensor(step.x0), rtol: 1e-4, "step \(step.index): x entering")
      XCTAssertEqual(step.h, Double(grid[step.index + 1] - grid[step.index]), accuracy: 1e-6,
        "step \(step.index): linear-frame h = σ' − σ")

      // Every k is the DATA PREDICTION at that row — RES4LYF's `denoised`.
      var k: [MLXArray] = []
      for r in 0..<scheduler.rows {
        let call = step.modelCalls[r]
        let xr: MLXArray
        if r == 0 {
          xr = x
        } else {
          xr = scheduler.rowSample(timestepIndex: step.index, row: r, x0: x, k: k)
          XCTAssertEqual(
            Double(scheduler.rowSigma(timestepIndex: step.index, row: r)), call.sTmp,
            accuracy: 1e-6, "step \(step.index) row \(r): substep sigma")
        }
        XCTAssertTraceClose(
          xr, try trace.tensor(call.xIn), rtol: 1e-4, "step \(step.index) row \(r): row sample")
        k.append(try trace.tensor(call.denoised))
      }

      x = scheduler.commit(timestepIndex: step.index, x0: x, k: k)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")
      XCTAssertTraceClose(x, try trace.tensor(step.xOut), rtol: 1e-4, "step \(step.index): T1 re-noises nothing")
    }

    // RES4LYF's model-free linear tail from σ_min to 0.
    let last = try XCTUnwrap(m.steps.last)
    let epsLast = (try trace.tensor(last.x0) - x) / Float(last.sigma - last.sigmaNext)
    let final = x - Float(m.sigmaMin) * epsLast
    XCTAssertTraceClose(final, try trace.tensor(m.final.x), rtol: 1e-4, "final x after the σ_min tail")
  }

  /// The same trace through the production driver, with the scripted denoiser
  /// standing in for the model — nothing read from the fixture but the initial
  /// latent and the answer. Also pins the stage's cost at 6 model evaluations
  /// (FDD §3.12 / AC-24: `ralston_3s` × 2 steps × 3 rows).
  func testRalston3sThroughTheDenoiseLoopReproducesTheTrace() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    let (_, grid, _, _) = try Self.stage2Grid()

    var scheduler: any ZImageScheduler = RalstonScheduler(
      stages: .three, numInferenceSteps: m.steps.count, sigmaValues: grid)
    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit)
    ) { latent, sigma in
      RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    XCTAssertEqual(stats.stepsRun, 2)
    XCTAssertEqual(stats.rowsAtStart, 3)
    XCTAssertEqual(stats.evaluateCalls, 6, "3 rows × 2 steps")
    XCTAssertEqual(stats.modelEvals, 6, "no CFG in this harness")
    XCTAssertEqual(stats.evaluateCalls, m.modelCallsTotal, "the oracle made the same 6 calls")

    let last = try XCTUnwrap(m.steps.last)
    XCTAssertTraceClose(x, try trace.tensor(last.xOut), rtol: 1e-4, "driver x after the last step")
  }
}
