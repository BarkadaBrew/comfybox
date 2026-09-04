import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E13, AC-26 — `ralston_3s` against RES4LYF's own step trace, driven by
/// the PRODUCTION path.
///
/// The published stage-2 recipe is `deis_3m` at 2 steps, and RES4LYF's order
/// ramp (`rk_coefficients_beta.py:1343,1376`, `multistep_extra_initial_steps = 1`)
/// swaps `ralston_3s` in for `step < order + 1` — so **both** steps of the
/// stage run `ralston_3s` and E18's `deis3m_bong2_*` traces are, verbatim,
/// `ralston_3s` traces. That makes them this WP's parity gate, not E14's.
///
/// Addendum A.1 is honoured, and after S-FIX-1 it is honoured with **no grid
/// surgery in the test**: the scheduler comes from `SchedulerFactory` exactly
/// as `Krea2Pipeline.makeScheduler` builds it, `RES4LYFSigmaPreparation`
/// inserts the `ModelSamplingFlux` σ_min inside the factory, and
/// `Krea2DenoiseLoop` applies the model-free σ_min → 0 conversion. Nothing here
/// hand-builds `schedule.dropLast() + [σ_min]` and nothing here applies the
/// tail by hand — that concealment is precisely the defect this file now gates.
///
/// The stage's `denoise 0.2` is expressed the way the production loop expresses
/// it: the full `bong_tangent` grid at `total_steps = int(steps / denoise) = 10`
/// with `startIndex = 8`. RES4LYF's node slices the same grid instead
/// (`sigmas[-(steps+1):]`), so the two agree sigma for sigma.
final class RalstonTraceParityTests: XCTestCase {

  /// Krea 2's registered `ModelSamplingFlux(shift = 1.15)` — the model sampling
  /// the traces were exported under, and therefore the table whose first entry
  /// RES4LYF's `prepare_sigmas` inserts before the trailing zero (A.1).
  static let tracedMu: Float = 1.15

  static func fluxSigmaMin() -> Float {
    SigmaSchedule.fluxSigmaTable(shift: tracedMu, tableSize: Krea2Sampling.fluxTableSize)[0]
  }

  /// The stage-2 scheduler as production builds it: `SchedulerFactory` with
  /// Krea 2's config, the traced `mu`, and RES4LYF sigma preparation on.
  static func stage2Scheduler() throws -> (
    scheduler: any ZImageScheduler, startIndex: Int, totalSteps: Int, steps: Int
  ) {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let stage2 = try XCTUnwrap(
      fixture["stage2_bong_tangent_denoise"] as? [String: Any],
      "comfy_sigmas.json: stage2_bong_tangent_denoise")
    let steps = try XCTUnwrap(stage2["steps"] as? Int)
    let totalSteps = try XCTUnwrap(stage2["total_steps"] as? Int)

    let scheduler = try SchedulerFactory.create(
      kind: .ralston3s,
      sigmaSchedule: .bongTangent,
      numInferenceSteps: totalSteps,
      config: Krea2Sampling.schedulerConfig(),
      mu: tracedMu,
      res4lyfSigmaPreparation: true)
    // `denoise 0.2` is a partial start on the full grid, not a re-generated one.
    return (scheduler, totalSteps - steps, totalSteps, steps)
  }

  // MARK: - The grid is ours, and it lands on the oracle's

  func testStage2GridComesFromOurOwnProducersAndIsRES4LYFPrepared() throws {
    let (scheduler, startIndex, totalSteps, steps) = try Self.stage2Scheduler()
    XCTAssertEqual(steps, 2)
    XCTAssertEqual(totalSteps, 10, "int(steps / denoise) at 2 steps / denoise 0.2")
    XCTAssertEqual(startIndex, 8)

    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    let grid = scheduler.sigmas.asArray(Float.self)

    // Preparation inserted σ_min before the trailing zero and dropped the zero:
    // 11 published sigmas → 12 prepared → 11 solver sigmas, still 10 steps.
    XCTAssertEqual(grid.count, totalSteps + 1)
    XCTAssertEqual(scheduler.numInferenceSteps, totalSteps)
    XCTAssertEqual(Double(Self.fluxSigmaMin()), m.sigmaMin, accuracy: 1e-9, "ModelSamplingFlux σ_min")
    XCTAssertEqual(scheduler.finalConversionSigma, Self.fluxSigmaMin())

    // The stage's own window: the published sigmas the node would have sliced,
    // then σ_min in place of the zero — i.e. `sigmas_run` minus its sentinel.
    let window = Array(grid[startIndex...])
    XCTAssertEqual(window.count, m.sigmasRun.count - 1)
    for (i, (g, w)) in zip(window, m.sigmasRun).enumerated() {
      XCTAssertEqual(Double(g), w, accuracy: 1e-6, "sigmas_run[\(i)]")
    }
    // …and the published schedule those came from, unmoved above the tail.
    for (i, w) in m.sigmasSchedule.dropLast().enumerated() {
      XCTAssertEqual(Double(grid[startIndex + i]), w, accuracy: 1e-6, "sigmas_schedule[\(i)]")
    }
  }

  // MARK: - AC-26 T1: row by row against the oracle

  func testRalston3sMatchesTheT1TraceRowByRow() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    let (base, startIndex, _, _) = try Self.stage2Scheduler()
    var scheduler = try XCTUnwrap(base as? RalstonScheduler)
    XCTAssertEqual(scheduler.modelOutputConvention, .dataPrediction)
    XCTAssertEqual(scheduler.rows, 3)
    let grid = scheduler.sigmas.asArray(Float.self)

    var x = try trace.tensor(m.xInit)
    for step in m.steps {
      let i = startIndex + step.index
      XCTAssertTraceClose(x, try trace.tensor(step.x0), rtol: 1e-4, "step \(step.index): x entering")
      XCTAssertEqual(step.h, Double(grid[i + 1] - grid[i]), accuracy: 1e-6,
        "step \(step.index): linear-frame h = σ' − σ")

      // Every k is the DATA PREDICTION at that row — RES4LYF's `denoised`.
      var k: [MLXArray] = []
      for r in 0..<scheduler.rows {
        let call = step.modelCalls[r]
        let xr: MLXArray
        if r == 0 {
          xr = x
        } else {
          xr = scheduler.rowSample(timestepIndex: i, row: r, x0: x, k: k)
          XCTAssertEqual(
            Double(scheduler.rowSigma(timestepIndex: i, row: r)), call.sTmp,
            accuracy: 1e-6, "step \(step.index) row \(r): substep sigma")
        }
        XCTAssertTraceClose(
          xr, try trace.tensor(call.xIn), rtol: 1e-4, "step \(step.index) row \(r): row sample")
        k.append(try trace.tensor(call.denoised))
      }

      x = scheduler.commit(timestepIndex: i, x0: x, k: k)
      XCTAssertTraceClose(x, try trace.tensor(step.xNext), rtol: 1e-4, "step \(step.index): x_next")
      XCTAssertTraceClose(x, try trace.tensor(step.xOut), rtol: 1e-4, "step \(step.index): T1 re-noises nothing")
    }
  }

  /// The same trace through the production driver, with the scripted denoiser
  /// standing in for the model — nothing read from the fixture but the initial
  /// latent and the answer, and nothing done to the grid or to the tail by the
  /// test. This is the S-FIX-1 gate: it fails on `final/x` if the σ_min
  /// preparation or the model-free conversion is missing from production.
  ///
  /// Also pins the stage's cost at 6 model evaluations (FDD §3.12 / AC-24:
  /// `ralston_3s` × 2 steps × 3 rows) — the conversion adds none.
  func testRalston3sThroughTheDenoiseLoopReproducesTheTrace() throws {
    let trace = try RES4LYFTraceFixture.load("deis3m_bong2_T1")
    let m = trace.manifest
    let (base, startIndex, _, _) = try Self.stage2Scheduler()

    var scheduler = base
    let (x, stats) = try Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: try trace.tensor(m.xInit), startIndex: startIndex
    ) { latent, sigma in
      RES4LYFScriptedDenoiser.velocity(latent, sigma: sigma)
    }

    XCTAssertEqual(stats.stepsRun, 2)
    XCTAssertEqual(stats.rowsAtStart, 3)
    XCTAssertEqual(stats.evaluateCalls, 6, "3 rows × 2 steps")
    XCTAssertEqual(stats.modelEvals, 6, "no CFG in this harness; the σ_min tail is model-free")
    XCTAssertEqual(stats.evaluateCalls, m.modelCallsTotal, "the oracle made the same 6 calls")
    XCTAssertEqual(stats.finalConversionSigma, Self.fluxSigmaMin())

    // The final latent — AFTER RES4LYF's σ_min → 0 conversion, which the
    // production loop applies and the test does not. `floor: 1.0` makes this an
    // ABSOLUTE 1e-4: the final latent is the data prediction at σ_min and its
    // magnitude has collapsed, so a tolerance relative to it would measure
    // float32 ulps of the trajectory rather than agreement with the oracle.
    XCTAssertTraceClose(
      x, try trace.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x after the σ_min tail")
  }
}
