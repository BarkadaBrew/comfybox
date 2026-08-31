import Foundation
import MLX
import XCTest

@testable import ZImage

/// Implicit-RK refinement (RES4LYF `full_iter`) — `heun_2s` re-iterated
/// `implicitStepsFull` times as a fixed point, against RES4LYF's own step
/// trace, driven by the PRODUCTION path.
///
/// Ground truth is `scripts/oracles/gen_implicit_fixture.py`, which runs
/// upstream `sample_rk_beta(rk_type="heun_2s", implicit_steps_full=…, eta=0,
/// bongmath off, guides off)` on the scripted denoiser and exports, per step
/// AND per full_iter pass, every model call's `x_in` / `s_eval` / `denoised` /
/// eps and the pass's `x_next`. Two fixtures:
///   * `explicit_heun2s`  — `implicit_steps_full = 0` (one pass; the plain
///                          explicit baseline the default path must reproduce).
///   * `implicit_heun2s`  — `implicit_steps_full = 1` (two passes).
///
/// The scoped regime (heun_2s, row_offset 1, eta 0, guides/bongmath off) makes
/// the whole refinement observable in one delta: pass > 0 re-anchors row 0 on
/// the previous pass's committed `x_next` evaluated at σ_next, while every
/// row's epsilon stays anchored at the step's x₀ and σ.
final class RES4LYFImplicitTests: XCTestCase {

  // MARK: - Fixture

  /// The implicit/explicit trace manifest — a leaner shape than the T1/T2/T3
  /// `RES4LYFTrace`, because this generator captures per-pass, not per-substep.
  struct ImplicitTrace: Decodable {
    struct Recipe: Decodable {
      let sampler: String
      let steps: Int
      let implicitStepsFull: Int
      let implicitStepsDiag: Int
    }
    struct Call: Decodable {
      let row: Int
      let sEval: Double
      let sigmaAnchor: Double
      let xIn: String
      let x0: String
      let denoised: String
      let eps: String
    }
    struct Pass: Decodable {
      let passIndex: Int
      let modelCalls: [Call]
      let x0: String
      let xNext: String
    }
    struct Step: Decodable {
      let index: Int
      let sigma: Double
      let sigmaNext: Double
      let h: Double
      let fullIters: [Pass]
    }
    struct Final: Decodable {
      let x: String
      let epsLast: String
    }
    let recipe: Recipe
    let sigmasSchedule: [Double]
    let sigmasRun: [Double]
    let sigmaMin: Double
    let sigmaMax: Double
    let xInit: String
    let steps: [Step]
    let final: Final
  }

  final class Fixture {
    let manifest: ImplicitTrace
    let tensors: [String: MLXArray]
    init(_ name: String) throws {
      let jsonURL = SchedulerOracleFixtures.url("res4lyf_trace_\(name).json")
      let stURL = SchedulerOracleFixtures.url("res4lyf_trace_\(name).safetensors")
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      manifest = try decoder.decode(ImplicitTrace.self, from: Data(contentsOf: jsonURL))
      tensors = try MLX.loadArrays(url: stURL)
    }
    func tensor(_ key: String) throws -> MLXArray {
      guard let t = tensors[key] else {
        throw NSError(domain: "ImplicitFixture", code: 1,
          userInfo: [NSLocalizedDescriptionKey: "missing tensor \(key)"])
      }
      return t
    }
  }

  /// The `heun_2s` scheduler on the fixture's own grid: `sigmas_run` minus the
  /// trailing sentinel zero (RES4LYF's prepared grid lands on σ_min), with the
  /// model-free σ_min → 0 conversion the driver applies after the last step.
  func makeScheduler(_ m: ImplicitTrace) throws -> RES4LYFHeunScheduler {
    let grid = m.sigmasRun.dropLast().map { Float($0) }   // drop the trailing 0
    XCTAssertEqual(grid.count, m.steps.count + 1, "grid is steps+1 solver sigmas")
    return RES4LYFHeunScheduler(
      stages: .two,
      numInferenceSteps: m.steps.count,
      sigmaValues: grid,
      finalConversionSigma: Float(m.sigmaMin))
  }

  // MARK: - implicitStepsFull = 1 reproduces the implicit trace, end to end

  func testHeun2sImplicitThroughDenoiseLoopReproducesFinal() throws {
    let fx = try Fixture("implicit_heun2s")
    let m = fx.manifest
    XCTAssertEqual(m.recipe.sampler, "heun_2s")
    XCTAssertEqual(m.recipe.implicitStepsFull, 1)
    XCTAssertEqual(m.recipe.implicitStepsDiag, 0)

    var scheduler: any ZImageScheduler = try makeScheduler(m)
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

  func testHeun2sImplicitPassByPass() throws {
    let fx = try Fixture("implicit_heun2s")
    let m = fx.manifest
    let scheduler = try makeScheduler(m)
    let grid = scheduler.sigmas.asArray(Float.self)

    // Each step is anchored on the ORACLE's own x₀ (not a self-carried
    // trajectory), so the per-pass anchoring is checked independently of
    // cross-step float32 drift. The full carried trajectory is validated
    // end-to-end by `testHeun2sImplicitThroughDenoiseLoopReproducesFinal`.
    for step in m.steps {
      let i = step.index
      let sigma = grid[i]
      let sigmaNext = grid[i + 1]
      XCTAssertEqual(step.h, Double(sigmaNext - sigma), accuracy: 1e-6,
        "step \(i): linear-frame h = σ' − σ")

      let x0 = try fx.tensor(step.fullIters[0].x0)  // the step's anchor (all passes)
      var committed = x0
      for pass in step.fullIters {
        // Feed the ORACLE's own data predictions as `k` (RalstonTraceParityTests
        // methodology): the check is then OUR rowSample / rowSigma / commit /
        // anchoring against upstream, not a re-derivation of the model output —
        // which the σ≈3e-4 schedule tail would otherwise amplify past float32.
        var k: [MLXArray] = []
        for (r, call) in pass.modelCalls.enumerated() {
          let xr: MLXArray
          let sr: Float
          if r == 0 {
            if pass.passIndex == 0 {
              xr = x0
              sr = sigma
            } else {
              // The fully-implicit re-anchor: previous pass's x_next at σ_next.
              xr = committed
              sr = sigmaNext
            }
          } else {
            var s = scheduler
            xr = s.rowSample(timestepIndex: i, row: r, x0: x0, k: k)
            sr = s.rowSigma(timestepIndex: i, row: r)
          }
          XCTAssertEqual(Double(sr), call.sEval, accuracy: 1e-6,
            "step \(i) pass \(pass.passIndex) row \(r): model-eval sigma")
          XCTAssertEqual(call.sigmaAnchor, Double(sigma), accuracy: 1e-6,
            "step \(i) pass \(pass.passIndex) row \(r): eps stays x₀/σ anchored")
          // rtol 2e-4 (vs 1e-4 elsewhere) for exactly one intermediate: the
          // re-anchored row at the σ≈3e-4 schedule tail. RES4LYF forms the
          // linear eps as `eps_unmoored + noise_anchor·(eps_anchor −
          // eps_unmoored)` (rk_method_beta.py:1100); at the re-anchored pass>0
          // row, `eps_unmoored = (x_tmp − denoised)/sub_sigma` divides by the
          // tiny tail σ and blows up, so `unmoored + (anchor − unmoored)`
          // loses ~3e-5 to float32 cancellation. The driver (like the existing
          // port for every tableau — RES4LYFTableau.epsilon) uses the
          // algebraically-equal clean anchor `(x₀ − denoised)/σ`, so this one
          // sample sits ~1.2e-5 (≈1.1e-4 rel) off the oracle's cancellation
          // artifact. The end-to-end final still holds at 1e-4 (below).
          XCTAssertTraceClose(xr, try fx.tensor(call.xIn), rtol: 2e-4,
            "step \(i) pass \(pass.passIndex) row \(r): the sample the model saw")
          k.append(try fx.tensor(call.denoised))
        }
        var s = scheduler
        committed = s.commit(timestepIndex: i, x0: x0, k: k)
        XCTAssertTraceClose(committed, try fx.tensor(pass.xNext), rtol: 2e-4,
          "step \(i) pass \(pass.passIndex): x_next (tail cancellation, see above)")
      }
    }
  }

  // MARK: - implicitStepsFull = 0 reproduces the plain explicit trace

  func testImplicitStepsFullZeroReproducesPlainExplicit() throws {
    let fx = try Fixture("explicit_heun2s")
    let m = fx.manifest
    XCTAssertEqual(m.recipe.implicitStepsFull, 0)
    for step in m.steps {
      XCTAssertEqual(step.fullIters.count, 1, "explicit fixture is one pass/step")
    }

    var scheduler: any ZImageScheduler = try makeScheduler(m)
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
    XCTAssertEqual(stats.evaluateCalls, 8, "implicitStepsFull 0 ⇒ plain explicit")
    XCTAssertTraceClose(
      x, try fx.tensor(m.final.x), rtol: 1e-4, floor: 1.0, "final x, plain explicit")

    // And the per-step x_next matches the oracle's single-pass trace, anchored
    // on the oracle's own x₀ each step (independent of cross-step drift).
    let grid = scheduler.sigmas.asArray(Float.self)
    for step in m.steps {
      let i = step.index
      let xr0 = try fx.tensor(step.fullIters[0].x0)
      var s = try makeScheduler(m)
      var k: [MLXArray] = []
      for (r, call) in step.fullIters[0].modelCalls.enumerated() {
        let sample = r == 0 ? xr0 : s.rowSample(timestepIndex: i, row: r, x0: xr0, k: k)
        let sr = r == 0 ? grid[i] : s.rowSigma(timestepIndex: i, row: r)
        XCTAssertEqual(Double(sr), call.sEval, accuracy: 1e-6, "step \(i) row \(r): sigma")
        XCTAssertTraceClose(sample, try fx.tensor(call.xIn), rtol: 1e-4,
          "step \(i) row \(r): row sample")
        k.append(try fx.tensor(call.denoised))
      }
      let committed = s.commit(timestepIndex: i, x0: xr0, k: k)
      XCTAssertTraceClose(committed, try fx.tensor(step.fullIters[0].xNext), rtol: 1e-4,
        "step \(i): plain-explicit x_next")
    }
  }
}
