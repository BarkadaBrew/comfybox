import Foundation
import XCTest

@testable import ZImage

/// WP-E18 — the pinned ComfyUI / RES4LYF sigma and DEIS-coefficient fixtures
/// (FDD-krea2-raw-recipe §5.2).
///
/// These fixtures are the oracle the schedule work packages test against:
///   * `comfy_sigmas.json`        — `beta`/`beta57` at 6/9/30 steps, `bong_tangent`
///                                  at 2/6/8/9/10/12/20, dumped from the pinned
///                                  upstream sources by the generator script.
///   * `res4lyf_deis_coeffs.json` — `get_deis_coeff_list(…, deis_mode="rhoab")`
///                                  for orders 2/3/4 on fixed sigma arrays.
///
/// The tests here pin the fixture to the values the FDD quotes (AC-19, AC-21),
/// so that **a mismatch between the pinned values and the dump means the port
/// source moved and the FDD is stale** — §5.2's stated purpose. They do not
/// exercise any Swift schedule: that is WP-E11/E12/E14's job.
final class OracleFixtureTests: XCTestCase {

  // MARK: - Provenance

  func testFixturesCarryPinnedProvenance() throws {
    for name in ["comfy_sigmas.json", "res4lyf_deis_coeffs.json"] {
      let fixture = try SchedulerOracleFixtures.json(name)
      let prov = try XCTUnwrap(fixture["provenance"] as? [String: Any], "\(name): provenance")
      for key in ["generator", "comfyui_sha", "res4lyf_sha", "comfyui_url", "res4lyf_url", "generated_at", "torch"] {
        XCTAssertNotNil(prov[key], "\(name): provenance.\(key) missing")
      }
      let comfySha = try XCTUnwrap(prov["comfyui_sha"] as? String)
      let r4Sha = try XCTUnwrap(prov["res4lyf_sha"] as? String)
      XCTAssertEqual(comfySha.count, 40, "\(name): comfyui_sha must be a full commit sha")
      XCTAssertEqual(r4Sha.count, 40, "\(name): res4lyf_sha must be a full commit sha")
    }
  }

  // MARK: - AC-21 pins: ComfyUI `beta`

  /// The FDD's AC-21 values (`beta(6, shift: 1.15)`) were derived under
  /// `ModelSamplingDiscreteFlow(shift=1.15)`. The dump under that class must
  /// reproduce them to 1e-5 — that is the "FDD is stale" alarm.
  func testBetaDiscreteFlowMatchesFDDPins() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let beta = try XCTUnwrap(fixture["beta"] as? [String: Any])
    let df = try XCTUnwrap(beta["discrete_flow"] as? [String: Any], "beta.discrete_flow")
    let six = try SchedulerOracleFixtures.doubles(df["6"], "beta.discrete_flow.6")
    let pinned: [Double] = [1.0, 0.919919, 0.751973, 0.535879, 0.304782, 0.104360, 0.0]
    XCTAssertEqual(six.count, pinned.count)
    for (i, (got, want)) in zip(six, pinned).enumerated() {
      XCTAssertEqual(got, want, accuracy: 1e-5, "beta(6, shift 1.15) DiscreteFlow i=\(i)")
    }

    // beta57 = (alpha 0.5, beta 0.7): same class, same steps, must be present and distinct.
    let beta57 = try XCTUnwrap(fixture["beta57"] as? [String: Any])
    let df57 = try XCTUnwrap(beta57["discrete_flow"] as? [String: Any])
    let six57 = try SchedulerOracleFixtures.doubles(df57["6"], "beta57.discrete_flow.6")
    XCTAssertEqual(six57.count, 7)
    XCTAssertEqual(six57.first, 1.0)
    XCTAssertEqual(six57.last, 0.0)
    XCTAssertNotEqual(six57[1], six[1], "beta57 is a different PPF from beta")

    // 9 and 30 steps are present, start at the table top, end at the exact 0
    // sentinel, and are strictly decreasing (ComfyUI de-duplicates).
    for key in ["9", "30"] {
      for (label, table) in [("beta", df), ("beta57", df57)] {
        let sigmas = try SchedulerOracleFixtures.doubles(table[key], "\(label).discrete_flow.\(key)")
        XCTAssertLessThanOrEqual(sigmas.count, Int(key)! + 1, "\(label) \(key): count")
        XCTAssertEqual(sigmas.first, 1.0, "\(label) \(key): σ₀")
        XCTAssertEqual(sigmas.last, 0.0, "\(label) \(key): trailing 0")
        for i in 1..<sigmas.count {
          XCTAssertLessThan(sigmas[i], sigmas[i - 1], "\(label) \(key): not decreasing at \(i)")
        }
      }
    }
  }

  /// What ComfyUI **actually** registers Krea 2 as is `ModelType.FLUX` →
  /// `ModelSamplingFlux(shift=1.15)` — a 10 000-entry table whose `shift` is a
  /// log-shift (`exp(1.15)`), not `ModelSamplingDiscreteFlow`'s linear 1.15.
  /// The fixture records both and names the class, so no consumer can read the
  /// DiscreteFlow numbers as the grid the published workflow ran on.
  func testBetaUnderComfyKrea2RegistrationIsFluxAndDiffers() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    XCTAssertEqual(
      fixture["comfy_krea2_model_sampling"] as? String, "ModelSamplingFlux",
      "the fixture must name the class ComfyUI builds for Krea 2")

    let samplings = try XCTUnwrap(fixture["model_samplings"] as? [String: Any])
    let flux = try XCTUnwrap(samplings["flux"] as? [String: Any])
    let df = try XCTUnwrap(samplings["discrete_flow"] as? [String: Any])
    XCTAssertEqual(flux["table_size"] as? Int, 10000)
    XCTAssertEqual(df["table_size"] as? Int, 1000)
    XCTAssertEqual(flux["shift"] as? Double, 1.15)
    XCTAssertEqual(df["shift"] as? Double, 1.15)
    let fluxMin = try XCTUnwrap(flux["sigma_min"] as? Double)
    let dfMin = try XCTUnwrap(df["sigma_min"] as? Double)
    // Flux: σ(1e-4) = e^1.15 / (e^1.15 + 9999); DiscreteFlow: 1.15e-3 / (1 + 0.15e-3).
    XCTAssertEqual(fluxMin, exp(1.15) / (exp(1.15) + 9999.0), accuracy: 1e-9)
    XCTAssertEqual(dfMin, 1.15e-3 / (1.0 + 0.15e-3), accuracy: 1e-9)

    let beta = try XCTUnwrap(fixture["beta"] as? [String: Any])
    let fluxSix = try SchedulerOracleFixtures.doubles(
      (beta["flux"] as? [String: Any])?["6"], "beta.flux.6")
    let dfSix = try SchedulerOracleFixtures.doubles(
      (beta["discrete_flow"] as? [String: Any])?["6"], "beta.discrete_flow.6")
    XCTAssertEqual(fluxSix.count, 7)
    XCTAssertEqual(fluxSix.first, 1.0)
    XCTAssertEqual(fluxSix.last, 0.0)
    XCTAssertGreaterThan(abs(fluxSix[1] - dfSix[1]), 1e-2,
      "the two classes produce materially different grids; the FDD's D3/D5/AC-21 assume DiscreteFlow")
  }

  // MARK: - AC-19 pins: RES4LYF `bong_tangent`

  func testBongTangentMatchesFDDPins() throws {
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let bong = try XCTUnwrap(fixture["bong_tangent"] as? [String: Any])
    let six = try SchedulerOracleFixtures.doubles(bong["6"], "bong_tangent.6")
    let pinned: [Double] = [1.0, 0.928970, 0.797686, 0.5, 0.185601, 0.056802, 0.0]
    XCTAssertEqual(six.count, pinned.count)
    for (i, (got, want)) in zip(six, pinned).enumerated() {
      XCTAssertEqual(got, want, accuracy: 1e-6, "bong_tangent(6) i=\(i)")
    }

    for steps in [2, 6, 8, 9, 10, 12, 20] {
      let sigmas = try SchedulerOracleFixtures.doubles(bong["\(steps)"], "bong_tangent.\(steps)")
      XCTAssertEqual(sigmas.count, steps + 1, "bong_tangent(\(steps)): steps+1 elements")
      XCTAssertEqual(sigmas.first, 1.0, "bong_tangent(\(steps)): starts at 1.0")
      XCTAssertEqual(sigmas.last, 0.0, "bong_tangent(\(steps)): ends at 0.0")
      XCTAssertTrue(sigmas.contains { abs($0 - 0.5) < 1e-12 }, "bong_tangent(\(steps)): exactly 0.5 at the join")
      for i in 1..<sigmas.count {
        XCTAssertLessThan(sigmas[i], sigmas[i - 1], "bong_tangent(\(steps)): not decreasing at \(i)")
      }
    }

    // AC-20's fixture half: identical under both model-sampling classes — the
    // schedule never consults the model.
    XCTAssertEqual(fixture["bong_tangent_shift_free"] as? Bool, true)

    // The reference stage 2: `get_sigmas(model, "bong_tangent", steps=2, denoise=0.2)`
    // is the tail of bong_tangent(10), not bong_tangent(2) scaled.
    let stage2 = try XCTUnwrap(fixture["stage2_bong_tangent_denoise"] as? [String: Any])
    XCTAssertEqual(stage2["steps"] as? Int, 2)
    XCTAssertEqual(stage2["denoise"] as? Double, 0.2)
    XCTAssertEqual(stage2["total_steps"] as? Int, 10)
    let tail = try SchedulerOracleFixtures.doubles(stage2["sigmas"], "stage2 sigmas")
    let ten = try SchedulerOracleFixtures.doubles(bong["10"], "bong_tangent.10")
    XCTAssertEqual(tail, Array(ten.suffix(3)))
  }

  // MARK: - AC-23 fixture: RES4LYF `rhoab` DEIS coefficients

  func testDEISCoefficientFixtureShapeAndOrderRamp() throws {
    let fixture = try SchedulerOracleFixtures.json("res4lyf_deis_coeffs.json")
    XCTAssertEqual(fixture["deis_mode"] as? String, "rhoab")
    XCTAssertEqual(fixture["source"] as? String, "RES4LYF beta/deis_coefficients.py")
    let cases = try XCTUnwrap(fixture["cases"] as? [[String: Any]])
    var ordersSeen = Set<Int>()
    for c in cases {
      let name = try XCTUnwrap(c["name"] as? String)
      let order = try XCTUnwrap(c["max_order"] as? Int)
      ordersSeen.insert(order)
      let sigmas = try SchedulerOracleFixtures.doubles(c["sigmas"], "\(name).sigmas")
      let coeffs = try XCTUnwrap(c["coeffs"] as? [[Any]], "\(name).coeffs")
      XCTAssertEqual(coeffs.count, sigmas.count - 1, "\(name): one coefficient row per interval")
      for (i, row) in coeffs.enumerated() {
        // RES4LYF's corrected ramp: order = min(i+1, max_order); order 1 → [].
        let expectedOrder = min(i + 1, order)
        XCTAssertEqual(row.count, expectedOrder == 1 ? 0 : expectedOrder,
          "\(name): interval \(i) carries \(row.count) coefficients, ramp says \(expectedOrder)")
      }

      // Order-2 closed form, recomputed in Double from the fixture's own sigmas:
      //   coeff_cur   = ((t_next − t_prev)² − (t_cur − t_prev)²) / (2 (t_cur − t_prev))
      //   coeff_prev1 = (t_next − t_cur)² / (2 (t_prev − t_cur))
      for i in 1..<coeffs.count where min(i + 1, order) == 2 {
        let tPrev = sigmas[i - 1], tCur = sigmas[i], tNext = sigmas[i + 1]
        let cur = (pow(tNext - tPrev, 2) - pow(tCur - tPrev, 2)) / (2 * (tCur - tPrev))
        let prev1 = pow(tNext - tCur, 2) / (2 * (tPrev - tCur))
        let row = try SchedulerOracleFixtures.doubles(coeffs[i], "\(name).coeffs[\(i)]")
        XCTAssertEqual(row[0], cur, accuracy: 1e-12, "\(name): order-2 coeff_cur at \(i)")
        XCTAssertEqual(row[1], prev1, accuracy: 1e-12, "\(name): order-2 coeff_prev1 at \(i)")
      }
    }
    XCTAssertEqual(ordersSeen, [2, 3, 4], "orders 2/3/4 are all pinned")
  }
}
