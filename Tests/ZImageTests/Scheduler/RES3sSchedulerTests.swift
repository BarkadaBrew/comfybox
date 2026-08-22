import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E13 — `res_3s`, the 3-row exponential-frame conformer
/// (FDD-krea2-raw-recipe §3.12, D20, D23, AC-25).
///
/// `res_3s` is the programme's one engineering-owned cut (§6): E18 exported no
/// step trace for it, so the gate here is (a) the coefficient formulas,
/// transcribed verbatim from RES4LYF `beta/rk_coefficients_beta.py:1825` and
/// `beta/phi_functions.py` at the pinned commit
/// `26036f647ca15d3048a193daf99a40cecfc3820d`, pinned as numbers; (b) the
/// exponential-integrator consistency identities those formulas must satisfy;
/// (c) a closed-form ODE — a constant data prediction, which the frame
/// integrates exactly; and (d) the measured order of accuracy, 3, which AC-25
/// asks for and which `res_3s` — unlike the anchored linear ralstons — has.
///
/// The frame: `h = −log(σ'/σ)`, row sigmas `σ·e^{−cᵣh}`, `εⱼ = denoisedⱼ − x₀`
/// (RES4LYF's `noise_anchor = 1.0` exponential branch, verified against the
/// `res2s_beta6` traces), `x_row = x₀ + h·Σaᵢⱼεⱼ`. Because
/// `Σⱼaᵢⱼ = cᵢ·φ₁(−cᵢh)`, that is algebraically the textbook exponential-RK
/// update `x_row = e^{−cᵢh}·x₀ + h·Σaᵢⱼ·denoisedⱼ` — the anchoring is exact
/// here, which is why the exponential family keeps its order.
final class RES3sSchedulerTests: XCTestCase {

  static func grid(steps: Int, from: Double = 1.0, to: Double = 0.05) -> [Float] {
    (0...steps).map { Float(from * exp(log(to / from) * Double($0) / Double(steps))) }
  }

  // MARK: - φ functions

  /// `phi_functions.py`'s `φⱼ(z)`, pinned. Upstream defaults to the
  /// 80-digit mpmath series (`use_analytic_solution = True`), so these are the
  /// series values; the closed form it falls back to loses ~1e-12 near zero.
  func testPhiMatchesUpstream() {
    let pins: [(Double, [Double])] = [
      (-0.5, [0.7869386805747332, 0.4261226388505337, 0.1477547222989326]),
      (-0.05, [0.9754115099857197, 0.49176980028560363, 0.16460399428792727]),
      (-2.0, [0.43233235838169365, 0.2838338208091532, 0.1080830895954234]),
      (-18.0, [0.05555555470944557, 0.05246913584947524, 0.024862825786140266]),
    ]
    for (z, want) in pins {
      for j in 1...3 {
        XCTAssertEqual(RES4LYFTableau.phi(j, z), want[j - 1], accuracy: 1e-12, "φ\(j)(\(z))")
      }
    }
    // The removable singularity at 0: φⱼ(0) = 1/j!.
    for (j, want) in [(1, 1.0), (2, 0.5), (3, 1.0 / 6.0)] {
      XCTAssertEqual(RES4LYFTableau.phi(j, 0), want, accuracy: 1e-15, "φ\(j)(0)")
      XCTAssertEqual(RES4LYFTableau.phi(j, -1e-9), want, accuracy: 1e-9, "φ\(j)(−1e-9)")
    }
  }

  // MARK: - The tableau is RES4LYF's

  /// `γ = (3c₃³ − 2c₃)/(c₂(2 − 3c₂))`, `a₃₂ = γc₂φ₂(−c₂h) + (c₃²/c₂)φ₂(−c₃h)`,
  /// `b₃ = φ₂(−h)/(γc₂ + c₃)`, `b₂ = γb₃`, then `gen_first_col_exp` fills
  /// `aᵢ₁ = cᵢφ₁(−cᵢh) − Σⱼ₌₂aᵢⱼ` and `b₁ = φ₁(−h) − b₂ − b₃`.
  func testTableauMatchesUpstreamFormulas() {
    struct Pin {
      let h: Double, c2: Double, gamma: Double
      let a21: Double, a31: Double, a32: Double, b: [Double]
    }
    let pins = [
      Pin(
        h: 0.5, c2: 0.5, gamma: 4.0,
        a21: 0.44239843385719024, a31: -0.9869316554112899, a32: 1.773870335986023,
        b: [0.0767342824905104, 0.5681635184673782, 0.14204087961684456]),
      Pin(
        h: 0.25, c2: 1.0 / 3.0, gamma: 3.0,
        a21: 0.31982234148270705, a31: -0.984036426335601, a32: 1.8688332940499814,
        b: [-0.03682819057057529, 0.6912187937137169, 0.23040626457123894]),
      Pin(
        h: 0.031392210724247180, c2: 0.5, gamma: 4.0,
        a21: 0.49609642399255527, a31: -0.9999391118246704, a32: 1.9844059706577777,
        b: [0.1597855759857233, 0.6597450262779072, 0.1649362565694768]),
    ]
    for pin in pins {
      let t = RES3sScheduler.tableau(h: pin.h, c2: pin.c2, c3: 1.0)
      let label = "h=\(pin.h) c2=\(pin.c2)"
      XCTAssertEqual(RES3sScheduler.gamma(c2: pin.c2, c3: 1.0), pin.gamma, accuracy: 1e-12, label)
      XCTAssertEqual(t.a[0], [0, 0, 0], "\(label): row 0 is the start sample")
      XCTAssertEqual(t.a[1][0], pin.a21, accuracy: 1e-12, "\(label): a21")
      XCTAssertEqual(t.a[1][1], 0, "\(label): strictly lower-triangular")
      XCTAssertEqual(t.a[2][0], pin.a31, accuracy: 1e-12, "\(label): a31")
      XCTAssertEqual(t.a[2][1], pin.a32, accuracy: 1e-12, "\(label): a32")
      XCTAssertEqual(t.a[2][2], 0, "\(label): strictly lower-triangular")
      for j in 0..<3 {
        XCTAssertEqual(t.b[j], pin.b[j], accuracy: 1e-12, "\(label): b[\(j)]")
      }
    }
  }

  /// The exponential-frame consistency identities `gen_first_col_exp` enforces:
  /// `Σⱼaᵢⱼ = cᵢ·φ₁(−cᵢh)` and `Σⱼbⱼ = φ₁(−h)`. Equivalent to
  /// `x_row = e^{−cᵢh}x₀ + h·Σaᵢⱼ·denoisedⱼ`, i.e. the exact solution of the
  /// linear part — which is what keeps the x₀ anchoring exact in this frame.
  func testExponentialConsistencyIdentities() {
    for h in [0.01, 0.1, 0.5, 2.0, 8.0] {
      for c2 in [0.25, 0.5, 0.75] {
        let c = [0.0, c2, 1.0]
        let t = RES3sScheduler.tableau(h: h, c2: c2, c3: 1.0)
        for i in 0..<3 {
          let want = c[i] * (c[i] == 0 ? 0 : RES4LYFTableau.phi(1, -h * c[i]))
          XCTAssertEqual(t.a[i].reduce(0, +), want, accuracy: 1e-12, "h=\(h) c2=\(c2) row \(i)")
        }
        XCTAssertEqual(
          t.b.reduce(0, +), RES4LYFTableau.phi(1, -h), accuracy: 1e-12, "h=\(h) c2=\(c2): Σb")
      }
    }
  }

  /// As `h → 0` the exponential tableau collapses onto its classical
  /// counterpart — Kutta's third-order method, `c = [0, 1/2, 1]`,
  /// `a₂₁ = 1/2`, `a₃₁ = −1`, `a₃₂ = 2`, `b = [1/6, 2/3, 1/6]`.
  func testCollapsesToKuttaThirdOrderAsStepSizeVanishes() {
    let t = RES3sScheduler.tableau(h: 1e-7, c2: 0.5, c3: 1.0)
    XCTAssertEqual(t.a[1][0], 0.5, accuracy: 1e-7)
    XCTAssertEqual(t.a[2][0], -1.0, accuracy: 1e-6)
    XCTAssertEqual(t.a[2][1], 2.0, accuracy: 1e-6)
    XCTAssertEqual(t.b[0], 1.0 / 6.0, accuracy: 1e-7)
    XCTAssertEqual(t.b[1], 2.0 / 3.0, accuracy: 1e-7)
    XCTAssertEqual(t.b[2], 1.0 / 6.0, accuracy: 1e-7)
  }

  // MARK: - Closed form: a constant data prediction

  /// With `denoised ≡ D`, `dx/dτ = D − x` in `τ = −log(σ/σ₀)` has the exact
  /// solution `x(σ) = D + (σ/σ₀)(x₀ − D)`. An exponential integrator whose
  /// weights sum to `φ₁(−h)` reproduces it in one step, and over the whole
  /// grid — this is the closed-form gate `res_3s` gets in place of a trace.
  func testExactOnAConstantDataPrediction() {
    let dConst: Float = -0.35
    let grid = Self.grid(steps: 4)
    var scheduler: any ZImageScheduler = RES3sScheduler(
      numInferenceSteps: 4, sigmaValues: grid, c2: 0.5)
    let (out, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: MLXArray([Float(0.7)], [1])
    ) { latent, sigma in (latent - dConst) / sigma }
    let want = dConst + (grid.last! / grid.first!) * (0.7 - dConst)
    XCTAssertEqual(out.item(Float.self), want, accuracy: 3e-6)
    XCTAssertEqual(stats.evaluateCalls, 12, "3 rows × 4 steps")
    XCTAssertEqual(stats.rowsAtStart, 3)
  }

  // MARK: - AC-25: order of accuracy

  /// Third order on the scripted field, measured against a 200 000-step RK4
  /// reference — the exponential frame does not lose the order the anchored
  /// linear frame does (`ExplicitRKSchedulerTests` pins that contrast).
  func testThirdOrderOnTheSyntheticODE() {
    let reference = ExplicitRKSchedulerTests.referenceSolution()
    var errors: [Double] = []
    // Coarser than the ralston sweep on purpose: `res_3s` converges fast
    // enough that at 64 steps the terminal error (1.3e-7 on a ~0.5 sample) is
    // already the float32 round-off floor of the MLX path, and the measured
    // order flattens to ~2.4 for reasons that have nothing to do with the
    // solver. 4 → 32 keeps every error two decades above the floor.
    for n in [4, 8, 16, 32] {
      let scheduler = RES3sScheduler(numInferenceSteps: n, sigmaValues: Self.grid(steps: n), c2: 0.5)
      errors.append(abs(ExplicitRKSchedulerTests.runThroughDriver(scheduler) - reference))
    }
    // Minor 6: the whole Richardson sweep, not only the finest pair.
    let orders = ExplicitRKSchedulerTests.observedOrders(errors)
    let context = "res_3s orders \(orders) from errors \(errors)"
    for (i, order) in orders.enumerated() {
      XCTAssertGreaterThan(order, i == orders.count - 1 ? 2.85 : 2.5, "pair \(i): \(context)")
    }
    XCTAssertTrue(
      zip(errors, errors.dropFirst()).allSatisfy { $0 > $1 },
      "error must fall monotonically as steps double: \(context)")
  }

  // MARK: - D23: c2 is honoured

  func testHonoursC2() throws {
    let grid = Self.grid(steps: 3)
    let h = Double(-log(grid[1] / grid[0]))
    for c2 in [0.25, 0.5, 0.75] {
      let s = RES3sScheduler(numInferenceSteps: 3, sigmaValues: grid, c2: Float(c2))
      XCTAssertEqual(Double(s.c2), c2, accuracy: 1e-6)
      XCTAssertEqual(
        Double(s.rowSigma(timestepIndex: 0, row: 1)), Double(grid[0]) * exp(-c2 * h),
        accuracy: 1e-6, "row 1 sits at σ·e^{−c₂h} for c2=\(c2)")
      XCTAssertEqual(
        Double(s.rowSigma(timestepIndex: 0, row: 2)), Double(grid[1]),
        accuracy: 1e-6, "c3 = 1 puts row 2 on σ_next")
      // A different c2 is a different solver, not a cosmetic label.
      let t = RES3sScheduler.tableau(h: h, c2: c2, c3: 1.0)
      let base = RES3sScheduler.tableau(h: h, c2: 0.5, c3: 1.0)
      if c2 != 0.5 { XCTAssertNotEqual(t.b[1], base.b[1], accuracy: 1e-6) }
    }

    // …and it arrives from the factory (the same seam res_2s uses).
    let config = FlowMatchSchedulerTests.makeConfig()
    let fromFactory = try SchedulerFactory.create(
      kind: .res3s, numInferenceSteps: 9, config: config, c2: 0.3)
    XCTAssertEqual((fromFactory as? RES3sScheduler)?.c2, 0.3)
    let defaulted = try SchedulerFactory.create(kind: .res3s, numInferenceSteps: 9, config: config)
    XCTAssertEqual((defaulted as? RES3sScheduler)?.c2, 0.5, "D23: c2 stays 0.5 unless asked")
  }

  // MARK: - Protocol conformance and wiring

  func testConventionRowsAndFrame() {
    let s = RES3sScheduler(numInferenceSteps: 4, sigmaValues: Self.grid(steps: 4))
    XCTAssertEqual(s.modelOutputConvention, .dataPrediction)
    XCTAssertEqual(s.rows, 3)
    XCTAssertEqual(s.frame, .exponential)
    XCTAssertFalse(s.requiresIntermediateEvaluation)
    XCTAssertEqual(s.numInferenceSteps, 4)
    XCTAssertEqual(s.sigmas.dim(0), 5)
    XCTAssertEqual(s.c3, 1.0, "c3 is not on the wire; upstream's default is 1.0")
  }

  func testWireNameResolvesAndIsAdvertised() throws {
    XCTAssertEqual(SchedulerKind(rawValue: "res_3s"), .res3s)
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("res_3s"), .res3s)
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("exponential/res_3s"), .res3s)
    XCTAssertTrue(RecipeNameResolver.advertisedSamplerNames.contains("res_3s"))
    XCTAssertTrue(RecipeNameResolver.validSamplerNames.contains("res_3s"))
  }

  // MARK: - The two poles in (c₂, c₃) — the review's finding 2

  /// `res_3s` has two `h`-independent poles, and neither is guarded upstream:
  /// `γ`'s denominator `c₂(2 − 3c₂)` (at c₂ → 0 and c₂ = 2/3) and `b₃`'s
  /// `γc₂ + c₃`, which the DEFAULT `c₃ = 1` reaches at `c₂ = 1` (γ = −1) — a
  /// value the `c₂ ∈ (0,1]` range admitted and the old `c₂ = 2/3` check missed.
  ///
  /// The pole is demonstrated, not assumed: the raw tableau at those substeps
  /// is non-finite, which is the NaN latent the guard exists to prevent.
  func testRejectsBothPoles() {
    for (c2, c3, label) in [(2.0 / 3.0, 1.0, "γ"), (1.0, 1.0, "b₃")] {
      XCTAssertNotNil(
        RES3sScheduler.unsupportedSubstepReason(c2: c2, c3: c3),
        "c2=\(c2) c3=\(c3) is the \(label) pole and must be refused")
      let broken = RES3sScheduler.tableau(h: 0.5, c2: c2, c3: c3)
      let entries = broken.a.flatMap { $0 } + broken.b
      XCTAssertTrue(
        entries.contains { !$0.isFinite },
        "c2=\(c2) c3=\(c3) must actually produce a non-finite tableau, else the guard is "
          + "decoration: a=\(broken.a) b=\(broken.b)")
    }
    // The refusal names the offending value and which denominator vanished.
    let reason = RES3sScheduler.unsupportedSubstepReason(c2: 1.0, c3: 1.0) ?? ""
    XCTAssertTrue(reason.contains("1.0"), reason)
    XCTAssertTrue(reason.contains("b₃"), reason)
  }

  /// A substep close to — but not at — the `b₃` pole is legal and finite. The
  /// guard refuses a singularity, not a neighbourhood, and `res_3s` still
  /// integrates the constant-x₀ closed form exactly there.
  func testValidSubstepNearThePoleStillRuns() {
    let c2: Float = 0.95  // γ = −1.2384, γc₂ + c₃ = −0.1765
    XCTAssertNil(RES3sScheduler.unsupportedSubstepReason(c2: Double(c2), c3: 1.0))
    let table = RES3sScheduler.tableau(h: 0.5, c2: Double(c2), c3: 1.0)
    XCTAssertTrue((table.a.flatMap { $0 } + table.b).allSatisfy { $0.isFinite })

    let dConst: Float = -0.35
    let grid = Self.grid(steps: 4)
    var scheduler: any ZImageScheduler = RES3sScheduler(
      numInferenceSteps: 4, sigmaValues: grid, c2: c2)
    let (out, _) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: MLXArray([Float(0.7)], [1])
    ) { latent, sigma in (latent - dConst) / sigma }
    let want = dConst + (grid.last! / grid.first!) * (0.7 - dConst)
    XCTAssertEqual(out.item(Float.self), want, accuracy: 3e-6)
  }

  /// The default substeps are nowhere near either pole.
  func testDefaultSubstepsAreLegal() {
    XCTAssertNil(RES3sScheduler.unsupportedSubstepReason(c2: 0.5, c3: 1.0))
    for c2 in stride(from: 0.05, through: 0.6, by: 0.05) {
      XCTAssertNil(RES3sScheduler.unsupportedSubstepReason(c2: c2, c3: 1.0), "c2=\(c2)")
    }
  }

  /// `res_3s` is an N-row conformer: `step` is a hard failure, not a quiet
  /// first-order render (review finding 1). Asserted via the predicate the
  /// server gate and the CLI branch on, since a `preconditionFailure` cannot
  /// be caught in-process.
  func testIsAnNRowConformerWithNoOneRowFallback() throws {
    XCTAssertTrue(SchedulerKind.res3s.isNRowTableau)
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(kind: .res3s, numInferenceSteps: 9, config: config)
    XCTAssertNotNil(scheduler as? TableauScheduler)
    XCTAssertNotNil(
      GeneratePayload.validateTableauSampler(
        ResolvedRecipeNames(
          scheduler: .res3s, schedulerRequested: "res_3s",
          sigmaSchedule: nil, sigmaScheduleRequested: nil),
        family: .flux1),
      "res_3s must be refused on a family whose loop cannot drive it")
  }
}
