import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E13 — the N-row tableau conformers `ralston_2s` / `ralston_3s` /
/// `ralston_4s` (FDD-krea2-raw-recipe §3.12, D1, D20, AC-25).
///
/// Weight-free and GPU-free: the model is a closed-form data-prediction field,
/// the reference is a Double RK4 integration of the same ODE at 200 000 steps.
///
/// Two things are pinned separately, and the distinction is the point:
///
///   1. **The tableaus are the RES4LYF ones.** Transcribed verbatim from
///      `beta/rk_coefficients_beta.py:1207` (`ralston_4s`), `:1241`
///      (`ralston_3s`), `:1294` (`ralston_2s`) at the pinned commit
///      `26036f647ca15d3048a193daf99a40cecfc3820d`, cross-checked for
///      `ralston_3s` against the `a_matrix` / `b_weights` / `c_nodes` recorded
///      in E18's `deis3m_bong2` step traces. Driven as a *classical* explicit
///      RK (`kⱼ = f(xⱼ, σⱼ)`) they reach classical order 2 / 3 / 4, which is
///      what proves the coefficients are transcribed correctly.
///
///   2. **The shipped conformer is RES4LYF-exact, and RES4LYF anchors.**
///      RES4LYF's linear-frame row derivative is `εⱼ = (x₀ − denoisedⱼ)/σ`
///      — the *step's* `x₀` and the *step's* `σ`, not the row's own sample and
///      substep sigma (`noise_anchor = 1.0`; verified against the trace
///      fixture, where the row-local form is off by 0.17 at row 1). That
///      anchoring costs the family its classical order: every `ralston_Ns`
///      is empirically **second order**, 3s and 4s included. AC-25 asks for
///      p≥3 / p≥4 here; the oracle says otherwise and the oracle wins (the
///      brief's "exactness over convenience"). Both facts are asserted below
///      so the discrepancy can never be mistaken for a port bug.
final class ExplicitRKSchedulerTests: XCTestCase {

  // MARK: - The scripted field (shared with the RES4LYF trace harness)

  /// `denoised(x, σ) = 0.5·tanh(x) + 0.25·σ − 0.1·x` — FDD §5.2's denoiser,
  /// in Double, so the reference solution and the samplers see one field.
  static func denoised(_ x: Double, _ sigma: Double) -> Double {
    0.5 * Foundation.tanh(x) + 0.25 * sigma - 0.1 * x
  }

  /// The probability-flow ODE the samplers integrate: `dx/dσ = (x − D)/σ`.
  static func derivative(_ x: Double, _ sigma: Double) -> Double {
    (x - denoised(x, sigma)) / sigma
  }

  static let x0: Double = 0.7
  static let sigmaStart: Double = 1.0
  static let sigmaEnd: Double = 0.05

  /// Classical RK4 at 200 000 steps — the closed-form-quality reference.
  static func referenceSolution() -> Double {
    let n = 200_000
    var x = x0
    let h = (sigmaEnd - sigmaStart) / Double(n)
    for i in 0..<n {
      let s = sigmaStart + Double(i) * h
      let k1 = derivative(x, s)
      let k2 = derivative(x + h / 2 * k1, s + h / 2)
      let k3 = derivative(x + h / 2 * k2, s + h / 2)
      let k4 = derivative(x + h * k3, s + h)
      x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    }
    return x
  }

  static func linearGrid(steps: Int) -> [Float] {
    (0...steps).map { Float(sigmaStart + (sigmaEnd - sigmaStart) * Double($0) / Double(steps)) }
  }

  /// Run a conformer through the production driver with the scripted field.
  static func runThroughDriver(_ scheduler: any ZImageScheduler) throws -> Double {
    var s = scheduler
    let (out, _) = try Krea2DenoiseLoop.run(
      scheduler: &s,
      initialSample: MLXArray([Float(x0)], [1])
    ) { latent, sigma in
      // The driver hands the model a velocity and converts per WP-E2.
      let x = latent.item(Float.self)
      let d = Float(denoised(Double(x), Double(sigma)))
      return MLXArray([(x - d) / sigma], [1])
    }
    return Double(out.item(Float.self))
  }

  /// Empirical order from a terminal-error sequence at doubling step counts.
  static func observedOrders(_ errors: [Double]) -> [Double] {
    (0..<(errors.count - 1)).map { log2(errors[$0] / errors[$0 + 1]) }
  }

  // MARK: - 1. The tableaus are RES4LYF's

  func testRalstonTableausMatchUpstreamSource() {
    let r5 = 5.0.squareRoot()

    let two = RalstonScheduler(stages: .two, numInferenceSteps: 4, sigmaValues: Self.linearGrid(steps: 4))
    XCTAssertEqual(two.cNodes, [0, 2.0 / 3.0])
    XCTAssertEqual(two.aMatrix, [[0, 0], [2.0 / 3.0, 0]])
    XCTAssertEqual(two.bWeights, [1.0 / 4.0, 3.0 / 4.0])

    let three = RalstonScheduler(stages: .three, numInferenceSteps: 4, sigmaValues: Self.linearGrid(steps: 4))
    XCTAssertEqual(three.cNodes, [0, 1.0 / 2.0, 3.0 / 4.0])
    XCTAssertEqual(three.aMatrix, [[0, 0, 0], [1.0 / 2.0, 0, 0], [0, 3.0 / 4.0, 0]])
    XCTAssertEqual(three.bWeights, [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0])

    let four = RalstonScheduler(stages: .four, numInferenceSteps: 4, sigmaValues: Self.linearGrid(steps: 4))
    XCTAssertEqual(four.cNodes, [0, 2.0 / 5.0, (14 - 3 * r5) / 16, 1])
    XCTAssertEqual(
      four.aMatrix,
      [
        [0, 0, 0, 0],
        [2.0 / 5.0, 0, 0, 0],
        [(-2889 + 1428 * r5) / 1024, (3785 - 1620 * r5) / 1024, 0, 0],
        [(-3365 + 2094 * r5) / 6040, (-975 - 3046 * r5) / 2552, (467040 + 203968 * r5) / 240845, 0],
      ])
    XCTAssertEqual(
      four.bWeights,
      [
        (263 + 24 * r5) / 1812, (125 - 1000 * r5) / 3828,
        (3426304 + 1661952 * r5) / 5924787, (30 - 4 * r5) / 123,
      ])
  }

  /// `ralston_3s` cross-checked against the tableau RES4LYF actually ran, as
  /// recorded in E18's `deis_3m` warm-up traces — a second, independent source
  /// for the same numbers (AC-24 pins that both steps run `ralston_3s`).
  func testRalston3sMatchesTheTableauRecordedInTheTrace() throws {
    let scheduler = RalstonScheduler(
      stages: .three, numInferenceSteps: 4, sigmaValues: Self.linearGrid(steps: 4))
    for tier in ["T1", "T2", "T3"] {
      let trace = try RES4LYFTraceFixture.load("deis3m_bong2_\(tier)")
      for step in trace.manifest.steps {
        XCTAssertEqual(step.rkType, "ralston_3s")
        XCTAssertEqual(step.aMatrix, scheduler.aMatrix, "\(trace.name) step \(step.index): a")
        XCTAssertEqual(step.bWeights[0], scheduler.bWeights, "\(trace.name) step \(step.index): b")
        // RES4LYF appends the final node 1 to the exported c list.
        XCTAssertEqual(
          Array(step.cNodes.dropLast()), scheduler.cNodes, "\(trace.name) step \(step.index): c")
      }
    }
  }

  /// Butcher order conditions, evaluated on the transcribed coefficients:
  /// row sums `Σⱼaᵢⱼ = cᵢ` (all), `Σb = 1` (order 1), `Σbc = 1/2` (order 2),
  /// `Σbc² = 1/3` and `Σᵢbᵢ(Σⱼaᵢⱼcⱼ) = 1/6` (order 3), `Σbc³ = 1/4` (order 4).
  func testRalstonTableausSatisfyTheirOrderConditions() {
    for (stages, order) in [(RalstonScheduler.Stages.two, 2), (.three, 3), (.four, 4)] {
      let s = RalstonScheduler(stages: stages, numInferenceSteps: 4, sigmaValues: Self.linearGrid(steps: 4))
      let a = s.aMatrix, b = s.bWeights, c = s.cNodes
      let label = "ralston_\(stages.rawValue)s"
      for i in 0..<s.rows {
        XCTAssertEqual(a[i].reduce(0, +), c[i], accuracy: 1e-12, "\(label): row \(i) sum ≠ c[\(i)]")
        for j in i..<s.rows {
          XCTAssertEqual(a[i][j], 0, "\(label): a[\(i)][\(j)] must be strictly lower-triangular")
        }
      }
      XCTAssertEqual(b.reduce(0, +), 1.0, accuracy: 1e-12, "\(label): Σb")
      if order >= 2 {
        XCTAssertEqual(zip(b, c).map(*).reduce(0, +), 0.5, accuracy: 1e-12, "\(label): Σbc")
      }
      if order >= 3 {
        XCTAssertEqual(
          zip(b, c).map { $0 * $1 * $1 }.reduce(0, +), 1.0 / 3.0, accuracy: 1e-12, "\(label): Σbc²")
        let bac = (0..<s.rows).map { i in b[i] * (0..<s.rows).map { a[i][$0] * c[$0] }.reduce(0, +) }
        XCTAssertEqual(bac.reduce(0, +), 1.0 / 6.0, accuracy: 1e-12, "\(label): Σbac")
      }
      if order >= 4 {
        XCTAssertEqual(
          zip(b, c).map { $0 * $1 * $1 * $1 }.reduce(0, +), 0.25, accuracy: 1e-12, "\(label): Σbc³")
      }
    }
  }

  /// Driven as a *classical* RK — row-local `kⱼ = f(xⱼ, σⱼ)` — the transcribed
  /// tableaus reach order 2 / 3 / 4. This is the coefficient-correctness gate;
  /// it deliberately does NOT go through the shipped conformer, which anchors.
  func testTableauCoefficientsAchieveClassicalOrder() {
    let reference = Self.referenceSolution()
    for (stages, expected) in [(RalstonScheduler.Stages.two, 2.0), (.three, 3.0), (.four, 4.0)] {
      var errors: [Double] = []
      for n in [8, 16, 32, 64] {
        let grid = Self.linearGrid(steps: n).map(Double.init)
        let s = RalstonScheduler(
          stages: stages, numInferenceSteps: n, sigmaValues: Self.linearGrid(steps: n))
        let a = s.aMatrix, b = s.bWeights, c = s.cNodes
        var x = Self.x0
        for i in 0..<n {
          let sigma = grid[i]
          let h = grid[i + 1] - sigma
          var k: [Double] = []
          for r in 0..<s.rows {
            let xr = x + h * (0..<r).map { a[r][$0] * k[$0] }.reduce(0, +)
            k.append(Self.derivative(xr, sigma + c[r] * h))
          }
          x += h * (0..<s.rows).map { b[$0] * k[$0] }.reduce(0, +)
        }
        errors.append(abs(x - reference))
      }
      // Minor 6: the WHOLE Richardson sweep, not just the finest pair —
      // a single good ratio can hide a non-monotone error curve. The band
      // widens at the coarse end because the asymptotic regime is not reached
      // there; the finest pair must be within 0.15 of the nominal order.
      let orders = Self.observedOrders(errors)
      let context = "ralston_\(stages.rawValue)s classical orders \(orders) from errors \(errors)"
      for (i, order) in orders.enumerated() {
        let floorAt = i == orders.count - 1 ? expected - 0.15 : expected - 0.75
        XCTAssertGreaterThan(order, floorAt, "pair \(i): \(context)")
      }
      XCTAssertTrue(
        zip(errors, errors.dropFirst()).allSatisfy { $0 > $1 },
        "error must fall monotonically as steps double: \(context)")
    }
  }

  // MARK: - 2. The shipped conformer is RES4LYF-exact, and therefore 2nd order

  /// AC-25, measured. RES4LYF's `x₀`-anchored epsilon perturbs every row
  /// derivative by `O(h²)`, so the local error is `O(h³)` and the global order
  /// is 2 for the whole family — `ralston_3s` and `ralston_4s` included. The
  /// assertion is two-sided on purpose: ≥2 proves the solver is consistent and
  /// correctly wired, <2.6 proves we did NOT quietly "fix" the anchoring into
  /// a classical RK and diverge from the oracle.
  func testAnchoredLinearFrameIsSecondOrderForEveryRalston() throws {
    let reference = Self.referenceSolution()
    for stages in [RalstonScheduler.Stages.two, .three, .four] {
      var errors: [Double] = []
      for n in [8, 16, 32, 64] {
        let scheduler = RalstonScheduler(
          stages: stages, numInferenceSteps: n, sigmaValues: Self.linearGrid(steps: n))
        errors.append(abs(try Self.runThroughDriver(scheduler) - reference))
      }
      // Minor 6: every pair in the sweep, not only the finest.
      let orders = Self.observedOrders(errors)
      let context = "ralston_\(stages.rawValue)s orders \(orders) from errors \(errors)"
      for (i, order) in orders.enumerated() {
        let floorAt = i == orders.count - 1 ? 1.85 : 1.5
        XCTAssertGreaterThan(order, floorAt, "pair \(i): \(context)")
        XCTAssertLessThan(
          order, 2.6,
          "pair \(i) reached order \(order): the RES4LYF x₀ anchoring is gone — that is a "
            + "divergence from the oracle, not an improvement. \(context)")
      }
      XCTAssertTrue(
        zip(errors, errors.dropFirst()).allSatisfy { $0 > $1 },
        "error must fall monotonically as steps double: \(context)")
    }
  }

  /// The one case the anchoring is exact for: a constant data prediction.
  /// `x(σ) = D + (σ/σ₀)(x₀ − D)` is reproduced to float32 precision in a
  /// single step by every ralston, which is why the anchoring is harmless on
  /// a well-behaved rectified-flow trajectory.
  func testExactOnAConstantDataPrediction() throws {
    let dConst: Float = -0.35
    for stages in [RalstonScheduler.Stages.two, .three, .four] {
      var scheduler: any ZImageScheduler = RalstonScheduler(
        stages: stages, numInferenceSteps: 3, sigmaValues: [1.0, 0.6, 0.25, 0.05])
      let (out, stats) = try Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: MLXArray([Float(0.7)], [1])
      ) { latent, sigma in (latent - dConst) / sigma }
      let want = dConst + Float(0.05 / 1.0) * (0.7 - dConst)
      XCTAssertEqual(out.item(Float.self), want, accuracy: 2e-6, "ralston_\(stages.rawValue)s")
      XCTAssertEqual(stats.evaluateCalls, 3 * stages.rawValue, "ralston_\(stages.rawValue)s evals")
      XCTAssertEqual(stats.rowsAtStart, stages.rawValue)
    }
  }

  // MARK: - Protocol conformance and wiring

  func testConventionRowsAndFrame() {
    for stages in [RalstonScheduler.Stages.two, .three, .four] {
      let s = RalstonScheduler(stages: stages, numInferenceSteps: 4, sigmaValues: Self.linearGrid(steps: 4))
      XCTAssertEqual(s.modelOutputConvention, .dataPrediction, "ralston_\(stages.rawValue)s")
      XCTAssertEqual(s.rows, stages.rawValue)
      XCTAssertEqual(s.frame, .linear)
      XCTAssertFalse(s.requiresIntermediateEvaluation, "the N-row protocol is not the 2-row one")
      XCTAssertEqual(s.numInferenceSteps, 4)
      XCTAssertEqual(s.sigmas.dim(0), 5)
    }
  }

  /// Row sigmas are `σ + cᵣ·h` on the linear frame, and row 0 is the grid sigma.
  func testRowSigmasAreTheLinearSubsteps() {
    let grid: [Float] = [1.0, 0.6, 0.25, 0.05]
    let s = RalstonScheduler(stages: .three, numInferenceSteps: 3, sigmaValues: grid)
    for i in 0..<3 {
      let h = grid[i + 1] - grid[i]
      for r in 0..<3 {
        XCTAssertEqual(
          s.rowSigma(timestepIndex: i, row: r), grid[i] + Float(s.cNodes[r]) * h, accuracy: 1e-6,
          "step \(i) row \(r)")
      }
      XCTAssertEqual(s.rowSigma(timestepIndex: i, row: 0), grid[i], accuracy: 1e-7)
    }
  }

  /// Every wire name resolves, is advertised, and builds its conformer.
  func testWireNamesResolveAndAreAdvertised() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let expected: [(String, SchedulerKind, Int)] = [
      ("ralston_2s", .ralston2s, 2), ("ralston_3s", .ralston3s, 3), ("ralston_4s", .ralston4s, 4),
    ]
    for (name, kind, rows) in expected {
      XCTAssertEqual(SchedulerKind(rawValue: name), kind)
      XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind(name), kind)
      // RES4LYF's UI groups these under `linear/`; a workflow value pastes verbatim.
      XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("linear/\(name)"), kind)
      XCTAssertTrue(
        RecipeNameResolver.advertisedSamplerNames.contains(name),
        "\(name) is accepted but not advertised — the E4 reconciliation AC")
      XCTAssertTrue(RecipeNameResolver.validSamplerNames.contains(name))
      let scheduler = try SchedulerFactory.create(kind: kind, numInferenceSteps: 9, config: config)
      let tableau = try XCTUnwrap(scheduler as? TableauScheduler, name)
      XCTAssertEqual(tableau.rows, rows, name)
      XCTAssertEqual(scheduler.modelOutputConvention, .dataPrediction, name)
    }
  }

  // MARK: - No 1-row fallback: the review's finding 1

  /// `SchedulerKind.isNRowTableau` is the one predicate the CLI and the server
  /// family gate branch on, so it must agree with what the factory actually
  /// builds. Checked in both directions: every kind that claims to be a
  /// tableau builds a `TableauScheduler` with `rows ≥ 2`, and no kind that
  /// denies it builds one.
  ///
  /// `step` on those conformers is a `preconditionFailure` (it would render
  /// first-order Euler under the tableau's name), so this predicate is the
  /// only thing standing between a wire name and a crash — it cannot drift.
  func testIsNRowTableauMatchesWhatTheFactoryBuilds() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    var claimed: Set<SchedulerKind> = []
    for kind in SchedulerKind.allCases {
      let scheduler = try SchedulerFactory.create(
        kind: kind, numInferenceSteps: 9, config: config, seed: 1)
      let tableau = scheduler as? TableauScheduler
      XCTAssertEqual(
        kind.isNRowTableau, tableau != nil,
        "\(kind.rawValue): isNRowTableau = \(kind.isNRowTableau) but conformance says otherwise")
      if let tableau {
        XCTAssertGreaterThanOrEqual(tableau.rows, 2, kind.rawValue)
        claimed.insert(kind)
      }
    }
    XCTAssertEqual(
      claimed, [.ralston2s, .ralston3s, .ralston4s, .res3s, .deis2m, .deis3m, .deis4m])
  }

  /// The server's family gate: a tableau sampler is a 400 on every family but
  /// `krea2`. Accepted-and-advertised as a NAME is unchanged (E4); what is
  /// family-scoped is whether the loop can honour it.
  ///
  /// K-FIX-1 / I5: the gate is now the family capability matrix, of which this
  /// is one row — so a NON-tableau sampler is no longer universally accepted
  /// either; it passes exactly on the families that drive it.
  func testTableauSamplersAreRefusedOffKrea2() throws {
    for kind in SchedulerKind.allCases {
      for family in WarmModelFamily.allCases {
        let names = ResolvedRecipeNames(
          scheduler: kind, schedulerRequested: kind.rawValue,
          sigmaSchedule: nil, sigmaScheduleRequested: nil)
        let error = GeneratePayload.validateFamilyRecipe(names, family: family)
        let familyRunsIt = FamilyRecipeMatrix.capability(for: family).samplers.contains(kind)
        if kind.isNRowTableau && family != .krea2 {
          let rejection = try XCTUnwrap(
            error, "\(kind.rawValue) on \(family.rawValue) must be refused")
          let message = try XCTUnwrap(rejection.errorDescription)
          XCTAssertTrue(message.contains(kind.rawValue), message)
          XCTAssertTrue(message.contains(family.rawValue), message)
          XCTAssertTrue(message.contains("Krea 2"), message)
        } else {
          XCTAssertEqual(
            error == nil, familyRunsIt,
            "\(kind.rawValue) on \(family.rawValue): \(String(describing: error))")
        }
      }
    }
    // A request with no sampler at all is never gated.
    XCTAssertNil(
      GeneratePayload.validateFamilyRecipe(
        ResolvedRecipeNames(
          scheduler: nil, schedulerRequested: nil, sigmaSchedule: nil, sigmaScheduleRequested: nil),
        family: .flux1))
  }

  /// The refusal names what the caller can use instead, and never offers a
  /// tableau sampler as the alternative.
  func testRefusalListsUsableSamplers() throws {
    let names = ResolvedRecipeNames(
      scheduler: .ralston4s, schedulerRequested: "linear/ralston_4s",
      sigmaSchedule: nil, sigmaScheduleRequested: nil)
    let error = try XCTUnwrap(GeneratePayload.validateFamilyRecipe(names, family: .flux1))
    let message = try XCTUnwrap(error.errorDescription)
    // The raw string the caller sent is echoed, prefix and all.
    XCTAssertTrue(message.contains("linear/ralston_4s"), message)
    for kind in SchedulerKind.allCases where !kind.isNRowTableau {
      XCTAssertTrue(message.contains(kind.rawValue), "must offer \(kind.rawValue): \(message)")
    }
  }
}
