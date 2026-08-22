// RES3sScheduler.swift — `res_3s`, the 3-row exponential-frame conformer
//
// WP-E13 (docs/FDD-krea2-raw-recipe.md §3.12, D20, D23). Transcribed from
// RES4LYF `beta/rk_coefficients_beta.py:1825` (`case "res_3s"`) and
// `beta/phi_functions.py` (`calculate_gamma`, `Phi.__call__`,
// `gen_first_col_exp`) at the pinned commit
// `26036f647ca15d3048a193daf99a40cecfc3820d`.
//
// `res_3s` is the programme's one engineering-owned cut (§6): E18 exported no
// step trace for it. Its gate is therefore the coefficient formulas pinned as
// numbers, the exponential-integrator consistency identities, the closed-form
// constant-x₀ solution the frame integrates exactly, and the measured order of
// accuracy (3 — see `RES3sSchedulerTests`).

import Foundation
import MLX

/// RES 3s: the 3-stage refined exponential solver, RES4LYF's `res_3s`.
///
/// Same frame as ``RES2sScheduler`` — `h = −log(σ'/σ)`, substeps at
/// `σ·e^{−cᵢh}`, data prediction in — with a third row and `φ₃` alongside
/// `φ₁`/`φ₂`. `c₂` is the request's substep parameter (default 0.5, D23);
/// `c₃` is upstream's, and is not on the wire.
///
/// The tableau is `h`-dependent, so it is recomputed per step (a handful of
/// `Double` operations) rather than cached at init.
public struct RES3sScheduler: TableauScheduler {

  /// Exponential frame: every `modelOutput` here is `x₀ = x − σ·v`.
  public let modelOutputConvention: ModelOutputConvention = .dataPrediction

  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int

  /// Second-row substep location in log-sigma space (`extra_options` `c2`
  /// upstream, the request's `c2` here). Upstream's default is 0.5.
  public let c2: Float
  /// Third-row substep location. Upstream's default is 1.0, which puts row 2
  /// exactly on `σ_next`. Not exposed on the wire (D23).
  public let c3: Float

  public let rows = 3
  public var frame: TableauFrame { .exponential }

  private let sigmaValues: [Float]

  /// - Parameters:
  ///   - numInferenceSteps: number of denoising steps.
  ///   - sigmaValues: `numInferenceSteps + 1` sigmas, monotonically decreasing.
  ///   - numTrainTimesteps: training timestep count for deriving `timesteps`.
  ///   - c2: second-row substep location. Must be in `(0, 1]` and not `2/3`,
  ///     where upstream's `γ` divides by zero.
  ///   - c3: third-row substep location.
  public init(
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000,
    c2: Float = 0.5,
    c3: Float = 1.0
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(
      sigmaValues.count == numInferenceSteps + 1,
      "sigmaValues must have numInferenceSteps + 1 elements")
    precondition(c2 > 0.0 && c2 <= 1.0, "c2 must be in (0, 1]")
    precondition(c3 > 0.0 && c3 <= 1.0, "c3 must be in (0, 1]")
    // Both of upstream's denominators, checked together. Neither is guarded in
    // RES4LYF; either one produces an inf tableau and a NaN latent.
    precondition(
      Self.unsupportedSubstepReason(c2: Double(c2), c3: Double(c3)) == nil,
      Self.unsupportedSubstepReason(c2: Double(c2), c3: Double(c3)) ?? "")

    self.c2 = c2
    self.c3 = c3
    self.sigmaValues = sigmaValues
    self.numInferenceSteps = numInferenceSteps
    self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
    let numTrainF = Float(numTrainTimesteps)
    let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }
    self.timesteps = MLXArray(timestepValues, [timestepValues.count])
  }

  // MARK: - N-row protocol

  public func rowSigma(timestepIndex: Int, row: Int) -> Float {
    precondition(row >= 0 && row < rows, "row \(row) outside 0..<\(rows)")
    let i = checked(timestepIndex)
    return RES4LYFTableau.rowSigma(
      frame: frame, sigma: sigmaValues[i], h: stepSize(timestepIndex: i),
      c: Float(cNodes[row]))
  }

  public mutating func rowSample(
    timestepIndex: Int, row: Int, x0: MLXArray, k: [MLXArray]
  ) -> MLXArray {
    precondition(row >= 1 && row < rows, "row 0 is the step's start sample")
    precondition(k.count == row, "row \(row) needs exactly \(row) prior outputs, got \(k.count)")
    let i = checked(timestepIndex)
    let h = stepSize(timestepIndex: i)
    let table = Self.tableau(h: Double(h), c2: Double(c2), c3: Double(c3))
    return RES4LYFTableau.advance(
      frame: frame, x0: x0, dataPredictions: k, weights: table.a[row], h: h,
      sigma: sigmaValues[i])
  }

  public mutating func commit(timestepIndex: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray {
    precondition(k.count == rows, "commit needs \(rows) outputs, got \(k.count)")
    let i = checked(timestepIndex)
    let h = stepSize(timestepIndex: i)
    let table = Self.tableau(h: Double(h), c2: Double(c2), c3: Double(c3))
    return RES4LYFTableau.advance(
      frame: frame, x0: x0, dataPredictions: k, weights: table.b, h: h, sigma: sigmaValues[i])
  }

  // MARK: - There is no 1-row fallback

  /// A `res_3s` step is three model evaluations. Reaching `step` means an
  /// N-row conformer was dispatched by a 1-row loop, which would render
  /// first-order exponential Euler under the name `res_3s` — a silent
  /// downgrade, so it fails hard. The reachable paths refuse the name first
  /// (`GeneratePayload.validateFamilyRecipe(_:family:)`, the CLI's
  /// `--scheduler`). See ``RalstonScheduler/step(modelOutput:timestepIndex:sample:)``.
  public mutating func step(
    modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray
  ) -> MLXArray {
    preconditionFailure(
      "res_3s is an N-row conformer driven by a 1-row loop: `step` would silently render "
        + "first-order exponential Euler under the name res_3s. Drive it through "
        + "Krea2DenoiseLoop (rowSigma / rowSample / commit), or refuse the sampler on this path.")
  }

  // MARK: - Coefficients

  var cNodes: [Double] { [0, Double(c2), Double(c3)] }

  /// `calculate_gamma` (`phi_functions.py:16`).
  static func gamma(c2: Double, c3: Double) -> Double {
    (3 * (c3 * c3 * c3) - 2 * c3) / (c2 * (2 - 3 * c2))
  }

  /// The two poles `res_3s`'s closed forms have in `(c₂, c₃)`, as a message —
  /// one source shared by the initializer's precondition and its tests.
  ///
  ///   * `γ = (3c₃³ − 2c₃)/(c₂(2 − 3c₂))` blows up as `c₂ → 0` or `c₂ → 2/3`;
  ///   * `b₃ = φ₂(−h)/(γc₂ + c₃)` blows up wherever `γc₂ + c₃ → 0`, which the
  ///     defaults reach at `c₂ = c₃ = 1` (γ = −1) — a value the old `c₂ ∈ (0,1]`
  ///     range admitted and the `c₂ = 2/3` check did not catch (WP-E13 review
  ///     finding 2).
  ///
  /// Both are `h`-independent, so one check at init covers every step.
  static func unsupportedSubstepReason(c2: Double, c3: Double) -> String? {
    let epsilon = 1e-6
    let gammaDenominator = c2 * (2 - 3 * c2)
    if abs(gammaDenominator) < epsilon {
      return "res_3s substep c2 = \(c2) is a pole of γ = (3c₃³ − 2c₃)/(c₂(2 − 3c₂)) "
        + "(c₂(2 − 3c₂) = \(gammaDenominator)); pick a c2 away from 0 and 2/3"
    }
    let g = (3 * (c3 * c3 * c3) - 2 * c3) / gammaDenominator
    let b3Denominator = g * c2 + c3
    if abs(b3Denominator) < epsilon {
      return "res_3s substeps c2 = \(c2), c3 = \(c3) are a pole of b₃ = φ₂(−h)/(γc₂ + c₃) "
        + "(γ = \(g), γc₂ + c₃ = \(b3Denominator)); pick another c2"
    }
    return nil
  }

  /// `case "res_3s"` (`rk_coefficients_beta.py:1825`) followed by
  /// `gen_first_col_exp` (`:3228`), in `Double`.
  ///
  /// `φ(j, i)` upstream is `φⱼ(−h·cᵢ₋₁)`, and returns 0 whenever that `c` is 0
  /// — which is why row 0 stays all-zero rather than becoming `0·φ₁(0)`.
  static func tableau(h: Double, c2: Double, c3: Double) -> (a: [[Double]], b: [Double]) {
    let c = [0.0, c2, c3]
    // φ(j) with no index is φⱼ(−h); φ(j, i) is φⱼ(−h·c[i−1]).
    func phi(_ j: Int) -> Double { RES4LYFTableau.phi(j, -h) }
    func phi(_ j: Int, _ i: Int) -> Double {
      let ci = c[i - 1]
      if ci == 0 { return 0 }
      return RES4LYFTableau.phi(j, -h * ci)
    }

    let g = gamma(c2: c2, c3: c3)
    let a32 = g * c2 * phi(2, 2) + (c3 * c3 / c2) * phi(2, 3)
    let b3 = (1 / (g * c2 + c3)) * phi(2)
    let b2 = g * b3

    var a: [[Double]] = [[0, 0, 0], [0, 0, 0], [0, a32, 0]]
    var b: [Double] = [0, b2, b3]
    // gen_first_col_exp: aᵢ₁ = cᵢ·φ₁(−cᵢh) − Σⱼ aᵢⱼ, b₁ = φ₁(−h) − Σⱼ bⱼ.
    for i in 0..<3 {
      a[i][0] = c[i] * phi(1, i + 1) - a[i].reduce(0, +)
    }
    b[0] = phi(1) - b.reduce(0, +)
    return (a, b)
  }

  // MARK: - Helpers

  private func stepSize(timestepIndex: Int) -> Float {
    let i = checked(timestepIndex)
    return RES4LYFTableau.stepSize(
      frame: frame, sigma: sigmaValues[i], sigmaNext: sigmaValues[i + 1])
  }

  private func checked(_ timestepIndex: Int) -> Int {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmaValues.count,
      "invalid timestep index \(timestepIndex)")
    return timestepIndex
  }
}
