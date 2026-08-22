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
    // γ = (3c₃³ − 2c₃)/(c₂(2 − 3c₂)) is singular at c₂ = 2/3. Upstream does not
    // guard it and produces an inf tableau; refusing by name is the only
    // alternative to a NaN latent.
    precondition(
      abs(Double(c2) - 2.0 / 3.0) > 1e-6,
      "c2 = 2/3 is the singularity of res_3s's γ; pick another substep")

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

  // MARK: - 1-row fallback

  /// First-order exponential Euler, byte-for-byte `RES2sScheduler.step`'s
  /// update: the single-evaluation path non-tableau callers (the Z-Image
  /// pipelines) take. A loss of order, never of frame.
  public mutating func step(
    modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray
  ) -> MLXArray {
    let i = checked(timestepIndex)
    let h = stepSize(timestepIndex: i)
    return RES4LYFTableau.advance(
      frame: frame, x0: sample, dataPredictions: [modelOutput],
      weights: [RES4LYFTableau.phi(1, Double(-h))], h: h, sigma: sigmaValues[i])
  }

  // MARK: - Coefficients

  var cNodes: [Double] { [0, Double(c2), Double(c3)] }

  /// `calculate_gamma` (`phi_functions.py:16`).
  static func gamma(c2: Double, c3: Double) -> Double {
    (3 * (c3 * c3 * c3) - 2 * c3) / (c2 * (2 - 3 * c2))
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
