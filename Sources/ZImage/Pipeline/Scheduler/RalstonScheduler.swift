// RalstonScheduler.swift — `ralston_2s` / `ralston_3s` / `ralston_4s`
//
// WP-E13 (docs/FDD-krea2-raw-recipe.md §3.12, D1, D20). The linear-frame
// N-row conformers, coefficients copied verbatim from RES4LYF
// `beta/rk_coefficients_beta.py` at the pinned commit
// `26036f647ca15d3048a193daf99a40cecfc3820d`:
//   `ralston_4s` :1207, `ralston_3s` :1241, `ralston_2s` :1294.
//
// These are also the DEIS warm-up samplers: RES4LYF's order ramp (:1343,
// :1376, `multistep_extra_initial_steps = 1`) swaps `ralston_{order}s` in
// while `step < order + 1`, so the published stage-2 recipe (`deis_3m`,
// 2 steps) is *entirely* `ralston_3s` — 6 model evaluations, true DEIS
// coefficients never engaging (AC-24). WP-E14 builds on this type.

import Foundation
import MLX

/// Ralston's explicit Runge–Kutta methods, in RES4LYF's linear frame.
///
/// **Takes the data prediction, not the velocity.** RES4LYF anchors every
/// row's derivative at the step's start sample and the step's sigma —
/// `εⱼ = (x₀ − denoisedⱼ)/σ` — so the conformer needs `denoisedⱼ`, and the
/// row sample it built is not enough to recover it. Callers convert through
/// ``ZImageScheduler/modelInput(velocity:sample:sigma:)``; the E3 driver does
/// it per row. See `RES4LYFTableau.swift` for why the anchoring costs this
/// family its classical order and why we keep it anyway.
public struct RalstonScheduler: TableauScheduler {

  /// How many rows — and therefore which tableau. Fixed for the life of the
  /// scheduler: the variable-row case is `deis_*` (WP-E14), not this one.
  public enum Stages: Int, Sendable, CaseIterable {
    case two = 2
    case three = 3
    case four = 4

    /// The RES4LYF sampler name (`linear/ralston_3s` in its UI).
    public var name: String { "ralston_\(rawValue)s" }
  }

  /// Linear frame, but still a data-prediction consumer — see the type doc.
  public let modelOutputConvention: ModelOutputConvention = .dataPrediction

  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int
  public let stages: Stages

  public var rows: Int { stages.rawValue }
  public var frame: TableauFrame { .linear }

  /// The Butcher tableau, strictly lower-triangular `a`, in `Double`.
  /// Internal so the tests can pin it against upstream and against the
  /// `a_matrix` / `b_weights` RES4LYF recorded in E18's step traces.
  let aMatrix: [[Double]]
  let bWeights: [Double]
  let cNodes: [Double]

  private let sigmaValues: [Float]

  /// - Parameters:
  ///   - stages: 2, 3 or 4 — which Ralston tableau.
  ///   - numInferenceSteps: number of denoising steps.
  ///   - sigmaValues: `numInferenceSteps + 1` sigmas, monotonically decreasing.
  ///   - numTrainTimesteps: training timestep count for deriving `timesteps`.
  public init(
    stages: Stages,
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(
      sigmaValues.count == numInferenceSteps + 1,
      "sigmaValues must have numInferenceSteps + 1 elements")

    let table = Self.tableau(for: stages)
    self.stages = stages
    self.aMatrix = table.a
    self.bWeights = table.b
    self.cNodes = table.c
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
    let sigma = sigmaValues[checked(timestepIndex)]
    return RES4LYFTableau.rowSigma(
      frame: frame, sigma: sigma, h: stepSize(timestepIndex: timestepIndex),
      c: Float(cNodes[row]))
  }

  public mutating func rowSample(
    timestepIndex: Int, row: Int, x0: MLXArray, k: [MLXArray]
  ) -> MLXArray {
    precondition(row >= 1 && row < rows, "row 0 is the step's start sample")
    precondition(k.count == row, "row \(row) needs exactly \(row) prior outputs, got \(k.count)")
    return RES4LYFTableau.advance(
      frame: frame, x0: x0, dataPredictions: k, weights: aMatrix[row],
      h: stepSize(timestepIndex: timestepIndex), sigma: sigmaValues[checked(timestepIndex)])
  }

  public mutating func commit(timestepIndex: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray {
    precondition(k.count == rows, "commit needs \(rows) outputs, got \(k.count)")
    return RES4LYFTableau.advance(
      frame: frame, x0: x0, dataPredictions: k, weights: bWeights,
      h: stepSize(timestepIndex: timestepIndex), sigma: sigmaValues[checked(timestepIndex)])
  }

  // MARK: - There is no 1-row fallback

  /// A `ralston_Ns` step is `rows` model evaluations. There is no meaningful
  /// one-evaluation version of it, so a caller that reaches `step` has
  /// dispatched an N-row conformer through a 1-row loop and is about to get a
  /// first-order Euler render under a third-order sampler's name.
  ///
  /// That is precisely the silent substitution this FDD exists to kill, so it
  /// is a hard failure here and the reachable paths refuse the name earlier:
  /// `GeneratePayload.validateTableauSampler(_:family:)` returns 400 on any
  /// non-Krea 2 family, and the CLI's `--scheduler` refuses before weights
  /// load. `RES2sScheduler` keeping a fallback is not a precedent — it is a
  /// 2-row scheduler and the Z-Image pipelines dispatch 2-row.
  public mutating func step(
    modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray
  ) -> MLXArray {
    preconditionFailure(
      "\(stages.name) is an N-row conformer driven by a 1-row loop: `step` would silently "
        + "render first-order Euler under the name \(stages.name). Drive it through "
        + "Krea2DenoiseLoop (rowSigma / rowSample / commit), or refuse the sampler on this path.")
  }

  // MARK: - Coefficients

  /// Verbatim from `rk_coefficients_beta.py`. `a` is padded to a full
  /// `rows × rows` strictly lower-triangular matrix; upstream stores ragged
  /// rows and an implicit leading zero.
  static func tableau(for stages: Stages) -> (a: [[Double]], b: [Double], c: [Double]) {
    switch stages {
    case .two:
      // `"ralston_2s"` (:1294)
      return (
        a: [[0, 0], [2.0 / 3.0, 0]],
        b: [1.0 / 4.0, 3.0 / 4.0],
        c: [0, 2.0 / 3.0]
      )
    case .three:
      // `"ralston_3s"` (:1241)
      return (
        a: [[0, 0, 0], [1.0 / 2.0, 0, 0], [0, 3.0 / 4.0, 0]],
        b: [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
        c: [0, 1.0 / 2.0, 3.0 / 4.0]
      )
    case .four:
      // `"ralston_4s"` (:1207) — Ralston (1962), the minimum-truncation-error
      // 4-stage method; every coefficient is a closed form in √5.
      let r5 = 5.0.squareRoot()
      return (
        a: [
          [0, 0, 0, 0],
          [2.0 / 5.0, 0, 0, 0],
          [(-2889 + 1428 * r5) / 1024, (3785 - 1620 * r5) / 1024, 0, 0],
          [
            (-3365 + 2094 * r5) / 6040, (-975 - 3046 * r5) / 2552,
            (467040 + 203968 * r5) / 240845, 0,
          ],
        ],
        b: [
          (263 + 24 * r5) / 1812, (125 - 1000 * r5) / 3828,
          (3426304 + 1661952 * r5) / 5924787, (30 - 4 * r5) / 123,
        ],
        c: [0, 2.0 / 5.0, (14 - 3 * r5) / 16, 1]
      )
    }
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
