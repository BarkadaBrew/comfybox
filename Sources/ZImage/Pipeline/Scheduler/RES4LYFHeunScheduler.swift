// RES4LYFHeunScheduler.swift — `heun_2s` / `heun_3s`
//
// Linear-frame N-row conformers (WP-E13 shape), coefficients copied VERBATIM
// from RES4LYF `beta/rk_coefficients_beta.py` at the pinned commit
// `26036f647ca15d3048a193daf99a40cecfc3820d` — the `heun_2s` / `heun_3s`
// entries of the `rk_coeff` dictionary (plain-arithmetic branch, NOT the
// exponential phi-function branch).
//
// Classical Heun explicit Runge–Kutta (RK2 explicit-trapezoidal, RK3), in
// RES4LYF's LINEAR frame — structurally identical to ``RalstonScheduler``,
// sharing the ``RES4LYFTableau`` machinery; only the Butcher tableau and the
// stage set differ. Like the ralstons, RES4LYF's `noise_anchor = 1.0`
// anchoring (every row's derivative at the step's x₀ and sigma) costs them
// their classical order — second order in practice — which is upstream's
// behaviour and therefore ours. See `RES4LYFTableau.swift`.

import Foundation
import MLX

/// Heun's explicit Runge–Kutta methods, in RES4LYF's linear frame.
///
/// See ``RalstonScheduler`` — the two are the same conformer (linear frame,
/// shared N-row tableau machinery, data-prediction consumer); they differ only
/// in the tableau and in the stage set (`heun` has no 4-row variant).
public struct RES4LYFHeunScheduler: TableauScheduler, RES4LYFFrameScheduler {

  /// Which tableau — `heun_2s` or `heun_3s`. Fixed for the scheduler's life.
  public enum Stages: Int, Sendable, CaseIterable {
    case two = 2
    case three = 3

    /// The RES4LYF sampler name (`heun_2s` / `heun_3s`).
    public var name: String { "heun_\(rawValue)s" }
  }

  /// Linear frame, but still a data-prediction consumer — see ``RalstonScheduler``.
  public let modelOutputConvention: ModelOutputConvention = .dataPrediction

  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int
  public let stages: Stages

  /// RES4LYF's model-free `σ_min → 0` conversion sigma when built from a
  /// ``RES4LYFSigmaPreparation``-prepared grid; `nil` otherwise.
  public let finalConversionSigma: Float?

  public var rows: Int { stages.rawValue }
  public var frame: TableauFrame { .linear }

  /// The Butcher tableau, strictly lower-triangular `a`, in `Double`. Internal
  /// so tests can pin it against upstream's `rk_coeff` values.
  let aMatrix: [[Double]]
  let bWeights: [Double]
  let cNodes: [Double]

  private let sigmaValues: [Float]

  /// - Parameters:
  ///   - stages: 2 or 3 — which Heun tableau.
  ///   - numInferenceSteps: number of denoising steps.
  ///   - sigmaValues: `numInferenceSteps + 1` sigmas, monotonically decreasing.
  ///   - numTrainTimesteps: training timestep count for deriving `timesteps`.
  public init(
    stages: Stages,
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000,
    finalConversionSigma: Float? = nil
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
    self.finalConversionSigma = finalConversionSigma
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

  // MARK: - There is no 1-row fallback (see RalstonScheduler)

  public mutating func step(
    modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray
  ) -> MLXArray {
    preconditionFailure(
      "\(stages.name) is an N-row conformer driven by a 1-row loop: `step` would silently "
        + "render first-order Euler under the name \(stages.name). Drive it through "
        + "Krea2DenoiseLoop (rowSigma / rowSample / commit), or refuse the sampler on this path.")
  }

  // MARK: - Coefficients

  /// Verbatim from `rk_coefficients_beta.py` (`heun_2s`, `heun_3s`). `a` is
  /// padded to a full `rows × rows` strictly lower-triangular matrix; upstream
  /// stores ragged rows with an implicit leading zero.
  static func tableau(for stages: Stages) -> (a: [[Double]], b: [Double], c: [Double]) {
    switch stages {
    case .two:
      // `"heun_2s"`: classical Heun / explicit trapezoidal (RK2).
      //   a = [[], [1]], b = [1/2, 1/2], c = [0, 1]
      return (
        a: [[0, 0], [1, 0]],
        b: [1.0 / 2.0, 1.0 / 2.0],
        c: [0, 1]
      )
    case .three:
      // `"heun_3s"`: Heun's third-order method (RK3).
      //   a = [[], [1/3], [0, 2/3]], b = [1/4, 0, 3/4], c = [0, 1/3, 2/3]
      return (
        a: [[0, 0, 0], [1.0 / 3.0, 0, 0], [0, 2.0 / 3.0, 0]],
        b: [1.0 / 4.0, 0, 3.0 / 4.0],
        c: [0, 1.0 / 3.0, 2.0 / 3.0]
      )
    }
  }

  // MARK: - Helpers

  /// ``RES4LYFFrameScheduler``: `NS.h` for this step.
  public func frameStepSize(timestepIndex: Int) -> Float {
    stepSize(timestepIndex: timestepIndex)
  }

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
