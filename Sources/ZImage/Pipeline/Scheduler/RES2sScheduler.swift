import Foundation
import MLX

/// RES 2s: Refined Exponential Solver, 2-stage.
///
/// A second-order exponential integrator using two model evaluations per
/// denoising step. The first model output is evaluated at the current sample,
/// then a substep at `c2 * h` is used for the second evaluation before the
/// exponential-integrator final update.
///
/// **Takes the data prediction, not the velocity.** The update
/// `x' = e^{-h}·x + h·Σ bᵢ·φ(−h)·kᵢ` with `h = −log(σ'/σ)` is the
/// exponential-integrator form in x₀: for rectified flow
/// `x_σ = σ·ε + (1−σ)·x₀` it gives `x' = (σ'/σ)·x + (1 − σ'/σ)·x₀`, which is
/// exactly the trajectory point at σ'. Fed a velocity it is dimensionally
/// wrong (FDD-krea2-raw-recipe D2). Callers convert through
/// ``ZImageScheduler/modelInput(velocity:sample:sigma:)``.
///
/// Reference: "Improved Order Analysis and Design of Exponential Integrator
/// for Diffusion Models Sampling" (arXiv:2308.02157).
public struct RES2sScheduler: ZImageScheduler, RES4LYFFrameScheduler {
  /// Exponential frame: every `modelOutput` here is `x₀ = x − σ·v`.
  public let modelOutputConvention: ModelOutputConvention = .dataPrediction

  /// `h = −log(σ'/σ)`, `ε = denoised − x₀` (WP-E16's T3 guards read this).
  public var frame: TableauFrame { .exponential }

  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int
  public let requiresIntermediateEvaluation: Bool = true
  public let c2: Float

  /// RES4LYF's model-free `σ_min → 0` conversion sigma when this scheduler was
  /// built from a ``RES4LYFSigmaPreparation``-prepared grid; `nil` otherwise
  /// (the Z-Image pipelines, and any direct construction).
  public let finalConversionSigma: Float?

  private var firstModelOutput: (timestepIndex: Int, output: MLXArray)?

  /// Create a RES 2s scheduler with pre-computed sigma values.
  ///
  /// - Parameters:
  ///   - numInferenceSteps: Number of denoising steps.
  ///   - sigmaValues: Array of `numInferenceSteps + 1` sigma values,
  ///     monotonically decreasing with a trailing zero.
  ///   - numTrainTimesteps: Training timestep count for deriving timesteps.
  ///   - c2: Second-stage substep location in log-sigma space. Defaults to 0.5.
  public init(
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000,
    c2: Float = 0.5,
    finalConversionSigma: Float? = nil
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(
      sigmaValues.count == numInferenceSteps + 1,
      "sigmaValues must have numInferenceSteps + 1 elements"
    )
    precondition(c2 > 0.0 && c2 <= 1.0, "c2 must be in (0, 1]")

    let numTrainF = Float(numTrainTimesteps)
    let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

    self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
    self.timesteps = MLXArray(timestepValues, [timestepValues.count])
    self.numInferenceSteps = numInferenceSteps
    self.c2 = c2
    self.finalConversionSigma = finalConversionSigma
  }

  /// Single-step fallback: first-order exponential Euler update.
  public mutating func step(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
      "invalid timestep index"
    )

    firstModelOutput = nil

    let h = stepSize(timestepIndex: timestepIndex)
    let decay = scalar(expf(-h), like: sample)
    let coeff = scalar(h * Self.phi1(-h), like: sample)
    return decay * sample + coeff * modelOutput
  }

  /// Compute the RES 2s intermediate sample at `c2 * h`.
  public mutating func intermediateStep(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray? {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
      "invalid timestep index"
    )

    firstModelOutput = (timestepIndex, modelOutput)

    let h = stepSize(timestepIndex: timestepIndex)
    let a21 = c2 * Self.phi1(-h * c2)
    let decay = scalar(expf(-c2 * h), like: sample)
    let coeff = scalar(h * a21, like: sample)
    return decay * sample + coeff * modelOutput
  }

  /// Sigma for the second model evaluation, located at `c2 * h`.
  public func intermediateSigma(timestepIndex: Int) -> Float? {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
      "invalid timestep index"
    )

    let sigma = sigmas[timestepIndex].item(Float.self)
    let h = stepSize(timestepIndex: timestepIndex)
    return sigma * expf(-c2 * h)
  }

  /// Final RES 2s update using the stored first output and second-stage output.
  public mutating func finalizeStep(
    originalOutput: MLXArray,
    intermediateOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
      "invalid timestep index"
    )

    let k1: MLXArray
    if let firstModelOutput, firstModelOutput.timestepIndex == timestepIndex {
      k1 = firstModelOutput.output
    } else {
      k1 = originalOutput
    }
    firstModelOutput = nil

    let h = stepSize(timestepIndex: timestepIndex)
    let phi1Value = Self.phi1(-h)
    let phi2Value = Self.phi2(-h)
    let b1 = phi1Value - phi2Value / c2
    let b2 = phi2Value / c2

    let decay = scalar(expf(-h), like: sample)
    let coeff1 = scalar(h * b1, like: sample)
    let coeff2 = scalar(h * b2, like: sample)
    return decay * sample + coeff1 * k1 + coeff2 * intermediateOutput
  }

  public mutating func reset() {
    firstModelOutput = nil
  }

  // MARK: - Phi Functions

  static func phi1ForTesting(_ z: Float) -> Float {
    phi1(z)
  }

  static func phi2ForTesting(_ z: Float) -> Float {
    phi2(z)
  }

  private static func phi1(_ z: Float) -> Float {
    if abs(z) < 1e-4 {
      return 1.0 + z / 2.0 + z * z / 6.0
    }
    return (expf(z) - 1.0) / z
  }

  private static func phi2(_ z: Float) -> Float {
    if abs(z) < 1e-4 {
      return 0.5 + z / 6.0 + z * z / 24.0
    }
    return (expf(z) - 1.0 - z) / (z * z)
  }

  // MARK: - Helpers

  /// ``RES4LYFFrameScheduler``: `NS.h`, the exponential frame's step size.
  public func frameStepSize(timestepIndex: Int) -> Float {
    stepSize(timestepIndex: timestepIndex)
  }

  private func stepSize(timestepIndex: Int) -> Float {
    let sigma = max(sigmas[timestepIndex].item(Float.self), 1e-8)
    let sigmaNext = max(sigmas[timestepIndex + 1].item(Float.self), 1e-8)
    return -logf(sigmaNext / sigma)
  }

  private func scalar(_ value: Float, like sample: MLXArray) -> MLXArray {
    MLXArray(value).asType(sample.dtype)
  }
}
