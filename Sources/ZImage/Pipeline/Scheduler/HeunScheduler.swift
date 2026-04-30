import Foundation
import MLX

/// Heun method: Second-order Runge-Kutta (explicit trapezoidal rule).
///
/// Requires two model evaluations per step:
/// 1. Evaluate at current point to get `d1` (velocity).
/// 2. Euler-step to the next sigma to get an intermediate sample.
/// 3. Evaluate at the intermediate point to get `d2`.
/// 4. Average `d1` and `d2` for the final step (trapezoidal rule).
///
/// This provides significantly better accuracy than first-order Euler
/// at 2x the computational cost per step. Use half the step count of
/// Euler for equivalent wall-clock time with better quality.
public struct HeunScheduler: ZImageScheduler {
  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int
  public let requiresIntermediateEvaluation: Bool = true

  /// Create a Heun scheduler with pre-computed sigma values.
  ///
  /// - Parameters:
  ///   - numInferenceSteps: Number of denoising steps.
  ///   - sigmaValues: Array of `numInferenceSteps + 1` sigma values,
  ///     monotonically decreasing with a trailing zero.
  ///   - numTrainTimesteps: Training timestep count for deriving timesteps.
  public init(
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(
      sigmaValues.count == numInferenceSteps + 1,
      "sigmaValues must have numInferenceSteps + 1 elements"
    )

    let numTrainF = Float(numTrainTimesteps)
    let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

    self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
    self.timesteps = MLXArray(timestepValues, [timestepValues.count])
    self.numInferenceSteps = numInferenceSteps
  }

  /// Single-step fallback: degrades to Euler when called directly
  /// without the intermediate evaluation protocol.
  public mutating func step(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
      "invalid timestep index"
    )
    let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
    return sample + modelOutput * dt
  }

  /// Euler step to the next sigma, producing the intermediate sample.
  ///
  /// The caller must evaluate the model at the returned sample and
  /// pass both predictions to ``finalizeStep(originalOutput:intermediateOutput:timestepIndex:sample:)``.
  public mutating func intermediateStep(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray? {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
      "invalid timestep index"
    )
    let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
    return sample + modelOutput * dt
  }

  /// Trapezoidal rule: average the two velocity predictions and step.
  ///
  /// - Parameters:
  ///   - originalOutput: Velocity prediction at the original sample.
  ///   - intermediateOutput: Velocity prediction at the intermediate sample.
  ///   - timestepIndex: The current step index.
  ///   - sample: The original latent sample (before the intermediate step).
  /// - Returns: The updated latent sample.
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
    let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
    let avgOutput = (originalOutput + intermediateOutput) / 2.0
    return sample + avgOutput * dt
  }
}
