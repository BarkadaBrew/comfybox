import Foundation
import MLX

/// DPM++ 2M: Second-order multistep solver (deterministic).
///
/// Caches the previous model output for second-order correction.
/// Falls back to Euler on the first step when no history is available.
///
/// Reference: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling
/// of Diffusion Probabilistic Models" (2022).
public struct DPMPlusPlus2MScheduler: ZImageScheduler {
  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int

  /// Cached previous model output for multistep correction.
  private var previousOutput: MLXArray?

  /// Create a DPM++ 2M scheduler with pre-computed sigma values.
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

    let result: MLXArray
    if let prev = previousOutput {
      // Second-order multistep: linear combination
      // D = (3/2) * v_t - (1/2) * v_{t-1}
      let combined = 1.5 * modelOutput - 0.5 * prev
      result = sample + combined * dt
    } else {
      // First step: Euler fallback (no history available)
      result = sample + modelOutput * dt
    }

    previousOutput = modelOutput
    return result
  }

  public mutating func reset() {
    previousOutput = nil
  }
}
