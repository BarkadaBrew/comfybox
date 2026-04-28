import Foundation
import MLX

/// First-order Euler ODE solver for flow-matching diffusion models.
///
/// This is the original ZImage scheduler, now conforming to the
/// ``ZImageScheduler`` protocol. When constructed with a
/// ``ZImageSchedulerConfig`` it delegates sigma computation to
/// ``SigmaSchedule/flow(numSteps:config:mu:)`` and preserves exact
/// backward compatibility with the original behavior.
public struct FlowMatchEulerScheduler: ZImageScheduler {
  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int

  /// Create a scheduler using the flow-match sigma schedule.
  ///
  /// This reproduces the original `FlowMatchEulerScheduler` behavior
  /// exactly, including optional dynamic shifting.
  public init(
    numInferenceSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float? = nil
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(config.numTrainTimesteps > 0, "numTrainTimesteps must be positive")

    // Delegate sigma computation to the shared SigmaSchedule.flow() function.
    let sigmaValues = SigmaSchedule.flow(
      numSteps: numInferenceSteps,
      config: config,
      mu: mu
    )

    // Derive timesteps from sigmas (excluding the trailing 0).
    let numTrainTimesteps = Float(config.numTrainTimesteps)
    let timestepValues = sigmaValues.dropLast().map { $0 * numTrainTimesteps }

    self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
    self.timesteps = MLXArray(timestepValues, [timestepValues.count])
    self.numInferenceSteps = numInferenceSteps
  }

  /// Create a scheduler with pre-computed sigma values.
  ///
  /// Use this initializer with non-flow sigma schedules (Karras,
  /// Exponential, Beta) where sigma values are computed externally.
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

  public func step(
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
}
