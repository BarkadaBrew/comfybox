import Foundation
import MLX

/// Flow-matching Euler scheduler for SeedVR2 super-resolution inference.
///
/// SeedVR2 uses a continuous-time flow-matching formulation where the model
/// predicts the velocity field between the noisy input and the clean output.
/// This scheduler implements the Euler discretization of that flow.
///
/// ## Timestep Computation
///
/// Given N inference steps and T training timesteps (default 1000):
///
///     step_size = T / N
///     timesteps = [T, T - step_size, ..., 0]    (N+1 values)
///     sigmas    = timesteps / T                  (normalized to [0, 1])
///
/// Each denoising step uses a pair (t, s) of consecutive timesteps.
///
/// ## Step Formula
///
///     pred_x_0    = sample - (t/T) * model_output
///     pred_noise  = sample + (1 - t/T) * model_output
///
///     if s == 0 (final step):
///         return pred_x_0
///     else:
///         return (1 - s/T) * pred_x_0 + (s/T) * pred_noise
///
/// For a single-step schedule (the default), one model evaluation suffices.
public struct SeedVR2EulerScheduler {

  /// Number of training timesteps the model was trained with.
  public let numTrainTimesteps: Int

  /// Number of inference steps to run.
  public let numInferenceSteps: Int

  /// Classifier-free guidance scale (unused in the step formula but stored for callers).
  public let guidance: Float

  /// The training timestep ceiling as a float for arithmetic.
  private let T: Float

  /// Timestep values: N+1 elements from T down to 0.
  public let timesteps: MLXArray

  /// Sigma values: timesteps / T, normalized to [0, 1].
  public let sigmas: MLXArray

  /// Creates a SeedVR2 Euler scheduler.
  ///
  /// - Parameters:
  ///   - numTrainTimesteps: Number of timesteps used during training. Default 1000.
  ///   - numInferenceSteps: Number of denoising steps at inference. Default 1.
  ///   - guidance: Classifier-free guidance scale. Default 1.0.
  public init(
    numTrainTimesteps: Int = 1000,
    numInferenceSteps: Int = 1,
    guidance: Float = 1.0
  ) {
    self.numTrainTimesteps = numTrainTimesteps
    self.numInferenceSteps = numInferenceSteps
    self.guidance = guidance
    self.T = Float(numTrainTimesteps)

    let stepSize = self.T / Float(numInferenceSteps)
    var values: [Float] = []
    for i in 0...numInferenceSteps {
      let t = self.T - Float(i) * stepSize
      values.append(max(t, 0.0))
    }

    self.timesteps = MLXArray(values)
    self.sigmas = MLXArray(values.map { v in v / Float(numTrainTimesteps) })
  }

  /// Performs a single Euler step.
  ///
  /// - Parameters:
  ///   - modelOutput: The model predicted velocity (same shape as sample).
  ///   - timestepIndex: Index into the timesteps array for the current step.
  ///     The next timestep is timesteps[timestepIndex + 1].
  ///   - sample: The current noisy latent.
  /// - Returns: The denoised (or partially denoised) latent.
  public func step(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray {
    let t = timesteps[timestepIndex]
    let s = timesteps[timestepIndex + 1]

    let tNorm = t / T
    let sNorm = s / T

    // Predicted clean sample and noise from flow matching.
    let predX0 = sample - tNorm * modelOutput
    let predNoise = sample + (1 - tNorm) * modelOutput

    // If this is the final step (s == 0), return the clean prediction directly.
    // Otherwise, interpolate between clean and noise at the next sigma level.
    let sValue = s.item(Float.self)
    if sValue > 0 {
      return (1 - sNorm) * predX0 + sNorm * predNoise
    } else {
      return predX0
    }
  }
}
