import Foundation
import MLX
import MLXRandom

/// DPM++ 2S Ancestral: Second-order solver with ancestral sampling.
///
/// Combines the DPM++ 2M multistep correction with controlled noise
/// injection for creative variation. Falls back to Euler on the first
/// step (no history). The `eta` parameter scales the ancestral noise
/// (1.0 = full ancestral, 0.0 = deterministic, identical to DPM++ 2M).
///
/// Reference: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling
/// of Diffusion Probabilistic Models" (2022).
public struct DPMPlusPlus2SAScheduler: ZImageScheduler {
  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int

  /// Ancestral noise scaling (0 = deterministic, 1 = full ancestral).
  private let eta: Float

  /// Cached previous model output for multistep correction.
  private var previousOutput: MLXArray?

  /// Random key for reproducible stochastic sampling.
  private var randomKey: MLXArray?

  /// Create a DPM++ 2S Ancestral scheduler with pre-computed sigma values.
  ///
  /// - Parameters:
  ///   - numInferenceSteps: Number of denoising steps.
  ///   - sigmaValues: Array of `numInferenceSteps + 1` sigma values,
  ///     monotonically decreasing with a trailing zero.
  ///   - numTrainTimesteps: Training timestep count for deriving timesteps.
  ///   - eta: Ancestral noise scaling (default 1.0).
  ///   - randomKey: Optional MLX random key for reproducibility.
  public init(
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000,
    eta: Float = 1.0,
    randomKey: MLXArray? = nil
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
    self.eta = eta
    self.randomKey = randomKey
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

    let sigmaT = sigmas[timestepIndex].asType(sample.dtype)
    let sigmaNext = sigmas[timestepIndex + 1].asType(sample.dtype)
    let dt = sigmaNext - sigmaT

    // Multistep velocity combination.
    let result: MLXArray
    if let prev = previousOutput {
      // Second-order: linear combination of current and previous velocities.
      let combined = 1.5 * modelOutput - 0.5 * prev
      result = sample + combined * dt
    } else {
      // First step: Euler fallback.
      result = sample + modelOutput * dt
    }

    previousOutput = modelOutput

    // Ancestral noise injection when eta > 0 and not at the final sigma.
    let epsScalar = MLXArray(Float(1e-8)).asType(sample.dtype)
    if eta > 0, MLX.all(sigmaNext .> epsScalar).item(Bool.self) {
      // sigma_up = sigma_next * eta * sqrt(1 - (sigma_t / sigma_next)^2)
      let ratio = sigmaT / MLX.maximum(sigmaNext, epsScalar)
      let sigmaUp = sigmaNext * MLXArray(eta).asType(sample.dtype)
        * MLX.sqrt(MLX.maximum(1.0 - ratio * ratio, epsScalar))

      let noise = nextNoise(shape: sample.shape)
      return result + sigmaUp * noise
    }

    return result
  }

  public mutating func reset() {
    previousOutput = nil
  }

  // MARK: - Private

  /// Generate noise and advance the random key for reproducibility.
  private mutating func nextNoise(shape: [Int]) -> MLXArray {
    guard let key = randomKey else {
      return MLXRandom.normal(shape)
    }
    let (k1, k2) = MLXRandom.split(key: key)
    let noise = MLXRandom.normal(shape, key: k1)
    randomKey = k2
    return noise
  }
}
