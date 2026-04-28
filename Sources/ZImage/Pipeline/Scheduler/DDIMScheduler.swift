import Foundation
import MLX
import MLXRandom

/// DDIM: Denoising Diffusion Implicit Models.
///
/// Adjustable `eta` controls stochasticity:
/// - `eta = 0`: fully deterministic (classic DDIM)
/// - `eta = 1`: equivalent to DDPM
///
/// Uses flow-matching velocity-prediction parameterization where
/// `alpha_t = 1 - sigma_t` (not noise-prediction DDPM math).
///
/// Reference: Song et al., "Denoising Diffusion Implicit Models" (2020).
public struct DDIMScheduler: ZImageScheduler {
  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int

  /// Stochasticity parameter: 0 = deterministic, 1 = full DDPM.
  private let eta: Float

  /// Random key for reproducible stochastic sampling.
  private var randomKey: MLXArray?

  /// Create a DDIM scheduler with pre-computed sigma values.
  ///
  /// - Parameters:
  ///   - numInferenceSteps: Number of denoising steps.
  ///   - sigmaValues: Array of `numInferenceSteps + 1` sigma values,
  ///     monotonically decreasing with a trailing zero.
  ///   - numTrainTimesteps: Training timestep count for deriving timesteps.
  ///   - eta: Stochasticity parameter (0 = deterministic, 1 = stochastic).
  ///   - randomKey: Optional MLX random key for reproducibility.
  public init(
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000,
    eta: Float = 0.0,
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

    // Flow-matching parameterization: alpha_t = 1 - t
    let t = sigmas[timestepIndex].asType(.float32)
    let tPrev = sigmas[timestepIndex + 1].asType(.float32)

    let alphaT = (1.0 - t).asType(sample.dtype)
    let alphaPrev = (1.0 - tPrev).asType(sample.dtype)

    // Clamp to avoid sqrt of negative or division by zero.
    let epsScalar = MLXArray(Float(1e-8)).asType(sample.dtype)
    let sqrtAlphaT = MLX.sqrt(MLX.maximum(alphaT, epsScalar))
    let sqrtOneMinusAlphaT = MLX.sqrt(MLX.maximum(1.0 - alphaT, epsScalar))

    // Predict x_0 from velocity prediction.
    let predX0 = (sample - sqrtOneMinusAlphaT * modelOutput) / sqrtAlphaT

    // Direction pointing towards x_t.
    let sqrtAlphaPrev = MLX.sqrt(MLX.maximum(alphaPrev, epsScalar))
    let sqrtOneMinusAlphaPrev = MLX.sqrt(MLX.maximum(1.0 - alphaPrev, epsScalar))
    let dirXt = sqrtOneMinusAlphaPrev * modelOutput

    // Deterministic component.
    var xPrev = sqrtAlphaPrev * predX0 + dirXt

    // Stochastic component when eta > 0.
    if eta > 0 {
      let oneMinusAlphaT = MLX.maximum(1.0 - alphaT, epsScalar)
      let oneMinusAlphaPrev = 1.0 - alphaPrev
      let variance = MLXArray(eta).asType(sample.dtype)
        * MLX.sqrt(oneMinusAlphaPrev / oneMinusAlphaT)
        * MLX.sqrt(1.0 - alphaT / MLX.maximum(alphaPrev, epsScalar))

      let noise = nextNoise(shape: sample.shape)
      xPrev = xPrev + variance.asType(sample.dtype) * noise
    }

    return xPrev
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
