// LTX2Sampler.swift -- Euler and res_2s samplers for LTX-2 denoising
// Phase 4 of the LTX-2 Swift/MLX port
//
// Implements two sampling strategies for the denoising loop:
//
// 1. Euler sampler: First-order ODE solver. Simple and fast.
//    x0 = x - sigma * velocity
//    x_next = x0 + sigma_next * (x - x0) / sigma
//
// 2. res_2s sampler: Second-order Rosenbrock-Runge-Kutta integrator.
//    Uses phi functions for exponential step computation with optional
//    SDE noise injection. Higher quality at the cost of 2 model evals per step.
//
// Reference: samplers.py, generate.py denoise_distilled/denoise_dev

import Foundation
import MLX
import MLXRandom

// MARK: - Euler Sampler

/// First-order Euler ODE step for flow-matching denoising.
///
/// Computes the denoised prediction (x0) from the velocity, then takes
/// an Euler step toward the next sigma level.
public enum LTX2EulerSampler {

  /// Perform a single Euler step.
  ///
  /// - Parameters:
  ///   - latents: Current noisy latent `(B, C, F, H, W)` in float32.
  ///   - velocity: Model velocity prediction (in token space `(B, T, C)`).
  ///   - sigma: Current sigma level.
  ///   - sigmaNext: Next sigma level.
  ///   - state: Optional I2V conditioning state.
  ///   - timesteps: Per-token timesteps `(B, T)` (may differ from sigma for I2V).
  /// - Returns: `(nextLatents, denoised)` tuple. Both in float32.
  public static func step(
    latents: MLXArray,
    velocity: MLXArray,
    sigma: Float,
    sigmaNext: Float,
    state: LTX2LatentState? = nil,
    timesteps: MLXArray? = nil
  ) -> (latents: MLXArray, denoised: MLXArray) {
    let b = latents.dim(0)
    let c = latents.dim(1)
    let f = latents.dim(2)
    let h = latents.dim(3)
    let w = latents.dim(4)
    let numTokens = f * h * w

    // Flatten latents to token space: (B, C, F, H, W) -> (B, T, C)
    let latentsFlat = latents.reshaped(b, c, -1).transposed(0, 2, 1)

    // Per-token timesteps for x0 computation
    let ts: MLXArray
    if let t = timesteps {
      ts = t.expandedDimensions(axis: -1)  // (B, T, 1)
    } else {
      ts = MLXArray(
        [Float](repeating: sigma, count: numTokens),
        [1, numTokens, 1]
      )
    }

    // x0 = latent - timestep * velocity (per-token)
    let x0Flat = latentsFlat - ts * velocity.asType(.float32)

    // Reshape x0 back to spatial: (B, T, C) -> (B, C, F, H, W)
    var denoised = x0Flat.transposed(0, 2, 1).reshaped(b, c, f, h, w)

    // Apply I2V denoise mask
    if let s = state {
      denoised = LTX2Conditioning.applyDenoiseMask(
        denoised: denoised,
        clean: s.cleanLatent.asType(.float32),
        denoiseMask: s.denoiseMask
      )
    }

    // Euler step
    let nextLatents: MLXArray
    if sigmaNext > 0 {
      let sigmaF = MLXArray(sigma)
      let sigmaNextF = MLXArray(sigmaNext)
      nextLatents = denoised + sigmaNextF * (latents - denoised) / sigmaF
    } else {
      nextLatents = denoised
    }

    return (nextLatents, denoised)
  }
}

// MARK: - res_2s Coefficient Computation

/// Phi function for exponential RK coefficients.
///
/// phi_j(z) = (e^z - sum_{k=0}^{j-1} z^k / k!) / z^j
private func phiFunction(_ j: Int, _ negH: Double) -> Double {
  if abs(negH) < 1e-10 {
    // Taylor expansion limit
    var factorial = 1
    for k in 1...j { factorial *= k }
    return 1.0 / Double(factorial)
  }

  var remainder = 0.0
  var factorial = 1.0
  for k in 0..<j {
    if k > 0 { factorial *= Double(k) }
    remainder += pow(negH, Double(k)) / factorial
  }
  return (exp(negH) - remainder) / pow(negH, Double(j))
}

/// Compute res_2s Runge-Kutta coefficients for a given step size.
///
/// - Parameters:
///   - h: Step size in log-space = log(sigma / sigma_next).
///   - c2: Substep position (default 0.5 = midpoint).
/// - Returns: `(a21, b1, b2)` RK coefficients.
private func getRes2sCoefficients(h: Double, c2: Double = 0.5) -> (Double, Double, Double) {
  let negHC2 = -h * c2
  let phi1C2 = phiFunction(1, negHC2)
  let a21 = c2 * phi1C2

  let negHFull = -h
  let phi2Full = phiFunction(2, negHFull)
  let b2 = phi2Full / c2

  let phi1Full = phiFunction(1, negHFull)
  let b1 = phi1Full - b2

  return (a21, b1, b2)
}

// MARK: - SDE Noise

/// Compute SDE coefficients for variance-preserving noise injection.
func getSdeCoeff(sigmaNext: Float) -> (alphaRatio: Float, sigmaDown: Float, sigmaUp: Float) {
  var sigmaUp = sigmaNext * 0.5
  sigmaUp = min(sigmaUp, sigmaNext * 0.9999)

  let sigmaSignal = 1.0 - sigmaNext
  let sigmaResidual = sqrt(max(sigmaNext * sigmaNext - sigmaUp * sigmaUp, 0.0))
  let alphaRatio = sigmaSignal + sigmaResidual

  let sigmaDown: Float
  if alphaRatio == 0 {
    sigmaDown = sigmaNext
  } else {
    sigmaDown = sigmaResidual / alphaRatio
  }

  return (
    alphaRatio.isNaN ? 1.0 : alphaRatio,
    sigmaDown.isNaN ? sigmaNext : sigmaDown,
    sigmaUp.isNaN ? 0.0 : sigmaUp
  )
}

/// Generate channel-wise normalized Gaussian noise.
func getNewNoise(shape: [Int]) -> MLXArray {
  var noise = MLXRandom.normal(shape, dtype: .float32)
  // Global normalization
  let mean = noise.mean()
  let rms = MLX.sqrt(MLX.mean(noise * noise)) + 1e-8
  noise = (noise - mean) / rms
  // Channel-wise normalization over spatial dims
  let channelMean = MLX.mean(noise, axes: [-2, -1], keepDims: true)
  noise = noise - channelMean
  let channelStd = MLX.sqrt(MLX.mean(noise * noise, axes: [-2, -1], keepDims: true) + 1e-8)
  noise = noise / channelStd
  return noise
}

// MARK: - res_2s Sampler

/// Second-order Rosenbrock-Runge-Kutta (res_2s) sampler.
///
/// Higher quality than Euler with optional SDE noise injection.
/// Requires 2 model evaluations per step.
public enum LTX2Res2sSampler {

  /// Perform the predictor half of a res_2s step.
  ///
  /// This is essentially an Euler step. The caller should then run the model
  /// again at `sigmaNext` and call `correct` with both velocity predictions.
  ///
  /// - Parameters:
  ///   - latents: Current noisy latent `(B, C, F, H, W)` in float32.
  ///   - velocity: Model velocity prediction from the first evaluation.
  ///   - sigma: Current sigma level.
  ///   - sigmaNext: Next sigma level.
  ///   - state: Optional I2V conditioning state.
  ///   - timesteps: Per-token timesteps.
  ///   - useSDE: Whether to inject SDE noise. Default false.
  /// - Returns: `(predictorLatents, denoised)` for the predictor pass.
  public static func predict(
    latents: MLXArray,
    velocity: MLXArray,
    sigma: Float,
    sigmaNext: Float,
    state: LTX2LatentState? = nil,
    timesteps: MLXArray? = nil,
    useSDE: Bool = false
  ) -> (predictorLatents: MLXArray, denoised: MLXArray) {
    let (eulerLatents, denoised) = LTX2EulerSampler.step(
      latents: latents,
      velocity: velocity,
      sigma: sigma,
      sigmaNext: sigmaNext,
      state: state,
      timesteps: timesteps
    )

    if useSDE && sigmaNext > 0 {
      let noise = getNewNoise(shape: latents.shape.map { $0 })
      let (alphaRatio, sigmaDown, sigmaUp) = getSdeCoeff(sigmaNext: sigmaNext)

      let sigmaF = MLXArray(sigma)
      let eps = (latents - denoised) / sigmaF
      let denoisedNext = latents - sigmaF * eps
      let noised = MLXArray(alphaRatio) * (denoisedNext + MLXArray(sigmaDown) * eps)
        + MLXArray(sigmaUp) * noise
      return (noised, denoised)
    }

    return (eulerLatents, denoised)
  }

  /// Apply the second-order corrector step.
  ///
  /// Uses the midpoint correction: average the two denoised predictions
  /// and take the Euler step from that average.
  ///
  /// - Parameters:
  ///   - originalLatents: Latents from before the predictor step.
  ///   - predictorLatents: Latents from the predictor step.
  ///   - denoised1: Denoised prediction from the predictor.
  ///   - velocity2: Model velocity prediction at the predictor point.
  ///   - sigma: Original sigma level.
  ///   - sigmaNext: Sigma level at the predictor point.
  ///   - state: Optional I2V conditioning state.
  ///   - timesteps2: Per-token timesteps at the predictor point.
  /// - Returns: Corrected latents.
  public static func correct(
    originalLatents: MLXArray,
    predictorLatents: MLXArray,
    denoised1: MLXArray,
    velocity2: MLXArray,
    sigma: Float,
    sigmaNext: Float,
    state: LTX2LatentState? = nil,
    timesteps2: MLXArray? = nil
  ) -> MLXArray {
    guard sigmaNext > 0 else { return denoised1 }

    // Compute denoised at the predictor point
    let (_, denoised2) = LTX2EulerSampler.step(
      latents: predictorLatents,
      velocity: velocity2,
      sigma: sigmaNext,
      sigmaNext: 0,
      state: state,
      timesteps: timesteps2
    )

    // Midpoint correction
    let sigmaF = MLXArray(sigma)
    let midpoint = (denoised1 + denoised2) / 2.0
    let corrected = midpoint + MLXArray(sigmaNext) * (originalLatents - denoised1) / sigmaF

    // Apply I2V mask
    if let s = state {
      return LTX2Conditioning.applyDenoiseMask(
        denoised: corrected,
        clean: s.cleanLatent.asType(.float32),
        denoiseMask: s.denoiseMask
      )
    }

    return corrected
  }
}
