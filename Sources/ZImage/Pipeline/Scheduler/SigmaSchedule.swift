import Foundation

/// Identifies a sigma schedule algorithm.
public enum SigmaScheduleKind: String, CaseIterable, Sendable {
  case flow = "flow"
  case karras = "karras"
  case exponential = "exponential"
  case beta = "beta"
  case beta57 = "beta57"
}

/// Pure-function sigma schedule generators.
///
/// Each function returns an array of `numSteps + 1` sigma values,
/// monotonically decreasing from `sigma_max` toward 0, with a trailing
/// zero sentinel. The trailing zero marks the end of the schedule and
/// is not used as a step sigma.
public enum SigmaSchedule {

  // MARK: - Flow-Match (current ZImage default)

  /// Flow-matching sigma schedule with optional dynamic shifting.
  ///
  /// This reproduces the existing `FlowMatchEulerScheduler` sigma computation
  /// exactly, preserving identical output for the default path.
  public static func flow(
    numSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float? = nil
  ) -> [Float] {
    let numTrainTimesteps = Float(config.numTrainTimesteps)
    let shift = config.shift

    let initSigmaMin: Float = 1.0 / numTrainTimesteps
    let shiftedSigmaMin = shift * initSigmaMin / (1 + (shift - 1) * initSigmaMin)
    let sigmaMax: Float = 1.0
    let sigmaMin: Float = shiftedSigmaMin

    let timesteps = linspace(
      sigmaMax * numTrainTimesteps,
      sigmaMin * numTrainTimesteps,
      count: numSteps
    )

    var sigmas = timesteps.map { $0 / numTrainTimesteps }

    if config.useDynamicShifting, let mu {
      sigmas = sigmas.map { sigma in
        exp(mu) / (exp(mu) + pow(1 / sigma - 1, 1.0))
      }
    } else if abs(shift - 1.0) > Float.ulpOfOne {
      sigmas = sigmas.map { sigma in
        let numerator = shift * sigma
        let denominator = 1 + (shift - 1) * sigma
        return denominator > 0 ? numerator / denominator : sigma
      }
    }

    sigmas.append(0.0)
    return sigmas
  }

  // MARK: - Karras (Karras et al. 2022)

  /// Karras sigma schedule -- optimized noise distribution.
  ///
  /// Produces sigmas spaced according to an inverse-power ramp.
  /// The default parameters match the standard Karras schedule used
  /// by Stable Diffusion and chroma-generate.
  public static func karras(
    numSteps: Int,
    sigmaMin: Float = 0.02,
    sigmaMax: Float = 100.0,
    rho: Float = 7.0
  ) -> [Float] {
    guard numSteps > 0 else { return [0.0] }
    let minInvRho = powf(sigmaMin, 1.0 / rho)
    let maxInvRho = powf(sigmaMax, 1.0 / rho)
    var sigmas: [Float] = (0..<numSteps).map { i in
      let t = Float(i) / Float(max(1, numSteps - 1))
      return powf(maxInvRho + t * (minInvRho - maxInvRho), rho)
    }
    sigmas.append(0.0)
    return sigmas
  }

  // MARK: - Exponential

  /// Exponential sigma schedule -- log-linear interpolation from max to min.
  ///
  /// Matches chroma-generate's `SigmaSchedule.sgm_uniform`, which is
  /// the standard exponential (log-uniform) schedule.
  public static func exponential(
    numSteps: Int,
    sigmaMin: Float = 0.02,
    sigmaMax: Float = 100.0
  ) -> [Float] {
    guard numSteps > 0 else { return [0.0] }
    let logMin = logf(sigmaMin)
    let logMax = logf(sigmaMax)
    var sigmas: [Float] = (0..<numSteps).map { i in
      let t = Float(i) / Float(max(1, numSteps - 1))
      return expf(logMax + t * (logMin - logMax))
    }
    sigmas.append(0.0)
    return sigmas
  }

  // MARK: - Beta

  /// Beta-distribution-inspired sigma schedule.
  ///
  /// Warps uniform timesteps through the beta CDF to redistribute step
  /// density. With alpha=beta=0.6 (U-shaped PDF), the CDF rises steeply
  /// at both edges and slowly through the middle. When used as the
  /// interpolation factor, this concentrates more sigma steps in the
  /// mid-noise range where denoising quality matters most, while
  /// spending fewer steps at extreme noise levels.
  ///
  /// The CDF is computed via numerical integration of the beta PDF at
  /// high resolution — no scipy dependency required.
  public static func beta(
    numSteps: Int,
    sigmaMin: Float = 0.02,
    sigmaMax: Float = 100.0,
    alpha: Float = 0.6,
    betaParam: Float = 0.6
  ) -> [Float] {
    guard numSteps > 0 else { return [0.0] }

    // Build CDF via midpoint-rule integration of beta PDF.
    // PDF(x; a, b) ∝ x^(a-1) * (1-x)^(b-1)
    // Midpoint rule avoids the singularities at x=0 and x=1 when alpha,beta < 1.
    let resolution = 10000
    var cdf = [Float](repeating: 0.0, count: resolution + 1)
    cdf[0] = 0.0  // CDF(0) = 0 by definition
    for i in 1...resolution {
      let xMid = (Float(i) - 0.5) / Float(resolution)  // midpoint of bin
      let pdfVal = powf(xMid, alpha - 1) * powf(1.0 - xMid, betaParam - 1)
      cdf[i] = cdf[i - 1] + pdfVal
    }
    // Normalize so CDF(1) = 1.
    let total = cdf[resolution]
    guard total > 0 else {
      return exponential(numSteps: numSteps, sigmaMin: sigmaMin, sigmaMax: sigmaMax)
    }
    for i in 1...resolution {
      cdf[i] /= total
    }

    // Evaluate CDF at uniform timesteps and use as interpolation factor.
    // CDF(t) warps the uniform spacing: steep CDF regions = big sigma
    // jumps (fewer effective steps), flat CDF regions = small sigma
    // changes (more effective steps concentrated there).
    let logMin = logf(sigmaMin)
    let logMax = logf(sigmaMax)
    var sigmas: [Float] = (0..<numSteps).map { i in
      let t = Float(i) / Float(max(1, numSteps - 1))
      let idx = min(Int(t * Float(resolution)), resolution)
      let warped = cdf[idx]
      return expf(logMax + (logMin - logMax) * warped)
    }
    sigmas.append(0.0)
    return sigmas
  }

  // MARK: - Helpers

  static func linspace(_ start: Float, _ end: Float, count: Int) -> [Float] {
    guard count > 1 else { return [start] }
    let step = (end - start) / Float(count - 1)
    return (0..<count).map { idx in start + Float(idx) * step }
  }
}
