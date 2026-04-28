import Foundation

/// Identifies a sigma schedule algorithm.
public enum SigmaScheduleKind: String, CaseIterable, Sendable {
  case flow = "flow"
  case karras = "karras"
  case exponential = "exponential"
  case beta = "beta"
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
  /// Uses a polynomial approximation of the beta PDF (no scipy dependency).
  /// Matches chroma-generate's fallback path in `SigmaSchedule.beta`.
  public static func beta(
    numSteps: Int,
    sigmaMin: Float = 0.02,
    sigmaMax: Float = 100.0,
    alpha: Float = 0.6,
    betaParam: Float = 0.6
  ) -> [Float] {
    guard numSteps > 0 else { return [0.0] }
    let logMin = logf(sigmaMin)
    let logMax = logf(sigmaMax)
    var sigmas: [Float] = (0..<numSteps).map { i in
      let x = 0.01 + 0.98 * Float(i) / Float(max(1, numSteps - 1))
      let betaApprox = powf(x * (1.0 - x), 0.5)
      let maxVal = powf(0.25, 0.5)  // peak of x*(1-x) at x=0.5
      let normalized = maxVal > 1e-10 ? betaApprox / maxVal : 1.0
      return expf(logMax + (logMin - logMax) * normalized)
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
