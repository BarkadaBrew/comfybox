import Foundation

/// Identifies a sigma schedule algorithm.
public enum SigmaScheduleKind: String, CaseIterable, Sendable {
  case flow = "flow"
  case karras = "karras"
  case exponential = "exponential"
  case beta = "beta"
  case beta57 = "beta57"
  /// Krea 2's native resolution-shifted warp over `linspace(1 → 0)` inclusive
  /// (see ``SigmaSchedule/krea2(numSteps:mu:sigmaExp:)``). Requires `mu`.
  case krea2 = "krea2"
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

  // MARK: - Krea 2 (native resolution-shifted warp)

  /// Krea 2's native timestep schedule, as Krea's reference `sampling.py` runs it.
  ///
  /// Emits `numSteps + 1` points over `linspace(1 → 0)` **inclusive of an exact
  /// 0**, each warped by `exp(mu) / (exp(mu) + (1/t − 1)^sigmaExp)`. The final
  /// point is the exact `0.0` sentinel every other schedule here appends.
  ///
  /// This is **not** ``flow(numSteps:config:mu:)``: `.flow` applies the same
  /// warp to `linspace(1 → shiftedSigmaMin)` and then appends 0, so the two
  /// grids share only their endpoints (the penultimate sigma differs). Krea 2
  /// must default to this schedule, never to `.flow` (FDD-krea2-raw-recipe §3.1).
  ///
  /// `Krea2Sampling.timesteps` delegates here, so the pipelines' schedule and
  /// the factory's `.krea2` grid are one body and cannot drift. The loop is the
  /// pre-change `timesteps` body verbatim — its float operation order is
  /// pinned by `Krea2SigmaScheduleTests.testMatchesPreChangeOracle`.
  ///
  /// - Parameters:
  ///   - numSteps: Number of denoising steps; the result has `numSteps + 1` entries.
  ///   - mu: Log-shift. Resolution-dependent by default (`Krea2Sampling.mu`),
  ///     or `log(shift)` when a request states an explicit shift (D3).
  ///   - sigmaExp: Exponent on `(1/t − 1)`; Krea's reference uses 1.0.
  public static func krea2(numSteps: Int, mu: Float, sigmaExp: Float = 1.0) -> [Float] {
    guard numSteps > 0 else { return [0.0] }
    let expMu = Foundation.exp(mu)
    var out: [Float] = []
    out.reserveCapacity(numSteps + 1)
    for i in 0...numSteps {
      let t = 1.0 - Float(i) / Float(numSteps)  // linspace 1 -> 0
      if t <= 0 {
        out.append(0)
      } else {
        let warped = expMu / (expMu + Foundation.pow(1.0 / t - 1.0, sigmaExp))
        out.append(warped)
      }
    }
    return out
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
