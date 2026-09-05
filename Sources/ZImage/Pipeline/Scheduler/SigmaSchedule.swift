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
  /// RES4LYF's `bong_tangent` — two arctan arcs joined at exactly 0.5, pure
  /// index arithmetic, **never consults the model** (no shift, no `mu`; FDD
  /// D6). Wire name is the upstream snake_case so a workflow value pastes
  /// verbatim. See ``SigmaSchedule/bongTangent(numSteps:)``.
  case bongTangent = "bong_tangent"
  /// ComfyUI's `simple` scheduler: `steps` evenly-spaced indices into the
  /// model's discrete sigma table, walked from the noisy (high-sigma) end
  /// down, then a trailing 0. Model-dependent — uses the same family sigma
  /// table `beta` does, so it needs `mu` under flux (Krea 2). Wire name
  /// `simple` matches ComfyUI so a workflow value pastes verbatim.
  case simple = "simple"
}

/// Pure-function sigma schedule generators.
///
/// Each function returns an array of `numSteps + 1` sigma values,
/// monotonically decreasing from `sigma_max` toward 0, with a trailing
/// zero sentinel. The trailing zero marks the end of the schedule and
/// is not used as a step sigma.
public enum SigmaSchedule {

  // MARK: - Flow-Match (current ZImage default)

  /// ComfyUI's `time_snr_shift` (`comfy/model_sampling.py:279-282`), the
  /// flow-matching "simple" shift `ModelSamplingAuraFlow` / `ModelSamplingSD3`
  /// apply:
  ///
  /// ```python
  /// def time_snr_shift(alpha, t):
  ///     if alpha == 1.0:
  ///         return t
  ///     return alpha * t / (1 + (alpha - 1) * t)
  /// ```
  ///
  /// `alpha == 1.0` is upstream's exact identity branch, not an optimisation:
  /// it is what makes "shift 1.0" mean "the unwarped grid" bit for bit.
  ///
  /// Distinct from the resolution-dependent DYNAMIC shift
  /// (`exp(mu) / (exp(mu) + (1/σ − 1))`, `ModelSamplingFlux`) that ``flow``
  /// applies when `config.useDynamicShifting`. The two never compose — see
  /// ``ZImageSchedulerConfig/applyingExplicitShift(_:)``.
  ///
  /// ``flow(numSteps:config:mu:)`` inlines this same expression (with an extra
  /// `denominator > 0` guard) rather than calling here, so its float operation
  /// order is provably the pre-#154 one; this function is the named, testable
  /// statement of the formula and the one
  /// ``discreteFlowSigmaTable(shift:numTrainTimesteps:)`` semantics match.
  ///
  /// **ComfyUI's `multiplier` cancels out of the sigma grid**, which is why
  /// `ModelSamplingAuraFlow` (multiplier 1.0) and `ModelSamplingSD3`
  /// (multiplier 1000) produce the SAME sigmas from the same `shift`.
  /// `normal_scheduler` walks `linspace(timestep(σ_max) → timestep(σ_min))` and
  /// maps each point back through `sigma(t) = time_snr_shift(shift, t / M)`;
  /// `timestep(σ) = σ · M`, so the `M` introduced by the endpoints is divided
  /// straight back out and the grid is multiplier-invariant. What the
  /// multiplier DOES scale is the timestep the model itself is fed — a
  /// property of how the DiT was trained, not of this schedule, and untouched
  /// by #154 (the Z-Image pipelines feed `σ · numTrainTimesteps`, as they
  /// always have). Pinned by `ModelSamplingShiftTests.testMultiplierCancels…`.
  public static func timeSNRShift(alpha: Float, t: Float) -> Float {
    if alpha == 1.0 { return t }
    return alpha * t / (1 + (alpha - 1) * t)
  }

  /// Flow-matching sigma schedule with optional dynamic shifting.
  ///
  /// This reproduces the existing `FlowMatchEulerScheduler` sigma computation
  /// exactly, preserving identical output for the default path.
  ///
  /// comfybox#154 does NOT add a shift parameter here: the single door for a
  /// per-request `ModelSamplingAuraFlow` shift is
  /// ``ZImageSchedulerConfig/applyingExplicitShift(_:)``, applied by the
  /// caller BEFORE the config reaches any schedule — which is what makes the
  /// table-backed schedules follow it too, exactly as they read the patched
  /// `model_sampling` upstream. A second door here would let the two disagree.
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
  ///     or the request's explicit `shift` itself — shift IS mu, ComfyUI's
  ///     `ModelSamplingFlux` parameterisation (D3 as amended by Addendum A.1).
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

  // MARK: - bong_tangent (RES4LYF)

  /// RES4LYF's `bong_tangent_scheduler`, ported literally from the pinned
  /// upstream source (`scripts/oracles/upstream/res4lyf/sigmas.py:4065-4098`,
  /// commit `26036f6`; FDD-krea2-raw-recipe §3.11, D6, AC-19/AC-20).
  ///
  /// Two arctan arcs — `start 1.0 → middle 0.5` and `middle 0.5 → end 0.0` —
  /// joined at **exactly** 0.5, emitting `numSteps + 1` values that end at an
  /// exact `0.0`. The upstream defaults are baked in (`pivot_1 = pivot_2 = 0.6`,
  /// `slope_1 = slope_2 = 0.2`, `pad = false`); no caller has asked for the
  /// knobs, and the published recipe uses the defaults.
  ///
  /// **Port literally, do not clean up.** The arc lengths come from
  /// `int((steps+2) · 0.6)` — integer truncation — so the schedule is
  /// deliberately non-smooth in `numSteps`; smoothing it would move every
  /// sigma off the oracle. Arithmetic is in `Double` (Python `float`) and cast
  /// to `Float` once at the end (upstream's `torch.tensor(list)` → float32), so
  /// the grid matches the fixture `comfy_sigmas.json` to the last bit rather
  /// than to a tolerance.
  ///
  /// **Model-free** (D6): upstream accepts `model_sampling` and never reads it
  /// in the body. `SchedulerFactory.resolveSigmas` therefore ignores `config`
  /// and `mu` for this schedule — there is no shift to compose on top, and
  /// `RenderRecipe` records `shift_applied: false` for it.
  ///
  /// - Precondition: `numSteps >= 2`. Upstream raises `ZeroDivisionError` for
  ///   0 and 1 (a one-point arc has `smax == smin`). `SchedulerFactory` refuses
  ///   those step counts by name (``SchedulerFactoryError/stepCountBelowMinimum(_:steps:minimum:)``)
  ///   before reaching this function; the precondition is the last line of
  ///   defence against a NaN grid, not the wire-level check.
  public static let bongTangentMinimumSteps = 2

  public static func bongTangent(numSteps: Int) -> [Float] {
    precondition(
      numSteps >= bongTangentMinimumSteps,
      "bong_tangent needs at least \(bongTangentMinimumSteps) steps (upstream divides by zero below that); got \(numSteps)")

    // bong_tangent_scheduler(model_sampling, steps, start=1.0, middle=0.5, end=0.0,
    //                        pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False)
    let start = 1.0, middle = 0.5, end = 0.0
    var pivot1 = 0.6, pivot2 = 0.6
    var slope1 = 0.2, slope2 = 0.2

    let steps = numSteps + 2                                                   // steps += 2
    let midpoint = Int((Double(steps) * pivot1 + Double(steps) * pivot2) / 2)  // int((steps*p1 + steps*p2)/2)
    pivot1 = Double(Int(Double(steps) * pivot1))                               // int(steps * pivot_1)
    pivot2 = Double(Int(Double(steps) * pivot2))                               // int(steps * pivot_2)
    slope1 = slope1 / (Double(steps) / 40)                                     // slope_1 / (steps/40)
    slope2 = slope2 / (Double(steps) / 40)                                     // slope_2 / (steps/40)

    let stage2Len = steps - midpoint
    let stage1Len = steps - stage2Len

    var tanSigmas1 = bongTangentArc(
      steps: stage1Len, slope: slope1, pivot: pivot1, start: start, end: middle)
    let tanSigmas2 = bongTangentArc(
      steps: stage2Len, slope: slope2, pivot: pivot2 - Double(stage1Len), start: middle, end: end)

    tanSigmas1.removeLast()  // tan_sigmas_1[:-1]
    // pad=False: no trailing extra 0 (tan_sigmas_2 already ends at `end`).

    return (tanSigmas1 + tanSigmas2).map { Float($0) }  // torch.tensor(list) → float32
  }

  /// `get_bong_tangent_sigmas(steps, slope, pivot, start, end)` — one arctan
  /// arc over `range(steps)`, normalised so `x = 0 → start` and
  /// `x = steps-1 → end`. Operation order is upstream's, in `Double`.
  private static func bongTangentArc(
    steps: Int, slope: Double, pivot: Double, start: Double, end: Double
  ) -> [Double] {
    let twoOverPi = 2.0 / Double.pi
    let smax = (twoOverPi * atan(-slope * (0.0 - pivot)) + 1) / 2
    let smin = (twoOverPi * atan(-slope * (Double(steps - 1) - pivot)) + 1) / 2

    let srange = smax - smin
    let sscale = start - end

    return (0..<steps).map { x in
      (((twoOverPi * atan(-slope * (Double(x) - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end
    }
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

  // MARK: - Beta (ComfyUI-exact)

  /// ComfyUI's `beta_scheduler` (`comfy/samplers.py:696-708`, pinned copy under
  /// `scripts/oracles/upstream/comfyui/`; FDD-krea2-raw-recipe D5, §3.11,
  /// AC-21/AC-22), ported literally:
  ///
  /// ```python
  /// total_timesteps = len(model_sampling.sigmas) - 1
  /// ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
  /// ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)
  /// sigs = [sigmas[int(t)] for consecutive-distinct t] + [0.0]
  /// ```
  ///
  /// The beta **PPF** (inverse CDF) is evaluated at `1 − i/steps`, scaled to
  /// the table's top index, rounded half-to-even (`numpy.rint`), and used as an
  /// index into the model's discrete sigma table. Consecutive repeats are
  /// dropped, so the result can have **fewer** than `numSteps + 1` entries;
  /// `SchedulerFactory` constructs with the produced count and the scheduler's
  /// `numInferenceSteps` reports it — never hidden (AC-22).
  ///
  /// This replaced, in place and under the same name, a CDF-integrator that
  /// interpolated in log-sigma space: at 6 steps / shift 1.15 its σ₁ was 0.1596
  /// where ComfyUI's is 0.9199 (D5; `BetaScheduleComfyParityTests.testBeforeAfterFixture`
  /// pins both). `CHANGELOG.md` announces the change.
  ///
  /// - Parameters:
  ///   - numSteps: Requested step count. `0` → `[0.0]`, as upstream.
  ///   - sigmaTable: The model's discrete sigma table, **ascending** (index 0 is
  ///     `σ_min`, the last entry `σ_max`), e.g. ``discreteFlowSigmaTable(shift:numTrainTimesteps:)``
  ///     or ``fluxSigmaTable(shift:tableSize:)``.
  ///   - alpha, betaParam: Beta-distribution shape. `beta` = (0.6, 0.6); `beta57` = (0.5, 0.7).
  /// ComfyUI `simple_scheduler`, statement for statement:
  ///   ss = len(sigmas) / steps
  ///   for x in range(steps): out.append(sigmas[-(1 + int(x*ss))])
  ///   out.append(0.0)
  /// `sigmaTable` is ascending (index 0 = smallest), so `-(1+k)` is
  /// `count-1-k`. No de-duplication (ComfyUI does none); with the flux table
  /// (~10k entries) vs a dozen-odd steps the indices never collide anyway.
  public static func simple(numSteps: Int, sigmaTable: [Float]) -> [Float] {
    precondition(!sigmaTable.isEmpty, "simple schedule needs a non-empty sigma table")
    guard numSteps > 0 else { return [0.0] }
    let count = sigmaTable.count
    let ss = Double(count) / Double(numSteps)
    var sigmas: [Float] = []
    sigmas.reserveCapacity(numSteps + 1)
    for x in 0..<numSteps {
      let idx = count - 1 - Int(Double(x) * ss)   // Int() truncates toward zero, == Python int()
      sigmas.append(sigmaTable[max(0, min(count - 1, idx))])
    }
    sigmas.append(0.0)
    return sigmas
  }

  public static func beta(
    numSteps: Int,
    sigmaTable: [Float],
    alpha: Float = 0.6,
    betaParam: Float = 0.6
  ) -> [Float] {
    precondition(!sigmaTable.isEmpty, "beta schedule needs a non-empty sigma table")
    guard numSteps > 0 else { return [0.0] }

    let totalTimesteps = Double(sigmaTable.count - 1)
    let a = Double(alpha), b = Double(betaParam)
    // numpy.linspace(0, 1, steps, endpoint=False) is `i * (1/steps)`; then `1 - …`.
    let step = 1.0 / Double(numSteps)

    var sigmas: [Float] = []
    sigmas.reserveCapacity(numSteps + 1)
    var lastIndex = -1.0
    for i in 0..<numSteps {
      let p = 1.0 - Double(i) * step
      let index = (betaPPF(p, alpha: a, beta: b) * totalTimesteps).rounded(.toNearestOrEven)
      if index != lastIndex {
        sigmas.append(sigmaTable[Int(index)])
      }
      lastIndex = index
    }
    sigmas.append(0.0)
    return sigmas
  }

  /// ``beta(numSteps:sigmaTable:alpha:betaParam:)`` over the discrete-flow table
  /// `σ[i] = shift·t / (1 + (shift − 1)·t)`, `t = (i+1)/T` — the signature
  /// FDD-krea2-raw-recipe §3.11 names for the `ModelSamplingDiscreteFlow`
  /// families. `SchedulerFactory` reads `shift` and `T` from a decoded
  /// scheduler config; Krea 2 is **not** on this table (its family is
  /// `ModelSamplingFlux`, built from `mu` — Addendum A.1).
  public static func beta(
    numSteps: Int,
    shift: Float,
    numTrainTimesteps: Int,
    alpha: Float = 0.6,
    betaParam: Float = 0.6
  ) -> [Float] {
    beta(
      numSteps: numSteps,
      sigmaTable: discreteFlowSigmaTable(shift: shift, numTrainTimesteps: numTrainTimesteps),
      alpha: alpha, betaParam: betaParam)
  }

  // MARK: Discrete sigma tables

  /// ComfyUI `ModelSamplingDiscreteFlow.set_parameters` (`model_sampling.py:298-302`):
  /// `σ[i] = time_snr_shift(shift, (i+1)/T) = shift·t / (1 + (shift − 1)·t)`,
  /// ascending, `σ[T−1] = 1.0` exactly. `shift == 1` is the identity (`σ = t`),
  /// as upstream special-cases it. Float32 arithmetic mirrors the torch tensor.
  public static func discreteFlowSigmaTable(shift: Float, numTrainTimesteps: Int) -> [Float] {
    precondition(numTrainTimesteps > 0, "numTrainTimesteps must be positive")
    let count = Float(numTrainTimesteps)
    return (0..<numTrainTimesteps).map { i in
      let t = Float(i + 1) / count
      if shift == 1.0 { return t }
      return shift * t / (1 + (shift - 1) * t)
    }
  }

  /// ComfyUI `ModelSamplingFlux.set_parameters` (`model_sampling.py:416-419`):
  /// `σ[i] = flux_time_shift(shift, 1.0, (i+1)/N) = e^shift / (e^shift + 1/t − 1)`,
  /// a 10 000-entry table by default. This is the class ComfyUI actually
  /// registers Krea 2 under (`supported_models.py`, E18's finding recorded in
  /// `comfy_sigmas.json`), where `shift` is a **log**-shift — the same
  /// parameterisation as `SigmaSchedule.krea2`'s `mu`, which is why the wire's
  /// `shift` for Krea 2 IS mu (FDD Addendum A.1). `SchedulerFactory` builds
  /// this table for `.flux` configs from the render's `mu`.
  public static func fluxSigmaTable(shift: Float, tableSize: Int = 10000) -> [Float] {
    precondition(tableSize > 0, "tableSize must be positive")
    let count = Float(tableSize)
    let expShift = Float(Foundation.exp(Double(shift)))
    return (0..<tableSize).map { i in
      let t = Float(i + 1) / count
      return expShift / (expShift + (1 / t - 1))
    }
  }

  // MARK: Beta distribution

  /// Inverse of the regularized incomplete beta function — `scipy.stats.beta.ppf`.
  ///
  /// Bisection in `Double` on ``regularizedIncompleteBeta(_:a:b:)`` to machine
  /// precision (~1e-15 in `x`). The symmetric midpoint `p = 0.5, a == b` is
  /// returned exactly (`I_0.5(a, a) = 0.5` by symmetry), so `rint(ppf·T)` lands
  /// on the same half-to-even tie as upstream instead of an ulp either side.
  ///
  /// Accuracy matters here: the FDD's proposed bisection on the old midpoint-rule
  /// CDF (10 000 bins) mis-rounds 2 of 6 indices at the AC-21 case and up to 17
  /// indices on the 10 000-entry flux table — the `rint` quantisation does not
  /// absorb a ~1e-3 CDF error (checked against scipy, 2026-08-22).
  static func betaPPF(_ p: Double, alpha a: Double, beta b: Double) -> Double {
    precondition(a > 0 && b > 0, "beta shape parameters must be positive")
    if p <= 0 { return 0 }
    if p >= 1 { return 1 }
    if a == b && p == 0.5 { return 0.5 }
    var lo = 0.0, hi = 1.0
    for _ in 0..<200 {
      let mid = 0.5 * (lo + hi)
      if mid == lo || mid == hi { break }  // Double exhausted
      if regularizedIncompleteBeta(mid, a: a, b: b) < p { lo = mid } else { hi = mid }
    }
    return 0.5 * (lo + hi)
  }

  /// Regularized incomplete beta function `I_x(a, b)` via Lentz's continued
  /// fraction (Numerical Recipes `betai`/`betacf`), with the `x > (a+1)/(a+b+2)`
  /// symmetry swap so the fraction always converges fast. Relative accuracy
  /// ~1e-14 for the shape range used here (`a, b ∈ [0.5, 0.7]`).
  static func regularizedIncompleteBeta(_ x: Double, a: Double, b: Double) -> Double {
    if x <= 0 { return 0 }
    if x >= 1 { return 1 }
    // Prefactor x^a (1−x)^b / (a B(a,b)), in log space.
    let lnBeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    let front = Foundation.exp(a * Foundation.log(x) + b * Foundation.log(1 - x) - lnBeta)
    if x < (a + 1) / (a + b + 2) {
      return front * betaContinuedFraction(x, a: a, b: b) / a
    } else {
      return 1 - front * betaContinuedFraction(1 - x, a: b, b: a) / b
    }
  }

  /// `betacf` — modified Lentz evaluation of the continued fraction for `I_x(a, b)`.
  private static func betaContinuedFraction(_ x: Double, a: Double, b: Double) -> Double {
    let tiny = 1e-300
    let eps = 1e-16
    let qab = a + b, qap = a + 1, qam = a - 1
    var c = 1.0
    var d = 1 - qab * x / qap
    if abs(d) < tiny { d = tiny }
    d = 1 / d
    var h = d
    for m in 1...300 {
      let mD = Double(m), m2 = 2 * mD
      // Even step.
      var aa = mD * (b - mD) * x / ((qam + m2) * (a + m2))
      d = 1 + aa * d
      if abs(d) < tiny { d = tiny }
      c = 1 + aa / c
      if abs(c) < tiny { c = tiny }
      d = 1 / d
      h *= d * c
      // Odd step.
      aa = -(a + mD) * (qab + mD) * x / ((a + m2) * (qap + m2))
      d = 1 + aa * d
      if abs(d) < tiny { d = tiny }
      c = 1 + aa / c
      if abs(c) < tiny { c = tiny }
      d = 1 / d
      let del = d * c
      h *= del
      if abs(del - 1) < eps { break }
    }
    return h
  }

  // MARK: - Helpers

  static func linspace(_ start: Float, _ end: Float, count: Int) -> [Float] {
    guard count > 1 else { return [start] }
    let step = (end - start) / Float(count - 1)
    return (0..<count).map { idx in start + Float(idx) * step }
  }
}
