import Foundation
import MLX
import MLXRandom

/// Identifies a sampler algorithm.
public enum SchedulerKind: String, CaseIterable, Sendable {
  case euler = "euler"
  case heun = "heun"
  case dpmplusplus2m = "dpmpp-2m"
  case dpmplusplus2sa = "dpmpp-2s-a"
  case deis = "deis"
  case ddim = "ddim"
  case res2s = "res_2s"
  // WP-E13 (§3.12, D20): the N-row tableau conformers. `ralston_*` are also
  // the DEIS warm-up samplers (AC-24); `res_3s` is the exponential 3-row.
  case ralston2s = "ralston_2s"
  case ralston3s = "ralston_3s"
  case ralston4s = "ralston_4s"
  case res3s = "res_3s"

  /// Whether this kind builds a ``TableauScheduler`` — an N-row conformer that
  /// takes `rows` model evaluations per step and can ONLY be driven by a loop
  /// that dispatches `rowSigma` / `rowSample` / `commit`.
  ///
  /// Today that is `Krea2DenoiseLoop` and nothing else: the Z-Image pipelines
  /// drive `step` directly, which for these samplers is a hard failure rather
  /// than a quiet first-order Euler (WP-E13 review finding 1). Callers on a
  /// non-Krea 2 path must refuse the name up front — see
  /// `GeneratePayload.validateTableauSampler(_:family:)` and the CLI's
  /// `--scheduler` parsing.
  ///
  /// Exhaustive on purpose (no `default`): a new kind must declare itself.
  public var isNRowTableau: Bool {
    switch self {
    case .ralston2s, .ralston3s, .ralston4s, .res3s:
      return true
    case .euler, .heun, .dpmplusplus2m, .dpmplusplus2sa, .deis, .ddim, .res2s:
      return false
    }
  }

  /// Whether this sampler is a port of a RES4LYF sampler and therefore obeys
  /// RES4LYF's `prepare_sigmas` / model-free-tail contract
  /// (``RES4LYFSigmaPreparation``): the schedule's trailing zero is a sentinel,
  /// the last solver step lands on the model's `sigma_min`, and `σ_min → 0` is
  /// a model-free conversion.
  ///
  /// `.deis` is deliberately **false** today: the shipped `DEISScheduler` is
  /// the k-diffusion multistep port, not RES4LYF's `deis_*`. WP-E14 replaces
  /// it with a RES4LYF conformer and flips this case with it.
  ///
  /// Exhaustive on purpose (no `default`): a new kind must declare itself.
  public var isRES4LYFFamily: Bool {
    switch self {
    case .res2s, .res3s, .ralston2s, .ralston3s, .ralston4s:
      return true
    case .euler, .heun, .dpmplusplus2m, .dpmplusplus2sa, .deis, .ddim:
      return false
    }
  }
}

/// Errors raised while resolving a sigma schedule.
public enum SchedulerFactoryError: Error, Equatable, CustomStringConvertible {
  /// The schedule is defined by the model's resolution shift and no `mu` was
  /// supplied. There is no default: `mu = 0` would be an unshifted linear grid
  /// presented as the model's own schedule (FDD-krea2-raw-recipe §3.1). Raised
  /// for `.krea2`, and — under a `.flux` model sampling (Krea 2) — for every
  /// table-backed schedule (`beta`, `beta57`, `karras`, `exponential`), whose
  /// Flux table is built from `mu` (Addendum A.1).
  case missingMu(SigmaScheduleKind)
  /// The schedule is undefined below `minimum` steps and the request asked for
  /// fewer. `bong_tangent` divides by zero upstream at 0 and 1 steps; refusing
  /// by name is the only alternative to a NaN grid (FDD-krea2-raw-recipe §3.11).
  case stepCountBelowMinimum(SigmaScheduleKind, steps: Int, minimum: Int)

  public var description: String {
    switch self {
    case .missingMu(let schedule):
      return "sigma schedule '\(schedule.rawValue)' requires mu (the model's resolution shift) and none was supplied"
    case .stepCountBelowMinimum(let schedule, let steps, let minimum):
      return "sigma schedule '\(schedule.rawValue)' needs at least \(minimum) steps; \(steps) requested"
    }
  }
}

/// Creates scheduler instances by kind.
public enum SchedulerFactory {

  /// Create a scheduler with the given configuration.
  ///
  /// - Parameters:
  ///   - kind: The sampler algorithm.
  ///   - sigmaSchedule: The sigma schedule to use. When `kind == .euler` and
  ///     the schedule is `.flow`, the model's native schedule is used for
  ///     exact backward compatibility.
  ///   - numInferenceSteps: Number of denoising steps.
  ///   - config: The model's scheduler configuration.
  ///   - mu: Dynamic shifting parameter (pass non-nil when `config.useDynamicShifting`).
  ///     **Required** for `.krea2`, which is defined by it; the factory throws
  ///     ``SchedulerFactoryError/missingMu(_:)`` rather than defaulting it.
  ///   - seed: Random seed for stochastic samplers (DPM++ 2S-A, DDIM with eta > 0).
  ///   - eta: DDIM stochasticity parameter (0 = deterministic, 1 = full DDPM).
  ///   - c2: `res_2s` / `res_3s` substep location in log-sigma space (D23).
  ///   - res4lyfSigmaPreparation: run RES4LYF's `prepare_sigmas` on the grid
  ///     when `kind.isRES4LYFFamily` — insert or replace the model's
  ///     `sigma_min` before the trailing zero, stop the solver there, and hand
  ///     the scheduler the model-free `σ_min → 0` conversion sigma
  ///     (``RES4LYFSigmaPreparation``). **Opt-in**: only the Krea 2 family
  ///     passes `true` (`Krea2Pipeline.makeScheduler`). The Z-Image pipelines
  ///     leave it `false`, so their `res_2s` behaviour is unchanged by this
  ///     seam. A no-op for every non-RES4LYF kind whatever the flag says.
  /// - Returns: A type-erased ``ZImageScheduler``.
  /// - Throws: ``SchedulerFactoryError`` when the schedule cannot be resolved,
  ///   including ``SchedulerFactoryError/missingMu(_:)`` when preparation is
  ///   asked for on a `.flux` model sampling with no `mu` — the Flux table has
  ///   no `sigma_min` without its shift, and `1e-8` is not a substitute.
  public static func create(
    kind: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind = .flow,
    numInferenceSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float? = nil,
    seed: UInt64? = nil,
    eta: Float? = nil,
    c2: Float = 0.5,
    res4lyfSigmaPreparation: Bool = false
  ) throws -> any ZImageScheduler {
    let published = try resolveSigmas(
      schedule: sigmaSchedule,
      numSteps: numInferenceSteps,
      config: config,
      mu: mu
    )

    // RES4LYF's `prepare_sigmas`, before anything reads a count. The published
    // grid ends at an exact 0; RES4LYF never solves through that zero (see
    // `RES4LYFSigmaPreparation`). Engaged only for the RES4LYF family, so the
    // default `.euler` + `.krea2` grid is `published` element for element.
    let sigmaValues: [Float]
    let finalConversionSigma: Float?
    if res4lyfSigmaPreparation && kind.isRES4LYFFamily {
      let prepared = RES4LYFSigmaPreparation.prepare(
        published: published,
        sigmaMin: try modelSigmaMin(schedule: sigmaSchedule, config: config, mu: mu))
      sigmaValues = prepared.solverSigmas
      finalConversionSigma = prepared.finalConversionSigma
    } else {
      sigmaValues = published
      finalConversionSigma = nil
    }

    // The produced count is authoritative: ComfyUI-style de-duplication
    // (`beta`/`beta57`) can return fewer than `numInferenceSteps + 1` sigmas,
    // and every scheduler init preconditions `sigmaValues.count == steps + 1`.
    // The scheduler's `numInferenceSteps` then reports the effective count
    // (FDD-krea2-raw-recipe D5, AC-22) — callers loop over it, not the request.
    // RES4LYF preparation moves this the same way, and by upstream's own rule:
    // `num_steps = len(sigmas) - 2` once the trailing zero is dropped.
    let effectiveSteps = sigmaValues.count - 1

    switch kind {
    case .euler:
      if sigmaSchedule == .flow {
        // Use the native flow-match init for exact backward compatibility.
        return FlowMatchEulerScheduler(
          numInferenceSteps: effectiveSteps,
          config: config,
          mu: config.useDynamicShifting ? mu : nil
        )
      }
      return FlowMatchEulerScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .dpmplusplus2m:
      return DPMPlusPlus2MScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .ddim:
      let randomKey = seed.map { MLXRandom.key($0) }
      return DDIMScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        eta: eta ?? 0.0,
        randomKey: randomKey
      )

    case .deis:
      return DEISScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .dpmplusplus2sa:
      let randomKey = seed.map { MLXRandom.key($0) }
      return DPMPlusPlus2SAScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        eta: eta ?? 1.0,
        randomKey: randomKey
      )

    case .heun:
      return HeunScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .res2s:
      return RES2sScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        c2: c2,
        finalConversionSigma: finalConversionSigma
      )

    // WP-E13: the N-row tableau conformers. Both families take the data
    // prediction (RES4LYF anchors every row at the step's x₀ and sigma), and
    // both are dispatched by `Krea2DenoiseLoop` through `TableauScheduler`.
    case .ralston2s, .ralston3s, .ralston4s:
      let stages: RalstonScheduler.Stages =
        kind == .ralston2s ? .two : (kind == .ralston3s ? .three : .four)
      return RalstonScheduler(
        stages: stages,
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        finalConversionSigma: finalConversionSigma
      )

    case .res3s:
      return RES3sScheduler(
        numInferenceSteps: effectiveSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        c2: c2,
        finalConversionSigma: finalConversionSigma
      )
    }
  }

  // MARK: - Sigma Resolution

  /// Resolve the float32 sigma grid for a schedule. Pure; internal so the
  /// schedule tests can pin grids without constructing a scheduler.
  static func resolveSigmas(
    schedule: SigmaScheduleKind,
    numSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float?
  ) throws -> [Float] {
    switch schedule {
    case .flow:
      return SigmaSchedule.flow(numSteps: numSteps, config: config, mu: mu)
    case .krea2:
      // mu is required: `mu ?? 0` would silently be an unshifted linear grid.
      guard let mu else { throw SchedulerFactoryError.missingMu(.krea2) }
      return SigmaSchedule.krea2(numSteps: numSteps, mu: mu)
    case .bongTangent:
      // Model-free by construction (D6): upstream takes `model_sampling` and
      // never reads it, so `config` and `mu` are deliberately ignored here —
      // Krea 2's resolution shift is NOT composed on top.
      guard numSteps >= SigmaSchedule.bongTangentMinimumSteps else {
        throw SchedulerFactoryError.stepCountBelowMinimum(
          .bongTangent, steps: numSteps, minimum: SigmaSchedule.bongTangentMinimumSteps)
      }
      return SigmaSchedule.bongTangent(numSteps: numSteps)
    case .karras:
      let (lo, hi) = try flowMatchingSigmaBounds(schedule: .karras, config: config, mu: mu)
      return SigmaSchedule.karras(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
    case .exponential:
      let (lo, hi) = try flowMatchingSigmaBounds(schedule: .exponential, config: config, mu: mu)
      return SigmaSchedule.exponential(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
    case .beta:
      // ComfyUI-exact (D5): beta PPF → rint → index into the model's discrete
      // sigma table, de-duplicated. Which table is the family's (A.1):
      // DiscreteFlow from config.shift / numTrainTimesteps, or Flux from mu.
      // May return fewer than numSteps + 1 sigmas; `create` constructs with
      // the produced count (AC-22).
      return SigmaSchedule.beta(
        numSteps: numSteps, sigmaTable: try sigmaTable(schedule: .beta, config: config, mu: mu))
    case .beta57:
      return SigmaSchedule.beta(
        numSteps: numSteps, sigmaTable: try sigmaTable(schedule: .beta57, config: config, mu: mu),
        alpha: 0.5, betaParam: 0.7)
    }
  }

  /// ComfyUI's `model_sampling.sigma_min` — the first (smallest) entry of the
  /// ACTIVE model-sampling table, which is what RES4LYF's `prepare_sigmas`
  /// inserts before the trailing zero (`SIGMA_MIN` in
  /// `rk_noise_sampler_beta.py`).
  ///
  /// Under `.flux` (Krea 2) the table is built from `mu`, so this is
  /// `e^mu / (e^mu + N − 1)` — `3.1575e-4` at Krea 2's registered 1.15 with
  /// `N = 10 000` (Addendum A.1) — and `mu` is **required**: without it there
  /// is no table and no `sigma_min`, and the `1e-8` frame floor the solver
  /// falls back to is precisely the bug this preparation exists to fix. Under
  /// `.discreteFlow` it is the shifted `1/numTrainTimesteps`, `mu`-free.
  static func modelSigmaMin(
    schedule: SigmaScheduleKind,
    config: ZImageSchedulerConfig,
    mu: Float?
  ) throws -> Float {
    let table = try sigmaTable(schedule: schedule, config: config, mu: mu)
    return table[0]
  }

  /// The model's discrete sigma table, ascending — ComfyUI's
  /// `model_sampling.sigmas` — for the family in `config.modelSampling`.
  ///
  /// `.discreteFlow` builds from the config's linear `shift` and
  /// `numTrainTimesteps` and never reads `mu`; `.flux` builds from `mu` and
  /// **requires** it — a Flux table without its shift has no default that is
  /// not a silent grid (FDD-krea2-raw-recipe Addendum A.1).
  static func sigmaTable(
    schedule: SigmaScheduleKind,
    config: ZImageSchedulerConfig,
    mu: Float?
  ) throws -> [Float] {
    switch config.modelSampling {
    case .discreteFlow:
      return SigmaSchedule.discreteFlowSigmaTable(
        shift: config.shift, numTrainTimesteps: config.numTrainTimesteps)
    case .flux(let tableSize):
      guard let mu else { throw SchedulerFactoryError.missingMu(schedule) }
      return SigmaSchedule.fluxSigmaTable(shift: mu, tableSize: tableSize)
    }
  }

  /// Derive sigma bounds that match the flow-matching model's native range —
  /// ComfyUI's `model_sampling.sigma_min` / `sigma_max`.
  ///
  /// Z-Image Turbo uses velocity-prediction flow matching where sigmas
  /// represent noise fraction in [0, 1]. Alternative schedules (karras,
  /// exponential) redistribute points within this range instead of using
  /// their DDPM defaults (0.02..100). Under `.discreteFlow` the floor is the
  /// shifted `1/numTrainTimesteps` (unchanged, `mu`-free); under `.flux` it is
  /// the Flux table's first entry, `e^mu / (e^mu + tableSize − 1)`, so the
  /// bounds move with `mu` exactly as `beta`/`beta57` do (Addendum A.1).
  /// `beta`/`beta57` take no bounds: they index the same table.
  private static func flowMatchingSigmaBounds(
    schedule: SigmaScheduleKind,
    config: ZImageSchedulerConfig,
    mu: Float?
  ) throws -> (sigmaMin: Float, sigmaMax: Float) {
    switch config.modelSampling {
    case .discreteFlow:
      let numTrainTimesteps = Float(config.numTrainTimesteps)
      let shift = config.shift
      let initSigmaMin = 1.0 / numTrainTimesteps
      let shiftedSigmaMin = shift * initSigmaMin / (1 + (shift - 1) * initSigmaMin)
      return (shiftedSigmaMin, 1.0)
    case .flux:
      let table = try sigmaTable(schedule: schedule, config: config, mu: mu)
      return (table[0], table[table.count - 1])
    }
  }
}
