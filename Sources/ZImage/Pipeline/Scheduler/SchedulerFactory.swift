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
}

/// Errors raised while resolving a sigma schedule.
public enum SchedulerFactoryError: Error, Equatable, CustomStringConvertible {
  /// The schedule is defined by the model's resolution shift and no `mu` was
  /// supplied. There is no default: `mu = 0` would be an unshifted linear grid
  /// presented as the model's own schedule (FDD-krea2-raw-recipe §3.1).
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
  ///   - c2: RES 2s second-stage substep location in log-sigma space.
  /// - Returns: A type-erased ``ZImageScheduler``.
  /// - Throws: ``SchedulerFactoryError`` when the schedule cannot be resolved.
  public static func create(
    kind: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind = .flow,
    numInferenceSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float? = nil,
    seed: UInt64? = nil,
    eta: Float? = nil,
    c2: Float = 0.5
  ) throws -> any ZImageScheduler {
    let sigmaValues = try resolveSigmas(
      schedule: sigmaSchedule,
      numSteps: numInferenceSteps,
      config: config,
      mu: mu
    )

    switch kind {
    case .euler:
      if sigmaSchedule == .flow {
        // Use the native flow-match init for exact backward compatibility.
        return FlowMatchEulerScheduler(
          numInferenceSteps: numInferenceSteps,
          config: config,
          mu: config.useDynamicShifting ? mu : nil
        )
      }
      return FlowMatchEulerScheduler(
        numInferenceSteps: numInferenceSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .dpmplusplus2m:
      return DPMPlusPlus2MScheduler(
        numInferenceSteps: numInferenceSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .ddim:
      let randomKey = seed.map { MLXRandom.key($0) }
      return DDIMScheduler(
        numInferenceSteps: numInferenceSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        eta: eta ?? 0.0,
        randomKey: randomKey
      )

    case .deis:
      return DEISScheduler(
        numInferenceSteps: numInferenceSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .dpmplusplus2sa:
      let randomKey = seed.map { MLXRandom.key($0) }
      return DPMPlusPlus2SAScheduler(
        numInferenceSteps: numInferenceSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        eta: eta ?? 1.0,
        randomKey: randomKey
      )

    case .heun:
      return HeunScheduler(
        numInferenceSteps: numInferenceSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps
      )

    case .res2s:
      return RES2sScheduler(
        numInferenceSteps: numInferenceSteps,
        sigmaValues: sigmaValues,
        numTrainTimesteps: config.numTrainTimesteps,
        c2: c2
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
      let (lo, hi) = flowMatchingSigmaBounds(config: config)
      return SigmaSchedule.karras(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
    case .exponential:
      let (lo, hi) = flowMatchingSigmaBounds(config: config)
      return SigmaSchedule.exponential(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
    case .beta:
      let (lo, hi) = flowMatchingSigmaBounds(config: config)
      return SigmaSchedule.beta(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
    case .beta57:
      let (lo, hi) = flowMatchingSigmaBounds(config: config)
      return SigmaSchedule.beta(
        numSteps: numSteps,
        sigmaMin: lo,
        sigmaMax: hi,
        alpha: 0.5,
        betaParam: 0.7
      )
    }
  }

  /// Derive sigma bounds that match the flow-matching model's native range.
  ///
  /// Z-Image Turbo uses velocity-prediction flow matching where sigmas
  /// represent noise fraction in [0, 1]. Alternative schedules (karras,
  /// exponential, beta) redistribute points within this range instead of
  /// using their DDPM defaults (0.02..100).
  private static func flowMatchingSigmaBounds(
    config: ZImageSchedulerConfig
  ) -> (sigmaMin: Float, sigmaMax: Float) {
    let numTrainTimesteps = Float(config.numTrainTimesteps)
    let shift = config.shift
    let initSigmaMin = 1.0 / numTrainTimesteps
    let shiftedSigmaMin = shift * initSigmaMin / (1 + (shift - 1) * initSigmaMin)
    return (shiftedSigmaMin, 1.0)
  }
}
