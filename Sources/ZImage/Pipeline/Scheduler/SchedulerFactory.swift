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
  ///   - seed: Random seed for stochastic samplers (DPM++ 2S-A, DDIM with eta > 0).
  ///   - eta: DDIM stochasticity parameter (0 = deterministic, 1 = full DDPM).
  /// - Returns: A type-erased ``ZImageScheduler``.
  public static func create(
    kind: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind = .flow,
    numInferenceSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float? = nil,
    seed: UInt64? = nil,
    eta: Float? = nil
  ) -> any ZImageScheduler {
    let sigmaValues = resolveSigmas(
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
    }
  }

  // MARK: - Sigma Resolution

  private static func resolveSigmas(
    schedule: SigmaScheduleKind,
    numSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float?
  ) -> [Float] {
    switch schedule {
    case .flow:
      return SigmaSchedule.flow(numSteps: numSteps, config: config, mu: mu)
    case .karras:
      let (lo, hi) = flowMatchingSigmaBounds(config: config)
      return SigmaSchedule.karras(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
    case .exponential:
      let (lo, hi) = flowMatchingSigmaBounds(config: config)
      return SigmaSchedule.exponential(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
    case .beta:
      let (lo, hi) = flowMatchingSigmaBounds(config: config)
      return SigmaSchedule.beta(numSteps: numSteps, sigmaMin: lo, sigmaMax: hi)
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
