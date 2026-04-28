import Foundation
import MLX

/// Identifies a sampler algorithm.
public enum SchedulerKind: String, CaseIterable, Sendable {
  case euler = "euler"
  // Phase 2+
  // case heun = "heun"
  // case dpmplusplus2m = "dpmpp-2m"
  // case dpmplusplus2sa = "dpmpp-2s-a"
  // case deis = "deis"
  // case ddim = "ddim"
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
  ///   - seed: Random seed for stochastic samplers (reserved for Phase 2).
  ///   - eta: DDIM stochasticity parameter (reserved for Phase 2).
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
      return SigmaSchedule.karras(numSteps: numSteps)
    case .exponential:
      return SigmaSchedule.exponential(numSteps: numSteps)
    case .beta:
      return SigmaSchedule.beta(numSteps: numSteps)
    }
  }
}
