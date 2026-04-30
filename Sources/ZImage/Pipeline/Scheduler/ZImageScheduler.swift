import MLX

/// A scheduler that advances latent samples through a denoising diffusion process.
///
/// Schedulers implement the ODE/SDE integration step for flow-matching models.
/// Single-evaluation schedulers (Euler, DDIM, DPM++ 2M) return `nil` from
/// ``intermediateStep(modelOutput:timestepIndex:sample:)``, while
/// multi-evaluation schedulers (Heun, RES) return an intermediate latent
/// that requires a second model forward pass.
public protocol ZImageScheduler {
  /// Sigma values for each step plus a trailing zero: shape `[numInferenceSteps + 1]`.
  var sigmas: MLXArray { get }

  /// Timestep values for each step: shape `[numInferenceSteps]`.
  var timesteps: MLXArray { get }

  /// Number of inference steps configured for this scheduler.
  var numInferenceSteps: Int { get }

  /// Whether this scheduler requires two model evaluations per step.
  var requiresIntermediateEvaluation: Bool { get }

  /// Perform a single denoising step.
  ///
  /// For single-evaluation schedulers this is the only call per step.
  /// For multi-evaluation schedulers, call ``intermediateStep(modelOutput:timestepIndex:sample:)``
  /// first, evaluate the model at the intermediate point, then call
  /// ``finalizeStep(originalOutput:intermediateOutput:timestepIndex:sample:)``
  /// with both predictions.
  ///
  /// - Parameters:
  ///   - modelOutput: The model's velocity prediction for the current sample.
  ///   - timestepIndex: The current step index in `[0, numInferenceSteps)`.
  ///   - sample: The current latent sample.
  /// - Returns: The updated latent sample.
  mutating func step(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray

  /// For multi-evaluation schedulers: compute the intermediate sample.
  ///
  /// Returns `nil` for single-evaluation schedulers. When non-nil,
  /// the caller must evaluate the model at the returned sample and
  /// then call ``finalizeStep(originalOutput:intermediateOutput:timestepIndex:sample:)``
  /// with both predictions.
  mutating func intermediateStep(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray?

  /// Finalize a multi-evaluation step using predictions at both
  /// the original and intermediate points.
  ///
  /// Only called when ``intermediateStep(modelOutput:timestepIndex:sample:)``
  /// returned non-nil.
  mutating func finalizeStep(
    originalOutput: MLXArray,
    intermediateOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray

  /// Reset any internal state (cached predictions, step counters).
  /// Call between generations when reusing a scheduler instance.
  mutating func reset()
}

// MARK: - Default implementations for single-evaluation schedulers

public extension ZImageScheduler {
  var requiresIntermediateEvaluation: Bool { false }

  mutating func intermediateStep(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray? {
    nil
  }

  mutating func finalizeStep(
    originalOutput: MLXArray,
    intermediateOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray {
    // Should never be called for single-eval schedulers.
    // Fall back to regular step with the original output.
    step(modelOutput: originalOutput, timestepIndex: timestepIndex, sample: sample)
  }

  mutating func reset() {}
}
