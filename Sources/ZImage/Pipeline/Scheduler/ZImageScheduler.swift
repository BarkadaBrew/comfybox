import MLX

/// The quantity a scheduler expects in `modelOutput`.
///
/// The pipelines always obtain a flow **velocity** `v = dx/dσ = ε − x₀` from
/// the transformer (for Z-Image the raw prediction is `x₀ − ε`, negated at
/// the call site) and convert it once per evaluation, after CFG combination,
/// through ``ZImageScheduler/modelInput(velocity:sample:sigma:)``. A solver
/// that integrates in the wrong quantity is dimensionally wrong, not merely
/// inaccurate — the `res_2s` correction in FDD-krea2-raw-recipe D2/§3.2.
public enum ModelOutputConvention: Sendable, Equatable {
  /// Linear-frame solvers integrate `dx/dσ = v` and take the velocity as-is.
  case velocity
  /// Exponential-frame solvers (`res_*`) integrate in `h = −log(σ'/σ)` and
  /// take the data prediction `x₀ = x − σ·v`.
  case dataPrediction
}

/// A scheduler that advances latent samples through a denoising diffusion process.
///
/// Schedulers implement the ODE/SDE integration step for flow-matching models.
/// Single-evaluation schedulers (Euler, DDIM, DPM++ 2M) return `nil` from
/// ``intermediateStep(modelOutput:timestepIndex:sample:)``, while
/// multi-evaluation schedulers (Heun, RES) return an intermediate latent
/// that requires a second model forward pass.
///
/// Every `modelOutput` argument below carries the quantity named by
/// ``modelOutputConvention`` — a velocity for every scheduler except the
/// exponential-frame RES family, which takes the data prediction.
public protocol ZImageScheduler {
  /// What quantity `step` / `intermediateStep` / `finalizeStep` expect in
  /// `modelOutput`. Linear-frame solvers integrate `dx/dσ = v` and take a
  /// velocity. Exponential-frame solvers (`res_2s`, later `res_3s`) integrate
  /// in `h = −log(σ'/σ)` and take the data prediction `x₀ = x − σ·v`.
  var modelOutputConvention: ModelOutputConvention { get }

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
  ///   - modelOutput: The model's prediction for the current sample, in the
  ///     quantity named by ``modelOutputConvention`` (velocity by default).
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

  /// Sigma value for the intermediate model evaluation.
  ///
  /// Multi-evaluation schedulers may evaluate the second stage at a
  /// substep between the current and next sigma. The default matches
  /// Heun's existing next-sigma intermediate point.
  func intermediateSigma(timestepIndex: Int) -> Float?

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
  /// Every existing scheduler takes a velocity; only the RES family overrides.
  var modelOutputConvention: ModelOutputConvention { .velocity }

  /// Convert the transformer's flow velocity into what this scheduler expects
  /// in `modelOutput`. Call once per model evaluation, **after** CFG
  /// combination (CFG is linear in `v`, so the order is safe), with the sigma
  /// the evaluation was made at (the grid sigma for the first evaluation,
  /// ``intermediateSigma(timestepIndex:)`` for the second).
  ///
  /// For `.velocity` this returns `velocity` itself — no new ops, so the
  /// default euler/flow path is byte-identical (AC-6). For `.dataPrediction`
  /// it returns `x₀ = sample − σ·velocity` in the sample's dtype (`Float *
  /// MLXArray` casts the scalar to the array's dtype).
  func modelInput(velocity: MLXArray, sample: MLXArray, sigma: Float) -> MLXArray {
    switch modelOutputConvention {
    case .velocity:
      return velocity
    case .dataPrediction:
      return sample - sigma * velocity
    }
  }

  var requiresIntermediateEvaluation: Bool { false }

  mutating func intermediateStep(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray? {
    nil
  }

  func intermediateSigma(timestepIndex: Int) -> Float? {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
      "invalid timestep index"
    )
    return sigmas[timestepIndex + 1].item(Float.self)
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

// MARK: - N-row protocol (WP-E3, FDD-krea2-raw-recipe D1 / §3.3)

/// A scheduler that takes `rows` model evaluations per step in explicit
/// Runge–Kutta form — `ralston_2s/3s/4s`, `res_3s`, the DEIS warm-up
/// (WP-E13/E14). Declared with the driver (D1) so the byte-identity gate is
/// run once: `Krea2DenoiseLoop` dispatches a `TableauScheduler` through
/// `rowSigma` / `rowSample` / `commit` and never through `step`.
///
/// Row 0 is always evaluated at the grid sigma on the step's start sample;
/// the driver does not call `rowSample` for it. Every `k[r]` handed back is
/// the model output **after** conversion through
/// ``ZImageScheduler/modelInput(velocity:sample:sigma:)`` at that row's
/// sigma, so a `.dataPrediction` tableau receives x₀ rows and a `.velocity`
/// tableau receives velocity rows — the same discipline as the 1- and 2-row
/// branches.
public protocol TableauScheduler: ZImageScheduler {
  /// Model evaluations per step. `modelEvals = stepsRun × rows × (CFG ? 2 : 1)`
  /// is reported, never discovered (§3.3).
  var rows: Int { get }

  /// The sigma row `row` of step `timestepIndex` is evaluated at. Row 0 is
  /// `sigmas[timestepIndex]`.
  func rowSigma(timestepIndex: Int, row: Int) -> Float

  /// The sample row `row` (≥ 1) is evaluated on, from the step's start
  /// sample `x0` and the converted outputs of rows `0 ..< row`.
  mutating func rowSample(timestepIndex: Int, row: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray

  /// Commit the step from the start sample and all `rows` converted outputs.
  mutating func commit(timestepIndex: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray
}
