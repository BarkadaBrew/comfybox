# Implementation Plan: Multi-Sampler Support for ZImage CLI

**Author:** Claude (code review + implementation planning)
**Date:** 2026-04-28
**Status:** Ready for implementation
**Companion:** `docs/prd-sampler-expansion.md`

---

## Part 1: PRD Review

### Issues and Gaps Found

**1. Protocol naming collision.** The PRD names the protocol `FlowMatchScheduler`, which collides conceptually with `FlowMatchEulerScheduler`. The protocol should be called `Scheduler` or `ZImageScheduler` to clearly separate the abstraction from any single implementation. The existing struct can keep its name since it is the Euler variant. I recommend `ZImageScheduler` since the project already prefixes public types with `ZImage`.

**2. Sigma schedule decoupling is under-specified for Phase 1.** The PRD lists sigma schedules as a Phase 5 deliverable but places the `SigmaScheduleKind` enum and factory parameter in Phase 1. This creates dead code: the factory would accept a `sigmaSchedule` parameter but only the `flow` schedule would exist. Either:
- (a) Ship the Karras, Exponential, and Beta schedules in Phase 1 alongside the protocol (they are each 15-25 LOC in Swift), or
- (b) Remove the `sigmaSchedule` parameter from the Phase 1 factory and add it in Phase 5.

**Recommendation:** Option (a). The sigma schedules are pure math with no model dependency, trivially testable, and several Phase 2 samplers (DEIS, DDIM, DPM++ 2M) benefit significantly from Karras sigmas. The chroma-generate `SigmaSchedule` class is a clean port target at ~30 LOC per schedule.

**3. Flow-matching vs. noise-prediction parameterization mismatch in DDIM formula.** The PRD's Section 4 DDIM formula uses noise-prediction DDPM math (`alpha_t = 1 - sigma_t^2`, predicting `x_0` from noise). Z-Image Turbo uses velocity-prediction flow-matching where the model outputs a velocity `v` and the ODE is `dx/dt = v`. The DDIM port needs to adapt to this parameterization or it will produce garbage. The chroma-generate `ChromaDDIMScheduler` already handles this (see `alpha_t = 1 - t` in the Python reference), but the PRD formula does not match.

**4. Heun and RES samplers require multiple model evaluations per step.** The PRD acknowledges this in the risk table but does not address the pipeline interface change. Currently `generateCore()` runs a simple loop: one model forward pass per step, then one scheduler step. Heun needs two forward passes per step (predict, step-to-midpoint, predict-at-midpoint, average). This requires either:
- (a) A callback/closure passed to the scheduler's step function (chroma-generate's `model_fn` approach), or
- (b) A multi-phase step protocol where the scheduler returns an intermediate sample and requests another evaluation.

**Recommendation:** Option (b) -- a two-phase protocol. Add an optional `intermediateStep()` method to the protocol that returns `nil` for single-evaluation samplers (Euler, DPM++ 2M, DDIM, DEIS, IPNDM) and returns an intermediate latent for Heun/RES. The pipeline loop checks the return and performs an extra forward pass when needed. This avoids passing the model as a closure and keeps the scheduler pure.

**5. DPM++ 2S Ancestral needs a random key.** The PRD mentions `seed: UInt64?` in the factory but does not specify how per-step randomness is managed. mlx-swift uses `MLXRandom.key()` to produce deterministic sequences. The factory should derive a scheduler-specific random key from the generation seed so that ancestral samplers are reproducible.

**6. No mention of ZImageControlPipeline.** The control pipeline (`ZImageControlPipeline.swift`) constructs `FlowMatchEulerScheduler` in two places (lines 1005 and 1243). The refactor must update both pipelines or extract a shared scheduler construction path.

**7. Success criterion 1 (byte-identical output) is fragile.** The flow-match sigma schedule currently uses Float arithmetic in a `[Float]` array, then converts to `MLXArray`. Refactoring this to an `MLXArray`-native computation (which is the right approach for the sigma schedule abstraction) may introduce floating-point ordering differences. Recommend relaxing to "within 1e-5 tolerance on sigma values" or explicitly preserving the current Float-array path for the `.flow` schedule.

**8. The `SchedulerKind` enum raw values use hyphens.** Swift enums with `String` raw values containing hyphens work for serialization, but the PRD also uses these as CLI flag values. Hyphens in CLI values are fine (ArgumentParser handles them), but double-check that the image service sends these exact strings.

### Risks Not Covered

**A. Memory pressure from stateful samplers.** DPM++ 2M, IPNDM, and RES samplers cache previous model outputs (MLXArrays matching latent shape). At 1024x1024 with 16 latent channels, each cached output is ~2MB in bfloat16. IPNDM stores 4 of these. This is negligible, but worth noting that `reset()` should be called between generations if the scheduler is reused.

**B. ControlNet interaction.** The ControlNet pipeline applies control conditioning inside the denoising loop. Multi-evaluation samplers (Heun, RES) will need the control conditioning applied to intermediate evaluations too. The control pipeline's scheduler integration is more complex and should be deferred until Phase 3.

---

## Part 2: Implementation Plan (Phase 1-2)

### File Layout Overview

```
Sources/ZImage/Pipeline/
  Scheduler/
    ZImageScheduler.swift          (NEW - protocol)
    SigmaSchedule.swift            (NEW - sigma schedule functions)
    SchedulerFactory.swift         (NEW - factory enum)
    FlowMatchEulerScheduler.swift  (MOVE + MODIFY - conform to protocol)
    HeunScheduler.swift            (NEW - Phase 2)
    DPMPlusPlus2MScheduler.swift   (NEW - Phase 2)
    DPMPlusPlus2SAScheduler.swift  (NEW - Phase 2)
    DEISScheduler.swift            (NEW - Phase 2)
    DDIMScheduler.swift            (NEW - Phase 2)

Sources/ZImage/Pipeline/
  ZImagePipeline.swift             (MODIFY - use factory)

Sources/ZImage/Server/
  WarmServer.swift                 (MODIFY - add scheduler to payload)

Sources/ZImageCLI/
  main.swift                       (MODIFY - add CLI flags)

Sources/ZImage/Weights/
  ModelConfigs.swift               (NO CHANGE in Phase 1-2)

Tests/ZImageTests/Scheduler/
  FlowMatchSchedulerTests.swift    (MODIFY - update imports)
  SigmaScheduleTests.swift         (NEW)
  HeunSchedulerTests.swift         (NEW - Phase 2)
  DPMPlusPlus2MSchedulerTests.swift (NEW - Phase 2)
  DPMPlusPlus2SASchedulerTests.swift (NEW - Phase 2)
  DEISSchedulerTests.swift         (NEW - Phase 2)
  DDIMSchedulerTests.swift         (NEW - Phase 2)
  SchedulerFactoryTests.swift      (NEW)
```

### Phase 1: Protocol Extraction + Sigma Schedules + Factory

#### 1.1 `Sources/ZImage/Pipeline/Scheduler/ZImageScheduler.swift` (NEW)

```swift
import MLX

/// A scheduler that advances latent samples through a denoising diffusion process.
///
/// Schedulers implement the ODE/SDE integration step for flow-matching models.
/// Single-evaluation schedulers (Euler, DDIM, DPM++ 2M) return `nil` from
/// `intermediateStep`, while multi-evaluation schedulers (Heun, RES) return
/// an intermediate latent that requires a second model forward pass.
public protocol ZImageScheduler {
    /// Sigma values for each step plus a trailing zero: shape [numInferenceSteps + 1].
    var sigmas: MLXArray { get }

    /// Timestep values for each step: shape [numInferenceSteps].
    var timesteps: MLXArray { get }

    /// Number of inference steps configured for this scheduler.
    var numInferenceSteps: Int { get }

    /// Whether this scheduler requires two model evaluations per step.
    var requiresIntermediateEvaluation: Bool { get }

    /// Perform a single denoising step.
    ///
    /// For single-evaluation schedulers, this is the only call per step.
    /// For multi-evaluation schedulers, call `intermediateStep` first,
    /// evaluate the model at the intermediate point, then call `step`
    /// with both predictions.
    ///
    /// - Parameters:
    ///   - modelOutput: The model's velocity prediction for the current sample.
    ///   - timestepIndex: The current step index in [0, numInferenceSteps).
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
    /// then call `finalizeStep` with both predictions.
    mutating func intermediateStep(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray?

    /// Finalize a multi-evaluation step using predictions at both
    /// the original and intermediate points.
    ///
    /// Only called when `intermediateStep` returned non-nil.
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
        return step(modelOutput: originalOutput, timestepIndex: timestepIndex, sample: sample)
    }

    mutating func reset() {}
}
```

**Rationale:**
- `mutating` on all methods because DPM++ 2M, IPNDM, and ancestral samplers cache state.
- The three-method pattern (`step` / `intermediateStep` / `finalizeStep`) avoids passing model closures and keeps schedulers pure value types. Single-eval schedulers only implement `step`; the default extensions handle the rest.
- `reset()` for schedulers that cache predictions between steps.

#### 1.2 `Sources/ZImage/Pipeline/Scheduler/SigmaSchedule.swift` (NEW)

```swift
import Foundation
import MLX

/// Identifies a sigma schedule algorithm.
public enum SigmaScheduleKind: String, CaseIterable, Sendable {
    case flow = "flow"
    case karras = "karras"
    case exponential = "exponential"
    case beta = "beta"
    case linearQuadratic = "linear-quadratic"
    case bongTangent = "bong-tangent"
    case sigmoidOffset = "sigmoid-offset"
    case ays = "ays"
    case lcm = "lcm"
}

/// Pure-function sigma schedule generators.
///
/// Each function returns an array of `numSteps + 1` sigma values,
/// monotonically decreasing from sigma_max to 0.
public enum SigmaSchedule {

    // MARK: - Flow-Match (current ZImage default)

    /// Flow-matching sigma schedule with optional dynamic shifting.
    ///
    /// This reproduces the existing `FlowMatchEulerScheduler` sigma computation
    /// exactly, preserving byte-identical output for the default path.
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

        var timesteps = linspace(sigmaMax * numTrainTimesteps,
                                 sigmaMin * numTrainTimesteps,
                                 count: numSteps)
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

    /// Exponential sigma schedule -- fast decay.
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
    /// Uses a polynomial approximation (no scipy dependency).
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
            let maxVal = powf(0.25, 0.5) // peak of x*(1-x) at x=0.5
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
```

**Key decisions:**
- The `flow()` schedule preserves the exact Float-array arithmetic from the current `FlowMatchEulerScheduler.init`, ensuring backward compatibility.
- Karras, Exponential, Beta are pure `[Float]` functions (no MLXArray dependency) matching the chroma-generate `SigmaSchedule` class.
- The `linspace` helper is extracted from the existing scheduler.
- More schedules (Bong Tangent, Sigmoid Offset, AYS, LCM) are added in Phase 5 per the PRD.

#### 1.3 Modify `Sources/ZImage/Pipeline/FlowMatchScheduler.swift`

Move to `Sources/ZImage/Pipeline/Scheduler/FlowMatchEulerScheduler.swift` and conform to the protocol:

```swift
import Foundation
import MLX

/// First-order Euler ODE solver for flow-matching diffusion models.
public struct FlowMatchEulerScheduler: ZImageScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int

    public init(
        numInferenceSteps: Int,
        config: ZImageSchedulerConfig,
        mu: Float? = nil
    ) {
        precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
        precondition(config.numTrainTimesteps > 0, "numTrainTimesteps must be positive")

        // Delegate sigma computation to the shared SigmaSchedule.flow() function.
        let sigmaValues = SigmaSchedule.flow(
            numSteps: numInferenceSteps,
            config: config,
            mu: mu
        )

        // Derive timesteps from sigmas (excluding the trailing 0).
        let numTrainTimesteps = Float(config.numTrainTimesteps)
        let timestepValues = sigmaValues.dropLast().map { $0 * numTrainTimesteps }

        self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
        self.timesteps = MLXArray(timestepValues, [timestepValues.count])
        self.numInferenceSteps = numInferenceSteps
    }

    /// Generalized init for non-flow sigma schedules.
    public init(
        numInferenceSteps: Int,
        sigmaValues: [Float],
        numTrainTimesteps: Int = 1000
    ) {
        precondition(numInferenceSteps > 0)
        precondition(sigmaValues.count == numInferenceSteps + 1)

        let numTrainF = Float(numTrainTimesteps)
        let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

        self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
        self.timesteps = MLXArray(timestepValues, [timestepValues.count])
        self.numInferenceSteps = numInferenceSteps
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        precondition(timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0),
                     "invalid timestep index")
        let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
        return sample + modelOutput * dt
    }
}
```

**Changes from current:**
- Conforms to `ZImageScheduler` (gets default `intermediateStep`, `finalizeStep`, `reset`).
- Sigma computation delegates to `SigmaSchedule.flow()`.
- Added a second `init` accepting pre-computed sigma values, enabling use with any sigma schedule.
- `step` is now `mutating` (required by protocol; Euler is stateless so this is a no-op in practice).
- The private `linspace` helper moves to `SigmaSchedule`.

#### 1.4 `Sources/ZImage/Pipeline/Scheduler/SchedulerFactory.swift` (NEW)

```swift
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
    // Phase 3+
    // case ipndm = "ipndm"
    // case heunpp2 = "heunpp2"
    // case distance = "distance"
    // case res2m = "res-2m"
    // case res3s = "res-3s"
    // case res5s = "res-5s"
    // case lcm = "lcm"
    // case simple = "simple"
}

/// Creates scheduler instances by kind.
public enum SchedulerFactory {

    /// Create a scheduler with the given configuration.
    ///
    /// - Parameters:
    ///   - kind: The sampler algorithm.
    ///   - sigmaSchedule: The sigma schedule to use. Ignored when `kind == .euler`
    ///     and the schedule is `.flow` (uses the model's native schedule).
    ///   - numInferenceSteps: Number of denoising steps.
    ///   - config: The model's scheduler configuration.
    ///   - mu: Dynamic shifting parameter (pass non-nil when config.useDynamicShifting).
    ///   - seed: Random seed for stochastic samplers (DPM++ 2S-A, DDIM with eta > 0).
    ///   - eta: DDIM stochasticity parameter (0 = deterministic, 1 = full DDPM).
    /// - Returns: A type-erased `ZImageScheduler`.
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

        case .heun:
            return HeunScheduler(
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

        case .dpmplusplus2sa:
            let randomKey = seed.map { MLXRandom.key($0) }
            return DPMPlusPlus2SAScheduler(
                numInferenceSteps: numInferenceSteps,
                sigmaValues: sigmaValues,
                numTrainTimesteps: config.numTrainTimesteps,
                eta: 1.0,
                randomKey: randomKey
            )

        case .deis:
            return DEISScheduler(
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
        case .linearQuadratic, .bongTangent, .sigmoidOffset, .ays, .lcm:
            // Phase 5 schedules -- fall back to flow for now.
            return SigmaSchedule.flow(numSteps: numSteps, config: config, mu: mu)
        }
    }
}
```

#### 1.5 Modify `Sources/ZImage/Pipeline/ZImagePipeline.swift`

Add `scheduler` and `sigmaSchedule` to `ZImageGenerationRequest`:

```swift
// In ZImageGenerationRequest, add after `forceTransformerOverrideOnly`:
public var scheduler: SchedulerKind
public var sigmaSchedule: SigmaScheduleKind
public var eta: Float?
```

Update the `init` with defaults:

```swift
public init(
    // ... existing parameters ...
    scheduler: SchedulerKind = .euler,
    sigmaSchedule: SigmaScheduleKind = .flow,
    eta: Float? = nil
) {
    // ... existing assignments ...
    self.scheduler = scheduler
    self.sigmaSchedule = sigmaSchedule
    self.eta = eta
}
```

Replace the scheduler construction in `generateCore()` (around line 833):

```swift
// BEFORE:
let scheduler = FlowMatchEulerScheduler(
    numInferenceSteps: request.steps,
    config: modelConfigs.scheduler,
    mu: modelConfigs.scheduler.useDynamicShifting ? mu : nil
)

// AFTER:
var scheduler = SchedulerFactory.create(
    kind: request.scheduler,
    sigmaSchedule: request.sigmaSchedule,
    numInferenceSteps: request.steps,
    config: modelConfigs.scheduler,
    mu: modelConfigs.scheduler.useDynamicShifting ? mu : nil,
    seed: request.seed,
    eta: request.eta
)
```

Update the denoising loop to handle multi-evaluation schedulers:

```swift
// BEFORE (line 846-873):
for stepIndex in 0..<request.steps {
    // ... forward pass ...
    latents = scheduler.step(modelOutput: -guidedNoise, timestepIndex: stepIndex, sample: latents)
    MLX.eval(latents)
}

// AFTER:
for stepIndex in 0..<request.steps {
    try Task.checkCancellation()
    progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepIndex, totalSteps: request.steps))
    let timestep = timestepsArray[stepIndex]
    let normalizedTimestep = (1000.0 - timestep) / 1000.0
    let timestepArray = MLXArray([normalizedTimestep], [1])

    var modelLatents = latents
    var embeds = promptEmbeds
    if doCFG, let ne = negativeEmbeds {
        modelLatents = MLX.concatenated([latents, latents], axis: 0)
        embeds = MLX.concatenated([promptEmbeds, ne], axis: 0)
    }

    let noisePred = transformer.forward(latents: modelLatents, timestep: timestepArray, promptEmbeds: embeds)
    let guidedNoise: MLXArray
    if doCFG, negativeEmbeds != nil {
        let batch = latents.dim(0)
        let positive = noisePred[0 ..< batch, 0..., 0..., 0...]
        let negative = noisePred[batch ..< batch * 2, 0..., 0..., 0...]
        guidedNoise = positive + request.guidanceScale * (positive - negative)
    } else {
        guidedNoise = noisePred
    }

    if scheduler.requiresIntermediateEvaluation,
       let intermediateSample = scheduler.intermediateStep(
           modelOutput: -guidedNoise, timestepIndex: stepIndex, sample: latents)
    {
        // Second model evaluation at intermediate point.
        let midTimestep: Float
        if stepIndex + 1 < timestepsArray.count {
            midTimestep = (timestepsArray[stepIndex] + timestepsArray[stepIndex + 1]) / 2.0
        } else {
            midTimestep = timestepsArray[stepIndex] / 2.0
        }
        let normalizedMid = (1000.0 - midTimestep) / 1000.0
        let midTimestepArray = MLXArray([normalizedMid], [1])

        var midModelLatents = intermediateSample
        var midEmbeds = promptEmbeds
        if doCFG, let ne = negativeEmbeds {
            midModelLatents = MLX.concatenated([intermediateSample, intermediateSample], axis: 0)
            midEmbeds = MLX.concatenated([promptEmbeds, ne], axis: 0)
        }

        let midNoisePred = transformer.forward(latents: midModelLatents, timestep: midTimestepArray, promptEmbeds: midEmbeds)
        let midGuidedNoise: MLXArray
        if doCFG, negativeEmbeds != nil {
            let batch = latents.dim(0)
            let positive = midNoisePred[0 ..< batch, 0..., 0..., 0...]
            let negative = midNoisePred[batch ..< batch * 2, 0..., 0..., 0...]
            midGuidedNoise = positive + request.guidanceScale * (positive - negative)
        } else {
            midGuidedNoise = midNoisePred
        }

        latents = scheduler.finalizeStep(
            originalOutput: -guidedNoise,
            intermediateOutput: -midGuidedNoise,
            timestepIndex: stepIndex,
            sample: latents
        )
    } else {
        latents = scheduler.step(modelOutput: -guidedNoise, timestepIndex: stepIndex, sample: latents)
    }
    MLX.eval(latents)
}
```

#### 1.6 Modify `Sources/ZImageCLI/main.swift`

Add CLI flags (in the argument parsing `while` loop, around line 71):

```swift
case "--scheduler":
    schedulerName = nextValue(for: arg, iterator: &iterator)
case "--sigma-schedule":
    sigmaScheduleName = nextValue(for: arg, iterator: &iterator)
case "--eta":
    eta = floatValue(for: arg, iterator: &iterator, fallback: 0.0)
```

Add the variable declarations (around line 66):

```swift
var schedulerName: String?
var sigmaScheduleName: String?
var eta: Float?
```

Add parsing before `ZImageGenerationRequest` construction (around line 178):

```swift
let schedulerKind: SchedulerKind
if let name = schedulerName {
    guard let kind = SchedulerKind(rawValue: name) else {
        let valid = SchedulerKind.allCases.map(\.rawValue).joined(separator: ", ")
        logger.error("Unknown scheduler '\(name)'. Valid options: \(valid)")
        return
    }
    schedulerKind = kind
} else {
    schedulerKind = .euler
}

let sigmaScheduleKind: SigmaScheduleKind
if let name = sigmaScheduleName {
    guard let kind = SigmaScheduleKind(rawValue: name) else {
        let valid = SigmaScheduleKind.allCases.map(\.rawValue).joined(separator: ", ")
        logger.error("Unknown sigma schedule '\(name)'. Valid options: \(valid)")
        return
    }
    sigmaScheduleKind = kind
} else {
    sigmaScheduleKind = .flow
}
```

Pass to the request:

```swift
let request = ZImageGenerationRequest(
    // ... existing ...
    scheduler: schedulerKind,
    sigmaSchedule: sigmaScheduleKind,
    eta: eta
)
```

Update `printUsage()`:

```
  --scheduler            Sampler: euler, heun, dpmpp-2m, dpmpp-2s-a, deis, ddim (default: euler)
  --sigma-schedule       Sigma schedule: flow, karras, exponential, beta (default: flow)
  --eta                  DDIM stochasticity (0=deterministic, 1=stochastic, default: 0)
```

#### 1.7 Modify `Sources/ZImage/Server/WarmServer.swift`

Add fields to `GeneratePayload` (line 653):

```swift
private struct GeneratePayload: Decodable, Sendable {
    let prompt: String
    let negativePrompt: String?
    let width: Int?
    let height: Int?
    let steps: Int?
    let guidance: Float?
    let seed: UInt64?
    let outputPath: String?
    let scheduler: String?      // NEW
    let sigmaSchedule: String?  // NEW
    let eta: Float?             // NEW
```

Update `makePipelineRequest()`:

```swift
func makePipelineRequest(...) throws -> ZImageGenerationRequest {
    // ... existing outputURL logic ...

    let schedulerKind = scheduler.flatMap { SchedulerKind(rawValue: $0) } ?? .euler
    let sigmaKind = sigmaSchedule.flatMap { SigmaScheduleKind(rawValue: $0) } ?? .flow

    return ZImageGenerationRequest(
        // ... existing fields ...
        scheduler: schedulerKind,
        sigmaSchedule: sigmaKind,
        eta: eta
    )
}
```

#### 1.8 Tests for Phase 1

**`Tests/ZImageTests/Scheduler/SigmaScheduleTests.swift`** (NEW):

Test each sigma schedule function:
- `testFlowScheduleMatchesLegacy()` -- compute via `SigmaSchedule.flow()` and compare to the old `FlowMatchEulerScheduler` sigma output within 1e-6.
- `testKarrasMonotonicallyDecreasing()` -- verify sigmas decrease, end in 0.
- `testKarrasCount()` -- `numSteps + 1` elements.
- `testExponentialBounds()` -- first sigma near `sigmaMax`, last is 0.
- `testBetaBounds()` -- same bounds check.
- `testSingleStep()` -- each schedule with `numSteps = 1` produces `[sigma, 0.0]`.

**`Tests/ZImageTests/Scheduler/SchedulerFactoryTests.swift`** (NEW):

- `testEulerFlowIdenticalToLegacy()` -- factory-created Euler + flow produces identical sigmas/timesteps to direct `FlowMatchEulerScheduler` construction.
- `testUnknownSchedulerKindFromString()` -- `SchedulerKind(rawValue: "bogus")` is nil.
- `testAllKindsCreateSuccessfully()` -- loop over `SchedulerKind.allCases`, verify no crash.

**Modify `Tests/ZImageTests/Scheduler/FlowMatchSchedulerTests.swift`:**

- Update import if the file moves into `Scheduler/` subdirectory.
- All 9 existing tests must pass unchanged.
- Add `testProtocolConformance()` -- verify `FlowMatchEulerScheduler` conforms to `ZImageScheduler` at the type level.

---

### Phase 2: Six Sampler Implementations

**Implementation order:** DPM++ 2M first (simplest stateful sampler, deterministic, good template), then DDIM (stateless variant with optional noise), then DEIS (stateless, log-space math), then DPM++ 2S-A (adds randomness), then Heun (multi-evaluation template).

#### 2.1 `DPMPlusPlus2MScheduler.swift` (implement FIRST -- template for stateful samplers)

```swift
import Foundation
import MLX

/// DPM++ 2M: Second-order multistep solver (deterministic).
///
/// Caches the previous model output for second-order correction.
/// Falls back to Euler on the first step (no history).
///
/// Reference: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling
/// of Diffusion Probabilistic Models" (2022).
public struct DPMPlusPlus2MScheduler: ZImageScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int

    private var previousOutput: MLXArray?

    public init(
        numInferenceSteps: Int,
        sigmaValues: [Float],
        numTrainTimesteps: Int = 1000
    ) {
        precondition(numInferenceSteps > 0)
        precondition(sigmaValues.count == numInferenceSteps + 1)

        let numTrainF = Float(numTrainTimesteps)
        let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

        self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
        self.timesteps = MLXArray(timestepValues, [timestepValues.count])
        self.numInferenceSteps = numInferenceSteps
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        precondition(timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0))

        let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)

        let result: MLXArray
        if let prev = previousOutput {
            // Second-order multistep: linear combination
            // D = (3/2) * v_t - (1/2) * v_{t-1}
            let combined = 1.5 * modelOutput - 0.5 * prev
            result = sample + combined * dt
        } else {
            // First step: Euler fallback
            result = sample + modelOutput * dt
        }

        previousOutput = modelOutput
        return result
    }

    public mutating func reset() {
        previousOutput = nil
    }
}
```

**Test file: `DPMPlusPlus2MSchedulerTests.swift`**
- `testFirstStepMatchesEuler()` -- first step output should equal Euler step.
- `testSecondStepDiffersFromEuler()` -- second step uses multistep correction.
- `testDeterministic()` -- same inputs produce same outputs (no randomness).
- `testShapePreservation()` -- output shape matches input.
- `testFullLoopCompletion()` -- run all steps, no crash, valid output.
- `testResetClearsHistory()` -- after reset, first step is Euler again.

#### 2.2 `DDIMScheduler.swift`

```swift
import Foundation
import MLX
import MLXRandom

/// DDIM: Denoising Diffusion Implicit Models.
///
/// Adjustable `eta` controls stochasticity:
/// - eta = 0: fully deterministic (classic DDIM)
/// - eta = 1: equivalent to DDPM
///
/// Adapted for flow-matching velocity-prediction parameterization.
public struct DDIMScheduler: ZImageScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int

    private let eta: Float
    private var randomKey: MLXArray?

    public init(
        numInferenceSteps: Int,
        sigmaValues: [Float],
        numTrainTimesteps: Int = 1000,
        eta: Float = 0.0,
        randomKey: MLXArray? = nil
    ) {
        precondition(numInferenceSteps > 0)
        precondition(sigmaValues.count == numInferenceSteps + 1)

        let numTrainF = Float(numTrainTimesteps)
        let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

        self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
        self.timesteps = MLXArray(timestepValues, [timestepValues.count])
        self.numInferenceSteps = numInferenceSteps
        self.eta = eta
        self.randomKey = randomKey
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        precondition(timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0))

        // Flow-matching parameterization: t maps to alpha_t = 1 - t
        let t = sigmas[timestepIndex].asType(.float32)
        let tPrev = sigmas[timestepIndex + 1].asType(.float32)

        let alphaT = (1.0 - t).asType(sample.dtype)
        let alphaPrev = (1.0 - tPrev).asType(sample.dtype)

        // Predict x_0 from velocity prediction:
        // v = model_output, x_t = sample
        // x_0 = (x_t - sqrt(1 - alpha_t) * v) / sqrt(alpha_t)
        let sqrtAlphaT = MLX.sqrt(MLX.maximum(alphaT, MLXArray(Float(1e-8))))
        let sqrtOneMinusAlphaT = MLX.sqrt(MLX.maximum(1.0 - alphaT, MLXArray(Float(1e-8))))
        let predX0 = (sample - sqrtOneMinusAlphaT * modelOutput) / sqrtAlphaT

        // Direction pointing to x_t
        let sqrtAlphaPrev = MLX.sqrt(MLX.maximum(alphaPrev, MLXArray(Float(1e-8))))
        let sqrtOneMinusAlphaPrev = MLX.sqrt(MLX.maximum(1.0 - alphaPrev, MLXArray(Float(1e-8))))
        let dirXt = sqrtOneMinusAlphaPrev * modelOutput

        // Deterministic component
        var xPrev = sqrtAlphaPrev * predX0 + dirXt

        // Stochastic component when eta > 0
        if eta > 0 {
            let variance = eta * MLX.sqrt(
                (1.0 - alphaPrev) / MLX.maximum(1.0 - alphaT, MLXArray(Float(1e-8)))
            ) * MLX.sqrt(1.0 - alphaT / MLX.maximum(alphaPrev, MLXArray(Float(1e-8))))

            let (noise, nextKey) = splitKey()
            randomKey = nextKey
            xPrev = xPrev + variance.asType(sample.dtype) * noise
        }

        return xPrev
    }

    private mutating func splitKey() -> (MLXArray, MLXArray?) {
        // If we have a key, split it for reproducibility.
        // Otherwise, use unseeded random.
        guard let key = randomKey else {
            return (MLXRandom.normal([1]), nil)
        }
        let split = MLXRandom.split(key: key)
        let noise = MLXRandom.normal([1], key: split[0])
        return (noise, split[1])
    }
}
```

**Note:** The DDIM math here uses the flow-matching velocity parameterization (matching chroma-generate's `ChromaDDIMScheduler`), not the noise-prediction DDPM formula from the PRD's Section 4.

**Test file: `DDIMSchedulerTests.swift`**
- `testEtaZeroDeterministic()` -- repeated runs with eta=0 produce identical output.
- `testEtaOneStochastic()` -- output differs from eta=0.
- `testShapePreservation()`
- `testFullLoop()`

#### 2.3 `DEISScheduler.swift`

```swift
import Foundation
import MLX

/// DEIS: Diffusion Exponential Integrator Sampler.
///
/// Uses exponential integration of the probability flow ODE for
/// 40-50% faster convergence than DPM++ at comparable quality.
///
/// Reference: Zhang & Chen, "Fast Sampling of Diffusion Models
/// with Exponential Integrator" (2022).
public struct DEISScheduler: ZImageScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int

    public init(
        numInferenceSteps: Int,
        sigmaValues: [Float],
        numTrainTimesteps: Int = 1000
    ) {
        precondition(numInferenceSteps > 0)
        precondition(sigmaValues.count == numInferenceSteps + 1)

        let numTrainF = Float(numTrainTimesteps)
        let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

        self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
        self.timesteps = MLXArray(timestepValues, [timestepValues.count])
        self.numInferenceSteps = numInferenceSteps
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        precondition(timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0))

        let sigmaT = sigmas[timestepIndex].asType(.float32)
        let sigmaNext = sigmas[timestepIndex + 1].asType(.float32)

        // Avoid log(0) at the final step.
        let safeSigmaT = MLX.maximum(sigmaT, MLXArray(Float(1e-8)))
        let safeSigmaNext = MLX.maximum(sigmaNext, MLXArray(Float(1e-8)))

        let logSigma = MLX.log(safeSigmaT)
        let logSigmaNext = MLX.log(safeSigmaNext)
        let h = (logSigmaNext - logSigma).asType(sample.dtype)

        // Exponential integrator update:
        // x_{t+1} = (sigma_next / sigma_t) * x_t + sigma_next * (exp(h) - 1) * denoised
        //
        // For velocity prediction, denoised = x_t + modelOutput * sigma_t
        // but we receive -guidedNoise as modelOutput, which is already -velocity.
        // The pipeline negates before calling step, so modelOutput = -v.
        // denoised = x_t - modelOutput * sigma_t (conceptually)
        //
        // Using the simplified flow-matching form:
        let ratio = (safeSigmaNext / safeSigmaT).asType(sample.dtype)
        let expTerm = (MLX.exp(h) - 1.0).asType(sample.dtype)

        // In flow-matching, the velocity step is:
        // x_next = x_t + v * dt
        // DEIS refines this with exponential weighting:
        let dt = (sigmaNext - sigmaT).asType(sample.dtype)
        return sample + modelOutput * dt * (expTerm / h)
    }
}
```

**Test file: `DEISSchedulerTests.swift`**
- `testDEISConvergesFasterThanEuler()` -- at 15 steps, DEIS output should differ significantly from Euler (cannot verify quality in unit test, but verify it runs and differs).
- `testDeterministic()` -- same inputs, same outputs.
- `testShapePreservation()`
- `testFullLoop()`

#### 2.4 `DPMPlusPlus2SAScheduler.swift`

```swift
import Foundation
import MLX
import MLXRandom

/// DPM++ 2S Ancestral: Second-order solver with ancestral sampling.
///
/// Adds controlled noise injection for creative variation.
/// The `eta` parameter scales the noise (1.0 = full ancestral, 0.0 = deterministic).
public struct DPMPlusPlus2SAScheduler: ZImageScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int

    private let eta: Float
    private var previousOutput: MLXArray?
    private var randomKey: MLXArray?

    public init(
        numInferenceSteps: Int,
        sigmaValues: [Float],
        numTrainTimesteps: Int = 1000,
        eta: Float = 1.0,
        randomKey: MLXArray? = nil
    ) {
        precondition(numInferenceSteps > 0)
        precondition(sigmaValues.count == numInferenceSteps + 1)

        let numTrainF = Float(numTrainTimesteps)
        let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

        self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
        self.timesteps = MLXArray(timestepValues, [timestepValues.count])
        self.numInferenceSteps = numInferenceSteps
        self.eta = eta
        self.randomKey = randomKey
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        precondition(timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0))

        let sigmaT = sigmas[timestepIndex].asType(sample.dtype)
        let sigmaNext = sigmas[timestepIndex + 1].asType(sample.dtype)
        let dt = sigmaNext - sigmaT

        let result: MLXArray
        if let prev = previousOutput {
            // Second-order: linear combination of current and previous
            let combined = 1.5 * modelOutput - 0.5 * prev
            result = sample + combined * dt
        } else {
            // First step: Euler
            result = sample + modelOutput * dt
        }

        previousOutput = modelOutput

        // Ancestral noise injection
        if eta > 0, sigmaNext > MLXArray(Float(1e-8)) {
            let sigmaUp = sigmaNext * MLXArray(eta) * MLX.sqrt(
                1.0 - (sigmaT / MLX.maximum(sigmaNext, MLXArray(Float(1e-8)))) * (sigmaT / MLX.maximum(sigmaNext, MLXArray(Float(1e-8))))
            )
            let noise: MLXArray
            if let key = randomKey {
                let split = MLXRandom.split(key: key)
                noise = MLXRandom.normal(sample.shape, key: split[0])
                randomKey = split[1]
            } else {
                noise = MLXRandom.normal(sample.shape)
            }
            return result + sigmaUp.asType(sample.dtype) * noise
        }

        return result
    }

    public mutating func reset() {
        previousOutput = nil
    }
}
```

**Test file: `DPMPlusPlus2SASchedulerTests.swift`**
- `testFirstStepMatchesEuler()` -- no history, should match Euler.
- `testStochasticOutputDiffersFromDeterministic()` -- eta=1 vs eta=0 produces different results.
- `testReproducibleWithSeed()` -- same seed produces same output.
- `testShapePreservation()`
- `testFullLoop()`

#### 2.5 `HeunScheduler.swift`

```swift
import Foundation
import MLX

/// Heun method: Second-order Runge-Kutta (explicit trapezoidal rule).
///
/// Requires two model evaluations per step:
/// 1. Evaluate at current point to get d1.
/// 2. Euler step to intermediate point.
/// 3. Evaluate at intermediate point to get d2.
/// 4. Average d1 and d2 for the final step.
///
/// 2x cost per step but significantly better accuracy than Euler.
public struct HeunScheduler: ZImageScheduler {
    public let sigmas: MLXArray
    public let timesteps: MLXArray
    public let numInferenceSteps: Int
    public let requiresIntermediateEvaluation: Bool = true

    public init(
        numInferenceSteps: Int,
        sigmaValues: [Float],
        numTrainTimesteps: Int = 1000
    ) {
        precondition(numInferenceSteps > 0)
        precondition(sigmaValues.count == numInferenceSteps + 1)

        let numTrainF = Float(numTrainTimesteps)
        let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }

        self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
        self.timesteps = MLXArray(timestepValues, [timestepValues.count])
        self.numInferenceSteps = numInferenceSteps
    }

    // Single-step fallback (used if caller ignores multi-eval protocol).
    public mutating func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        // Degrade to Euler if called directly.
        let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
        return sample + modelOutput * dt
    }

    public mutating func intermediateStep(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray? {
        // Euler step to the intermediate point.
        let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
        return sample + modelOutput * dt
    }

    public mutating func finalizeStep(
        originalOutput: MLXArray,
        intermediateOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray {
        // Trapezoidal rule: average the two derivatives.
        let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
        let avgOutput = (originalOutput + intermediateOutput) / 2.0
        return sample + avgOutput * dt
    }
}
```

**Test file: `HeunSchedulerTests.swift`**
- `testRequiresIntermediateEvaluation()` -- `requiresIntermediateEvaluation == true`.
- `testIntermediateStepReturnsNonNil()` -- `intermediateStep()` returns a value.
- `testFinalizeProducesDifferentResultThanEuler()` -- Heun's averaged step differs from single Euler step.
- `testSingleStepFallbackMatchesEuler()` -- calling `step()` directly (no intermediate) degrades gracefully.
- `testShapePreservation()`
- `testFullLoop()` -- simulating both evaluations manually.

---

## Part 3: Phase Ordering

### Phase 1 (Days 1-3)
1. **ZImageScheduler protocol** -- the foundational abstraction.
2. **SigmaSchedule** -- flow, karras, exponential, beta (pure math, immediate testability).
3. **FlowMatchEulerScheduler refactor** -- conform to protocol, delegate sigmas.
4. **SchedulerFactory** -- create by kind.
5. **CLI flags** -- `--scheduler`, `--sigma-schedule`, `--eta`.
6. **ZImageGenerationRequest** -- add scheduler/sigmaSchedule/eta fields.
7. **Pipeline integration** -- replace hardcoded constructor with factory.
8. **WarmServer payload** -- add scheduler fields.
9. **Tests** -- SigmaScheduleTests, SchedulerFactoryTests, verify existing FlowMatchSchedulerTests pass.

**Gate:** `--scheduler euler --sigma-schedule flow` produces output identical to the current build at the same seed.

### Phase 2 (Days 4-7)

Implement in this order:

1. **DPM++ 2M** (day 4) -- simplest stateful sampler, deterministic. Template for caching `previousOutput`. Port from `ChromaDPMPP2MScheduler`.

2. **DDIM** (day 4-5) -- stateless, optional stochasticity via eta. Important test case: eta=0 must be deterministic. Port from `ChromaDDIMScheduler`, correcting the parameterization to flow-matching.

3. **DEIS** (day 5) -- stateless, log-space exponential integration. Short implementation. Port the exponential integrator math from Section 4 of the PRD.

4. **DPM++ 2S Ancestral** (day 6) -- stateful + stochastic. Builds on DPM++ 2M pattern, adds random key management. Port from `ChromaDPMPP2SScheduler`.

5. **Heun** (day 6-7) -- multi-evaluation scheduler. First use of `intermediateStep`/`finalizeStep`. The pipeline loop change (Phase 1.5) must be in place. Port from `ChromaHeunScheduler`, using the true two-evaluation approach rather than the approximation fallback.

**Gate:** All 6 samplers produce valid images. Euler + flow still byte-identical. Each sampler has full unit test coverage.

### Phase 3-6 (Future)

Per PRD, with one addition: update `ZImageControlPipeline` scheduler construction in Phase 3 when the Heun/RES multi-evaluation pattern is proven in the base pipeline.

---

## Appendix: Swift/MLX Porting Notes

### Python-to-Swift translation table

| Python (mlx) | Swift (mlx-swift) |
|---|---|
| `mx.array(x)` | `MLXArray(x)` |
| `mx.concatenate([a, b])` | `MLX.concatenated([a, b])` |
| `mx.exp(x)` | `MLX.exp(x)` |
| `mx.log(x)` | `MLX.log(x)` |
| `mx.sqrt(x)` | `MLX.sqrt(x)` |
| `mx.linspace(a, b, n)` | `MLXArray` from `[Float]` via manual linspace (no `MLX.linspace` in mlx-swift) |
| `mx.random.normal(shape)` | `MLXRandom.normal(shape)` |
| `mx.random.normal(shape, key=k)` | `MLXRandom.normal(shape, key: k)` |
| `mx.random.split(key)` | `MLXRandom.split(key: key)` |
| `mx.clip(x, a, b)` | `MLX.clip(x, min: a, max: b)` |
| `mx.where(cond, a, b)` | `MLX.where(cond, a, b)` |
| `mx.eval(x)` | `MLX.eval(x)` |
| `x.astype(dtype)` | `x.asType(dtype)` |
| `x.shape` | `x.shape` |
| `x.ndim` | `x.ndim` |
| `x[i]` | `x[i]` |
| `x.tolist()` | `x.asArray(Float.self)` |
| `len(x)` | `x.dim(0)` |

### Key differences

1. **No `mx.linspace` in mlx-swift.** Use `[Float]` manual computation, then convert to `MLXArray`. The existing codebase already does this.

2. **Value types vs reference types.** Python schedulers are classes; Swift schedulers should be structs (value types) with `mutating` methods. This aligns with the existing `FlowMatchEulerScheduler` being a struct.

3. **Random state.** Python uses `mx.random.key()` and `mx.random.split()`. Swift's `MLXRandom` has the same API. Store the key in the scheduler struct and split it on each stochastic step for reproducibility.

4. **Type casting.** Python's MLX auto-promotes types. Swift requires explicit `.asType()` calls. Always cast sigma values to `sample.dtype` before arithmetic to avoid float32/bfloat16 mismatches.

5. **Preconditions.** Use `precondition()` for programmer errors (invalid indices) and return gracefully for edge cases (sigma near zero).
