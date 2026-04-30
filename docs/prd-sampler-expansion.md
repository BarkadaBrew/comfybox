# PRD: Multi-Sampler Support for ZImage CLI

**Author:** Bree (BaristaBree)
**Date:** 2026-04-28
**Status:** Draft
**Repo:** BarkadaBrew/zimage.swift (fork of mzbac/zimage.swift)
**License:** MIT
**Reference:** chroma-generate scheduler catalog (15 samplers, 8+ sigma schedules)

---

## 1. Problem Statement

ZImage CLI currently hardcodes a single scheduler: `FlowMatchEulerScheduler` — a first-order Euler ODE solver for rectified flow models. While functional, Euler is the simplest possible solver.

Todd's chroma-generate project demonstrates the value of sampler diversity — it implements 15 schedulers and 8 sigma schedules, each producing meaningfully different results. Key findings from chroma-generate testing:

- **RES 3S + Bong Tangent** — best overall quality (95/100), superior anatomical accuracy
- **DEIS + Exponential** — best speed/quality balance (88/100, 3-4x faster than RES)
- **IPNDM + Karras** — best for portraits, excellent skin tones
- **RES 2M + Sigmoid Offset** — best photorealism, good color accuracy
- **Euler + Beta** — fastest acceptable quality for prototyping
- **LCM** — ultra-fast 4-8 step generation for real-time use

The image service already has `scheduler` field plumbing (`ResolvedJobRequest.scheduler`, `--scheduler` CLI passthrough) — it just needs ZImage to accept and use it.

## 2. Current Architecture

### Scheduler (63 lines)

```
Sources/ZImage/Pipeline/FlowMatchScheduler.swift
```

`FlowMatchEulerScheduler` is a value type with:
- `init(numInferenceSteps:config:mu:)` — builds sigma schedule with optional dynamic shifting
- `step(modelOutput:timestepIndex:sample:) -> MLXArray` — single Euler step: `sample + modelOutput * dt`
- `sigmas: MLXArray` / `timesteps: MLXArray` / `numInferenceSteps: Int`

### Pipeline Integration

In `ZImagePipeline.generateCore()`:

```swift
let scheduler = FlowMatchEulerScheduler(
    numInferenceSteps: request.steps,
    config: modelConfigs.scheduler,
    mu: modelConfigs.scheduler.useDynamicShifting ? mu : nil
)
for stepIndex in 0..<request.steps {
    // transformer forward pass
    latents = scheduler.step(modelOutput: -guidedNoise, timestepIndex: stepIndex, sample: latents)
    MLX.eval(latents)
}
```

### Config

`ZImageSchedulerConfig` (from `scheduler/scheduler_config.json`):
- `numTrainTimesteps: Int` (1000)
- `shift: Float`
- `useDynamicShifting: Bool`
- `baseShift/maxShift: Float?`
- `baseImageSeqLen/maxImageSeqLen: Int?`

### Tests

`Tests/ZImageTests/Scheduler/FlowMatchSchedulerTests.swift` — 9 test cases covering init, timesteps, sigmas, dynamic shifting, step math, edge cases. Uses a `makeConfig()` helper.

### Image Service Plumbing (coffeeshop-image-service)

`generator.ts` line 225: `if (request.scheduler) args.push('--scheduler', request.scheduler);`

Already wired. Just needs ZImage to accept the flag.

## 3. Proposed Design

### 3.1 Protocol: `FlowMatchScheduler`

Extract the common interface into a protocol:

```swift
public protocol FlowMatchScheduler {
    var sigmas: MLXArray { get }
    var timesteps: MLXArray { get }
    var numInferenceSteps: Int { get }

    mutating func step(
        modelOutput: MLXArray,
        timestepIndex: Int,
        sample: MLXArray
    ) -> MLXArray
}
```

Note: `mutating` because DPM++ 2M needs to cache the previous step's output. Euler and DDIM are stateless.

### 3.2 Sampler Implementations

Full catalog derived from chroma-generate's proven implementations, prioritized for ZImage Turbo's flow-matching architecture.

#### Tier 1 — Core (Phase 1-2)

| Sampler | Key | Complexity | Notes |
|---------|-----|------------|-------|
| **Euler** | `euler` | Existing | Current behavior, no change. First-order ODE. |
| **Heun** | `heun` | ~70 LOC | 2nd-order Runge-Kutta. 2x cost per step but better quality. |
| **DPM++ 2M** | `dpmpp-2m` | ~80 LOC | Second-order multistep. Caches previous output. Better convergence at low steps. Deterministic. |
| **DPM++ 2S Ancestral** | `dpmpp-2s-a` | ~90 LOC | Chroma's default. Ancestral sampling adds creative variation. |
| **DEIS** | `deis` | ~100 LOC | Diffusion Exponential Integrator. 40-50% faster than DPM++ at similar quality. Best speed/quality balance. |
| **DDIM** | `ddim` | ~70 LOC | Adjustable `eta` (0=deterministic, 1=stochastic). Research baseline. |

#### Tier 2 — Advanced (Phase 3)

| Sampler | Key | Complexity | Notes |
|---------|-----|------------|-------|
| **IPNDM** | `ipndm` | ~120 LOC | Improved Pseudo Numerical Methods. 4th-order. Best for portraits/skin. Needs 30+ steps. |
| **Heun++** | `heunpp2` | ~90 LOC | Enhanced Heun with stochasticity params (s_churn, s_noise). |
| **Distance** | `distance` | ~150 LOC | Dynamic CFG support (cosine/linear/exponential curves). Best for photorealism. |

#### Tier 3 — RES Family (Phase 4)

| Sampler | Key | Complexity | Notes |
|---------|-----|------------|-------|
| **RES 2M** | `res-2m` | ~100 LOC | Refined Explicit Solver, 2nd order. Richardson extrapolation. Superior to DPM++ 2M. |
| **RES 3S** | `res-3s` | ~130 LOC | 3rd order, 3 substeps per step. Best overall quality in chroma benchmarks (95/100). |
| **RES 5S** | `res-5s` | ~180 LOC | 5th order adaptive Runge-Kutta. Ultimate quality. 5x compute cost. |

#### Tier 4 — Experimental (Phase 5)

| Sampler | Key | Complexity | Notes |
|---------|-----|------------|-------|
| **LCM** | `lcm` | ~80 LOC | Latent Consistency Model. 4-8 steps, ultra-fast. Needs special sigma mapping. |
| **Simple** | `simple` | ~40 LOC | Pure Euler without corrections. Baseline for testing. |
| **RescaleCFG** | `rescale-cfg` | ~60 LOC | Wrapper — enables safe high-CFG (15-30) on any base sampler. |

### 3.3 Sigma Schedules

Separate concern from samplers. Each sampler can pair with any sigma schedule. Chroma-generate findings:

| Schedule | Key | Notes | Best Pairing |
|----------|-----|-------|--------------|
| **Flow-Match** | `flow` | Current ZImage default. Dynamic shifting. | Euler (existing behavior) |
| **Karras** | `karras` | Industry standard. Optimized noise distribution. | IPNDM, DPM++ 2M |
| **Beta** | `beta` | Quality-focused, gradual transitions. | RES 5S, artistic content |
| **Exponential** | `exponential` | Aggressive noise reduction, fast. | DEIS, Euler |
| **Bong Tangent** | `bong-tangent` | Non-linear tangent transitions. Superior detail. | RES 3S (highest quality combo) |
| **Sigmoid Offset** | `sigmoid-offset` | S-curve transitions, adjustable offset. | RES 2M, photographic |
| **Linear Quadratic** | `linear-quadratic` | Smooth generation. | General |
| **AYS** | `ays` | Align Your Steps — optimized for exact step counts. | Any |
| **LCM** | `lcm` | Maps low steps to original training distribution. | LCM sampler only |

### 3.4 Factory

```swift
public enum SchedulerKind: String, CaseIterable, Sendable {
    case euler = "euler"
    case heun = "heun"
    case heunpp2 = "heunpp2"
    case dpmplusplus2m = "dpmpp-2m"
    case dpmplusplus2sa = "dpmpp-2s-a"
    case deis = "deis"
    case ddim = "ddim"
    case ipndm = "ipndm"
    case distance = "distance"
    case res2m = "res-2m"
    case res3s = "res-3s"
    case res5s = "res-5s"
    case lcm = "lcm"
    case simple = "simple"
    case rescaleCfg = "rescale-cfg"
}

public enum SigmaScheduleKind: String, CaseIterable, Sendable {
    case flow = "flow"              // current default
    case karras = "karras"
    case beta = "beta"
    case exponential = "exponential"
    case bongTangent = "bong-tangent"
    case sigmoidOffset = "sigmoid-offset"
    case linearQuadratic = "linear-quadratic"
    case ays = "ays"
    case lcm = "lcm"
}

public enum SchedulerFactory {
    public static func create(
        kind: SchedulerKind,
        sigmaSchedule: SigmaScheduleKind = .flow,
        numInferenceSteps: Int,
        config: ZImageSchedulerConfig,
        mu: Float? = nil,
        seed: UInt64? = nil,
        eta: Float? = nil       // for DDIM
    ) -> any FlowMatchScheduler
}
```

### 3.5 CLI Flags

Add to `ZImageCLI/main.swift`:

```swift
@Option(name: .long, help: "Sampler: euler, heun, dpmpp-2m, dpmpp-2s-a, deis, ddim, ipndm, res-2m, res-3s, res-5s, lcm, simple")
var scheduler: String = "euler"

@Option(name: .long, help: "Sigma schedule: flow, karras, beta, exponential, bong-tangent, sigmoid-offset, ays, lcm")
var sigmaSchedule: String = "flow"

@Option(name: .long, help: "DDIM eta (0=deterministic, 1=stochastic)")
var eta: Float?
```

Pass through to `ZImageGenerationRequest`:

```swift
public struct ZImageGenerationRequest {
    // ... existing fields ...
    public var scheduler: SchedulerKind       // default .euler
    public var sigmaSchedule: SigmaScheduleKind  // default .flow
    public var eta: Float?                    // DDIM only
}
```

### 3.5 Pipeline Change

In `generateCore()`, replace direct construction with factory:

```swift
let scheduler = SchedulerFactory.create(
    kind: request.scheduler,
    numInferenceSteps: request.steps,
    config: modelConfigs.scheduler,
    mu: modelConfigs.scheduler.useDynamicShifting ? mu : nil,
    seed: request.seed
)
```

The denoising loop stays identical — just calls `scheduler.step()`.

### 3.6 Warm Server Protocol

The warm server's `/v1/generate` endpoint needs a `scheduler` field in the JSON body. The response format is unchanged.

## 4. Sampler Math Reference

All formulas below use **flow-matching (velocity prediction)** parameterization, not noise-prediction DDPM. The sigma schedule feeds into each — only the step update rule changes.

### Euler (existing, 1st order)
```
x_{t+1} = x_t + v_t * dt
where dt = sigma_{t+1} - sigma_t
```

### Heun (2nd order Runge-Kutta)
```
d1 = model(x_t, sigma_t)
x_hat = x_t + dt * d1
d2 = model(x_hat, sigma_{t+1})
x_{t+1} = x_t + dt * (d1 + d2) / 2
```

### DPM++ 2M (2nd order multistep)
```
# First step: Euler fallback
x_1 = x_0 + v_0 * dt_0

# Subsequent steps:
r = dt_t / dt_{t-1}
D = (1 + 1/(2r)) * v_t - (1/(2r)) * v_{t-1}
x_{t+1} = x_t + D * dt_t
```

### DPM++ 2S Ancestral
```
sigma_down = sigma_{t+1}
sigma_up = sqrt(sigma_t^2 - sigma_down^2)
d = (x_t - denoised) / sigma_t
x_mid = x_t + (sigma_down - sigma_t) / 2 * d
d2 = (x_mid - model(x_mid)) / sigma_mid
x_{t+1} = x_t + (sigma_down - sigma_t) * d2 + sigma_up * noise
```

### DEIS (Exponential Integrator)
```
# Uses exponential integration of the probability flow ODE
log_sigma = log(sigma_t)
log_sigma_next = log(sigma_{t+1})
h = log_sigma_next - log_sigma
x_{t+1} = (sigma_{t+1}/sigma_t) * x_t + sigma_{t+1} * (exp(h) - 1) * denoised
```

### DDIM
```
alpha_t = 1 - sigma_t^2
pred_x0 = (x_t - sigma_t * v_t) / sqrt(alpha_t)
dir_xt = sqrt(1 - alpha_{t+1} - eta^2 * sigma_{t+1}^2) * v_t
noise = eta * sigma_{t+1} * randn
x_{t+1} = sqrt(alpha_{t+1}) * pred_x0 + dir_xt + noise
```

### RES 2M (Richardson extrapolation, 2nd order)
```
d = (x - denoised) / t
x_mid = x + 0.5 * h * d
d_mid = (x_mid - model(x_mid)) / t_mid
d2 = (d_mid - d) / (0.5 * h)
x_{t+1} = x + h * d_mid + (h^2 / 12) * d2
```

### RES 3S (3rd order, 3 substeps)
```
d1 = (x - denoised) / t
d2 = (x + h/3 * d1 - model(...)) / (t + h/3)
d3 = (x + 2h/3 * d2 - model(...)) / (t + 2h/3)
d_combined = (2*d3 + 3*d2 - d1) / 4    # standard
d_combined = (9*d3 - 3*d2 + d1 - prev_d) / 6  # with history
x_{t+1} = x + h * d_combined
```

### RES 5S (5th order adaptive Runge-Kutta)
```
Uses Cash-Karp Butcher tableau (5 evaluations per step).
Optional Richardson extrapolation from history for even higher accuracy.
Reference: chroma-generate/res4lyf_samplers.py
```

### Sigma Schedule: Bong Tangent
```
Non-linear transitions using tangent function for superior detail emergence.
Pairs with RES samplers for highest quality output.
Reference: chroma-generate/mlx_schedulers.py — SigmaSchedule.bong_tangent()
```

## 5. Testing Strategy

### Unit Tests (per sampler)

Mirror the existing `FlowMatchSchedulerTests` structure:
- Init with default/custom steps
- Sigma monotonicity and bounds
- Step produces different output from input
- Full loop through all timesteps produces valid output
- Shape preservation

### Comparative Tests

- Same prompt + seed + steps with Euler vs DPM++ 2M: output differs
- Euler Ancestral with same seed: output differs from Euler (stochastic term)
- DDIM with eta=0: deterministic (same output on repeat)
- DPM++ 2M at 8 steps vs Euler at 16 steps: visual quality comparison (manual)

### Integration Test

- Round-trip: CLI `--scheduler dpmpp-2m` produces a valid image file
- Unknown scheduler name: clean error message, not crash

## 6. Recommended Pairings (from chroma-generate benchmarks)

| Use Case | Sampler | Sigma Schedule | Steps | CFG | Quality |
|----------|---------|----------------|-------|-----|---------|
| **Maximum quality** | res-3s | bong-tangent | 30-40 | 2.5-3.5 | 95/100 |
| **Speed/quality balance** | deis | exponential | 20-25 | 2.5-3.0 | 88/100 |
| **Portraits** | ipndm | karras | 25-30 | 3.0-4.0 | 90/100 |
| **Photorealism** | res-2m | sigmoid-offset | 25-30 | 2.5-3.0 | 92/100 |
| **Fast prototyping** | euler | beta | 15-20 | 4.0-6.0 | 75/100 |
| **Deterministic** | ddim | karras | 20-25 | 3.0-4.0 | — |
| **Ultra-fast** | lcm | lcm | 4-8 | 1.0-3.0 | — |
| **Default (balanced)** | dpmpp-2s-a | karras | 20-36 | 3.5-4.5 | 85/100 |

**Note:** CFG values above are from chroma-generate testing. Z-Image Turbo is distilled for CFG=1.0 — these pairings need empirical validation on our pipeline. Euler at CFG=1.0 remains the safe default.

## 7. Scope and Effort

| Phase | Deliverable | Estimate |
|-------|-------------|----------|
| Phase 1 | Protocol + SigmaSchedule abstraction + Euler refactor + factory + CLI flags + tests | 2-3 days |
| Phase 2 | Heun + DPM++ 2M + DPM++ 2S-A + DEIS + DDIM + tests | 3-4 days |
| Phase 3 | IPNDM + Heun++ + Distance + tests | 2-3 days |
| Phase 4 | RES 2M + RES 3S + RES 5S (port from chroma res4lyf_samplers.py) + tests | 3-4 days |
| Phase 5 | LCM + Simple + RescaleCFG + sigma schedules (Karras, Beta, Exponential, Bong Tangent, Sigmoid) | 2-3 days |
| Phase 6 | Warm server protocol + image service integration + preset combos | 1 day |

**Total: 13-18 days of focused work.**
- Phase 1-2 (8 samplers + framework): 5-7 days — covers the most impactful samplers
- Phase 3-4 (advanced + RES family): 5-7 days — unlocks highest quality
- Phase 5-6 (experimental + integration): 3-4 days — completeness

### Implementation Approach

Codex 5.5 will review this PRD and produce an implementation plan. Development on a feature branch off the current zimage.swift fork, with controlled experimentation:

1. Branch: `bree/sampler-expansion`
2. Each sampler gets a PR with unit tests + a comparison render vs Euler at same seed
3. Sigma schedules are independent PRs (they compose with any sampler)
4. Integration PR wires `--scheduler` and `--sigma-schedule` CLI flags end-to-end

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Flow-matching vs noise-prediction math | Wrong output, artifacts | Port from chroma-generate's MLX implementations (same framework); validate against diffusers |
| Z-Image Turbo distilled for Euler at CFG=1 | Other samplers may produce worse results | Empirical testing per sampler; Euler remains default; document which samplers work well |
| RES samplers need multiple model evaluations per step | 3-5x slower per step | Document clearly; fewer steps needed for same quality (RES 3S at 15 steps ≈ Euler at 45) |
| Ancestral/SDE samplers need random state | Thread safety | Pass seed through factory, create per-step random keys from MLXRandom |
| DPM++/RES first-step edge case | No history cache on step 0 | Fall back to Euler for first step (standard practice) |
| Breaking change to `ZImageGenerationRequest` | Downstream consumers | Default to `.euler` + `.flow`, fully backward compatible |
| Sigma schedule math from chroma is Python/numpy | Needs Swift/MLX port | Math is straightforward (< 30 lines each); chroma source is the reference |

## 9. Non-Goals

- Dynamic CFG curves (distance scheduler has this but it's a separate feature)
- Adaptive step-size solvers (runtime step count adjustment)
- Sampler-specific guidance scale auto-tuning
- UI/gallery A/B comparison tool
- Warm server sampler hot-swap without restart

## 10. Reference Material

- `chroma-generate/mlx_schedulers.py` — All sigma schedules + base scheduler classes (MLX-native)
- `chroma-generate/res4lyf_samplers.py` — RES 2M/3S/5S implementations (MLX-native)
- `chroma-generate/chroma/chroma/chromasampler.py` — Flow-matching base sampler
- `chroma-generate/SCHEDULERS_GUIDE.md` — Full scheduler catalog with benchmarks
- `chroma-generate/optimal_sampler_scheduler_guide.md` — Tested pairings and recommendations
- `chroma-generate/compare_best_samplers.sh` — Comparison testing script

## 11. Success Criteria

1. `--scheduler euler --sigma-schedule flow` produces byte-identical output to current behavior (same seed)
2. `--scheduler deis --sigma-schedule exponential` at 15 steps produces comparable quality to Euler at 30 steps
3. `--scheduler res-3s --sigma-schedule bong-tangent` at 15 steps produces visibly superior detail
4. `--scheduler ddim --eta 0` is fully deterministic (identical output on repeated runs)
5. All existing tests pass unchanged
6. Each new sampler has unit tests matching the FlowMatchSchedulerTests pattern
7. Image service routes `scheduler` + `sigmaSchedule` fields end-to-end without code changes
8. CLI `--scheduler unknown` produces a clean error listing valid options
