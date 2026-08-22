// RES4LYFSDENoise.swift — RES4LYF's `eta` SDE, ported exactly (parity tier T2).
//
// WP-E15 (docs/FDD-krea2-raw-recipe.md §3.13, D18, AC-26/27/28), against the
// pinned upstream at `scripts/oracles/upstream/res4lyf/beta/`, commit
// `26036f647ca15d3048a193daf99a40cecfc3820d`.
//
// `eta = 0.5` is the workflow's value and the node's default. It is NOT
// decoration: on a variance-preserving flow model in `noise_mode_sde = "hard"`
// every step is split into a shorter deterministic move plus a re-noise, and
// the trajectory that results is a different one — not the same trajectory
// with jitter.
//
// The split (`rk_noise_sampler_beta.py`, `get_sde_step` `case "hard"` at :444
// feeding `get_sde_coeff`'s VP branch at :279-293):
//
//     σ_up   = η · σ'                                (hard mode: eta_ratio = η)
//     σ_res  = sqrt(σ'² − σ_up²)
//     α      = (σ_max − σ') + σ_res
//     σ_down = σ_res / α
//
// and the swap itself (`swap_noise_step` :588-613, `swap_noise_substep`
// :618-645), written in the epsilon the step ALREADY implies rather than in a
// fresh model call:
//
//     eps'      = (x₀ − x_next) / (σ − σ_target)
//     denoised' = x₀ − σ · eps'
//     x         = α · (denoised' + σ_down · eps') + σ_up · noise · s_noise
//
// Two things about that formula are easy to get wrong and are the reason this
// file exists rather than an inline three-liner:
//
//   1. **It is anchored at the STEP's σ and x₀**, not at the target. `σ` is
//      `NS.sigma` — the sigma the step started from — even in the substep
//      swap, where `σ_target` is the substep's own `s_[row]`. `noise_anchor`
//      is 1.0 and this is what that means.
//   2. **RES4LYF re-noises SUBSTEPS as well as steps.** `eta_substep` is set
//      equal to `eta` by `ClownsharKSampler_Beta`, and `rk_sampler_beta.py`
//      :1874 swaps noise into `x_[row + row_offset]` after every non-final
//      row's update — so the next row's model call is evaluated on a re-noised
//      sample. A port that only implemented the step-level swap reproduces
//      neither T2 trace: for `res_2s` the second row's input IS the re-noised
//      substep sample (`step00/sub0/x_post == step00/call1/x_in` in the
//      fixture, to the bit).
//
// The row's injected noise reaches the step's result ONLY through that model
// call: upstream rebuilds every row sample from `x₀` and the epsilon history
// (`update_substep`), exactly as `TableauScheduler.rowSample` /
// `ZImageScheduler.intermediateStep` do here, so nothing else has to change.
//
// What is NOT modelled, deliberately, because the Krea 2 path never sets it:
// `overshoot` / `overshoot_substep` (0 — which makes `h` and `h_new` the plain
// deterministic step sizes, so the tableau geometry is untouched by eta),
// `noise_boost_*` (0), brownian/fractal noise samplers (gaussian), guides,
// masks, implicit iterations, and `scale_av_noise` (a no-op unless an
// audio/video split is declared, `:165-169`).

import Foundation
import MLX

// MARK: - The hard-mode variance-preserving split

/// RES4LYF's `get_sde_step(noise_mode = "hard")` on a variance-preserving flow
/// model, in closed form and in `Double`.
///
/// `Double` for the same reason `rhoab` is (§3.13): at the schedule tail the
/// sigmas are `1.2e-1 → 4.3e-2 → 3.2e-4` and `σ_res = sqrt(σ'² − σ_up²)` is a
/// difference of squares of numbers that are close together. Upstream computes
/// it in the sampler's float64; narrowing to `Float` before the subtraction
/// throws away the digits the square root then amplifies.
public enum RES4LYFSDESplit {

  /// `σ_up`, `σ_down` and the `alpha_ratio` the swap rescales by.
  public struct Split: Equatable, Sendable {
    /// How much noise is added: `η · σ'`, clamped as upstream clamps it.
    public let up: Double
    /// The sigma the deterministic part of the step actually integrates to.
    public let down: Double
    /// The variance-preserving rescale applied to the deterministic part.
    public let alpha: Double
  }

  /// The split for one target sigma.
  ///
  /// - Parameters:
  ///   - sigmaNext: the sigma this move targets — the step's `σ'` for a step
  ///     swap, the substep's `s_[row]` for a substep swap.
  ///   - eta: `eta` (step) or `eta_substep` (substep).
  ///   - sigmaMax: the model sampling's `sigma_max`; 1.0 for Krea 2's
  ///     `ModelSamplingFlux`. `alpha` is `(σ_max − σ') + σ_res`, so this is not
  ///     a cosmetic parameter.
  /// - Returns: the identity split `(0, σ', 1)` whenever there is no noise to
  ///   add — `eta == 0`, or a zero target, which is upstream's own early
  ///   return in both `swap_noise_step` and `swap_noise_substep`.
  public static func hardVP(sigmaNext: Double, eta: Double, sigmaMax: Double = 1.0) -> Split {
    guard eta != 0, sigmaNext != 0 else { return Split(up: 0, down: sigmaNext, alpha: 1) }

    // `case "hard": eta_ratio = eta` → `sud = sigma_base · eta_ratio` with
    // `sigma_base = sigma_next` (`:444-445`, `:494-496`).
    var up = sigmaNext * eta
    // `get_sde_coeff` :281-286 — "Maximum VPSDE noise level exceeded": at
    // η ≥ 1 the residual would be sqrt(≤ 0), so upstream backs off to 0.9999·σ'.
    if up >= sigmaNext {
      up = eta >= 1 ? sigmaNext * 0.9999 : sigmaNext * eta
    }
    let residual = (sigmaNext * sigmaNext - up * up).squareRoot()
    let alpha = (sigmaMax - sigmaNext) + residual
    return Split(up: up, down: residual / alpha, alpha: alpha)
  }
}

// MARK: - Where a latent keeps its channels

/// Which axis of the working latent carries the model's channels, for the
/// per-channel z-score upstream applies to every draw.
///
/// It is a parameter and not an assumption because Krea 2's loop does NOT run
/// on the VAE-domain latent: `Krea2Pipeline` patchifies `(1, C, H, W)` into
/// `(1, tokens, C·p·p)` before the first step and unpatchifies after the last,
/// so the array the injector sees has no channel axis at all.
public enum RES4LYFNoiseLayout: Equatable, Sendable {
  /// `(B, C, …)` — the oracle's latent layout, and the VAE-domain latent.
  case channelsAtAxis1
  /// Krea 2's patchified working latent `(B, tokens, C·p·p)`: channel `c` owns
  /// the contiguous slice `c·p² ..< (c+1)·p²` of the last axis
  /// (`Krea2Pipeline.patchify` reshapes `(b, c, h, p, w, p)` and transposes to
  /// `(b, h, w, c, p, p)`, so `c` is the OUTER index of the trailing group).
  case patchifiedTrailing(channels: Int)
  /// No channel structure the injector can see: normalise over the whole
  /// tensor. Never used by the Krea 2 path; here so a caller with an opaque
  /// latent has a documented option rather than a wrong one.
  case whole
}

// MARK: - The noise stream

/// One draw of unit noise, already carrying upstream's
/// `normalize_zscore(channelwise = True)` — the injector multiplies by `σ_up`
/// and does not normalise again.
///
/// A seam, not an abstraction for its own sake: MLX's PRNG is not torch's, so
/// the T2 trace gates cannot re-derive the oracle's realisation and instead
/// feed the fixture's RECORDED noise tensors through this protocol. Everything
/// else in the SDE — the split, the epsilon, the substep the swap happens at,
/// and the ORDER of the draws — is then still under test.
public protocol RES4LYFNoiseStream: AnyObject {
  /// The next draw, shaped and typed like `sample`.
  func next(like sample: MLXArray) -> MLXArray
}

/// The production stream: gaussian noise from a seeded, self-contained MLX key
/// chain, z-scored per channel.
///
/// **Self-contained on purpose.** It never touches MLX's global RNG, which
/// `Krea2Pipeline` seeds with the request seed to sample the initial latent —
/// a stream that consumed the global generator would move that latent as a
/// side effect of turning `eta` on, and the two would stop being separable.
public final class RES4LYFGaussianNoiseStream: RES4LYFNoiseStream {
  private var key: MLXArray
  private let layout: RES4LYFNoiseLayout

  public init(seed: UInt64, layout: RES4LYFNoiseLayout) {
    self.key = MLXRandom.key(seed)
    self.layout = layout
  }

  public func next(like sample: MLXArray) -> MLXArray {
    // Split rather than reseed: the chain is a sequence, and `seed + n` keys
    // would be a family of unrelated streams whose ordering means nothing.
    let keys = MLXRandom.split(key: key, into: 2)
    key = keys[0]
    let raw = MLXRandom.normal(sample.shape, dtype: .float32, key: keys[1])
    return RES4LYFNoiseNormalization.zscore(raw, layout: layout).asType(sample.dtype)
  }
}

/// Upstream's `normalize_zscore(noise, channelwise = True)`.
///
/// Per channel, mean 0 and **unbiased** (`ddof = 1`) standard deviation 1 —
/// torch's `Tensor.std()` default. Verified against the fixtures: every
/// recorded `noise` tensor has per-channel mean < 3e-8 and per-channel
/// `ddof = 1` std of exactly 1.0, while its GLOBAL std is 0.9926.
public enum RES4LYFNoiseNormalization {

  public static func zscore(_ noise: MLXArray, layout: RES4LYFNoiseLayout) -> MLXArray {
    switch layout {
    case .whole:
      return normalise(noise, axes: Array(0..<noise.ndim))

    case .channelsAtAxis1:
      guard noise.ndim >= 3 else { return normalise(noise, axes: Array(0..<noise.ndim)) }
      // Reduce over everything but batch and channel, so each (b, c) plane is
      // its own population — what `channelwise` means at batch 1, which is the
      // only batch this pipeline renders.
      return normalise(noise, axes: Array(2..<noise.ndim))

    case .patchifiedTrailing(let channels):
      let last = noise.dim(noise.ndim - 1)
      guard channels > 0, last % channels == 0 else {
        // A latent whose trailing axis is not a whole number of channel groups
        // is not the layout this case describes; normalising it per "channel"
        // would be an invention. Fall back to the honest whole-tensor z-score.
        return normalise(noise, axes: Array(0..<noise.ndim))
      }
      let per = last / channels
      var grouped = noise.shape
      grouped[noise.ndim - 1] = channels
      grouped.append(per)
      // (…, C, p²): reduce over every axis but batch and C — i.e. the token
      // axis and the within-patch axis.
      let reshaped = noise.reshaped(grouped)
      let axes = Array(1..<(reshaped.ndim - 2)) + [reshaped.ndim - 1]
      return normalise(reshaped, axes: axes).reshaped(noise.shape)
    }
  }

  private static func normalise(_ x: MLXArray, axes: [Int]) -> MLXArray {
    guard !axes.isEmpty else { return x }
    let mean = x.mean(axes: axes, keepDims: true)
    let centred = x - mean
    let std = MLX.std(centred, axes: axes, keepDims: true, ddof: 1)
    return centred / std
  }
}

// MARK: - The injector

/// RES4LYF's SDE re-noise, as a ``SDENoiseInjector``.
///
/// **One instance per run.** It owns two noise streams whose ORDER is part of
/// the reproduction (upstream draws step noise from `noise_sampler` and
/// substep noise from `noise_sampler2`, two generators with different seeds),
/// and it caches the grid it is driven on.
public final class RES4LYFSDENoiseInjector: SDENoiseInjector {

  /// `eta` — the step-level SDE strength. 0 makes every hook a no-op.
  public let eta: Double
  /// `eta_substep`. `ClownsharKSampler_Beta` sets it equal to `eta`, and the
  /// T2 fixtures record exactly that; it is separate here because upstream's
  /// two knobs are separate.
  public let etaSubstep: Double
  /// `s_noise` — a multiplier on the injected noise, 1.0 on this path.
  public let sNoise: Double
  /// `s_noise_substep`.
  public let sNoiseSubstep: Double
  /// The model sampling's `sigma_max`; 1.0 under `ModelSamplingFlux`.
  public let sigmaMax: Double

  private let stepNoise: RES4LYFNoiseStream
  private let substepNoise: RES4LYFNoiseStream

  /// The scheduler's grid, read once. `ZImageScheduler.sigmas` is an MLXArray
  /// and pulling it per step would be a GPU sync per step for numbers that do
  /// not change during a run.
  private var grid: [Float]?

  public init(
    eta: Double,
    etaSubstep: Double,
    sNoise: Double = 1.0,
    sNoiseSubstep: Double = 1.0,
    sigmaMax: Double = 1.0,
    stepNoise: RES4LYFNoiseStream,
    substepNoise: RES4LYFNoiseStream
  ) {
    self.eta = eta
    self.etaSubstep = etaSubstep
    self.sNoise = sNoise
    self.sNoiseSubstep = sNoiseSubstep
    self.sigmaMax = sigmaMax
    self.stepNoise = stepNoise
    self.substepNoise = substepNoise
  }

  /// The production convenience: `eta` on both knobs, `s_noise` 1.0, and the
  /// two seeded gaussian streams upstream's seeds describe.
  public convenience init(
    eta: Double,
    stageSeed: UInt64,
    layout: RES4LYFNoiseLayout,
    sigmaMax: Double = 1.0
  ) {
    self.init(
      eta: eta, etaSubstep: eta, sNoise: 1.0, sNoiseSubstep: 1.0, sigmaMax: sigmaMax,
      stepNoise: RES4LYFGaussianNoiseStream(
        seed: Self.stepNoiseSeed(stageSeed: stageSeed), layout: layout),
      substepNoise: RES4LYFGaussianNoiseStream(
        seed: Self.substepNoiseSeed(stageSeed: stageSeed), layout: layout))
  }

  // MARK: Seeds

  /// `rk_sampler_beta.py:364` — `MAX_STEPS`, the offset upstream uses so the
  /// step and substep generators cannot reuse a seed (`constants.py:1`).
  public static let substepSeedOffset: UInt64 = 10_000

  /// The step stream's seed: `stage seed + 1`.
  ///
  /// `SharkSampler` hands `sample_rk_beta` `noise_seed = seed + 1` — the
  /// fixture generator passes exactly that and the manifests record it
  /// (`seed: 4242`, `noise_seed_sde: 4243`). It is derived from the STAGE's
  /// seed, so a staged render's two stages have two streams and changing only
  /// `stage2.seed` changes only stage 2 (AC-27).
  public static func stepNoiseSeed(stageSeed: UInt64) -> UInt64 { stageSeed &+ 1 }

  /// The substep stream's seed: `noise_seed + MAX_STEPS`.
  public static func substepNoiseSeed(stageSeed: UInt64) -> UInt64 {
    stepNoiseSeed(stageSeed: stageSeed) &+ substepSeedOffset
  }

  // MARK: The hooks

  /// `swap_noise_step` (`rk_sampler_beta.py:2027`): after the step's commit,
  /// before the model-free tail — which is why S-FIX-1 records the step's
  /// `(x₀, x_next)` BEFORE this hook runs and applies the tail to the sample
  /// this hook leaves behind (`:1997`, `:2017-2022`, `:2202`).
  public func inject(
    sample: inout MLXArray, x0: MLXArray, timestepIndex: Int, scheduler: any ZImageScheduler
  ) {
    guard eta != 0 else { return }
    let sigmas = gridValues(scheduler)
    guard timestepIndex + 1 < sigmas.count else { return }
    let sigma = Double(sigmas[timestepIndex])
    let sigmaNext = Double(sigmas[timestepIndex + 1])
    let split = RES4LYFSDESplit.hardVP(sigmaNext: sigmaNext, eta: eta, sigmaMax: sigmaMax)
    guard split.up != 0 else { return }
    swap(
      &sample, x0: x0, sigma: sigma, target: sigmaNext, split: split,
      noise: stepNoise.next(like: sample), sNoise: sNoise)
  }

  /// `swap_noise_substep` (`rk_sampler_beta.py:1874`): after row `row`'s sample
  /// has been built from `x₀` and the rows before it, and BEFORE the model is
  /// evaluated on it.
  ///
  /// `row` is the driver's row index — the row whose sample this is — so it is
  /// upstream's `row + row_offset`, and its target sigma is that row's own
  /// evaluation sigma. The final row is never passed here: upstream's
  /// `row < rows − row_offset − multistep_stages` guard
  /// (`rk_noise_sampler_beta.py:363`) leaves the last update unnoised, and the
  /// step-level swap is what moves it to `σ'`.
  public func injectSubstep(
    sample: inout MLXArray, x0: MLXArray, timestepIndex: Int, row: Int,
    scheduler: any ZImageScheduler
  ) {
    guard etaSubstep != 0 else { return }
    let sigmas = gridValues(scheduler)
    guard timestepIndex < sigmas.count else { return }
    let sigma = Double(sigmas[timestepIndex])
    guard let target = Self.substepSigma(scheduler, timestepIndex: timestepIndex, row: row) else {
      // A scheduler that produced a row sample and will not say what sigma it
      // is evaluated at cannot be re-noised — and guessing σ' is exactly the
      // silent substitution D18 exists to forbid.
      preconditionFailure(
        "eta != 0 needs the substep sigma for row \(row) of step \(timestepIndex), and this "
          + "scheduler reports none")
    }
    let split = RES4LYFSDESplit.hardVP(
      sigmaNext: Double(target), eta: etaSubstep, sigmaMax: sigmaMax)
    guard split.up != 0 else { return }
    swap(
      &sample, x0: x0, sigma: sigma, target: Double(target), split: split,
      noise: substepNoise.next(like: sample), sNoise: sNoiseSubstep)
  }

  // MARK: Internals

  /// The one arithmetic statement, shared by both swaps because upstream's two
  /// functions differ only in which sigma, which split and which generator
  /// they use. Anchored at the STEP's `σ` and `x₀` in both cases.
  ///
  /// Computed in `float32` and cast back: the working latent is `bfloat16`,
  /// and `eps' = (x₀ − x_next)/(σ − σ_target)` is a cancellation over a
  /// difference that shrinks to `4e-2` at the tail — 8 mantissa bits are not
  /// enough to divide by it. Upstream's `work_dtype` is float32 for the same
  /// reason. At `eta == 0` this is never reached, so the default path pays
  /// neither the cast nor the precision.
  private func swap(
    _ x: inout MLXArray, x0: MLXArray, sigma: Double, target: Double,
    split: RES4LYFSDESplit.Split, noise: MLXArray, sNoise: Double
  ) {
    let dtype = x.dtype
    let denominator = Float(sigma - target)
    precondition(
      denominator != 0,
      "the SDE swap needs a non-degenerate move (σ \(sigma) == target \(target))")
    let x0f = x0.asType(.float32)
    let eps = (x0f - x.asType(.float32)) / denominator
    let denoised = x0f - Float(sigma) * eps
    let noised =
      Float(split.alpha) * (denoised + Float(split.down) * eps)
      + Float(split.up) * noise.asType(.float32) * Float(sNoise)
    x = noised.asType(dtype)
  }

  private func gridValues(_ scheduler: any ZImageScheduler) -> [Float] {
    if let grid { return grid }
    let values = scheduler.sigmas.asArray(Float.self)
    grid = values
    return values
  }

  /// The sigma row `row` is evaluated at — upstream's `s_[row]`.
  ///
  /// A tableau scheduler answers directly; the 2-row branch's single non-final
  /// row is the scheduler's `intermediateSigma`, which for `res_2s` is the
  /// genuine substep `σ·e^{−c₂h}` and NOT `σ_{i+1}` (§3.3).
  static func substepSigma(
    _ scheduler: any ZImageScheduler, timestepIndex: Int, row: Int
  ) -> Float? {
    if let tableau = scheduler as? TableauScheduler {
      guard row >= 0 && row < tableau.rows else { return nil }
      return tableau.rowSigma(timestepIndex: timestepIndex, row: row)
    }
    guard row == 1, scheduler.requiresIntermediateEvaluation else { return nil }
    return scheduler.intermediateSigma(timestepIndex: timestepIndex)
  }
}
