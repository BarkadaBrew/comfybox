// RES4LYFBongMath.swift — RES4LYF's `bongmath` fixed point, ported exactly
// (parity tier T3).
//
// WP-E16 (docs/FDD-krea2-raw-recipe.md §3.13, §4 AC-26, D18), against the
// pinned upstream at `scripts/oracles/upstream/res4lyf/beta/`, commit
// `26036f647ca15d3048a193daf99a40cecfc3820d`:
//   * `rk_method_beta.py:695-767`  — `RK_Method.bong_iter`, the fixed point.
//   * `rk_sampler_beta.py:1967-1974` — where it is called and what guards it.
//   * `rk_method_beta.py:985-1000` / `:1116-1128` — `get_epsilon` in the
//     exponential and linear frames (`noise_anchor = 1.0`).
//
// WHAT IT DOES, and why it is not decoration.
//
// The SDE (T2, WP-E15) re-noises every non-final ROW of the tableau: after
// `update_substep` builds `x_[row+1] = x₀ + h·Σⱼ a₍row+1₎ⱼ·εⱼ`, `swap_noise_substep`
// moves it off that manifold. The row is then no longer something the tableau
// could have produced from `x₀`, and every later row, the commit, the step-level
// swap and the model-free tail are all still written in `x₀`.
//
// `bongmath` repairs that by moving the ANCHOR instead of the row: it solves
//
//     x₀  such that   x₀ + h·Σⱼ a₍row+1₎ⱼ·εⱼ(x₀)  ==  x_[row+1]      (the re-noised one)
//
// by 100 rounds of the obvious fixed-point iteration
//
//     x₀ ← x_[row+1] − h·Σⱼ a₍row+1₎ⱼ·εⱼ(x₀_prev)
//
// which upstream writes out longhand (rebuilding `x_[rr]` and `eps_[rr]` for
// every earlier row each round). Two facts make the port short:
//
//   1. **`εⱼ` is a pure function of `x₀` and the row's data prediction.** At
//      `noise_anchor = 1.0` — the value `ClownsharKSampler_Beta` passes and the
//      fixtures record — `get_epsilon` collapses to the x₀-anchored form
//      (`denoisedⱼ − x₀` exponential, `(x₀ − denoisedⱼ)/σ` linear), with no
//      dependence on `x_[rr]` or the substep sigma at all. Upstream's
//      per-round rewrite of `x_[rr]` is therefore dead for our purposes: those
//      rows have already been evaluated and are never read again.
//   2. **`x₀ + h·Σⱼ aᵢⱼ·εⱼ(x₀)` is exactly `rowSample`.** So one round is
//      `x₀ ← target − (rowSample(x₀) − x₀)`, and this file needs no copy of any
//      Butcher tableau — it drives the scheduler's own.
//
// **It makes no model call.** `bong_iter` touches only `data_`, which the rows
// already produced; E18's traces record `model_calls_total` 12 and 6 at T1, T2
// and T3 alike. ``iterate`` returns 0 for that reason, and it returns it rather
// than the loop assuming it (§3.3: reported, never discovered).
//
// CONVERGENCE. One round is an affine map with linear part `h·Σⱼ aᵢⱼ·(∂εⱼ/∂x₀)`.
// In the exponential frame that is `1 − e^{−cᵢh}` (because `Σⱼ aᵢⱼ = cᵢφ₁(−cᵢh)`),
// strictly inside `(0, 1)` for every `h > 0`; in the linear frame it is
// `−cᵢ·h/σ` with `|h| < σ`, so `< 1` too. The iteration is a contraction on
// both frames and reaches float32's fixed point well inside upstream's 100
// rounds — which is why upstream can afford a fixed count and no residual
// check. ``RES4LYFBongMathParityTests.testTheFixedPointIsAFixedPoint`` measures
// the residual on the production grid rather than trusting that argument.

import Foundation
import MLX

/// RES4LYF's `bong_iter`, as a ``BongMath``.
///
/// **One instance per run.** It records what it did — how many rebases it made
/// and how many upstream's guards refused — so ``Krea2RunTrace`` can report the
/// fixed point rather than assert it.
public final class RES4LYFBongMath: BongMath {

  /// `for i in range(100)` (`rk_method_beta.py:721`). A fixed count, not a
  /// tolerance: upstream has no residual test, and matching the count is what
  /// makes the T3 traces reproduce bit-for-bit rather than nearly.
  public static let fixedPointIterations = 100

  /// `if sigma > 0.03` (`rk_sampler_beta.py:1971`). Below it upstream leaves
  /// the anchor alone — the tail is where `h` explodes and the fixed point
  /// would be solving for a step it no longer describes.
  public static let minimumSigma: Double = 0.03

  /// The model sampling's `sigma_min` — `3.1575e-4` under Krea 2's registered
  /// `ModelSamplingFlux(1.15)`. `s_[row] > σ_min` is one of upstream's guards.
  public let sigmaMin: Double
  /// The model sampling's `sigma_max`; 1.0 for Krea 2. `h < σ_max/2` is the
  /// guard that refuses the last steps of a 6-step `res_2s` run.
  public let sigmaMax: Double
  /// Rounds per rebase. Upstream's 100; a parameter only so the contraction
  /// can be measured at other counts in a test.
  public let iterations: Int

  /// Rebases actually performed — one per row whose guards passed.
  public private(set) var rebases = 0
  /// Rows upstream's guards refused. Counted, not silent: a run that expected
  /// the fixed point and got none should be able to see that in the record.
  public private(set) var refusals = 0
  /// Grid indices the fixed point ran on, in order, de-duplicated. The shape
  /// of E18's per-step `bongmath` array.
  public private(set) var rebasedSteps: [Int] = []
  /// Model evaluations this hook made: 0, always. Kept as a counter rather
  /// than a constant so the number reported is the number that happened.
  public private(set) var extraModelEvals = 0

  /// The scheduler's grid, read once — the same reason
  /// ``RES4LYFSDENoiseInjector`` caches it.
  private var grid: [Float]?

  public init(
    sigmaMin: Double,
    sigmaMax: Double = 1.0,
    iterations: Int = RES4LYFBongMath.fixedPointIterations
  ) {
    precondition(iterations > 0, "the fixed point needs at least one round")
    self.sigmaMin = sigmaMin
    self.sigmaMax = sigmaMax
    self.iterations = iterations
  }

  // MARK: - The hook

  /// `bong_iter` for one row: re-derive `x0` so `buildRowSample(x0)` is
  /// `rowSample` again.
  ///
  /// - Parameters:
  ///   - x0: the step's anchor, updated in place. Upstream's `x_0`.
  ///   - rowSample: `x_[row + 1]` — the row's sample **after** the T2 substep
  ///     re-noise, which is the whole reason the anchor has to move.
  ///   - row: upstream's `row`, i.e. the index of the row whose UPDATE built
  ///     `rowSample`. The driver calls with `driverRow − 1`, and never for the
  ///     committing row (upstream's `row < rows − row_offset` guard, which the
  ///     driver satisfies structurally).
  ///   - buildRowSample: the scheduler's own `x₀ + h·Σⱼ aᵢⱼ·εⱼ(x₀)` for this
  ///     row. Supplied by the driver because only it knows which branch built
  ///     the sample.
  ///   - evaluate: unused. The seam carries it because ``BongMath`` promises a
  ///     conformer MAY evaluate and must then say so; RES4LYF's never does.
  /// - Returns: extra model evaluations made — 0.
  public func iterate(
    x0: inout MLXArray,
    rowSample: MLXArray,
    timestepIndex: Int,
    row: Int,
    scheduler: any ZImageScheduler,
    buildRowSample: (_ x0: MLXArray) -> MLXArray,
    evaluate: (_ latent: MLXArray, _ sigma: Float) -> MLXArray
  ) -> Int {
    let sigmas = gridValues(scheduler)
    guard timestepIndex >= 0, timestepIndex + 1 < sigmas.count else {
      refusals += 1
      return 0
    }

    // The two things this hook cannot proceed without come FIRST, so that a
    // scheduler it does not know how to interrogate fails loudly on every
    // call rather than only on the steps whose guards would have passed.
    //
    // `h` is the FRAME's step size (`NS.h`), not `σ' − σ`: guessing the linear
    // form for an exponential sampler is off by a factor of 20 at the tail and
    // would refuse the wrong steps — the silent substitution D18 forbids.
    guard let framed = scheduler as? RES4LYFFrameScheduler else {
      preconditionFailure(
        "bongmath needs the frame step size `h` for step \(timestepIndex), and this scheduler "
          + "is not a RES4LYFFrameScheduler; σ' − σ is not a valid substitute for the "
          + "exponential frame (§3.13)")
    }
    // `NS.s_[row]` — the sigma the row being rebased was evaluated at.
    guard let rowSigma = Self.rowSigma(
      scheduler, timestepIndex: timestepIndex, row: row, grid: sigmas)
    else {
      preconditionFailure(
        "bongmath needs the substep sigma for row \(row) of step \(timestepIndex), and this "
          + "scheduler reports none")
    }

    // Upstream's three guards, all of which must hold
    // (`rk_sampler_beta.py:1967` and `:1971`):
    //     NS.s_[row] > RK.sigma_min      NS.h < RK.sigma_max/2      sigma > 0.03
    let sigma = Double(sigmas[timestepIndex])
    let h = Double(framed.frameStepSize(timestepIndex: timestepIndex))
    guard Double(rowSigma) > sigmaMin, h < sigmaMax / 2, sigma > Self.minimumSigma else {
      refusals += 1
      return 0
    }

    // The iteration itself, in float32 — upstream's `work_dtype`, and the same
    // reason ``RES4LYFSDENoiseInjector.swap`` uses it: the production latent is
    // bfloat16 and 8 mantissa bits will not carry a contraction to its fixed
    // point. The anchor is handed back in the sample's own dtype, so the loop's
    // latent stays exactly as wide as it was.
    let dtype = x0.dtype
    let target = rowSample.asType(.float32)
    var candidate = x0.asType(.float32)
    for _ in 0..<iterations {
      // `x_0 = x_[row+1] − h·zum(row+1, eps_)`, written through the
      // scheduler's own row arithmetic: `h·zum(…) == buildRowSample(x) − x`.
      candidate = target - (buildRowSample(candidate) - candidate)
      // Materialise each round: 100 chained lazy graphs over a full-resolution
      // latent would hold ~100 intermediates alive at once.
      MLX.eval(candidate)
    }
    x0 = candidate.asType(dtype)

    rebases += 1
    if rebasedSteps.last != timestepIndex { rebasedSteps.append(timestepIndex) }
    return 0
  }

  // MARK: - Internals

  /// `NS.s_[row]` — the sigma row `row` was evaluated at.
  ///
  /// Row 0 is always the step's own sigma, on every branch; rows above it are
  /// the scheduler's substeps, which is the same question
  /// ``RES4LYFSDENoiseInjector/substepSigma(_:timestepIndex:row:)`` answers for
  /// the T2 swap, so it is answered in one place.
  static func rowSigma(
    _ scheduler: any ZImageScheduler, timestepIndex: Int, row: Int, grid: [Float]
  ) -> Float? {
    guard row > 0 else { return grid[timestepIndex] }
    return RES4LYFSDENoiseInjector.substepSigma(
      scheduler, timestepIndex: timestepIndex, row: row)
  }

  private func gridValues(_ scheduler: any ZImageScheduler) -> [Float] {
    let count = scheduler.sigmas.dim(0)
    if let grid {
      // The cache is only sound while the hook is driven on ONE run of ONE
      // scheduler, which is what "one instance per run" means. A grid that
      // changed length underneath it would silently guard the wrong steps.
      precondition(
        grid.count == count,
        "bongmath cached a \(grid.count)-sigma grid and is now driven on a \(count)-sigma one; "
          + "build one RES4LYFBongMath per run")
      return grid
    }
    let values = scheduler.sigmas.asArray(Float.self)
    grid = values
    return values
  }
}
