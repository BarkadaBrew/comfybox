// Krea2DenoiseLoop.swift — the one denoise-loop driver for Krea 2.
//
// WP-E3 (docs/FDD-krea2-raw-recipe.md §3.3, D1, D2). A pure function: it
// knows nothing about transformers, conditioning or VAEs. `evaluate` is the
// real transformer (CFG already combined) in production and a closed-form
// velocity field in the tests, which is what makes the byte-identity gate,
// the eval-count matrix and the reset discipline assertable with no weights.
//
// Three dispatch branches from the first commit (D1), in this order:
//   N-row  `TableauScheduler`      rows × evaluate → commit
//   2-row  `requiresIntermediateEvaluation` → intermediateStep / finalizeStep
//   1-row  `step`
// The default euler + krea2 path takes the 1-row branch, whose update
// (`FlowMatchEulerScheduler.step`: `sample + modelOutput * dt`, dt cast to the
// sample dtype from a float32 difference) is the same arithmetic as the
// pre-change inline `img + (tp - tc) * v` — asserted bit-for-bit in
// `Krea2DenoiseLoopTests.testDefaultPathBitIdenticalToInlineLoop`.

import Foundation
import MLX

/// Parity tier T2 seam (WP-E15): RES4LYF SDE `eta`. The conformer owns its
/// RNG and its `sigma_down` / `sigma_up` split. Reference-typed so a stateful
/// injector persists across steps and across rows.
///
/// TWO hooks, because RES4LYF re-noises twice per step and the second one is
/// not optional decoration:
///
///   * ``injectSubstep(sample:x0:timestepIndex:row:scheduler:)`` runs after a
///     non-final row's sample has been built and BEFORE the model is evaluated
///     on it (`rk_sampler_beta.py:1874`) — so the row's evaluation sees the
///     re-noised sample. `res_2s` reproduces its T2 trace only with this.
///   * ``inject(sample:x0:timestepIndex:scheduler:)`` runs once per step,
///     after the step's commit (`rk_sampler_beta.py:2029`). The `x0` it is
///     handed is the one T3 may have re-derived during the row loop, because
///     that is the `x_0` upstream's `swap_noise_step` reads.
///
/// Both are handed the step's own `x0` — the sample the step started from —
/// because upstream's swap is written in `eps' = (x₀ − x)/(σ − σ_target)` and
/// anchored at the STEP's sigma, in the substep case too. The driver is the
/// only party that still has `x0` when the hooks fire.
///
/// Ordering with S-FIX-1's model-free tail is fixed and load-bearing: the tail
/// epsilon is computed from the PRE-swap `(x₀, x_next)` and the tail is applied
/// to the POST-swap sample (`rk_sampler_beta.py:1997, 2017-2022, 2202`).
public protocol SDENoiseInjector: AnyObject {
  func inject(
    sample: inout MLXArray, x0: MLXArray, timestepIndex: Int, scheduler: any ZImageScheduler)

  /// Re-noise the sample row `row` (≥ 1, never the committing row) will be
  /// evaluated on. Defaulted to a no-op so an injector that models only the
  /// step-level swap stays valid.
  func injectSubstep(
    sample: inout MLXArray, x0: MLXArray, timestepIndex: Int, row: Int,
    scheduler: any ZImageScheduler)
}

extension SDENoiseInjector {
  public func injectSubstep(
    sample: inout MLXArray, x0: MLXArray, timestepIndex: Int, row: Int,
    scheduler: any ZImageScheduler
  ) {}
}

/// Parity tier T3 seam (WP-E16): `bongmath`.
///
/// **It is a ROW hook, not a step hook** — the one correction E16 made to E3's
/// declared seam, and the reason is upstream's, not stylistic. `bong_iter`
/// runs inside RES4LYF's row loop (`rk_sampler_beta.py:1967-1974`), after a
/// non-final row's sample has been built AND re-noised by the T2 substep swap,
/// and what it rewrites is the STEP's `x₀` — not the sample. Everything after
/// it in the step then reads the re-derived anchor: the remaining rows, the
/// commit, the T2 step-level swap, and the epsilon the model-free tail is
/// written in. A hook that ran once after the commit could not reach any of
/// them, and E18's traces measure the difference (`step/x_0` is 1.89 away from
/// the sample the step started on, at step 0 of `res2s_beta6_T3`).
///
/// `buildRowSample` is supplied by the driver because only the driver knows
/// which branch built this row — the N-row `TableauScheduler.rowSample` or the
/// 2-row `intermediateStep`. The conformer inverts it and never has to carry a
/// copy of anyone's Butcher tableau.
///
/// A conformer MAY evaluate the model — hence `evaluate`, and hence the return
/// value, which is added to `Stats.evaluateCalls`, so the eval count stays
/// reported rather than discovered (§3.3). ``RES4LYFBongMath`` returns 0:
/// upstream's fixed point is algebraic.
public protocol BongMath: AnyObject {
  func iterate(
    x0: inout MLXArray,
    rowSample: MLXArray,
    timestepIndex: Int,
    row: Int,
    scheduler: any ZImageScheduler,
    buildRowSample: (_ x0: MLXArray) -> MLXArray,
    evaluate: (_ latent: MLXArray, _ sigma: Float) -> MLXArray
  ) -> Int
}

public enum Krea2DenoiseLoop {

  /// What the loop did — the numbers `RenderRecipe` (WP-E10) reports.
  ///
  /// `modelEvals` is the counted truth. `rowsAtStart` is a label, and the two
  /// are only related by `stepsRun × rowsAtStart × cfg` for a scheduler whose
  /// row count is CONSTANT across the run — which is every scheduler that
  /// exists today, and deliberately not every scheduler that will. See
  /// ``Stats/rowsAtStart``.
  public struct Stats: Equatable, Sendable {
    /// Steps actually taken: `numInferenceSteps − startIndex`.
    public let stepsRun: Int

    /// Model evaluations the scheduler asked for at the FIRST step of this run
    /// (step `startIndex`): 1, 2, or a `TableauScheduler`'s `rows`.
    ///
    /// It is sampled once, on purpose, because it is a description of the
    /// sampler and not an accounting quantity. The driver re-reads
    /// `tableau.rows` on every step, so a scheduler that changes its row count
    /// mid-run is dispatched correctly — but this field will still say what
    /// the first step wanted. `deis_3m` is exactly that case (AC-24: a
    /// `ralston_3s` warm-up over steps 0…3, then multistep at 1 row), and for
    /// it `stepsRun × rowsAtStart` would OVERCOUNT by a wide margin.
    ///
    /// Never multiply this to obtain a cost. ``modelEvals`` is counted.
    public let rowsAtStart: Int

    /// Calls actually made to `evaluate` — one per row per step, whatever the
    /// rows were on that step, plus any a T3 hook reported making.
    public let evaluateCalls: Int

    /// Transformer forwards: `evaluateCalls × modelEvalsPerEvaluate`. The CFG
    /// multiplier lives in the caller, which is the only party that knows
    /// whether one `evaluate` is one pass or two. This is the number §3.3
    /// requires be reported rather than discovered, and it is a count, not a
    /// product of labels.
    public let modelEvals: Int

    /// The sigma RES4LYF's model-free `σ_min → 0` conversion ran from, or `nil`
    /// when the run finished without one.
    ///
    /// Non-nil means the last solver step landed on the model's `sigma_min`
    /// and the loop then converted to zero with `x − σ_min·eps`, reusing the
    /// last step's own epsilon (`rk_sampler_beta.py:1997,2202`). It costs **no**
    /// model evaluation, so it is deliberately absent from ``modelEvals`` and
    /// ``evaluateCalls``; it is not a solver step either, so ``stepsRun``
    /// excludes it too. ``Krea2RunTrace`` reports the grid including the zero
    /// it landed on.
    public let finalConversionSigma: Float?

    public init(
      stepsRun: Int,
      rowsAtStart: Int,
      evaluateCalls: Int,
      modelEvals: Int,
      finalConversionSigma: Float? = nil
    ) {
      self.stepsRun = stepsRun
      self.rowsAtStart = rowsAtStart
      self.evaluateCalls = evaluateCalls
      self.modelEvals = modelEvals
      self.finalConversionSigma = finalConversionSigma
    }
  }

  /// Drive `scheduler` from `startIndex` to the end of its grid.
  ///
  /// - Parameters:
  ///   - scheduler: reset before the first step (multistep caches —
  ///     `DPMPlusPlus2MScheduler.previousOutput`, `RES2sScheduler.firstModelOutput`
  ///     — never leak across runs, AC-13). `inout` so stochastic samplers keep
  ///     their key state for the caller's record.
  ///   - initialSample: the latent at `sigmas[startIndex]` (pure noise for
  ///     t2i; the float32-mixed source for img2img, §3.3).
  ///   - startIndex: first grid index to step from (img2img's partial start).
  ///   - modelEvalsPerEvaluate: transformer forwards per `evaluate` call —
  ///     2 under CFG, else 1. Only affects `Stats.modelEvals`.
  ///   - evaluate: `(latent, sigma) → velocity`. `sigma` is Krea 2's `t`
  ///     directly — the pipelines pass it into the transformer with no
  ///     `(1000 − t)/1000` renormalisation.
  ///   - noise / bongmath: T2 / T3 hooks; `nil` (the default) is today's path.
  ///   - progress: `(stepsCompletedIndex + 1, numInferenceSteps)` after each
  ///     step, exactly as the inline loop reported it.
  public static func run(
    scheduler: inout any ZImageScheduler,
    initialSample: MLXArray,
    startIndex: Int = 0,
    modelEvalsPerEvaluate: Int = 1,
    evaluate: (_ latent: MLXArray, _ sigma: Float) -> MLXArray,
    noise: SDENoiseInjector? = nil,
    bongmath: BongMath? = nil,
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> (sample: MLXArray, stats: Stats) {
    let total = scheduler.numInferenceSteps
    precondition(startIndex >= 0 && startIndex <= total, "startIndex \(startIndex) outside 0...\(total)")
    precondition(modelEvalsPerEvaluate >= 1, "modelEvalsPerEvaluate must be ≥ 1")

    // 1. Reset multistep state before the first step.
    scheduler.reset()

    // The grid as Floats, read once: `evaluate` receives exactly the stored
    // float32 values (`ts[i]` before this WP), with no per-step GPU sync.
    let sigmas = scheduler.sigmas.asArray(Float.self)

    // The row count at the first step of this run. Dispatch below re-reads
    // `tableau.rows` every step; this is the label, not the multiplier.
    let rowsAtStart: Int
    if let tableau = scheduler as? TableauScheduler {
      rowsAtStart = tableau.rows
    } else if scheduler.requiresIntermediateEvaluation {
      rowsAtStart = 2
    } else {
      rowsAtStart = 1
    }

    var x = initialSample
    var evaluateCalls = 0

    // RES4LYF's model-free tail (`rk_sampler_beta.py:1997,2202`) needs the LAST
    // step's own epsilon, `(x₀ − x_next)/(σ − σ_next)`, taken BEFORE that step's
    // T2 re-noise — exactly as upstream computes it. Recorded per step so the
    // conversion after the loop reads no scheduler state and makes no model call.
    //
    // Read once: the conversion sigma is a property of the grid, and a run that
    // has none (every non-RES4LYF sampler, the default euler path included)
    // must not pay to retain a step's latents for it.
    let tailSigma = scheduler.finalConversionSigma
    var lastStep: (index: Int, x0: MLXArray, xNext: MLXArray, sigma: Float, sigmaNext: Float)?

    for i in startIndex..<total {
      // comfybox#304: step-boundary cancellation check, matching the
      // Flux2/Fibo idiom (Flux2Pipeline.swift, FiboPipeline.swift) — one
      // check per sampler step, CancellationError propagates unmodified.
      try Task.checkCancellation()
      let sigma = sigmas[i]
      // The step's own `x₀`. Retained only when someone downstream needs it —
      // the model-free tail, a T2 injector whose swap is written in it, or the
      // T3 fixed point, which REWRITES it — so the default euler path still
      // keeps no second latent alive per step.
      //
      // `var`, because bongmath moves the anchor mid-step and every consumer
      // below (the later rows, the commit, the T2 step swap, `lastStep`) must
      // see the moved one. With no T3 hook it is assigned once and never
      // reassigned, so the pre-E16 arithmetic is untouched op for op.
      var stepStart = (tailSigma != nil || noise != nil || bongmath != nil) ? x : nil

      if var tableau = scheduler as? TableauScheduler {
        // 3. N-row: rows evaluations at rowSigma / rowSample, then commit.
        var k: [MLXArray] = []
        k.reserveCapacity(tableau.rows)
        for r in 0..<tableau.rows {
          let anchor = stepStart ?? x
          var xr = r == 0 ? x : tableau.rowSample(timestepIndex: i, row: r, x0: anchor, k: k)
          let sr = r == 0 ? sigma : tableau.rowSigma(timestepIndex: i, row: r)
          // T2 substep re-noise: rows 1 ..< rows are upstream's non-final rows
          // (`row + row_offset` for `row < rows − row_offset`), and row 0 is
          // the step's start sample, which is never re-noised.
          if r > 0, let noise, let stepStart {
            noise.injectSubstep(
              sample: &xr, x0: stepStart, timestepIndex: i, row: r, scheduler: tableau)
          }
          // T3: re-anchor on the row the T2 swap just moved. Upstream's `row`
          // is the row whose UPDATE built this sample, i.e. `r − 1`; it never
          // runs for the committing row, which is why the driver's `r` stops
          // at `rows − 1` and this call is inside the loop rather than after
          // it (`rk_method_beta.py:713` — `row < rows − row_offset`).
          if r > 0, let bongmath, var anchor = stepStart {
            let history = k
            evaluateCalls += bongmath.iterate(
              x0: &anchor, rowSample: xr, timestepIndex: i, row: r - 1, scheduler: tableau,
              buildRowSample: { [tableau] x0 in
                var t = tableau
                return t.rowSample(timestepIndex: i, row: r, x0: x0, k: history)
              },
              evaluate: evaluate)
            stepStart = anchor
          }
          let v = evaluate(xr, sr)
          evaluateCalls += 1
          k.append(tableau.modelInput(velocity: v, sample: xr, sigma: sr))
        }
        x = tableau.commit(timestepIndex: i, x0: stepStart ?? x, k: k)
        scheduler = tableau
      } else {
        // 2. First evaluation at the grid sigma; convert per WP-E2.
        let v = evaluate(x, sigma)
        evaluateCalls += 1
        let out = scheduler.modelInput(velocity: v, sample: x, sigma: sigma)

        if scheduler.requiresIntermediateEvaluation,
           var mid = scheduler.intermediateStep(modelOutput: out, timestepIndex: i, sample: x) {
          // 4. 2-row: the second evaluation at the scheduler's own substep
          //    (res_2s: σ·e^{−c₂h}, a genuine substep — not σ_{i+1}).
          // Fail loud: σ_{i+1} is the WRONG substep (§3.3), so there is no
          // fallback to substitute. A scheduler that says it needs a second
          // evaluation and then will not say where owes the caller an answer.
          guard let midSigma = scheduler.intermediateSigma(timestepIndex: i) else {
            preconditionFailure(
              "scheduler requires an intermediate evaluation at step \(i) but returned no "
                + "intermediateSigma; σ_{i+1} is not a valid substitute (§3.3)")
          }
          // T2 substep re-noise: the 2-row branch's one non-final row. For
          // `res_2s` this is the difference between reproducing the T2 trace
          // and not — the second evaluation happens on the re-noised sample.
          if let noise, let stepStart {
            noise.injectSubstep(
              sample: &mid, x0: stepStart, timestepIndex: i, row: 1, scheduler: scheduler)
          }
          // T3, at the same position as in the N-row branch: after the one
          // non-final row has been built and re-noised, before it is evaluated.
          if let bongmath, var anchor = stepStart {
            let snapshot = scheduler
            evaluateCalls += bongmath.iterate(
              x0: &anchor, rowSample: mid, timestepIndex: i, row: 0, scheduler: scheduler,
              buildRowSample: { x0 in
                var s = snapshot
                guard let row = s.intermediateStep(
                  modelOutput: out, timestepIndex: i, sample: x0)
                else {
                  preconditionFailure(
                    "scheduler built an intermediate sample at step \(i) and then refused to "
                      + "rebuild it; the T3 fixed point has nothing to invert")
                }
                return row
              },
              evaluate: evaluate)
            stepStart = anchor
          }
          let vMid = evaluate(mid, midSigma)
          evaluateCalls += 1
          let outMid = scheduler.modelInput(velocity: vMid, sample: mid, sigma: midSigma)
          x = scheduler.finalizeStep(
            originalOutput: out, intermediateOutput: outMid, timestepIndex: i,
            sample: stepStart ?? x)
        } else {
          // 5. 1-row.
          x = scheduler.step(modelOutput: out, timestepIndex: i, sample: x)
        }
      }

      if let stepStart { lastStep = (i, stepStart, x, sigma, sigmas[i + 1]) }

      // 6. The T2 step swap runs AFTER the commit and after `lastStep` has
      //    recorded the pre-swap `(x₀, x_next)` the model-free tail needs —
      //    upstream computes that epsilon before the swap and applies the tail
      //    to the post-swap sample. Both read the anchor T3 may have moved:
      //    `swap_noise_step(x_0, x_next)` and `eps = (x_0 − x_next)/(σ − σ')`
      //    are written in the POST-bongmath `x_0` (`rk_sampler_beta.py:1997`,
      //    `:2029`), and E18 exports `step/x_0` from exactly there.
      //
      //    There is no T3 hook here: upstream's `bong_iter` refuses the
      //    committing row (`row < rows − row_offset`), so every call it makes
      //    has already happened inside the row loop above.
      if let noise, let stepStart {
        noise.inject(sample: &x, x0: stepStart, timestepIndex: i, scheduler: scheduler)
      }

      // 7. Materialise, report.
      MLX.eval(x)
      progress?(i + 1, total)
    }

    // 8. RES4LYF's model-free final conversion, σ_min → 0.
    //
    // The prepared grid (`RES4LYFSigmaPreparation`) stops the solver ON the
    // model's `sigma_min`; upstream then converts to zero with the flow model's
    // `calculate_denoised(σ_min, eps, x) = x − σ_min·eps`, reusing the epsilon
    // the last step already implies. No model evaluation, no scheduler step —
    // so `modelEvals` and `stepsRun` are untouched by it. Upstream guards the
    // tail on having reached the end of the grid (`step == len(sigmas) − 2`);
    // a run that took no step at all reached nothing and gets no conversion.
    var conversionSigma: Float? = nil
    if let tailSigma, let last = lastStep, last.index == total - 1 {
      let denominator = last.sigma - last.sigmaNext
      precondition(
        denominator != 0,
        "final conversion needs a non-degenerate last step (σ \(last.sigma) == σ' \(last.sigmaNext))")
      let eps = (last.x0 - last.xNext) / MLXArray(denominator).asType(last.x0.dtype)
      x = x - MLXArray(tailSigma).asType(x.dtype) * eps
      MLX.eval(x)
      conversionSigma = tailSigma
    }

    let stats = Stats(
      stepsRun: total - startIndex,
      rowsAtStart: rowsAtStart,
      evaluateCalls: evaluateCalls,
      modelEvals: evaluateCalls * modelEvalsPerEvaluate,
      finalConversionSigma: conversionSigma)
    return (x, stats)
  }
}
