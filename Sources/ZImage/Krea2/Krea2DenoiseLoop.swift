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

/// Parity tier T2 seam (WP-E15): RES4LYF SDE `eta`. Called once per step,
/// after the step's commit, with the scheduler the step was taken on. The
/// conformer owns its RNG and its `sigma_down` / `sigma_up` split; E3
/// guarantees only that `nil` is today's path. Reference-typed so a
/// stateful injector persists across steps.
public protocol SDENoiseInjector: AnyObject {
  func inject(sample: inout MLXArray, timestepIndex: Int, scheduler: any ZImageScheduler)
}

/// Parity tier T3 seam (WP-E16): `bongmath`. Called once per step after the
/// T2 hook; may evaluate the model and returns how many times it did, so the
/// model-eval count stays reported rather than discovered.
public protocol BongMath: AnyObject {
  func iterate(
    sample: inout MLXArray, timestepIndex: Int, scheduler: any ZImageScheduler,
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
  ) -> (sample: MLXArray, stats: Stats) {
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

    for i in startIndex..<total {
      let sigma = sigmas[i]

      if var tableau = scheduler as? TableauScheduler {
        // 3. N-row: rows evaluations at rowSigma / rowSample, then commit.
        var k: [MLXArray] = []
        k.reserveCapacity(tableau.rows)
        for r in 0..<tableau.rows {
          let xr = r == 0 ? x : tableau.rowSample(timestepIndex: i, row: r, x0: x, k: k)
          let sr = r == 0 ? sigma : tableau.rowSigma(timestepIndex: i, row: r)
          let v = evaluate(xr, sr)
          evaluateCalls += 1
          k.append(tableau.modelInput(velocity: v, sample: xr, sigma: sr))
        }
        x = tableau.commit(timestepIndex: i, x0: x, k: k)
        scheduler = tableau
      } else {
        // 2. First evaluation at the grid sigma; convert per WP-E2.
        let v = evaluate(x, sigma)
        evaluateCalls += 1
        let out = scheduler.modelInput(velocity: v, sample: x, sigma: sigma)

        if scheduler.requiresIntermediateEvaluation,
           let mid = scheduler.intermediateStep(modelOutput: out, timestepIndex: i, sample: x) {
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
          let vMid = evaluate(mid, midSigma)
          evaluateCalls += 1
          let outMid = scheduler.modelInput(velocity: vMid, sample: mid, sigma: midSigma)
          x = scheduler.finalizeStep(
            originalOutput: out, intermediateOutput: outMid, timestepIndex: i, sample: x)
        } else {
          // 5. 1-row.
          x = scheduler.step(modelOutput: out, timestepIndex: i, sample: x)
        }
      }

      // 6. T2 / T3 hooks — nil today.
      noise?.inject(sample: &x, timestepIndex: i, scheduler: scheduler)
      if let bongmath {
        evaluateCalls += bongmath.iterate(
          sample: &x, timestepIndex: i, scheduler: scheduler, evaluate: evaluate)
      }

      // 7. Materialise, report.
      MLX.eval(x)
      progress?(i + 1, total)
    }

    let stats = Stats(
      stepsRun: total - startIndex,
      rowsAtStart: rowsAtStart,
      evaluateCalls: evaluateCalls,
      modelEvals: evaluateCalls * modelEvalsPerEvaluate)
    return (x, stats)
  }
}
