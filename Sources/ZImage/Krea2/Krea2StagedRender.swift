// Krea2StagedRender.swift — WP-E17: the second stage of a two-stage render
// (FDD-krea2-raw-recipe §3.14, D4; AC-27, AC-30, AC-31, AC-32).
//
// ONE render, two stages. The reference workflow's detail pass is not a second
// HTTP call: it re-noises the LATENT to σ ≈ 0.117 and solves again, with no VAE
// round-trip and no re-tokenised prompt. A client-composed version cannot
// express that (§3.14), so the stage lives here, between the loop and the one
// `vae.decode`.
//
// Three things in this file, in the order the render uses them:
//
//   1. ``Krea2Pipeline/Stage2`` — the request shape, and `resolved(against:)`,
//      which fills what the stage did not state from the render's OWN recipe
//      (never from an invented default) and derives the stage seed.
//   2. ``Krea2StagedRender/publishedSigmas(schedule:steps:denoise:config:mu:)``
//      — the STRETCH-AND-TAIL. `total = int(steps/denoise)` → build the
//      schedule at `total` → take `sigmas[-(steps+1):]`
//      (`res4lyf_sigmas.py:1397-1429`, truncation at `:1402`). At the published
//      recipe that is `bong_tangent(10)[-3:]` = `[0.117461, 0.043265, 0.0]` —
//      it starts at σ ≈ 0.117, **not** at 0.2, and it is not a slice of stage 1.
//   3. ``Krea2StagedRender/runStage2(...)`` — re-noise, then the same
//      ``Krea2DenoiseLoop`` a second time, on the stage's own prepared grid,
//      with its own sampler / schedule / eta / bongmath / seed.
//
// Stage 1 is untouched: a request without `stage2` executes exactly the
// statements it executed before this WP (`Krea2Pipeline.generateStaged` skips
// this file entirely), which is what makes the byte-identity gates AC-1/AC-2/5
// hold by construction rather than by measurement.
//
// `Krea2ImageToImagePipeline` is also untouched. Its `strength → startIndex`
// rule is the established img2img contract on `/v1/generate`'s `image_path`
// path; stage 2 is a distinct, differently-specified mechanism (AC-30 pins that
// they stay distinguishable).

import Foundation
import MLX
import MLXRandom

/// Fail-loud errors for the stage-2 request (§3.14: `denoise ≤ 0` is a hard
/// error; nothing here is ever clamped or substituted).
public enum Krea2StageError: Error, Equatable, CustomStringConvertible {
  /// `stage2.denoise` is not a finite number in `(0, 1]`.
  case invalidDenoise(Double)
  /// `stage2.steps` is not positive.
  case invalidSteps(Int)
  /// The stretched schedule produced too few sigmas to take a step — e.g. a
  /// `beta` grid whose de-duplication collapsed the tail (D5/AC-22).
  case degenerateStageGrid(schedule: String, steps: Int, denoise: Double, produced: Int)
  /// `generateWithRecipe` was handed a staged request. It returns ONE trace and
  /// a staged render has more than one; losing the second silently would make
  /// `applied` describe half a render, so the caller is told to use
  /// `generateStaged` instead.
  case stagedRequestNeedsStagedCall(stages: Int)

  public var description: String {
    switch self {
    case .invalidDenoise(let value):
      return "stage2.denoise must be a finite number in (0, 1] (got \(value)); it is the fraction "
        + "of the schedule the stage runs, and there is no value to substitute"
    case .invalidSteps(let value):
      return "stage2.steps must be positive (got \(value))"
    case .degenerateStageGrid(let schedule, let steps, let denoise, let produced):
      return "stage2 {schedule: \(schedule), steps: \(steps), denoise: \(denoise)} stretches to a "
        + "grid of \(produced) sigmas — too few to take a step; raise denoise or steps"
    case .stagedRequestNeedsStagedCall(let stages):
      return "this request ran \(stages) stages; generateWithRecipe returns one trace — "
        + "call generateStaged(_:progress:) to receive them all"
    }
  }
}

// MARK: - The request shape (D4)

extension Krea2Pipeline {

  /// The optional second stage of one render (§3.14). Additive: a request
  /// without it runs exactly as it did before WP-E17.
  ///
  /// `steps` and `denoise` are non-optional because they are the two fields
  /// that DECIDE the grid — a default for either would be an invented recipe.
  /// Everything else is `nil`-means-"the render's own value": the stage
  /// continues this render with this sampler, this schedule, this guidance,
  /// rather than something the engine chose. The family's published pairing
  /// (`deis_3m` + `bong_tangent` for `raw-turbo`) is the CLIENT's policy table
  /// (WP-C8), sent explicitly, not an engine default.
  public struct Stage2: Sendable, Equatable {
    /// Steps the stage runs. The stretched schedule is built at
    /// `int(steps/denoise)` and the last `steps + 1` sigmas are taken.
    public var steps: Int
    /// The fraction of the schedule this stage runs, in `(0, 1]`.
    ///
    /// **`Double`, deliberately** — every other float on `GeneratePayload`
    /// decodes as `Float`, and this one must not. Python's
    /// `int(steps/denoise)` truncates a *double*: `2/0.2` is
    /// `10.000000000000002` → 10, while `Float(9)/Float(0.3)` lands on the
    /// other side of the integer from `9/0.3` and would silently select a
    /// 29-step grid where upstream builds 30 — a different tail, a different
    /// picture, and no error anywhere (§3.14, AC-31).
    public var denoise: Double
    /// `nil` → the render's own sampler.
    public var sampler: SchedulerKind?
    /// `nil` → the render's own sigma schedule.
    public var sigmaSchedule: SigmaScheduleKind?
    /// D22: the raw schedule name the caller sent for THIS stage.
    public var sigmaScheduleRequested: String?
    /// `nil` → the render's own guidance.
    public var guidance: Float?
    /// `nil` → the render's own eta.
    public var eta: Float?
    /// `nil` → the render's own bongmath.
    public var bongmath: Bool?
    /// `nil` → the stage-1 seed `&+ 1` (§3.14). Recorded either way, so the
    /// record never has to be re-derived to reproduce the render (AC-27).
    public var seed: UInt64?
    /// `nil` → the render's own `c2` (not on the wire, D23).
    public var c2: Float?

    public init(
      steps: Int, denoise: Double,
      sampler: SchedulerKind? = nil, sigmaSchedule: SigmaScheduleKind? = nil,
      sigmaScheduleRequested: String? = nil,
      guidance: Float? = nil, eta: Float? = nil, bongmath: Bool? = nil,
      seed: UInt64? = nil, c2: Float? = nil
    ) {
      self.steps = steps
      self.denoise = denoise
      self.sampler = sampler
      self.sigmaSchedule = sigmaSchedule
      self.sigmaScheduleRequested = sigmaScheduleRequested
      self.guidance = guidance
      self.eta = eta
      self.bongmath = bongmath
      self.seed = seed
      self.c2 = c2
    }

    /// Fill the unstated fields from the render this stage belongs to, and
    /// derive the stage seed. Pure — asserted without weights.
    public func resolved(against request: Krea2Pipeline.Request) -> Krea2ResolvedStage {
      Krea2ResolvedStage(
        index: 1,
        sampler: sampler ?? request.sampler,
        sigmaSchedule: sigmaSchedule ?? request.sigmaSchedule,
        sigmaScheduleRequested: sigmaScheduleRequested,
        steps: steps,
        denoise: denoise,
        guidance: guidance ?? request.guidance,
        eta: eta ?? request.eta,
        bongmath: bongmath ?? request.bongmath,
        c2: c2 ?? request.c2,
        // `&+` so a seed at `UInt64.max` wraps rather than trapping mid-render;
        // the value is recorded, so the wrap is visible rather than surprising.
        seed: seed ?? (request.seed &+ 1))
    }
  }
}

/// One stage's recipe with nothing left to resolve — what the run and the
/// record both read.
public struct Krea2ResolvedStage: Sendable, Equatable {
  /// 0-based position in `applied.stages[]`.
  public let index: Int
  public let sampler: SchedulerKind
  public let sigmaSchedule: SigmaScheduleKind
  public let sigmaScheduleRequested: String?
  public let steps: Int
  public let denoise: Double
  public let guidance: Float
  public let eta: Float
  public let bongmath: Bool
  public let c2: Float
  public let seed: UInt64

  public init(
    index: Int, sampler: SchedulerKind, sigmaSchedule: SigmaScheduleKind,
    sigmaScheduleRequested: String?, steps: Int, denoise: Double,
    guidance: Float, eta: Float, bongmath: Bool, c2: Float, seed: UInt64
  ) {
    self.index = index
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.sigmaScheduleRequested = sigmaScheduleRequested
    self.steps = steps
    self.denoise = denoise
    self.guidance = guidance
    self.eta = eta
    self.bongmath = bongmath
    self.c2 = c2
    self.seed = seed
  }
}

// MARK: - The stage

public enum Krea2StagedRender {

  // MARK: Stretch-and-tail (§3.14 step 2, AC-31)

  /// `total = int(steps / denoise)` — RES4LYF's `get_sigmas`
  /// (`res4lyf_sigmas.py:1402`), in **`Double`**, which is the arithmetic
  /// Python does and the arithmetic the tail selection is sensitive to.
  ///
  /// - Throws: ``Krea2StageError/invalidSteps(_:)`` /
  ///   ``Krea2StageError/invalidDenoise(_:)``. `denoise ≤ 0` is a hard error
  ///   (§3.14) and so is a non-finite or `> 1` value: this is a fraction of a
  ///   schedule, and there is nothing to clamp it to that would not be a
  ///   different render than the one asked for.
  public static func stretchedStepCount(steps: Int, denoise: Double) throws -> Int {
    guard steps > 0 else { throw Krea2StageError.invalidSteps(steps) }
    guard denoise.isFinite, denoise > 0, denoise <= 1 else {
      throw Krea2StageError.invalidDenoise(denoise)
    }
    return Int((Double(steps) / denoise).rounded(.towardZero))
  }

  /// The stage's PUBLISHED grid: build `schedule` at the stretched step count,
  /// then take the last `steps + 1` sigmas.
  ///
  /// Not a truncation of stage 1's grid and not a rescaling of it — a
  /// different schedule, evaluated at a different length, of which the tail is
  /// taken. `{steps: 2, denoise: 0.2, bong_tangent}` is `bong_tangent(10)[-3:]`
  /// = `[0.117461, 0.043265, 0.0]`.
  ///
  /// `denoise == 1.0` is the identity: `total == steps` and the whole grid
  /// comes back, so a stage that runs the full schedule is expressible and is
  /// the same object stage 1 would have built.
  static func publishedSigmas(
    schedule: SigmaScheduleKind, steps: Int, denoise: Double,
    config: ZImageSchedulerConfig, mu: Float?
  ) throws -> [Float] {
    let total = try stretchedStepCount(steps: steps, denoise: denoise)
    let full = try SchedulerFactory.resolveSigmas(
      schedule: schedule, numSteps: total, config: config, mu: mu)
    // `sigmas[-(steps+1):]`. ComfyUI-exact `beta` de-duplication can leave the
    // full grid shorter than `total + 1`, in which case this takes what there
    // is — exactly as the Python slice does — and the produced count is
    // authoritative (D5/AC-22).
    return Array(full.suffix(steps + 1))
  }

  // MARK: The stage-2 scheduler, on its OWN prepared grid

  /// Build the scheduler stage 2 walks: the stretched tail, put through
  /// RES4LYF's `prepare_sigmas` in its own right.
  ///
  /// "Its own" is the load-bearing word. The stage does NOT inherit stage 1's
  /// grid, its `sigma_min` insertion or its model-free tail: it gets the same
  /// treatment applied to ITS published grid, so a RES4LYF stage 2 ends its
  /// last solver step on the model's `sigma_min` and finishes with the
  /// model-free `σ_min → 0` conversion, and a non-RES4LYF stage 2 keeps the
  /// schedule's trailing `0.0` sentinel and gets no conversion — the same rule
  /// `Krea2Pipeline.makeScheduler` applies to stage 1.
  ///
  /// **Why this does not go through `SchedulerFactory.create`:** the factory
  /// builds its grid from a STEP COUNT, and the stretch-and-tail is not
  /// expressible as one (there is no `n` for which `bong_tangent(n)` equals
  /// `bong_tangent(10)[-3:]`). The construction below is therefore the
  /// factory's own switch over an explicit grid, and
  /// `Krea2StagedSigmaTests.testSeamAgreesWithFactoryAtDenoiseOne` pins the
  /// two against each other for every `SchedulerKind` at `denoise == 1.0`, so a
  /// change to `SchedulerFactory.create` that this seam does not follow is a
  /// test failure rather than a silently divergent stage 2. (Folding it back
  /// into the factory as a `denoise:` parameter is the right end state; it is
  /// deferred because `Sources/ZImage/Pipeline/Scheduler/*` is another lane's
  /// working set while WP-E16 is in flight.)
  static func makeScheduler(
    sampler: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind,
    steps: Int,
    denoise: Double,
    shift: Krea2Sampling.ScheduleShift,
    seed: UInt64,
    c2: Float
  ) throws -> any ZImageScheduler {
    let published = try publishedSigmas(
      schedule: sigmaSchedule, steps: steps, denoise: denoise,
      config: shift.config, mu: shift.mu)

    let sigmaValues: [Float]
    let finalConversionSigma: Float?
    // S-FIX-1's rule, per stage: the Krea 2 family always asks for RES4LYF
    // preparation, and it is a no-op for every non-RES4LYF sampler.
    if sampler.isRES4LYFFamily {
      let prepared = RES4LYFSigmaPreparation.prepare(
        published: published,
        sigmaMin: try SchedulerFactory.modelSigmaMin(
          schedule: sigmaSchedule, config: shift.config, mu: shift.mu))
      sigmaValues = prepared.solverSigmas
      finalConversionSigma = prepared.finalConversionSigma
    } else {
      sigmaValues = published
      finalConversionSigma = nil
    }

    guard sigmaValues.count >= 2 else {
      throw Krea2StageError.degenerateStageGrid(
        schedule: sigmaSchedule.rawValue, steps: steps, denoise: denoise,
        produced: sigmaValues.count)
    }
    // The produced count is authoritative, exactly as in the factory.
    let effectiveSteps = sigmaValues.count - 1
    let trainSteps = shift.config.numTrainTimesteps

    switch sampler {
    case .euler:
      // Always the explicit-grid init. The factory's `(.euler, .flow)` branch
      // uses `FlowMatchEulerScheduler(numInferenceSteps:config:mu:)`, which
      // REBUILDS the flow grid from the config and would discard the tail.
      // The two agree whenever the grid is the whole schedule, which is what
      // the parity test asserts.
      return FlowMatchEulerScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps)

    case .dpmplusplus2m:
      return DPMPlusPlus2MScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps)

    case .ddim:
      return DDIMScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps, eta: 0.0, randomKey: MLXRandom.key(seed))

    case .deis:
      return DEISScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps)

    case .dpmplusplus2sa:
      return DPMPlusPlus2SAScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps, eta: 1.0, randomKey: MLXRandom.key(seed))

    case .heun:
      return HeunScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps)

    case .res2s:
      return RES2sScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps, c2: c2, finalConversionSigma: finalConversionSigma)

    case .ralston2s, .ralston3s, .ralston4s:
      let stages: RalstonScheduler.Stages =
        sampler == .ralston2s ? .two : (sampler == .ralston3s ? .three : .four)
      return RalstonScheduler(
        stages: stages, numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps, finalConversionSigma: finalConversionSigma)

    case .res3s:
      return RES3sScheduler(
        numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps, c2: c2, finalConversionSigma: finalConversionSigma)

    case .deis2m, .deis3m, .deis4m:
      let order: DEISMultistepScheduler.Order =
        sampler == .deis2m ? .two : (sampler == .deis3m ? .three : .four)
      return DEISMultistepScheduler(
        order: order, numInferenceSteps: effectiveSteps, sigmaValues: sigmaValues,
        numTrainTimesteps: trainSteps, finalConversionSigma: finalConversionSigma)
    }
  }

  /// Everything about the stage that can be refused WITHOUT model work (D18):
  /// its tier fields, its `eta`/sampler pairing, and whether its stretched grid
  /// exists at all.
  ///
  /// Called at the top of ``Krea2Pipeline/generateStaged(_:progress:)``, before
  /// the noise draw and before the first transformer forward — a stage-2 field
  /// that is going to be a 400 must not cost a stage-1 render first.
  /// Deterministic and side-effect free (scheduler construction touches no
  /// global RNG state), so running it here and again in ``runStage2(...)``
  /// changes nothing but the moment of the refusal.
  /// - Returns: the steps the stage will actually take — its EFFECTIVE count,
  ///   which `beta` de-duplication and the `sigma_min` insertion can both move
  ///   (D5/AC-22). The caller uses it to report progress over the whole render
  ///   instead of restarting the bar at the second stage.
  @discardableResult
  static func preflight(
    stage: Krea2ResolvedStage, shift: Krea2Sampling.ScheduleShift
  ) throws -> Int {
    try Krea2Pipeline.validateTiers(eta: stage.eta, bongmath: stage.bongmath)
    _ = try Krea2Pipeline.makeSDEInjector(
      eta: stage.eta, sampler: stage.sampler, stageSeed: stage.seed,
      layout: Krea2Pipeline.sdeNoiseLayout)
    return try makeScheduler(
      sampler: stage.sampler, sigmaSchedule: stage.sigmaSchedule, steps: stage.steps,
      denoise: stage.denoise, shift: shift, seed: stage.seed, c2: stage.c2
    ).numInferenceSteps
  }

  /// One progress bar over both stages.
  ///
  /// The loop reports `(i + 1, total)` against the grid IT is walking, so a
  /// two-stage render would otherwise publish 6/6 (100%) and then 1/2 (50%)
  /// — a bar that goes backwards halfway through every published render
  /// (WP-E10 wired `/health.progress_percent` to exactly this callback).
  /// Pure and separate so the monotonicity is asserted without weights.
  struct Progress: Equatable {
    let stage1Steps: Int
    let stage2Steps: Int
    var total: Int { stage1Steps + stage2Steps }
    /// Stage 1's `(step, total)` rewritten against the whole render.
    func stage1(_ step: Int) -> (Int, Int) { (step, total) }
    /// Stage 2's, offset past stage 1.
    func stage2(_ step: Int) -> (Int, Int) { (stage1Steps + step, total) }
  }

  // MARK: The re-noise (§3.14 step 3)

  /// `x₂ = σ₂[0]·ε + (1 − σ₂[0])·x₁`, applied to the LATENT — no VAE
  /// round-trip, no re-tokenised prompt (AC-30).
  ///
  /// **The mix runs in float32, from the scheduler's own 0-d `MLXArray`**, not
  /// from a Swift `Float`. That is the §3.3 / AC-2 trap in the one place it can
  /// still be sprung: mlx-swift converts a `Float` operand to the ARRAY's dtype
  /// first (`MLXArray+Ops.swift:253-255`), so `sigma.item(Float.self)` would
  /// run the whole interpolation in bf16 and move every staged render. This
  /// goes through ``Krea2Sampling/mixSourceLatent(noise:source:sigma:dtype:)``,
  /// whose precondition makes that a crash rather than a drift — the same body
  /// the img2img path uses, so the two cannot diverge.
  ///
  /// `ε` is drawn NCHW at the stage-1 latent shape, from
  /// `MLXRandom.seed(stage seed)` — the same stream shape and the same global
  /// seeding the t2i path uses for its initial noise, so "same payload, same
  /// pixels" (AC-27) covers the stage-2 draw too.
  static func renoise(
    stageOneLatent: MLXArray,
    sigma: MLXArray,
    seed: UInt64,
    patch: Int, hTok: Int, wTok: Int,
    latentHeight: Int, latentWidth: Int,
    dtype: DType
  ) -> MLXArray {
    let source = Krea2Sampling.unpatchify(
      stageOneLatent, patch: patch, h: hTok, w: wTok, c: Krea2VAE.latentChannels)
    MLXRandom.seed(seed)
    let eps = MLXRandom.normal([1, Krea2VAE.latentChannels, latentHeight, latentWidth])
      .asType(dtype)
    let mixed = Krea2Sampling.mixSourceLatent(
      noise: eps, source: source, sigma: sigma, dtype: dtype)
    return Krea2Sampling.patchify(mixed, patch: patch)
  }

  // MARK: The run (§3.14 steps 3–4)

  /// Run stage 2 on the patchified stage-1 latent and hand back the patchified
  /// stage-2 latent plus the record of what it did.
  ///
  /// Takes and returns the LATENT: the one `vae.decode` belongs to the caller,
  /// after this returns (AC-30 — exactly one decode, zero encodes).
  ///
  /// `evaluate` is handed the stage's guidance so the caller's CFG branch can
  /// key on the STAGE's value rather than the render's; the loop's
  /// `modelEvalsPerEvaluate` is derived from the same number, so the stage's
  /// cost is counted rather than predicted.
  static func runStage2(
    stage: Krea2ResolvedStage,
    shift: Krea2Sampling.ScheduleShift,
    stageOneLatent: MLXArray,
    patch: Int, hTok: Int, wTok: Int,
    latentHeight: Int, latentWidth: Int,
    dtype: DType,
    width: Int, height: Int,
    negativePromptApplied: String?,
    evaluate: (_ latent: MLXArray, _ sigma: Float, _ guidance: Float) -> MLXArray,
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> (sample: MLXArray, trace: Krea2RunTrace) {
    // D18, per stage: an unimplemented tier is refused before any model work,
    // and `eta` on a sampler RES4LYF's SDE is not defined against is refused by
    // name. The stage gets the same gates the render does — a field that is a
    // 400 on stage 1 cannot become a silent default on stage 2.
    try Krea2Pipeline.validateTiers(eta: stage.eta, bongmath: stage.bongmath)
    let sdeNoise = try Krea2Pipeline.makeSDEInjector(
      eta: stage.eta, sampler: stage.sampler, stageSeed: stage.seed,
      layout: Krea2Pipeline.sdeNoiseLayout)

    var scheduler = try makeScheduler(
      sampler: stage.sampler, sigmaSchedule: stage.sigmaSchedule, steps: stage.steps,
      denoise: stage.denoise, shift: shift, seed: stage.seed, c2: stage.c2)

    // The re-noise σ is the stage grid's FIRST sigma — σ ≈ 0.117 at the
    // published recipe, which is what "stretch-and-tail" buys over reading
    // `denoise` as a starting sigma (that would be 0.2).
    let start = renoise(
      stageOneLatent: stageOneLatent, sigma: scheduler.sigmas[0], seed: stage.seed,
      patch: patch, hTok: hTok, wTok: wTok,
      latentHeight: latentHeight, latentWidth: latentWidth, dtype: dtype)

    let useCFG = stage.guidance > 1.0
    let (denoised, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler,
      initialSample: start,
      // The stage walks its OWN grid from the top: the stretch already put the
      // start where `denoise` says. `startIndex` is img2img's mechanism, and
      // §3.14 keeps the two distinguishable on purpose (AC-30).
      startIndex: 0,
      modelEvalsPerEvaluate: useCFG ? 2 : 1,
      evaluate: { latent, sigma in evaluate(latent, sigma, stage.guidance) },
      noise: sdeNoise,
      progress: progress)

    let trace = Krea2RunTrace(
      sampler: stage.sampler,
      sigmaSchedule: stage.sigmaSchedule,
      sigmaScheduleRequested: stage.sigmaScheduleRequested,
      shift: shift,
      scheduler: scheduler,
      stats: stats,
      stepsRequested: stage.steps,
      startIndex: 0,
      denoise: Float(stage.denoise),
      guidance: stage.guidance,
      eta: stage.eta,
      bongmath: stage.bongmath,
      seed: stage.seed,
      width: width,
      height: height,
      negativePromptApplied: negativePromptApplied)
    return (denoised, trace)
  }
}
