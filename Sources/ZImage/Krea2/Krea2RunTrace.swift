// Krea2RunTrace.swift — what one Krea 2 render actually did.
//
// WP-E3 (docs/FDD-krea2-raw-recipe.md §3.3, §3.10). The generate calls HAND
// this back; the pipeline keeps no copy. §3.3 is explicit that there is no
// shared "last recipe" state — the caller reads the value it was handed, so
// two concurrent renders can never read each other's record.
//
// This is deliberately the loop's half of `RenderRecipe` (§3.10) and nothing
// more: the schedule grid, the sampler that walked it, and the counts the
// loop is the only witness to. What loaded (model file, quantization, VAE,
// LoRAs) is read back from the pipeline by WP-E10, which owns the mapping
// into `RenderRecipe.Stage`.

import Foundation
import MLX

public struct Krea2RunTrace: Sendable, Equatable {

  // — the recipe as resolved —

  /// The sampler that ran (`RenderRecipe.Stage.sampler`).
  public let sampler: SchedulerKind
  /// The sigma schedule that ran (`RenderRecipe.Stage.sigmaSchedule`).
  public let sigmaSchedule: SigmaScheduleKind
  /// The raw schedule name the caller sent when it differed from the resolved
  /// kind (Krita's `normal` → `flow`, D22); `nil` when the caller sent nothing
  /// or sent the resolved name itself.
  public let sigmaScheduleRequested: String?

  // — the schedule grid (D3, as amended by Addendum A.1: shift IS mu) —

  /// The log-shift the grid was warped by.
  public let mu: Float
  /// The effective linear shift, `e^mu` — the record's human-facing number.
  public let shift: Float
  /// `"dynamic"` (resolution-derived) or `"explicit"` (request-stated).
  public let shiftSource: String
  /// The full grid the loop walked, trailing 0.
  ///
  /// `stepsEffective + 1` values on the ordinary path. On a RES4LYF-prepared
  /// grid it is `stepsEffective + 2`: the solver sigmas end on the model's
  /// `sigma_min` and the trailing 0 is where the model-free conversion landed,
  /// so the record shows the grid as actually walked rather than the published
  /// one (``finalConversionSigma``).
  public let sigmas: [Float]

  /// The sigma RES4LYF's model-free `σ_min → 0` conversion ran from, or `nil`
  /// when the run had none. When set, the last entry of ``sigmas`` was reached
  /// WITHOUT a model evaluation — it is not one of ``stepsRun``'s steps and it
  /// contributes nothing to ``modelEvals``.
  public let finalConversionSigma: Float?

  // — what actually ran —

  /// Steps the request asked for.
  public let stepsRequested: Int
  /// Steps the grid actually has: ComfyUI-exact `beta`/`beta57` de-duplication
  /// can produce fewer than requested (D5, AC-22).
  public let stepsEffective: Int
  /// Steps taken — `stepsEffective − startIndex`; img2img and a second stage
  /// start partway down the grid.
  public let stepsRun: Int
  /// Transformer forwards: `stepsRun × rows × (guidance > 1 ? 2 : 1)`.
  /// Counted by the loop, never predicted (§3.3).
  ///
  /// That product is a description, not a formula: `deis_Nm`'s row count FALLS
  /// mid-run when its order ramp completes, and this field is the loop's count
  /// either way (WP-E14, `Krea2DenoiseLoop.Stats.modelEvals`).
  public let modelEvals: Int

  // — the order ramp, when there was one (WP-E14, §3.12 / AC-24) —

  /// The sampler that actually ran during a DEIS warm-up (`"ralston_3s"`), or
  /// `nil` when the run had no warm-up — which is every sampler but `deis_Nm`,
  /// and `deis_Nm` itself on a run that took no step.
  ///
  /// It exists because the published stage-2 recipe (`deis_3m`, 2 steps) is
  /// ENTIRELY warm-up: RES4LYF swaps `ralston_3s` in while
  /// `step < order + multistep_extra_initial_steps` (the latter defaulting to
  /// 1), so the DEIS coefficients never engage and the stage costs 6 model
  /// evaluations. Without this field that fact has to be rediscovered from the
  /// upstream source every time.
  public let warmupSampler: String?

  /// How many of ``stepsRun`` used ``warmupSampler``. 0 when there was none.
  public let warmupSteps: Int
  /// First grid index stepped from. 0 for text-to-image.
  public let startIndex: Int
  /// The denoise fraction the start index came from. 1.0 for text-to-image.
  public let denoise: Float

  // — request echo the record needs —

  public let guidance: Float
  /// RES4LYF SDE eta (T2).
  public let eta: Float

  /// What the T2 SDE actually ran with, or `nil` when this render had none
  /// (`eta == 0`) — WP-E15, §3.13.
  ///
  /// Additive and optional on purpose: it is the record of a thing that mostly
  /// does not happen, and `eta: 0` renders must not grow a block of zeros in
  /// their provenance. ``eta`` stays a plain field because `RenderRecipe.Stage`
  /// already reads it.
  public let sde: SDEParameters?
  /// RES4LYF bongmath (T3). `false` until WP-E16.
  public let bongmath: Bool
  public let seed: UInt64
  /// Width/height AFTER the alignment round-up — what was rendered, not what
  /// was asked for.
  public let width: Int
  public let height: Int

  /// The RES4LYF SDE settings one stage ran under.
  ///
  /// Everything here is either a knob the request set or a value derived from
  /// it by a rule that lives in one place — so a render that looks wrong can
  /// be re-derived from the record instead of from the source. The seeds are
  /// included because they are the only part of the SDE that is NOT visible in
  /// the request: they are derived from the stage's seed by upstream's own
  /// offsets (`seed + 1`, `+ MAX_STEPS`), and without them "same payload, same
  /// output" (AC-27) is a claim rather than a check.
  public struct SDEParameters: Sendable, Equatable {
    /// `eta` — the step-level SDE strength.
    public let eta: Float
    /// `eta_substep`. `ClownsharKSampler_Beta` sets it equal to `eta`.
    public let etaSubstep: Float
    /// `noise_mode_sde`. `"hard"` is the only mode this port implements, and
    /// it is the one the workflow uses.
    public let noiseMode: String
    /// `s_noise` — the multiplier on the injected noise.
    public let sNoise: Float
    /// The step noise stream's seed: the stage's seed + 1.
    public let stepNoiseSeed: UInt64
    /// The substep noise stream's seed: the step seed + `MAX_STEPS`.
    public let substepNoiseSeed: UInt64

    public init(
      eta: Float, etaSubstep: Float, noiseMode: String, sNoise: Float,
      stepNoiseSeed: UInt64, substepNoiseSeed: UInt64
    ) {
      self.eta = eta
      self.etaSubstep = etaSubstep
      self.noiseMode = noiseMode
      self.sNoise = sNoise
      self.stepNoiseSeed = stepNoiseSeed
      self.substepNoiseSeed = substepNoiseSeed
    }

    /// What ``Krea2Pipeline`` builds its injector with, for a stage that has
    /// one. `nil` at `eta == 0`, so the two agree by construction rather than
    /// by a comment.
    public static func forRun(eta: Float, stageSeed: UInt64) -> SDEParameters? {
      guard eta != 0 else { return nil }
      return SDEParameters(
        eta: eta, etaSubstep: eta, noiseMode: "hard", sNoise: 1.0,
        stepNoiseSeed: RES4LYFSDENoiseInjector.stepNoiseSeed(stageSeed: stageSeed),
        substepNoiseSeed: RES4LYFSDENoiseInjector.substepNoiseSeed(stageSeed: stageSeed))
    }
  }

  public init(
    sampler: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind,
    sigmaScheduleRequested: String?,
    mu: Float,
    shift: Float,
    shiftSource: String,
    sigmas: [Float],
    finalConversionSigma: Float? = nil,
    warmupSampler: String? = nil,
    warmupSteps: Int = 0,
    stepsRequested: Int,
    stepsEffective: Int,
    stepsRun: Int,
    modelEvals: Int,
    startIndex: Int,
    denoise: Float,
    guidance: Float,
    eta: Float,
    bongmath: Bool,
    seed: UInt64,
    width: Int,
    height: Int
  ) {
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.sigmaScheduleRequested = sigmaScheduleRequested
    self.mu = mu
    self.shift = shift
    self.shiftSource = shiftSource
    self.sigmas = sigmas
    self.finalConversionSigma = finalConversionSigma
    self.warmupSampler = warmupSampler
    self.warmupSteps = warmupSteps
    self.stepsRequested = stepsRequested
    self.stepsEffective = stepsEffective
    self.stepsRun = stepsRun
    self.modelEvals = modelEvals
    self.startIndex = startIndex
    self.denoise = denoise
    self.guidance = guidance
    self.eta = eta
    self.sde = SDEParameters.forRun(eta: eta, stageSeed: seed)
    self.bongmath = bongmath
    self.seed = seed
    self.width = width
    self.height = height
  }
}

// MARK: - Built from what the generate paths hold

extension Krea2RunTrace {

  /// The text-to-image trace. `denoise` is 1.0 and `startIndex` 0; both are
  /// parameters rather than constants so `generateImg2ImgWithRecipe` builds
  /// its trace through the same body.
  init(
    request: Krea2Pipeline.Request,
    shift: Krea2Sampling.ScheduleShift,
    scheduler: any ZImageScheduler,
    stats: Krea2DenoiseLoop.Stats,
    startIndex: Int,
    denoise: Float,
    width: Int,
    height: Int
  ) {
    self.init(
      sampler: request.sampler,
      sigmaSchedule: request.sigmaSchedule,
      sigmaScheduleRequested: Self.requestedName(
        request.sigmaScheduleRequested, resolved: request.sigmaSchedule),
      shift: shift, scheduler: scheduler, stats: stats,
      stepsRequested: request.steps, startIndex: startIndex, denoise: denoise,
      guidance: request.guidance, eta: request.eta, bongmath: request.bongmath,
      seed: request.seed, width: width, height: height)
  }

  /// The img2img trace.
  init(
    request: Krea2Pipeline.Img2ImgRequest,
    shift: Krea2Sampling.ScheduleShift,
    scheduler: any ZImageScheduler,
    stats: Krea2DenoiseLoop.Stats,
    startIndex: Int,
    denoise: Float,
    width: Int,
    height: Int
  ) {
    self.init(
      sampler: request.sampler,
      sigmaSchedule: request.sigmaSchedule,
      sigmaScheduleRequested: Self.requestedName(
        request.sigmaScheduleRequested, resolved: request.sigmaSchedule),
      shift: shift, scheduler: scheduler, stats: stats,
      stepsRequested: request.steps, startIndex: startIndex, denoise: denoise,
      guidance: request.guidance, eta: request.eta, bongmath: request.bongmath,
      seed: request.seed, width: width, height: height)
  }

  /// The shared body: everything the loop and the scheduler are the witnesses to.
  private init(
    sampler: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind,
    sigmaScheduleRequested: String?,
    shift: Krea2Sampling.ScheduleShift,
    scheduler: any ZImageScheduler,
    stats: Krea2DenoiseLoop.Stats,
    stepsRequested: Int,
    startIndex: Int,
    denoise: Float,
    guidance: Float,
    eta: Float,
    bongmath: Bool,
    seed: UInt64,
    width: Int,
    height: Int
  ) {
    self.init(
      sampler: sampler,
      sigmaSchedule: sigmaSchedule,
      sigmaScheduleRequested: sigmaScheduleRequested,
      mu: shift.mu,
      shift: shift.shift,
      shiftSource: shift.source.rawValue,
      sigmas: Self.walkedGrid(scheduler: scheduler),
      finalConversionSigma: stats.finalConversionSigma,
      // The scheduler is the only witness to its own order ramp; a sampler
      // that does not ramp reports none rather than an invented one (WP-E14).
      warmupSampler: Self.warmUpSampler(of: scheduler),
      warmupSteps: Self.warmUpSteps(of: scheduler),
      stepsRequested: stepsRequested,
      stepsEffective: scheduler.numInferenceSteps,
      stepsRun: stats.stepsRun,
      modelEvals: stats.modelEvals,
      startIndex: startIndex,
      denoise: denoise,
      guidance: guidance,
      eta: eta,
      bongmath: bongmath,
      seed: seed,
      width: width,
      height: height)
  }

  /// The grid as walked: the scheduler's own sigmas, plus the `0.0` a
  /// RES4LYF-prepared run's model-free conversion lands on. A prepared
  /// scheduler's `sigmas` stop at the model's `sigma_min` (the zero is a
  /// sentinel, never a solver target), so without this the record would omit
  /// where the render actually finished.
  private static func walkedGrid(scheduler: any ZImageScheduler) -> [Float] {
    let solver = scheduler.sigmas.asArray(Float.self)
    guard scheduler.finalConversionSigma != nil else { return solver }
    return solver + [0.0]
  }

  /// The warm-up sampler a ramping conformer actually ran, or `nil`.
  ///
  /// Read off the scheduler AFTER the run, through
  /// ``WarmUpReportingScheduler``: only the conformer knows how many of its
  /// steps were the ramp's, and only the run that just finished knows how many
  /// steps it took at all (a 2-step `deis_3m` warms up twice; an 8-step one
  /// warms up four times). `internal` so the tests can pin both halves without
  /// building a whole `RenderRecipe`.
  static func warmUpSampler(of scheduler: any ZImageScheduler) -> String? {
    (scheduler as? WarmUpReportingScheduler)?.warmUpSampler
  }

  /// How many steps of the finished run used it; 0 for every non-ramping
  /// sampler.
  static func warmUpSteps(of scheduler: any ZImageScheduler) -> Int {
    (scheduler as? WarmUpReportingScheduler)?.warmUpSteps ?? 0
  }

  /// D22: the requested name is recorded only when it is not simply the
  /// resolved kind's own spelling — the alias is what is worth seeing.
  private static func requestedName(_ raw: String?, resolved: SigmaScheduleKind) -> String? {
    guard let raw, raw != resolved.rawValue else { return nil }
    return raw
  }
}

// MARK: - What the record reads off the trace (WP-E10)

extension Krea2RunTrace {

  /// CFG ran: two model evaluations per step, and the negative prompt was
  /// actually conditioned on. `false` at guidance ≤ 1, where the record must
  /// omit `negative_prompt` because it did not apply (AC-61).
  public var cfgActive: Bool { guidance > 1.0 }

  /// First three sigmas of the grid the loop walked (`RenderRecipe.Stage.sigmaHead`).
  public var sigmaHead: [Float] { Array(sigmas.prefix(3)) }

  /// Last three sigmas of the grid (`RenderRecipe.Stage.sigmaTail`).
  public var sigmaTail: [Float] { Array(sigmas.suffix(3)) }
}
