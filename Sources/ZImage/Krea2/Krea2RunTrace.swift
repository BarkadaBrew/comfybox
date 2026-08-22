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
  public let modelEvals: Int
  /// First grid index stepped from. 0 for text-to-image.
  public let startIndex: Int
  /// The denoise fraction the start index came from. 1.0 for text-to-image.
  public let denoise: Float

  // — request echo the record needs —

  public let guidance: Float
  /// K-FIX-1 / Codex I4 — the negative prompt the pipeline ACTUALLY encoded,
  /// resolved at the CFG branch itself:
  ///
  ///   * `nil`  — guidance ≤ 1, the CFG branch never ran, nothing applied.
  ///   * `""`   — CFG ran and the caller sent no negative, so the pipeline
  ///              encoded the empty string and paid a second model pass for
  ///              it (`request.negativePrompt ?? ""`).
  ///   * text   — CFG ran against that text.
  ///
  /// Every provenance sink (`applied`, PNG metadata, `/health.last_recipe`,
  /// async status) reads this instead of the request payload, so an omitted
  /// negative under CFG can no longer be recorded as "no negative prompt".
  /// Invariant, pinned by test: non-nil exactly when ``cfgActive``.
  public let negativePromptApplied: String?
  /// RES4LYF SDE eta (T2). 0 until WP-E15.
  public let eta: Float
  /// RES4LYF bongmath (T3). `false` until WP-E16.
  public let bongmath: Bool
  public let seed: UInt64
  /// Width/height AFTER the alignment round-up — what was rendered, not what
  /// was asked for.
  public let width: Int
  public let height: Int

  public init(
    sampler: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind,
    sigmaScheduleRequested: String?,
    mu: Float,
    shift: Float,
    shiftSource: String,
    sigmas: [Float],
    finalConversionSigma: Float? = nil,
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
    height: Int,
    negativePromptApplied: String? = nil
  ) {
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.sigmaScheduleRequested = sigmaScheduleRequested
    self.mu = mu
    self.shift = shift
    self.shiftSource = shiftSource
    self.sigmas = sigmas
    self.finalConversionSigma = finalConversionSigma
    self.stepsRequested = stepsRequested
    self.stepsEffective = stepsEffective
    self.stepsRun = stepsRun
    self.modelEvals = modelEvals
    self.startIndex = startIndex
    self.denoise = denoise
    self.guidance = guidance
    self.negativePromptApplied = negativePromptApplied
    self.eta = eta
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
    height: Int,
    negativePromptApplied: String?
  ) {
    self.init(
      sampler: request.sampler,
      sigmaSchedule: request.sigmaSchedule,
      sigmaScheduleRequested: Self.requestedName(
        request.sigmaScheduleRequested, resolved: request.sigmaSchedule),
      shift: shift, scheduler: scheduler, stats: stats,
      stepsRequested: request.steps, startIndex: startIndex, denoise: denoise,
      guidance: request.guidance, eta: request.eta, bongmath: request.bongmath,
      seed: request.seed, width: width, height: height,
      negativePromptApplied: negativePromptApplied)
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
    height: Int,
    negativePromptApplied: String?
  ) {
    self.init(
      sampler: request.sampler,
      sigmaSchedule: request.sigmaSchedule,
      sigmaScheduleRequested: Self.requestedName(
        request.sigmaScheduleRequested, resolved: request.sigmaSchedule),
      shift: shift, scheduler: scheduler, stats: stats,
      stepsRequested: request.steps, startIndex: startIndex, denoise: denoise,
      guidance: request.guidance, eta: request.eta, bongmath: request.bongmath,
      seed: request.seed, width: width, height: height,
      negativePromptApplied: negativePromptApplied)
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
    height: Int,
    negativePromptApplied: String?
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
      height: height,
      negativePromptApplied: negativePromptApplied)
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
  ///
  /// This is the SAME predicate the pipelines branch on (`useCFG`), which is
  /// why ``negativePromptApplied`` is non-nil exactly when this is true.
  public var cfgActive: Bool { guidance > 1.0 }

  /// What a pipeline encodes for the CFG pass, given the branch it took and
  /// the request it was handed. One spelling of the I4 rule, so the two
  /// pipelines and the fixtures cannot drift from each other.
  public static func negativePromptApplied(cfgActive: Bool, requested: String?) -> String? {
    cfgActive ? (requested ?? "") : nil
  }

  /// First three sigmas of the grid the loop walked (`RenderRecipe.Stage.sigmaHead`).
  public var sigmaHead: [Float] { Array(sigmas.prefix(3)) }

  /// Last three sigmas of the grid (`RenderRecipe.Stage.sigmaTail`).
  public var sigmaTail: [Float] { Array(sigmas.suffix(3)) }
}
