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
  /// The full grid the loop walked, `stepsEffective + 1` values, trailing 0.
  public let sigmas: [Float]

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
    self.stepsRequested = stepsRequested
    self.stepsEffective = stepsEffective
    self.stepsRun = stepsRun
    self.modelEvals = modelEvals
    self.startIndex = startIndex
    self.denoise = denoise
    self.guidance = guidance
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
      sigmas: scheduler.sigmas.asArray(Float.self),
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

  /// D22: the requested name is recorded only when it is not simply the
  /// resolved kind's own spelling — the alias is what is worth seeing.
  private static func requestedName(_ raw: String?, resolved: SigmaScheduleKind) -> String? {
    guard let raw, raw != resolved.rawValue else { return nil }
    return raw
  }
}
