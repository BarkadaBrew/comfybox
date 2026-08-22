// Krea2RunTrace.swift — What the Krea 2 denoise loop actually did (WP-E10,
// FDD §3.10 "every field is read back").
//
// `RenderRecipe` must never echo the request. The loop therefore COUNTS its
// own steps and model evaluations through `Krea2RunCounter` and hands the
// grid it walked, the shift it resolved and the geometry it rounded to back
// as a `Krea2RunTrace`; the server turns that into the record. Nothing here
// is derived from `Krea2Pipeline.Request` after the fact.

import Foundation

/// Loop-side counter. One instance per run; the loop calls `eval()` once per
/// transformer call (twice per step under CFG) and `step()` once per
/// scheduler step it completes.
public final class Krea2RunCounter {
  public private(set) var steps = 0
  public private(set) var evals = 0
  public init() {}
  public func step() { steps += 1 }
  public func eval() { evals += 1 }
}

/// The facts of one completed run, as counted and resolved by the loop.
public struct Krea2RunTrace: Sendable, Equatable {
  /// Geometry AFTER the pipeline's round-up to the 16-px alignment.
  public let width: Int
  public let height: Int
  public let seed: UInt64
  /// The schedule shift as resolved for this run (FDD D3): the log-shift the
  /// warp took, its effective linear shift, and where it came from.
  public let mu: Float
  public let shift: Float
  public let shiftSource: String
  /// The sigma grid the loop walked — `steps_effective + 1` values, 1.0 → 0.0.
  public let sigmas: [Float]
  public let stepsRequested: Int
  /// Where the walk began (0 for text-to-image; img2img starts mid-grid).
  public let startIndex: Int
  public let stepsRun: Int
  public let modelEvals: Int
  public let guidance: Float
  public let denoise: Float
  /// The sampler and schedule that ran, by their wire names.
  public let sampler: String
  public let sigmaSchedule: String

  public init(
    width: Int, height: Int, seed: UInt64,
    mu: Float, shift: Float, shiftSource: String,
    sigmas: [Float], stepsRequested: Int, startIndex: Int,
    guidance: Float, denoise: Float,
    sampler: String, sigmaSchedule: String,
    counter: Krea2RunCounter
  ) {
    self.width = width
    self.height = height
    self.seed = seed
    self.mu = mu
    self.shift = shift
    self.shiftSource = shiftSource
    self.sigmas = sigmas
    self.stepsRequested = stepsRequested
    self.startIndex = startIndex
    self.stepsRun = counter.steps
    self.modelEvals = counter.evals
    self.guidance = guidance
    self.denoise = denoise
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
  }

  /// Intervals in the grid (`sigmas.count - 1`); a de-duplicating schedule
  /// can make this smaller than `stepsRequested` (FDD D5, AC-22).
  public var stepsEffective: Int { max(0, sigmas.count - 1) }
  public var cfgActive: Bool { guidance > 1.0 }
  public var sigmaHead: [Float] { Array(sigmas.prefix(3)) }
  public var sigmaTail: [Float] { Array(sigmas.suffix(3)) }
}
