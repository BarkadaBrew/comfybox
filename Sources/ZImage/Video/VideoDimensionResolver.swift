// VideoDimensionResolver.swift — the ONE place LTX-2 render dims are decided.
//
// comfybox#405: i2v ignored the source image's aspect and emitted landscape
// (704x448 at "480p"), so a 9:16 portrait source rendered squeezed. The aspect
// derivation existed (`WarmServer.deriveVideoDims`) but it was reachable only
// from one inline branch of `prepareLocalVideo`, it chased the pixel BUDGET
// hard enough to overshoot the requested resolution label, and a request that
// named neither a resolution nor explicit
// dims fell all the way through to the hard-coded 704x448 landscape default
// with nothing recorded about why.
//
// This file is deliberately dependency-free (Foundation only) and pure: every
// decision is a value in, a value out, so the sizes callers ACTUALLY send can
// be validated in unit tests rather than only the ones convenient to reach
// (see intent.md — agents run unit tests only).
//
// Rules the resolver enforces, in order:
//
//  1. Budget (magnitude): explicit width/height > named resolution label >
//     preset dims > server config defaults > 704x448 engine default. This is
//     unchanged from `WarmServer.resolvedLTX2RequestDims` — the priority chain
//     is a production contract.
//  2. Shape (aspect):
//       - i2v: the SOURCE image's aspect wins, fitted into the budget's pixel
//         area (reason `.sourceAspect`);
//       - t2v whose budget carries no orientation of its own (no explicit
//         dims, no resolution label): an `aspect_ratio` label decides the
//         shape (reason `.explicit`) instead of rendering the landscape
//         default sideways;
//       - otherwise the budget keeps its own shape, /64-snapped (reason
//         `.explicit` when the caller sent dims, else `.default`). Every t2v
//         request that reached a shape before #405 is byte-identical.
//  3. Both axes are multiples of 64 with a floor of 256. LTX-2 renders at
//     32-multiples that are NOT 64-multiples (e.g. 480) exhibit progressive
//     haze/ghosting (#219) — every clean render in the 07-13 bisect used /64
//     dims, every hazy one used 480.
//
// Why a neighbourhood SEARCH and not per-axis rounding: rounding each axis to
// /64 independently compounds error in opposite directions. A 1664x896 source
// (aspect 1.857) at a 448x704 budget produced 768x384 (aspect 2.000) — the
// height's ideal 412.1 sat almost exactly on a 64-boundary midpoint and
// rounded DOWN while the width rounded up, a 7.7% distortion that visibly
// squashes the subject (2026-08-01). Searching the /64 neighbourhood and
// keeping the pair whose aspect is closest to the target fixes that.

import Foundation

/// Why the resolver produced the dims it did — recorded on the response and
/// the render trace (comfybox#405) so a wrong-shaped clip is diagnosable
/// after the fact instead of requiring a log archaeology session.
public enum VideoDimensionReason: String, Equatable, Sendable {
  /// The output aspect came from the i2v source image.
  case sourceAspect = "source_aspect"
  /// The caller pinned the shape: explicit width/height, or an `aspect_ratio`
  /// label on a t2v request whose budget carried no orientation.
  case explicit = "explicit"
  /// No source image and no caller dims — the named resolution / preset /
  /// server default budget, snapped.
  case `default` = "default"
}

/// The resolver's full answer: the dims to render at, why, and the budget and
/// source it reasoned from (both echoed so a trace is self-explaining).
public struct ResolvedVideoDimensions: Equatable, Sendable {
  public let width: Int
  public let height: Int
  public let reason: VideoDimensionReason
  /// The pixel-area budget the shape was fitted into, after the priority chain.
  public let budgetWidth: Int
  public let budgetHeight: Int
  /// The i2v source image size, when one was supplied and readable.
  public let sourceWidth: Int?
  public let sourceHeight: Int?

  public init(
    width: Int, height: Int, reason: VideoDimensionReason,
    budgetWidth: Int, budgetHeight: Int,
    sourceWidth: Int? = nil, sourceHeight: Int? = nil
  ) {
    self.width = width
    self.height = height
    self.reason = reason
    self.budgetWidth = budgetWidth
    self.budgetHeight = budgetHeight
    self.sourceWidth = sourceWidth
    self.sourceHeight = sourceHeight
  }

  /// True when the resolver moved the dims off the requested budget — the
  /// condition the server logs.
  public var adjusted: Bool { width != budgetWidth || height != budgetHeight }
}

public enum VideoDimensionResolver {

  /// The LTX-2 engine default when nothing else names a size.
  public static let defaultWidth = 704
  public static let defaultHeight = 448

  /// Snap a render dimension to the nearest multiple of 64 (floor 256), the
  /// /64 rule the pipeline enforces (#219).
  public static func snap64(_ value: Int) -> Int {
    max(256, Int((Double(value) / 64.0).rounded()) * 64)
  }

  /// Parse an `aspect_ratio` label into a width/height ratio.
  ///
  /// Accepts `W:H` and `WxH` with positive integers or decimals ("9:16",
  /// "16:9", "1:1", "4:5", "1.85:1"). Returns nil for anything else — an
  /// unparseable label must fall back to the source aspect, never to a
  /// silently wrong shape.
  public static func aspect(fromLabel label: String?) -> Double? {
    guard let raw = label?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
          !raw.isEmpty
    else { return nil }
    let parts = raw.split(whereSeparator: { $0 == ":" || $0 == "x" || $0 == "/" })
    guard parts.count == 2,
          let w = Double(parts[0]), let h = Double(parts[1]),
          w > 0, h > 0, w.isFinite, h.isFinite
    else { return nil }
    return w / h
  }

  /// Fit `aspect` into the pixel-area budget on the /64 grid.
  ///
  /// The candidate must not exceed the budget area (the label means what it
  /// says: a "480p" request should not silently render 15% more pixels than
  /// 480p). The single exception is the small-budget pathology: below ~2x the
  /// 256 floor the cap can pin one axis and force a badly stretched pair
  /// (a halved two-stage budget once hit 19% distortion that way), so if the
  /// best in-budget candidate is still more than 10% off the target aspect,
  /// a larger clip is preferred over a distorted one.
  public static func fit(
    aspect: Double, budgetWidth: Int, budgetHeight: Int
  ) -> (width: Int, height: Int) {
    guard aspect > 0, aspect.isFinite else {
      return (snap64(budgetWidth), snap64(budgetHeight))
    }
    let budget = Double(max(budgetWidth, 64) * max(budgetHeight, 64))
    let idealW = (budget * aspect).squareRoot()
    let idealH = idealW / aspect
    let baseW = Int((idealW / 64.0).rounded())
    let baseH = Int((idealH / 64.0).rounded())

    // Aspect fidelity dominates, but area is not free: without the small area
    // term a candidate a third of the requested size wins on a 0.7-point
    // aspect gain (576x1024 at a 480p budget picked 384x704 — 270k px — over
    // 448x768 — 344k px). The weight is bounded by that pair: it must stay
    // small enough that a 0.6-point aspect gain never buys a 7-point area
    // loss, and large enough to break near-ties toward the requested budget.
    let areaWeight = 0.05
    func search(areaCap: Double) -> (w: Int, h: Int, aspectErr: Double, score: Double)? {
      var best: (w: Int, h: Int, aspectErr: Double, score: Double)?
      for dw in -2...2 {
        for dh in -2...2 {
          let w = max(256, (baseW + dw) * 64)
          let h = max(256, (baseH + dh) * 64)
          let area = Double(w * h)
          guard area <= budget * areaCap else { continue }
          let aspectErr = abs(Double(w) / Double(h) - aspect) / aspect
          let areaErr = abs(area - budget) / budget
          let score = aspectErr + areaWeight * areaErr
          guard let b = best else { best = (w, h, aspectErr, score); continue }
          if score < b.score - 1e-12 { best = (w, h, aspectErr, score) }
        }
      }
      return best
    }

    // Stay inside the requested budget by default. Only a badly distorted
    // in-budget best (the 256-floor pathology) justifies overshooting it.
    var pick = search(areaCap: 1.0)
    if pick == nil || pick!.aspectErr > 0.10,
       let relaxed = search(areaCap: 1.6),
       relaxed.aspectErr < (pick?.aspectErr ?? .infinity) - 1e-9 {
      pick = relaxed
    }
    guard let chosen = pick else {
      return (snap64(Int(idealW.rounded())), snap64(Int(idealH.rounded())))
    }
    return (chosen.w, chosen.h)
  }

  /// The pixel-area budget, by the production priority chain:
  /// explicit width/height > named resolution > preset dims > server config
  /// defaults > the 704x448 engine default.
  public static func budget(
    requestWidth: Int?, requestHeight: Int?,
    namedWidth: Int?, namedHeight: Int?,
    presetWidth: Int?, presetHeight: Int?,
    configWidth: Int?, configHeight: Int?
  ) -> (width: Int, height: Int) {
    (
      width: requestWidth ?? namedWidth ?? presetWidth ?? configWidth ?? defaultWidth,
      height: requestHeight ?? namedHeight ?? presetHeight ?? configHeight ?? defaultHeight
    )
  }

  /// Resolve the dims a video request should render at.
  ///
  /// - Parameters:
  ///   - sourceWidth/sourceHeight: the i2v init image's pixel size, or nil for
  ///     t2v (or an unreadable image — in which case the budget's own shape is
  ///     kept, exactly as before #405).
  ///   - aspectRatio: the caller's `aspect_ratio` label.
  ///
  /// Why `aspectRatio` does NOT override the source image on i2v: real
  /// traffic sends a DEFAULTED `"16:9"` alongside an i2v request (the MCP
  /// tool's schema documents 16:9 as the default, and the production logs show
  /// i2v renders arriving with a landscape budget while the source is portrait
  /// or square). Treating that default as an intent would force every such
  /// render landscape — which is exactly the bug #405 is about, not a fix for
  /// it. The source image is ground truth for i2v; the label sizes the budget.
  ///
  /// Where the label DOES decide the shape is a t2v request whose budget came
  /// from neither explicit dims nor a named resolution — i.e. it fell through
  /// to a preset / server default / the 704x448 engine default. Today that
  /// request renders landscape no matter what orientation it asked for. This
  /// case is purely additive: any request carrying explicit width/height or a
  /// resolution label already has the orientation baked into its budget by
  /// `videoDims`, and is left byte-identical (comfybox#405 ruling 3).
  public static func resolve(
    requestWidth: Int?, requestHeight: Int?,
    namedWidth: Int? = nil, namedHeight: Int? = nil,
    presetWidth: Int? = nil, presetHeight: Int? = nil,
    configWidth: Int? = nil, configHeight: Int? = nil,
    sourceWidth: Int? = nil, sourceHeight: Int? = nil,
    aspectRatio: String? = nil
  ) -> ResolvedVideoDimensions {
    let budgetDims = budget(
      requestWidth: requestWidth, requestHeight: requestHeight,
      namedWidth: namedWidth, namedHeight: namedHeight,
      presetWidth: presetWidth, presetHeight: presetHeight,
      configWidth: configWidth, configHeight: configHeight)

    let hasSource = (sourceWidth ?? 0) > 0 && (sourceHeight ?? 0) > 0
    let callerSizedBudget = requestWidth != nil || requestHeight != nil
    let budgetAlreadyOriented =
      callerSizedBudget || namedWidth != nil || namedHeight != nil

    // i2v: the source image is ground truth for the SHAPE; the budget only
    // decides the magnitude.
    if hasSource {
      let sw = sourceWidth!, sh = sourceHeight!
      let fitted = fit(
        aspect: Double(sw) / Double(sh),
        budgetWidth: budgetDims.width, budgetHeight: budgetDims.height)
      return ResolvedVideoDimensions(
        width: fitted.width, height: fitted.height,
        reason: .sourceAspect,
        budgetWidth: budgetDims.width, budgetHeight: budgetDims.height,
        sourceWidth: sw, sourceHeight: sh)
    }

    // t2v with an orientation the budget does not already carry: honour it
    // instead of rendering the landscape default sideways.
    if !budgetAlreadyOriented, let target = aspect(fromLabel: aspectRatio) {
      let fitted = fit(
        aspect: target,
        budgetWidth: budgetDims.width, budgetHeight: budgetDims.height)
      return ResolvedVideoDimensions(
        width: fitted.width, height: fitted.height,
        reason: .explicit,
        budgetWidth: budgetDims.width, budgetHeight: budgetDims.height)
    }

    // t2v (or an unreadable source): keep the budget's own shape, /64-snapped.
    return ResolvedVideoDimensions(
      width: snap64(budgetDims.width), height: snap64(budgetDims.height),
      reason: callerSizedBudget ? .explicit : .default,
      budgetWidth: budgetDims.width, budgetHeight: budgetDims.height,
      sourceWidth: nil, sourceHeight: nil)
  }
}
