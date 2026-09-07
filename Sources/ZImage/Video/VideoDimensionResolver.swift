// VideoDimensionResolver.swift — the ONE place LTX-2 render dims are decided.
//
// comfybox#405: i2v could emit landscape for a portrait source. The aspect
// derivation existed (`WarmServer.deriveVideoDims`) but it was reachable only
// from one inline branch of `prepareLocalVideo`: it ran only when the init
// image's pixel size could be read, the MCP tool could not size an i2v render
// at all (it dropped caller width/height whenever `image_path` was set), a
// t2v request naming an orientation but no dims fell through to the
// hard-coded 704x448 landscape default, and nothing anywhere recorded what
// was resolved or why.
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
//         area (reason `.sourceAspect`), by the SAME search the engine runs
//         today — every currently-produced shape is byte-identical;
//       - t2v whose budget carries no orientation of its own (no explicit
//         dims, no resolution label): an `aspect_ratio` label swaps the budget
//         axes (reason `.explicit`) instead of rendering the landscape default
//         sideways. Pixel count preserved exactly;
//       - otherwise the budget keeps its own shape, /64-snapped (reason
//         `.explicit` when the caller sent dims, else `.default`).
//  3. Both axes are multiples of 64 with a floor of 256. LTX-2 renders at
//     32-multiples that are NOT 64-multiples (e.g. 480) exhibit progressive
//     haze/ghosting (#219) — every clean render in the 07-13 bisect used /64
//     dims, every hazy one used 480.
//  4. An upper safety clamp (`maxVideoLongEdge`/`maxVideoPixels`) so a
//     degenerate source cannot hand the pipeline an absurd allocation. It is
//     applied to the generator dims AND to the predicted output dims.
//
// NOTHING MAY SHRINK. The first cut of this file tightened the fit's area cap
// from 1.25x to 1.0x; a replay over all 474 `I2V: adjusted` lines in the
// production log showed 62 renders (13%) losing 17-35% of their pixels. The
// cap is back at 1.25x and the search is the original one, verbatim.
// `scripts/replay-i2v-dims.py` re-checks that over the log on demand.
//
// PREDICTIONS ARE LABELLED AS PREDICTIONS. The output size of a two-stage
// render is not `dims x 2`: `refine_scale`'s builtin is 1.5 (the production
// plist sets 1.35), the refine upsamples the LATENT grid so the pixel result
// is quantised by `spatialCompression`, and `LTX2RefineGate` can skip the
// refine outright. `predictedOutputDims` models all three; the authoritative
// number is the one the ENCODER measured
// (`LTX2VideoResult.outputWidth/outputHeight`), carried on the terminal trace
// event as `output_width`/`output_height`. Assuming x2 here is what made
// comfybox#409 look like a bug in the file rather than in the log line.
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
  /// The dims actually handed to the generator when two-stage is on — the
  /// refine upsamples them by the resolved `refine_scale` to `width`x`height`.
  /// nil for a single-scale render, where generator and output agree.
  public let stage1Width: Int?
  public let stage1Height: Int?

  public init(
    width: Int, height: Int, reason: VideoDimensionReason,
    budgetWidth: Int, budgetHeight: Int,
    sourceWidth: Int? = nil, sourceHeight: Int? = nil,
    stage1Width: Int? = nil, stage1Height: Int? = nil
  ) {
    self.width = width
    self.height = height
    self.reason = reason
    self.budgetWidth = budgetWidth
    self.budgetHeight = budgetHeight
    self.sourceWidth = sourceWidth
    self.sourceHeight = sourceHeight
    self.stage1Width = stage1Width
    self.stage1Height = stage1Height
  }

  /// True when the resolver moved the dims off the requested budget — the
  /// condition the server logs.
  public var adjusted: Bool { width != budgetWidth || height != budgetHeight }
}

public enum VideoDimensionResolver {

  /// The LTX-2 engine default when nothing else names a size.
  public static let defaultWidth = 704
  public static let defaultHeight = 448

  /// Upper ceiling for a VIDEO render, in pixels per spatial axis.
  ///
  /// The first cut used `imageMemoryCaps` (4096 / 4096², PR #363) — the IMAGE
  /// path's cap. No LTX-2 render survives 4096×4096, so it was not a ceiling
  /// in any useful sense.
  ///
  /// The numeral comes from the LTX-2 transformer's own spatial position
  /// ceiling, `positionalEmbeddingMaxPos: [20, 2048, 2048]`
  /// (`LTX2Transformer.swift`) — the only hard architectural limit the video
  /// stack states about spatial extent. **Read here in PIXEL space, not latent
  /// space, which is deliberately conservative**: in latent units 2048 would
  /// be 65 536 px, a ceiling that bounds nothing. 2048 px is 1.5× the largest
  /// render in 474 logged production lines (1344×768), so it is a verified
  /// no-op on real traffic while still rejecting a degenerate 256×17792.
  ///
  /// This is a judgement call and is flagged as one: the engine has NO
  /// video max-dims constant to inherit. The two existing LTX volume gates
  /// (`plain_decode_max_vol` 4500, `refine_max_vol` 12000) are frame-count-
  /// dependent decode/refine PATH-SELECTION gates, not admission caps — a
  /// logged 1344×768×289f render is 37 296 latent units, far past both, and
  /// renders fine via the streamed decode. Using either as a dimension clamp
  /// would shrink shapes that ship today.
  public static let maxVideoLongEdge = 2048
  /// Companion area ceiling — `maxVideoLongEdge²`, i.e. the square case.
  public static let maxVideoPixels = 2048 * 2048

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
  /// This is the pre-#405 `WarmServer.deriveVideoDims` search, moved here
  /// UNCHANGED — same +/-1 neighbourhood, same 1.25x area cap, same relaxation,
  /// same ranking (aspect error first, pixel budget as the tie-break). #405 is
  /// an ASPECT ticket: a review replay over all 474 `I2V: adjusted` lines in
  /// the production log showed that tightening the cap to 1.0 cost 62 renders
  /// (13%) 17-35% of their pixels — an undisclosed quality regression, and
  /// Todd's standing rule is quality over speed. So every source whose aspect
  /// was already being honoured keeps the EXACT dims it gets today; only the
  /// paths where no aspect was applied at all change.
  ///
  /// Rounding each axis to /64 independently compounds error in opposite
  /// directions: a 1664x896 source (aspect 1.857) at a 448x704 budget produced
  /// 768x384 (aspect 2.000) — the height's ideal 412.1 sat almost exactly on a
  /// 64-boundary midpoint and rounded DOWN while the width rounded up, a 7.7%
  /// distortion that visibly squashes the subject (2026-08-01). Searching the
  /// /64 neighbourhood and keeping the pair whose aspect is closest fixes that.
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

    func search(areaCap: Double) -> (w: Int, h: Int, aspectErr: Double, areaErr: Double)? {
      var best: (w: Int, h: Int, aspectErr: Double, areaErr: Double)?
      for dw in -1...1 {
        for dh in -1...1 {
          let w = max(256, (baseW + dw) * 64)
          let h = max(256, (baseH + dh) * 64)
          let area = Double(w * h)
          guard area <= budget * areaCap else { continue }
          let aspectErr = abs(Double(w) / Double(h) - aspect) / aspect
          let areaErr = abs(area - budget) / budget
          if let b = best {
            let better = aspectErr < b.aspectErr - 1e-9
              || (abs(aspectErr - b.aspectErr) <= 1e-9 && areaErr < b.areaErr)
            if better { best = (w, h, aspectErr, areaErr) }
          } else {
            best = (w, h, aspectErr, areaErr)
          }
        }
      }
      return best
    }

    // Prefer staying near the budget; but at small budgets the 256 floor pins
    // one axis and the tight cap can force a badly stretched pair (a halved
    // two-stage budget hit 19% that way), so allow a larger clip rather than
    // distort.
    var pick = search(areaCap: 1.25)
    if pick == nil || pick!.aspectErr > 0.03, let relaxed = search(areaCap: 1.6),
       relaxed.aspectErr < (pick?.aspectErr ?? .infinity) - 1e-9 {
      pick = relaxed
    }
    guard let chosen = pick else {
      return (snap64(Int(idealW.rounded())), snap64(Int(idealH.rounded())))
    }
    return (chosen.w, chosen.h)
  }

  /// Upper safety clamp: no render may exceed `maxLongEdge` on its longer axis
  /// or `maxPixels` in total. Defaults are the VIDEO ceiling above (see
  /// `maxVideoLongEdge` for where the number comes from and why it is not the
  /// image path's 4096). It exists so a degenerate source (a 10x9999 strip, a
  /// corrupt EXIF size) cannot hand the pipeline a 256x17792 allocation, and
  /// is a verified no-op at every shape in the production log.
  ///
  /// Aspect is preserved by scaling BOTH axes by one factor and flooring each
  /// to /64 (flooring, not rounding, so the result cannot land back above the
  /// cap). The 256 floor still wins — at 256x256 the caps are satisfied by any
  /// sane configuration.
  public static func clamp(
    width: Int, height: Int,
    maxLongEdge: Int = maxVideoLongEdge, maxPixels: Int = maxVideoPixels
  ) -> (width: Int, height: Int) {
    let edgeCap = max(256, maxLongEdge)
    let pixelCap = max(256 * 256, maxPixels)
    guard max(width, height) > edgeCap || width * height > pixelCap else {
      return (width, height)
    }
    let edgeScale = Double(edgeCap) / Double(max(width, height))
    let pixelScale = (Double(pixelCap) / Double(width * height)).squareRoot()
    let scale = min(edgeScale, pixelScale)
    func floor64(_ v: Double) -> Int { max(256, Int(v / 64.0) * 64) }
    var w = floor64(Double(width) * scale)
    var h = floor64(Double(height) * scale)
    // The 256 floor can push an axis back over a cap on a degenerate strip.
    // Step the LONGER axis down until both caps hold or it hits the floor.
    var guardCount = 0
    while (max(w, h) > edgeCap || w * h > pixelCap) && max(w, h) > 256 && guardCount < 1024 {
      if w >= h { w = max(256, w - 64) } else { h = max(256, h - 64) }
      guardCount += 1
    }
    return (w, h)
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
  ///   - hasInitImage: whether the request supplied an init image AT ALL —
  ///     distinct from whether its size could be READ. A request with an
  ///     unreadable init image is still i2v: swapping its budget axes on an
  ///     `aspect_ratio` label would override a source image the resolver
  ///     simply failed to measure (review round 2, item 4). Such a request
  ///     keeps the budget's shape and the server warns.
  ///   - maxLongEdge/maxPixels: the upper safety clamp (see `clamp`).
  ///
  /// Why `aspectRatio` does NOT override the source image on i2v: real
  /// traffic sends a DEFAULTED `"16:9"` alongside an i2v request (the MCP
  /// tool's schema documents 16:9 as the default, and the production logs show
  /// i2v renders arriving with a landscape budget while the source is portrait
  /// or square). Treating that default as an intent would force every such
  /// render landscape — which is exactly the bug #405 is about, not a fix for
  /// it. The source image is ground truth for i2v; the label sizes the budget.
  ///
  /// Where the label DOES decide the shape is a request with NO init image at
  /// all whose budget came from neither explicit dims nor a named resolution — i.e. it fell through
  /// to a preset / server default / the 704x448 engine default, which is
  /// landscape. That request renders landscape today no matter what
  /// orientation it asked for. It is corrected by ORIENTING the budget (the
  /// axes are swapped, the pixel count is preserved exactly — 704x448 with
  /// "9:16" becomes 448x704, the same dims the MCP tool already synthesizes
  /// client-side), never by refitting: nothing may shrink. Any request that
  /// carries explicit width/height or a resolution label already has its
  /// orientation baked into the budget by `videoDims`, and is left
  /// byte-identical (comfybox#405 review ruling 1).
  public static func resolve(
    requestWidth: Int?, requestHeight: Int?,
    namedWidth: Int? = nil, namedHeight: Int? = nil,
    presetWidth: Int? = nil, presetHeight: Int? = nil,
    configWidth: Int? = nil, configHeight: Int? = nil,
    sourceWidth: Int? = nil, sourceHeight: Int? = nil,
    aspectRatio: String? = nil,
    hasInitImage: Bool = false,
    maxLongEdge: Int = maxVideoLongEdge, maxPixels: Int = maxVideoPixels
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

    func answer(
      _ dims: (width: Int, height: Int), _ reason: VideoDimensionReason
    ) -> ResolvedVideoDimensions {
      let capped = clamp(
        width: dims.width, height: dims.height,
        maxLongEdge: maxLongEdge, maxPixels: maxPixels)
      return ResolvedVideoDimensions(
        width: capped.width, height: capped.height, reason: reason,
        budgetWidth: budgetDims.width, budgetHeight: budgetDims.height,
        sourceWidth: hasSource ? sourceWidth : nil,
        sourceHeight: hasSource ? sourceHeight : nil)
    }

    // i2v: the source image is ground truth for the SHAPE; the budget only
    // decides the magnitude.
    if hasSource {
      return answer(
        fit(aspect: Double(sourceWidth!) / Double(sourceHeight!),
            budgetWidth: budgetDims.width, budgetHeight: budgetDims.height),
        .sourceAspect)
    }

    // t2v BY CONSTRUCTION (no init image was supplied at all) with an
    // orientation the budget does not already carry: orient the budget rather
    // than render the landscape default sideways. Axis swap only — the pixel
    // count is preserved exactly. Gated on `hasInitImage`, not on whether the
    // source size parsed: an i2v whose init image could not be measured must
    // NOT have its shape decided by a (usually defaulted) aspect_ratio label.
    if !hasInitImage, !budgetAlreadyOriented, let target = aspect(fromLabel: aspectRatio) {
      let budgetIsPortrait = budgetDims.height > budgetDims.width
      let wantPortrait = target < 1.0
      if budgetIsPortrait != wantPortrait, target != 1.0 {
        return answer(
          (snap64(budgetDims.height), snap64(budgetDims.width)), .explicit)
      }
    }

    // t2v (or an unreadable source): keep the budget's own shape, /64-snapped.
    return answer(
      (snap64(budgetDims.width), snap64(budgetDims.height)),
      callerSizedBudget ? .explicit : .default)
  }

  /// PREDICTED output dims for a render, given the dims the generator will be
  /// handed and the resolved refine configuration.
  ///
  /// This is a prediction and is named as one. The first cut of #405 hardcoded
  /// x2 — the two-stage halving convention's own assumption — but that is
  /// wrong at the shipping configuration in two independent ways:
  ///
  ///  * `refine_scale`'s builtin is **1.5**, not 2 (`LTX2ConfigResolver`), and
  ///    the pipeline clamps it to [1, 2] (`LTX2Pipeline`). The refine upsamples
  ///    the LATENT grid by that factor and the VAE decodes at
  ///    `spatialCompression` per latent unit, so the pixel result is
  ///    `round(dim / spatialCompression * scale) * spatialCompression` — NOT
  ///    `dim * scale`, because the latent rounding quantises it.
  ///  * `LTX2RefineGate` can skip the refine entirely (volume gate, missing
  ///    upsampler), in which case the output is the generator dims unchanged.
  ///
  /// So a two-stage render's real output is one of three sizes, and only the
  /// encoder knows which. `LTX2VideoResult.outputWidth/outputHeight` carry the
  /// MEASURED pair; this function exists for the submitted event, which has to
  /// commit to a number before the render runs.
  ///
  /// - Parameters:
  ///   - refineScale: the RESOLVED scale (`LTX2ResolvedVideoConfig.refineScale`),
  ///     clamped here exactly as `LTX2Pipeline` clamps it.
  ///   - refineWillSkip: pass true when the refine is already known not to run.
  /// Reduce the GENERATOR dims until their predicted output fits the ceiling.
  ///
  /// Review round 2, item 2: the clamp has to bound the FINAL output, not only
  /// the dims handed to the pipeline — a two-stage render can predict past the
  /// ceiling from generator dims that are themselves under it. Clamping the
  /// prediction alone would only make the REPORTED number wrong (the file
  /// would still be whatever the encoder wrote), so the reduction is applied
  /// where it actually binds: the generator dims. One pass suffices — the /64
  /// floor only ever reduces further.
  ///
  /// Returns the generator dims unchanged when the prediction already fits,
  /// which is every shape in the production log (largest predicted output:
  /// 2016x1152 from a 1344x768 two-stage render).
  public static func generatorDimsFittingOutputCeiling(
    generatorWidth: Int, generatorHeight: Int,
    twoStage: Bool, refineScale: Float,
    spatialCompression: Int = 32, refineWillSkip: Bool = false,
    maxLongEdge: Int = maxVideoLongEdge, maxPixels: Int = maxVideoPixels
  ) -> (width: Int, height: Int) {
    let predicted = predictedOutputDims(
      generatorWidth: generatorWidth, generatorHeight: generatorHeight,
      twoStage: twoStage, refineScale: refineScale,
      spatialCompression: spatialCompression, refineWillSkip: refineWillSkip)
    let edgeCap = max(256, maxLongEdge)
    let pixelCap = max(256 * 256, maxPixels)
    guard max(predicted.width, predicted.height) > edgeCap
      || predicted.width * predicted.height > pixelCap
    else { return (generatorWidth, generatorHeight) }
    let scale = min(
      Double(edgeCap) / Double(max(predicted.width, predicted.height)),
      (Double(pixelCap) / Double(predicted.width * predicted.height)).squareRoot())
    func floor64(_ v: Double) -> Int { max(256, Int(v / 64.0) * 64) }
    return (
      floor64(Double(generatorWidth) * scale),
      floor64(Double(generatorHeight) * scale))
  }

  public static func predictedOutputDims(
    generatorWidth: Int, generatorHeight: Int,
    twoStage: Bool, refineScale: Float,
    spatialCompression: Int = 32, refineWillSkip: Bool = false
  ) -> (width: Int, height: Int) {
    guard twoStage, !refineWillSkip else { return (generatorWidth, generatorHeight) }
    let scale = max(1.0, min(2.0, refineScale))
    let comp = max(1, spatialCompression)
    func scaled(_ dim: Int) -> Int {
      let lat = max(1, dim / comp)
      return max(1, Int((Float(lat) * scale).rounded())) * comp
    }
    return (scaled(generatorWidth), scaled(generatorHeight))
  }
}
