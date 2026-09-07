// VideoDimensionResolverTests.swift — comfybox#405
//
// LTX-2 i2v ignored the source image's aspect and emitted landscape (704x448
// at "480p"), so a 9:16 portrait source rendered squeezed. These pin the ONE
// pure resolver every video caller now shares:
//
//   * a portrait source renders portrait, a landscape source landscape, a
//     square source square — at the requested resolution budget, /64;
//   * t2v and any request that already carried a shape is byte-identical to
//     the pre-#405 behaviour (the regression guard);
//   * the reason is reported so the response/trace can record WHY.
//
// The production cases below are taken verbatim from the engine's own log
// (~/.comfybox/serve.err.log, "LTX-2 I2V: adjusted …") so this is validated at
// the sizes Kira's scheduler actually sends, not at convenient ones.

import XCTest
@testable import ZImage

final class VideoDimensionResolverTests: XCTestCase {

  private func resolveI2V(
    source: (Int, Int), budget: (Int, Int), aspectRatio: String? = nil
  ) -> ResolvedVideoDimensions {
    VideoDimensionResolver.resolve(
      requestWidth: budget.0, requestHeight: budget.1,
      sourceWidth: source.0, sourceHeight: source.1,
      aspectRatio: aspectRatio, hasInitImage: true)
  }

  // MARK: - The ticket's shapes at a 480p budget (832x480)
  //
  // The values are the ones the engine produces TODAY. The first cut of this
  // PR tightened the fit's area cap to 1.0x, which would have made these
  // 448x768 / 768x448 / 576x576 — a 25%, 25% and 19% pixel loss. The review
  // replay found 62 renders (13% of the log) losing 17-35% that way, so the
  // cap is back at 1.25x: #405 is an ASPECT ticket, not a sizing one.

  func testPortraitSourceAt480pRendersPortrait() {
    // 576x1024 (9:16) source, 480p budget. The bug: landscape output.
    let r = resolveI2V(source: (576, 1024), budget: (832, 480))
    XCTAssertEqual(r.width, 512)
    XCTAssertEqual(r.height, 896)
    XCTAssertEqual(r.reason, .sourceAspect)
    XCTAssertLessThan(r.width, r.height, "a 9:16 source must not render landscape")
  }

  func testLandscapeSourceAt480pRendersLandscape() {
    let r = resolveI2V(source: (1024, 576), budget: (832, 480))
    XCTAssertEqual(r.width, 896)
    XCTAssertEqual(r.height, 512)
    XCTAssertEqual(r.reason, .sourceAspect)
  }

  func testSquareSourceAt480pRendersSquare() {
    let r = resolveI2V(source: (1024, 1024), budget: (832, 480))
    XCTAssertEqual(r.width, r.height, "a square source must render square")
    XCTAssertEqual(r.width, 640)
    XCTAssertEqual(r.reason, .sourceAspect)
  }

  // MARK: - The /64 rule and the 256 floor (#219 haze)

  func testEveryResolvedAxisIsAMultipleOf64AboveTheFloor() {
    let sources = [(576, 1024), (1024, 576), (1024, 1024), (896, 1664), (1664, 896),
                   (832, 1216), (1344, 768), (768, 1024), (896, 1120)]
    let budgets = [(832, 480), (480, 832), (1280, 720), (704, 448), (384, 640)]
    for source in sources {
      for budget in budgets {
        let r = resolveI2V(source: source, budget: budget)
        XCTAssertEqual(r.width % 64, 0, "\(source) @ \(budget) -> \(r.width)x\(r.height)")
        XCTAssertEqual(r.height % 64, 0, "\(source) @ \(budget) -> \(r.width)x\(r.height)")
        XCTAssertGreaterThanOrEqual(r.width, 256)
        XCTAssertGreaterThanOrEqual(r.height, 256)
      }
    }
  }

  func testOrientationAlwaysFollowsTheSource() {
    let cases: [(source: (Int, Int), budget: (Int, Int))] = [
      ((576, 1024), (832, 480)),   // portrait source, landscape budget
      ((1024, 576), (480, 832)),   // landscape source, portrait budget
      ((896, 1664), (832, 480)),
      ((1664, 896), (480, 832)),
      ((768, 1024), (704, 448)),
    ]
    for c in cases {
      let r = resolveI2V(source: c.source, budget: c.budget)
      let sourcePortrait = c.source.1 > c.source.0
      XCTAssertEqual(
        r.height > r.width, sourcePortrait,
        "source \(c.source) @ budget \(c.budget) resolved \(r.width)x\(r.height)")
    }
  }

  func testAspectErrorStaysSmallAcrossProductionSources() {
    // The 2026-08-01 regression was a 7.7% squash. Nothing should exceed 5%
    // at a normal budget.
    let cases: [((Int, Int), (Int, Int))] = [
      ((576, 1024), (832, 480)), ((896, 1664), (480, 832)), ((1664, 896), (832, 480)),
      ((1664, 896), (448, 704)), ((896, 1120), (480, 832)), ((1024, 1024), (832, 480)),
      ((1344, 768), (832, 480)), ((832, 1216), (704, 448)), ((1024, 640), (480, 832)),
    ]
    for (source, budget) in cases {
      let r = resolveI2V(source: source, budget: budget)
      let want = Double(source.0) / Double(source.1)
      let got = Double(r.width) / Double(r.height)
      let err = abs(got - want) / want
      XCTAssertLessThan(
        err, 0.05,
        "source \(source) @ \(budget) -> \(r.width)x\(r.height): \(Int(err * 100))% off aspect")
    }
  }

  func testTinyBudgetPrefersALargerClipOverADistortedOne() {
    // 224x352 budget (a halved two-stage budget) with a 1.857 source: the 256
    // floor pins the height, and everything inside the budget is >40% off
    // aspect. Overshooting the budget beats squashing the subject.
    let r = resolveI2V(source: (1664, 896), budget: (224, 352))
    let err = abs(Double(r.width) / Double(r.height) - 1664.0 / 896.0) / (1664.0 / 896.0)
    XCTAssertLessThan(err, 0.10)
    XCTAssertGreaterThan(r.width, r.height)
  }

  // MARK: - Regression guard: t2v and explicit dims are unchanged (ruling 3)

  func testT2VDefaultIsTheUnchangedLandscapeEngineDefault() {
    let r = VideoDimensionResolver.resolve(requestWidth: nil, requestHeight: nil)
    XCTAssertEqual(r.width, 704)
    XCTAssertEqual(r.height, 448)
    XCTAssertEqual(r.reason, .default)
  }

  func testT2VNamedResolutionBudgetsAreUnchanged() {
    // videoDims maps the label upstream; the resolver must pass them through
    // untouched (all already /64 except 480, which snaps to 512 as before).
    let landscape480 = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, namedWidth: 832, namedHeight: 480,
      aspectRatio: "16:9")
    XCTAssertEqual(landscape480.width, 832)
    XCTAssertEqual(landscape480.height, 512)   // 480 -> 512, the pre-#405 snap

    let portrait480 = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, namedWidth: 480, namedHeight: 832,
      aspectRatio: "9:16")
    XCTAssertEqual(portrait480.width, 512)
    XCTAssertEqual(portrait480.height, 832)

    let p720 = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, namedWidth: 1280, namedHeight: 720,
      aspectRatio: "16:9")
    XCTAssertEqual(p720.width, 1280)
    XCTAssertEqual(p720.height, 704)           // 720 -> 704, the pre-#405 snap
  }

  func testT2VExplicitDimsWinAndAreOnlySnapped() {
    let r = VideoDimensionResolver.resolve(
      requestWidth: 960, requestHeight: 576,
      namedWidth: 832, namedHeight: 480, presetWidth: 704, presetHeight: 448,
      aspectRatio: "9:16")
    XCTAssertEqual(r.width, 960)
    XCTAssertEqual(r.height, 576)
    XCTAssertEqual(r.reason, .explicit)
  }

  func testExplicitAspectRatioNeverOverridesTheSourceImage() {
    // Real traffic sends a DEFAULTED "16:9" on i2v requests. Honouring that as
    // an intent would force a portrait source landscape — the #405 bug itself.
    let r = resolveI2V(source: (576, 1024), budget: (832, 480), aspectRatio: "16:9")
    XCTAssertLessThan(r.width, r.height)
    XCTAssertEqual(r.reason, .sourceAspect)
  }

  func testUnorientedT2VBudgetHonoursTheAspectLabel() {
    // Additive: a t2v request with no explicit dims and no resolution label
    // used to render the 704x448 landscape default however it was oriented.
    // The correction is an AXIS SWAP — the pixel count is preserved exactly
    // (nothing may shrink), and 448x704 is the same pair the MCP tool already
    // synthesizes client-side for "9:16".
    let portrait = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, aspectRatio: "9:16")
    XCTAssertEqual(portrait.width, 448)
    XCTAssertEqual(portrait.height, 704)
    XCTAssertEqual(portrait.reason, .explicit)
    XCTAssertEqual(portrait.width * portrait.height, 704 * 448, "an axis swap must not shrink")

    // A landscape label on the already-landscape default changes nothing.
    let landscape = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, aspectRatio: "16:9")
    XCTAssertEqual(landscape.width, 704)
    XCTAssertEqual(landscape.height, 448)

    // 1:1 has no orientation to correct — leave the budget alone.
    let square = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, aspectRatio: "1:1")
    XCTAssertEqual(square.width, 704)
    XCTAssertEqual(square.height, 448)

    // A resolution label already carries the orientation: untouched.
    let named = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, namedWidth: 832, namedHeight: 480,
      aspectRatio: "9:16")
    XCTAssertEqual(named.width, 832)
    XCTAssertEqual(named.height, 512)
  }

  // MARK: - Budget priority chain (unchanged production contract)

  func testBudgetPriorityChain() {
    XCTAssertEqual(
      VideoDimensionResolver.budget(
        requestWidth: 1, requestHeight: 2, namedWidth: 3, namedHeight: 4,
        presetWidth: 5, presetHeight: 6, configWidth: 7, configHeight: 8).width, 1)
    XCTAssertEqual(
      VideoDimensionResolver.budget(
        requestWidth: nil, requestHeight: nil, namedWidth: 3, namedHeight: 4,
        presetWidth: 5, presetHeight: 6, configWidth: 7, configHeight: 8).width, 3)
    XCTAssertEqual(
      VideoDimensionResolver.budget(
        requestWidth: nil, requestHeight: nil, namedWidth: nil, namedHeight: nil,
        presetWidth: 5, presetHeight: 6, configWidth: 7, configHeight: 8).width, 5)
    XCTAssertEqual(
      VideoDimensionResolver.budget(
        requestWidth: nil, requestHeight: nil, namedWidth: nil, namedHeight: nil,
        presetWidth: nil, presetHeight: nil, configWidth: 7, configHeight: 8).width, 7)
    XCTAssertEqual(
      VideoDimensionResolver.budget(
        requestWidth: nil, requestHeight: nil, namedWidth: nil, namedHeight: nil,
        presetWidth: nil, presetHeight: nil, configWidth: nil, configHeight: nil).width, 704)
  }

  // MARK: - aspect_ratio label parsing

  func testAspectLabelParsing() {
    XCTAssertEqual(VideoDimensionResolver.aspect(fromLabel: "16:9")!, 16.0 / 9.0, accuracy: 1e-9)
    XCTAssertEqual(VideoDimensionResolver.aspect(fromLabel: "9:16")!, 9.0 / 16.0, accuracy: 1e-9)
    XCTAssertEqual(VideoDimensionResolver.aspect(fromLabel: " 1:1 ")!, 1.0, accuracy: 1e-9)
    XCTAssertEqual(VideoDimensionResolver.aspect(fromLabel: "4x5")!, 0.8, accuracy: 1e-9)
    XCTAssertNil(VideoDimensionResolver.aspect(fromLabel: nil))
    XCTAssertNil(VideoDimensionResolver.aspect(fromLabel: ""))
    XCTAssertNil(VideoDimensionResolver.aspect(fromLabel: "widescreen"))
    XCTAssertNil(VideoDimensionResolver.aspect(fromLabel: "16:0"))
    XCTAssertNil(VideoDimensionResolver.aspect(fromLabel: "-16:9"))
  }

  // MARK: - Reason reporting (ruling 4)

  func testReasonAndBudgetAreReported() {
    let i2v = resolveI2V(source: (576, 1024), budget: (832, 480))
    XCTAssertEqual(i2v.reason.rawValue, "source_aspect")
    XCTAssertEqual(i2v.budgetWidth, 832)
    XCTAssertEqual(i2v.budgetHeight, 480)
    XCTAssertEqual(i2v.sourceWidth, 576)
    XCTAssertEqual(i2v.sourceHeight, 1024)
    XCTAssertTrue(i2v.adjusted)

    let t2v = VideoDimensionResolver.resolve(requestWidth: 704, requestHeight: 448)
    XCTAssertEqual(t2v.reason.rawValue, "explicit")
    XCTAssertFalse(t2v.adjusted)
    XCTAssertNil(t2v.sourceWidth)

    let fallback = VideoDimensionResolver.resolve(requestWidth: nil, requestHeight: nil)
    XCTAssertEqual(fallback.reason.rawValue, "default")
  }

  func testUnreadableSourceFallsBackToTheBudgetShape() {
    // imagePixelSize returning nil must not crash or invent an aspect.
    let r = VideoDimensionResolver.resolve(
      requestWidth: 480, requestHeight: 704, sourceWidth: nil, sourceHeight: nil)
    XCTAssertEqual(r.width, 512)   // 480 snaps to 512
    XCTAssertEqual(r.height, 704)
    XCTAssertEqual(r.reason, .explicit)

    let degenerate = VideoDimensionResolver.resolve(
      requestWidth: 480, requestHeight: 704, sourceWidth: 0, sourceHeight: 0)
    XCTAssertEqual(degenerate.width, 512)
    XCTAssertEqual(degenerate.height, 704)
  }

  // MARK: - Production regression pins (review ruling 1: nothing may change)
  //
  // Every distinct (budget, source) -> dims triple that appears in the live
  // engine log (~/.comfybox/serve.err.log, all 474 "LTX-2 I2V: adjusted"
  // lines). A scripted replay of the old and new algorithms over that log
  // reports 0/474 differences and an exhaustive budget x source sweep reports
  // none outside the >4096 safety clamp; these pin the highest-volume shapes
  // in the test suite so the equivalence cannot silently rot.
  func testEveryProductionShapeIsByteIdenticalToTheDeployedEngine() {
    let cases: [(budget: (Int, Int), source: (Int, Int), want: (Int, Int), n: Int)] = [
      ((480, 832), (896, 1664), (448, 832), 141),
      ((832, 480), (1664, 896), (832, 448), 118),
      ((1280, 720), (1664, 896), (1280, 704), 57),
      ((480, 832), (896, 1120), (512, 640), 50),
      ((832, 480), (1024, 1024), (640, 640), 18),
      ((384, 640), (576, 1024), (320, 576), 11),
      ((480, 832), (576, 1024), (512, 896), 9),
      ((480, 832), (832, 1216), (576, 832), 6),
      ((480, 480), (1024, 1024), (448, 448), 6),
      ((704, 448), (1024, 1024), (576, 576), 5),
      ((480, 832), (1024, 1024), (640, 640), 4),
      ((1280, 720), (1024, 1024), (960, 960), 4),
      ((832, 480), (1344, 768), (896, 512), 3),
      ((480, 832), (448, 704), (448, 704), 3),
      ((224, 352), (1664, 896), (448, 256), 3),
      ((832, 480), (1024, 576), (896, 512), 2),
      ((832, 480), (768, 1024), (576, 768), 2),
      ((704, 448), (1280, 768), (640, 384), 2),
      ((480, 832), (1024, 640), (832, 512), 2),
      ((384, 256), (1280, 768), (448, 256), 2),
      ((960, 576), (1024, 640), (1024, 640), 1),
      ((896, 512), (1280, 768), (960, 576), 1),
      ((832, 480), (768, 768), (640, 640), 1),
      ((448, 704), (1664, 896), (832, 448), 1),
    ]
    for c in cases {
      let r = resolveI2V(source: c.source, budget: c.budget)
      XCTAssertEqual(
        [r.width, r.height], [c.want.0, c.want.1],
        "budget \(c.budget) + source \(c.source) (\(c.n) renders in the log) must not change")
      XCTAssertEqual(r.reason, .sourceAspect)
    }
  }

  // MARK: - Upper safety clamp (review ruling 3)

  func testDegenerateStripIsClamped() {
    // A 10x9999 source: the fit falls through to its ideal-dims fallback and
    // would hand the pipeline a 256x17728 allocation.
    let r = VideoDimensionResolver.resolve(
      requestWidth: 704, requestHeight: 448,
      sourceWidth: 10, sourceHeight: 9999, hasInitImage: true)
    XCTAssertLessThanOrEqual(
      max(r.width, r.height), VideoDimensionResolver.maxVideoLongEdge,
      "long edge must be capped")
    XCTAssertLessThanOrEqual(
      r.width * r.height, VideoDimensionResolver.maxVideoPixels,
      "pixel count must be capped")
    XCTAssertEqual(r.width % 64, 0)
    XCTAssertEqual(r.height % 64, 0)
    XCTAssertGreaterThanOrEqual(r.width, 256)
    XCTAssertGreaterThanOrEqual(r.height, 256)
    XCTAssertLessThan(r.width, r.height, "an extreme portrait source still renders portrait")
  }

  func testClampIsANoOpForEveryRealRenderSize() {
    // The largest dims in the production log are 1344x768; the largest
    // PREDICTED output across all 474 logged renders is 2016x1152. The clamp
    // must never touch either. (A replay over the log confirms 0/474 for both
    // — see scripts/replay-i2v-dims.py.)
    for dims in [(1344, 768), (2016, 1152), (1280, 704), (960, 960), (832, 448), (256, 256)] {
      let c = VideoDimensionResolver.clamp(width: dims.0, height: dims.1)
      XCTAssertEqual([c.width, c.height], [dims.0, dims.1], "\(dims) must not be clamped")
    }
  }

  func testVideoCeilingIsNotTheImagePathCap() {
    // Review round 2, item 2: 4096/4096^2 is the IMAGE cap (imageMemoryCaps,
    // PR #363) and no LTX render survives it, so it was not a ceiling at all.
    XCTAssertEqual(VideoDimensionResolver.maxVideoLongEdge, 2048)
    XCTAssertEqual(VideoDimensionResolver.maxVideoPixels, 2048 * 2048)
    let imageSized = VideoDimensionResolver.clamp(width: 4096, height: 4096)
    XCTAssertLessThanOrEqual(max(imageSized.width, imageSized.height), 2048)
  }

  func testClampHonoursCustomCaps() {
    let c = VideoDimensionResolver.clamp(
      width: 1344, height: 768, maxLongEdge: 704, maxPixels: 16_777_216)
    XCTAssertLessThanOrEqual(max(c.width, c.height), 704)
    XCTAssertEqual(c.width % 64, 0)
    XCTAssertEqual(c.height % 64, 0)
    // Aspect preserved within a /64 step.
    XCTAssertEqual(Double(c.width) / Double(c.height), 1344.0 / 768.0, accuracy: 0.1)

    let byPixels = VideoDimensionResolver.clamp(
      width: 2048, height: 2048, maxLongEdge: 2048, maxPixels: 1_048_576)
    XCTAssertLessThanOrEqual(byPixels.width * byPixels.height, 1_048_576)
  }

  // MARK: - Predicted vs measured output dims (review round 2, item 1)
  //
  // The first cut hardcoded x2 — the two-stage convention's own assumption.
  // Production `refine_scale` is 1.5 (LTX2ConfigResolver's builtin, clamped to
  // [1, 2] by LTX2Pipeline) and LTX2RefineGate can skip the refine outright,
  // so x2 named a size no shipping render produces.

  func testPredictionUsesTheResolvedRefineScaleNotATwoTimesAssumption() {
    // 512x320 generator dims at the SHIPPING scale: latent 16x10 -> round(x1.5)
    // = 24x15 -> 768x480. The x2 assumption would have said 1024x640.
    let shipping = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: 512, generatorHeight: 320, twoStage: true, refineScale: 1.5)
    XCTAssertEqual([shipping.width, shipping.height], [768, 480])
    XCTAssertNotEqual([shipping.width, shipping.height], [1024, 640], "x2 is not the shipping scale")

    // scale 2.0 (the upper clamp) does reproduce the old doubling.
    let doubled = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: 512, generatorHeight: 320, twoStage: true, refineScale: 2.0)
    XCTAssertEqual([doubled.width, doubled.height], [1024, 640])

    // Out-of-range scales are clamped exactly as LTX2Pipeline clamps them.
    XCTAssertEqual(
      [VideoDimensionResolver.predictedOutputDims(
        generatorWidth: 512, generatorHeight: 320, twoStage: true, refineScale: 9.0).width],
      [1024])
    XCTAssertEqual(
      [VideoDimensionResolver.predictedOutputDims(
        generatorWidth: 512, generatorHeight: 320, twoStage: true, refineScale: 0.1).width],
      [512])
  }

  /// comfybox#409, solved. The report: a t2v probe submitted at 512x320 with
  /// two-stage produced a 704x448 mp4 while the engine logged "output will be
  /// 1024x640". The FILE was right and the LOG was wrong — the deployed
  /// `LTX2_REFINE_SCALE` is 1.35 (production launchd plist; the builtin is
  /// 1.5, and only the two-stage log message ever assumed 2), and the refine
  /// upsamples the LATENT grid:
  ///
  ///   512/32 = 16 -> round(16 x 1.35) = 22 -> 704
  ///   320/32 = 10 -> round(10 x 1.35) = 14 -> 448
  ///
  /// `deliveryShortEdge` is 480 and the short edge is 448, so no delivery
  /// downscale applies. 704x448 is exactly correct.
  func testIssue409ProductionScaleReproducesTheDeliveredFile() {
    let p = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: 512, generatorHeight: 320, twoStage: true, refineScale: 1.35)
    XCTAssertEqual([p.width, p.height], [704, 448], "the mp4 comfybox#409 reported")
    XCTAssertNotEqual(
      [p.width, p.height], [1024, 640],
      "1024x640 was the x2 assumption in the log message, not a real output size")
  }

  func testRefineSkippedOrSingleScaleOutputsTheGeneratorDims() {
    let skipped = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: 832, generatorHeight: 448, twoStage: true, refineScale: 1.5,
      refineWillSkip: true)
    XCTAssertEqual([skipped.width, skipped.height], [832, 448])

    let single = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: 832, generatorHeight: 448, twoStage: false, refineScale: 1.5)
    XCTAssertEqual([single.width, single.height], [832, 448])
  }

  func testPredictionAtTheRealTwoStageWorkingPoints() {
    // Halved branch: a 960x576 request paints stage 1 at 512x320, and at the
    // shipping 1.5x the file comes back 768x480 — NOT the 1024x640 the
    // convention's x2 assumption predicts.
    let s1 = WarmServer.stageOneDims(finalWidth: 960, finalHeight: 576)
    XCTAssertTrue(s1.halved)
    XCTAssertEqual([s1.width, s1.height], [512, 320])
    let halved = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: s1.width, generatorHeight: s1.height, twoStage: true, refineScale: 1.5)
    XCTAssertEqual([halved.width, halved.height], [768, 480])

    // Below-the-floor branch: 704x448 is NOT halved and is painted as-is, so
    // the file comes back 1056x672 at 1.5x (the engine's warning says 1408x896
    // — that warning is computed from the same x2 assumption and is wrong).
    let floored = WarmServer.stageOneDims(finalWidth: 704, finalHeight: 448)
    XCTAssertFalse(floored.halved)
    let doubledBranch = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: floored.width, generatorHeight: floored.height,
      twoStage: true, refineScale: 1.5)
    XCTAssertEqual([doubledBranch.width, doubledBranch.height], [1056, 672])
  }

  func testTheOutputCeilingBindsTheGeneratorDimsNotJustTheReport() {
    // Every logged shape is already under the ceiling — the largest predicted
    // output in the production log is 2016x1152 from a 1344x768 two-stage
    // render, so nothing real is touched.
    let untouched = VideoDimensionResolver.generatorDimsFittingOutputCeiling(
      generatorWidth: 1344, generatorHeight: 768, twoStage: true, refineScale: 1.5)
    XCTAssertEqual([untouched.width, untouched.height], [1344, 768])
    let pred = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: 1344, generatorHeight: 768, twoStage: true, refineScale: 1.5)
    XCTAssertEqual([pred.width, pred.height], [2016, 1152])
    XCTAssertLessThanOrEqual(max(pred.width, pred.height), VideoDimensionResolver.maxVideoLongEdge)

    // A request whose refine would carry it past the ceiling is painted
    // smaller, so the REPORTED size stays truthful instead of being a clamped
    // lie about a file the encoder wrote at full size.
    let bound = VideoDimensionResolver.generatorDimsFittingOutputCeiling(
      generatorWidth: 1920, generatorHeight: 1088, twoStage: true, refineScale: 2.0)
    XCTAssertLessThan(bound.width, 1920)
    XCTAssertEqual(bound.width % 64, 0)
    XCTAssertEqual(bound.height % 64, 0)
    let boundPrediction = VideoDimensionResolver.predictedOutputDims(
      generatorWidth: bound.width, generatorHeight: bound.height,
      twoStage: true, refineScale: 2.0)
    XCTAssertLessThanOrEqual(
      max(boundPrediction.width, boundPrediction.height),
      VideoDimensionResolver.maxVideoLongEdge)
    XCTAssertLessThanOrEqual(
      boundPrediction.width * boundPrediction.height, VideoDimensionResolver.maxVideoPixels)
  }

  func testResolvedDimensionsCarryStage1Separately() {
    let d = ResolvedVideoDimensions(
      width: 768, height: 480, reason: .sourceAspect,
      budgetWidth: 960, budgetHeight: 576,
      sourceWidth: 1024, sourceHeight: 640,
      stage1Width: 512, stage1Height: 320)
    XCTAssertEqual(d.width, 768)           // predicted output at 1.5x
    XCTAssertEqual(d.stage1Width, 512)     // what the generator paints
    XCTAssertEqual(d.stage1Height, 320)

    let single = ResolvedVideoDimensions(
      width: 832, height: 448, reason: .sourceAspect,
      budgetWidth: 832, budgetHeight: 480)
    XCTAssertNil(single.stage1Width)
  }

  // MARK: - i2v with an unreadable init image (review round 2, item 4)

  func testUnreadableInitImagePlusAspectLabelDoesNotSwapTheBudget() {
    // The request IS i2v — the resolver just could not measure the source.
    // A (usually defaulted) "9:16" must not get to decide the shape.
    let unreadable = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil,
      sourceWidth: nil, sourceHeight: nil,
      aspectRatio: "9:16", hasInitImage: true)
    XCTAssertEqual(unreadable.width, 704)
    XCTAssertEqual(unreadable.height, 448, "an i2v budget must not be swapped by a label")
    XCTAssertEqual(unreadable.reason, .default)

    // Same request with NO init image at all IS t2v, and does swap.
    let t2v = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil,
      sourceWidth: nil, sourceHeight: nil,
      aspectRatio: "9:16", hasInitImage: false)
    XCTAssertEqual(t2v.width, 448)
    XCTAssertEqual(t2v.height, 704)
    XCTAssertEqual(t2v.reason, .explicit)
  }
}
