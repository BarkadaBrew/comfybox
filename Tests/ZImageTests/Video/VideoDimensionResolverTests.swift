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
      aspectRatio: aspectRatio)
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
    // would hand the pipeline a 256x17792 allocation.
    let r = VideoDimensionResolver.resolve(
      requestWidth: 704, requestHeight: 448,
      sourceWidth: 10, sourceHeight: 9999)
    XCTAssertLessThanOrEqual(max(r.width, r.height), 4096, "long edge must be capped")
    XCTAssertLessThanOrEqual(r.width * r.height, 16_777_216, "pixel count must be capped")
    XCTAssertEqual(r.width % 64, 0)
    XCTAssertEqual(r.height % 64, 0)
    XCTAssertGreaterThanOrEqual(r.width, 256)
    XCTAssertGreaterThanOrEqual(r.height, 256)
    XCTAssertLessThan(r.width, r.height, "an extreme portrait source still renders portrait")
  }

  func testClampIsANoOpForEveryRealRenderSize() {
    // The largest dims in the production log are 1344x768 — three times under
    // the long-edge cap. The clamp must never touch a real request.
    for dims in [(1344, 768), (1280, 704), (960, 960), (832, 448), (256, 256), (4096, 4096)] {
      let c = VideoDimensionResolver.clamp(width: dims.0, height: dims.1)
      XCTAssertEqual([c.width, c.height], [dims.0, dims.1])
    }
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
      width: 2048, height: 2048, maxLongEdge: 4096, maxPixels: 1_048_576)
    XCTAssertLessThanOrEqual(byPixels.width * byPixels.height, 1_048_576)
  }

  // MARK: - Reported dims are the FINAL output dims (review ruling 2)

  func testOutputDimsAccountForTheTwoStageDoubling() {
    // Single scale: what the generator gets is what the caller receives.
    let single = VideoDimensionResolver.outputDims(
      generatorWidth: 832, generatorHeight: 448, twoStage: false)
    XCTAssertEqual([single.width, single.height], [832, 448])

    // Two-stage, halved branch: 960x576 request -> stage 1 paints 512x320 ->
    // the refine doubles back to 1024x640, which is what the file has.
    let s1 = WarmServer.stageOneDims(finalWidth: 960, finalHeight: 576)
    XCTAssertTrue(s1.halved)
    let halved = VideoDimensionResolver.outputDims(
      generatorWidth: s1.width, generatorHeight: s1.height, twoStage: true)
    XCTAssertEqual([halved.width, halved.height], [1024, 640])

    // Two-stage, below-the-floor branch: a 704x448 request is NOT halved and
    // is treated as stage-1 dims, so the output is 1408x896 — the size the
    // engine's own warning names, and the one `resolved_*` must report.
    let floored = WarmServer.stageOneDims(finalWidth: 704, finalHeight: 448)
    XCTAssertFalse(floored.halved)
    let doubled = VideoDimensionResolver.outputDims(
      generatorWidth: floored.width, generatorHeight: floored.height, twoStage: true)
    XCTAssertEqual([doubled.width, doubled.height], [1408, 896])
  }

  func testResolvedDimensionsCarryStage1Separately() {
    let d = ResolvedVideoDimensions(
      width: 1024, height: 640, reason: .sourceAspect,
      budgetWidth: 960, budgetHeight: 576,
      sourceWidth: 1024, sourceHeight: 640,
      stage1Width: 512, stage1Height: 320)
    XCTAssertEqual(d.width, 1024)          // what the caller receives
    XCTAssertEqual(d.stage1Width, 512)     // what the generator painted
    XCTAssertEqual(d.stage1Height, 320)

    let single = ResolvedVideoDimensions(
      width: 832, height: 448, reason: .sourceAspect,
      budgetWidth: 832, budgetHeight: 480)
    XCTAssertNil(single.stage1Width)
  }
}
