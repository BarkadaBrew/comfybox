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

  // MARK: - The ticket's pinned examples (480p budget = 832x480 landscape)

  func testPortraitSourceAt480pRendersPortrait() {
    // 576x1024 (9:16) source, 480p budget. The bug: 704x448 landscape.
    let r = resolveI2V(source: (576, 1024), budget: (832, 480))
    XCTAssertEqual(r.width, 448)
    XCTAssertEqual(r.height, 768)
    XCTAssertEqual(r.reason, .sourceAspect)
    XCTAssertLessThan(r.width, r.height, "a 9:16 source must not render landscape")
  }

  func testLandscapeSourceAt480pRendersLandscape() {
    let r = resolveI2V(source: (1024, 576), budget: (832, 480))
    XCTAssertEqual(r.width, 768)
    XCTAssertEqual(r.height, 448)
    XCTAssertEqual(r.reason, .sourceAspect)
  }

  func testSquareSourceAt480pRendersNearestValidSquareInBudget() {
    // 1:1 source. 576x576 (331 776 px) is the largest /64 square inside the
    // 480p budget (399 360 px); 640x640 would overshoot it by 2.6%.
    let r = resolveI2V(source: (1024, 1024), budget: (832, 480))
    XCTAssertEqual(r.width, r.height, "a square source must render square")
    XCTAssertEqual(r.width, 576)
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
    let r = VideoDimensionResolver.resolve(
      requestWidth: nil, requestHeight: nil, aspectRatio: "9:16")
    XCTAssertLessThan(r.width, r.height)
    XCTAssertEqual(r.width % 64, 0)
    XCTAssertEqual(r.height % 64, 0)
    XCTAssertEqual(r.reason, .explicit)
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

  // MARK: - Production regression pins
  //
  // Taken verbatim from the live engine log (~/.comfybox/serve.err.log,
  // "LTX-2 I2V: adjusted <budget> -> <result> (source WxH)"). The three marked
  // CHANGED are the ones #405 deliberately moves; everything else must keep
  // rendering at exactly the size Kira's scheduler has been getting.
  func testProductionCasesFromTheLiveEngineLog() {
    let cases: [(budget: (Int, Int), source: (Int, Int), want: (Int, Int), note: String)] = [
      ((480, 832), (896, 1664), (448, 832), "141 renders — unchanged"),
      ((832, 480), (1664, 896), (832, 448), "118 renders — unchanged"),
      ((1280, 720), (1664, 896), (1280, 704), "57 renders — unchanged"),
      ((480, 832), (896, 1120), (512, 640), "50 renders — unchanged"),
      ((384, 640), (576, 1024), (320, 576), "scheduler i2v — unchanged"),
      ((832, 480), (1024, 1024), (576, 576), "CHANGED 640x640 -> 576x576 (was 2.6% over the 480p budget)"),
      ((480, 832), (576, 1024), (448, 768), "CHANGED 512x896 -> 448x768 (the ticket's pinned example)"),
      ((448, 704), (1664, 896), (704, 384), "CHANGED 768x384 -> 704x384 (7.7% squash -> 1.3%)"),
    ]
    for c in cases {
      let r = resolveI2V(source: c.source, budget: c.budget)
      XCTAssertEqual(
        [r.width, r.height], [c.want.0, c.want.1],
        "budget \(c.budget) + source \(c.source): \(c.note)")
    }
  }
}
