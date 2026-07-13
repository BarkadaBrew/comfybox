// VideoDimsDerivationTests.swift — LTX-2 render-dim derivation (#219)
//
// LTX-2 renders at dims that are 32-multiples but NOT 64-multiples (e.g. 480)
// exhibit progressive haze; I2V output must match the source image aspect.
// These pin WarmServer.snapDim64 / deriveVideoDims, the pure helpers behind
// the server-side dim adjustment in prepareLocalVideo.

import XCTest
@testable import ZImage

final class VideoDimsDerivationTests: XCTestCase {

  // MARK: snapDim64

  func testSnapExactMultiplesUnchanged() {
    XCTAssertEqual(WarmServer.snapDim64(704), 704)
    XCTAssertEqual(WarmServer.snapDim64(448), 448)
    XCTAssertEqual(WarmServer.snapDim64(512), 512)
  }

  func testSnapHazeBucket480RoundsToSafeDim() {
    // 480 = 32*15 — the empirically hazy bucket. 480/64 = 7.5 rounds to 8 -> 512.
    XCTAssertEqual(WarmServer.snapDim64(480), 512)
  }

  func testSnapRoundsToNearest() {
    XCTAssertEqual(WarmServer.snapDim64(465), 448)   // 7.27 -> 7
    XCTAssertEqual(WarmServer.snapDim64(672), 704)   // 10.5 rounds half-away-from-zero -> 11
  }

  func testSnapFloor256() {
    XCTAssertEqual(WarmServer.snapDim64(10), 256)
    XCTAssertEqual(WarmServer.snapDim64(0), 256)
  }

  // MARK: deriveVideoDims

  func testPortraitSourceGetsPortraitDims() {
    // 832x1216 source (aspect 0.684) with the default 704x448 budget.
    let dims = WarmServer.deriveVideoDims(
      sourceWidth: 832, sourceHeight: 1216, budgetWidth: 704, budgetHeight: 448)
    XCTAssertLessThan(dims.width, dims.height, "portrait source must yield portrait dims")
    XCTAssertEqual(dims.width % 64, 0)
    XCTAssertEqual(dims.height % 64, 0)
    // Area stays within ~35% of the requested budget.
    let budgetArea = 704 * 448
    XCTAssertLessThan(abs(dims.width * dims.height - budgetArea), Int(Double(budgetArea) * 0.35))
  }

  func testLandscapeSourceKeepsLandscapeDims() {
    let dims = WarmServer.deriveVideoDims(
      sourceWidth: 1056, sourceHeight: 672, budgetWidth: 704, budgetHeight: 448)
    XCTAssertGreaterThan(dims.width, dims.height)
    XCTAssertEqual(dims.width % 64, 0)
    XCTAssertEqual(dims.height % 64, 0)
  }

  func testMatchedAspectAlreadySafeIsStable() {
    // A 704x448-shaped source with the same budget should stay 704x448.
    let dims = WarmServer.deriveVideoDims(
      sourceWidth: 1408, sourceHeight: 896, budgetWidth: 704, budgetHeight: 448)
    XCTAssertEqual(dims.width, 704)
    XCTAssertEqual(dims.height, 448)
  }

  func testDegenerateSourceFallsBackToSnappedBudget() {
    let dims = WarmServer.deriveVideoDims(
      sourceWidth: 0, sourceHeight: 0, budgetWidth: 480, budgetHeight: 704)
    XCTAssertEqual(dims.width, 512)   // 480 snaps to 512
    XCTAssertEqual(dims.height, 704)
  }

  func testSquareSource() {
    let dims = WarmServer.deriveVideoDims(
      sourceWidth: 1024, sourceHeight: 1024, budgetWidth: 704, budgetHeight: 448)
    XCTAssertEqual(dims.width, dims.height)
    XCTAssertEqual(dims.width % 64, 0)
    XCTAssertGreaterThanOrEqual(dims.width, 512)
  }
}
