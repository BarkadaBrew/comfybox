import XCTest
@testable import ZImage

/// The rewriter-proof trigger guarantee (Todd 2026-08-11): applied LoRAs'
/// triggers are re-asserted on the FINAL prompt engine-side.
final class LoRATriggerGuardTests: XCTestCase {
  func testPrefixesMissingTriggers() {
    XCTAssertEqual(
      LoRATriggerGuard.ensure(prompt: "a portrait at dusk", triggers: ["Dazzin", "c2n0n"]),
      "Dazzin, c2n0n, a portrait at dusk")
  }
  func testLeavesPresentTriggersAlone() {
    let p = "Dazzin, c2n0n, a portrait at dusk"
    XCTAssertEqual(LoRATriggerGuard.ensure(prompt: p, triggers: ["Dazzin", "c2n0n"]), p)
  }
  func testPartialAndCaseInsensitive() {
    XCTAssertEqual(
      LoRATriggerGuard.ensure(prompt: "dazzin smiles softly", triggers: ["Dazzin", "c2n0n"]),
      "c2n0n, dazzin smiles softly")
  }
  func testEmptyTriggersNoOp() {
    XCTAssertEqual(LoRATriggerGuard.ensure(prompt: "x", triggers: []), "x")
  }
}
