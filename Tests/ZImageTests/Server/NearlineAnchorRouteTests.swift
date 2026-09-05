// NearlineAnchorRouteTests.swift — #273: POST /v1/nearline/anchor decode
// shape and the GET /v1/nearline item wire shape (the additive `anchored`
// key). Pure decode/serialization: no server, no weights.

import XCTest
@testable import ZImage

final class NearlineAnchorRouteTests: XCTestCase {

  // MARK: - Request decode

  func testAnchorBodyDecodesAllFields() throws {
    let json = """
      {"kind":"lora","id":"kroma-lora-v0.3.safetensors","anchored":true}
      """
    let body = try JSONDecoder().decode(WarmServer.NearlineAnchorBody.self, from: Data(json.utf8))
    XCTAssertEqual(body.kind, "lora")
    XCTAssertEqual(body.id, "kroma-lora-v0.3.safetensors")
    XCTAssertTrue(body.anchored)
  }

  func testAnchorBodyDecodesModelKindAndFalse() throws {
    let json = """
      {"kind":"model","id":"kroma-v0.2.safetensors","anchored":false}
      """
    let body = try JSONDecoder().decode(WarmServer.NearlineAnchorBody.self, from: Data(json.utf8))
    XCTAssertEqual(body.kind, "model")
    XCTAssertFalse(body.anchored)
  }

  func testAnchorBodyMissingAnchoredFieldFailsToDecode() {
    let json = """
      {"kind":"lora","id":"x.safetensors"}
      """
    XCTAssertThrowsError(try JSONDecoder().decode(WarmServer.NearlineAnchorBody.self, from: Data(json.utf8)))
  }

  func testAnchorBodyMissingIdFieldFailsToDecode() {
    let json = """
      {"kind":"lora","anchored":true}
      """
    XCTAssertThrowsError(try JSONDecoder().decode(WarmServer.NearlineAnchorBody.self, from: Data(json.utf8)))
  }

  // MARK: - Response shape (GET /v1/nearline items[])

  func testNearlineItemJSONIncludesAnchoredKey() {
    let item = NearlineItem(
      name: "pinned.safetensors", path: "/vol/pinned.safetensors", sizeMB: 12, kind: "lora",
      stagedPath: "/local/pinned.safetensors", anchored: true)
    let dict = WarmServer.nearlineItemJSON(item, iso: ISO8601DateFormatter())

    XCTAssertEqual(dict["anchored"] as? Bool, true)
    XCTAssertEqual(dict["name"] as? String, "pinned.safetensors")
    XCTAssertEqual(dict["staged"] as? Bool, true)
    XCTAssertEqual(dict["staged_path"] as? String, "/local/pinned.safetensors")
  }

  func testNearlineItemJSONReportsAnchoredFalseByDefault() {
    let item = NearlineItem(name: "free.safetensors", path: "/vol/free.safetensors", sizeMB: 4, kind: "lora")
    let dict = WarmServer.nearlineItemJSON(item, iso: ISO8601DateFormatter())
    XCTAssertEqual(dict["anchored"] as? Bool, false)
  }

  /// The whole payload must serialize through Foundation's JSON writer —
  /// catches an accidental non-JSON-object value sneaking into the dict.
  func testNearlineItemJSONIsValidJSONObject() throws {
    let item = NearlineItem(name: "x.safetensors", path: "/vol/x.safetensors", sizeMB: 1, kind: "model", anchored: true)
    let dict = WarmServer.nearlineItemJSON(item, iso: ISO8601DateFormatter())
    XCTAssertTrue(JSONSerialization.isValidJSONObject(dict))
  }

  // MARK: - #273 fix round 1 (C2): error -> HTTP status mapping

  func testInsufficientCapacityMapsTo507() {
    let status = WarmServer.httpStatus(
      for: .insufficientCapacity(needMB: 100, freeMB: 10, anchoredMB: 90))
    XCTAssertEqual(status, 507)
  }

  func testUnknownItemMapsTo404() {
    XCTAssertEqual(WarmServer.httpStatus(for: .unknownItem("x")), 404)
  }

  func testSourceMissingMapsTo404() {
    XCTAssertEqual(WarmServer.httpStatus(for: .sourceMissing("/vol/x")), 404)
  }
}
