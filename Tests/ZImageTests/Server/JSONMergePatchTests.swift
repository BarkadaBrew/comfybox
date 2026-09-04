// JSONMergePatchTests.swift — pure RFC 7386 semantics, independent of
// ServerConfigStore (FDD-ui-api-parity §3.3). ServerConfigStoreTests covers
// the same semantics through the actual PATCH /v1/config write path; this
// file pins the algorithm itself.

import XCTest
@testable import ZImage

final class JSONMergePatchTests: XCTestCase {

  /// Parse a JSON string into the same `[String: Any]` shape production code
  /// actually works with (`JSONSerialization.jsonObject`) — sidesteps Swift
  /// dictionary-literal type inference, which does not reliably unify sibling
  /// branches of DIFFERENT inferred value types (e.g. one branch mixing Int +
  /// Double, another Int-only) into `[String: Any]` at every nesting level the
  /// way hand-written nested literals might suggest.
  private func jsonObject(_ json: String) -> [String: Any] {
    (try! JSONSerialization.jsonObject(with: Data(json.utf8))) as! [String: Any]
  }

  func testOmittedKeysAreUnchanged() {
    let target: [String: Any] = ["a": 1, "b": 2]
    let result = JSONMergePatch.apply(patch: ["a": 10], to: target) as? [String: Any]
    XCTAssertEqual(result?["a"] as? Int, 10)
    XCTAssertEqual(result?["b"] as? Int, 2)
  }

  func testNullDeletesTheKey() {
    let target: [String: Any] = ["a": 1, "b": 2]
    let result = JSONMergePatch.apply(patch: ["a": NSNull()], to: target) as? [String: Any]
    XCTAssertNil(result?["a"])
    XCTAssertEqual(result?["b"] as? Int, 2)
  }

  func testNullOnAnAbsentKeyIsHarmless() {
    let target: [String: Any] = ["a": 1]
    let result = JSONMergePatch.apply(patch: ["z": NSNull()], to: target) as? [String: Any]
    XCTAssertEqual(result?.count, 1)
    XCTAssertEqual(result?["a"] as? Int, 1)
  }

  func testNestedObjectMergesRatherThanReplaces() {
    let target: [String: Any] = ["outer": ["x": 1, "y": 2]]
    let result = JSONMergePatch.apply(patch: ["outer": ["y": 20]], to: target) as? [String: Any]
    let outer = result?["outer"] as? [String: Any]
    XCTAssertEqual(outer?["x"] as? Int, 1, "keys the patch doesn't mention survive")
    XCTAssertEqual(outer?["y"] as? Int, 20)
  }

  func testNonObjectPatchReplacesWholesale() {
    // RFC 7386 §2: if the patch (at any level) is not an object, it replaces
    // the target outright — an array, string, number or bool patch value
    // never merges field-by-field.
    let target: [String: Any] = ["list": [1, 2, 3]]
    let result = JSONMergePatch.apply(patch: ["list": [9]], to: target) as? [String: Any]
    XCTAssertEqual(result?["list"] as? [Int], [9])
  }

  func testMissingTargetKeyTreatedAsEmptyObjectForNestedMerge() {
    // Patching a nested object that doesn't exist yet on the target creates it.
    let target: [String: Any] = [:]
    let result = JSONMergePatch.apply(patch: ["outer": ["x": 1]], to: target) as? [String: Any]
    let outer = result?["outer"] as? [String: Any]
    XCTAssertEqual(outer?["x"] as? Int, 1)
  }

  func testTopLevelPatchThatIsNotAnObjectReplacesTargetEntirely() {
    // Degenerate but well-defined by the RFC: patching with a scalar at the
    // top replaces the whole document. ServerConfigStore's route handler
    // rejects this before it reaches the algorithm (config is always an
    // object), but the pure function itself must still do the RFC-correct
    // thing rather than crash.
    let result = JSONMergePatch.apply(patch: "scalar", to: ["a": 1])
    XCTAssertEqual(result as? String, "scalar")
  }

  func testDeeplyNestedMergePreservesSiblingBranches() {
    let target = jsonObject("""
    { "renderDefaults": { "byFamily": {
        "fibo": { "steps": 30, "guidance": 4.0 },
        "chroma": { "steps": 28 }
    } } }
    """)
    let patch = jsonObject("""
    { "renderDefaults": { "byFamily": { "fibo": { "steps": 40 } } } }
    """)
    let result = JSONMergePatch.apply(patch: patch, to: target) as? [String: Any]
    let byFamily = (result?["renderDefaults"] as? [String: Any])?["byFamily"] as? [String: Any]
    let fibo = byFamily?["fibo"] as? [String: Any]
    let chroma = byFamily?["chroma"] as? [String: Any]
    XCTAssertEqual(fibo?["steps"] as? Int, 40, "the patched field updates")
    XCTAssertEqual(fibo?["guidance"] as? Double, 4.0, "a sibling field in the SAME object survives")
    XCTAssertEqual(chroma?["steps"] as? Int, 28, "an entirely different family branch is untouched")
  }
}
