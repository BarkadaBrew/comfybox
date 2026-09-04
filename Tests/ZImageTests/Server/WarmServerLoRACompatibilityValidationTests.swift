import XCTest

@testable import ZImage

/// #313 (review round 1, Important finding 1): `POST /v1/loras/{id}/update`
/// accepted `model_compatibility` unvalidated — unlike its sibling
/// `krea2_relative` a few lines above it in `WarmServer.swift`, which 400s on
/// an unrecognized value. `WarmServer.validateModelCompatibilityTags` is the
/// pure, static piece the route now calls before building the `LoRAEntryPatch`
/// (mirrors `WarmServerRejectionTests`' pattern: exercise the pure validator +
/// `WarmServer.errorResponse(for:)`, no listening server needed).
final class WarmServerLoRACompatibilityValidationTests: XCTestCase {

  private func bodyString(_ response: HTTPResponse) -> String {
    String(decoding: response.body, as: UTF8.self)
  }

  func testKnownTagsPassThroughUnchanged() throws {
    for valid in LoRAScanner.knownCompatibilityTags {
      XCTAssertEqual(try WarmServer.validateModelCompatibilityTags([valid]), [valid])
    }
  }

  func testMixedCaseKnownTagPasses() throws {
    XCTAssertEqual(try WarmServer.validateModelCompatibilityTags(["Z-Image"]), ["Z-Image"])
  }

  func testUnknownTagIs400NamingValueAndValidSet() {
    XCTAssertThrowsError(try WarmServer.validateModelCompatibilityTags(["not-a-real-family"])) { error in
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      let body = bodyString(response)
      XCTAssertTrue(body.contains("not-a-real-family"), body)
      for valid in LoRAScanner.knownCompatibilityTags {
        XCTAssertTrue(body.contains(valid), "400 body must list valid tag '\(valid)': \(body)")
      }
    }
  }

  /// One valid tag alongside one bogus tag must still 400 — partial
  /// acceptance would silently drop the caller's intent for the bad entry.
  func testOneUnknownTagAmongValidOnesStill400s() {
    XCTAssertThrowsError(try WarmServer.validateModelCompatibilityTags(["ltx", "not-real"])) { error in
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      XCTAssertTrue(bodyString(response).contains("not-real"))
    }
  }

  func testEmptyArrayIs400() {
    XCTAssertThrowsError(try WarmServer.validateModelCompatibilityTags([])) { error in
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      XCTAssertTrue(bodyString(response).contains("model_compatibility"))
    }
  }
}
