import XCTest
@testable import ZImage

/// Repro for the lost request-level tuning (2026-08-11): the wire body
/// carries tuning.stage1_sigmas (verified in the render trace) but the
/// render resolves (env). Decode the EXACT failing body shape.
final class LocalVideoRequestDecodeTests: XCTestCase {
  func testTuningSurvivesFullBodyDecode() throws {
    let body = """
    {"prompt":"x","preset":"kira-video-avocado","width":480,"height":832,
     "frames":97,"seed":4242,"fps":24,"audio":true,"enhance":false,
     "source":"tuning",
     "tuning":{"stage1_sigmas":[1.0,0.955,0.893,0.812,0.715,0.603,0.482,0.241,0.121,0.0]},
     "output_path":"/tmp/x.mp4"}
    """
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let req = try decoder.decode(WarmServer.LocalVideoRequest.self, from: Data(body.utf8))
    XCTAssertNotNil(req.tuning, "tuning must decode")
    XCTAssertEqual(req.tuning?.stage1Sigmas?.count, 10, "sigmas must survive")
  }

  // MARK: - comfybox#307: top-level `two_pass` convenience field

  private func decodeLocalVideoRequest(_ json: String) throws -> WarmServer.LocalVideoRequest {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(WarmServer.LocalVideoRequest.self, from: Data(json.utf8))
  }

  func testTwoPassTrueDecodes() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":true}"#)
    XCTAssertEqual(req.twoPass, true)
  }

  func testTwoPassFalseDecodes() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":false}"#)
    XCTAssertEqual(req.twoPass, false)
  }

  func testTwoPassExplicitNullDecodesAsNil() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x","two_pass":null}"#)
    XCTAssertNil(req.twoPass)
  }

  func testTwoPassAbsentDecodesAsNil() throws {
    let req = try decodeLocalVideoRequest(#"{"prompt":"x"}"#)
    XCTAssertNil(req.twoPass, "no two_pass key, no tuning key — additive field must not be required")
  }

  /// `two_pass` is additive: it must not disturb the pre-existing
  /// `tuning.two_stage` decode path exercised above.
  func testTwoPassAlongsideExistingTuningBlockBothDecode() throws {
    let req = try decodeLocalVideoRequest(#"""
      {"prompt":"x","two_pass":true,"tuning":{"refine_scale":1.35}}
      """#)
    XCTAssertEqual(req.twoPass, true)
    XCTAssertEqual(req.tuning?.refineScale, 1.35)
  }
}
