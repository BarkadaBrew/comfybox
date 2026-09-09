import XCTest
@testable import ZImage

final class LTX2ImageRequestTests: XCTestCase {
  private func decode(_ json: String) throws -> GeneratePayload {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  func testLTXEngineDecodesAndValidatesNativeDefaults() throws {
    let payload = try decode(#"{"prompt":"a butterfly","engine":"ltx2"}"#)
    XCTAssertTrue(payload.usesLTX2ImageEngine)
    XCTAssertNoThrow(try payload.validateEngine())
    XCTAssertNoThrow(try payload.validateLTX2ImageFields())
  }

  func testDefaultEngineDoesNotSelectLTX() throws {
    let payload = try decode(#"{"prompt":"a butterfly","engine":"default"}"#)
    XCTAssertFalse(payload.usesLTX2ImageEngine)
    XCTAssertNoThrow(try payload.validateEngine())
  }

  func testLTXImageRejectsInvalidSpatialGrid() throws {
    let payload = try decode(
      #"{"prompt":"a butterfly","engine":"ltx2","width":1279,"height":704}"#)
    XCTAssertThrowsError(try payload.validateLTX2ImageFields())
  }

  func testLTXImageRejectsImg2ImgFieldsUntilNativePathIsWired() throws {
    let payload = try decode(
      #"{"prompt":"change it","engine":"ltx2","image_path":"/tmp/source.png"}"#)
    XCTAssertThrowsError(try payload.validateLTX2ImageFields())
  }

  func testServerFramePolicyPreservesTheNativeSingleFrameRequest() {
    XCTAssertEqual(
      WarmServer.resolvedLTX2Frames(
        requestFrames: 1, videoConfigDefaults: VideoDefaultValues(frames: 121)),
      1)
  }

  func testLTXImageRejectsImagePresetFieldsItCannotApply() throws {
    let payload = try decode(
      #"{"prompt":"a butterfly","engine":"ltx2","preset":"portrait"}"#)
    XCTAssertThrowsError(try payload.validateLTX2ImageFields())
  }

  func testUnknownEngineFailsLoud() throws {
    let payload = try decode(#"{"prompt":"a butterfly","engine":"something-else"}"#)
    XCTAssertThrowsError(try payload.validateEngine())
  }
}
