import XCTest
@testable import ZImage

/// Regression tests for the /v1/generate decode path, which uses
/// `.convertFromSnakeCase`. That strategy rewrites incoming JSON keys to
/// camelCase BEFORE matching CodingKey stringValues, so any explicit snake_case
/// CodingKey rawValue silently fails to match (the inpaint bug: mask + base
/// image dropped to nil, turning inpaint into plain txt2img).
final class GeneratePayloadDecodeTests: XCTestCase {

    /// Mirrors WarmServer.decode(_:from:).
    private func decode(_ json: String) throws -> GeneratePayload {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return try d.decode(GeneratePayload.self, from: Data(json.utf8))
    }

    func testInpaintImageAndMaskDecodeThroughSnakeCase() throws {
        // base64 of "hi" so Data(base64Encoded:) succeeds.
        let json = """
        {"prompt":"x","width":512,"height":512,
         "inpaint_image_base64":"aGk=","mask_base64":"aGk=",
         "mask_grow":8,"mask_feather":8,"denoise":0.9}
        """
        let p = try decode(json)
        XCTAssertNotNil(p.inpaintImageData, "inpaint image must survive .convertFromSnakeCase")
        XCTAssertNotNil(p.maskData, "mask must survive .convertFromSnakeCase")
        XCTAssertEqual(p.maskGrow, 8)
        XCTAssertEqual(p.maskFeather, 8)
        XCTAssertEqual(p.denoise, 0.9)
    }

    func testInitImageBase64DecodesForImg2Img() throws {
        let p = try decode(#"{"prompt":"x","init_image_base64":"aGk=","image_strength":0.55}"#)
        XCTAssertNotNil(p.initImageData, "init_image_base64 must survive .convertFromSnakeCase")
        XCTAssertEqual(p.imageStrength, 0.55)
    }

    func testSourceDecodes() throws {
        let p = try decode(#"{"prompt":"x","source":"desktop"}"#)
        XCTAssertEqual(p.source, "desktop")
    }

    func testPlainRequestHasNoInpaintData() throws {
        let p = try decode(#"{"prompt":"x","width":1024,"height":1024}"#)
        XCTAssertNil(p.inpaintImageData)
        XCTAssertNil(p.maskData)
    }
}

// MARK: - WP-E4 (FDD-krea2-raw-recipe D25, AC-15a): the `sampler` key alias

extension GeneratePayloadDecodeTests {

  private func decodeE4(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  /// `sampler` is a decoded alias of the wire key `scheduler`. It can never be
  /// silently ignored: `{"sampler":"res_2s"}` resolves to `.res2s`, both keys
  /// with different values is a 400 (`mutuallyExclusive`), both equal succeeds.
  func testSamplerKeyAlias() throws {
    let aliased = try decodeE4(#"{"prompt":"x","sampler":"res_2s"}"#)
    XCTAssertEqual(aliased.scheduler, "res_2s", "sampler must land on the scheduler field, not be dropped")
    XCTAssertEqual(try aliased.validateRecipeNames().scheduler, .res2s)

    XCTAssertThrowsError(try decodeE4(#"{"prompt":"x","scheduler":"res_2s","sampler":"euler"}"#)) { error in
      guard case WarmServerError.mutuallyExclusive(let message) = error else {
        return XCTFail("expected WarmServerError.mutuallyExclusive, got \(error)")
      }
      XCTAssertTrue(message.contains("scheduler"), message)
      XCTAssertTrue(message.contains("sampler"), message)
      XCTAssertTrue(message.contains("res_2s") && message.contains("euler"), "message names both values: \(message)")
    }

    let agreeing = try decodeE4(#"{"prompt":"x","scheduler":"res_2s","sampler":"res_2s"}"#)
    XCTAssertEqual(agreeing.scheduler, "res_2s")
    XCTAssertEqual(try agreeing.validateRecipeNames().scheduler, .res2s)

    // The legacy key alone is byte-identical to today.
    let legacy = try decodeE4(#"{"prompt":"x","scheduler":"heun"}"#)
    XCTAssertEqual(legacy.scheduler, "heun")
  }
}
