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
