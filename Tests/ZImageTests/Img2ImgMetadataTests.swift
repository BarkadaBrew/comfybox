import XCTest
@testable import ZImage

final class Img2ImgMetadataTests: XCTestCase {
    /// Write a tiny valid PNG to a temp path so makeImg2ImgPipelineRequest's
    /// file-existence + dimension checks pass.
    private func makeTempPNG() throws -> String {
        let path = NSTemporaryDirectory() + "img2img-test-\(UUID().uuidString).png"
        // 1x1 white PNG.
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        try Data(base64Encoded: b64)!.write(to: URL(fileURLWithPath: path))
        return path
    }

    func testImg2ImgPipelineRequestCarriesContentModeAndSource() throws {
        let src = try makeTempPNG()
        defer { try? FileManager.default.removeItem(atPath: src) }
        let req = Img2ImgRequest(
            prompt: "a cat", sourceImagePath: src,
            contentMode: "banana", source: "desktop")
        let pipeline = try ZImagePipeline.makeImg2ImgPipelineRequestForTesting(req)
        let json = try XCTUnwrap(pipeline.embeddedMetadata().parametersJSON)
        XCTAssertTrue(json.contains("\"content_mode\":\"banana\""), json)
        XCTAssertTrue(json.contains("\"source\":\"desktop\""), json)
    }

    func testImg2ImgPipelineRequestOmitsNilFields() throws {
        let src = try makeTempPNG()
        defer { try? FileManager.default.removeItem(atPath: src) }
        let req = Img2ImgRequest(prompt: "a cat", sourceImagePath: src)
        let pipeline = try ZImagePipeline.makeImg2ImgPipelineRequestForTesting(req)
        let json = try XCTUnwrap(pipeline.embeddedMetadata().parametersJSON)
        XCTAssertFalse(json.contains("content_mode"), json)
        XCTAssertFalse(json.contains("\"source\""), json)
    }
}
