import XCTest
@testable import ZImage

final class ImageMetadataContentModeTests: XCTestCase {
    func testGenerationEmbedsContentMode() throws {
        let m = QwenImageIO.ImageMetadata.generation(
            prompt: "a cat", seed: 7, steps: 9, guidance: 0, width: 1024, height: 1024,
            model: "cyberrealisticZImage_v50", contentMode: "avocado")
        let json = try XCTUnwrap(m.parametersJSON)
        XCTAssertTrue(json.contains("\"content_mode\""))
        XCTAssertTrue(json.contains("avocado"))
    }

    func testGenerationOmitsEmptyContentMode() throws {
        let m = QwenImageIO.ImageMetadata.generation(prompt: "a cat", contentMode: nil)
        let json = try XCTUnwrap(m.parametersJSON)
        XCTAssertFalse(json.contains("content_mode"))
    }

    func testGenerationOmitsEmptyStringContentMode() throws {
        let m = QwenImageIO.ImageMetadata.generation(prompt: "x", contentMode: "")
        let json = try XCTUnwrap(m.parametersJSON)
        XCTAssertFalse(json.contains("content_mode"))
    }
}
