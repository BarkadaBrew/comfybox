import Foundation
import XCTest
@testable import ZImage

final class ImageMetadataContentModeTests: XCTestCase {
    func testLTXEngineAndKindAreEmbeddedForRecipeHandoff() throws {
        let metadata = QwenImageIO.ImageMetadata.generation(
            prompt: "a cat", model: "LTX-2.3", engine: "ltx2", kind: "t2i")
        let json = try XCTUnwrap(metadata.parametersJSON)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(object["engine"] as? String, "ltx2")
        XCTAssertEqual(object["kind"] as? String, "t2i")
    }

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

/// The preset the caller named belongs in the metadata, not only in the
/// filename.
///
/// `ComfyBoxOutputNaming.defaultFilename(presetId:)` has always put the preset
/// in the NAME — `comfybox-kroma-v0.3-base-krea-kira-apple-…` — while every
/// writer dropped it from the metadata, so the catalog's `preset` column was
/// empty for every asset ever rendered. `MetadataReader` already reads a
/// `preset` key; nothing was writing one.
final class ImageMetadataPresetTests: XCTestCase {
    func testGenerationEmbedsPreset() throws {
        let m = QwenImageIO.ImageMetadata.generation(
            prompt: "a cat", model: "kroma-v0.3-base", preset: "krea-kira")
        let json = try XCTUnwrap(m.parametersJSON)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(object["preset"] as? String, "krea-kira")
    }

    /// The live store keys presets by UUID, so the value is often not a slug.
    func testGenerationEmbedsAPresetUUIDVerbatim() throws {
        let id = "5F2973F3-DB57-48F7-991F-65C5A843C54C"
        let m = QwenImageIO.ImageMetadata.generation(prompt: "x", preset: id)
        let json = try XCTUnwrap(m.parametersJSON)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(object["preset"] as? String, id)
    }

    func testGenerationOmitsAnAbsentPreset() throws {
        let json = try XCTUnwrap(
            QwenImageIO.ImageMetadata.generation(prompt: "x", preset: nil).parametersJSON)
        XCTAssertFalse(json.contains("preset"))
    }

    func testGenerationOmitsAnEmptyPreset() throws {
        let json = try XCTUnwrap(
            QwenImageIO.ImageMetadata.generation(prompt: "x", preset: "").parametersJSON)
        XCTAssertFalse(json.contains("preset"))
    }

    /// A render with no preset must produce byte-identical metadata to before
    /// the field existed — the same guarantee #399 made for `style`.
    func testAPresetlessRenderIsUnchanged() throws {
        let withParam = QwenImageIO.ImageMetadata.generation(
            prompt: "a cat", seed: 7, steps: 9, model: "m", preset: nil)
        let without = QwenImageIO.ImageMetadata.generation(
            prompt: "a cat", seed: 7, steps: 9, model: "m")
        XCTAssertEqual(withParam.parametersJSON, without.parametersJSON)
    }
}
