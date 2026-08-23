import XCTest
@testable import ComfyBoxDesktop

final class ImageRecipeTests: XCTestCase {
    func testFromParamsReconstructsRecipe() throws {
        let params: [String: Any] = [
            "prompt": "a cat", "negative_prompt": "blurry",
            "seed": 12345, "steps": 9, "guidance": 3.5,
            "width": 896, "height": 1152,
            "model": "cyberrealisticZImage_v50", "content_mode": "banana",
            "loras": [
                ["name": "Anneliese_Zbase3", "scale": 0.8],
                ["name": "kroma-v0.3", "scale": 0.55, "role": "kroma"],
                ["name": "Z-Breast-Slider", "scale": -3],
            ],
        ]
        let r = try XCTUnwrap(ImageRecipe.from(params: params))
        XCTAssertEqual(r.preset.promptTemplate, "a cat")
        XCTAssertEqual(r.preset.negativePrompt, "blurry")
        XCTAssertEqual(r.preset.seed, 12345)
        XCTAssertEqual(r.preset.steps, 9)
        XCTAssertEqual(r.preset.guidance, 3.5)
        XCTAssertEqual(r.preset.width, 896)
        XCTAssertEqual(r.preset.height, 1152)
        XCTAssertEqual(r.preset.modelId, "cyberrealisticZImage_v50")
        XCTAssertEqual(r.contentMode, .banana)
        XCTAssertEqual(r.preset.loras.map(\.filename),
                       ["Anneliese_Zbase3.safetensors", "Z-Breast-Slider.safetensors"])
        XCTAssertEqual(r.preset.loras.map(\.scale), [0.8, -3])
        XCTAssertEqual(r.preset.kroma, PresetKroma(strength: 0.55, file: "kroma-v0.3.safetensors"))
    }

    func testFromParamsEmptyReturnsNil() {
        XCTAssertNil(ImageRecipe.from(params: [:]))
    }

    func testFromParamsNoLorasNoMode() throws {
        let r = try XCTUnwrap(ImageRecipe.from(params: ["prompt": "x", "seed": 7]))
        XCTAssertTrue(r.preset.loras.isEmpty)
        XCTAssertNil(r.contentMode)
    }
}
