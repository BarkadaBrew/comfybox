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
        // Todd 2026-09-04: kroma is a regular LoRA — the `role: "kroma"` wire
        // entry stays in `loras[]`, in declared order, like any other role.
        XCTAssertEqual(r.preset.loras.map(\.filename), [
            "Anneliese_Zbase3.safetensors", "kroma-v0.3.safetensors", "Z-Breast-Slider.safetensors",
        ])
        XCTAssertEqual(r.preset.loras.map(\.scale), [0.8, 0.55, -3])
        XCTAssertEqual(r.preset.loras[1].role, "kroma")
    }

    func testFromParamsEmptyReturnsNil() {
        XCTAssertNil(ImageRecipe.from(params: [:]))
    }

    func testFromParamsNoLorasNoMode() throws {
        let r = try XCTUnwrap(ImageRecipe.from(params: ["prompt": "x", "seed": 7]))
        XCTAssertTrue(r.preset.loras.isEmpty)
        XCTAssertNil(r.contentMode)
    }

    func testLTXMetadataRestoresNativeEngineWithoutTreatingTransformerAsPoolModel() throws {
        let r = try XCTUnwrap(ImageRecipe.from(params: [
            "prompt": "a landscape",
            "engine": "ltx2",
            "kind": "t2i",
            "model": "LTX-2.3",
            "width": 1280,
            "height": 704,
            "steps": 8,
            "guidance": 1,
        ]))
        XCTAssertEqual(r.preset.engine, "ltx2")
        XCTAssertNil(r.preset.modelId)
        XCTAssertEqual(r.preset.width, 1280)
        XCTAssertEqual(r.preset.height, 704)
    }

    func testGenericT2IKindDoesNotSelectLTXWithoutEngineOrLTXModel() throws {
        let r = try XCTUnwrap(ImageRecipe.from(params: [
            "prompt": "a landscape",
            "kind": "t2i",
            "model": "krea-2-turbo",
        ]))
        XCTAssertNil(r.preset.engine)
        XCTAssertEqual(r.preset.modelId, "krea-2-turbo")
    }
}
