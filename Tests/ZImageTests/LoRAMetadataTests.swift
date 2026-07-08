import XCTest
@testable import ZImage

final class LoRAMetadataTests: XCTestCase {
    private func loras(_ json: String) throws -> [[String: Any]] {
        let obj = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
        return (obj["loras"] as? [[String: Any]]) ?? []
    }

    func testGenerationEmbedsLoraStack() throws {
        let m = QwenImageIO.ImageMetadata.generation(
            prompt: "x",
            loras: [
                .local("/models/Anneliese_Zbase3.safetensors", scale: 0.8),
                .local("/models/deedee_amateur_photography_zimage_base_and_turbo_v1.safetensors", scale: 0.4),
                .local("/models/Z-Breast-Slider.safetensors", scale: -3),
            ])
        let arr = try loras(try XCTUnwrap(m.parametersJSON))
        XCTAssertEqual(arr.count, 3)
        XCTAssertEqual(arr[0]["name"] as? String, "Anneliese_Zbase3")          // extension stripped
        XCTAssertEqual(arr[0]["scale"] as? Double, 0.8)
        XCTAssertEqual(arr[2]["name"] as? String, "Z-Breast-Slider")
        XCTAssertEqual(arr[2]["scale"] as? Double, -3)
    }

    func testGenerationOmitsEmptyLoraStack() throws {
        let m = QwenImageIO.ImageMetadata.generation(prompt: "x", loras: [])
        let json = try XCTUnwrap(m.parametersJSON)
        XCTAssertFalse(json.contains("loras"), json)
    }
}
