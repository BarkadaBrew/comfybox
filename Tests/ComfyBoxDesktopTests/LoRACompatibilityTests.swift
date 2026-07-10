import XCTest
@testable import ComfyBoxDesktop

final class LoRACompatibilityTests: XCTestCase {

    func testFamilyNormalization() {
        XCTAssertEqual(LoRACompatibility.family(from: "ZImageTurbo"), "z-image")
        XCTAssertEqual(LoRACompatibility.family(from: "z-image-turbo"), "z-image")
        XCTAssertEqual(LoRACompatibility.family(from: "zeta-chroma"), "z-image")
        XCTAssertEqual(LoRACompatibility.family(from: "Krea 2"), "krea2")
        // Real server model id formats (hyphenated) — regression: the family
        // detector originally only recognized space/no-separator variants,
        // so "Only krea2 LoRAs" never matched the actual active model string.
        XCTAssertEqual(LoRACompatibility.family(from: "krea-2-turbo"), "krea2")
        XCTAssertEqual(LoRACompatibility.family(from: "krea/Krea-2-Turbo"), "krea2")
        XCTAssertEqual(LoRACompatibility.family(from: "krea2-turbo-q8"), "krea2")
        XCTAssertEqual(LoRACompatibility.family(from: "Flux.1 Krea"), "flux")   // Krea 1 = Flux-based
        XCTAssertEqual(LoRACompatibility.family(from: "krea-dev"), "flux")
        XCTAssertEqual(LoRACompatibility.family(from: "Flux.1 D"), "flux")
        XCTAssertEqual(LoRACompatibility.family(from: "flux2-klein-9b"), "flux2")
        XCTAssertEqual(LoRACompatibility.family(from: "Qwen"), "qwen")
        XCTAssertEqual(LoRACompatibility.family(from: "Pony"), "sdxl")
        XCTAssertEqual(LoRACompatibility.family(from: "SD 1.5"), "sd15")
        XCTAssertEqual(LoRACompatibility.family(from: "unknown"), "")
    }

    func testKrea2IsDistinctFromFlux() {
        // The whole point: a Krea 2 LoRA must NOT read as Flux-compatible.
        XCTAssertNotEqual(LoRACompatibility.family(from: "Krea 2"),
                          LoRACompatibility.family(from: "Flux.1 D"))
    }

    func testCompatibleMatch() {
        XCTAssertEqual(
            LoRACompatibility.status(loraCompatibility: "z-image", modelIdentifier: "ZImageTurbo"),
            .compatible)
    }

    func testIncompatibleMismatch() {
        let status = LoRACompatibility.status(loraCompatibility: "Krea 2", modelIdentifier: "z-image-turbo")
        if case .incompatible(let l, let m) = status {
            XCTAssertEqual(l, "krea2"); XCTAssertEqual(m, "z-image")
        } else { XCTFail("expected incompatible, got \(status)") }
    }

    func testUnknownWhenEitherSideUnknown() {
        XCTAssertEqual(LoRACompatibility.status(loraCompatibility: "unknown", modelIdentifier: "ZImageTurbo"), .unknown)
        XCTAssertEqual(LoRACompatibility.status(loraCompatibility: "z-image", modelIdentifier: nil), .unknown)
    }

    func testLabels() {
        XCTAssertEqual(LoRACompatibility.label(for: "krea2"), "Krea 2")
        XCTAssertEqual(LoRACompatibility.label(for: "z-image"), "Z-Image")
        XCTAssertEqual(LoRACompatibility.label(for: ""), "?")
    }
}
