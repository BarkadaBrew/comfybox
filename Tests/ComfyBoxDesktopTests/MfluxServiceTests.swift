import XCTest
@testable import ComfyBoxDesktop

final class MfluxServiceTests: XCTestCase {

    func testGenerateArgsCore() {
        let args = MfluxService.generateArgs(.init(
            model: "dev", prompt: "a fox", width: 1024, height: 1536, steps: 25,
            output: "/tmp/out.png"))
        XCTAssertEqual(args, [
            "--model", "dev", "--prompt", "a fox",
            "--width", "1024", "--height", "1536",
            "--steps", "25", "--output", "/tmp/out.png",
        ])
    }

    func testGenerateArgsFull() {
        let args = MfluxService.generateArgs(.init(
            model: "z-image-turbo", prompt: "p", negativePrompt: "blurry",
            width: 1280, height: 1280, steps: 9, guidance: 3.5, seed: 42, quantize: 8,
            loraPaths: ["/a.safetensors", "/b.safetensors"], loraScales: [0.8, 0.5],
            imagePath: "/ref.png", imageStrength: 0.6, lowRam: true, output: "/tmp/o.png"))
        XCTAssertTrue(args.contains("--negative-prompt"))
        XCTAssertEqual(valueAfter("--guidance", in: args), "3.5")
        XCTAssertEqual(valueAfter("--seed", in: args), "42")
        XCTAssertEqual(valueAfter("--quantize", in: args), "8")
        XCTAssertEqual(valueAfter("--image-path", in: args), "/ref.png")
        XCTAssertEqual(valueAfter("--image-strength", in: args), "0.6")
        XCTAssertTrue(args.contains("--low-ram"))
        // LoRA paths and scales follow their flags (variadic).
        let lpIdx = args.firstIndex(of: "--lora-paths")!
        XCTAssertEqual(args[lpIdx + 1], "/a.safetensors")
        XCTAssertEqual(args[lpIdx + 2], "/b.safetensors")
    }

    func testGenerateArgsLoraStyleForIdentity() {
        let args = MfluxService.generateArgs(.init(
            model: "dev-krea", prompt: "portrait", imagePath: "/face.png",
            imageStrength: 0.65, loraStyle: "identity", output: "/o.png"))
        XCTAssertEqual(valueAfter("--lora-style", in: args), "identity")
        XCTAssertEqual(valueAfter("--image-path", in: args), "/face.png")
        XCTAssertEqual(valueAfter("--image-strength", in: args), "0.65")
    }

    func testGenerateArgsOmitsOptional() {
        let args = MfluxService.generateArgs(.init(model: "schnell", prompt: "x", output: "/o.png"))
        XCTAssertFalse(args.contains("--quantize"))
        XCTAssertFalse(args.contains("--seed"))
        XCTAssertFalse(args.contains("--lora-paths"))
        XCTAssertFalse(args.contains("--image-path"))
        XCTAssertFalse(args.contains("--low-ram"))
    }

    func testTrainArgsConfigVsResume() {
        let config = MfluxService.trainArgs(.init(model: "dev", configPath: "/t.json"))
        XCTAssertEqual(valueAfter("--config", in: config), "/t.json")
        XCTAssertEqual(valueAfter("--model", in: config), "dev")

        // Resume takes precedence over config.
        let resume = MfluxService.trainArgs(.init(configPath: "/t.json", resumePath: "/ckpt.zip"))
        XCTAssertEqual(valueAfter("--resume", in: resume), "/ckpt.zip")
        XCTAssertFalse(resume.contains("--config"))
    }

    func testTrainArgsDryRun() {
        let args = MfluxService.trainArgs(.init(configPath: "/t.json", dryRun: true))
        XCTAssertTrue(args.contains("--dry-run"))
    }

    func testSaveArgsQuantizeAndBake() {
        let args = MfluxService.saveArgs(.init(
            model: "z-image-turbo", path: "/out/model", baseModel: "z-image-turbo",
            quantize: 4, loraPaths: ["/l.safetensors"], loraScales: [0.6]))
        XCTAssertEqual(valueAfter("--model", in: args), "z-image-turbo")
        XCTAssertEqual(valueAfter("--path", in: args), "/out/model")
        XCTAssertEqual(valueAfter("--base-model", in: args), "z-image-turbo")
        XCTAssertEqual(valueAfter("--quantize", in: args), "4")
        XCTAssertEqual(valueAfter("--lora-paths", in: args), "/l.safetensors")
    }

    func testParseVersion() {
        let blob = "Name: mflux\nVersion: 0.16.4\nSummary: FLUX on MLX\n"
        XCTAssertEqual(MfluxService.parseVersion(fromPipShow: blob), "0.16.4")
        XCTAssertNil(MfluxService.parseVersion(fromPipShow: "no version here"))
    }

    func testModelVariants() {
        XCTAssertEqual(MfluxModel.zImageTurbo.rawValue, "z-image-turbo")
        XCTAssertTrue(MfluxModel.allCases.contains(.qwen))
        XCTAssertTrue(MfluxModel.allCases.contains(.flux2Klein9b))
    }

    private func valueAfter(_ flag: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
}
