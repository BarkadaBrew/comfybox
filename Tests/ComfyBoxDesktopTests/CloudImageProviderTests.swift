import XCTest
@testable import ComfyBoxDesktop

final class CloudImageProviderTests: XCTestCase {

    func testDefaultModels() {
        XCTAssertEqual(CloudProvider.replicate.defaultModel, "black-forest-labs/flux-schnell")
        XCTAssertEqual(CloudProvider.fal.defaultModel, "fal-ai/flux/schnell")
        XCTAssertTrue(CloudProvider.local.defaultModel.isEmpty)
        XCTAssertFalse(CloudProvider.local.needsKey)
        XCTAssertTrue(CloudProvider.replicate.needsKey)
    }

    func testClientFallsBackToDefaultModel() {
        let c = CloudImageClient(provider: .fal, model: "", apiKey: "k")
        XCTAssertEqual(c.model, "fal-ai/flux/schnell")
    }

    func testAspectRatioMapping() {
        XCTAssertEqual(CloudImageClient.aspectRatio(width: 1024, height: 1024), "1:1")
        XCTAssertEqual(CloudImageClient.aspectRatio(width: 1024, height: 576), "16:9")
        XCTAssertEqual(CloudImageClient.aspectRatio(width: 576, height: 1024), "9:16")
        XCTAssertEqual(CloudImageClient.aspectRatio(width: 1024, height: 1536), "2:3")
        XCTAssertEqual(CloudImageClient.aspectRatio(width: 0, height: 0), "1:1")
    }

    func testReplicateInputBody() {
        let p = CloudImageParams(prompt: "a cat", width: 1024, height: 576, steps: 8, seed: 42)
        let input = CloudImageClient.replicateInput(p)
        XCTAssertEqual(input["prompt"] as? String, "a cat")
        XCTAssertEqual(input["aspect_ratio"] as? String, "16:9")
        XCTAssertEqual(input["num_inference_steps"] as? Int, 4, "schnell caps steps at 4")
        XCTAssertEqual(input["seed"] as? Int, 42)
        XCTAssertEqual(input["output_format"] as? String, "png")
    }

    func testReplicateInputOmitsSeedWhenRandom() {
        let input = CloudImageClient.replicateInput(
            CloudImageParams(prompt: "x", width: 512, height: 512, steps: 4, seed: 0))
        XCTAssertNil(input["seed"])
    }

    func testFalInputBody() {
        let p = CloudImageParams(prompt: "a dog", width: 768, height: 1024, steps: 12, seed: 7)
        let input = CloudImageClient.falInput(p)
        XCTAssertEqual(input["prompt"] as? String, "a dog")
        let size = input["image_size"] as? [String: Int]
        XCTAssertEqual(size?["width"], 768)
        XCTAssertEqual(size?["height"], 1024)
        XCTAssertEqual(input["num_inference_steps"] as? Int, 12)
        XCTAssertEqual(input["seed"] as? Int, 7)
    }

    func testParseReplicateSucceededSingleAndArray() {
        let single = #"{"id":"abc","status":"succeeded","output":"https://x/y.png"}"#.data(using: .utf8)!
        let r1 = CloudImageClient.parseReplicate(single)
        XCTAssertEqual(r1.status, "succeeded")
        XCTAssertEqual(r1.id, "abc")
        XCTAssertEqual(r1.urls, ["https://x/y.png"])

        let array = #"{"id":"z","status":"succeeded","output":["https://a/1.png","https://a/2.png"]}"#.data(using: .utf8)!
        XCTAssertEqual(CloudImageClient.parseReplicate(array).urls.count, 2)
    }

    func testParseReplicateProcessing() {
        let processing = #"{"id":"p","status":"processing","output":null}"#.data(using: .utf8)!
        let r = CloudImageClient.parseReplicate(processing)
        XCTAssertEqual(r.status, "processing")
        XCTAssertTrue(r.urls.isEmpty)
    }

    func testParseFalImages() {
        let json = #"{"images":[{"url":"https://fal/out.png","width":1024,"height":1024}]}"#.data(using: .utf8)!
        XCTAssertEqual(CloudImageClient.parseFal(json), ["https://fal/out.png"])
    }

    func testParseFalEmptyOnGarbage() {
        XCTAssertTrue(CloudImageClient.parseFal(Data("not json".utf8)).isEmpty)
    }
}
