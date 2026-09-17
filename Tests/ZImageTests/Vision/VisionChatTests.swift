import XCTest

@testable import ZImage

/// FDD-mlx-serve-tuning WP4 (§0.5, D7): the local vision call must resolve its
/// endpoint from the ComfyBox config (env override first, never a hard-coded
/// LM Studio fallback) and must give a thinking model enough budget to answer.
final class VisionChatTests: XCTestCase {

  private func configJSON(vision: [String: String]?, captioning: [String: String]? = nil) -> Data {
    var providers: [String: Any] = [:]
    if let vision { providers["vision"] = vision }
    if let captioning { providers["captioning"] = captioning }
    return try! JSONSerialization.data(withJSONObject: ["port": 7870, "providers": providers])
  }

  // MARK: - Endpoint resolution

  func testConfigVisionProviderIsUsed() {
    let cfg = configJSON(vision: ["baseUrl": "http://127.0.0.1:11234/v1/", "model": "Muse-Glimmer-30B-heretic-MLX-Q6"])
    let ep = VisionChat.resolveEndpoint(environment: [:], configJSON: cfg)
    XCTAssertEqual(ep?.baseURL, "http://127.0.0.1:11234/v1", "trailing slash trimmed")
    XCTAssertEqual(ep?.model, "Muse-Glimmer-30B-heretic-MLX-Q6")
    XCTAssertEqual(ep?.source, .config)
  }

  func testCaptioningIsTheFallbackProvider() {
    let cfg = configJSON(vision: nil, captioning: ["baseUrl": "http://h:1/v1", "model": "cap", "apiKey": "k"])
    let ep = VisionChat.resolveEndpoint(environment: [:], configJSON: cfg)
    XCTAssertEqual(ep?.model, "cap")
    XCTAssertEqual(ep?.apiKey, "k")
  }

  func testEnvironmentOverridesConfig() {
    let cfg = configJSON(vision: ["baseUrl": "http://127.0.0.1:11234/v1", "model": "glimmer"])
    let env = ["COMFYBOX_VISION_URL": "http://test:9/v1", "COMFYBOX_VISION_MODEL": "stub-vl"]
    let ep = VisionChat.resolveEndpoint(environment: env, configJSON: cfg)
    XCTAssertEqual(ep?.baseURL, "http://test:9/v1")
    XCTAssertEqual(ep?.model, "stub-vl")
    XCTAssertEqual(ep?.source, .environment)
  }

  func testNoProviderMeansNilNeverLMStudio() {
    XCTAssertNil(VisionChat.resolveEndpoint(environment: [:], configJSON: configJSON(vision: nil)))
    XCTAssertNil(VisionChat.resolveEndpoint(environment: [:], configJSON: nil), "unreadable config → nil, not :1234")
  }

  // MARK: - Request body

  func testBodyGivesThinkingModelsHeadroom() {
    let body = VisionChat.body(model: "m", prompt: "look", base64PNG: "AAAA")
    XCTAssertEqual(body["model"] as? String, "m")
    XCTAssertGreaterThanOrEqual(body["max_tokens"] as? Int ?? 0, 1200, "300 let Glimmer spend its whole budget reasoning")
    XCTAssertEqual(body["reasoning_budget"] as? Int, 256)
    let content = (body["messages"] as? [[String: Any]])?.first?["content"] as? [[String: Any]]
    let url = (content?.first { ($0["type"] as? String) == "image_url" }?["image_url"] as? [String: Any])?["url"] as? String
    XCTAssertEqual(url, "data:image/png;base64,AAAA")
  }

  // MARK: - Reply parsing

  private func reply(content: String?, reasoning: String? = nil) -> Data {
    var msg: [String: Any] = [:]
    if let content { msg["content"] = content }
    if let reasoning { msg["reasoning_content"] = reasoning }
    return try! JSONSerialization.data(withJSONObject: ["choices": [["message": msg]]])
  }

  func testParseText() {
    XCTAssertEqual(VisionChat.parseReply(reply(content: "  CLEAN \n")), .text("CLEAN"))
  }

  func testParseEmptyContentAfterReasoningIsNamed() {
    XCTAssertEqual(VisionChat.parseReply(reply(content: "", reasoning: String(repeating: "x", count: 948))), .emptyAfterReasoning(948))
  }

  func testParseMalformed() {
    XCTAssertEqual(VisionChat.parseReply(Data("not json".utf8)), .malformed)
    XCTAssertEqual(VisionChat.parseReply(reply(content: nil)), .malformed)
  }
}
