import XCTest
@testable import ComfyBoxDesktop

final class VisionServiceTests: XCTestCase {

    func testRequestBodyShape() {
        let body = VisionService.requestBody(model: "qwen2-vl", base64PNG: "AAAA")
        XCTAssertEqual(body["model"] as? String, "qwen2-vl")
        let messages = body["messages"] as? [[String: Any]]
        let content = messages?.first?["content"] as? [[String: Any]]
        XCTAssertEqual(content?.count, 2)
        let imageBlock = content?.first(where: { ($0["type"] as? String) == "image_url" })
        let url = (imageBlock?["image_url"] as? [String: Any])?["url"] as? String
        XCTAssertEqual(url, "data:image/png;base64,AAAA")
    }

    func testParseCleanJSON() {
        let d = VisionService.parseDescription(from: #"{"caption":"A cat on a couch","tags":["Cat","couch","INDOOR"]}"#)
        XCTAssertEqual(d?.caption, "A cat on a couch")
        XCTAssertEqual(d?.tags, ["cat", "couch", "indoor"])   // lowercased
    }

    func testParseFencedOrPrefixed() {
        let reply = "Here you go:\n```json\n{\"caption\":\"Sunset\",\"tags\":[\"sky\",\"orange\"]}\n```"
        let d = VisionService.parseDescription(from: reply)
        XCTAssertEqual(d?.caption, "Sunset")
        XCTAssertEqual(d?.tags, ["sky", "orange"])
    }

    func testParseFallbackToPlainCaption() {
        let d = VisionService.parseDescription(from: "A lone tree on a hill.")
        XCTAssertEqual(d?.caption, "A lone tree on a hill.")
        XCTAssertEqual(d?.tags, [])
    }

    func testParseEmptyIsNil() {
        XCTAssertNil(VisionService.parseDescription(from: "   "))
    }

    func testExtractJSONObjectBalanced() {
        XCTAssertEqual(VisionService.extractJSONObject(from: "x {\"a\":{\"b\":1}} y"), "{\"a\":{\"b\":1}}")
        XCTAssertNil(VisionService.extractJSONObject(from: "no braces"))
    }
}
