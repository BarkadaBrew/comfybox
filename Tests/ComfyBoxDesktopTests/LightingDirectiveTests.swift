import XCTest
@testable import ComfyBoxDesktop

final class LightingDirectiveTests: XCTestCase {

    func testEmptyByDefault() {
        let d = LightingDirective()
        XCTAssertTrue(d.isEmpty)
        XCTAssertEqual(d.phrase, "")
    }

    func testComposesInOrder() {
        let d = LightingDirective(direction: .left, quality: .soft, mood: .goldenHour)
        XCTAssertEqual(d.phrase, "lit from the left, soft light, warm golden hour light")
    }

    func testPartialSelection() {
        XCTAssertEqual(LightingDirective(direction: .back).phrase, "backlit")
        XCTAssertEqual(LightingDirective(mood: .lowKey).phrase, "low-key lighting")
    }

    func testAppendedToPrompt() {
        let d = LightingDirective(direction: .rim, mood: .studio)
        XCTAssertEqual(d.appended(to: "a portrait"), "a portrait, rim-lit, studio lighting")
        XCTAssertEqual(d.appended(to: "  "), "rim-lit, studio lighting")
    }

    func testAppendedEmptyIsNoop() {
        XCTAssertEqual(LightingDirective().appended(to: "a portrait"), "a portrait")
    }
}
