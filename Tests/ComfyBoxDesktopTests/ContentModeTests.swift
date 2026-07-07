import XCTest
@testable import ComfyBoxDesktop

final class ContentModeTests: XCTestCase {
    func testRawValuesMatchServerModeStrings() {
        XCTAssertEqual(ContentMode.neutral.rawValue, "neutral")
        XCTAssertEqual(ContentMode.banana.rawValue, "banana")
        XCTAssertEqual(ContentMode.avocado.rawValue, "avocado")
    }

    func testAllCasesInDisplayOrder() {
        XCTAssertEqual(ContentMode.allCases, [.neutral, .banana, .avocado])
    }

    func testLabelsCarryEmoji() {
        XCTAssertTrue(ContentMode.neutral.label.contains("Neutral"))
        XCTAssertTrue(ContentMode.banana.label.contains("🍌"))
        XCTAssertTrue(ContentMode.avocado.label.contains("🥑"))
    }
}
