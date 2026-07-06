import XCTest
@testable import ComfyBoxDesktop

final class NSFWGateTests: XCTestCase {

    func testContentRating() {
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "banana"))
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "Avocado"))
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "explicit"))
        XCTAssertFalse(ContentRating.isNSFW(contentMode: "apple"))
        XCTAssertFalse(ContentRating.isNSFW(contentMode: nil))
        XCTAssertFalse(ContentRating.isNSFW(contentMode: ""))
    }

    func testHashIsStableAndSalted() {
        let a = NSFWGate.hash("hunter2")
        XCTAssertEqual(a, NSFWGate.hash("hunter2"), "same input → same hash")
        XCTAssertNotEqual(a, NSFWGate.hash("hunter3"))
        XCTAssertNotEqual(a, "hunter2", "never the plaintext")
        XCTAssertEqual(a.count, 64, "SHA-256 hex")
    }

    func testVerifyWithNoPasswordSetAllows() {
        // With nothing configured, verify() should not gate (returns true).
        NSFWGate.setPassword(nil)
        XCTAssertFalse(NSFWGate.isConfigured)
        XCTAssertTrue(NSFWGate.verify("anything"))
    }

    func testSetVerifyClear() {
        NSFWGate.setPassword("s3cret")
        XCTAssertTrue(NSFWGate.isConfigured)
        XCTAssertTrue(NSFWGate.verify("s3cret"))
        XCTAssertFalse(NSFWGate.verify("wrong"))
        NSFWGate.setPassword(nil)   // cleanup
        XCTAssertFalse(NSFWGate.isConfigured)
    }

    func testFilterModes() {
        XCTAssertEqual(NSFWFilterMode.allCases.count, 3)
        XCTAssertEqual(NSFWFilterMode.blur.rawValue, "Blur NSFW")
    }
}
