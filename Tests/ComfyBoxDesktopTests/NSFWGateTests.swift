import XCTest
@testable import ComfyBoxDesktop
import ComfyBoxCatalog

final class NSFWGateTests: XCTestCase {

    func testContentRating() {
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "banana"))
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "Avocado"))
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "explicit"))
        XCTAssertFalse(ContentRating.isNSFW(contentMode: "apple"))
        XCTAssertFalse(ContentRating.isNSFW(contentMode: nil))
        XCTAssertFalse(ContentRating.isNSFW(contentMode: ""))
    }

    /// The gate's vocabulary must not be a second, independently-maintained copy
    /// of the catalog's. It was, and the two copies disagreed about the ONLY case
    /// that matters — an unrecognised string. The catalog withholds it (tierRank
    /// fails closed HIGH); a `Set.contains` membership test showed it unblurred.
    /// Every spelling the catalog knows must get the same verdict from both ends.
    func testGateAgreesWithTheCatalogOnEverySpelling() {
        for spelling in CATALOG_TIER_SPELLINGS {
            XCTAssertEqual(
                ContentRating.isNSFW(contentMode: spelling),
                isWithheld(tier: spelling, ceiling: CATALOG_STRICTEST_CEILING),
                "gate and catalog disagree about '\(spelling)'")
        }
        // The derived set is exactly the above-SFW rungs — no more, no less.
        XCTAssertEqual(ContentRating.nsfwModes,
                       ["banana", "avocado", "nsfw", "explicit", "suggestive"])
        XCTAssertFalse(ContentRating.nsfwModes.contains("apple"),
                       "apple collapses to neutral and is SFW")
        XCTAssertFalse(ContentRating.nsfwModes.contains("neutral"))
    }

    /// FAIL CLOSED on unknown vocabulary. A content mode the ladder does not
    /// recognise is withheld by the catalog and by the gallery server, so the
    /// desktop must blur/hide it too — otherwise a tier string this build has
    /// never heard of (a newer daemon's vocabulary, a typo, a hand-edited
    /// sidecar) renders unblurred in a gate whose whole job is not to do that.
    func testUnknownContentModeIsNSFW() {
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "hardcore"))
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "durian"))
        XCTAssertTrue(ContentRating.isNSFW(contentMode: "  Hardcore  "),
                      "case and surrounding space must not open the gate")
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
