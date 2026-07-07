import XCTest
@testable import ComfyBoxDesktop

final class SidecarServiceTests: XCTestCase {

    func testEmbedArgsMapsStandardFields() {
        let m = SidecarService.Metadata(description: "a red barn",
                                        keywords: ["barn", "rural"],
                                        parametersJSON: #"{"seed":7}"#)
        let a = SidecarService.embedArgs(m, path: "/img.png")
        XCTAssertTrue(a.contains("-overwrite_original"))
        XCTAssertTrue(a.contains("-EXIF:ImageDescription=a red barn"))
        XCTAssertTrue(a.contains("-XMP-dc:Description=a red barn"))
        XCTAssertTrue(a.contains("-IPTC:Caption-Abstract=a red barn"))   // Finder Description
        XCTAssertTrue(a.contains("-IPTC:Keywords=barn"))                 // Finder Keywords
        XCTAssertTrue(a.contains("-XMP-dc:Subject=rural"))
        XCTAssertTrue(a.contains(#"-EXIF:UserComment={"seed":7}"#))
        XCTAssertEqual(a.last, "/img.png")
    }

    func testKeywordsDedupeAndLowercase() {
        let k = SidecarService.keywords(tags: ["Barn", "barn", " Sky "], character: "Anneliese", contentMode: "apple")
        XCTAssertEqual(k, ["barn", "sky", "anneliese", "apple"])
    }

    func testKeywordsHandlesEmpty() {
        XCTAssertEqual(SidecarService.keywords(tags: ["", "  "], character: nil, contentMode: nil), [])
    }

    func testReadArgs() {
        let a = SidecarService.readArgs(path: "/img.png")
        XCTAssertEqual(a.first, "-j")
        XCTAssertEqual(a.last, "/img.png")
    }
}
