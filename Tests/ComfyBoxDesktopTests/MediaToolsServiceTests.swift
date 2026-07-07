import XCTest
@testable import ComfyBoxDesktop

final class MediaToolsServiceTests: XCTestCase {

    func testConvertArgsPlain() {
        let a = MediaToolsService.convertArgs(source: "/a.png", output: "/a.jpg", transform: .none, quality: 92)
        XCTAssertEqual(a, ["/a.png", "-quality", "92", "/a.jpg"])
    }

    func testConvertArgsWithTransform() {
        let a = MediaToolsService.convertArgs(source: "/a.png", output: "/a-grayscale.png", transform: .grayscale)
        XCTAssertEqual(a, ["/a.png", "-colorspace", "Gray", "/a-grayscale.png"])
    }

    func testVideoArgs() {
        let a = MediaToolsService.videoArgs(listPath: "/list.txt", fps: 12, output: "/out.mp4")
        XCTAssertEqual(a.first, "-y")
        XCTAssertTrue(a.contains("libx264"))
        XCTAssertTrue(a.contains("/list.txt"))
        XCTAssertEqual(a.last, "/out.mp4")
        XCTAssertTrue(a.contains("12"))  // fps
    }

    func testConcatListBodyRepeatsLastFrame() {
        let body = MediaToolsService.concatListBody(images: ["/1.png", "/2.png"], fps: 2)
        // two files + last repeated = 3 "file" lines
        XCTAssertEqual(body.components(separatedBy: "file '").count - 1, 3)
        XCTAssertTrue(body.contains("duration 0.5"))
    }

    func testConcatListEscapesQuotes() {
        let body = MediaToolsService.concatListBody(images: ["/wei'rd.png"], fps: 1)
        XCTAssertTrue(body.contains("'\\''"))
    }

    func testFormatExtensions() {
        XCTAssertEqual(MediaToolsService.ImageFormat.jpeg.ext, "jpg")
        XCTAssertEqual(MediaToolsService.ImageFormat.webp.ext, "webp")
    }
}
