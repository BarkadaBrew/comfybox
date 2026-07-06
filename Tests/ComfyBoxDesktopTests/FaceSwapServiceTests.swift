import XCTest
@testable import ComfyBoxDesktop

final class FaceSwapServiceTests: XCTestCase {

    func testSwapArgsSingleFace() {
        let args = FaceSwapService.swapArgs(
            script: "/p/swap.py", source: "/s.png", target: "/t.png",
            output: "/o.png", allFaces: false)
        XCTAssertEqual(args, ["/p/swap.py", "/s.png", "/t.png", "/o.png"])
    }

    func testSwapArgsAllFaces() {
        let args = FaceSwapService.swapArgs(
            script: "/p/swap.py", source: "/s.png", target: "/t.png",
            output: "/o.png", allFaces: true)
        XCTAssertEqual(args.last, "--all")
        XCTAssertEqual(args.count, 5)
    }

    @MainActor
    func testPaths() {
        let s = FaceSwapService(projectDirectory: "/faceswap")
        XCTAssertEqual(s.pythonPath, "/faceswap/.venv/bin/python")
        XCTAssertEqual(s.scriptPath, "/faceswap/swap.py")
        XCTAssertEqual(s.modelPath, "/faceswap/models/inswapper_128.onnx")
    }
}
