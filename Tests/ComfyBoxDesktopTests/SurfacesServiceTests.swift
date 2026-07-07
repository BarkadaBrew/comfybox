import XCTest
@testable import ComfyBoxDesktop

final class SurfacesServiceTests: XCTestCase {

    func testBridgeReachable() {
        let s = SurfacesService.comfyBridgeSurface(port: 7870, reachable: true)
        XCTAssertEqual(s.health, .ok)
        XCTAssertEqual(s.endpoint, "http://127.0.0.1:7870")
    }

    func testBridgeUnreachable() {
        let s = SurfacesService.comfyBridgeSurface(port: 7870, reachable: false)
        XCTAssertEqual(s.health, .off)
        XCTAssertNotNil(s.hint)
    }

    func testConfigFlagParsing() throws {
        let tmp = NSTemporaryDirectory() + "kritarc-test-\(UUID().uuidString)"
        try "[python]\nenable_ai_diffusion=true\nenable_kritamcp=false\n".write(toFile: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        XCTAssertTrue(SurfacesService.configFlag("enable_ai_diffusion", in: tmp))
        XCTAssertFalse(SurfacesService.configFlag("enable_kritamcp", in: tmp))
        XCTAssertFalse(SurfacesService.configFlag("enable_missing", in: tmp))
    }

    func testPluginVersionExtraction() throws {
        let tmp = NSTemporaryDirectory() + "init-test-\(UUID().uuidString).py"
        try "\"\"\"doc\"\"\"\n\n__version__ = \"1.51.0\"\n".write(toFile: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        XCTAssertEqual(SurfacesService.pluginVersion(at: tmp), "1.51.0")
    }
}
