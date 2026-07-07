import XCTest
@testable import ZImage

/// P5: video routing must honor explicit backend/model intent so an on-device
/// "ltx" request never silently renders on paid Replicate cloud.
final class VideoBackendIntentTests: XCTestCase {

    private func req(backend: String? = nil, model: String? = nil) -> VideoGenerateRequest {
        VideoGenerateRequest(prompt: "x", backend: backend, model: model)
    }

    func testExplicitLocal() {
        XCTAssertEqual(req(backend: "local").backendIntent, .local)
        XCTAssertEqual(req(backend: "ltx").backendIntent, .local)
        XCTAssertEqual(req(model: "ltx").backendIntent, .local)
        XCTAssertEqual(req(model: "LTX-2").backendIntent, .local)
    }

    func testExplicitCloud() {
        XCTAssertEqual(req(backend: "replicate").backendIntent, .cloud)
        XCTAssertEqual(req(backend: "cloud").backendIntent, .cloud)
    }

    func testUnspecified() {
        XCTAssertEqual(req().backendIntent, .unspecified)
        XCTAssertEqual(req(model: "something-else").backendIntent, .unspecified)
    }
}
