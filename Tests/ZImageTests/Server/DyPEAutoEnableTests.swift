import XCTest
@testable import ZImage

/// Covers which DyPE tier a request ends up with when the caller sends no
/// explicit `dype`.
///
/// This path matters more than it looks: the callers that most want DyPE (the
/// Kira daemon's "HQ 2K rerender", which upsizes the longest edge to 2048)
/// don't forward a `dype` argument at all, so the auto-enable branch is what
/// actually decides their tier. `.ntk` is the real ceiling — `.yarn` is an
/// unimplemented stub that warns and falls back to NTK.
final class DyPEAutoEnableTests: XCTestCase {

    private func configuration() -> WarmServerConfiguration {
        WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
    }

    // MARK: - The shared resolver

    func testResolverAutoEnablesNTKAboveBaseResolution() {
        let payload = GeneratePayload(prompt: "x")
        let config = payload.resolvedDyPEConfig(width: 2048, height: 2048)

        XCTAssertTrue(config.enabled)
        XCTAssertEqual(config.method, .ntk)
    }

    func testResolverLeavesBaseResolutionDisabled() {
        let payload = GeneratePayload(prompt: "x")
        XCTAssertFalse(payload.resolvedDyPEConfig(width: 1024, height: 1024).enabled)
    }

    func testResolverHonoursExplicitNone() {
        let payload = GeneratePayload(prompt: "x", dype: "none")
        XCTAssertFalse(payload.resolvedDyPEConfig(width: 2048, height: 2048).enabled,
                       "'none' must opt out even above base resolution")
    }

    func testResolverHonoursExplicitOff() {
        let payload = GeneratePayload(prompt: "x", dype: "off")
        XCTAssertFalse(payload.resolvedDyPEConfig(width: 2048, height: 2048).enabled)
    }

    func testResolverRejectsUnknownValuesByDisabling() {
        let payload = GeneratePayload(prompt: "x", dype: "banana")
        XCTAssertFalse(payload.resolvedDyPEConfig(width: 2048, height: 2048).enabled)
    }

    // MARK: - txt2img (makePipelineRequest)

    func testHighResWithoutExplicitDypeAutoEnablesNTK() throws {
        let payload = GeneratePayload(prompt: "x", width: 2048, height: 2048)

        let request = try payload.makePipelineRequest(
            configuration: configuration(), activeLoRAs: [])

        XCTAssertTrue(request.dyPE.enabled, "above base resolution DyPE must engage")
        XCTAssertEqual(request.dyPE.method, .ntk)
    }

    func testNonSquareHighResAlsoAutoEnables() throws {
        // Only the longest edge needs to exceed base resolution.
        let payload = GeneratePayload(prompt: "x", width: 2048, height: 1152)

        let request = try payload.makePipelineRequest(
            configuration: configuration(), activeLoRAs: [])

        XCTAssertTrue(request.dyPE.enabled)
    }

    // MARK: - img2img (makeImg2ImgRequest) — the HQ-rerender shape

    func testImg2ImgHighResWithoutExplicitDypeAutoEnablesNTK() throws {
        let payload = GeneratePayload(
            prompt: "x", width: 2048, height: 2048,
            imagePath: "/tmp/source.png", imageStrength: 0.75)

        let request = try payload.makeImg2ImgRequest(
            configuration: configuration(), activeLoRAs: [])

        XCTAssertTrue(request.dyPE.enabled)
        XCTAssertEqual(request.dyPE.method, .ntk)
    }

    // MARK: - Guards: what must NOT change

    func testBaseResolutionLeavesDyPEDisabled() throws {
        // DyPE rewrites RoPE frequencies only when scale > 1.0, so at 1024 it is
        // a no-op anyway — it must stay off rather than pay setup.
        let payload = GeneratePayload(prompt: "x", width: 1024, height: 1024)

        let request = try payload.makePipelineRequest(
            configuration: configuration(), activeLoRAs: [])

        XCTAssertFalse(request.dyPE.enabled)
    }

    func testExplicitNoneDisablesEvenAtHighRes() throws {
        let payload = GeneratePayload(prompt: "x", width: 2048, height: 2048, dype: "none")

        let request = try payload.makePipelineRequest(
            configuration: configuration(), activeLoRAs: [])

        XCTAssertFalse(request.dyPE.enabled)
    }
}
