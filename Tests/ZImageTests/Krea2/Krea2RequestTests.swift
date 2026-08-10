import XCTest
import MLX
@testable import ZImage

/// The Krea-2 requests carry DyPE from the server down to the transformer.
/// Constructing a Request loads no weights, so these are cheap.
final class Krea2RequestTests: XCTestCase {

    func testRequestDefaultsToDisabledDyPE() {
        let request = Krea2Pipeline.Request(prompt: "x")
        XCTAssertFalse(request.dyPE.enabled,
                       "existing callers must keep vanilla RoPE")
    }

    func testRequestCarriesDyPE() {
        let request = Krea2Pipeline.Request(
            prompt: "x", width: 2048, height: 2048, dyPE: .ntk)
        XCTAssertTrue(request.dyPE.enabled)
        XCTAssertEqual(request.dyPE.method, .ntk)
    }

    func testImg2ImgRequestDefaultsToDisabledDyPE() {
        let source = MLX.zeros([1, 64, 64, 3])
        let request = Krea2Pipeline.Img2ImgRequest(prompt: "x", sourceImage: source)
        XCTAssertFalse(request.dyPE.enabled)
    }

    func testImg2ImgRequestCarriesDyPE() {
        // The HQ 2K rerender path: upsized source, DyPE on.
        let source = MLX.zeros([1, 64, 64, 3])
        let request = Krea2Pipeline.Img2ImgRequest(
            prompt: "x", sourceImage: source,
            width: 2048, height: 2048, strength: 0.75, dyPE: .ntk)
        XCTAssertTrue(request.dyPE.enabled)
        XCTAssertEqual(request.dyPE.method, .ntk)
    }
}
