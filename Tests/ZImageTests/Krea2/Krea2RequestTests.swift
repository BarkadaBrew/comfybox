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

// MARK: - WP-E12: the explicit `shift` field (FDD-krea2-raw-recipe D3)

extension Krea2RequestTests {

    func testRequestDefaultsToNilShift() {
        let request = Krea2Pipeline.Request(prompt: "x")
        XCTAssertNil(request.shift, "nil = today's resolution-dependent mu; existing renders are unmoved")
    }

    func testRequestCarriesShift() {
        let request = Krea2Pipeline.Request(prompt: "x", shift: 1.15)
        XCTAssertEqual(request.shift, 1.15)
    }

    func testImg2ImgRequestDefaultsToNilShift() {
        let source = MLX.zeros([1, 64, 64, 3])
        let request = Krea2Pipeline.Img2ImgRequest(prompt: "x", sourceImage: source)
        XCTAssertNil(request.shift)
    }

    func testImg2ImgRequestCarriesShift() {
        let source = MLX.zeros([1, 64, 64, 3])
        let request = Krea2Pipeline.Img2ImgRequest(prompt: "x", sourceImage: source, shift: 1.15)
        XCTAssertEqual(request.shift, 1.15)
    }
}

// MARK: - WP-E3: sampler / schedule / tier fields (FDD-krea2-raw-recipe §3.3, D1, D18, D23)

extension Krea2RequestTests {

  /// The defaults ARE today's behaviour: euler over the native krea2 warp at
  /// the resolution-dependent mu, no SDE, no bongmath, res_2s substep 0.5.
  func testRequestRecipeDefaultsPreserveToday() {
    let request = Krea2Pipeline.Request(prompt: "x")
    XCTAssertEqual(request.sampler, .euler)
    XCTAssertEqual(request.sigmaSchedule, .krea2)
    XCTAssertNil(request.shift)
    XCTAssertEqual(request.eta, 0)
    XCTAssertFalse(request.bongmath)
    XCTAssertEqual(request.c2, 0.5)
  }

  func testRequestCarriesRecipeFields() {
    let request = Krea2Pipeline.Request(
      prompt: "x", shift: 1.15, sampler: .res2s, sigmaSchedule: .beta57, eta: 0.5, bongmath: true, c2: 0.3)
    XCTAssertEqual(request.sampler, .res2s)
    XCTAssertEqual(request.sigmaSchedule, .beta57)
    XCTAssertEqual(request.shift, 1.15)
    XCTAssertEqual(request.eta, 0.5)
    XCTAssertTrue(request.bongmath)
    XCTAssertEqual(request.c2, 0.3)
  }

  func testImg2ImgRequestRecipeDefaultsPreserveToday() {
    let source = MLX.zeros([1, 64, 64, 3])
    let request = Krea2Pipeline.Img2ImgRequest(prompt: "x", sourceImage: source)
    XCTAssertEqual(request.sampler, .euler)
    XCTAssertEqual(request.sigmaSchedule, .krea2)
    XCTAssertNil(request.shift)
    XCTAssertEqual(request.eta, 0)
    XCTAssertFalse(request.bongmath)
    XCTAssertEqual(request.c2, 0.5)
  }

  func testImg2ImgRequestCarriesRecipeFields() {
    let source = MLX.zeros([1, 64, 64, 3])
    let request = Krea2Pipeline.Img2ImgRequest(
      prompt: "x", sourceImage: source, shift: 1.15,
      sampler: .dpmplusplus2m, sigmaSchedule: .flow, eta: 0.25, bongmath: true, c2: 0.4)
    XCTAssertEqual(request.sampler, .dpmplusplus2m)
    XCTAssertEqual(request.sigmaSchedule, .flow)
    XCTAssertEqual(request.shift, 1.15)
    XCTAssertEqual(request.eta, 0.25)
    XCTAssertTrue(request.bongmath)
    XCTAssertEqual(request.c2, 0.4)
  }
}
