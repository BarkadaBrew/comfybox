import XCTest

@testable import ZImage

/// WP-E4 (FDD-krea2-raw-recipe D18, AC-28 regression half) — `eta` has shipped
/// on the Z-Image path since April (DDIM η / DPM++ 2S-A ancestral η). The tier
/// gate for RES4LYF SDE `eta` lives inside `runKrea2Generate`, NOT at the
/// decoder, so a Z-Image `{"scheduler":"ddim","eta":0.5}` must keep decoding,
/// validating and reaching the pipeline request unchanged.
final class ZImageEtaRegressionTests: XCTestCase {

  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  private func makeConfiguration() -> WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  func testDDIMWithEtaStillReachesThePipelineRequest() throws {
    let payload = try decode(#"{"prompt":"x","scheduler":"ddim","eta":0.5}"#)
    let names = try payload.validateRecipeNames()
    XCTAssertEqual(names.scheduler, .ddim)
    let request = try payload.makePipelineRequest(configuration: makeConfiguration(), activeLoRAs: [])
    XCTAssertEqual(request.schedulerKind, .ddim)
    XCTAssertEqual(request.eta, 0.5)
    XCTAssertEqual(request.sigmaSchedule, .flow)
  }

  func testDPMPP2SAWithEtaViaAliasStillReachesThePipelineRequest() throws {
    let payload = try decode(#"{"prompt":"x","scheduler":"dpmpp_2s_ancestral","sigma_schedule":"beta57","eta":1.0}"#)
    let request = try payload.makePipelineRequest(configuration: makeConfiguration(), activeLoRAs: [])
    XCTAssertEqual(request.schedulerKind, .dpmplusplus2sa)
    XCTAssertEqual(request.sigmaSchedule, .beta57)
    XCTAssertEqual(request.eta, 1.0)
  }

  /// Absent names keep today's defaults (euler / flow) — the defaults live at
  /// the request builder, not in the resolver.
  func testAbsentNamesKeepEulerFlowDefaults() throws {
    let payload = try decode(#"{"prompt":"x"}"#)
    let request = try payload.makePipelineRequest(configuration: makeConfiguration(), activeLoRAs: [])
    XCTAssertEqual(request.schedulerKind, .euler)
    XCTAssertEqual(request.sigmaSchedule, .flow)
    XCTAssertNil(request.eta)
  }

  /// The img2img builder goes through the same resolver: an unknown name
  /// throws instead of rendering euler.
  func testImg2ImgBuilderRejectsUnknownSampler() throws {
    let payload = try decode(#"{"prompt":"x","image_path":"/nonexistent.png","scheduler":"uni_pc"}"#)
    XCTAssertThrowsError(try payload.makeImg2ImgRequest(configuration: makeConfiguration(), activeLoRAs: [])) { error in
      guard case WarmServerError.unknownSampler(let name, _) = error else { return XCTFail("\(error)") }
      XCTAssertEqual(name, "uni_pc")
    }
  }
}
