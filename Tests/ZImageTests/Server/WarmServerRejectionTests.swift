import XCTest

@testable import ZImage

/// WP-E4 (FDD-krea2-raw-recipe §3.4, AC-15, AC-28) — the new `WarmServerError`
/// cases map to HTTP 400 with a message that names the offending value and
/// the valid set. The route handlers (`/v1/generate`, `/v1/generate/async`)
/// both decode through `decodedGeneratePayload`, which now runs
/// `validateRecipeNames()` BEFORE anything is enqueued — so a rejected name
/// never reaches a render, never increments `successful_render_count` and
/// never writes a file. (Route-level assertion lives in the E2E suite.)
final class WarmServerRejectionTests: XCTestCase {

  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  private func bodyString(_ response: HTTPResponse) -> String {
    String(decoding: response.body, as: UTF8.self)
  }

  func testUnknownSamplerIs400NamingValueAndValidSet() throws {
    let payload = try decode(#"{"prompt":"x","scheduler":"uni_pc"}"#)
    var caught: Error?
    do { _ = try payload.validateRecipeNames() } catch { caught = error }
    let error = try XCTUnwrap(caught)
    let response = WarmServer.errorResponse(for: error)
    XCTAssertEqual(response.status, 400)
    let body = bodyString(response)
    XCTAssertTrue(body.contains("uni_pc"), body)
    for valid in RecipeNameResolver.validSamplerNames {
      XCTAssertTrue(body.contains(valid), "400 body must list valid sampler '\(valid)': \(body)")
    }
  }

  /// Same rejection through the D25 alias key.
  func testUnknownSamplerViaSamplerAliasIs400() throws {
    let payload = try decode(#"{"prompt":"x","sampler":"uni_pc"}"#)
    XCTAssertThrowsError(try payload.validateRecipeNames()) { error in
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      XCTAssertTrue(bodyString(response).contains("uni_pc"))
    }
  }

  func testUnknownSigmaScheduleIs400() throws {
    let payload = try decode(#"{"prompt":"x","sigma_schedule":"ays"}"#)
    XCTAssertThrowsError(try payload.validateRecipeNames()) { error in
      let response = WarmServer.errorResponse(for: error)
      XCTAssertEqual(response.status, 400)
      let body = bodyString(response)
      XCTAssertTrue(body.contains("ays"), body)
      XCTAssertTrue(body.contains("flow"), body)
    }
  }

  func testMutuallyExclusiveKeysIs400() {
    let error = WarmServerError.mutuallyExclusive("scheduler='res_2s' and sampler='euler' disagree")
    let response = WarmServer.errorResponse(for: error)
    XCTAssertEqual(response.status, 400)
    XCTAssertTrue(bodyString(response).contains("disagree"))
  }

  /// D18: an unimplemented tier on the Krea 2 path is a 400 naming the
  /// unsupported value — never a downgrade.
  func testUnsupportedRecipeFieldIs400() {
    let error = WarmServerError.unsupportedRecipeField(
      field: "eta", value: "0.5", family: "krea2", reason: "RES4LYF SDE eta (T2) is not implemented yet")
    let response = WarmServer.errorResponse(for: error)
    XCTAssertEqual(response.status, 400)
    let body = bodyString(response)
    XCTAssertTrue(body.contains("eta"), body)
    XCTAssertTrue(body.contains("0.5"), body)
    XCTAssertTrue(body.contains("krea2"), body)
  }

  /// Krea 2 tier gates (D18, §3.4): `eta != 0` is refused before T2; a sampler
  /// or schedule the Krea 2 loop does not yet honour is refused rather than
  /// silently rendered as euler / native warp (removed by WP-E3 when the loop
  /// takes the sampler). `eta: 0`, euler and the flow aliases pass.
  func testKrea2TierGates() throws {
    let eta = try decode(#"{"prompt":"x","eta":0.5}"#)
    XCTAssertThrowsError(try eta.validateKrea2TierGates(try eta.validateRecipeNames())) { error in
      guard case WarmServerError.unsupportedRecipeField(let field, let value, let family, _) = error else {
        return XCTFail("\(error)")
      }
      XCTAssertEqual(field, "eta"); XCTAssertEqual(value, "0.5"); XCTAssertEqual(family, "krea2")
    }
    let etaZero = try decode(#"{"prompt":"x","eta":0}"#)
    XCTAssertNoThrow(try etaZero.validateKrea2TierGates(try etaZero.validateRecipeNames()))

    let sampler = try decode(#"{"prompt":"x","scheduler":"res_2s"}"#)
    XCTAssertThrowsError(try sampler.validateKrea2TierGates(try sampler.validateRecipeNames())) { error in
      guard case WarmServerError.unsupportedRecipeField(let field, let value, _, _) = error else {
        return XCTFail("\(error)")
      }
      XCTAssertEqual(field, "scheduler"); XCTAssertEqual(value, "res_2s")
    }
    let schedule = try decode(#"{"prompt":"x","sigma_schedule":"karras"}"#)
    XCTAssertThrowsError(try schedule.validateKrea2TierGates(try schedule.validateRecipeNames())) { error in
      guard case WarmServerError.unsupportedRecipeField(let field, let value, _, _) = error else {
        return XCTFail("\(error)")
      }
      XCTAssertEqual(field, "sigma_schedule"); XCTAssertEqual(value, "karras")
    }
    // Krita's default style on the bridge: euler / normal → passes.
    let kritaDefault = try decode(#"{"prompt":"x","scheduler":"euler","sigma_schedule":"normal"}"#)
    XCTAssertNoThrow(try kritaDefault.validateKrea2TierGates(try kritaDefault.validateRecipeNames()))
    let bare = try decode(#"{"prompt":"x"}"#)
    XCTAssertNoThrow(try bare.validateKrea2TierGates(try bare.validateRecipeNames()))
  }

  /// Decoding errors stay 400 and untouched by the new arms.
  func testDecodingErrorStill400() {
    let error: Error
    do {
      _ = try decode(#"{"width":512}"#)
      return XCTFail("missing prompt must fail decode")
    } catch let e { error = e }
    XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
  }
}
