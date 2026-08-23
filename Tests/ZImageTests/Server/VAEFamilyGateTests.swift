import XCTest
@testable import ZImage

/// WP-E10 "E9b" (FDD Addendum A.2, E9 review MAJOR): `payload.vae` on a
/// non-krea2 family used to be silently ignored — the caller named a decoder
/// and got the family's default with no error and no log. It is now refused
/// with a 400 naming the field and the family, the same shape as the D3
/// `shift` gate (`GeneratePayload.validateShift`): a pure function over
/// (field, family) so it is asserted here without a server.
final class VAEFamilyGateTests: XCTestCase {

  func testAbsentVAEIsFineOnEveryFamily() {
    for family in WarmModelFamily.allCases {
      XCTAssertNil(GeneratePayload.vaeGate(nil, family: family), "\(family)")
    }
  }

  func testVAEIsAcceptedOnKrea2() {
    XCTAssertNil(GeneratePayload.vaeGate("~/LocalModels/vae/Wan2_1_VAE_fp32.safetensors", family: .krea2))
  }

  func testVAEOnAnotherFamilyIsRefusedNamingFieldAndFamily() throws {
    for family in WarmModelFamily.allCases where family != .krea2 {
      let error = try XCTUnwrap(
        GeneratePayload.vaeGate("~/LocalModels/vae/Wan2_1_VAE_fp32.safetensors", family: family),
        "\(family) must refuse, not ignore, a named VAE")
      guard case .unsupportedRecipeField(let field, let value, let named, _) = error else {
        return XCTFail("expected unsupportedRecipeField, got \(error)")
      }
      XCTAssertEqual(field, "vae")
      XCTAssertEqual(value, "~/LocalModels/vae/Wan2_1_VAE_fp32.safetensors")
      XCTAssertEqual(named, family.rawValue)
      let text = error.localizedDescription
      XCTAssertTrue(text.contains("vae"), text)
      XCTAssertTrue(text.contains(family.rawValue), text)
      // And it maps to a 400 (the caller's error), never a 500.
      XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
    }
  }
}
