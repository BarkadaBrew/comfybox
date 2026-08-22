import XCTest
import Foundation
@testable import ZImage

/// WP-E3 — `runKrea2Generate` forwards `scheduler` / `sigma_schedule` /
/// `shift` / `eta` into the Krea 2 request (FDD-krea2-raw-recipe §3.3, D11,
/// D22, D25). The forwarding is a pure function of the payload so it is
/// asserted here without a server or weights.
final class Krea2RecipeForwardingTests: XCTestCase {

  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  /// A bare payload yields the pre-change recipe exactly (AC-1/AC-5).
  func testBarePayloadIsTodaysRecipe() throws {
    let p = try decode(#"{"prompt":"x"}"#)
    let fields = try p.krea2RecipeFields()
    XCTAssertEqual(fields.sampler, .euler)
    XCTAssertEqual(fields.sigmaSchedule, .krea2)
    XCTAssertNil(fields.shift)
    XCTAssertEqual(fields.eta, 0)
    XCTAssertNil(fields.samplerRequested)
    XCTAssertNil(fields.sigmaScheduleRequested)
  }

  func testNamedRecipeIsForwarded() throws {
    let p = try decode(#"{"prompt":"x","scheduler":"res_2s","sigma_schedule":"beta57","shift":1.15,"eta":0}"#)
    let fields = try p.krea2RecipeFields()
    XCTAssertEqual(fields.sampler, .res2s)
    XCTAssertEqual(fields.sigmaSchedule, .beta57)
    XCTAssertEqual(fields.shift, 1.15)
    XCTAssertEqual(fields.eta, 0)
    XCTAssertEqual(fields.samplerRequested, "res_2s")
    XCTAssertEqual(fields.sigmaScheduleRequested, "beta57")
  }

  /// `sampler` is the decoded alias of `scheduler` (D25); RES4LYF prefixes strip.
  func testSamplerAliasAndPrefixForward() throws {
    let p = try decode(#"{"prompt":"x","sampler":"exponential/res_2s"}"#)
    let fields = try p.krea2RecipeFields()
    XCTAssertEqual(fields.sampler, .res2s)
    XCTAssertEqual(fields.samplerRequested, "exponential/res_2s")
  }

  /// D22: Krita's `normal` is a declared alias of `flow`, which on Krea 2 is
  /// the shifted 1→1/1000 grid (D11) — the alias is visible, not silent.
  func testKritaNormalForwardsAsFlowWithTheRequestedName() throws {
    let p = try decode(#"{"prompt":"x","scheduler":"euler","sigma_schedule":"normal"}"#)
    let fields = try p.krea2RecipeFields()
    XCTAssertEqual(fields.sampler, .euler)
    XCTAssertEqual(fields.sigmaSchedule, .flow)
    XCTAssertEqual(fields.sigmaScheduleRequested, "normal")
  }

  /// `eta` is forwarded as the RES4LYF SDE eta and is still gated at the
  /// server before T2 (D18) — the gate refuses, the forwarding does not hide it.
  func testEtaForwardsAndIsGated() throws {
    let p = try decode(#"{"prompt":"x","eta":0.5}"#)
    XCTAssertEqual(try p.krea2RecipeFields().eta, 0.5)
    XCTAssertThrowsError(try p.validateKrea2TierGates(try p.validateRecipeNames()))
  }

  func testUnknownNamesStillThrow() throws {
    XCTAssertThrowsError(try decode(#"{"prompt":"x","scheduler":"uni_pc"}"#).krea2RecipeFields())
    XCTAssertThrowsError(try decode(#"{"prompt":"x","sigma_schedule":"ays"}"#).krea2RecipeFields())
  }
}
