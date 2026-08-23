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

  /// `eta` is forwarded as the RES4LYF SDE eta whatever the sampler, and the
  /// gate — not the forwarding — decides whether the render may have it
  /// (D18). Since WP-E15 that decision is by SAMPLER: honoured on a RES4LYF
  /// port, 400 on the default `euler`.
  func testEtaForwardsAndIsGatedBySampler() throws {
    let bare = try decode(#"{"prompt":"x","eta":0.5}"#)
    XCTAssertEqual(try bare.krea2RecipeFields().eta, 0.5)
    XCTAssertEqual(try bare.krea2RecipeFields().sampler, .euler)
    XCTAssertThrowsError(try bare.validateKrea2TierGates(try bare.validateRecipeNames()))

    let res2s = try decode(#"{"prompt":"x","scheduler":"res_2s","eta":0.5}"#)
    XCTAssertEqual(try res2s.krea2RecipeFields().eta, 0.5)
    XCTAssertEqual(try res2s.krea2RecipeFields().sampler, .res2s)
    XCTAssertNoThrow(try res2s.validateKrea2TierGates(try res2s.validateRecipeNames()))
  }

  /// WP-E16: `bongmath` is the eta arm's twin on the wire — forwarded whatever
  /// the sampler, gated by SAMPLER, and absent means `false` (a bare payload
  /// is byte-identical to before the key existed).
  func testBongmathForwardsAndIsGatedBySampler() throws {
    let bare = try decode(#"{"prompt":"x"}"#)
    XCTAssertFalse(try bare.krea2RecipeFields().bongmath, "absent is false, not nil")
    XCTAssertNoThrow(try bare.validateKrea2TierGates(try bare.validateRecipeNames()))

    let euler = try decode(#"{"prompt":"x","bongmath":true}"#)
    XCTAssertTrue(try euler.krea2RecipeFields().bongmath)
    XCTAssertEqual(try euler.krea2RecipeFields().sampler, .euler)
    XCTAssertThrowsError(
      try euler.validateKrea2TierGates(try euler.validateRecipeNames())
    ) { error in
      guard case WarmServerError.unsupportedRecipeField(let field, let value, let family, let why) =
        error
      else { return XCTFail("\(error)") }
      XCTAssertEqual(field, "bongmath")
      XCTAssertEqual(value, "true")
      XCTAssertEqual(family, "krea2")
      XCTAssertTrue(why.contains("euler"), "the refusal must name the sampler: \(why)")
    }

    // Explicit false is not a refusal on any sampler.
    let off = try decode(#"{"prompt":"x","bongmath":false}"#)
    XCTAssertFalse(try off.krea2RecipeFields().bongmath)
    XCTAssertNoThrow(try off.validateKrea2TierGates(try off.validateRecipeNames()))

    let res2s = try decode(#"{"prompt":"x","scheduler":"res_2s","bongmath":true,"eta":0.5}"#)
    let fields = try res2s.krea2RecipeFields()
    XCTAssertTrue(fields.bongmath)
    XCTAssertEqual(fields.eta, 0.5, "T3 composes with T2 on one payload")
    XCTAssertNoThrow(try res2s.validateKrea2TierGates(try res2s.validateRecipeNames()))
  }

  func testUnknownNamesStillThrow() throws {
    XCTAssertThrowsError(try decode(#"{"prompt":"x","scheduler":"uni_pc"}"#).krea2RecipeFields())
    XCTAssertThrowsError(try decode(#"{"prompt":"x","sigma_schedule":"ays"}"#).krea2RecipeFields())
  }
}
