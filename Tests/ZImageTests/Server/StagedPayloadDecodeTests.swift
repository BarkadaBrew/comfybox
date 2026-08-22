import XCTest
@testable import ZImage

/// WP-E17 — the `stage2` wire field (FDD-krea2-raw-recipe §3.14, D4, D25;
/// AC-28's rejection matrix, AC-29's "one payload").
///
/// Pure decode + gate assertions: no server, no weights. Everything here is a
/// function of the payload, which is why the forwarding can be pinned at all.
final class StagedPayloadDecodeTests: XCTestCase {

  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  /// The published recipe, in ONE payload (AC-29's request half).
  private static let referencePayload = """
  {"prompt":"x","scheduler":"res_2s","sigma_schedule":"beta","shift":1.15,
   "steps":6,"guidance":1.0,"eta":0.5,
   "stage2":{"scheduler":"deis_3m","sigma_schedule":"bong_tangent",
             "steps":2,"denoise":0.2,"guidance":1.0,"eta":0.5,"seed":null}}
  """

  // MARK: - Decode

  func testReferenceStageTwoDecodesThroughSnakeCase() throws {
    let p = try decode(Self.referencePayload)
    let s2 = try XCTUnwrap(p.stage2)
    XCTAssertEqual(s2.steps, 2)
    XCTAssertEqual(s2.denoise, 0.2)
    XCTAssertEqual(s2.scheduler, "deis_3m")
    XCTAssertEqual(s2.sigmaSchedule, "bong_tangent")
    XCTAssertEqual(s2.guidance, 1.0)
    XCTAssertEqual(s2.eta, 0.5)
    XCTAssertNil(s2.seed, "an explicit null seed decodes as absent — stage1 &+ 1")
    XCTAssertNil(s2.bongmath)
  }

  /// A payload with no `stage2` is the single-stage request it always was.
  func testNoStageTwoIsAbsentNotDefaulted() throws {
    let p = try decode(#"{"prompt":"x","steps":9}"#)
    XCTAssertNil(p.stage2)
    XCTAssertNil(p.detailPass)
    XCTAssertNil(p.detailDenoise)
    XCTAssertNil(try p.krea2Stage2Fields())
  }

  /// `denoise` decodes as **Double**, and the value is carried at Double
  /// precision to the arithmetic that selects the tail (§3.14, AC-31). A
  /// `Float` round-trip of `0.3` would be `0.30000001192092896`, and
  /// `9 / that` truncates to 29 rather than 30.
  func testDenoiseKeepsDoublePrecision() throws {
    let p = try decode(
      #"{"prompt":"x","scheduler":"deis_3m","stage2":{"steps":9,"denoise":0.3}}"#)
    let denoise = try XCTUnwrap(p.stage2).denoise
    XCTAssertEqual(denoise, 0.3)
    XCTAssertEqual(
      try Krea2StagedRender.stretchedStepCount(steps: 9, denoise: denoise), 30,
      "a Float round-trip here would select 29")
  }

  /// D25 on the stage too: `sampler` is an accepted alias of `scheduler`, and
  /// the two disagreeing is a 400 rather than a coin flip.
  func testStageTwoSamplerAlias() throws {
    let aliased = try decode(
      #"{"prompt":"x","stage2":{"steps":2,"denoise":0.2,"sampler":"deis_3m"}}"#)
    XCTAssertEqual(try XCTUnwrap(aliased.stage2).scheduler, "deis_3m")
    XCTAssertEqual(try XCTUnwrap(try aliased.krea2Stage2Fields()).sampler, .deis3m)

    XCTAssertThrowsError(
      try decode(
        #"{"prompt":"x","stage2":{"steps":2,"denoise":0.2,"scheduler":"deis_3m","sampler":"euler"}}"#))
  }

  /// The two grid-deciding fields are required: there is no default for either
  /// that would not be an invented recipe.
  func testStepsAndDenoiseAreRequired() {
    XCTAssertThrowsError(try decode(#"{"prompt":"x","stage2":{"denoise":0.2}}"#))
    XCTAssertThrowsError(try decode(#"{"prompt":"x","stage2":{"steps":2}}"#))
  }

  // MARK: - Forwarding

  /// What the pipeline is handed: resolved kinds, the raw alias recorded, and
  /// the unstated fields left `nil` so the RENDER's own values fill them.
  func testForwardsResolvedKindsAndKeepsTheAlias() throws {
    let p = try decode("""
      {"prompt":"x","stage2":{"steps":2,"denoise":0.2,"scheduler":"multistep/deis_3m",
                              "sigma_schedule":"normal","bongmath":false,"seed":900}}
      """)
    let s2 = try XCTUnwrap(try p.krea2Stage2Fields())
    XCTAssertEqual(s2.sampler, .deis3m)
    XCTAssertEqual(s2.sigmaSchedule, .flow, "D22: 'normal' is an alias of flow")
    XCTAssertEqual(s2.sigmaScheduleRequested, "normal")
    XCTAssertEqual(s2.seed, 900)
    XCTAssertEqual(s2.bongmath, false)
    XCTAssertNil(s2.guidance, "unstated → the render's own guidance")
    XCTAssertNil(s2.eta)
  }

  func testUnknownStageTwoSamplerIsRefusedByName() throws {
    let p = try decode(#"{"prompt":"x","stage2":{"steps":2,"denoise":0.2,"scheduler":"res_5s"}}"#)
    XCTAssertThrowsError(try p.krea2Stage2Fields()) { error in
      guard case .unknownSampler(let name, _)? = error as? WarmServerError else {
        return XCTFail("expected unknownSampler, got \(error)")
      }
      XCTAssertEqual(name, "res_5s")
    }
  }

  // MARK: - AC-28: the rejection matrix

  func testRejectionMatrix() throws {
    let staged = try decode(Self.referencePayload)

    // `stage2` on a non-krea2 family is a 400 naming the field and the family —
    // never a silently single-stage render.
    for family in [WarmModelFamily.flux1, .flux2, .fibo, .chroma] {
      guard case .unsupportedRecipeField(let field, _, let named, _)? =
              GeneratePayload.stage2Gate(staged, family: family)
      else { return XCTFail("stage2 must be refused on \(family.rawValue)") }
      XCTAssertEqual(field, "stage2")
      XCTAssertEqual(named, family.rawValue)
    }
    XCTAssertNil(GeneratePayload.stage2Gate(staged, family: .krea2))

    // `stage2.denoise: 0` — §3.14's hard error.
    for bad in ["0", "-0.2", "1.5"] {
      let p = try decode(#"{"prompt":"x","stage2":{"steps":2,"denoise":\#(bad)}}"#)
      guard case .unsupportedRecipeField(let field, _, _, _)? =
              GeneratePayload.stage2Gate(p, family: .krea2)
      else { return XCTFail("denoise \(bad) must be refused") }
      XCTAssertEqual(field, "stage2.denoise")
    }

    // `stage2.steps: 0`.
    let zeroSteps = try decode(#"{"prompt":"x","stage2":{"steps":0,"denoise":0.2}}"#)
    guard case .unsupportedRecipeField(let stepsField, _, _, _)? =
            GeneratePayload.stage2Gate(zeroSteps, family: .krea2)
    else { return XCTFail("stage2.steps 0 must be refused") }
    XCTAssertEqual(stepsField, "stage2.steps")

    // `stage2` naming a sampler the family cannot run. Krea 2 runs them all, so
    // the row that bites is a tableau sampler on a fixed-Euler family — which
    // the family gate above already refuses at the `stage2` field itself.
    let tableau = try decode(
      #"{"prompt":"x","stage2":{"steps":2,"denoise":0.2,"scheduler":"ralston_4s"}}"#)
    XCTAssertNotNil(GeneratePayload.stage2Gate(tableau, family: .flux1))
    XCTAssertNil(GeneratePayload.stage2Gate(tableau, family: .krea2))
  }

  /// The stage's tier gates are evaluated on what it will RUN with — an
  /// unstated field inherits the render's — and they are 400s at dispatch, not
  /// 500s from an unmapped pipeline throw halfway through the render.
  func testStageTwoTierGates() throws {
    // T3 is WP-E16's; `stage2.bongmath: true` is refused by name.
    let bong = try decode(#"{"prompt":"x","stage2":{"steps":2,"denoise":0.2,"bongmath":true}}"#)
    guard case .unsupportedRecipeField(let bongField, _, _, _)? =
            GeneratePayload.stage2Gate(bong, family: .krea2)
    else { return XCTFail("stage2.bongmath must be refused until T3 lands") }
    XCTAssertEqual(bongField, "stage2.bongmath")

    // An INHERITED eta on a stage that names a non-RES4LYF sampler: the render
    // is legal (res_2s + eta), the stage is not, and the pairing that matters
    // is the resolved one.
    let inherited = try decode("""
      {"prompt":"x","scheduler":"res_2s","eta":0.5,
       "stage2":{"steps":2,"denoise":0.2,"scheduler":"euler"}}
      """)
    guard case .unsupportedRecipeField(let etaField, let etaValue, _, let reason)? =
            GeneratePayload.stage2Gate(inherited, family: .krea2)
    else { return XCTFail("an inherited eta on a euler stage 2 must be refused") }
    XCTAssertEqual(etaField, "stage2.eta")
    XCTAssertEqual(etaValue, "0.5")
    XCTAssertTrue(reason.contains("euler"), "the refusal names the sampler: \(reason)")

    // The stage INHERITING a RES4LYF sampler is fine.
    XCTAssertNil(GeneratePayload.stage2Gate(
      try decode("""
        {"prompt":"x","scheduler":"res_2s","eta":0.5,"stage2":{"steps":2,"denoise":0.2}}
        """),
      family: .krea2))
  }

  /// The pipeline's own refusals are the CALLER's error even when no wire gate
  /// pre-empted them — a non-server caller and any uncovered path still get a
  /// 400 naming the field rather than a 500 naming nothing.
  func testPipelineRefusalsMapTo400() {
    for error: Error in [
      Krea2StageError.invalidDenoise(0),
      Krea2StageError.degenerateStageGrid(schedule: "beta", steps: 4, denoise: 0.9, produced: 1),
      Krea2ScheduleError.tierNotImplemented(field: "bongmath", value: "true", tier: "T3"),
      SchedulerFactoryError.stepCountBelowMinimum(.bongTangent, steps: 1, minimum: 2),
    ] {
      XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400, "\(error)")
    }
  }

  /// `detail_denoise` / `detail_pass` are the MCP TOOL-SCHEMA spelling (§3.17,
  /// AC-68a): the client expands `detail_pass` into `stage2` from its family
  /// policy table. The engine has no such table, so it refuses both rather
  /// than dropping them — a request that names a detail pass and silently
  /// renders one stage is exactly the failure this programme exists to kill.
  func testDetailPassKeysAreRefusedNotDropped() throws {
    // Addendum A.2 / C3: `detail_denoise` without `detail_pass` is an orphan.
    let orphan = try decode(#"{"prompt":"x","detail_denoise":0.2}"#)
    XCTAssertEqual(orphan.detailDenoise, 0.2)
    guard case .orphanField(let field, let requires, _)? =
            GeneratePayload.stage2Gate(orphan, family: .krea2)
    else { return XCTFail("orphan detail_denoise must be refused") }
    XCTAssertEqual(field, "detail_denoise")
    XCTAssertEqual(requires, "detail_pass")

    // NaN included (the same A.2 row).
    let nan = GeneratePayload(prompt: "x", detailDenoise: Double.nan)
    XCTAssertNotNil(GeneratePayload.stage2Gate(nan, family: .krea2))

    // With `detail_pass`, the engine says what to send instead.
    let withPass = try decode(#"{"prompt":"x","detail_pass":true,"detail_denoise":0.2}"#)
    guard case .unsupportedRecipeField(let f, _, _, let reason)? =
            GeneratePayload.stage2Gate(withPass, family: .krea2)
    else { return XCTFail("detail_pass must be refused") }
    XCTAssertEqual(f, "detail_pass")
    XCTAssertTrue(reason.contains("stage2"), "the refusal must name the engine's own field")

    // A payload with neither key is untouched.
    XCTAssertNil(GeneratePayload.stage2Gate(try decode(#"{"prompt":"x"}"#), family: .krea2))
  }
}
