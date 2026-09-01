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

  /// WP-E13 (review finding 1): an N-row tableau sampler is a 400 on any
  /// family whose denoise loop takes one model evaluation per step. It stays
  /// accepted and advertised (E4) — the gate is family-scoped and lives at
  /// dispatch, beside `validateShift`, not at the decoder (D18).
  ///
  /// Before this gate, `{"scheduler":"ralston_4s"}` on a Z-Image model decoded
  /// fine and rendered first-order Euler under the name `ralston_4s`: no 400,
  /// no warning, no telemetry. That is the substitution §3.4 exists to kill.
  func testTableauSamplerOffKrea2Is400() throws {
    for name in ["ralston_2s", "ralston_3s", "ralston_4s", "res_3s"] {
      let payload = try decode(#"{"prompt":"x","scheduler":"\#(name)"}"#)
      let names = try payload.validateRecipeNames()

      // Krea 2 dispatches the N-row protocol, so it is honoured there.
      XCTAssertNil(
        GeneratePayload.validateFamilyRecipe(names, family: .krea2),
        "\(name) must render on krea2")

      for family in WarmModelFamily.allCases where family != .krea2 {
        let error = try XCTUnwrap(
          GeneratePayload.validateFamilyRecipe(names, family: family),
          "\(name) on \(family.rawValue) must be refused")
        guard case WarmServerError.unsupportedSampler(let got, let gotFamily, _) = error else {
          return XCTFail("\(error) is not .unsupportedSampler")
        }
        XCTAssertEqual(got, name)
        XCTAssertEqual(gotFamily, family.rawValue)

        let response = WarmServer.errorResponse(for: error)
        XCTAssertEqual(response.status, 400, "\(name) on \(family.rawValue)")
        let body = bodyString(response)
        XCTAssertTrue(body.contains(name), body)
        XCTAssertTrue(body.contains(family.rawValue), body)
        // …and it says where the sampler DOES work, plus what to use instead.
        XCTAssertTrue(body.contains("Krea 2"), body)
        XCTAssertTrue(body.contains("euler"), body)
      }
    }

    // Every non-tableau sampler is untouched on the families that DRIVE it —
    // including the 2-row `res_2s`, which the Z-Image loop does dispatch.
    // K-FIX-1 / I5: "every family" was the silence; the gate is now the family
    // capability matrix, so a family that cannot run a sampler refuses it
    // instead of rendering euler under its name.
    for kind in SchedulerKind.allCases where !kind.isNRowTableau {
      let payload = try decode(#"{"prompt":"x","scheduler":"\#(kind.rawValue)"}"#)
      let names = try payload.validateRecipeNames()
      for family in WarmModelFamily.allCases {
        let runs = FamilyRecipeMatrix.capability(for: family).samplers.contains(kind)
        XCTAssertEqual(
          GeneratePayload.validateFamilyRecipe(names, family: family) == nil, runs,
          "\(kind.rawValue) on \(family.rawValue)")
      }
    }
  }

  /// Krea 2 tier gates (D18, §3.4). Since WP-E3 the loop takes the sampler and
  /// the schedule, so every name the resolver accepts passes the gate —
  /// `res_2s`, `karras`, Krita's `euler`/`normal`, and a bare request alike.
  ///
  /// WP-E15 changed what `eta` means here, not whether it is checked. T2 has
  /// landed, so `eta != 0` is APPLIED on a RES4LYF sampler and refused — 400,
  /// naming the sampler — on anything else. A bare `{"eta":0.5}` still 400s,
  /// because the Krea 2 default sampler is `euler` and RES4LYF's SDE is not
  /// defined against it.
  func testKrea2TierGates() throws {
    let eta = try decode(#"{"prompt":"x","eta":0.5}"#)
    XCTAssertThrowsError(try eta.validateKrea2TierGates(try eta.validateRecipeNames())) { error in
      guard case WarmServerError.unsupportedRecipeField(let field, let value, let family, let reason) = error else {
        return XCTFail("\(error)")
      }
      XCTAssertEqual(field, "eta"); XCTAssertEqual(value, "0.5"); XCTAssertEqual(family, "krea2")
      XCTAssertTrue(reason.contains("euler"), "the refusal names the sampler: \(reason)")
      XCTAssertTrue(reason.contains("res_2s"), "…and what to send instead: \(reason)")
    }
    let etaZero = try decode(#"{"prompt":"x","eta":0}"#)
    XCTAssertNoThrow(try etaZero.validateKrea2TierGates(try etaZero.validateRecipeNames()))

    // WP-E15: on a RES4LYF sampler the same eta is honoured, not refused.
    for sampler in ["res_2s", "res_3s", "ralston_3s", "deis_3m"] {
      let ok = try decode(#"{"prompt":"x","scheduler":"\#(sampler)","eta":0.5}"#)
      XCTAssertNoThrow(
        try ok.validateKrea2TierGates(try ok.validateRecipeNames()), "eta 0.5 on \(sampler)")
    }
    // …and on the samplers where `eta` already means something else, refused.
    for sampler in ["ddim", "dpmpp-2s-a", "heun"] {
      let bad = try decode(#"{"prompt":"x","scheduler":"\#(sampler)","eta":0.5}"#)
      XCTAssertThrowsError(
        try bad.validateKrea2TierGates(try bad.validateRecipeNames()), "eta 0.5 on \(sampler)")
    }

    // WP-E3: the sampler and the schedule are honoured, not gated.
    let sampler = try decode(#"{"prompt":"x","scheduler":"res_2s"}"#)
    XCTAssertNoThrow(try sampler.validateKrea2TierGates(try sampler.validateRecipeNames()))
    let schedule = try decode(#"{"prompt":"x","sigma_schedule":"karras"}"#)
    XCTAssertNoThrow(try schedule.validateKrea2TierGates(try schedule.validateRecipeNames()))
    for name in RecipeNameResolver.validSamplerNames {
      let p = try decode(#"{"prompt":"x","scheduler":"\#(name)"}"#)
      XCTAssertNoThrow(try p.validateKrea2TierGates(try p.validateRecipeNames()), name)
    }
    for name in RecipeNameResolver.validSigmaScheduleNames {
      let p = try decode(#"{"prompt":"x","sigma_schedule":"\#(name)"}"#)
      XCTAssertNoThrow(try p.validateKrea2TierGates(try p.validateRecipeNames()), name)
    }
    // Krita's default style on the bridge: euler / normal → passes.
    let kritaDefault = try decode(#"{"prompt":"x","scheduler":"euler","sigma_schedule":"normal"}"#)
    XCTAssertNoThrow(try kritaDefault.validateKrea2TierGates(try kritaDefault.validateRecipeNames()))
    let bare = try decode(#"{"prompt":"x"}"#)
    XCTAssertNoThrow(try bare.validateKrea2TierGates(try bare.validateRecipeNames()))
  }

  /// M1 (ClownsharK adversarial review): `projector_scale` is validated where
  /// the payload is applied — non-finite or outside the Desktop dial's 0…3
  /// clamp range is a 400 naming the value, never a clamp. Absent → the
  /// neutral 1.0; every in-range value (including the boundaries) passes
  /// through unchanged.
  // Implicit-RK batch-2 review F1: negative implicit_steps trapped the
  // denoise-loop precondition (process abort); unbounded hung renders.
  func testImplicitStepsOutOfRangeIsRejected() throws {
    for json in [
      #"{"prompt":"x","implicit_steps":-1}"#,
      #"{"prompt":"x","implicit_steps":9}"#,
      #"{"prompt":"x","implicit_steps":1000000}"#,
    ] {
      let payload = try decode(json)
      XCTAssertThrowsError(try payload.validatedImplicitSteps(), json) { error in
        guard case WarmServerError.implicitStepsOutOfRange = error else {
          return XCTFail("expected implicitStepsOutOfRange, got \(error) for \(json)")
        }
      }
    }
  }

  func testImplicitStepsAbsentAndValidPassThrough() throws {
    XCTAssertEqual(try decode(#"{"prompt":"x"}"#).validatedImplicitSteps(), 0)
    XCTAssertEqual(try decode(#"{"prompt":"x","implicit_steps":0}"#).validatedImplicitSteps(), 0)
    XCTAssertEqual(try decode(#"{"prompt":"x","implicit_steps":4}"#).validatedImplicitSteps(), 4)
    XCTAssertEqual(try decode(#"{"prompt":"x","implicit_steps":8}"#).validatedImplicitSteps(), 8)
  }

  func testC2RejectsInvalidRangeAndRES3sPole() throws {
    for json in [
      #"{"prompt":"x","c2":0}"#,
      #"{"prompt":"x","c2":-0.1}"#,
      #"{"prompt":"x","c2":1.01}"#,
      #"{"prompt":"x","c2":0.6666667}"#,
    ] {
      XCTAssertThrowsError(try decode(json).validatedC2(), json) { error in
        guard case WarmServerError.c2OutOfRange = error else {
          return XCTFail("expected c2OutOfRange, got \(error) for \(json)")
        }
        XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
      }
    }

    XCTAssertEqual(try decode(#"{"prompt":"x"}"#).validatedC2(), 0.5)
    XCTAssertEqual(try decode(#"{"prompt":"x","c2":0.05}"#).validatedC2(), 0.05)
    XCTAssertEqual(try decode(#"{"prompt":"x","c2":1}"#).validatedC2(), 1)
  }

  func testProjectorScaleOutOfRangeIs400() throws {
    // Out of range → refused by value. Non-finite cannot arrive via JSON --
    // Foundation refuses overflow literals at decode with its own 400; the
    // non-finite validator arm is wire-unreachable defense-in-depth.
    for json in [
      #"{"prompt":"x","projector_scale":-0.01}"#,
      #"{"prompt":"x","projector_scale":3.01}"#,
      #"{"prompt":"x","projector_scale":-1}"#,
    ] {
      let payload = try decode(json)
      XCTAssertThrowsError(try payload.validatedProjectorScale(), json) { error in
        guard case WarmServerError.projectorScaleOutOfRange(let value) = error else {
          return XCTFail("expected projectorScaleOutOfRange, got \(error)")
        }
        let response = WarmServer.errorResponse(for: error)
        XCTAssertEqual(response.status, 400)
        let body = bodyString(response)
        XCTAssertTrue(body.contains("projector_scale"), body)
        XCTAssertTrue(body.contains(value), "message must name the value: \(body)")
        XCTAssertTrue(body.contains("0.0...3.0"), "message must name the range: \(body)")
      }
    }

    // In range (boundaries included) → applied as sent; absent → neutral 1.0.
    for (json, expected) in [
      (#"{"prompt":"x","projector_scale":0}"#, Float(0)),
      (#"{"prompt":"x","projector_scale":3}"#, Float(3)),
      (#"{"prompt":"x","projector_scale":1.35}"#, Float(1.35)),
      (#"{"prompt":"x"}"#, Float(1.0)),
    ] {
      XCTAssertEqual(try decode(json).validatedProjectorScale(), expected, json)
    }
  }

  /// M2 (ClownsharK adversarial review): an unknown `noise_type` is a 400
  /// naming the value and the valid set — it must never silently degrade to
  /// gaussian. Absent stays gaussian (the default, not a coercion).
  func testUnknownNoiseTypeIs400() throws {
    for name in ["gaussain", "perlin", "GAUSSIAN", "fractal "] {
      let payload = try decode(#"{"prompt":"x","noise_type":"\#(name)"}"#)
      XCTAssertThrowsError(try payload.validatedNoiseType(), name) { error in
        guard case WarmServerError.unknownNoiseType(let got, let valid) = error else {
          return XCTFail("expected unknownNoiseType for '\(name)', got \(error)")
        }
        XCTAssertEqual(got, name)
        XCTAssertEqual(Set(valid), Set(RES4LYFNoiseType.allCases.map(\.rawValue)))
        let response = WarmServer.errorResponse(for: error)
        XCTAssertEqual(response.status, 400)
        let body = bodyString(response)
        XCTAssertTrue(body.contains(name), body)
        for v in valid { XCTAssertTrue(body.contains(v), "400 body must list '\(v)': \(body)") }
      }
    }

    // Every real noise type resolves; absent stays gaussian.
    for kind in RES4LYFNoiseType.allCases {
      let payload = try decode(#"{"prompt":"x","noise_type":"\#(kind.rawValue)"}"#)
      XCTAssertEqual(try payload.validatedNoiseType(), kind, kind.rawValue)
    }
    XCTAssertEqual(try decode(#"{"prompt":"x"}"#).validatedNoiseType(), .gaussian)
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
