import XCTest

@testable import ZImage

/// WP-E4 (FDD-krea2-raw-recipe §3.4, D22, D25) — fail-loud sampler / sigma
/// schedule name resolution. Replaces the `parseSchedulerKind` /
/// `parseSigmaScheduleKind` coercions that turned every unknown name into
/// euler / flow silently.
final class SamplerNameResolutionTests: XCTestCase {

  // MARK: - AC-16: the alias table survives, and the aliases are visible

  func testAliasTable() throws {
    // Sampler aliases (ComfyUI / RES4LYF spellings → SchedulerKind).
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("res_2s"), .res2s)
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("dpmpp_2m"), .dpmplusplus2m)
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("dpmpp_2s_ancestral"), .dpmplusplus2sa)
    // The enum's own raw values resolve too.
    for kind in SchedulerKind.allCases {
      XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind(kind.rawValue), kind, kind.rawValue)
    }
    // RES4LYF UI prefixes are stripped so a workflow value pastes verbatim.
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("exponential/res_2s"), .res2s)
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("multistep/dpmpp_2m"), .dpmplusplus2m)

    // Sigma schedule aliases.
    XCTAssertEqual(try RecipeNameResolver.resolveSigmaScheduleKind("beta57"), .beta57)
    for kind in SigmaScheduleKind.allCases {
      XCTAssertEqual(try RecipeNameResolver.resolveSigmaScheduleKind(kind.rawValue), kind, kind.rawValue)
    }
    // D22 (reversed from v1): the four ComfyUI schedule names stay as declared
    // aliases of `.flow` — Krita's built-in styles send them by default.
    for name in ["normal", "simple", "sgm_uniform", "ddim_uniform"] {
      XCTAssertEqual(try RecipeNameResolver.resolveSigmaScheduleKind(name), .flow, name)
    }

    // Absent is absent — never a default smuggled in by the resolver.
    XCTAssertNil(try RecipeNameResolver.resolveSchedulerKind(nil))
    XCTAssertNil(try RecipeNameResolver.resolveSigmaScheduleKind(nil))
  }

  /// The alias is visible, not silent: the resolved names carry both what ran
  /// and what was requested (the record's `sigma_schedule_requested`, D22).
  func testResolvedNamesCarryTheRequestedString() throws {
    let payload = try decode(#"{"prompt":"x","scheduler":"dpmpp_2m","sigma_schedule":"normal"}"#)
    let names = try payload.validateRecipeNames()
    XCTAssertEqual(names.scheduler, .dpmplusplus2m)
    XCTAssertEqual(names.schedulerRequested, "dpmpp_2m")
    XCTAssertEqual(names.sigmaSchedule, .flow)
    XCTAssertEqual(names.sigmaScheduleRequested, "normal")

    let bare = try decode(#"{"prompt":"x"}"#)
    let bareNames = try bare.validateRecipeNames()
    XCTAssertNil(bareNames.scheduler)
    XCTAssertNil(bareNames.schedulerRequested)
    XCTAssertNil(bareNames.sigmaSchedule)
    XCTAssertNil(bareNames.sigmaScheduleRequested)
  }

  // MARK: - AC-15 / AC-28: unknown names throw, naming the value and the valid set

  func testUnknownSamplerThrowsNamingValueAndValidSet() {
    for name in ["uni_pc", "res_5s", "dpmpp_2m_sde", "lcm", "euler_ancestral", "uni_pc_bh2", "dpmpp_sde_gpu", "dpmpp_2m_sde_gpu"] {
      XCTAssertThrowsError(try RecipeNameResolver.resolveSchedulerKind(name), name) { error in
        guard case WarmServerError.unknownSampler(let offending, let valid) = error else {
          return XCTFail("expected unknownSampler for '\(name)', got \(error)")
        }
        XCTAssertEqual(offending, name)
        XCTAssertEqual(Set(valid), Set(RecipeNameResolver.validSamplerNames))
        XCTAssertTrue(valid.contains("euler"))
        XCTAssertTrue(valid.contains("res_2s"))
        let message = error.localizedDescription
        XCTAssertTrue(message.contains("'\(name)'"), "message must name the value: \(message)")
        for v in valid { XCTAssertTrue(message.contains(v), "message must list '\(v)': \(message)") }
      }
    }
  }

  func testUnknownSigmaScheduleThrowsNamingValueAndValidSet() {
    for name in ["ays", "linear", "sigmoid_offset", "bong_tangent_missing"] {
      XCTAssertThrowsError(try RecipeNameResolver.resolveSigmaScheduleKind(name), name) { error in
        guard case WarmServerError.unknownSigmaSchedule(let offending, let valid) = error else {
          return XCTFail("expected unknownSigmaSchedule for '\(name)', got \(error)")
        }
        XCTAssertEqual(offending, name)
        XCTAssertEqual(Set(valid), Set(RecipeNameResolver.validSigmaScheduleNames))
        XCTAssertTrue(valid.contains("flow"))
        XCTAssertTrue(valid.contains("normal"), "kept aliases are part of the valid set")
        let message = error.localizedDescription
        XCTAssertTrue(message.contains("'\(name)'"), "message must name the value: \(message)")
      }
    }
  }

  /// The payload-level validation surfaces the same errors: a request body
  /// with an unknown name never yields a payload.
  func testPayloadValidationRejectsUnknownNames() throws {
    let p1 = try decode(#"{"prompt":"x","scheduler":"uni_pc"}"#)
    XCTAssertThrowsError(try p1.validateRecipeNames()) { error in
      guard case WarmServerError.unknownSampler(let name, _) = error else { return XCTFail("\(error)") }
      XCTAssertEqual(name, "uni_pc")
    }
    let p2 = try decode(#"{"prompt":"x","sigma_schedule":"ays"}"#)
    XCTAssertThrowsError(try p2.validateRecipeNames()) { error in
      guard case WarmServerError.unknownSigmaSchedule(let name, _) = error else { return XCTFail("\(error)") }
      XCTAssertEqual(name, "ays")
    }
  }

  /// The valid sets are exactly the enum cases plus the declared aliases —
  /// nothing the resolver accepts is missing from the message, and nothing in
  /// the message is unresolvable.
  func testValidSetsAreClosedOverTheResolver() throws {
    let samplers = RecipeNameResolver.validSamplerNames
    XCTAssertEqual(Set(samplers),
                   Set(SchedulerKind.allCases.map(\.rawValue)).union(RecipeNameResolver.samplerAliases.keys))
    for name in samplers { XCTAssertNotNil(try RecipeNameResolver.resolveSchedulerKind(name), name) }

    let schedules = RecipeNameResolver.validSigmaScheduleNames
    XCTAssertEqual(Set(schedules),
                   Set(SigmaScheduleKind.allCases.map(\.rawValue)).union(RecipeNameResolver.sigmaScheduleAliases.keys))
    for name in schedules { XCTAssertNotNil(try RecipeNameResolver.resolveSigmaScheduleKind(name), name) }
  }

  // MARK: - AC-16a: every Krita default either resolves or is refused by name

  /// Values read from Krita AI Diffusion's `style.py` (`_sampler_map` /
  /// `_scheduler_map`, verified on disk 2026-08-22). Each either resolves or
  /// throws an error that names it — never becomes euler/flow silently. The
  /// bridge end-to-end half of this criterion is `BridgeKrea2VariantTests`
  /// (WP-E19).
  func testKritaStyleMatrix() {
    let kritaSamplers = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde_gpu", "dpmpp_sde_gpu", "uni_pc_bh2", "lcm", "ddim"]
    let kritaSchedules = ["normal", "karras", "ddim_uniform", "sgm_uniform"]
    var resolvedSamplers: [String] = []
    for name in kritaSamplers {
      do {
        let kind = try RecipeNameResolver.resolveSchedulerKind(name)
        XCTAssertNotNil(kind)
        resolvedSamplers.append(name)
      } catch {
        guard case WarmServerError.unknownSampler(let offending, _) = error else {
          return XCTFail("'\(name)' threw something other than unknownSampler: \(error)")
        }
        XCTAssertEqual(offending, name)
      }
    }
    // The default style (Euler → euler / normal) and the DPM++ 2M and DDIM
    // styles resolve; the rest are refused by name.
    XCTAssertEqual(Set(resolvedSamplers), ["euler", "dpmpp_2m", "ddim"])
    for name in kritaSchedules {
      XCTAssertNoThrow(try RecipeNameResolver.resolveSigmaScheduleKind(name), "Krita schedule '\(name)' must resolve")
    }
  }

  // MARK: - helpers

  /// Mirrors WarmServer.decode(_:from:).
  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }
}
