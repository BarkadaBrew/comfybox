// FamilyRecipeMatrixTests.swift — K-FIX-1 / Codex engine review I5.
//
// Name resolution (WP-E4) is family-agnostic by design: a name is wrong for
// EVERY family, so `decodedGeneratePayload` rejects unknown spellings up
// front. What was missing is the other half — a KNOWN name that the family
// about to render cannot honour. The only gate was
// `validateTableauSampler`, which refused N-row tableaus outside Krea 2 and
// nothing else, so:
//
//   * Chroma has native `.heun` and `.beta` (`ChromaSchedulerType`) and
//     `runChromaGenerate` passed NEITHER — `scheduler: "heun"` resolved,
//     validated, and rendered Euler pixels under the name "heun".
//   * Flux 2 and FIBO run fixed flow-match Euler loops and accepted
//     `dpmpp_2m`, `karras`, `beta`… rendering Euler under every one.
//
// One matrix now answers both questions at the dispatch point: does this
// family implement this pair, and if so what does it map to. These tests are
// generated from that matrix — every family × every sampler × every schedule
// — so a new `SchedulerKind` or family must declare itself rather than
// inheriting a silent pass.

import XCTest

@testable import ZImage

final class FamilyRecipeMatrixTests: XCTestCase {

  private func names(_ sampler: SchedulerKind?, _ schedule: SigmaScheduleKind?)
    -> ResolvedRecipeNames
  {
    ResolvedRecipeNames(
      scheduler: sampler, schedulerRequested: sampler?.rawValue,
      sigmaSchedule: schedule, sigmaScheduleRequested: schedule?.rawValue)
  }

  // MARK: - The whole table, family × sampler × schedule

  /// Table-driven: every combination is either accepted or 400, and the
  /// verdict comes from the matrix rather than from a hand-listed set. The
  /// assertion is that acceptance is EXACTLY the declared capability — no
  /// family silently tolerates a name it does not implement.
  func testEveryFamilySamplerScheduleCombinationMatchesTheDeclaredCapability() {
    for family in WarmModelFamily.allCases {
      let capability = FamilyRecipeMatrix.capability(for: family)
      for sampler in SchedulerKind.allCases {
        for schedule in SigmaScheduleKind.allCases {
          let error = FamilyRecipeMatrix.validate(names(sampler, schedule), family: family)
          let expected = capability.samplers.contains(sampler)
            && capability.sigmaSchedules.contains(schedule)
            && capability.isPairSupported(sampler: sampler, schedule: schedule)
          XCTAssertEqual(
            error == nil, expected,
            "\(family.rawValue) + \(sampler.rawValue) + \(schedule.rawValue): "
              + "expected \(expected ? "accepted" : "400"), got \(error == nil ? "accepted" : "400")")
        }
      }
    }
  }

  /// An absent field is absent: a request that names neither is accepted by
  /// every family (that is the default path every existing caller takes).
  func testAbsentFieldsAreAcceptedEverywhere() {
    for family in WarmModelFamily.allCases {
      XCTAssertNil(FamilyRecipeMatrix.validate(names(nil, nil), family: family), family.rawValue)
    }
  }

  /// Every refusal names the offending value AND the family — the WP-E4
  /// posture, so a 400 is actionable without reading the source.
  func testEveryRefusalNamesTheValueAndTheFamily() {
    for family in WarmModelFamily.allCases {
      for sampler in SchedulerKind.allCases {
        for schedule in SigmaScheduleKind.allCases {
          guard let error = FamilyRecipeMatrix.validate(names(sampler, schedule), family: family)
          else { continue }
          let response = WarmServer.errorResponse(for: error)
          XCTAssertEqual(response.status, 400)
          let body = String(decoding: response.body, as: UTF8.self)
          XCTAssertTrue(body.contains(family.rawValue), body)
          XCTAssertTrue(
            body.contains(sampler.rawValue) || body.contains(schedule.rawValue), body)
        }
      }
    }
  }

  // MARK: - The specific silences Codex found

  /// Chroma implements heun and beta natively and used to ignore both.
  func testChromaHonoursItsNativeHeunAndBeta() throws {
    XCTAssertNil(FamilyRecipeMatrix.validate(names(.heun, .flow), family: .chroma))
    XCTAssertEqual(
      FamilyRecipeMatrix.chromaSchedulerType(sampler: .heun, schedule: .flow), .heun)
    XCTAssertEqual(
      FamilyRecipeMatrix.chromaSchedulerType(sampler: .heun, schedule: .beta), .beta)
    XCTAssertEqual(
      FamilyRecipeMatrix.chromaSchedulerType(sampler: nil, schedule: nil), .euler)
    XCTAssertEqual(
      FamilyRecipeMatrix.chromaSchedulerType(sampler: .euler, schedule: .flow), .euler)
  }

  /// Chroma's `.beta` IS beta-timesteps + heun stepping (`ChromaPipeline`
  /// sets `useHeun` for it), so `beta` with the euler sampler has no honest
  /// mapping — a 400 naming what it needs, never a silent upgrade to heun.
  func testChromaBetaWithEulerIsRefusedRatherThanSilentlyHeun() throws {
    let error = try XCTUnwrap(
      FamilyRecipeMatrix.validate(names(.euler, .beta), family: .chroma))
    XCTAssertNil(FamilyRecipeMatrix.chromaSchedulerType(sampler: .euler, schedule: .beta))
    let body = String(decoding: WarmServer.errorResponse(for: error).body, as: UTF8.self)
    XCTAssertTrue(body.contains("beta"), body)
    XCTAssertTrue(body.contains("heun"), body)
  }

  /// Flux 2 / FIBO run fixed flow-match Euler loops. `euler` (and the default
  /// `flow`) pass — that is what the ComfyUI bridge sends on every render —
  /// and everything else is a 400 instead of Euler under another name.
  func testFixedLoopFamiliesAcceptOnlyWhatTheyRun() {
    for family in [WarmModelFamily.flux2, .fibo] {
      XCTAssertNil(FamilyRecipeMatrix.validate(names(.euler, .flow), family: family))
      XCTAssertNil(FamilyRecipeMatrix.validate(names(nil, nil), family: family))
      XCTAssertNotNil(FamilyRecipeMatrix.validate(names(.heun, nil), family: family))
      XCTAssertNotNil(FamilyRecipeMatrix.validate(names(.dpmplusplus2m, nil), family: family))
      XCTAssertNotNil(FamilyRecipeMatrix.validate(names(.res2s, nil), family: family))
      XCTAssertNotNil(FamilyRecipeMatrix.validate(names(nil, .karras), family: family))
      XCTAssertNotNil(FamilyRecipeMatrix.validate(names(nil, .beta), family: family))
    }
  }

  /// E13's `validateTableauSampler` is now one ROW of the matrix, not a
  /// sibling gate: N-row tableaus run under the Krea 2 loop and nowhere else.
  func testTableauSamplersAreOneRowOfTheMatrix() {
    for sampler in SchedulerKind.allCases where sampler.isNRowTableau {
      XCTAssertNil(
        FamilyRecipeMatrix.validate(names(sampler, .krea2), family: .krea2),
        "\(sampler.rawValue) runs on krea2")
      for family in WarmModelFamily.allCases where family != .krea2 {
        XCTAssertNotNil(
          FamilyRecipeMatrix.validate(names(sampler, nil), family: family),
          "\(sampler.rawValue) must be refused on \(family.rawValue)")
      }
    }
  }

  /// Z-Image (flux1) keeps every sampler and schedule it drives through
  /// `SchedulerFactory` today — including `ddim`, whose `eta` is a shipped
  /// Z-Image parameter (AC-28's regression).
  func testZImageKeepsItsShippedRecipeSurface() {
    for sampler in [SchedulerKind.euler, .heun, .dpmplusplus2m, .dpmplusplus2sa, .deis, .ddim, .res2s] {
      XCTAssertNil(
        FamilyRecipeMatrix.validate(names(sampler, .flow), family: .flux1), sampler.rawValue)
    }
    for schedule in [SigmaScheduleKind.flow, .karras, .exponential, .beta, .beta57] {
      XCTAssertNil(
        FamilyRecipeMatrix.validate(names(.euler, schedule), family: .flux1), schedule.rawValue)
    }
  }

  // MARK: - Advertised options come from the same matrix

  /// The bridge's `/object_info` lists are served before a model is chosen,
  /// so they advertise the UNION — but the union is computed from the matrix,
  /// not maintained beside it, so a name can never be advertised that no
  /// family implements.
  func testAdvertisedListsAreExactlyTheMatrixUnion() {
    let samplerUnion = Set(
      WarmModelFamily.allCases.flatMap { FamilyRecipeMatrix.capability(for: $0).samplers })
    XCTAssertEqual(samplerUnion, Set(SchedulerKind.allCases),
                   "every SchedulerKind must be implemented by at least one family")
    XCTAssertEqual(
      Set(FamilyRecipeMatrix.advertisedSamplerNames(for: nil)),
      Set(RecipeNameResolver.advertisedSamplerNames))

    let scheduleUnion = Set(
      WarmModelFamily.allCases.flatMap { FamilyRecipeMatrix.capability(for: $0).sigmaSchedules })
    XCTAssertEqual(scheduleUnion, Set(SigmaScheduleKind.allCases),
                   "every SigmaScheduleKind must be implemented by at least one family")
  }

  /// Per-family advertisement is the same table read the other way.
  func testPerFamilyAdvertisedNamesMatchTheCapability() {
    for family in WarmModelFamily.allCases {
      let capability = FamilyRecipeMatrix.capability(for: family)
      for name in FamilyRecipeMatrix.advertisedSamplerNames(for: family) {
        let kind = try? RecipeNameResolver.resolveSchedulerKind(name)
        XCTAssertTrue(
          capability.samplers.contains(try! XCTUnwrap(kind)),
          "\(family.rawValue) advertises '\(name)' it does not implement")
      }
      for name in FamilyRecipeMatrix.advertisedSigmaScheduleNames(for: family) {
        let kind = try? RecipeNameResolver.resolveSigmaScheduleKind(name)
        XCTAssertTrue(
          capability.sigmaSchedules.contains(try! XCTUnwrap(kind)),
          "\(family.rawValue) advertises '\(name)' it does not implement")
      }
    }
  }

  // MARK: - Public desktop option facade

  /// The desktop selectors consume the public facade, not a copied list. Pin
  /// that each family sees exactly the matrix capability and that model paths
  /// are classified the same way as `/health` family names.
  func testPublicSamplingCatalogUsesTheFamilyMatrix() {
    XCTAssertEqual(SamplingRecipeCatalog.canonicalFamily("krea2"), "krea2")
    XCTAssertEqual(
      SamplingRecipeCatalog.canonicalFamily("/models/krea2_raw_bf16.safetensors"),
      "krea2")
    XCTAssertEqual(SamplingRecipeCatalog.canonicalFamily("Tongyi-MAI/Z-Image-Turbo"), "flux1")

    for family in WarmModelFamily.allCases {
      let capability = FamilyRecipeMatrix.capability(for: family)
      XCTAssertEqual(
        SamplingRecipeCatalog.samplerNames(forModelFamily: family.rawValue),
        FamilyRecipeMatrix.supportedSamplerNames(for: family),
        family.rawValue)
      XCTAssertEqual(
        Set(SamplingRecipeCatalog.sigmaScheduleNames(forModelFamily: family.rawValue)),
        Set(SigmaScheduleKind.allCases.filter {
          capability.sigmaSchedules.contains($0)
            && capability.isPairSupported(sampler: .euler, schedule: $0)
        }.map(\.rawValue)),
        family.rawValue)
    }

    XCTAssertEqual(
      SamplingRecipeCatalog.defaultSigmaScheduleName(forModelFamily: "krea2"), "krea2")
    XCTAssertEqual(
      SamplingRecipeCatalog.defaultSigmaScheduleName(forModelFamily: "flux1"), "flow")
  }

  /// Chroma is the one family with a pair constraint: beta is available only
  /// after choosing Heun. The UI list and its validation must agree with the
  /// render gate so an advertised selection never becomes a later 400.
  func testPublicSamplingCatalogFiltersUnsupportedPairs() {
    XCTAssertFalse(
      SamplingRecipeCatalog.sigmaScheduleNames(forModelFamily: "chroma").contains("beta"))
    XCTAssertTrue(
      SamplingRecipeCatalog.sigmaScheduleNames(
        forModelFamily: "chroma", sampler: "heun").contains("beta"))
    XCTAssertFalse(
      SamplingRecipeCatalog.supports(
        sampler: "euler", sigmaSchedule: "beta", forModelFamily: "chroma"))
    XCTAssertTrue(
      SamplingRecipeCatalog.supports(
        sampler: "heun", sigmaSchedule: "beta", forModelFamily: "chroma"))
    XCTAssertFalse(
      SamplingRecipeCatalog.supports(
        sampler: "uni_pc", sigmaSchedule: nil, forModelFamily: "krea2"))
  }
}
