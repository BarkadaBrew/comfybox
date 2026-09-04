import XCTest
@testable import ZImage

/// Issue #22 — DyPE memory management: pre-flight checks and resolution caps.
/// PR #363 review round 1: the O(tokens²) attention term was deleted (the
/// fused `MLXFast.scaledDotProductAttention` kernel never materializes that
/// tensor) and the memory-budget check is advisory by default
/// (`ImageMemoryCapsConfig.enforceMemoryEstimate = false`) — see
/// `ImageMemoryPreflight.swift`'s header for the full rationale.
///
/// `ImageMemoryPreflight` is pure (no live memory probing, no model I/O), so
/// every test here injects explicit byte figures — never
/// `MemoryProbe.systemAvailableMemoryBytes()` or `ServerConfigStore.shared`
/// directly (see `GeneratePayload.validateImageMemoryPreflight`'s injectable
/// `caps`/`availableBytes` parameters).
final class ImageMemoryPreflightTests: XCTestCase {

  private var gb: UInt64 { 1024 * 1024 * 1024 }

  private func bodyJSON(_ response: HTTPResponse) throws -> [String: Any] {
    let object = try JSONSerialization.jsonObject(with: response.body)
    return try XCTUnwrap(object as? [String: Any])
  }

  // MARK: - estimateBytes: pinned regression points

  /// flux1 (Z-Image-Turbo) at the training resolution — DyPE is a no-op here
  /// (`ZImageTransformer2D.forward` only rescales when `hScale > 1.0`), so
  /// the with/without-DyPE estimates must be IDENTICAL.
  func testEstimateFlux1At1024IsPinnedAndDyPEInvariant() {
    let noDype = ImageMemoryPreflight.estimateBytes(width: 1024, height: 1024, family: .flux1, dype: false)
    let withDype = ImageMemoryPreflight.estimateBytes(width: 1024, height: 1024, family: .flux1, dype: true)
    XCTAssertEqual(noDype, 610_271_232)
    XCTAssertEqual(withDype, noDype, "1024px is the DyPE base resolution — hScale is exactly 1.0, never > 1.0")
  }

  /// flux1 at 2048px: DyPE now applies (tokens exceed the base-resolution
  /// token count) — the estimate must be STRICTLY larger than without DyPE.
  func testEstimateFlux1At2048PinnedWithAndWithoutDyPE() {
    let noDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .flux1, dype: false)
    let withDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .flux1, dype: true)
    XCTAssertEqual(noDype, 2_441_084_928)
    XCTAssertEqual(withDype, 2_445_475_840)
    XCTAssertGreaterThan(withDype, noDype, "DyPE recomputes a per-token frequency table above base resolution")
  }

  /// flux1 at 4096px. PR #363 review: with the O(tokens²) attention term
  /// removed, the estimate is now O(tokens) — 4x the pixels (2048→4096)
  /// should give roughly 4x the estimate, NOT the ~16x a quadratic term
  /// would have produced. Pinned as a regression guard in both directions:
  /// catches an accidental re-introduction of a quadratic term just as much
  /// as a formula that's gone flat.
  func testEstimateFlux1At4096PinnedAndScalesLinearlyNotQuadratically() {
    let at2048 = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .flux1, dype: false)
    let at4096 = ImageMemoryPreflight.estimateBytes(width: 4096, height: 4096, family: .flux1, dype: false)
    XCTAssertEqual(at4096, 9_764_339_712)
    let ratio = Double(at4096) / Double(at2048)
    XCTAssertEqual(ratio, 4.0, accuracy: 0.05, "O(tokens) formula: 4x the pixels should give ~4x the estimate, not ~16x")
  }

  /// krea2 (the production model, `intent.md`) is heavier than flux1 at the
  /// SAME resolution — a larger `features`/FFN width than Z-Image-Turbo.
  func testEstimateKrea2IsLargerThanFlux1AtSameResolution() {
    let flux1 = ImageMemoryPreflight.estimateBytes(width: 1024, height: 1024, family: .flux1, dype: false)
    let krea2 = ImageMemoryPreflight.estimateBytes(width: 1024, height: 1024, family: .krea2, dype: false)
    XCTAssertEqual(flux1, 610_271_232)
    XCTAssertEqual(krea2, 962_592_768)
    XCTAssertGreaterThan(krea2, flux1)
  }

  func testEstimateKrea2At2048IsPinned() {
    let noDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .krea2, dype: false)
    let withDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .krea2, dype: true)
    XCTAssertEqual(noDype, 3_850_371_072)
    XCTAssertEqual(withDype, 3_854_761_984)
  }

  /// C1, PR #363 review: the required sanity assertion — krea2@2048² must be
  /// comfortably under 12GB now that the O(tokens²) term is gone. (It was
  /// ~59GB before this fix, which alone would have refused every 2K krea2
  /// render even with krea2 resident, since ~75GB resident already eats most
  /// of a 128GB machine's headroom.)
  func testKrea2At2048SanityAssertionUnder12GB() {
    let estimate = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .krea2, dype: true)
    XCTAssertLessThan(estimate, 12 * gb, "krea2@2048² must stay well under 12GB — see PR #363 review C1")
  }

  /// Unrecognized families (flux2/fibo/chroma have no compiled-in
  /// transformer constants) must default to the LARGER (krea2) profile —
  /// never under-estimate. Distinct from `resolvedFamily`'s OWN fallback
  /// (I4: lighter, when `model` is unspecified) — this is about a family
  /// that WAS explicitly resolved to something this file cannot model.
  func testUnresolvedFamiliesDefaultToTheLargerKrea2Profile() {
    let krea2 = ImageMemoryPreflight.estimateBytes(width: 1536, height: 1536, family: .krea2, dype: false)
    for family: WarmModelFamily in [.flux2, .fibo, .chroma] {
      XCTAssertEqual(
        ImageMemoryPreflight.estimateBytes(width: 1536, height: 1536, family: family, dype: false),
        krea2, "\(family) must fall back to the krea2 profile, not under-estimate")
    }
  }

  func testEstimateIsZeroForNonPositiveDimensions() {
    XCTAssertEqual(ImageMemoryPreflight.estimateBytes(width: 0, height: 1024, family: .flux1, dype: false), 0)
    XCTAssertEqual(ImageMemoryPreflight.estimateBytes(width: 1024, height: -1, family: .flux1, dype: false), 0)
  }

  // MARK: - I6: overflow safety

  func testEstimateSaturatesRatherThanTrapsOnPathologicalDimensions() {
    // Beyond tokensPerAxisOverflowGuard (20,000 tokens/axis ≈ 320,000px) —
    // must return UInt64.max, never trap.
    let huge = ImageMemoryPreflight.estimateBytes(width: 10_000_000, height: 10_000_000, family: .krea2, dype: true)
    XCTAssertEqual(huge, UInt64.max)
  }

  func testDecideResolutionRefusesNonPositiveDimensionsWithoutTrapping() {
    let d = ImageMemoryPreflight.decideResolution(width: -5, height: 100, caps: .default)
    XCTAssertFalse(d.allow)
  }

  func testDecideResolutionRefusesPathologicallyLargeDimensionsWithoutTrapping() {
    // Int.max would overflow a plain `width * height` Int multiply — must
    // refuse cleanly via the overflow-safe UInt64 path, not trap.
    let d = ImageMemoryPreflight.decideResolution(width: Int.max, height: 2, caps: .default)
    XCTAssertFalse(d.allow)
  }

  func testMulSatSaturatesInsteadOfTrapping() {
    XCTAssertEqual(ImageMemoryPreflight.mulSat(UInt64.max, 2), UInt64.max)
    XCTAssertEqual(ImageMemoryPreflight.mulSat(3, 4), 12)
  }

  func testAddSatSaturatesInsteadOfTrapping() {
    XCTAssertEqual(ImageMemoryPreflight.addSat(UInt64.max, 1), UInt64.max)
    XCTAssertEqual(ImageMemoryPreflight.addSat(3, 4), 7)
  }

  // MARK: - decide(estimate:available:cap:)

  func testDecideAllowsWhenEstimateFitsUnderCap() {
    let d = ImageMemoryPreflight.decide(estimate: 10 * gb, available: 100 * gb, cap: 90 * gb)
    XCTAssertTrue(d.allow)
  }

  func testDecideAllowsAtExactCap() {
    let d = ImageMemoryPreflight.decide(estimate: 90 * gb, available: 100 * gb, cap: 90 * gb)
    XCTAssertTrue(d.allow, "an estimate exactly at the cap must be admitted, not refused")
  }

  func testDecideRefusesWhenEstimateExceedsCap() {
    let d = ImageMemoryPreflight.decide(estimate: 91 * gb, available: 100 * gb, cap: 90 * gb)
    XCTAssertFalse(d.allow)
    XCTAssertTrue(d.reason.contains("MB"), d.reason)
  }

  // MARK: - decideResolution(width:height:caps:)

  func testDecideResolutionAllowsWithinCaps() {
    let d = ImageMemoryPreflight.decideResolution(width: 2048, height: 2048, caps: .default)
    XCTAssertTrue(d.allow)
  }

  func testDecideResolutionAllowsExactlyAtLongEdgeAndPixelCap() {
    // 4096x4096: long edge == maxLongEdge AND pixels == maxPixels, both defaults.
    let caps = ImageMemoryCapsConfig()
    XCTAssertEqual(caps.maxLongEdge, 4096)
    XCTAssertEqual(caps.maxPixels, 4096 * 4096)
    let d = ImageMemoryPreflight.decideResolution(width: 4096, height: 4096, caps: caps)
    XCTAssertTrue(d.allow, "a request exactly at the cap must pass, not be refused")
  }

  func testDecideResolutionRefusesLongEdgeOverCap() {
    let d = ImageMemoryPreflight.decideResolution(width: 6000, height: 6000, caps: .default)
    XCTAssertFalse(d.allow)
    XCTAssertTrue(d.reason.contains("6000"), d.reason)
    XCTAssertTrue(d.reason.contains("4096"), d.reason)
  }

  func testDecideResolutionRefusesPixelsOverCapEvenUnderLongEdge() {
    // Long edge 4096 (at the cap) but a taller-than-square shape pushes
    // pixels past maxPixels.
    let d = ImageMemoryPreflight.decideResolution(width: 4096, height: 4097, caps: .default)
    XCTAssertFalse(d.allow)
  }

  // MARK: - resolvedFamily: I4 (PR #363 review)

  func testResolvedFamilyUsesExplicitModelRegardlessOfWarmFamily() {
    XCTAssertEqual(ImageMemoryPreflight.resolvedFamily(model: "krea2-raw", warmFamily: .flux1), .krea2)
    XCTAssertEqual(ImageMemoryPreflight.resolvedFamily(model: "z-image-turbo", warmFamily: .krea2), .flux1)
  }

  func testResolvedFamilyUsesWarmFamilyWhenModelIsNil() {
    XCTAssertEqual(ImageMemoryPreflight.resolvedFamily(model: nil, warmFamily: .krea2), .krea2)
    XCTAssertEqual(ImageMemoryPreflight.resolvedFamily(model: nil, warmFamily: .flux1), .flux1)
  }

  func testResolvedFamilyFallsBackToLighterFlux1WhenNothingIsKnown() {
    XCTAssertEqual(ImageMemoryPreflight.resolvedFamily(model: nil, warmFamily: nil), .flux1,
                   "I4: never default an unspecified model to the heaviest profile")
  }

  // MARK: - validate(...): the combined gate, returns Outcome, throws WarmServerError

  func testValidateThrowsResolutionCapBeforeAnyMemoryProbing() {
    XCTAssertThrowsError(
      try ImageMemoryPreflight.validate(
        width: 6000, height: 6000, family: .krea2, dype: true,
        caps: .default, availableBytes: 200 * gb)
    ) { error in
      guard case let .imageMemoryPreflightRefused(code, reason, estimate, available, cap) =
        error as? WarmServerError else {
        return XCTFail("expected .imageMemoryPreflightRefused, got \(error)")
      }
      XCTAssertEqual(code, "resolution_cap")
      XCTAssertNil(estimate, "resolution cap refuses before the byte estimate is even computed")
      XCTAssertNil(available)
      XCTAssertNotNil(cap)
      XCTAssertTrue(reason.contains("6000"), reason)
    }
  }

  /// C1b: with `enforceMemoryEstimate` false (the default), an estimate that
  /// exceeds the budget does NOT throw — it returns an `Outcome` with
  /// `withinBudget == false` so the caller can log/surface it.
  func testValidateDoesNotThrowWhenMemoryEstimateExceedsBudgetByDefault() throws {
    var caps = ImageMemoryCapsConfig.default
    XCTAssertFalse(caps.enforceMemoryEstimate, "advisory by default")
    caps.maxLongEdge = 4096
    caps.maxPixels = 4096 * 4096
    let outcome = try ImageMemoryPreflight.validate(
      width: 2048, height: 2048, family: .krea2, dype: true,
      caps: caps, availableBytes: 1 * gb)
    XCTAssertFalse(outcome.withinBudget)
    XCTAssertEqual(outcome.estimateBytes, 3_854_761_984)
    XCTAssertEqual(outcome.availableBytes, 1 * gb)
    XCTAssertFalse(outcome.reason.isEmpty)
  }

  /// The flip side: with `enforceMemoryEstimate` true, the SAME
  /// over-budget request throws.
  func testValidateThrowsInsufficientMemoryOnlyWhenEnforced() {
    var caps = ImageMemoryCapsConfig.default
    caps.enforceMemoryEstimate = true
    XCTAssertThrowsError(
      try ImageMemoryPreflight.validate(
        width: 2048, height: 2048, family: .krea2, dype: true,
        caps: caps, availableBytes: 1 * gb)
    ) { error in
      guard case let .imageMemoryPreflightRefused(code, _, estimate, available, cap) =
        error as? WarmServerError else {
        return XCTFail("expected .imageMemoryPreflightRefused, got \(error)")
      }
      XCTAssertEqual(code, "insufficient_memory")
      XCTAssertEqual(estimate, 3_854_761_984)
      XCTAssertEqual(available, 1 * gb)
      XCTAssertEqual(cap, UInt64(Double(1 * gb) * 0.9))
    }
  }

  /// The test plan's second half: 2048² on a machine with real M3 Max-class
  /// headroom must pass (allow, and be within budget) both by default and
  /// when enforced.
  func testValidateAllows2048SquareWithGenerousHeadroom() throws {
    let outcome = try ImageMemoryPreflight.validate(
      width: 2048, height: 2048, family: .krea2, dype: true,
      caps: .default, availableBytes: 100 * gb)
    XCTAssertTrue(outcome.withinBudget)
  }

  /// The test plan's first half, as a deterministic unit test: 6000x6000
  /// refuses immediately, with no dependence on live machine memory or on
  /// `enforceMemoryEstimate` — the resolution cap is unconditional.
  func testValidateRefuses6000SquareRegardlessOfAvailableMemoryOrEnforcement() {
    for available: UInt64 in [0, 1 * gb, 1000 * gb] {
      for enforce in [false, true] {
        var caps = ImageMemoryCapsConfig.default
        caps.enforceMemoryEstimate = enforce
        XCTAssertThrowsError(
          try ImageMemoryPreflight.validate(
            width: 6000, height: 6000, family: .krea2, dype: false,
            caps: caps, availableBytes: available))
      }
    }
  }

  // MARK: - HTTP shape: REST keeps 413, additive JSON fields

  func testResolutionCapRefusalMapsTo413WithAdditiveFields() throws {
    var caught: Error?
    do {
      try ImageMemoryPreflight.validate(
        width: 6000, height: 6000, family: .krea2, dype: false,
        caps: .default, availableBytes: 100 * gb)
    } catch {
      caught = error
    }
    let error = try XCTUnwrap(caught)
    let response = WarmServer.errorResponse(for: error)
    XCTAssertEqual(response.status, 413)
    let json = try bodyJSON(response)
    XCTAssertEqual(json["success"] as? Bool, false)
    XCTAssertEqual(json["error_code"] as? String, "resolution_cap")
    XCTAssertNotNil(json["error"] as? String)
    XCTAssertNil(json["estimate_bytes"], "resolution-cap refusal never computed an estimate")
  }

  func testInsufficientMemoryRefusalMapsTo413NamingAllThreeNumbersWhenEnforced() throws {
    var caps = ImageMemoryCapsConfig.default
    caps.enforceMemoryEstimate = true
    var caught: Error?
    do {
      try ImageMemoryPreflight.validate(
        width: 2048, height: 2048, family: .krea2, dype: true,
        caps: caps, availableBytes: 1 * gb)
    } catch {
      caught = error
    }
    let error = try XCTUnwrap(caught)
    let response = WarmServer.errorResponse(for: error)
    XCTAssertEqual(response.status, 413)
    let json = try bodyJSON(response)
    XCTAssertEqual(json["error_code"] as? String, "insufficient_memory")
    let estimate = try XCTUnwrap(json["estimate_bytes"] as? NSNumber)
    let available = try XCTUnwrap(json["available_bytes"] as? NSNumber)
    let cap = try XCTUnwrap(json["cap_bytes"] as? NSNumber)
    XCTAssertEqual(estimate.uint64Value, 3_854_761_984)
    XCTAssertEqual(available.uint64Value, 1 * gb)
    XCTAssertEqual(cap.uint64Value, UInt64(Double(1 * gb) * 0.9))
  }

  /// M9 (PR #363 review): the bridge downgrades the SAME refusal to 400.
  func testBridgeErrorResponseUses400ForTheSameRefusal() throws {
    var caught: Error?
    do {
      try ImageMemoryPreflight.validate(
        width: 6000, height: 6000, family: .krea2, dype: false,
        caps: .default, availableBytes: 100 * gb)
    } catch {
      caught = error
    }
    let error = try XCTUnwrap(caught as? WarmServerError)
    let response = ComfyBridge.bridgeErrorResponse(for: error)
    XCTAssertEqual(response.status, 400)
    let json = try bodyJSON(response)
    XCTAssertEqual(json["error_code"] as? String, "resolution_cap")
  }

  func testBridgeErrorResponseDefersToSharedMappingForOtherErrors() {
    let response = ComfyBridge.bridgeErrorResponse(for: .mutuallyExclusive("x and y disagree"))
    XCTAssertEqual(response.status, 400, "mutuallyExclusive is already a 400 in the shared mapping")
  }

  /// Pre-existing refusals (no `imageMemoryPreflightRefused` involved) must
  /// stay byte-identical: no `error_code`/`estimate_bytes`/... keys at all —
  /// `Encodable`'s `encodeIfPresent` omits nil optionals rather than encoding
  /// `null`.
  func testPreExistingErrorPayloadShapeIsUnchanged() throws {
    let response = WarmServer.errorResponse(for: WarmServerError.mutuallyExclusive("x and y disagree"))
    let json = try bodyJSON(response)
    XCTAssertNil(json["error_code"])
    XCTAssertNil(json["estimate_bytes"])
    XCTAssertNil(json["available_bytes"])
    XCTAssertNil(json["cap_bytes"])
    XCTAssertEqual(Set(json.keys), Set(["success", "error"]))
  }

  // MARK: - GeneratePayload.validateImageMemoryPreflight() wiring

  func testPayloadWiringUsesRequestWidthHeightAndDefaultsFamilyToLighterProfile() {
    var payload = GeneratePayload(prompt: "x", width: 6000, height: 6000)
    XCTAssertThrowsError(
      try payload.validateImageMemoryPreflight(caps: .default, availableBytes: 200 * gb)
    ) { error in
      guard case let .imageMemoryPreflightRefused(code, _, _, _, _) = error as? WarmServerError else {
        return XCTFail("expected .imageMemoryPreflightRefused, got \(error)")
      }
      XCTAssertEqual(code, "resolution_cap")
    }
  }

  func testPayloadWiringHonoursExplicitModelFamily() throws {
    var payload = GeneratePayload(prompt: "x", width: 2048, height: 2048, model: "krea2-raw")
    let outcome = try payload.validateImageMemoryPreflight(caps: .default, availableBytes: 100 * gb)
    XCTAssertEqual(outcome?.estimateBytes, 3_854_761_984, "explicit krea2 model must use the krea2 profile")
  }

  func testPayloadWiringUsesWarmFamilyWhenModelIsNil() throws {
    var payload = GeneratePayload(prompt: "x", width: 2048, height: 2048)
    let outcome = try payload.validateImageMemoryPreflight(warmFamily: .krea2, caps: .default, availableBytes: 100 * gb)
    XCTAssertEqual(outcome?.estimateBytes, 3_854_761_984)
  }

  func testPayloadWiringSkipsWhenDimensionsAreOmitted() throws {
    // No width/height at all — resolves to the small engine default later in
    // the pipeline, never DyPE territory. Must not throw even under a
    // starved injected budget, and must return nil (skipped).
    var payload = GeneratePayload(prompt: "x")
    let outcome = try payload.validateImageMemoryPreflight(caps: .default, availableBytes: 0)
    XCTAssertNil(outcome)
  }

  func testPayloadWiringStampsMemoryFieldsOnSelf() throws {
    var payload = GeneratePayload(prompt: "x", width: 2048, height: 2048)
    XCTAssertNil(payload.memoryEstimateBytes)
    _ = try payload.validateImageMemoryPreflight(warmFamily: .flux1, caps: .default, availableBytes: 100 * gb)
    XCTAssertEqual(payload.memoryEstimateBytes, 2_445_475_840)
    XCTAssertEqual(payload.memoryAvailableBytes, 100 * gb)
  }

  func testPayloadWiringLogsAWarningWhenAdvisoryBudgetIsExceeded() throws {
    var payload = GeneratePayload(prompt: "x", width: 2048, height: 2048)
    var logged: [String] = []
    _ = try payload.validateImageMemoryPreflight(
      warmFamily: .krea2, caps: .default, availableBytes: 1 * gb, log: { logged.append($0) })
    XCTAssertTrue(logged.contains { $0.contains("advisory") }, "expected an advisory warning to be logged: \(logged)")
  }

  func testPayloadWiringDoesNotLogWhenWithinBudget() throws {
    var payload = GeneratePayload(prompt: "x", width: 1024, height: 1024)
    var logged: [String] = []
    _ = try payload.validateImageMemoryPreflight(
      warmFamily: .flux1, caps: .default, availableBytes: 100 * gb, log: { logged.append($0) })
    XCTAssertTrue(logged.isEmpty, "no warning expected when within budget: \(logged)")
  }

  // MARK: - C3: wiring tests through the ACTUAL route function

  private func makePresetStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-memory-preflight-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  /// C3: through the REAL `WarmServer.decodedGeneratePayload` (the exact
  /// static function `/v1/generate`/`/v1/generate/async` call, per
  /// `GeneratePresetRouteTests`) — a 6000×6000 body throws
  /// `.imageMemoryPreflightRefused(code: "resolution_cap", ...)`.
  /// Deleting `validateImageMemoryPreflight`'s call site inside
  /// `decodedGeneratePayload` makes this test fail (no throw).
  func testDecodedGeneratePayloadRefuses6000SquareAsResolutionCap() throws {
    let store = try makePresetStore()
    XCTAssertThrowsError(
      try WarmServer.decodedGeneratePayload(
        from: Data(#"{"prompt":"x","width":6000,"height":6000}"#.utf8),
        store: store, configuration: configuration)
    ) { error in
      guard case .imageMemoryPreflightRefused(let code, _, _, _, _) = error as? WarmServerError else {
        return XCTFail("expected .imageMemoryPreflightRefused, got \(error)")
      }
      XCTAssertEqual(code, "resolution_cap")
    }
  }

  /// C3: a 2048² body passes through the same real function (advisory
  /// budget check never throws by default).
  func testDecodedGeneratePayloadAccepts2048Square() throws {
    let store = try makePresetStore()
    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","width":2048,"height":2048}"#.utf8),
      store: store, configuration: configuration)
    XCTAssertEqual(payload.width, 2048)
    XCTAssertNotNil(payload.memoryEstimateBytes, "the preflight ran and stamped the estimate")
  }

  /// C2: `gateSubmission: false` (crash-recovery replay's own posture) must
  /// skip the gate entirely — the SAME 6000×6000 body that throws under the
  /// default `gateSubmission: true` must pass cleanly here.
  func testDecodedGeneratePayloadWithGateSubmissionFalseNeverRefusesReplay() throws {
    let store = try makePresetStore()
    let payload = try WarmServer.decodedGeneratePayload(
      from: Data(#"{"prompt":"x","width":6000,"height":6000}"#.utf8),
      store: store, configuration: configuration, gateSubmission: false)
    XCTAssertEqual(payload.width, 6000)
    XCTAssertNil(payload.memoryEstimateBytes, "the preflight did not run at all on replay")
  }
}
