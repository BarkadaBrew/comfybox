import XCTest
@testable import ZImage

/// Issue #22 — DyPE memory management: pre-flight checks and resolution caps.
///
/// `ImageMemoryPreflight` is pure (no live memory probing, no model I/O), so
/// every test here injects explicit byte figures — never
/// `MemoryProbe.systemAvailableMemoryBytes()` or `ServerConfigStore.shared`
/// directly (see `GeneratePayload.validateImageMemoryPreflight`'s injectable
/// `caps`/`availableBytes` parameters).
final class ImageMemoryPreflightTests: XCTestCase {

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
    XCTAssertEqual(noDype, 3_611_295_744)
    XCTAssertEqual(withDype, noDype, "1024px is the DyPE base resolution — hScale is exactly 1.0, never > 1.0")
  }

  /// flux1 at 2048px: DyPE now applies (tokens exceed the base-resolution
  /// token count) — the estimate must be STRICTLY larger than without DyPE,
  /// pinned to the recomputed-frequency-table + retained-position-ids delta.
  func testEstimateFlux1At2048PinnedWithAndWithoutDyPE() {
    let noDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .flux1, dype: false)
    let withDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .flux1, dype: true)
    XCTAssertEqual(noDype, 38_604_374_016)
    XCTAssertEqual(withDype, 38_608_764_928)
    XCTAssertGreaterThan(withDype, noDype, "DyPE recomputes a per-token frequency table above base resolution")
  }

  /// flux1 at 4096px — the O(tokens²) attention term must dominate: going
  /// from 2048→4096 doubles each axis (4x pixels) but roughly 16x the
  /// dominant attention term. This is the mechanism behind issue #22's
  /// reported 2048px slowdown, pinned as a regression guard so a future edit
  /// to the formula cannot accidentally flatten it back to linear.
  func testEstimateFlux1At4096PinnedAndQuadraticallyLarger() {
    let at2048 = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .flux1, dype: false)
    let at4096 = ImageMemoryPreflight.estimateBytes(width: 4096, height: 4096, family: .flux1, dype: false)
    XCTAssertEqual(at4096, 540_964_552_704)
    // 4x the pixels; the estimate must grow well past 4x (quadratic attention term).
    XCTAssertGreaterThan(Double(at4096), Double(at2048) * 8.0)
  }

  /// krea2 (the production model, `intent.md`) is heavier than flux1 at the
  /// SAME resolution — a larger `features`/`heads`/FFN width than Z-Image-Turbo.
  func testEstimateKrea2IsLargerThanFlux1AtSameResolution() {
    let flux1 = ImageMemoryPreflight.estimateBytes(width: 1024, height: 1024, family: .flux1, dype: false)
    let krea2 = ImageMemoryPreflight.estimateBytes(width: 1024, height: 1024, family: .krea2, dype: false)
    XCTAssertEqual(flux1, 3_611_295_744)
    XCTAssertEqual(krea2, 5_133_828_096)
    XCTAssertGreaterThan(krea2, flux1)
  }

  func testEstimateKrea2At2048IsPinned() {
    let noDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .krea2, dype: false)
    let withDype = ImageMemoryPreflight.estimateBytes(width: 2048, height: 2048, family: .krea2, dype: true)
    XCTAssertEqual(noDype, 59_190_018_048)
    XCTAssertEqual(withDype, 59_194_408_960)
  }

  /// Unrecognized/unresolved families (flux2/fibo/chroma have no compiled-in
  /// transformer constants) must default to the LARGER (krea2) profile —
  /// never under-estimate.
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

  // MARK: - decide(estimate:available:cap:)

  func testDecideAllowsWhenEstimateFitsUnderCap() {
    let d = ImageMemoryPreflight.decide(estimate: 10 * gb, available: 100 * gb, cap: 90 * gb)
    XCTAssertTrue(d.allow)
    XCTAssertTrue(d.reason.contains("10240MB") || d.reason.contains("10485760") || d.reason.contains("MB"), d.reason)
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

  // MARK: - validate(...): the combined gate, thrown as WarmServerError

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

  func testValidateThrowsInsufficientMemoryNamingEstimateAvailableAndCap() {
    // 2048px krea2+DyPE (~55GB estimate) against a starved machine (only
    // 10GB free) — well within resolution caps, refused on memory instead.
    XCTAssertThrowsError(
      try ImageMemoryPreflight.validate(
        width: 2048, height: 2048, family: .krea2, dype: true,
        caps: .default, availableBytes: 10 * gb)
    ) { error in
      guard case let .imageMemoryPreflightRefused(code, reason, estimate, available, cap) =
        error as? WarmServerError else {
        return XCTFail("expected .imageMemoryPreflightRefused, got \(error)")
      }
      XCTAssertEqual(code, "insufficient_memory")
      XCTAssertEqual(estimate, 59_194_408_960)
      XCTAssertEqual(available, 10 * gb)
      XCTAssertEqual(cap, UInt64(Double(10 * gb) * 0.9))
      XCTAssertFalse(reason.isEmpty)
    }
  }

  /// The test plan's second half: 2048² on a machine with real M3 Max-class
  /// headroom (well over 100GB free) must pass BOTH gates.
  func testValidateAllows2048SquareWithGenerousHeadroom() throws {
    try ImageMemoryPreflight.validate(
      width: 2048, height: 2048, family: .krea2, dype: true,
      caps: .default, availableBytes: 100 * gb)
  }

  /// The test plan's first half, as a deterministic unit test: 6000x6000
  /// refuses immediately, with no dependence on live machine memory.
  func testValidateRefuses6000SquareRegardlessOfAvailableMemory() {
    for available: UInt64 in [0, 1 * gb, 1000 * gb] {
      XCTAssertThrowsError(
        try ImageMemoryPreflight.validate(
          width: 6000, height: 6000, family: .krea2, dype: false,
          caps: .default, availableBytes: available))
    }
  }

  // MARK: - HTTP shape: 413, additive JSON fields

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

  func testInsufficientMemoryRefusalMapsTo413NamingAllThreeNumbers() throws {
    var caught: Error?
    do {
      try ImageMemoryPreflight.validate(
        width: 2048, height: 2048, family: .krea2, dype: true,
        caps: .default, availableBytes: 10 * gb)
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
    XCTAssertEqual(estimate.uint64Value, 59_194_408_960)
    XCTAssertEqual(available.uint64Value, 10 * gb)
    XCTAssertEqual(cap.uint64Value, UInt64(Double(10 * gb) * 0.9))
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

  func testPayloadWiringUsesRequestWidthHeightAndDefaultsFamilyToKrea2() {
    let payload = GeneratePayload(prompt: "x", width: 6000, height: 6000)
    XCTAssertThrowsError(
      try payload.validateImageMemoryPreflight(caps: .default, availableBytes: 200 * gb)
    ) { error in
      guard case let .imageMemoryPreflightRefused(code, _, _, _, _) = error as? WarmServerError else {
        return XCTFail("expected .imageMemoryPreflightRefused, got \(error)")
      }
      XCTAssertEqual(code, "resolution_cap")
    }
  }

  func testPayloadWiringHonoursExplicitModelFamily() {
    // flux1 at 2048px needs far less than krea2 at the same resolution
    // (pinned above) — an explicit `model: "z-image"` must resolve to the
    // lighter flux1 profile and therefore pass where the krea2 default
    // (tested elsewhere) would not, at a headroom in between the two.
    let payload = GeneratePayload(prompt: "x", width: 2048, height: 2048, model: "z-image")
    // flux1@2048 with DyPE ≈ 35.96GB; give it a cap comfortably above that
    // but BELOW krea2@2048's ≈55.13GB estimate.
    XCTAssertNoThrow(
      try payload.validateImageMemoryPreflight(caps: .default, availableBytes: 45 * gb))
  }

  func testPayloadWiringSkipsWhenDimensionsAreOmitted() {
    // No width/height at all — resolves to the small engine default later in
    // the pipeline, never DyPE territory. Must not throw even under a
    // starved injected budget.
    let payload = GeneratePayload(prompt: "x")
    XCTAssertNoThrow(
      try payload.validateImageMemoryPreflight(caps: .default, availableBytes: 0))
  }

  func testPayloadWiringPassesAt2048WithGenerousHeadroom() {
    let payload = GeneratePayload(prompt: "x", width: 2048, height: 2048)
    XCTAssertNoThrow(
      try payload.validateImageMemoryPreflight(caps: .default, availableBytes: 100 * gb))
  }

  private var gb: UInt64 { 1024 * 1024 * 1024 }
}
