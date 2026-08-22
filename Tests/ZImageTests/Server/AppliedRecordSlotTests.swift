// AppliedRecordSlotTests.swift — K-FIX-1 round 2, the C4 client-lane finding.
//
// A synthesized Swift `Codable` OMITS a nil Optional, so `applied: nil` and
// "this family has no record" were the same bytes on the wire: an absent key.
// That made ENGINE-INCOMPLETE provenance — a Krea 2 render whose read-back was
// refused because the pipeline's loaded LoRA configs and bind reports
// disagreed (§3.10's fail-closed rule) — indistinguishable from a Flux/Chroma
// render that never had a record to give.
//
// The contract these tests pin, on all three JSON sinks:
//
//   key ABSENT  — no Krea 2 provenance applies (other family / none yet).
//   `null`      — a Krea 2 render completed and its record was REFUSED.
//   object      — the record.
//
// Asserted against the literal JSON TEXT, not a re-parsed dictionary: parsing
// erases exactly the distinction under test (`json["applied"]` is nil for an
// absent key AND for an `NSNull`).

import Foundation
import XCTest

@testable import ZImage

final class AppliedRecordSlotTests: XCTestCase {

  private func encodedText<T: Encodable>(_ value: T) throws -> String {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return String(decoding: try encoder.encode(value), as: UTF8.self)
  }

  private func recipe() -> RenderRecipe { RenderRecipeFixture.recipe(steps: 9) }

  // MARK: - The slot itself

  func testSlotEncodesNullWhenTheRecordWasRefused() throws {
    XCTAssertEqual(try encodedText(AppliedRecordSlot.refused), "null")
    XCTAssertEqual(try encodedText(AppliedRecordSlot(record: nil)), "null")
  }

  func testSlotRoundTrips() throws {
    let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase

    let refused = try decoder.decode(
      AppliedRecordSlot.self, from: try encoder.encode(AppliedRecordSlot.refused))
    XCTAssertNil(refused.record, "null decodes back to a PRESENT slot holding no record")

    let r = recipe()
    let carried = try decoder.decode(
      AppliedRecordSlot.self, from: try encoder.encode(AppliedRecordSlot(record: r)))
    XCTAssertEqual(carried.record, r)
  }

  // MARK: - Sink 1: GenerateResponse

  func testGenerateResponseDistinguishesRefusedFromAbsent() throws {
    let refused = GenerateResponse(
      success: true, outputPath: "/x.png", durationMs: 10, applied: .refused)
    let text = try encodedText(refused)
    XCTAssertTrue(text.contains("\"applied\":null"),
                  "a refused Krea 2 record must be a literal null: \(text)")

    // Another family: no slot at all — the key is absent.
    let otherFamily = GenerateResponse(success: true, outputPath: "/x.png", durationMs: 10)
    let absent = try encodedText(otherFamily)
    XCTAssertFalse(absent.contains("\"applied\""),
                   "a non-krea2 render must omit the key entirely: \(absent)")

    // And the record itself still encodes as an object.
    let carried = try encodedText(GenerateResponse(
      success: true, outputPath: "/x.png", durationMs: 10,
      applied: AppliedRecordSlot(record: recipe())))
    XCTAssertTrue(carried.contains("\"applied\":{"), carried)
    XCTAssertFalse(carried.contains("\"applied\":null"), carried)
  }

  // MARK: - Sink 4: ImageJobStatus (async)

  func testImageJobStatusDistinguishesRefusedFromAbsent() throws {
    func status(_ applied: AppliedRecordSlot?) -> ImageJobStatus {
      ImageJobStatus(
        jobId: "j", status: .succeeded, source: "api", outputPath: "/y.png", durationMs: 5,
        error: nil, elapsedMs: 6, preemptRefused: nil, etaSec: nil, applied: applied)
    }
    let refused = try encodedText(status(.refused))
    XCTAssertTrue(refused.contains("\"applied\":null"), refused)

    let absent = try encodedText(status(nil))
    XCTAssertFalse(absent.contains("\"applied\""), absent)

    let carried = try encodedText(status(AppliedRecordSlot(record: recipe())))
    XCTAssertTrue(carried.contains("\"applied\":{"), carried)

    // Persisted pre-upgrade JSON (no key at all) still decodes — AC-64 —
    // and reads as ABSENT, not as refused.
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    let old = #"{"job_id":"j1","status":"succeeded","source":"api","output_path":"/x.png","duration_ms":10,"error":null,"elapsed_ms":12}"#
    let decoded = try decoder.decode(ImageJobStatus.self, from: Data(old.utf8))
    XCTAssertNil(decoded.applied, "an absent key is ABSENT, never a refused slot")

    // A persisted `"applied":null` decodes as REFUSED — a present slot.
    let refusedJSON = #"{"job_id":"j1","status":"succeeded","source":"api","output_path":"/x.png","duration_ms":10,"error":null,"elapsed_ms":12,"applied":null}"#
    let decodedRefused = try decoder.decode(ImageJobStatus.self, from: Data(refusedJSON.utf8))
    XCTAssertNotNil(decodedRefused.applied, "null is a PRESENT slot")
    XCTAssertNil(decodedRefused.appliedRecord)
  }

  // MARK: - Sink 3: /health.last_recipe

  func testHealthLastRecipeDistinguishesRefusedFromAbsent() throws {
    func health(_ slot: AppliedRecordSlot?) -> HealthResponse {
      HealthResponse(
        status: "ok", model: "/m", modelFamily: "krea2", modelVariant: "raw", modelAlias: nil,
        buildSha: BuildInfo.gitSHA, textEncoderPath: nil, loaded: true, loras: [],
        uptimeSeconds: 1, renderCount: 1, failedRenderCount: 0, pendingCount: 0, maxPending: 8,
        isRendering: false, isPaused: false, activeRequestAgeMs: nil, currentJobId: nil,
        progressPercent: nil, memoryUsageBytes: 0, memoryUsageMB: 0, lastRenderDurationMs: 1,
        lastError: nil, lastRecipe: slot)
    }
    let refused = try encodedText(health(.refused))
    XCTAssertTrue(refused.contains("\"last_recipe\":null"), refused)

    let absent = try encodedText(health(nil))
    XCTAssertFalse(absent.contains("\"last_recipe\""),
                   "no Krea 2 render yet ⇒ the key is absent: \(absent)")

    let carried = try encodedText(health(AppliedRecordSlot(record: recipe())))
    XCTAssertTrue(carried.contains("\"last_recipe\":{"), carried)
  }

  // MARK: - The PNG sidecar makes the same distinction

  /// The PNG is the fourth sink and the one that outlives the response. It
  /// must not read as "a render with no provenance" when the engine actually
  /// refused the record.
  func testPNGSidecarDistinguishesRefusedFromAbsent() throws {
    // Refused: `applied` present and null.
    let refused = QwenImageIO.ImageMetadata.generation(
      prompt: "p", seed: 1, steps: 9, guidance: 1.0, width: 1024, height: 1024,
      model: "krea2-raw", appliedSlot: .refused)
    let refusedJSON = try XCTUnwrap(refused.parametersJSON)
    XCTAssertTrue(refusedJSON.contains("\"applied\":null"), refusedJSON)

    // Another family: no key.
    let absent = QwenImageIO.ImageMetadata.generation(
      prompt: "p", seed: 1, steps: 9, guidance: 1.0, width: 1024, height: 1024, model: "z-image")
    let absentJSON = try XCTUnwrap(absent.parametersJSON)
    XCTAssertFalse(absentJSON.contains("\"applied\""), absentJSON)

    // Carried: the object, as before.
    let carried = QwenImageIO.ImageMetadata.generation(
      prompt: "p", seed: 1, steps: 9, guidance: 1.0, width: 1024, height: 1024,
      model: "krea2-raw", applied: recipe())
    let carriedJSON = try XCTUnwrap(carried.parametersJSON)
    XCTAssertTrue(carriedJSON.contains("\"applied\":{"), carriedJSON)
    XCTAssertFalse(carriedJSON.contains("\"applied\":null"), carriedJSON)
  }
}
