import XCTest

@testable import ZImage

/// WP-E4 (FDD-krea2-raw-recipe D22, AC-18) — a persisted queue job that fails
/// replay validation is marked FAILED with the reason recorded (visible on
/// `GET /v1/generate/status/{id}`), never rendered and never silently
/// dropped. `recoverPersistedQueue` decodes through `decodedGeneratePayload`
/// (the same fail-loud path the live routes use) and records the failure
/// through `ImageJobTracker.recordFailedReplay`; both halves are asserted
/// here without a server.
final class PersistedQueueRecoveryTests: XCTestCase {

  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  func testInvalidRecipeJobFailsLoud() throws {
    let body = Data(#"{"prompt":"x","scheduler":"uni_pc"}"#.utf8)
    let job = PersistedQueueJob(id: "recovered-1", kind: "generate", source: "api", enqueuedAt: Date(), rawBody: body)

    // The replayed body fails validation by name.
    var failure: Error?
    do { _ = try decode(String(decoding: job.rawBody, as: UTF8.self)).validateRecipeNames() } catch { failure = error }
    let error = try XCTUnwrap(failure, "a persisted uni_pc job must fail replay validation")
    guard case WarmServerError.unknownSampler(let name, _) = error else { return XCTFail("\(error)") }
    XCTAssertEqual(name, "uni_pc")

    // And the failure is recorded against the job's own id with the reason.
    let tracker = ImageJobTracker()
    tracker.recordFailedReplay(jobId: job.id, source: job.source, error: error)
    let status = try XCTUnwrap(tracker.status(jobId: "recovered-1"), "a failed replay must be queryable, not dropped")
    XCTAssertEqual(status.status, .failed)
    XCTAssertEqual(status.source, "api")
    XCTAssertNil(status.outputPath, "never rendered")
    let reason = try XCTUnwrap(status.error)
    XCTAssertTrue(reason.contains("uni_pc"), "reason must name the value: \(reason)")
  }

  func testValidPersistedJobStillDecodes() throws {
    let body = #"{"prompt":"x","scheduler":"dpmpp_2m","sigma_schedule":"normal"}"#
    let names = try decode(body).validateRecipeNames()
    XCTAssertEqual(names.scheduler, .dpmplusplus2m)
    XCTAssertEqual(names.sigmaSchedule, .flow)
  }
}
