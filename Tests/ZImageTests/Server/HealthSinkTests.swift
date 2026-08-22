import XCTest
@testable import ZImage

/// WP-E10 — sink 3, `/health` (FDD §3.10; §7.3 smoke steps c/e; Addendum
/// A.2 `model_alias`). The response carries `last_recipe` (the same record
/// the sync response returned), `model_alias` beside the resolved `model`,
/// and `build_sha` so a clobbered binary is detectable from outside.
final class HealthSinkTests: XCTestCase {

  private func encode(_ health: HealthResponse) throws -> [String: Any] {
    let e = JSONEncoder(); e.keyEncodingStrategy = .convertToSnakeCase
    return try XCTUnwrap(JSONSerialization.jsonObject(with: e.encode(health)) as? [String: Any])
  }

  private func sample(lastRecipe: RenderRecipe?, alias: String?) -> HealthResponse {
    HealthResponse(
      status: "ok", model: "/Users/me/LocalModels/krea2-raw", modelFamily: "krea2", modelVariant: "raw",
      modelAlias: alias, buildSha: BuildInfo.gitSHA,
      textEncoderPath: nil, loaded: true, loras: [], uptimeSeconds: 1, renderCount: 1, failedRenderCount: 0,
      pendingCount: 0, maxPending: 8, isRendering: false, isPaused: false, activeRequestAgeMs: nil,
      currentJobId: nil, progressPercent: nil, memoryUsageBytes: 0, memoryUsageMB: 0,
      lastRenderDurationMs: 1234, lastError: nil, lastRecipe: lastRecipe)
  }

  /// The same builder every other sink test uses (AC-62: the four sinks
  /// carry one record, not four look-alikes).
  private func recipe() -> RenderRecipe {
    RenderRecipeFixture.recipe(steps: 9)
  }

  func testHealthCarriesLastRecipeAliasAndBuildSha() throws {
    let json = try encode(sample(lastRecipe: recipe(), alias: "krea2-raw"))
    XCTAssertEqual(json["model"] as? String, "/Users/me/LocalModels/krea2-raw")
    XCTAssertEqual(json["model_alias"] as? String, "krea2-raw")
    XCTAssertEqual(json["model_variant"] as? String, "raw")
    XCTAssertEqual(json["build_sha"] as? String, BuildInfo.gitSHA)
    let last = try XCTUnwrap(json["last_recipe"] as? [String: Any])
    XCTAssertEqual(last["base_variant"] as? String, "raw")
    XCTAssertEqual((last["stages"] as? [[String: Any]])?.first?["steps_run"] as? Int, 9)
    XCTAssertEqual(last["quantization"] as? String, "q8")
  }

  /// Before any krea2 render, and on other families, the /health body still
  /// carries the keys — as JSON null — so a client decodes them
  /// unconditionally (AC-64 second half; the same telemetry contract as
  /// `current_job_id`). `WarmServer.healthJSON` is the route's body builder.
  func testAbsentRecipeAndAliasArePresentAsNullInTheHealthBody() throws {
    let data = try XCTUnwrap(WarmServer.healthJSON(sample(lastRecipe: nil, alias: nil), videoAvailable: false, activeVideoJobs: 0))
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    XCTAssertTrue(json["last_recipe"] is NSNull, "\(String(describing: json["last_recipe"]))")
    XCTAssertTrue(json["model_alias"] is NSNull)
    XCTAssertTrue(json["current_job_id"] is NSNull, "the pre-existing contract is kept")
    XCTAssertEqual(json["build_sha"] as? String, BuildInfo.gitSHA)
    XCTAssertEqual((json["video"] as? [String: Any])?["backend"] as? String, "none")

    // With a record: the same builder carries it through intact.
    let withData = try XCTUnwrap(WarmServer.healthJSON(sample(lastRecipe: recipe(), alias: "krea2-raw"), videoAvailable: true, activeVideoJobs: 2))
    let with = try XCTUnwrap(JSONSerialization.jsonObject(with: withData) as? [String: Any])
    XCTAssertEqual((with["last_recipe"] as? [String: Any])?["quantization"] as? String, "q8")
    XCTAssertEqual(with["model_alias"] as? String, "krea2-raw")
  }

  /// `build_sha` is a git short sha (7–40 hex, optional `-dirty`) or the
  /// committed placeholder — never empty, never a version string.
  func testBuildShaFormat() {
    let sha = BuildInfo.gitSHA
    XCTAssertFalse(sha.isEmpty)
    let ok = sha == BuildInfo.placeholder
      || sha.range(of: #"^[0-9a-f]{7,40}(-dirty)?$"#, options: .regularExpression) != nil
    XCTAssertTrue(ok, "unexpected build sha '\(sha)'")
    XCTAssertEqual(BuildInfo.placeholder, "unknown")
    XCTAssertEqual(BuildInfo.isKnown, sha != BuildInfo.placeholder)
  }
}
