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
    sample(slot: lastRecipe.map(AppliedRecordSlot.init(record:)), alias: alias)
  }

  private func sample(slot: AppliedRecordSlot?, alias: String?) -> HealthResponse {
    HealthResponse(
      status: "ok", model: "/Users/me/LocalModels/krea2-raw", modelFamily: "krea2", modelVariant: "raw",
      modelAlias: alias, buildSha: BuildInfo.gitSHA,
      textEncoderPath: nil, loaded: true, loras: [], uptimeSeconds: 1, renderCount: 1, failedRenderCount: 0,
      pendingCount: 0, maxPending: 8, isRendering: false, isPaused: false, activeRequestAgeMs: nil,
      currentJobId: nil, progressPercent: nil, memoryUsageBytes: 0, memoryUsageMB: 0,
      lastRenderDurationMs: 1234, lastError: nil, lastRecipe: slot)
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

  func testHealthAdvertisesReadyLocalLTX2WithRequiredAndOptionalAssets() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-video-health-\(UUID().uuidString)", isDirectory: true)
    let weights = root.appendingPathComponent("weights", isDirectory: true)
    let gemma = root.appendingPathComponent("gemma", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: gemma, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try Data().write(to: weights.appendingPathComponent("local-monolith.safetensors"))
    try Data().write(to: gemma.appendingPathComponent("model.safetensors"))
    try Data("{}".utf8).write(to: gemma.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: gemma.appendingPathComponent("tokenizer.json"))

    let data = try XCTUnwrap(WarmServer.healthJSON(
      sample(lastRecipe: nil, alias: nil), videoAvailable: false, activeVideoJobs: 0,
      localVideoWeightsPath: weights.path, localVideoGemmaPath: gemma.path,
      localVideoUpsamplerPath: nil))
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let video = try XCTUnwrap(json["video"] as? [String: Any])
    XCTAssertEqual(video["available"] as? Bool, true)
    XCTAssertEqual(video["backend"] as? String, "local_ltx2")
    let readiness = try XCTUnwrap(video["readiness"] as? [String: Any])
    XCTAssertEqual(readiness["ready"] as? Bool, true)
    XCTAssertEqual((readiness["required_assets"] as? [[String: Any]])?.count, 2)
    let optional = try XCTUnwrap((readiness["optional_assets"] as? [[String: Any]])?.first)
    XCTAssertEqual(optional["name"] as? String, "ltx2_upsampler")
    XCTAssertEqual(optional["required"] as? Bool, false)
    XCTAssertEqual(optional["valid"] as? Bool, true, "an absent optional upsampler must not disable core video")
  }

  func testHealthRejectsIncompleteLocalLTX2Assets() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-video-health-missing-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let data = try XCTUnwrap(WarmServer.healthJSON(
      sample(lastRecipe: nil, alias: nil), videoAvailable: false, activeVideoJobs: 0,
      localVideoWeightsPath: root.appendingPathComponent("missing-weights").path,
      localVideoGemmaPath: root.appendingPathComponent("missing-gemma").path))
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let video = try XCTUnwrap(json["video"] as? [String: Any])
    XCTAssertEqual(video["available"] as? Bool, false)
    XCTAssertEqual(video["backend"] as? String, "none")
    let readiness = try XCTUnwrap(video["readiness"] as? [String: Any])
    XCTAssertEqual(readiness["ready"] as? Bool, false)
    let required = try XCTUnwrap(readiness["required_assets"] as? [[String: Any]])
    XCTAssertTrue(required.allSatisfy { ($0["valid"] as? Bool) == false })
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

/// `/health.last_recipe` must never describe a model that is no longer
/// resident. A base handoff (a different Raw/Turbo file, or a switch out of
/// the family entirely) invalidates the record: keeping it would put a
/// `krea2-raw` provenance block beside a `model: z-image-turbo` line, which
/// is exactly the silent mismatch this document exists to prevent.
final class LastRecipeRetentionTests: XCTestCase {

  private var recipe: RenderRecipe { RenderRecipeFixture.recipe() }

  func testTheRecordSurvivesAReactivationOfTheSameBase() {
    XCTAssertEqual(
      RenderRecipe.retained(recipe, activeTransformerFile: recipe.baseModelFile),
      recipe)
  }

  func testADifferentTransformerFileDropsTheRecord() {
    XCTAssertNil(RenderRecipe.retained(recipe, activeTransformerFile: "/Users/me/LocalModels/kroma-v0.2/turbo.safetensors"))
  }

  /// Another family has no krea2 transformer to compare against — the record
  /// goes, rather than being reported next to a foreign model.
  func testAnotherFamilyDropsTheRecord() {
    XCTAssertNil(RenderRecipe.retained(recipe, activeTransformerFile: nil))
  }

  func testNoRecordStaysNoRecord() {
    XCTAssertNil(RenderRecipe.retained(nil, activeTransformerFile: "/x/raw.safetensors"))
  }

  // MARK: - The rule as every call site states it: (family, resident file)

  /// A re-activation of the same Krea 2 checkpoint keeps the record.
  func testSameFamilySameFileKeepsTheRecord() {
    XCTAssertEqual(
      RenderRecipe.retained(recipe, family: .krea2, krea2TransformerFile: recipe.baseModelFile),
      recipe)
  }

  /// **The #218 eviction path.** `releaseImageModelsForVideo()` nils
  /// `krea2Pipeline` to vacate ~22 GB for the LTX-2 stack; the family is still
  /// `.krea2` but nothing is resident. `/health` used to publish a full
  /// provenance block beside `loaded: false` for the whole video render — tens
  /// of minutes of a record for a checkpoint that is not in memory.
  func testAFullImageStackEvictionDropsTheRecord() {
    XCTAssertNil(
      RenderRecipe.retained(recipe, family: .krea2, krea2TransformerFile: nil),
      "no resident Krea 2 pipeline → no record, even though the family has not changed")
  }

  /// A base handoff inside the family: a different checkpoint, so a different
  /// record — never the old one.
  func testAHandoffToAnotherKrea2CheckpointDropsTheRecord() {
    XCTAssertNil(RenderRecipe.retained(
      recipe, family: .krea2,
      krea2TransformerFile: "/Users/me/LocalModels/kroma-v0.2/turbo.safetensors"))
  }

  /// Every non-Krea-2 family drops it, whatever happens to be resident —
  /// `prepare()`'s other arms and `poolActivate`'s other cases both land here.
  func testEveryOtherFamilyDropsTheRecord() {
    for family in WarmModelFamily.allCases where family != .krea2 {
      XCTAssertNil(
        RenderRecipe.retained(recipe, family: family, krea2TransformerFile: recipe.baseModelFile),
        "\(family) must not publish a Krea 2 record")
    }
  }
}
