import XCTest

@testable import ZImage

/// WP-E5 — variant defaults (FDD §3.5, AC-5b). `runKrea2Generate` resolves
/// `steps`/`guidance` through `Krea2Variant` instead of the literals `?? 9` /
/// `?? 1.0`; the resolution is a pure function so it is asserted here with no
/// weights. (`applied.stages[0]` reporting both is WP-E10's provenance record.)
final class Krea2VariantDefaultsTests: XCTestCase {

  func testVariantTable() {
    XCTAssertEqual(Krea2Variant.turbo.transformerFilename, "turbo.safetensors")
    XCTAssertEqual(Krea2Variant.raw.transformerFilename, "raw.safetensors")
    XCTAssertFalse(Krea2Variant.turbo.supportsGuidance)
    XCTAssertTrue(Krea2Variant.raw.supportsGuidance)
    XCTAssertEqual(Krea2Variant.turbo.defaultSteps, 9)
    XCTAssertEqual(Krea2Variant.raw.defaultSteps, 30)
    // 1.0 == off on both. 3.5 is the client's raw-stock policy, NOT an engine
    // default: an engine default that doubles model evals is a surprise.
    XCTAssertEqual(Krea2Variant.turbo.defaultGuidance, 1.0)
    XCTAssertEqual(Krea2Variant.raw.defaultGuidance, 1.0)
    XCTAssertEqual(Krea2Variant.turbo.bridgeStepClamp, 12)
    XCTAssertNil(Krea2Variant.raw.bridgeStepClamp)
    XCTAssertEqual(Krea2Variant(rawValue: "raw"), .raw)
    XCTAssertEqual(Krea2Variant(rawValue: "turbo"), .turbo)
    XCTAssertNil(Krea2Variant(rawValue: "base"))
  }

  func testRawDefaultsReachTheGeneratePath() {
    // No steps, no guidance on a .raw pipeline → 30 / 1.0 (not 3.5).
    XCTAssertEqual(Krea2Variant.raw.resolvedSteps(nil), 30)
    XCTAssertEqual(Krea2Variant.raw.resolvedGuidance(nil), 1.0)
    // Explicit values pass through untouched.
    XCTAssertEqual(Krea2Variant.raw.resolvedSteps(52), 52)
    XCTAssertEqual(Krea2Variant.raw.resolvedGuidance(3.5), 3.5)
  }

  func testTurboDefaultsUnchanged() {
    XCTAssertEqual(Krea2Variant.turbo.resolvedSteps(nil), 9)
    XCTAssertEqual(Krea2Variant.turbo.resolvedGuidance(nil), 1.0)
    XCTAssertEqual(Krea2Variant.turbo.resolvedSteps(6), 6)
    XCTAssertEqual(Krea2Variant.turbo.resolvedGuidance(1.5), 1.5)
  }

  func testPoolEntryCarriesTheVariantBack() async throws {
    // `PoolEntry.detectedInfo` carries the loaded variant from loadPipeline to
    // poolActivate, where the coordinator stores it as `currentKrea2Variant`.
    let pool = ModelPool(
      budgetMB: 40960, textEncoderPath: nil, maxSequenceLength: 512,
      forceTransformerOverrideOnly: false, logger: .init(label: "test"))
    await pool.registerExisting(
      poolKey: "krea2-raw", modelSpec: "krea2-raw", family: .krea2,
      box: PipelineBox(pipeline: NSObject()), vramEstimateMB: 22528, detectedInfo: Krea2Variant.raw)
    let entry = try await pool.activate(modelId: "krea2-raw")
    XCTAssertEqual(entry.detectedInfo as? Krea2Variant, .raw)
  }
}
