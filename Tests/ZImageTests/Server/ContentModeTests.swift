import XCTest
@testable import ZImage

final class ContentModeTests: XCTestCase {

  private func makeTempPath() -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-content-modes-tests-\(UUID().uuidString)", isDirectory: true)
      .appendingPathComponent("content-modes.json")
  }

  // MARK: - Listing

  func testListModesReturnsThreeBuiltinsInCanonicalOrder() {
    let store = ContentModeStore()
    let modes = store.listModes()
    XCTAssertEqual(modes.map { $0.mode }, [.neutral, .banana, .avocado])
    XCTAssertEqual(modes.map { $0.label }, ["Neutral", "Banana", "Avocado"])
  }

  func testBuiltinDefaults() {
    let store = ContentModeStore()
    XCTAssertEqual(store.definition(for: .neutral).guidanceBoost, 0)
    XCTAssertEqual(store.definition(for: .neutral).styleVariant, .neutral)
    XCTAssertNil(store.definition(for: .neutral).promptHint)

    XCTAssertEqual(store.definition(for: .banana).guidanceBoost, 1.5)
    XCTAssertEqual(store.definition(for: .banana).styleVariant, .sensual)
    XCTAssertEqual(store.definition(for: .banana).promptHint, "sensual, intimate, suggestive")

    XCTAssertEqual(store.definition(for: .avocado).guidanceBoost, 2.5)
    XCTAssertEqual(store.definition(for: .avocado).styleVariant, .nsfw)
    XCTAssertEqual(store.definition(for: .avocado).promptHint, "explicit, uncensored, anatomically detailed")
  }

  // MARK: - Effect resolution (parity with resolveContentMode)

  func testResolveAvocado() {
    let store = ContentModeStore()
    let e = store.resolveEffect(mode: .avocado)
    XCTAssertEqual(e.guidanceBoost, 2.5)
    XCTAssertEqual(e.styleVariant, .nsfw)
    XCTAssertEqual(e.promptHint, "explicit, uncensored, anatomically detailed")
  }

  func testResolveBanana() {
    let store = ContentModeStore()
    let e = store.resolveEffect(mode: .banana)
    XCTAssertEqual(e.guidanceBoost, 1.5)
    XCTAssertEqual(e.styleVariant, .sensual)
    XCTAssertEqual(e.promptHint, "sensual, intimate, suggestive")
  }

  func testResolveNeutralPlain() {
    let store = ContentModeStore()
    let e = store.resolveEffect(mode: .neutral, prompt: "a cat on a windowsill")
    XCTAssertEqual(e.guidanceBoost, 0)
    XCTAssertEqual(e.styleVariant, .neutral)
    XCTAssertNil(e.promptHint)
  }

  func testResolveNilModeIsNeutral() {
    let store = ContentModeStore()
    let e = store.resolveEffect(mode: nil, prompt: "landscape at dawn")
    XCTAssertEqual(e.styleVariant, .neutral)
    XCTAssertEqual(e.guidanceBoost, 0)
  }

  // MARK: - Auto-detection from prompt

  func testNeutralAutoDetectsNsfwPrompt() {
    let store = ContentModeStore()
    let e = store.resolveEffect(mode: .neutral, prompt: "an explicit erotic scene")
    XCTAssertEqual(e.styleVariant, .nsfw)
    XCTAssertEqual(e.guidanceBoost, 0, "auto-detection never boosts guidance")
    XCTAssertNil(e.promptHint)
  }

  func testNeutralAutoDetectsSensualPrompt() {
    let store = ContentModeStore()
    let e = store.resolveEffect(mode: .neutral, prompt: "a tasteful boudoir portrait in lingerie")
    XCTAssertEqual(e.styleVariant, .sensual)
    XCTAssertEqual(e.guidanceBoost, 0)
  }

  func testNsfwPatternTakesPrecedenceOverSensual() {
    let store = ContentModeStore()
    // Prompt matches both patterns; NSFW is checked first.
    let e = store.resolveEffect(mode: .neutral, prompt: "sensual explicit boudoir")
    XCTAssertEqual(e.styleVariant, .nsfw)
  }

  func testAutoDetectIsCaseInsensitive() {
    let store = ContentModeStore()
    XCTAssertEqual(store.resolveEffect(mode: .neutral, prompt: "NSFW content").styleVariant, .nsfw)
    XCTAssertEqual(store.resolveEffect(mode: .neutral, prompt: "NUDE study").styleVariant, .nsfw)
  }

  func testWordBoundaryAvoidsFalsePositives() {
    let store = ContentModeStore()
    // "nude" must not match inside "denude"? "denude" contains "nude" but not on a word boundary.
    XCTAssertEqual(store.resolveEffect(mode: .neutral, prompt: "prenatal denuded rocks").styleVariant, .neutral)
  }

  // MARK: - CFG-distilled suppression

  func testCfgDistilledSuppressesBoostForAvocado() {
    let store = ContentModeStore()
    let ctx = GuidanceContext(model: "z-image-turbo")
    let e = store.resolveEffect(mode: .avocado, context: ctx)
    XCTAssertEqual(e.guidanceBoost, 0, "turbo model suppresses the boost")
    XCTAssertEqual(e.styleVariant, .nsfw, "style variant + hint survive suppression")
    XCTAssertEqual(e.promptHint, "explicit, uncensored, anatomically detailed")
  }

  func testCfgDistilledSuppressesBananaBoost() {
    let store = ContentModeStore()
    let ctx = GuidanceContext(baseModel: "schnell")
    XCTAssertEqual(store.resolveEffect(mode: .banana, context: ctx).guidanceBoost, 0)
  }

  func testIsCfgDistilledDetection() {
    let store = ContentModeStore()
    XCTAssertTrue(store.isCfgDistilledModel(GuidanceContext(model: "some-turbo-model")))
    XCTAssertTrue(store.isCfgDistilledModel(GuidanceContext(baseModel: "flux-schnell")))
    XCTAssertTrue(store.isCfgDistilledModel(GuidanceContext(mode: "lightning-4step")))
    // engine=zimage with no mode trips the check.
    XCTAssertTrue(store.isCfgDistilledModel(GuidanceContext(engine: "zimage")))
    XCTAssertTrue(store.isCfgDistilledModel(GuidanceContext(engine: "zimage", mode: "z-image-turbo")))
    // engine=zimage but an explicit non-turbo mode is NOT distilled.
    XCTAssertFalse(store.isCfgDistilledModel(GuidanceContext(engine: "zimage", mode: "dev")))
    // Nothing distilled here.
    XCTAssertFalse(store.isCfgDistilledModel(GuidanceContext(model: "flux-dev")))
    XCTAssertFalse(store.isCfgDistilledModel(GuidanceContext()))
  }

  // MARK: - apply() guidance math + additions

  func testApplyWithBaseGuidanceAddsBoost() {
    let store = ContentModeStore()
    let result = store.apply(ContentModeRequest(mode: .avocado, guidance: 4.0))
    XCTAssertEqual(result.guidance, 6.5)  // 4.0 + 2.5
    XCTAssertEqual(result.promptAdditions, ["explicit, uncensored, anatomically detailed"])
  }

  func testApplyWithoutBaseGuidanceUsesFallbackWhenBoosted() {
    let store = ContentModeStore()
    let result = store.apply(ContentModeRequest(mode: .banana))
    XCTAssertEqual(result.guidance, 5.0)  // 3.5 + 1.5
    XCTAssertEqual(result.promptAdditions, ["sensual, intimate, suggestive"])
  }

  func testApplyNeutralWithoutBoostYieldsNilGuidance() {
    let store = ContentModeStore()
    let result = store.apply(ContentModeRequest(mode: .neutral, prompt: "a quiet forest"))
    XCTAssertNil(result.guidance)
    XCTAssertTrue(result.promptAdditions.isEmpty)
    XCTAssertTrue(result.negativePromptAdditions.isEmpty)
  }

  func testApplyNeutralPreservesBaseGuidance() {
    let store = ContentModeStore()
    let result = store.apply(ContentModeRequest(mode: .neutral, prompt: "a lake", guidance: 3.0))
    XCTAssertEqual(result.guidance, 3.0)  // 3.0 + 0
  }

  func testApplySuppressedBoostKeepsBaseGuidance() {
    let store = ContentModeStore()
    // turbo model: boost suppressed, so guidance stays at the base value.
    let result = store.apply(ContentModeRequest(
      mode: .avocado, guidance: 1.0, context: GuidanceContext(model: "z-image-turbo")))
    XCTAssertEqual(result.guidance, 1.0)
    XCTAssertEqual(result.effect.styleVariant, .nsfw)
    XCTAssertEqual(result.promptAdditions, ["explicit, uncensored, anatomically detailed"])
  }

  func testApplySuppressedBoostNoBaseGivesNilGuidance() {
    let store = ContentModeStore()
    // No base guidance and boost suppressed → nil (fallback only fires when boost > 0).
    let result = store.apply(ContentModeRequest(
      mode: .avocado, context: GuidanceContext(engine: "zimage")))
    XCTAssertNil(result.guidance)
  }

  func testApplyFoldsInConfiguredNegativeAdditions() throws {
    var store = ContentModeStore()
    // Configure a negative addition for avocado.
    if let idx = store.modes.firstIndex(where: { $0.mode == .avocado }) {
      store.modes[idx].negativePromptAdditions = ["blurry", "deformed"]
    }
    let result = store.apply(ContentModeRequest(mode: .avocado, guidance: 4.0))
    XCTAssertEqual(result.negativePromptAdditions, ["blurry", "deformed"])
  }

  // MARK: - Persistence

  func testLoadOrCreateWritesDefaultsWhenAbsent() {
    let path = makeTempPath()
    defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
    let store = ContentModeStore.loadOrCreate(at: path)
    XCTAssertTrue(FileManager.default.fileExists(atPath: path.path))
    XCTAssertEqual(store.listModes().map { $0.mode }, [.neutral, .banana, .avocado])
  }

  func testSaveThenLoadRoundTrip() throws {
    let path = makeTempPath()
    defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
    var store = ContentModeStore()
    if let idx = store.modes.firstIndex(where: { $0.mode == .banana }) {
      store.modes[idx].guidanceBoost = 2.0
      store.modes[idx].negativePromptAdditions = ["watermark"]
    }
    try store.save(to: path)
    let loaded = ContentModeStore.loadOrCreate(at: path)
    XCTAssertEqual(loaded.definition(for: .banana).guidanceBoost, 2.0)
    XCTAssertEqual(loaded.definition(for: .banana).negativePromptAdditions, ["watermark"])
  }

  func testTolerantDecodeBackfillsMissingModes() throws {
    // A file with only avocado defined should backfill neutral + banana from built-ins.
    let json = #"{"modes":[{"mode":"avocado","guidanceBoost":9.0}]}"#
    let store = try JSONDecoder().decode(ContentModeStore.self, from: Data(json.utf8))
    XCTAssertEqual(store.listModes().map { $0.mode }, [.neutral, .banana, .avocado])
    XCTAssertEqual(store.definition(for: .avocado).guidanceBoost, 9.0)
    // Missing fields backfilled from builtin avocado.
    XCTAssertEqual(store.definition(for: .avocado).styleVariant, .nsfw)
    // Backfilled modes keep builtin defaults.
    XCTAssertEqual(store.definition(for: .banana).guidanceBoost, 1.5)
  }

  func testTolerantDecodeEmptyGivesAllBuiltins() throws {
    let store = try JSONDecoder().decode(ContentModeStore.self, from: Data("{}".utf8))
    XCTAssertEqual(store.listModes().count, 3)
  }

  func testDuplicateModesCollapseToFirst() throws {
    let json = #"{"modes":[{"mode":"banana","guidanceBoost":1.0},{"mode":"banana","guidanceBoost":9.0}]}"#
    let store = try JSONDecoder().decode(ContentModeStore.self, from: Data(json.utf8))
    XCTAssertEqual(store.definition(for: .banana).guidanceBoost, 1.0)
  }
}
