import XCTest
@testable import ZImage

/// Verifies PromptRepositoryStore's upsert-by-composite-key semantics,
/// filtering, and atomic persistence (#234). Mirrors QueuePersistenceTests'
/// isolation convention exactly — every path resolves inside a per-test temp
/// directory via `isolateComfyBoxStateDirectory()`, never the LIVE
/// ~/.comfybox/prompt-repository.json.
final class PromptRepositoryStoreTests: XCTestCase {

  private var stateDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    stateDirectory = try isolateComfyBoxStateDirectory()
  }

  private func entry(
    modelId: Int, versionId: Int, name: String = "Some LoRA", baseModel: String = "Z-Image",
    act: String? = nil, trainedWords: [String] = [], tags: [String] = []
  ) -> PromptRepositoryEntry {
    PromptRepositoryEntry(
      sourceModelId: modelId, sourceVersionId: versionId, modelName: name, baseModel: baseModel,
      actTaxonomy: act, trainedWords: trainedWords, tags: tags)
  }

  // MARK: - Upsert

  func testUpsertCreatesNewEntries() {
    let result = PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 10),
      entry(modelId: 2, versionId: 20),
    ])
    XCTAssertEqual(result, PromptRepositoryStore.UpsertResult(created: 2, updated: 0))
    XCTAssertEqual(PromptRepositoryStore.loadAll().count, 2)
  }

  func testUpsertByCompositeKeyUpdatesInPlaceRatherThanDuplicating() {
    PromptRepositoryStore.upsert([entry(modelId: 1, versionId: 10, trainedWords: ["old"])])
    let result = PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 10, name: "Renamed LoRA", trainedWords: ["new", "words"]),
    ])
    XCTAssertEqual(result, PromptRepositoryStore.UpsertResult(created: 0, updated: 1))

    let all = PromptRepositoryStore.loadAll()
    XCTAssertEqual(all.count, 1, "re-harvesting the same (modelId, versionId) must not duplicate")
    XCTAssertEqual(all.first?.id, "1-10")
    XCTAssertEqual(all.first?.modelName, "Renamed LoRA")
    XCTAssertEqual(all.first?.trainedWords, ["new", "words"])
  }

  func testUpsertIdIsTheCompositeKey() {
    PromptRepositoryStore.upsert([entry(modelId: 42, versionId: 7)])
    XCTAssertEqual(PromptRepositoryStore.loadAll().first?.id, "42-7")
  }

  func testUpsertOfEmptyBatchIsANoOp() {
    PromptRepositoryStore.upsert([entry(modelId: 1, versionId: 10)])
    let result = PromptRepositoryStore.upsert([])
    XCTAssertEqual(result, PromptRepositoryStore.UpsertResult(created: 0, updated: 0))
    XCTAssertEqual(PromptRepositoryStore.loadAll().count, 1)
  }

  func testUpsertPreservesExistingEntriesNotInTheIncomingBatch() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 10),
      entry(modelId: 2, versionId: 20),
    ])
    PromptRepositoryStore.upsert([entry(modelId: 1, versionId: 10, name: "Updated")])
    let all = PromptRepositoryStore.loadAll()
    XCTAssertEqual(all.count, 2)
    XCTAssertTrue(all.contains { $0.id == "2-20" })
  }

  // MARK: - Query

  func testQueryFiltersByBaseModelCaseInsensitively() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 1, baseModel: "Z-Image"),
      entry(modelId: 2, versionId: 2, baseModel: "SDXL 1.0"),
    ])
    let results = PromptRepositoryStore.query(baseModel: "z-image")
    XCTAssertEqual(results.map(\.id), ["1-1"])
  }

  func testQueryFiltersByAct() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 1, act: "pose"),
      entry(modelId: 2, versionId: 2, act: "style"),
      entry(modelId: 3, versionId: 3, act: nil),
    ])
    XCTAssertEqual(PromptRepositoryStore.query(act: "pose").map(\.id), ["1-1"])
  }

  func testQueryFiltersByTag() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 1, tags: ["anime", "portrait"]),
      entry(modelId: 2, versionId: 2, tags: ["realistic"]),
    ])
    XCTAssertEqual(PromptRepositoryStore.query(tag: "ANIME").map(\.id), ["1-1"])
  }

  func testQueryFiltersByKeywordAcrossNameTrainedWordsAndTags() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 1, name: "Sunset Glow LoRA", trainedWords: ["glow"]),
      entry(modelId: 2, versionId: 2, name: "Unrelated", trainedWords: [], tags: ["glowing skin"]),
      entry(modelId: 3, versionId: 3, name: "Nothing matches here"),
    ])
    let results = PromptRepositoryStore.query(keyword: "glow")
    XCTAssertEqual(Set(results.map(\.id)), Set(["1-1", "2-2"]))
  }

  func testQueryWithNoFiltersReturnsEverything() {
    PromptRepositoryStore.upsert([entry(modelId: 1, versionId: 1), entry(modelId: 2, versionId: 2)])
    XCTAssertEqual(PromptRepositoryStore.query().count, 2)
  }

  func testQueryCombinesMultipleFiltersWithAND() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 1, baseModel: "Z-Image", act: "pose"),
      entry(modelId: 2, versionId: 2, baseModel: "Z-Image", act: "style"),
      entry(modelId: 3, versionId: 3, baseModel: "SDXL 1.0", act: "pose"),
    ])
    XCTAssertEqual(
      PromptRepositoryStore.query(baseModel: "Z-Image", act: "pose").map(\.id), ["1-1"])
  }

  // MARK: - Persistence / atomicity

  func testPersistenceRoundTripsAcrossReload() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 1, name: "A", trainedWords: ["a", "b"], tags: ["x"]),
    ])
    // Simulate a fresh process by re-reading straight off disk.
    let reloaded = PromptRepositoryStore.loadAll()
    XCTAssertEqual(reloaded.count, 1)
    XCTAssertEqual(reloaded.first?.modelName, "A")
    XCTAssertEqual(reloaded.first?.trainedWords, ["a", "b"])
    XCTAssertEqual(reloaded.first?.tags, ["x"])
  }

  func testLoadWithNoFileReturnsEmpty() {
    try? FileManager.default.removeItem(at: PromptRepositoryStore.path)
    XCTAssertEqual(PromptRepositoryStore.loadAll(), [])
  }

  /// Repeated upserts (simulating repeated harvests) never corrupt the file —
  /// every intermediate state loads back cleanly, matching
  /// QueuePersistenceTests' round-trip-after-every-mutation style.
  func testRepeatedUpsertsNeverCorruptTheFile() {
    for i in 0..<25 {
      PromptRepositoryStore.upsert([entry(modelId: i, versionId: i, name: "Model \(i)")])
      let all = PromptRepositoryStore.loadAll()
      XCTAssertEqual(all.count, i + 1, "corrupted or lost state after upsert #\(i)")
    }
    XCTAssertEqual(PromptRepositoryStore.loadAll().count, 25)
  }

  /// The snapshot this suite writes lands in the temp directory, never in
  /// `~/.comfybox` — asserted on the file that was actually created.
  func testTheSnapshotIsWrittenInsideTheTempDirectory() {
    PromptRepositoryStore.upsert([entry(modelId: 1, versionId: 1)])
    XCTAssertTrue(FileManager.default.fileExists(atPath: PromptRepositoryStore.path.path))
    XCTAssertEqual(
      PromptRepositoryStore.path.deletingLastPathComponent().standardizedFileURL.path,
      stateDirectory.path)
  }

  // MARK: - Description truncation

  func testDescriptionExcerptIsTruncatedToTheLimit() {
    let longText = String(repeating: "x", count: PromptRepositoryEntry.descriptionExcerptLimit + 100)
    let entry = PromptRepositoryEntry(
      sourceModelId: 1, sourceVersionId: 1, modelName: "A", baseModel: "Z-Image",
      descriptionExcerpt: longText)
    XCTAssertEqual(entry.descriptionExcerpt?.count, PromptRepositoryEntry.descriptionExcerptLimit)
  }

  func testDescriptionExcerptShorterThanLimitIsUnchanged() {
    let entry = PromptRepositoryEntry(
      sourceModelId: 1, sourceVersionId: 1, modelName: "A", baseModel: "Z-Image",
      descriptionExcerpt: "short")
    XCTAssertEqual(entry.descriptionExcerpt, "short")
  }

  func testEmptyDescriptionExcerptBecomesNil() {
    let entry = PromptRepositoryEntry(
      sourceModelId: 1, sourceVersionId: 1, modelName: "A", baseModel: "Z-Image",
      descriptionExcerpt: "")
    XCTAssertNil(entry.descriptionExcerpt)
  }
}
