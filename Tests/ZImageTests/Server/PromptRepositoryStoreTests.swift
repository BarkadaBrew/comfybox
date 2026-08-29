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
    act: String? = nil, trainedWords: [String] = [], tags: [String] = [],
    harvestedAt: Date = Date()
  ) -> PromptRepositoryEntry {
    PromptRepositoryEntry(
      sourceModelId: modelId, sourceVersionId: versionId, modelName: name, baseModel: baseModel,
      actTaxonomy: act, trainedWords: trainedWords, tags: tags, harvestedAt: harvestedAt)
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

  // MARK: - Duplicate-id self-heal (adversarial review P2: no trapping)

  /// Write a persisted file that carries TWO entries with the same id — the
  /// exact shape that made `Dictionary(uniqueKeysWithValues:)` trap at
  /// runtime. Loading must not crash, the LAST entry must win, and the next
  /// save must persist the deduped form.
  func testAPersistedFileWithDuplicateIdsDoesNotTrapAndLastWins() throws {
    let json = """
    {"entries": [
      {"id": "1-10", "sourceModelId": 1, "sourceVersionId": 10, "modelName": "Stale Copy",
       "baseModel": "Z-Image", "trainedWords": ["stale"], "tags": [],
       "harvestedAt": "2026-08-01T00:00:00Z"},
      {"id": "2-20", "sourceModelId": 2, "sourceVersionId": 20, "modelName": "Innocent Bystander",
       "baseModel": "Z-Image", "trainedWords": [], "tags": [],
       "harvestedAt": "2026-08-02T00:00:00Z"},
      {"id": "1-10", "sourceModelId": 1, "sourceVersionId": 10, "modelName": "Fresh Copy",
       "baseModel": "Z-Image", "trainedWords": ["fresh"], "tags": [],
       "harvestedAt": "2026-08-03T00:00:00Z"}
    ]}
    """
    try Data(json.utf8).write(to: PromptRepositoryStore.path)

    // Load: no crash, dedupe applied, last writer won.
    let loaded = PromptRepositoryStore.loadAll()
    XCTAssertEqual(loaded.count, 2, "duplicate id must collapse to one entry")
    XCTAssertEqual(loaded.map(\.id), ["1-10", "2-20"], "first position, last value")
    XCTAssertEqual(loaded.first?.modelName, "Fresh Copy")
    XCTAssertEqual(loaded.first?.trainedWords, ["fresh"])

    // Upsert over the corrupt file: also no crash, and the subsequent save
    // self-heals — re-reading the raw file shows the dup is gone for good.
    PromptRepositoryStore.upsert([entry(modelId: 3, versionId: 30)])
    let raw = try Data(contentsOf: PromptRepositoryStore.path)
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let healed = try decoder.decode(PromptRepositoryState.self, from: raw)
    XCTAssertEqual(healed.entries.count, 3)
    XCTAssertEqual(
      healed.entries.map(\.id).sorted(), ["1-10", "2-20", "3-30"],
      "the saved file must be deduped, not just the in-memory view")
  }

  /// An incoming batch that repeats an id (same model version twice in one
  /// page) must not trap either — last occurrence wins, counted once.
  func testAnIncomingBatchWithDuplicateIdsDoesNotTrapAndLastWins() {
    let result = PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 10, name: "First Occurrence"),
      entry(modelId: 1, versionId: 10, name: "Second Occurrence"),
    ])
    XCTAssertEqual(result, PromptRepositoryStore.UpsertResult(created: 1, updated: 0))
    let all = PromptRepositoryStore.loadAll()
    XCTAssertEqual(all.count, 1)
    XCTAssertEqual(all.first?.modelName, "Second Occurrence")
  }

  func testDedupedLastWinsKeepsFirstPositionWithLastValue() {
    let entries = [
      entry(modelId: 1, versionId: 1, name: "A-old"),
      entry(modelId: 2, versionId: 2, name: "B"),
      entry(modelId: 1, versionId: 1, name: "A-new"),
      entry(modelId: 3, versionId: 3, name: "C"),
    ]
    let deduped = PromptRepositoryStore.dedupedLastWins(entries)
    XCTAssertEqual(deduped.map(\.id), ["1-1", "2-2", "3-3"])
    XCTAssertEqual(deduped.first?.modelName, "A-new")
  }

  // MARK: - Concurrency (adversarial review P2: serialized writes)

  /// Concurrent upserts (two harvests' per-page batches racing) must all
  /// land — the pre-lock load→merge→save race silently dropped whole
  /// batches. 8 tasks x 25 distinct entries: every one must be present.
  func testConcurrentUpsertsAllLandWithoutClobberingEachOther() async {
    let tasks = 8
    let perTask = 25
    await withTaskGroup(of: Void.self) { group in
      for t in 0..<tasks {
        group.addTask {
          for i in 0..<perTask {
            PromptRepositoryStore.upsert([
              PromptRepositoryEntry(
                sourceModelId: 1000 + t, sourceVersionId: i,
                modelName: "Racer \(t)", baseModel: "Z-Image")
            ])
          }
        }
      }
      await group.waitForAll()
    }
    let all = PromptRepositoryStore.loadAll()
    XCTAssertEqual(
      all.count, tasks * perTask,
      "lost update: concurrent upserts clobbered each other's batches")
    for t in 0..<tasks {
      for i in 0..<perTask {
        XCTAssertTrue(
          all.contains { $0.id == "\(1000 + t)-\(i)" },
          "entry \(1000 + t)-\(i) was lost in the race")
      }
    }
  }

  // MARK: - Store cap / eviction (adversarial review P2: unbounded growth)

  func testUpsertEvictsOldestHarvestedAtFirstWhenOverCap() {
    let base = Date(timeIntervalSince1970: 1_700_000_000)
    // Seed 5 entries, oldest first (cap for the test is 5; default 5000).
    PromptRepositoryStore.upsert(
      (0..<5).map { entry(modelId: $0, versionId: $0, harvestedAt: base.addingTimeInterval(Double($0))) },
      cap: 5)
    XCTAssertEqual(PromptRepositoryStore.loadAll().count, 5)

    // Two fresh entries push it to 7 — the two OLDEST (0-0, 1-1) must go.
    PromptRepositoryStore.upsert(
      [
        entry(modelId: 100, versionId: 100, harvestedAt: base.addingTimeInterval(100)),
        entry(modelId: 101, versionId: 101, harvestedAt: base.addingTimeInterval(101)),
      ],
      cap: 5)
    let all = PromptRepositoryStore.loadAll()
    XCTAssertEqual(all.count, 5, "the store must be capped, not grow forever")
    let ids = Set(all.map(\.id))
    XCTAssertFalse(ids.contains("0-0"), "oldest entry must be evicted first")
    XCTAssertFalse(ids.contains("1-1"), "second-oldest entry must be evicted next")
    XCTAssertTrue(ids.isSuperset(of: ["2-2", "3-3", "4-4", "100-100", "101-101"]))
  }

  func testUpsertAtExactlyTheCapEvictsNothing() {
    let base = Date(timeIntervalSince1970: 1_700_000_000)
    PromptRepositoryStore.upsert(
      (0..<5).map { entry(modelId: $0, versionId: $0, harvestedAt: base.addingTimeInterval(Double($0))) },
      cap: 5)
    XCTAssertEqual(Set(PromptRepositoryStore.loadAll().map(\.id)),
                   Set(["0-0", "1-1", "2-2", "3-3", "4-4"]))
  }

  func testDefaultCapIs5000() {
    XCTAssertEqual(PromptRepositoryStore.defaultMaxEntries, 5000)
  }

  // MARK: - Query result cap (adversarial review P2: repo route)

  func testQueryLimitCapsResults() {
    PromptRepositoryStore.upsert((0..<10).map { entry(modelId: $0, versionId: $0) })
    XCTAssertEqual(PromptRepositoryStore.query(limit: 3).count, 3)
    XCTAssertEqual(PromptRepositoryStore.query(limit: 100).count, 10, "limit above size is harmless")
    XCTAssertEqual(PromptRepositoryStore.query().count, 10, "nil limit preserves old behavior")
  }

  func testQueryLimitAppliesAfterFiltering() {
    PromptRepositoryStore.upsert([
      entry(modelId: 1, versionId: 1, baseModel: "Z-Image"),
      entry(modelId: 2, versionId: 2, baseModel: "SDXL 1.0"),
      entry(modelId: 3, versionId: 3, baseModel: "Z-Image"),
      entry(modelId: 4, versionId: 4, baseModel: "Z-Image"),
    ])
    let results = PromptRepositoryStore.query(baseModel: "Z-Image", limit: 2)
    XCTAssertEqual(results.count, 2)
    XCTAssertTrue(results.allSatisfy { $0.baseModel == "Z-Image" })
  }
}
