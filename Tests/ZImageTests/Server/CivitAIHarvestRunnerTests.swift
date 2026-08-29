import XCTest

@testable import ZImage

/// The harvest paging loop (#234, adversarial review follow-up), exercised
/// through `CivitAIHarvestRunner.run(request:timeBudget:interPageDelay:
/// fetchPage:upsert:)` — the network (`fetchPage`) and persistence
/// (`upsert`) seams injected, so no live CivitAI calls and no real store
/// writes unless a test wants them.
final class CivitAIHarvestRunnerTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    // Belt and braces: even though every test injects `upsert`, nothing in
    // this suite may ever resolve the LIVE ~/.comfybox.
    try isolateComfyBoxStateDirectory()
  }

  // MARK: - Fixture pages

  /// Decode a fixture `CivitAIModelsPage` (the wire models only expose
  /// `init(from:)`) with `count` single-version models starting at
  /// `startId`, and an optional next cursor.
  private func fixturePage(
    count: Int, startId: Int, versionsPerModel: Int = 1, nextCursor: String?
  ) throws -> CivitAIModelsPage {
    let items = (0..<count).map { i -> String in
      let modelId = startId + i
      let versions = (0..<versionsPerModel).map { v in
        """
        {"id": \(modelId * 100 + v), "name": "v\(v)", "baseModel": "Z-Image",
         "trainedWords": ["word-\(modelId)-\(v)"]}
        """
      }.joined(separator: ",")
      return """
      {"id": \(modelId), "name": "Model \(modelId)", "type": "LORA",
       "modelVersions": [\(versions)]}
      """
    }.joined(separator: ",")
    let metadata = nextCursor.map { #", "metadata": {"nextCursor": "\#($0)"}"# } ?? ""
    let json = #"{"items": [\#(items)]\#(metadata)}"#
    return try JSONDecoder().decode(CivitAIModelsPage.self, from: Data(json.utf8))
  }

  private func request(limit: Int) -> CivitAIHarvestRequestBody {
    var body = CivitAIHarvestRequestBody()
    body.limit = limit
    return body
  }

  // MARK: - Limit clamping (P1-2a)

  func testClampedLimitCapsAt200() {
    XCTAssertEqual(CivitAIHarvestRunner.clampedLimit(10000), 200)
    XCTAssertEqual(CivitAIHarvestRunner.clampedLimit(201), 200)
    XCTAssertEqual(CivitAIHarvestRunner.clampedLimit(200), 200)
    XCTAssertEqual(CivitAIHarvestRunner.clampedLimit(24), 24)
    XCTAssertEqual(CivitAIHarvestRunner.clampedLimit(0), 1)
    XCTAssertEqual(CivitAIHarvestRunner.clampedLimit(-5), 1)
    XCTAssertEqual(CivitAIHarvestRunner.maxModelsPerHarvest, 200)
  }

  /// limit=10000 through the actual loop: the run scans exactly 200 models
  /// (2 pages of 100) no matter how many more the fake upstream offers.
  func testRunWithHugeLimitStopsAtTheServerSideCap() async throws {
    var pageRequests: [(cursor: String?, pageLimit: Int)] = []
    var upsertBatchSizes: [Int] = []

    let summary = try await CivitAIHarvestRunner.run(
      request: request(limit: 10000),
      interPageDelay: 0,
      fetchPage: { cursor, pageLimit in
        pageRequests.append((cursor, pageLimit))
        // Upstream always has more: every page is full and offers a cursor.
        let start = pageRequests.count * 1000
        return try self.fixturePage(count: pageLimit, startId: start, nextCursor: "cursor-\(start)")
      },
      upsert: { entries in
        upsertBatchSizes.append(entries.count)
        return PromptRepositoryStore.UpsertResult(created: entries.count, updated: 0)
      })

    XCTAssertEqual(summary.modelsScanned, 200, "clamped to maxModelsPerHarvest, not 10000")
    XCTAssertEqual(summary.versionsHarvested, 200)
    XCTAssertEqual(summary.created, 200)
    XCTAssertFalse(summary.truncated)
    XCTAssertEqual(pageRequests.map(\.pageLimit), [100, 100], "per-page limit stays <= 100")
    XCTAssertEqual(upsertBatchSizes, [100, 100])
  }

  // MARK: - Per-page upsert (P1-2b)

  /// Three-page fixture: entries must be persisted page-by-page — one upsert
  /// call per page, each bounded to that page's entries, with the store
  /// state observably growing BETWEEN pages (not one terminal upsert).
  func testEntriesAreUpsertedPerPageNotAccumulatedAcrossPages() async throws {
    // 3 pages x 10 models x 2 versions each = 60 entries total.
    var storeSizeWhenPageWasFetched: [Int] = []
    var persisted = 0
    var upsertBatchSizes: [Int] = []
    var fetchCount = 0

    let summary = try await CivitAIHarvestRunner.run(
      request: request(limit: 30),
      interPageDelay: 0,
      fetchPage: { cursor, pageLimit in
        fetchCount += 1
        // Observable seam: how much had already been persisted when each
        // page begins? Pages 2 and 3 must see the previous pages' entries
        // already in the store.
        storeSizeWhenPageWasFetched.append(persisted)
        XCTAssertEqual(pageLimit, min(30 - (fetchCount - 1) * 10, 100))
        let isLast = fetchCount == 3
        return try self.fixturePage(
          count: 10, startId: fetchCount * 1000, versionsPerModel: 2,
          nextCursor: isLast ? nil : "cursor-\(fetchCount)")
      },
      upsert: { entries in
        XCTAssertLessThanOrEqual(
          entries.count, 20, "peak accumulation must be bounded to one page's entries")
        upsertBatchSizes.append(entries.count)
        persisted += entries.count
        return PromptRepositoryStore.UpsertResult(created: entries.count, updated: 0)
      })

    XCTAssertEqual(upsertBatchSizes, [20, 20, 20], "one upsert per page, page-sized")
    XCTAssertEqual(
      storeSizeWhenPageWasFetched, [0, 20, 40],
      "each later page must observe the earlier pages already persisted")
    XCTAssertEqual(summary.modelsScanned, 30)
    XCTAssertEqual(summary.versionsHarvested, 60)
    XCTAssertEqual(summary.created, 60)
    XCTAssertFalse(summary.truncated)
  }

  /// The default `upsert` really is `PromptRepositoryStore.upsert` — a run
  /// against the isolated store persists across pages and dedupes repeats.
  func testRunPersistsIntoThePromptRepositoryStoreByDefault() async throws {
    var fetchCount = 0
    let summary = try await CivitAIHarvestRunner.run(
      request: request(limit: 4),
      interPageDelay: 0,
      fetchPage: { _, _ in
        fetchCount += 1
        // Page 2 repeats page 1's models (same ids) — upsert, not duplicate.
        return try self.fixturePage(
          count: 2, startId: 500, nextCursor: fetchCount == 1 ? "more" : nil)
      })
    XCTAssertEqual(summary.modelsScanned, 4)
    XCTAssertEqual(summary.created, 2)
    XCTAssertEqual(summary.updated, 2)
    XCTAssertEqual(PromptRepositoryStore.loadAll().count, 2)
  }

  // MARK: - Time budget / cancellation (P1-2c)

  func testAnExhaustedTimeBudgetStopsBeforeTheFirstPage() async throws {
    var fetched = false
    let summary = try await CivitAIHarvestRunner.run(
      request: request(limit: 100),
      timeBudget: 0,  // already elapsed
      interPageDelay: 0,
      fetchPage: { _, _ in
        fetched = true
        return try self.fixturePage(count: 1, startId: 1, nextCursor: nil)
      })
    XCTAssertFalse(fetched, "an exhausted budget must not fetch another page")
    XCTAssertTrue(summary.truncated)
    XCTAssertEqual(summary.modelsScanned, 0)
    XCTAssertEqual(summary.versionsHarvested, 0)
  }

  /// Budget expires while page 1 is in flight: page 1's entries are already
  /// upserted (per-page persistence), the loop stops before page 2, and the
  /// summary reports exactly what landed with truncated=true.
  func testTimeBudgetExpiryMidRunReturnsWhatWasHarvestedSoFar() async throws {
    var upsertBatchSizes: [Int] = []
    let summary = try await CivitAIHarvestRunner.run(
      request: request(limit: 100),
      timeBudget: 0.05,
      interPageDelay: 0,
      fetchPage: { _, _ in
        // Slow upstream: by the time this page returns, the budget is gone.
        try await Task.sleep(nanoseconds: 80_000_000)
        return try self.fixturePage(count: 10, startId: 1, nextCursor: "next")
      },
      upsert: { entries in
        upsertBatchSizes.append(entries.count)
        return PromptRepositoryStore.UpsertResult(created: entries.count, updated: 0)
      })
    XCTAssertTrue(summary.truncated)
    XCTAssertEqual(summary.modelsScanned, 10, "counts reflect what was actually harvested")
    XCTAssertEqual(summary.created, 10)
    XCTAssertEqual(upsertBatchSizes, [10], "page 1 was persisted before the stop")
  }

  func testTaskCancellationStopsTheHarvestWithPersistedPagesIntact() async throws {
    var upsertCalls = 0
    let summary = try await CivitAIHarvestRunner.run(
      request: request(limit: 100),
      interPageDelay: 0,
      fetchPage: { _, _ in
        // Cancel the surrounding task from inside page 1's fetch — the loop
        // must notice before fetching page 2.
        withUnsafeCurrentTask { $0?.cancel() }
        return try self.fixturePage(count: 5, startId: 1, nextCursor: "next")
      },
      upsert: { entries in
        upsertCalls += 1
        return PromptRepositoryStore.UpsertResult(created: entries.count, updated: 0)
      })
    XCTAssertTrue(summary.truncated)
    XCTAssertEqual(summary.modelsScanned, 5)
    XCTAssertEqual(upsertCalls, 1, "page 1 persisted; no page 2")
  }

  func testDefaultTimeBudgetIsSixtySeconds() {
    XCTAssertEqual(CivitAIHarvestRunner.timeBudgetSeconds, 60)
  }
}
