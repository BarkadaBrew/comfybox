import XCTest
@testable import ZImage

/// Route param parsing for the three new /v1/civitai/* routes (#234) —
/// independent of hitting real CivitAI, matching the style of
/// WarmServerRejectionTests (pure parsing/validation functions tested
/// directly, not the fileprivate `respond(to:)` dispatch itself).
final class CivitAIConduitRouteParsingTests: XCTestCase {

  // MARK: - GET /v1/civitai/search query parsing

  func testSearchQueryDefaultsWhenEmpty() {
    let q = CivitAISearchQuery(queryParameters: [:])
    XCTAssertEqual(q.query, "")
    XCTAssertEqual(q.types, [])
    XCTAssertNil(q.baseModel)
    XCTAssertEqual(q.sort, .mostDownloaded)
    XCTAssertEqual(q.period, .allTime)
    XCTAssertFalse(q.nsfw)
    XCTAssertNil(q.cursor)
    XCTAssertEqual(q.limit, 24)
    XCTAssertEqual(q.site, "civitai.com")
  }

  func testSearchQueryParsesAllFields() {
    let q = CivitAISearchQuery(queryParameters: [
      "query": "anime",
      "types": "LORA,Checkpoint",
      "base_model": "Z-Image",
      "sort": "Newest",
      "period": "Week",
      "nsfw": "true",
      "cursor": "abc123",
      "limit": "50",
      "site": "civitai.red",
    ])
    XCTAssertEqual(q.query, "anime")
    XCTAssertEqual(q.types, ["LORA", "Checkpoint"])
    XCTAssertEqual(q.baseModel, "Z-Image")
    XCTAssertEqual(q.sort, .newest)
    XCTAssertEqual(q.period, .week)
    XCTAssertTrue(q.nsfw)
    XCTAssertEqual(q.cursor, "abc123")
    XCTAssertEqual(q.limit, 50)
    XCTAssertEqual(q.site, "civitai.red")
    XCTAssertEqual(q.baseURL.absoluteString, "https://civitai.red")
  }

  func testSearchQueryDecodesPercentEncodedFreeText() {
    // MCPToolExecutor percent-encodes free-text params (a raw space can't
    // survive an HTTP request line); the query struct must decode it back.
    let q = CivitAISearchQuery(queryParameters: ["query": "anime%20girl"])
    XCTAssertEqual(q.query, "anime girl")
  }

  func testSearchQuerySortParsingIsLenientToCaseAndSeparators() {
    for raw in ["most_liked", "Most Liked", "MOST-LIKED", "most liked"] {
      XCTAssertEqual(
        CivitAISearchQuery(queryParameters: ["sort": raw]).sort, .mostLiked, "failed for '\(raw)'")
    }
  }

  func testSearchQueryUnknownSortFallsBackToDefault() {
    XCTAssertEqual(CivitAISearchQuery(queryParameters: ["sort": "not_a_real_sort"]).sort, .mostDownloaded)
  }

  func testSearchQueryPeriodParsingIsLenientToCaseAndSeparators() {
    for raw in ["all_time", "AllTime", "all-time"] {
      XCTAssertEqual(
        CivitAISearchQuery(queryParameters: ["period": raw]).period, .allTime, "failed for '\(raw)'")
    }
    XCTAssertEqual(CivitAISearchQuery(queryParameters: ["period": "month"]).period, .month)
  }

  func testSearchQueryTypesSplitsOnCommasAndTrimsWhitespace() {
    let q = CivitAISearchQuery(queryParameters: ["types": " LORA , Checkpoint ,,"])
    XCTAssertEqual(q.types, ["LORA", "Checkpoint"])
  }

  func testSearchQueryEmptyStringsBecomeNilNotEmpty() {
    let q = CivitAISearchQuery(queryParameters: ["base_model": "", "cursor": "", "site": ""])
    XCTAssertNil(q.baseModel)
    XCTAssertNil(q.cursor)
    XCTAssertEqual(q.site, "civitai.com")
  }

  // MARK: - POST /v1/civitai/harvest body parsing

  private func decodeHarvestBody(_ json: String) throws -> CivitAIHarvestRequestBody {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(CivitAIHarvestRequestBody.self, from: Data(json.utf8))
  }

  func testHarvestRequestBodyDefaultsOnEmptyObject() throws {
    let body = try decodeHarvestBody("{}")
    XCTAssertEqual(body, CivitAIHarvestRequestBody())
  }

  func testHarvestRequestBodyDecodesSnakeCaseKeys() throws {
    let body = try decodeHarvestBody(#"""
    {"query":"portrait","types":["LORA"],"base_model":"Z-Image","sort":"Highest Rated",
     "period":"Month","nsfw":true,"limit":100,"site":"civitai.red"}
    """#)
    XCTAssertEqual(body.query, "portrait")
    XCTAssertEqual(body.types, ["LORA"])
    XCTAssertEqual(body.baseModel, "Z-Image")
    XCTAssertEqual(body.sort, "Highest Rated")
    XCTAssertEqual(body.period, "Month")
    XCTAssertTrue(body.nsfw)
    XCTAssertEqual(body.limit, 100)
    XCTAssertEqual(body.site, "civitai.red")
    XCTAssertEqual(body.resolvedBaseURL.absoluteString, "https://civitai.red")
  }

  func testHarvestRequestBodyToleratesUnknownExtraFields() throws {
    let body = try decodeHarvestBody(#"{"query":"x","unexpected_field":123}"#)
    XCTAssertEqual(body.query, "x")
  }

  func testHarvestRequestBodySortAndPeriodParseThroughTheSameLenientLogicAsSearch() throws {
    let body = try decodeHarvestBody(#"{"sort":"most_liked","period":"all_time"}"#)
    XCTAssertEqual(CivitAISortPeriodParsing.parseSort(body.sort), .mostLiked)
    XCTAssertEqual(CivitAISortPeriodParsing.parsePeriod(body.period), .allTime)
  }

  // MARK: - GET /v1/civitai/repo query parsing

  func testRepoQueryDefaultsToAllNilWhenEmpty() {
    let q = CivitAIRepoQuery(queryParameters: [:])
    XCTAssertNil(q.baseModel)
    XCTAssertNil(q.act)
    XCTAssertNil(q.tag)
    XCTAssertNil(q.keyword)
  }

  func testRepoQueryParsesAllFilters() {
    let q = CivitAIRepoQuery(queryParameters: [
      "base_model": "Z-Image", "act": "pose", "tag": "anime", "keyword": "sunset",
    ])
    XCTAssertEqual(q.baseModel, "Z-Image")
    XCTAssertEqual(q.act, "pose")
    XCTAssertEqual(q.tag, "anime")
    XCTAssertEqual(q.keyword, "sunset")
  }

  func testRepoQueryEmptyStringsBecomeNil() {
    let q = CivitAIRepoQuery(queryParameters: ["base_model": "", "act": "", "tag": "", "keyword": ""])
    XCTAssertNil(q.baseModel)
    XCTAssertNil(q.act)
    XCTAssertNil(q.tag)
    XCTAssertNil(q.keyword)
  }

  func testRepoQueryDecodesPercentEncodedKeyword() {
    let q = CivitAIRepoQuery(queryParameters: ["keyword": "anime%20girl"])
    XCTAssertEqual(q.keyword, "anime girl")
  }
}
