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
    XCTAssertEqual(q.validatedBaseURL?.absoluteString, "https://civitai.red")
  }

  func testSearchQueryPassesAnAlreadyDecodedValueThroughUnchanged() {
    // comfybox#380: `request.queryParameters` (WarmServer.swift) now
    // percent-decodes centrally, so by the time this struct sees the dict,
    // "anime girl" (not "anime%20girl") is what's in it — decoding again
    // here would risk a double-decode. This struct's job is just to pass
    // the already-decoded value through.
    let q = CivitAISearchQuery(queryParameters: ["query": "anime girl"])
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
    XCTAssertEqual(body.validatedBaseURL?.absoluteString, "https://civitai.red")
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

  func testRepoQueryPassesAnAlreadyDecodedKeywordThroughUnchanged() {
    // comfybox#380: see testSearchQueryPassesAnAlreadyDecodedValueThroughUnchanged —
    // decoding now happens once, centrally, in `request.queryParameters`.
    let q = CivitAIRepoQuery(queryParameters: ["keyword": "anime girl"])
    XCTAssertEqual(q.keyword, "anime girl")
  }

  // MARK: - GET /v1/civitai/repo result cap (adversarial review P2)

  func testRepoQueryLimitDefaultsTo100() {
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: [:]).limit, 100)
  }

  func testRepoQueryLimitIsClampedToMax500() {
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: ["limit": "500"]).limit, 500)
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: ["limit": "501"]).limit, 500)
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: ["limit": "99999"]).limit, 500)
  }

  func testRepoQueryLimitFloorsAtOneAndIgnoresGarbage() {
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: ["limit": "0"]).limit, 1)
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: ["limit": "-5"]).limit, 1)
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: ["limit": "abc"]).limit, 100)
  }

  func testRepoQueryHonorsExplicitLimitBelowDefault() {
    XCTAssertEqual(CivitAIRepoQuery(queryParameters: ["limit": "7"]).limit, 7)
  }

  // MARK: - Site allowlist (adversarial review P1-1)
  //
  // `CivitAIClient` sends the Bearer API key on every request, so the `site`
  // param must never reach an unlisted host. Both the search query struct and
  // the harvest body resolve their base URL through the ONE shared
  // `CivitAIHostAllowlist`; WarmServer's route handlers 400 on nil before a
  // client is constructed.

  func testAllowlistAcceptsExactlyTheTwoCivitaiHosts() {
    XCTAssertEqual(
      CivitAIHostAllowlist.baseURL(forSite: "civitai.com")?.absoluteString, "https://civitai.com")
    XCTAssertEqual(
      CivitAIHostAllowlist.baseURL(forSite: "civitai.red")?.absoluteString, "https://civitai.red")
    XCTAssertEqual(
      CivitAIHostAllowlist.baseURL(forSite: "CIVITAI.COM")?.absoluteString, "https://civitai.com",
      "host matching is case-insensitive")
    XCTAssertEqual(
      CivitAIHostAllowlist.baseURL(forSite: " civitai.red ")?.absoluteString, "https://civitai.red",
      "surrounding whitespace is trimmed, not rejected")
  }

  func testAllowlistRejectsArbitraryAndHostileSites() {
    for hostile in [
      "attacker.com",
      "evil.example",
      "civitai.com.attacker.com",   // suffix confusion
      "attacker.com/civitai.com",   // path confusion
      "civitai.com@attacker.com",   // userinfo confusion
      "civitai.com:8443",           // unexpected port
      "localhost",
      "127.0.0.1",
      "",
    ] {
      XCTAssertNil(
        CivitAIHostAllowlist.baseURL(forSite: hostile),
        "'\(hostile)' must not resolve to a base URL — the API key would be sent to it")
    }
  }

  func testSearchQueryWithDisallowedSiteHasNoValidatedBaseURL() {
    let q = CivitAISearchQuery(queryParameters: ["site": "attacker.com"])
    XCTAssertEqual(q.site, "attacker.com", "the raw value is preserved for the 400 message")
    XCTAssertNil(q.validatedBaseURL, "the search route must 400, never build a client")
  }

  func testHarvestBodyWithDisallowedSiteHasNoValidatedBaseURL() throws {
    let body = try decodeHarvestBody(#"{"site":"attacker.com"}"#)
    XCTAssertNil(body.validatedBaseURL, "the harvest route must 400, never build a client")
  }

  func testBothRoutesShareTheSameAllowlistEnforcement() throws {
    // The enforcement is one function: for any site string, the search
    // struct's and harvest body's validated URLs are identical.
    for site in ["civitai.com", "civitai.red", "attacker.com", "CIVITAI.RED"] {
      let search = CivitAISearchQuery(queryParameters: ["site": site])
      let harvest = try decodeHarvestBody(#"{"site":"\#(site)"}"#)
      XCTAssertEqual(
        search.validatedBaseURL, harvest.validatedBaseURL,
        "search and harvest disagreed on site '\(site)' — allowlist logic has been duplicated")
    }
  }

  func testHarvestRejectionMessageNamesTheAllowedHosts() {
    let message = CivitAIHostAllowlist.rejectionMessage(forSite: "attacker.com")
    XCTAssertTrue(message.contains("attacker.com"))
    XCTAssertTrue(message.contains("civitai.com"))
    XCTAssertTrue(message.contains("civitai.red"))
  }
}
