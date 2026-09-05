// QueryParametersPercentDecodingTests.swift — comfybox#380.
//
// `HTTPRequest.queryParameters` (WarmServer.swift) used to split the raw
// query string on `&`/`=` without percent-decoding at all. A custom model
// path (or any other free-text query value) containing a space, `#`, a
// literal `%`, or non-ASCII text is stat'ed as the literal encoded string —
// `loadable: false` — because the desktop and MCPToolExecutor both
// percent-encode those values before sending them (an unencoded space can't
// survive an HTTP request line at all).
//
// Two layers are tested here:
//   1. `HTTPRequest.queryParameters` directly — the actual decode logic,
//      including the "+ is not a space" and malformed-escape-fallback
//      choices.
//   2. The full round trip through `GET /v1/model/family`, driven with a
//      whole `HTTPRequest` via `WarmServerQueueProbe.syncModelFamilyRoute` —
//      the existing sync-route test harness pattern (see
//      `InterruptTargetTests.testTheSyncRouteHandlesAWholeRequestAndAuditsIt`)
//      — so the test proves the fix at the actual layer the bug lived in,
//      not just at `ModelFamilyDetector.detect` (already covered by
//      `ModelFamilyDetectionTests`, which never touches query parsing).

import XCTest

@testable import ZImage

final class QueryParametersPercentDecodingTests: XCTestCase {

  // MARK: - HTTPRequest.queryParameters, direct

  private func request(query: String?) -> HTTPRequest {
    HTTPRequest(method: "GET", path: "/v1/model/family", queryString: query, headers: [:], body: Data())
  }

  func testDecodesASpaceInAValue() {
    let params = request(query: "model=a%20b").queryParameters
    XCTAssertEqual(params["model"], "a b")
  }

  func testDecodesAHashInAValue() {
    // '#' can never appear literally in a query string sent by a spec-
    // respecting client (it is the fragment delimiter), so a real filename
    // containing one arrives percent-encoded.
    let params = request(query: "model=track%231.safetensors").queryParameters
    XCTAssertEqual(params["model"], "track#1.safetensors")
  }

  func testDecodesALiteralPercentSign() {
    // "%25" is the encoding of a literal "%" — the case the ticket calls out
    // by name, and the one most likely to be double-decoded by mistake.
    let params = request(query: "model=100%25off.safetensors").queryParameters
    XCTAssertEqual(params["model"], "100%off.safetensors")
  }

  func testDecodesNonASCIIText() {
    let params = request(query: "model=caf%C3%A9.safetensors").queryParameters
    XCTAssertEqual(params["model"], "café.safetensors")
  }

  func testPlusIsNotTreatedAsASpace() {
    // Deliberate: this is a URI query per RFC 3986 (`%XX` triplets only), not
    // `application/x-www-form-urlencoded`. `+` is a legal, unreserved-in-
    // practice query character (`CharacterSet.urlQueryAllowed` does not
    // require it to be escaped), so a caller sending a path with a literal
    // `+` never encodes it — decoding it to a space here would corrupt the
    // value FormEncoding would only ever have chosen for a `x-www-form-
    // urlencoded` body, not a query string.
    let params = request(query: "model=z-image%2Bextra.safetensors").queryParameters
    XCTAssertEqual(params["model"], "z-image+extra.safetensors")

    // A literal '+' with no percent-encoding around it at all must also
    // survive unchanged.
    let literal = request(query: "model=a+b").queryParameters
    XCTAssertEqual(literal["model"], "a+b")
  }

  func testDecodesKeysToo() {
    let params = request(query: "mod%65l=krea2").queryParameters
    XCTAssertEqual(params["model"], "krea2")
  }

  func testFallsBackToTheRawSubstringOnAMalformedEscape() {
    // `removingPercentEncoding` returns nil for an incomplete/invalid escape
    // (a trailing `%`, or `%` not followed by two hex digits) — the fix must
    // not turn a bad escape into a dropped/empty parameter.
    let trailing = request(query: "model=abc%").queryParameters
    XCTAssertEqual(trailing["model"], "abc%")

    let nonHex = request(query: "model=50%zz").queryParameters
    XCTAssertEqual(nonHex["model"], "50%zz")
  }

  func testEmptyQueryStringStillYieldsAnEmptyDictionary() {
    XCTAssertEqual(request(query: nil).queryParameters, [:])
    XCTAssertEqual(request(query: "").queryParameters, [:])
  }

  func testAKeyWithNoEqualsSignStillYieldsAnEmptyStringValue() {
    let params = request(query: "flag").queryParameters
    XCTAssertEqual(params["flag"], "")
  }

  // MARK: - Full round trip: GET /v1/model/family

  /// Percent-encode the way the desktop/MCPToolExecutor actually do — RFC
  /// 3986 query encoding (`CharacterSet.urlQueryAllowed`), not form encoding.
  private func percentEncodeQueryValue(_ raw: String) -> String {
    raw.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? raw
  }

  private func routeJSON(_ response: HTTPResponse) throws -> [String: Any] {
    try XCTUnwrap(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
  }

  /// The exact production symptom (comfybox#380): a custom model path with a
  /// space, `#`, a literal `%`, and non-ASCII text — percent-encoded by the
  /// caller as it must be to survive the request line — round-trips through
  /// `GET /v1/model/family` to the real on-disk file, which the route then
  /// reports `loadable: true` for. Before the fix this stat'ed the literal
  /// (still-encoded) string, which never exists on disk, and reported
  /// `loadable: false`.
  func testAModelPathWithSpaceHashPercentAndNonASCIIRoundTripsThroughModelFamily() throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let scratch = FileManager.default.temporaryDirectory
      .appendingPathComponent("model-family-query-decode-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: scratch) }

    let fileName = "odd model #1 100% café.safetensors"
    let file = scratch.appendingPathComponent(fileName)
    FileManager.default.createFile(atPath: file.path, contents: Data([0]))

    let rawSpec = file.path
    let encodedSpec = percentEncodeQueryValue(rawSpec)
    // Sanity: the whole point of the test is that the encoded form differs
    // from the raw form (otherwise this would pass even with no decoding).
    XCTAssertNotEqual(encodedSpec, rawSpec)

    let request = HTTPRequest(
      method: "GET", path: "/v1/model/family", queryString: "model=\(encodedSpec)",
      headers: [:], body: Data())
    let response = probe.syncModelFamilyRoute(request: request)

    XCTAssertEqual(response.status, 200)
    let json = try routeJSON(response)
    XCTAssertEqual(json["model"] as? String, rawSpec, "echoed verbatim, decoded")
    XCTAssertEqual(json["spec"] as? String, file.standardizedFileURL.path)
    XCTAssertEqual(json["loadable"] as? Bool, true, "the on-disk file must resolve as loadable once decoded")
    XCTAssertNil(json["reason"])
  }

  /// Same route, `+` case: a literal `+` in a model path must reach the
  /// filesystem check unchanged, not turned into a space that no longer
  /// names the real file.
  func testAModelPathWithALiteralPlusRoundTripsThroughModelFamilyWithoutBecomingASpace() throws {
    try isolateComfyBoxStateDirectory()
    let probe = makeQueueProbe()

    let scratch = FileManager.default.temporaryDirectory
      .appendingPathComponent("model-family-plus-decode-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: scratch) }

    let fileName = "z-image+extra.safetensors"
    let file = scratch.appendingPathComponent(fileName)
    FileManager.default.createFile(atPath: file.path, contents: Data([0]))

    let rawSpec = file.path
    // `+` is allowed under `.urlQueryAllowed`, so a real caller sends it
    // unescaped — construct the query string directly rather than via
    // `addingPercentEncoding` to pin that exact wire shape.
    let queryString = "model=" + rawSpec.replacingOccurrences(of: " ", with: "%20")

    let request = HTTPRequest(
      method: "GET", path: "/v1/model/family", queryString: queryString, headers: [:], body: Data())
    let response = probe.syncModelFamilyRoute(request: request)

    XCTAssertEqual(response.status, 200)
    let json = try routeJSON(response)
    XCTAssertEqual(json["spec"] as? String, file.standardizedFileURL.path)
    XCTAssertEqual(json["loadable"] as? Bool, true, "a literal '+' must not have become a space")
  }
}
