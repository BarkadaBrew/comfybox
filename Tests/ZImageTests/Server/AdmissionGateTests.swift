import XCTest
@testable import ZImage

/// "Local mode" (Todd 2026-09-15): a runtime admission switch so no automation
/// grabs the engine while a human test runs, with the mode visible to every
/// caller via /health and /v1/queue.
final class AdmissionGateTests: XCTestCase {
  func testBootsOpenAndAdmitsEverything() {
    let g = AdmissionGate()
    XCTAssertEqual(g.snapshot().mode, .open)
    XCTAssertNil(g.snapshot().since)
    XCTAssertNil(g.refusal(method: "POST", path: "/v1/video/generate/async", isLoopbackPeer: false, source: "scheduler"))
    XCTAssertNil(g.refusal(method: "POST", path: "/prompt", isLoopbackPeer: false, source: nil))
  }

  func testLocalModeDefersRemoteSubmitsButAdmitsLoopback() {
    let g = AdmissionGate()
    g.set(mode: .local)
    XCTAssertEqual(g.snapshot().mode, .local)
    XCTAssertNotNil(g.snapshot().since)
    // Remote scheduler → deferred.
    let why = g.refusal(method: "POST", path: "/v1/video/generate/async", isLoopbackPeer: false, source: "scheduler")
    XCTAssertNotNil(why)
    XCTAssertTrue(why!.contains("local mode"))
    // Same route from this Mac → admitted.
    XCTAssertNil(g.refusal(method: "POST", path: "/v1/video/generate/async", isLoopbackPeer: true, source: "ladder-a1"))
    // The ComfyUI bridge submit is gated too.
    XCTAssertNotNil(g.refusal(method: "POST", path: "/prompt", isLoopbackPeer: false, source: nil))
  }

  func testReadRoutesAreNeverGatedSoCallersCanSeeTheMode() {
    let g = AdmissionGate()
    g.set(mode: .local)
    for (m, p) in [("GET", "/health"), ("GET", "/v1/queue"), ("GET", "/v1/queue/admission"),
                   ("GET", "/v1/video/status/abc"), ("GET", "/v1/presets"), ("POST", "/v1/queue/pause"),
                   ("POST", "/v1/queue/admission"), ("POST", "/v1/presets"), ("POST", "/v1/workflows")] {
      XCTAssertNil(g.refusal(method: m, path: p, isLoopbackPeer: false, source: nil), "\(m) \(p) must not be gated")
    }
    XCTAssertNotNil(g.refusal(method: "POST", path: "/v1/workflows/abc/run", isLoopbackPeer: false, source: nil), "a workflow RUN is a submit")
  }

  func testAllowedSourcePrefixLetsANamedRemoteCallerThrough() {
    let g = AdmissionGate()
    g.set(mode: .local, allowSources: [" ladder- ", "", "invest-"])
    XCTAssertEqual(g.snapshot().allowSources, ["ladder-", "invest-"], "trimmed, empties dropped")
    XCTAssertNil(g.refusal(method: "POST", path: "/v1/video/generate/async", isLoopbackPeer: false, source: "ladder-a3"))
    XCTAssertNotNil(g.refusal(method: "POST", path: "/v1/video/generate/async", isLoopbackPeer: false, source: "scheduler"))
    XCTAssertNotNil(g.refusal(method: "POST", path: "/v1/video/generate/async", isLoopbackPeer: false, source: nil))
  }

  func testBackToOpenAdmitsAgain() {
    let g = AdmissionGate()
    g.set(mode: .local)
    g.set(mode: .open)
    XCTAssertEqual(g.snapshot().mode, .open)
    XCTAssertNil(g.refusal(method: "POST", path: "/v1/generate", isLoopbackPeer: false, source: nil))
  }

  func testSubmitSourceExtractionIsLenient() {
    XCTAssertEqual(AdmissionGate.submitSource(from: Data(#"{"prompt":"x","source":"scheduler"}"#.utf8)), "scheduler")
    XCTAssertNil(AdmissionGate.submitSource(from: Data(#"{"prompt":"x"}"#.utf8)))
    XCTAssertNil(AdmissionGate.submitSource(from: Data("not json".utf8)))
    XCTAssertNil(AdmissionGate.submitSource(from: Data()))
  }

  func testRefusalBodyCarriesTheDeferralContract() throws {
    let enc = JSONEncoder(); enc.keyEncodingStrategy = .convertToSnakeCase
    let data = try enc.encode(AdmissionRefusal(admissionMode: "local", error: "x"))
    let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    XCTAssertEqual(obj["success"] as? Bool, false)
    XCTAssertEqual(obj["deferred"] as? Bool, true)
    XCTAssertEqual(obj["admission_mode"] as? String, "local")
  }
}

final class AdmissionGateMCPTests: XCTestCase {
  func testMCPExecutorRecognisesTheLocalModeRefusal() {
    XCTAssertTrue(MCPToolExecutor.isLocalModeRefusal(Data(#"{"success":false,"deferred":true,"admission_mode":"local","error":"x"}"#.utf8)))
    XCTAssertFalse(MCPToolExecutor.isLocalModeRefusal(Data(#"{"error_code":"queue_recovery_in_progress"}"#.utf8)))
    XCTAssertFalse(MCPToolExecutor.isLocalModeRefusal(Data(#"{"admission_mode":"open"}"#.utf8)))
    XCTAssertFalse(MCPToolExecutor.isLocalModeRefusal(Data("nope".utf8)))
  }
}
