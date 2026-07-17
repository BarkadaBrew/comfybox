import XCTest
@testable import ComfyBoxDesktop

final class KiraClientTests: XCTestCase {

  /// Real kira-daemon /health payload captured 2026-07-16.
  private let livePayload = """
    {"status":"ok","name":"kira","energy":1,"isRunning":true,"isPaused":false,
     "persona":{"role":"companion","stateNamespace":"Kira"},
     "network":{"role":"companion","listeners":[
       {"name":"api","address":"127.0.0.1","port":3787,"family":"IPv4"}]},
     "renderControls":{"autonomousRenderEnabled":false,
       "stateFile":"/home/todd/.kira/render-controls.json","blockedCount":0},
     "tools":["pipeline_status","vault_read","vault_write"]}
    """

  func testHealthSnapshotParsesLivePayload() throws {
    let snapshot = try XCTUnwrap(KiraHealthSnapshot.parse(Data(livePayload.utf8)))
    XCTAssertEqual(snapshot.status, "ok")
    XCTAssertEqual(snapshot.name, "kira")
    XCTAssertTrue(snapshot.isRunning)
    XCTAssertFalse(snapshot.isPaused)
    XCTAssertEqual(snapshot.energy, 1.0)
    XCTAssertEqual(snapshot.autonomousRenderEnabled, false)
    XCTAssertEqual(snapshot.toolCount, 3)
  }

  func testHealthSnapshotTolerantOfMissingFields() throws {
    let snapshot = try XCTUnwrap(KiraHealthSnapshot.parse(Data(#"{"status":"ok"}"#.utf8)))
    XCTAssertEqual(snapshot.status, "ok")
    XCTAssertFalse(snapshot.isRunning)
    XCTAssertNil(snapshot.energy)
    XCTAssertNil(snapshot.autonomousRenderEnabled)
    XCTAssertNil(snapshot.toolCount)
    // Non-JSON is a nil, not a crash or a phantom value.
    XCTAssertNil(KiraHealthSnapshot.parse(Data("plainly not json".utf8)))
  }

  func testBindingURLAndLocality() {
    // Default is loopback — correct for the interim SSH tunnel AND for the
    // post-migration local daemon (no binding change when Kira moves home).
    let binding = KiraHostBinding()
    XCTAssertEqual(binding.host, "127.0.0.1")
    XCTAssertEqual(binding.port, 3787)
    XCTAssertEqual(binding.baseURL?.absoluteString, "http://127.0.0.1:3787")
    XCTAssertTrue(binding.isLocal)

    let remote = KiraHostBinding(host: "10.0.100.232", port: 3787)
    XCTAssertFalse(remote.isLocal)
  }
}
