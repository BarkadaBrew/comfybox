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

  /// Structure mirrors the live /v1/kira/state payload (2026-07-17), content
  /// sanitized — the wire payload is private conversational state.
  private let statePayload = """
    {"ok":true,"state":{"v":1,
      "now":{"mood":"glowing","energy":"wired","arcPhase":"make the thing"},
      "relationship":{"scores":{"closeness":83.3,"warmth":34.9,"desire":70.9,"playfulness":47.0},
        "dynamic":"companion and photographer peer","facts":["fact one"]},
      "active":{"campaignBeat":"Muse arc","agenda":["item one","item two"],"openThreads":[]},
      "recent":{"topics":["t1"],"lines":["Yesterday: 3 exchanges."]}}}
    """

  func testStateSnapshotParsesLiveShape() throws {
    let snapshot = try XCTUnwrap(KiraStateSnapshot.parse(Data(statePayload.utf8)))
    XCTAssertEqual(snapshot.mood, "glowing")
    XCTAssertEqual(snapshot.energy, "wired")
    XCTAssertEqual(snapshot.arcPhase, "make the thing")
    XCTAssertEqual(snapshot.scores.map(\.name), ["closeness", "warmth", "desire", "playfulness"])
    XCTAssertEqual(snapshot.scores[0].value, 83.3, accuracy: 0.01)
    XCTAssertEqual(snapshot.campaignBeat, "Muse arc")
    XCTAssertEqual(snapshot.agenda, ["item one", "item two"])
    XCTAssertEqual(snapshot.recentLines.count, 1)
    XCTAssertFalse(snapshot.worldPresent, "world slice absent until A3 lands")
    // Tolerance: an empty state object parses with everything absent.
    let minimal = try XCTUnwrap(KiraStateSnapshot.parse(Data(#"{"state":{}}"#.utf8)))
    XCTAssertNil(minimal.mood)
    XCTAssertTrue(minimal.scores.isEmpty)
  }

  func testSchedulerStatusParses() throws {
    let payload = #"{"ok":true,"paused":false,"config":{"enabled":true,"intervalMinutes":30,"imageCount":2,"videoCount":1,"videoMode":"mixed"}}"#
    let status = try XCTUnwrap(KiraSchedulerStatus.parse(Data(payload.utf8)))
    XCTAssertFalse(status.paused)
    XCTAssertTrue(status.enabled)
    XCTAssertEqual(status.intervalMinutes, 30)
    XCTAssertEqual(status.imageCount, 2)
    XCTAssertEqual(status.videoCount, 1)
    XCTAssertEqual(status.videoMode, "mixed")
    XCTAssertNil(KiraSchedulerStatus.parse(Data(#"{"ok":true}"#.utf8)), "no paused field → nil")
  }

  func testSuggestionParses() throws {
    let item: [String: Any] = [
      "id": "989dd184171c", "kind": "image",
      "text": "Kira on the balcony at golden hour",
      "status": "pending", "createdAt": "2026-07-17T18:19:02.687Z",
    ]
    let suggestion = try XCTUnwrap(KiraSuggestion.parse(item))
    XCTAssertEqual(suggestion.id, "989dd184171c")
    XCTAssertEqual(suggestion.kind, "image")
    XCTAssertEqual(suggestion.status, "pending")
    // Missing required field → nil, not a crash.
    XCTAssertNil(KiraSuggestion.parse(["id": "x", "kind": "image"]))
    // Status defaults to pending when absent.
    let noStatus = try XCTUnwrap(KiraSuggestion.parse(["id": "y", "kind": "arc", "text": "beach week"]))
    XCTAssertEqual(noStatus.status, "pending")
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
