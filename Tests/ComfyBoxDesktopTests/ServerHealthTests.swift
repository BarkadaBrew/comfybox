import XCTest
@testable import ComfyBoxDesktop

final class ServerHealthTests: XCTestCase {

    private let sample = """
    {
      "summary": "1 container down",
      "system": {"disk": "85%", "memory": "60%", "uptime": "53d"},
      "containers": [
        {"name": "deepwiki", "state": "exited", "restart_policy": "no", "suppressed": true},
        {"name": "grafana", "state": "running", "restart_policy": "unless-stopped"}
      ],
      "mac_pipeline": [{"name": "ComfyBox", "status": "up"}],
      "suppressed_alerts": [
        {"alert": "deepwiki down", "reason": "muted", "since": "2026-05-14"}
      ],
      "checked_at": "2026-07-06T14:00:00Z"
    }
    """

    func testDecodesBareObject() {
        let h = ServerHealthService.decodeHealth(from: Data(sample.utf8))
        XCTAssertNotNil(h)
        XCTAssertEqual(h?.system?.disk, "85%")
        XCTAssertEqual(h?.containers?.count, 2)
    }

    func testSuppressedAlwaysPresentAndParsed() {
        let h = ServerHealthService.decodeHealth(from: Data(sample.utf8))!
        XCTAssertEqual(h.suppressed.count, 1)
        XCTAssertEqual(h.suppressed.first?.alert, "deepwiki down")
        XCTAssertEqual(h.suppressed.first?.since, "2026-05-14")
    }

    func testSuppressedEmptyWhenAbsent() {
        let h = ServerHealthService.decodeHealth(from: Data("{\"summary\":\"ok\"}".utf8))
        XCTAssertEqual(h?.suppressed.count, 0, "empty is a valid, always-present state")
    }

    func testSevenWeekTrapDetected() {
        let h = ServerHealthService.decodeHealth(from: Data(sample.utf8))!
        let deepwiki = h.containers!.first { $0.name == "deepwiki" }!
        XCTAssertFalse(deepwiki.isRunning)
        XCTAssertTrue(deepwiki.isSuppressed)
        XCTAssertTrue(deepwiki.lacksRestartPolicy, "restart_policy 'no' is the silent-death trap")
        XCTAssertEqual(h.problemContainers.count, 1)
    }

    func testMCPContentEnvelope() {
        let escaped = sample.replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "")
        let envelope = "{\"content\":[{\"type\":\"text\",\"text\":\"\(escaped)\"}]}"
        let h = ServerHealthService.decodeHealth(from: Data(envelope.utf8))
        XCTAssertNotNil(h, "should dig health out of an MCP content envelope")
        XCTAssertEqual(h?.suppressed.count, 1)
    }

    func testMacPipelineStatusField() {
        let h = ServerHealthService.decodeHealth(from: Data(sample.utf8))!
        XCTAssertEqual(h.macPipeline?.first?.name, "ComfyBox")
        XCTAssertTrue(h.macPipeline?.first?.isHealthy == true)
    }
}
