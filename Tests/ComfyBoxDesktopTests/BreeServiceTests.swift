import XCTest
@testable import ComfyBoxDesktop

final class BreeServiceTests: XCTestCase {

    func testFormatEntryHasTimestampHeaderAndBody() {
        // 2026-07-06 14:05 local.
        var comps = DateComponents()
        comps.year = 2026; comps.month = 7; comps.day = 6; comps.hour = 14; comps.minute = 5
        let date = Calendar.current.date(from: comps)!
        let entry = BreeService.formatEntry("  Please render the batch.  ", timestamp: date)
        XCTAssertTrue(entry.contains("## 2026-07-06 14:05 — Desktop"), "header with stamp+author")
        XCTAssertTrue(entry.contains("Please render the batch."), "trimmed body present")
        XCTAssertFalse(entry.contains("  Please"), "leading whitespace trimmed")
        XCTAssertTrue(entry.hasPrefix("\n"), "entry is appended with a leading newline separator")
    }

    func testFormatEntryCustomAuthor() {
        let entry = BreeService.formatEntry("hi", timestamp: Date(), author: "ComfyBox")
        XCTAssertTrue(entry.contains("— ComfyBox"))
    }

    func testDefaultDirectoryExpandsTilde() {
        XCTAssertFalse(BreeService.defaultDirectory.hasPrefix("~"))
        XCTAssertTrue(BreeService.defaultDirectory.contains("Handoff"))
    }

    @MainActor
    func testSendAppendsWithoutDestroyingHistory() throws {
        let tmp = NSTemporaryDirectory() + "bree-test-\(UUID().uuidString)"
        let bree = BreeService(handoffDirectory: tmp)
        bree.send("first")
        bree.send("second")
        let written = try String(contentsOfFile: bree.outboxPath, encoding: .utf8)
        XCTAssertTrue(written.contains("first"))
        XCTAssertTrue(written.contains("second"))
        // Order preserved (append-only).
        XCTAssertLessThan(written.range(of: "first")!.lowerBound, written.range(of: "second")!.lowerBound)
        try? FileManager.default.removeItem(atPath: tmp)
    }
}
