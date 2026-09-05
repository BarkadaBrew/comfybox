// SQLiteRowDrainTests.swift — pins `drainSQLiteRows`, now the shared home for
// the primitive both CatalogStore/CatalogSchema (this module) and DAMStore
// (ComfyBoxDesktop) build their row loops on (#357, following #263/#356).
//
// #263: `while sqlite3_step(stmt) == SQLITE_ROW` cannot distinguish "the
// result set is finished" (`SQLITE_DONE`) from "the step itself failed
// partway through" (`SQLITE_BUSY`, `SQLITE_IOERR`, `SQLITE_CORRUPT`, …) —
// both fall out of the loop identically, silently truncating the result.
// Fault-injecting a real mid-iteration SQLite error into a live connection is
// impractical to do deterministically, so this pins the primitive directly
// with a stubbed `step` closure — the same approach `DrainSQLiteRowsTests`
// used before the function moved here.

import XCTest
import SQLite3
@testable import ComfyBoxCatalog

final class SQLiteRowDrainTests: XCTestCase {
    func testCleanFinishWalksEveryRowAndReturnsDone() {
        var remaining = [SQLITE_ROW, SQLITE_ROW, SQLITE_ROW, SQLITE_DONE]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        XCTAssertEqual(rc, SQLITE_DONE)
        XCTAssertEqual(seen, 3)
    }

    func testMidIterationErrorStopsAndReportsTheFailingCodeNotDone() {
        // Two good rows, then the step call itself fails — the historical bug
        // treated this identically to "no more rows".
        var remaining = [SQLITE_ROW, SQLITE_ROW, SQLITE_IOERR]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        XCTAssertEqual(rc, SQLITE_IOERR)
        XCTAssertNotEqual(rc, SQLITE_DONE)
        XCTAssertEqual(seen, 2, "onRow must not fire for the failing step")
    }

    func testBusyIsNotTreatedAsDoneEither() {
        var remaining = [SQLITE_ROW, SQLITE_BUSY]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        XCTAssertEqual(rc, SQLITE_BUSY)
        XCTAssertEqual(seen, 1)
    }

    func testEmptyResultSetStillReportsDoneNotAnError() {
        var remaining = [SQLITE_DONE]
        var seen = 0
        let rc = drainSQLiteRows(step: { remaining.removeFirst() }) { seen += 1 }
        XCTAssertEqual(rc, SQLITE_DONE)
        XCTAssertEqual(seen, 0)
    }
}
