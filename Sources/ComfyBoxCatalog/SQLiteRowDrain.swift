// SQLiteRowDrain.swift — the shared `sqlite3_step` row-draining primitive.
//
// Moved here from ComfyBoxDesktop/DAM/DAMStore.swift (#356) so both the DAM
// store and the catalog store (CatalogStore.swift, CatalogSchema.swift) share
// ONE implementation instead of two copies drifting apart. ComfyBoxDesktop
// already depends on ComfyBoxCatalog (DAMStore.swift itself does
// `import ComfyBoxCatalog`), so this is the shared home, not the other way
// around. See #357.

import SQLite3

/// Drives a `sqlite3_step` loop to completion, invoking `onRow` for every
/// `SQLITE_ROW` and returning the terminal step code once rows are
/// exhausted. A bare `while sqlite3_step(stmt) == SQLITE_ROW` cannot tell
/// "the result set is finished" (`SQLITE_DONE`) from "the step itself
/// failed partway through" (`SQLITE_BUSY`, `SQLITE_IOERR`,
/// `SQLITE_CORRUPT`, …) — both look identical from inside the loop, so a
/// mid-iteration error silently truncates the result instead of surfacing
/// (#263, #357). Callers must compare the returned code against
/// `SQLITE_DONE` and throw on anything else — or, for a read-only listing
/// where a partial result is an acceptable degradation, log the non-DONE
/// code rather than silently accepting it as "no more rows".
///
/// `step` is a closure rather than a bound `OpaquePointer` so this can be
/// unit-tested with a stub that fails after N rows, without standing up a
/// real (and hard to fault-inject on purpose) SQLite connection.
///
/// Out of scope here, deliberately: none of this function's callers check
/// `sqlite3_finalize`'s own return code. `sqlite3_finalize` can surface an
/// error that occurred during the statement's *last* `sqlite3_step` call —
/// which, for every loop this function now guards, is already caught by the
/// `rc != SQLITE_DONE` check callers perform before `finalize` ever runs;
/// the residual gap is prepared-but-never-stepped and single-shot
/// (INSERT/UPDATE/DELETE) statements, where a step failure is already
/// checked separately and finalize is unlikely to add new information.
public func drainSQLiteRows(step: () -> Int32, onRow: () -> Void) -> Int32 {
    var rc = step()
    while rc == SQLITE_ROW {
        onRow()
        rc = step()
    }
    return rc
}
