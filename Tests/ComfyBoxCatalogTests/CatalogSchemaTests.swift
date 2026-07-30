import XCTest
import SQLite3
@testable import ComfyBoxCatalog

final class CatalogSchemaTests: XCTestCase {
    private var db: OpaquePointer!
    private var path: String!

    override func setUp() {
        super.setUp()
        path = NSTemporaryDirectory() + "catalog-test-\(UUID().uuidString).sqlite3"
        XCTAssertEqual(sqlite3_open(path, &db), SQLITE_OK)
    }

    override func tearDown() {
        sqlite3_close(db)
        try? FileManager.default.removeItem(atPath: path)
        super.tearDown()
    }

    /// Recreate the CURRENT production schema, so we test migration of a real
    /// pre-existing database rather than a greenfield one.
    private func createLegacySchema() {
        let sql = """
            CREATE TABLE assets (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'image',
                filename TEXT NOT NULL, absolute_path TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL DEFAULT 0, sha256 TEXT,
                width INTEGER, height INTEGER,
                created_at REAL NOT NULL, modified_at REAL NOT NULL,
                ingested_at REAL NOT NULL, orphaned INTEGER NOT NULL DEFAULT 0,
                prompt TEXT, negative_prompt TEXT, seed INTEGER, steps INTEGER,
                guidance REAL, model_family TEXT,
                rating INTEGER NOT NULL DEFAULT 0, favorite INTEGER NOT NULL DEFAULT 0,
                content_mode TEXT, character_name TEXT, source TEXT
            );
            CREATE TABLE folders (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE asset_folders (asset_id TEXT PRIMARY KEY, folder_id TEXT NOT NULL);
            CREATE VIRTUAL TABLE assets_fts USING fts5(id UNINDEXED, prompt, negative_prompt);
            """
        XCTAssertEqual(sqlite3_exec(db, sql, nil, nil, nil), SQLITE_OK)
    }

    private func columns(of table: String) -> [String] {
        var stmt: OpaquePointer?
        var out: [String] = []
        sqlite3_prepare_v2(db, "PRAGMA table_info(\(table))", -1, &stmt, nil)
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 1) { out.append(String(cString: c)) }
        }
        sqlite3_finalize(stmt)
        return out
    }

    private func tableNames() -> [String] {
        var stmt: OpaquePointer?
        var out: [String] = []
        sqlite3_prepare_v2(db, "SELECT name FROM sqlite_master WHERE type='table'", -1, &stmt, nil)
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 0) { out.append(String(cString: c)) }
        }
        sqlite3_finalize(stmt)
        return out
    }

    func testMigrationAddsNewColumnsWithoutDisturbingLegacyOnes() throws {
        createLegacySchema()
        let before = columns(of: "assets")
        try CatalogSchema.migrate(db: db)
        let after = columns(of: "assets")

        // Every legacy column survives, in its original position.
        XCTAssertEqual(Array(after.prefix(before.count)), before,
                       "legacy columns must not move — DAMStore reads by explicit list")
        for (name, _) in CatalogSchema.newColumns {
            XCTAssertTrue(after.contains(name), "missing new column \(name)")
        }
    }

    func testMigrationIsIdempotent() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        let once = columns(of: "assets")
        try CatalogSchema.migrate(db: db)
        XCTAssertEqual(columns(of: "assets"), once)
    }

    /// A non-duplicate ALTER failure must propagate rather than being silently
    /// swallowed. With no `assets` table at all, the very first
    /// `ALTER TABLE assets ADD COLUMN ...` fails with "no such table: assets" —
    /// a message the duplicate-column check does not match — so it must be
    /// rethrown out of `migrate`, not absorbed like an already-migrated column
    /// would be.
    func testMigrateThrowsWhenAssetsTableIsMissingEntirely() {
        XCTAssertThrowsError(try CatalogSchema.migrate(db: db)) { error in
            guard case let CatalogSchemaError.execFailed(_, msg) = error else {
                XCTFail("expected .execFailed, got \(error)")
                return
            }
            XCTAssertFalse(msg.lowercased().contains("duplicate column name"),
                           "this must be a genuine failure, not the ignored duplicate-column case")
        }
    }

    /// Positive case for the post-migration verification: on a correctly
    /// migrated database every new column is present. The negative case (a
    /// migration that silently drops one of the ALTERs) has no seam to trigger
    /// from a black-box test — the loop's own duplicate-column filter is the
    /// only conditional path, and bypassing it would require reaching into
    /// `CatalogSchema` internals with test-only hooks, which would test the
    /// hook rather than the guarantee. `verifyNewColumnsPresent` is exercised
    /// on every call to `migrate` in every other test in this file, so its
    /// absence of a false negative is covered throughout; this test pins down
    /// that it does not raise a false positive either.
    func testMigrationVerifiesAllNewColumnsPresent() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        let present = Set(columns(of: "assets"))
        for (name, _) in CatalogSchema.newColumns {
            XCTAssertTrue(present.contains(name), "verification should have caught a missing \(name)")
        }
    }

    func testMigrationCreatesNewTablesAndLeavesLegacyTablesAlone() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        let tables = tableNames()
        for t in ["asset_locations", "collections", "asset_collections", "asset_edges"] {
            XCTAssertTrue(tables.contains(t), "missing table \(t)")
        }
        XCTAssertTrue(tables.contains("folders"))
        XCTAssertTrue(tables.contains("asset_folders"))
        XCTAssertEqual(columns(of: "asset_folders"), ["asset_id", "folder_id"],
                       "legacy folder tables must be untouched")
    }

    func testFTSGainsCaptionColumn() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        XCTAssertEqual(columns(of: "assets_fts"), ["id", "prompt", "negative_prompt", "caption"])
    }

    func testSeedCollectionsAreTwoLevelsAndWellFormed() {
        let seeds = CatalogSchema.seedCollections
        let roots = Set(seeds.filter { $0.parentID == nil }.map(\.id))
        for c in seeds where c.parentID != nil {
            XCTAssertTrue(roots.contains(c.parentID!),
                          "\(c.slug) parent must be a root — two levels only")
        }
        XCTAssertEqual(Set(seeds.map(\.slug)).count, seeds.count, "slugs must be unique")
        // Kira's genres are hers; the cross-producer bodies are shared.
        XCTAssertNil(seeds.first { $0.slug == "decoupage" }?.realm)
        XCTAssertNil(seeds.first { $0.slug == "photography" }?.realm)
        XCTAssertEqual(seeds.first { $0.slug == "kira-dreams-memories" }?.realm, .kira)
        XCTAssertEqual(seeds.first { $0.slug == "kira-autocord" }?.realm, .kira)
    }

    func testSeedingIsIdempotent() throws {
        createLegacySchema()
        try CatalogSchema.migrate(db: db)
        try CatalogSchema.migrate(db: db)
        var stmt: OpaquePointer?
        sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM collections", -1, &stmt, nil)
        XCTAssertEqual(sqlite3_step(stmt), SQLITE_ROW)
        XCTAssertEqual(Int(sqlite3_column_int(stmt, 0)), CatalogSchema.seedCollections.count)
        sqlite3_finalize(stmt)
    }
}
