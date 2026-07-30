// CatalogSchema.swift — the ONLY file that writes DDL.
//
// Migration is strictly additive. `DAMStore` in the desktop app reads `assets`
// with an explicit column list (DAMStore.swift:223, :519), so columns appended
// at the end are invisible to it and cannot break it. Never drop, rename or
// reorder an existing column; never touch folders / asset_folders /
// secured_assets, which are legacy and deliberately left alone.

import Foundation
import SQLite3

public enum CatalogSchemaError: Error, LocalizedError {
    case execFailed(String, String)

    public var errorDescription: String? {
        switch self {
        case let .execFailed(sql, msg): return "SQL failed: \(msg) — \(sql)"
        }
    }
}

public enum CatalogSchema {

    /// Columns appended to `assets`, in order. Name → SQLite type.
    public static let newColumns: [(String, String)] = [
        ("realm", "TEXT"),
        ("sealed", "INTEGER NOT NULL DEFAULT 0"),
        ("lane", "TEXT"),
        ("arc", "TEXT"),
        ("theme", "TEXT"),
        ("stock", "TEXT"),
        ("genre", "TEXT"),
        ("family", "TEXT"),
        ("style", "TEXT"),
        ("preset", "TEXT"),
        ("loras", "TEXT"),
        ("render_id", "TEXT"),
        ("caption", "TEXT"),
        ("caption_source", "TEXT"),
        ("prompt_raw", "TEXT"),
        ("mode", "TEXT"),
        ("duration_ms", "INTEGER"),
        ("fps", "REAL"),
        ("frames", "INTEGER"),
        ("resolution", "TEXT"),
        ("aspect_ratio", "TEXT"),
    ]

    /// The seeded bodies of work. Ids are stable literals so re-seeding is a
    /// no-op and so the rule table can reference them by id.
    public static let seedCollections: [CatalogCollection] = [
        // Shared — any producer contributes.
        CatalogCollection(id: "col-decoupage", slug: "decoupage", name: "Decoupage",
                          description: "Tile designs — desktop tool, tile engine, Krita, Kira."),
        CatalogCollection(id: "col-photography", slug: "photography", name: "Photography",
                          description: "Art photography from any producer."),
        CatalogCollection(id: "col-adult", slug: "adult", name: "Adult",
                          parentID: nil, description: "Adult entertainment."),
        // Kira's realm — hers alone.
        CatalogCollection(id: "col-kira-still-life", slug: "kira-still-life",
                          name: "Still Life", realm: .kira,
                          description: "Her still-life practice."),
        CatalogCollection(id: "col-kira-autocord", slug: "kira-autocord",
                          name: "Autocord Photography", parentID: "col-photography",
                          realm: .kira, description: "Her TLR practice, any film stock, any genre."),
        CatalogCollection(id: "col-kira-decoupage", slug: "kira-decoupage",
                          name: "Decoupage Designs", parentID: "col-decoupage",
                          realm: .kira, description: "Her tile designs."),
        CatalogCollection(id: "col-kira-nightlife", slug: "kira-nightlife",
                          name: "Nightlife", realm: .kira,
                          description: "Voyeuristic, fun, sometimes crazy."),
        CatalogCollection(id: "col-kira-erotic-portraiture", slug: "kira-erotic-portraiture",
                          name: "Erotic Portraiture", parentID: "col-adult",
                          realm: .kira, description: "Erotic film photography of her and her friends."),
        CatalogCollection(id: "col-kira-adult-scenes", slug: "kira-adult-scenes",
                          name: "Adult Scenes", parentID: "col-adult",
                          realm: .kira, description: "Stills and clips alike."),
        CatalogCollection(id: "col-kira-dreams-memories", slug: "kira-dreams-memories",
                          name: "Dreams & Memories", realm: .kira,
                          description: "The t2v work she makes to describe a dream or a memory."),
    ]

    public static func migrate(db: OpaquePointer?) throws {
        // Appended columns. A duplicate-column error means an earlier run already
        // added it, which is success — hence `try?`, matching the existing
        // `source` migration idiom at DAMStore.swift:623.
        for (name, type) in newColumns {
            _ = try? exec(db, "ALTER TABLE assets ADD COLUMN \(name) \(type)")
        }
        _ = try? exec(db, "UPDATE assets SET realm = 'shared' WHERE realm IS NULL")

        try exec(db, """
            CREATE TABLE IF NOT EXISTS asset_locations (
                asset_id TEXT NOT NULL,
                host TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime REAL NOT NULL,
                PRIMARY KEY (asset_id, host, path)
            )
            """)
        try exec(db, "CREATE INDEX IF NOT EXISTS idx_locations_path ON asset_locations(path)")

        try exec(db, """
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                parent_id TEXT,
                realm TEXT,
                description TEXT,
                created_at REAL NOT NULL
            )
            """)
        try exec(db, """
            CREATE TABLE IF NOT EXISTS asset_collections (
                asset_id TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                manual INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (asset_id, collection_id)
            )
            """)
        try exec(db, "CREATE INDEX IF NOT EXISTS idx_asset_collections_col ON asset_collections(collection_id)")

        try exec(db, """
            CREATE TABLE IF NOT EXISTS asset_edges (
                from_asset_id TEXT NOT NULL,
                to_asset_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                PRIMARY KEY (from_asset_id, to_asset_id, relation)
            )
            """)
        try exec(db, "CREATE INDEX IF NOT EXISTS idx_edges_to ON asset_edges(to_asset_id)")

        try exec(db, "CREATE INDEX IF NOT EXISTS idx_assets_realm ON assets(realm, created_at DESC)")

        try migrateFTS(db: db)
        try seed(db: db)
    }

    /// FTS5 has no ALTER. Rebuild the virtual table with the caption column and
    /// repopulate from `assets`, which is the source of truth for the text.
    private static func migrateFTS(db: OpaquePointer?) throws {
        if ftsHasCaption(db: db) { return }
        try exec(db, "DROP TABLE IF EXISTS assets_fts")
        try exec(db, """
            CREATE VIRTUAL TABLE assets_fts USING fts5(
                id UNINDEXED, prompt, negative_prompt, caption
            )
            """)
        try exec(db, """
            INSERT INTO assets_fts (id, prompt, negative_prompt, caption)
            SELECT id, COALESCE(prompt, ''), COALESCE(negative_prompt, ''), COALESCE(caption, '')
            FROM assets WHERE sealed = 0
            """)
    }

    private static func ftsHasCaption(db: OpaquePointer?) -> Bool {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "PRAGMA table_info(assets_fts)", -1, &stmt, nil) == SQLITE_OK else {
            return false
        }
        defer { sqlite3_finalize(stmt) }
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 1), String(cString: c) == "caption" { return true }
        }
        return false
    }

    private static func seed(db: OpaquePointer?) throws {
        let now = Date().timeIntervalSince1970
        for c in seedCollections {
            let sql = """
                INSERT OR IGNORE INTO collections (id, slug, name, parent_id, realm, description, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw CatalogSchemaError.execFailed(sql, String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }
            bindText(stmt, 1, c.id)
            bindText(stmt, 2, c.slug)
            bindText(stmt, 3, c.name)
            bindText(stmt, 4, c.parentID)
            bindText(stmt, 5, c.realm?.rawValue)
            bindText(stmt, 6, c.description)
            sqlite3_bind_double(stmt, 7, now)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw CatalogSchemaError.execFailed(sql, String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    @discardableResult
    static func exec(_ db: OpaquePointer?, _ sql: String) throws -> Bool {
        var errMsg: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if rc != SQLITE_OK {
            let msg = errMsg.flatMap { String(cString: $0) } ?? "unknown"
            sqlite3_free(errMsg)
            throw CatalogSchemaError.execFailed(sql, msg)
        }
        return true
    }

    static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    static func bindText(_ stmt: OpaquePointer?, _ idx: Int32, _ value: String?) {
        if let v = value {
            sqlite3_bind_text(stmt, idx, (v as NSString).utf8String, -1, transient)
        } else {
            sqlite3_bind_null(stmt, idx)
        }
    }
}
