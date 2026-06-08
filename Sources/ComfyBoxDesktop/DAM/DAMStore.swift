// DAMStore.swift — Digital Asset Management SQLite store
//
// Actor-isolated SQLite database for tracking generated assets.
// Uses system SQLite3 framework (zero dependencies). WAL mode
// for concurrent reads. FTS5 for full-text prompt search.
//
// Database location: ~/.comfybox/dam.sqlite3

import Foundation
import SQLite3

public actor DAMStore {
    private var db: OpaquePointer?
    private let dbPath: String

    private init(db: OpaquePointer, dbPath: String) {
        self.db = db
        self.dbPath = dbPath
    }

    /// Create and initialize a DAMStore. Opens the database, enables WAL mode,
    /// and creates tables if they do not exist.
    public static func open(path: String? = nil) async throws -> DAMStore {
        let resolvedPath: String
        if let path = path {
            resolvedPath = path
        } else {
            let comfyboxDir = NSString(string: "~/.comfybox").expandingTildeInPath
            try FileManager.default.createDirectory(
                atPath: comfyboxDir,
                withIntermediateDirectories: true
            )
            resolvedPath = (comfyboxDir as NSString).appendingPathComponent("dam.sqlite3")
        }

        var dbHandle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(resolvedPath, &dbHandle, flags, nil)
        guard rc == SQLITE_OK, let handle = dbHandle else {
            let msg = dbHandle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown error"
            if let h = dbHandle { sqlite3_close(h) }
            throw DAMStoreError.openFailed(resolvedPath, msg)
        }

        let store = DAMStore(db: handle, dbPath: resolvedPath)
        try await store.initialize()
        return store
    }

    /// Perform post-init setup within actor isolation.
    private func initialize() throws {
        try execute("PRAGMA journal_mode=WAL")
        try createTables()
    }

    deinit {
        if let db = db {
            sqlite3_close(db)
        }
    }

    // MARK: - Public API

    /// Insert a new asset into the database.
    public func insertAsset(_ asset: DAMAsset) throws {
        let sql = """
            INSERT OR REPLACE INTO assets (
                id, kind, filename, absolute_path, file_size, sha256,
                width, height, created_at, modified_at, ingested_at, orphaned,
                prompt, negative_prompt, seed, steps, guidance,
                model_family, rating, favorite, content_mode, character_name
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6,
                ?7, ?8, ?9, ?10, ?11, ?12,
                ?13, ?14, ?15, ?16, ?17,
                ?18, ?19, ?20, ?21, ?22
            )
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (asset.id as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 2, (asset.kind as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 3, (asset.filename as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 4, (asset.absolutePath as NSString).utf8String, -1, nil)
        sqlite3_bind_int64(stmt, 5, sqlite3_int64(asset.fileSize))
        bindOptionalText(stmt, 6, asset.sha256)
        bindOptionalInt(stmt, 7, asset.width)
        bindOptionalInt(stmt, 8, asset.height)
        sqlite3_bind_double(stmt, 9, asset.createdAt.timeIntervalSince1970)
        sqlite3_bind_double(stmt, 10, asset.modifiedAt.timeIntervalSince1970)
        sqlite3_bind_double(stmt, 11, asset.ingestedAt.timeIntervalSince1970)
        sqlite3_bind_int(stmt, 12, asset.orphaned ? 1 : 0)
        bindOptionalText(stmt, 13, asset.prompt)
        bindOptionalText(stmt, 14, asset.negativePrompt)
        bindOptionalInt(stmt, 15, asset.seed)
        bindOptionalInt(stmt, 16, asset.steps)
        bindOptionalDouble(stmt, 17, asset.guidance)
        bindOptionalText(stmt, 18, asset.modelFamily)
        sqlite3_bind_int(stmt, 19, Int32(asset.rating))
        sqlite3_bind_int(stmt, 20, asset.favorite ? 1 : 0)
        bindOptionalText(stmt, 21, asset.contentMode)
        bindOptionalText(stmt, 22, asset.characterName)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw DAMStoreError.insertFailed(lastError)
        }

        // Update FTS index.
        if let prompt = asset.prompt {
            try insertFTS(id: asset.id, prompt: prompt, negativePrompt: asset.negativePrompt)
        }
    }

    /// Fetch assets ordered by creation date (newest first).
    public func fetchAssets(limit: Int = 50, offset: Int = 0) throws -> [DAMAsset] {
        let sql = """
            SELECT id, kind, filename, absolute_path, file_size, sha256,
                   width, height, created_at, modified_at, ingested_at, orphaned,
                   prompt, negative_prompt, seed, steps, guidance,
                   model_family, rating, favorite, content_mode, character_name
            FROM assets
            ORDER BY created_at DESC
            LIMIT ?1 OFFSET ?2
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_int(stmt, 1, Int32(limit))
        sqlite3_bind_int(stmt, 2, Int32(offset))

        var results: [DAMAsset] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append(assetFromRow(stmt))
        }
        return results
    }

    /// Return all known absolute_path values from the database.
    /// Used by AssetIngestor to avoid re-ingesting existing assets.
    public func allAssetPaths() throws -> [String] {
        let sql = "SELECT absolute_path FROM assets"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        var paths: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let path = columnText(stmt, 0) {
                paths.append(path)
            }
        }
        return paths
    }

    /// Total number of assets in the database.
    public func assetCount() throws -> Int {
        let sql = "SELECT COUNT(*) FROM assets"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        guard sqlite3_step(stmt) == SQLITE_ROW else {
            return 0
        }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    /// Full-text search across prompts. Returns matching assets.
    public func searchPrompts(query: String, limit: Int = 50) throws -> [DAMAsset] {
        let sql = """
            SELECT a.id, a.kind, a.filename, a.absolute_path, a.file_size, a.sha256,
                   a.width, a.height, a.created_at, a.modified_at, a.ingested_at, a.orphaned,
                   a.prompt, a.negative_prompt, a.seed, a.steps, a.guidance,
                   a.model_family, a.rating, a.favorite, a.content_mode, a.character_name
            FROM assets_fts fts
            JOIN assets a ON a.id = fts.id
            WHERE assets_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (query as NSString).utf8String, -1, nil)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var results: [DAMAsset] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append(assetFromRow(stmt))
        }
        return results
    }

    // MARK: - Private

    private var lastError: String {
        db.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
    }

    private func execute(_ sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if rc != SQLITE_OK {
            let msg = errMsg.flatMap { String(cString: $0) } ?? "unknown"
            sqlite3_free(errMsg)
            throw DAMStoreError.execFailed(sql, msg)
        }
    }

    private func createTables() throws {
        try execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'image',
                filename TEXT NOT NULL,
                absolute_path TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL DEFAULT 0,
                sha256 TEXT,
                width INTEGER,
                height INTEGER,
                created_at REAL NOT NULL,
                modified_at REAL NOT NULL,
                ingested_at REAL NOT NULL,
                orphaned INTEGER NOT NULL DEFAULT 0,
                prompt TEXT,
                negative_prompt TEXT,
                seed INTEGER,
                steps INTEGER,
                guidance REAL,
                model_family TEXT,
                rating INTEGER NOT NULL DEFAULT 0,
                favorite INTEGER NOT NULL DEFAULT 0,
                content_mode TEXT,
                character_name TEXT
            )
            """)

        try execute("CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at DESC)")

        // FTS5 virtual table for full-text prompt search.
        try execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
                id UNINDEXED,
                prompt,
                negative_prompt
            )
            """)
    }

    private func insertFTS(id: String, prompt: String, negativePrompt: String?) throws {
        let sql = "INSERT OR REPLACE INTO assets_fts (id, prompt, negative_prompt) VALUES (?1, ?2, ?3)"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            // FTS insert failure is non-fatal.
            return
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 2, (prompt as NSString).utf8String, -1, nil)
        bindOptionalText(stmt, 3, negativePrompt)
        sqlite3_step(stmt)
    }

    private func assetFromRow(_ stmt: OpaquePointer?) -> DAMAsset {
        DAMAsset(
            id: columnText(stmt, 0) ?? "",
            kind: columnText(stmt, 1) ?? "image",
            filename: columnText(stmt, 2) ?? "",
            absolutePath: columnText(stmt, 3) ?? "",
            fileSize: sqlite3_column_int64(stmt, 4),
            sha256: columnText(stmt, 5),
            width: columnOptionalInt(stmt, 6),
            height: columnOptionalInt(stmt, 7),
            createdAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 8)),
            modifiedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 9)),
            ingestedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 10)),
            orphaned: sqlite3_column_int(stmt, 11) != 0,
            prompt: columnText(stmt, 12),
            negativePrompt: columnText(stmt, 13),
            seed: columnOptionalInt(stmt, 14),
            steps: columnOptionalInt(stmt, 15),
            guidance: columnOptionalDouble(stmt, 16),
            modelFamily: columnText(stmt, 17),
            rating: Int(sqlite3_column_int(stmt, 18)),
            favorite: sqlite3_column_int(stmt, 19) != 0,
            contentMode: columnText(stmt, 20),
            characterName: columnText(stmt, 21)
        )
    }

    // MARK: - SQLite Helpers

    private func bindOptionalText(_ stmt: OpaquePointer?, _ index: Int32, _ value: String?) {
        if let value = value {
            sqlite3_bind_text(stmt, index, (value as NSString).utf8String, -1, nil)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func bindOptionalInt(_ stmt: OpaquePointer?, _ index: Int32, _ value: Int?) {
        if let value = value {
            sqlite3_bind_int64(stmt, index, sqlite3_int64(value))
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func bindOptionalDouble(_ stmt: OpaquePointer?, _ index: Int32, _ value: Double?) {
        if let value = value {
            sqlite3_bind_double(stmt, index, value)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func columnText(_ stmt: OpaquePointer?, _ index: Int32) -> String? {
        guard let cStr = sqlite3_column_text(stmt, index) else { return nil }
        return String(cString: cStr)
    }

    private func columnOptionalInt(_ stmt: OpaquePointer?, _ index: Int32) -> Int? {
        if sqlite3_column_type(stmt, index) == SQLITE_NULL { return nil }
        return Int(sqlite3_column_int64(stmt, index))
    }

    private func columnOptionalDouble(_ stmt: OpaquePointer?, _ index: Int32) -> Double? {
        if sqlite3_column_type(stmt, index) == SQLITE_NULL { return nil }
        return sqlite3_column_double(stmt, index)
    }
}

// MARK: - Errors

public enum DAMStoreError: Error, LocalizedError {
    case openFailed(String, String)
    case prepareFailed(String)
    case insertFailed(String)
    case execFailed(String, String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let path, let msg):
            return "Failed to open database at \(path): \(msg)"
        case .prepareFailed(let msg):
            return "SQL prepare failed: \(msg)"
        case .insertFailed(let msg):
            return "Insert failed: \(msg)"
        case .execFailed(let sql, let msg):
            return "SQL exec failed (\(sql)): \(msg)"
        }
    }
}
