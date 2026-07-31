// DAMStore.swift — Digital Asset Management SQLite store
//
// Actor-isolated SQLite database for tracking generated assets.
// Uses system SQLite3 framework (zero dependencies). WAL mode
// for concurrent reads. FTS5 for full-text prompt search.
//
// Database location: ~/.comfybox/dam.sqlite3

import ComfyBoxCatalog
import Foundation
import SQLite3

/// SQLite destructor telling SQLite to copy bound text immediately.
/// Passing nil (SQLITE_STATIC) with autoreleased `NSString.utf8String`
/// pointers risks the buffer being released before the statement runs.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

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
        // This file has more than one writer: the catalog tooling holds a single
        // long write transaction for the whole of `refileAll`, and without a
        // timeout a render landing during that window takes an immediate
        // SQLITE_BUSY and loses its DAM row outright. Matches the catalog side.
        try execute("PRAGMA busy_timeout = 5000")
        try createTables()
    }

    deinit {
        if let db = db {
            sqlite3_close(db)
        }
    }

    // MARK: - Public API

    /// Insert a new asset, or update an existing one. If the asset's path is
    /// already tracked under a different id (re-ingest), the existing id and
    /// user annotations (rating, favorite) are preserved and generation
    /// metadata is only overwritten when the new record provides it.
    /// Returns the asset as stored.
    @discardableResult
    public func insertAsset(_ asset: DAMAsset) throws -> DAMAsset {
        let asset = try mergedWithExisting(asset)
        // A TRUE UPSERT, never `INSERT OR REPLACE`.
        //
        // `OR REPLACE` is DELETE-then-INSERT, so every column this statement
        // does not name reverts to its default. That was survivable while this
        // table had 23 columns and the DAM owned all of them; it now has 45,
        // because the catalog migrated into this same file. Under `OR REPLACE`
        // a single ♥ click — toggleFavorite reads the row, flips one flag and
        // writes it back through here — reset 22 catalog columns on that row:
        // `realm` to NULL (the next open promotes NULL to 'shared', quietly
        // moving a Kira-private asset out of her realm) and `sealed` to 0 (the
        // row unseals, and the FTS write below then indexes its prompt,
        // permanently breaking "a sealed row is unreachable by its own
        // prompt"). Backfill cannot restore either.
        //
        // `mergedWithExisting` does NOT protect against this: it returns early
        // when the path's owner is this very id, which is exactly the update
        // path.
        //
        // Two conflict targets because the table has two unique indexes.
        // `id` is the ordinary case. `absolute_path` is a concurrency backstop:
        // `mergedWithExisting` normally rewrites an id collision on a tracked
        // path into an id conflict, but another writer can claim the path in
        // between — and losing that race must update the owning row, not delete
        // it.
        let ownedColumns = [
            "kind", "filename", "file_size", "sha256", "width", "height",
            "created_at", "modified_at", "ingested_at", "orphaned",
            "prompt", "negative_prompt", "seed", "steps", "guidance",
            "model_family", "rating", "favorite", "content_mode",
            "character_name", "source",
        ]
        let setClause = ownedColumns.map { "\($0) = excluded.\($0)" }.joined(separator: ", ")
        let sql = """
            INSERT INTO assets (
                id, kind, filename, absolute_path, file_size, sha256,
                width, height, created_at, modified_at, ingested_at, orphaned,
                prompt, negative_prompt, seed, steps, guidance,
                model_family, rating, favorite, content_mode, character_name, source
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6,
                ?7, ?8, ?9, ?10, ?11, ?12,
                ?13, ?14, ?15, ?16, ?17,
                ?18, ?19, ?20, ?21, ?22, ?23
            )
            ON CONFLICT(id) DO UPDATE SET \(setClause), absolute_path = excluded.absolute_path
            ON CONFLICT(absolute_path) DO UPDATE SET \(setClause)
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (asset.id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, (asset.kind as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 3, (asset.filename as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 4, (asset.absolutePath as NSString).utf8String, -1, SQLITE_TRANSIENT)
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
        bindOptionalText(stmt, 23, asset.source)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw DAMStoreError.insertFailed(lastError)
        }

        // Which id the path actually ended up under. Normally `asset.id`; in the
        // `ON CONFLICT(absolute_path)` backstop the pre-existing row keeps its
        // own id, and FTS + thumbnails are keyed by id, so indexing under the id
        // we *tried* to write would leave an entry pointing at no row.
        let stored = try assetWithStoredIdentity(asset)

        // Update FTS index. Delete first — FTS5 has no unique constraint on
        // id, so a bare INSERT would accumulate duplicate rows on updates.
        try deleteFTS(id: stored.id)
        // A SEALED row is never indexed, whatever its prompt column happens to
        // hold. "A sealed row is unreachable by its own prompt" rested entirely
        // on that column being NULL — which is true of a row the catalog wrote
        // and NOT true of one whose text arrived by another path (a sidecar
        // re-ingest, a hand-repaired row). The DAM has no `sealed` concept of
        // its own, so it asks the file; a DAM-only database has no such column
        // and the answer is simply "not sealed".
        if let prompt = stored.prompt, !isSealed(id: stored.id) {
            try insertFTS(id: stored.id, prompt: prompt, negativePrompt: stored.negativePrompt)
        }

        return stored
    }

    /// `asset` re-labelled with the id the row at its path actually carries.
    private func assetWithStoredIdentity(_ asset: DAMAsset) throws -> DAMAsset {
        let sql = "SELECT id FROM assets WHERE absolute_path = ?1"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, (asset.absolutePath as NSString).utf8String, -1, SQLITE_TRANSIENT)
        guard sqlite3_step(stmt) == SQLITE_ROW,
              let id = columnText(stmt, 0), id != asset.id else { return asset }
        return DAMAsset(
            id: id, kind: asset.kind, filename: asset.filename,
            absolutePath: asset.absolutePath, fileSize: asset.fileSize,
            sha256: asset.sha256, width: asset.width, height: asset.height,
            createdAt: asset.createdAt, modifiedAt: asset.modifiedAt,
            ingestedAt: asset.ingestedAt, orphaned: asset.orphaned,
            prompt: asset.prompt, negativePrompt: asset.negativePrompt,
            seed: asset.seed, steps: asset.steps, guidance: asset.guidance,
            modelFamily: asset.modelFamily, rating: asset.rating, favorite: asset.favorite,
            contentMode: asset.contentMode, characterName: asset.characterName,
            source: asset.source)
    }

    /// Delete rows whose backing file no longer exists (deleted out from
    /// under the DAM), cleaning FTS + folder mappings + returning their ids
    /// so the caller can drop cached thumbnails. Secured assets are skipped —
    /// their path points into the vault and is expected to be non-obvious.
    ///
    /// So are assets that live on ANOTHER HOST. This table is now shared with
    /// the catalog, which indexes Kira's and Bree's server trees as well as this
    /// Mac's; roughly 1,300 of the 2,994 rows in the live database name a path
    /// that exists only on a server. "The file isn't here" is not evidence that
    /// such a row is an orphan — it is the normal case — and pruning them would
    /// silently delete half the catalog on the next gallery load.
    /// Above this fraction of the library, a sweep refuses to delete anything.
    /// A mass disappearance is far more likely to be an unmounted volume, a
    /// revoked Pictures permission or iCloud eviction than that many genuine
    /// deletions — and this runs unattended on every unfiltered gallery load,
    /// so "probably a mount problem" must not be spelled DELETE.
    public static let pruneCircuitBreakerFraction = 0.05
    /// Below this many rows the fraction is meaningless (1 of 3 is 33%), so a
    /// sweep may always remove at least this many.
    public static let pruneCircuitBreakerFloor = 5

    @discardableResult
    public func pruneOrphans() throws -> [String] {
        let secured = try securedAssetIds()
        let elsewhere = try assetIDsHostedElsewhere()
        let all = try fetchAssets(limit: 100_000)

        // Decide the whole sweep before performing any of it.
        var candidates: [String] = []
        for asset in all {
            guard !secured.contains(asset.id), !elsewhere.contains(asset.id) else { continue }
            if !FileManager.default.fileExists(atPath: asset.absolutePath) {
                candidates.append(asset.id)
            }
        }

        let ceiling = max(Self.pruneCircuitBreakerFloor,
                          Int(Double(all.count) * Self.pruneCircuitBreakerFraction))
        guard candidates.count <= ceiling else {
            throw DAMStoreError.pruneRefused(candidates: candidates.count, total: all.count)
        }

        for id in candidates {
            try deleteAsset(id: id)  // FTS + folders + collections + locations
        }
        return candidates
    }

    /// Ids the catalog records a copy of on a host other than this Mac.
    ///
    /// `asset_locations` is the catalog's table, created by its migration in the
    /// same file. A DAM-only database (a test fixture, or an install that has
    /// never run the backfill) does not have it, so its absence means "nothing
    /// is hosted elsewhere" rather than an error.
    private func assetIDsHostedElsewhere() throws -> Set<String> {
        let exists = "SELECT 1 FROM sqlite_master WHERE type='table' AND name='asset_locations'"
        var check: OpaquePointer?
        guard sqlite3_prepare_v2(db, exists, -1, &check, nil) == SQLITE_OK else { return [] }
        let hasTable = sqlite3_step(check) == SQLITE_ROW
        sqlite3_finalize(check)
        guard hasTable else { return [] }

        var stmt: OpaquePointer?
        let sql = "SELECT DISTINCT asset_id FROM asset_locations WHERE host != 'mac'"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }
        var out: Set<String> = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 0) { out.insert(String(cString: c)) }
        }
        return out
    }

    /// All asset creation timestamps (for activity stats). Lightweight — one
    /// column, served by the created_at index.
    public func assetCreationTimestamps() throws -> [Date] {
        let sql = "SELECT created_at FROM assets ORDER BY created_at ASC"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        var results: [Date] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append(Date(timeIntervalSince1970: sqlite3_column_double(stmt, 0)))
        }
        return results
    }

    /// Creation timestamps grouped by asset kind (image / video / voice / …),
    /// for per-type activity stats and heatmaps.
    public func assetCreationTimestampsByKind() throws -> [String: [Date]] {
        let sql = "SELECT kind, created_at FROM assets ORDER BY created_at ASC"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        var results: [String: [Date]] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            let kind = (sqlite3_column_text(stmt, 0).map { String(cString: $0) }) ?? "image"
            let date = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 1))
            results[kind, default: []].append(date)
        }
        return results
    }

    /// Distinct prompts from generated assets with use counts, most recent
    /// first (backs the prompt-history section of the Prompt Library).
    public func promptHistory(limit: Int = 100) throws -> [(prompt: String, count: Int, lastUsed: Date)] {
        let sql = """
            SELECT prompt, COUNT(*), MAX(created_at)
            FROM assets
            WHERE prompt IS NOT NULL AND prompt != ''
              AND id NOT IN (SELECT asset_id FROM secured_assets)
            GROUP BY prompt
            ORDER BY MAX(created_at) DESC
            LIMIT ?1
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(limit))

        var results: [(prompt: String, count: Int, lastUsed: Date)] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append((
                prompt: String(cString: sqlite3_column_text(stmt, 0)),
                count: Int(sqlite3_column_int(stmt, 1)),
                lastUsed: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))
            ))
        }
        return results
    }

    /// Fetch assets ordered by creation date (newest first).
    public func fetchAssets(limit: Int = 50, offset: Int = 0) throws -> [DAMAsset] {
        let sql = """
            SELECT id, kind, filename, absolute_path, file_size, sha256,
                   width, height, created_at, modified_at, ingested_at, orphaned,
                   prompt, negative_prompt, seed, steps, guidance,
                   model_family, rating, favorite, content_mode, character_name, source
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
    ///
    /// The desktop search box is raw user text and FTS5 MATCH is a grammar, not
    /// a string: `sun-set`, `a:b` or a lone `"` is a syntax error, which
    /// `sqlite3_step` reports as a failed step and the caller shows as an empty
    /// result. Same sanitiser the catalog uses, so both search boxes read the
    /// same words the same way.
    public func searchPrompts(query: String, limit: Int = 50) throws -> [DAMAsset] {
        guard let match = CatalogStore.ftsMatchExpression(query) else { return [] }
        let sql = """
            SELECT a.id, a.kind, a.filename, a.absolute_path, a.file_size, a.sha256,
                   a.width, a.height, a.created_at, a.modified_at, a.ingested_at, a.orphaned,
                   a.prompt, a.negative_prompt, a.seed, a.steps, a.guidance,
                   a.model_family, a.rating, a.favorite, a.content_mode, a.character_name
            FROM assets_fts fts
            JOIN assets a ON a.id = fts.id
            WHERE assets_fts MATCH ?1
              AND a.id NOT IN (SELECT asset_id FROM secured_assets)
            ORDER BY rank
            LIMIT ?2
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (match as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var results: [DAMAsset] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append(assetFromRow(stmt))
        }
        return results
    }

    /// Delete an asset row and everything keyed to it: its FTS entry, its folder
    /// mapping, and — since the catalog migrated into this same file — its
    /// collection filings and its recorded locations. Unknown ids are a no-op.
    /// Does not touch files on disk — see AssetIngestor.deleteAsset for the full
    /// file + thumbnail + database removal.
    ///
    /// The live schema has no foreign keys, so nothing cascades on its own:
    /// leaving `asset_collections` and `asset_locations` behind leaves rows
    /// pointing at an asset that no longer exists, which inflates every
    /// collection count and hands `assetID(forPath:)` a dead id.
    public func deleteAsset(id: String) throws {
        let sql = "DELETE FROM assets WHERE id = ?1"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw DAMStoreError.execFailed(sql, lastError)
        }
        try deleteFTS(id: id)
        try runSimple("DELETE FROM asset_folders WHERE asset_id = ?1", text: [id])
        // Catalog-side tables. Absent in a DAM-only database (a test fixture, or
        // an install that has never run the backfill), so their absence is not
        // an error.
        for table in ["asset_collections", "asset_locations"] where hasTable(table) {
            try runSimple("DELETE FROM \(table) WHERE asset_id = ?1", text: [id])
        }
        // Edges name their endpoints from_/to_, not asset_id — an edge is a
        // relation between two assets, and losing either end makes it dangling.
        if hasTable("asset_edges") {
            try runSimple("DELETE FROM asset_edges WHERE from_asset_id = ?1 OR to_asset_id = ?1",
                          text: [id])
        }
    }

    /// Whether the catalog has sealed this row. False whenever the question
    /// cannot be asked — a DAM-only database has no `sealed` column at all.
    private func isSealed(id: String) -> Bool {
        var stmt: OpaquePointer?
        let sql = "SELECT sealed FROM assets WHERE id = ?1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return false }
        return sqlite3_column_int(stmt, 0) != 0
    }

    /// Whether a table exists in this database. Table names here are compile-time
    /// literals, never caller input.
    private func hasTable(_ name: String) -> Bool {
        var stmt: OpaquePointer?
        let sql = "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, (name as NSString).utf8String, -1, SQLITE_TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    // MARK: - Asset security

    /// Mark an asset secured: record its original location and point the
    /// asset row at the secured file. File movement is the ingestor's job.
    public func secureAsset(id: String, securedPath: String, originalPath: String) throws {
        let sql = "INSERT OR REPLACE INTO secured_assets (asset_id, original_path, secured_at) VALUES (?1, ?2, ?3)"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, (originalPath as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 3, Date().timeIntervalSince1970)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw DAMStoreError.execFailed(sql, lastError)
        }
        try updateAssetPath(id: id, path: securedPath)
    }

    /// Clear the secured mark and restore the asset row's path. Returns the
    /// original path (for the ingestor to move the file back to), or nil if
    /// the asset wasn't secured.
    @discardableResult
    public func unsecureAsset(id: String) throws -> String? {
        let sql = "SELECT original_path FROM secured_assets WHERE asset_id = ?1"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        var originalPath: String?
        if sqlite3_step(stmt) == SQLITE_ROW {
            originalPath = String(cString: sqlite3_column_text(stmt, 0))
        }
        sqlite3_finalize(stmt)

        guard let originalPath else { return nil }
        try runSimple("DELETE FROM secured_assets WHERE asset_id = ?1", text: [id])
        try updateAssetPath(id: id, path: originalPath)
        return originalPath
    }

    /// Ids of all secured assets (for client-side gallery filtering).
    public func securedAssetIds() throws -> Set<String> {
        let sql = "SELECT asset_id FROM secured_assets"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        var ids = Set<String>()
        while sqlite3_step(stmt) == SQLITE_ROW {
            ids.insert(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return ids
    }

    /// Point an asset row at a new file location (secure/unsecure moves).
    private func updateAssetPath(id: String, path: String) throws {
        let filename = (path as NSString).lastPathComponent
        try runSimple(
            "UPDATE assets SET absolute_path = ?2, filename = ?3 WHERE id = ?1",
            text: [id, path, filename])
    }

    // MARK: - Folders

    /// Create a folder and return it.
    @discardableResult
    public func createFolder(name: String, id: String = UUID().uuidString) throws -> DAMFolder {
        let folder = DAMFolder(id: id, name: name, createdAt: Date())
        let sql = "INSERT INTO folders (id, name, created_at) VALUES (?1, ?2, ?3)"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, (folder.id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, (folder.name as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 3, folder.createdAt.timeIntervalSince1970)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw DAMStoreError.insertFailed(lastError)
        }
        return folder
    }

    /// All folders, sorted by name (case-insensitive).
    public func listFolders() throws -> [DAMFolder] {
        let sql = "SELECT id, name, created_at FROM folders ORDER BY name COLLATE NOCASE ASC"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        var results: [DAMFolder] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append(DAMFolder(
                id: String(cString: sqlite3_column_text(stmt, 0)),
                name: String(cString: sqlite3_column_text(stmt, 1)),
                createdAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))
            ))
        }
        return results
    }

    /// Rename a folder. Unknown ids are a no-op.
    public func renameFolder(id: String, name: String) throws {
        try runSimple("UPDATE folders SET name = ?2 WHERE id = ?1", text: [id, name])
    }

    /// Delete a folder. Its assets are unfiled, not deleted.
    public func deleteFolder(id: String) throws {
        try runSimple("DELETE FROM asset_folders WHERE folder_id = ?1", text: [id])
        try runSimple("DELETE FROM folders WHERE id = ?1", text: [id])
    }

    /// File assets into a folder, or unfile them when `toFolder` is nil.
    /// An asset lives in at most one folder; refiling replaces the mapping.
    public func assignAssets(ids: [String], toFolder folderId: String?) throws {
        for assetId in ids {
            if let folderId {
                try runSimple(
                    "INSERT OR REPLACE INTO asset_folders (asset_id, folder_id) VALUES (?1, ?2)",
                    text: [assetId, folderId])
            } else {
                try runSimple("DELETE FROM asset_folders WHERE asset_id = ?1", text: [assetId])
            }
        }
    }

    /// The full asset→folder mapping (assets without a folder are absent).
    public func folderAssignments() throws -> [String: String] {
        let sql = "SELECT asset_id, folder_id FROM asset_folders"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        var results: [String: String] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            results[String(cString: sqlite3_column_text(stmt, 0))] =
                String(cString: sqlite3_column_text(stmt, 1))
        }
        return results
    }

    /// Asset counts per folder id.
    public func folderCounts() throws -> [String: Int] {
        let sql = "SELECT folder_id, COUNT(*) FROM asset_folders GROUP BY folder_id"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        var results: [String: Int] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            results[String(cString: sqlite3_column_text(stmt, 0))] = Int(sqlite3_column_int(stmt, 1))
        }
        return results
    }

    /// Prepare, bind text parameters in order, and step a one-shot statement.
    private func runSimple(_ sql: String, text: [String]) throws {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }
        for (index, value) in text.enumerated() {
            sqlite3_bind_text(stmt, Int32(index + 1), (value as NSString).utf8String, -1, SQLITE_TRANSIENT)
        }
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw DAMStoreError.execFailed(sql, lastError)
        }
    }

    /// Delete multiple assets by id.
    public func deleteAssets(ids: [String]) throws {
        for id in ids {
            try deleteAsset(id: id)
        }
    }

    /// Fetch a single asset by absolute path, or nil if not tracked.
    public func fetchAsset(byPath path: String) throws -> DAMAsset? {
        let sql = """
            SELECT id, kind, filename, absolute_path, file_size, sha256,
                   width, height, created_at, modified_at, ingested_at, orphaned,
                   prompt, negative_prompt, seed, steps, guidance,
                   model_family, rating, favorite, content_mode, character_name, source
            FROM assets
            WHERE absolute_path = ?1
            LIMIT 1
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DAMStoreError.prepareFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (path as NSString).utf8String, -1, SQLITE_TRANSIENT)

        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return assetFromRow(stmt)
    }

    // MARK: - Private

    /// If the asset's path is already tracked under a different id, keep the
    /// existing id and user annotations so a re-ingest does not reset them
    /// (and does not orphan the thumbnail keyed by id). An insert with a
    /// matching id is an intentional update and is stored as-is.
    private func mergedWithExisting(_ asset: DAMAsset) throws -> DAMAsset {
        guard let existing = try fetchAsset(byPath: asset.absolutePath),
              existing.id != asset.id else {
            return asset
        }

        return DAMAsset(
            id: existing.id,
            kind: asset.kind,
            filename: asset.filename,
            absolutePath: asset.absolutePath,
            fileSize: asset.fileSize,
            sha256: asset.sha256 ?? existing.sha256,
            width: asset.width ?? existing.width,
            height: asset.height ?? existing.height,
            createdAt: asset.createdAt,
            modifiedAt: asset.modifiedAt,
            ingestedAt: existing.ingestedAt,
            orphaned: asset.orphaned,
            prompt: asset.prompt ?? existing.prompt,
            negativePrompt: asset.negativePrompt ?? existing.negativePrompt,
            seed: asset.seed ?? existing.seed,
            steps: asset.steps ?? existing.steps,
            guidance: asset.guidance ?? existing.guidance,
            modelFamily: asset.modelFamily ?? existing.modelFamily,
            rating: existing.rating,
            favorite: existing.favorite,
            contentMode: asset.contentMode ?? existing.contentMode,
            characterName: asset.characterName ?? existing.characterName
        )
    }

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
                character_name TEXT,
                source TEXT
            )
            """)

        // Migration: add `source` to pre-existing databases (idempotent — a
        // duplicate-column error on already-migrated DBs is ignored).
        _ = try? execute("ALTER TABLE assets ADD COLUMN source TEXT")

        try execute("CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at DESC)")

        // Virtual folders. Membership lives in a separate mapping table (one
        // folder per asset) so INSERT OR REPLACE re-ingests of an asset row
        // can't wipe its filing.
        try execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """)
        try execute("""
            CREATE TABLE IF NOT EXISTS asset_folders (
                asset_id TEXT PRIMARY KEY,
                folder_id TEXT NOT NULL
            )
            """)
        try execute("CREATE INDEX IF NOT EXISTS idx_asset_folders_folder ON asset_folders(folder_id)")

        // Secured (sensitive) assets. Separate mapping table so INSERT OR
        // REPLACE re-ingests can't reset the flag; remembers where the file
        // came from so unsecuring can put it back.
        try execute("""
            CREATE TABLE IF NOT EXISTS secured_assets (
                asset_id TEXT PRIMARY KEY,
                original_path TEXT NOT NULL,
                secured_at REAL NOT NULL
            )
            """)

        // FTS5 virtual table for full-text prompt search.
        try execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
                id UNINDEXED,
                prompt,
                negative_prompt
            )
            """)
    }

    private func deleteFTS(id: String) throws {
        let sql = "DELETE FROM assets_fts WHERE id = ?1"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            // FTS delete failure is non-fatal.
            return
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_step(stmt)
    }

    private func insertFTS(id: String, prompt: String, negativePrompt: String?) throws {
        let sql = "INSERT INTO assets_fts (id, prompt, negative_prompt) VALUES (?1, ?2, ?3)"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            // FTS insert failure is non-fatal.
            return
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, (prompt as NSString).utf8String, -1, SQLITE_TRANSIENT)
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
            characterName: columnText(stmt, 21),
            source: columnText(stmt, 22)
        )
    }

    // MARK: - SQLite Helpers

    private func bindOptionalText(_ stmt: OpaquePointer?, _ index: Int32, _ value: String?) {
        if let value = value {
            sqlite3_bind_text(stmt, index, (value as NSString).utf8String, -1, SQLITE_TRANSIENT)
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
    /// The orphan sweep would have removed an implausible share of the library
    /// and refused, deleting nothing. See `DAMStore.pruneCircuitBreakerFraction`.
    case pruneRefused(candidates: Int, total: Int)

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
        case .pruneRefused(let candidates, let total):
            return "\(candidates) of \(total) images look missing — that is usually an "
                + "unmounted volume or a revoked folder permission, not deletions. "
                + "Nothing was removed."
        }
    }
}
