// CatalogStore.swift — the query layer over ~/.comfybox/dam.sqlite3.
//
// Two rules are enforced HERE, in the store, rather than in any caller:
//
//   1. REALM LOCK. `CatalogQuery.scope` is set by the service from which tool
//      is calling — never from client input. When scope == .kira the WHERE
//      clause carries `realm = 'kira'` unconditionally, so no combination of
//      other filters can widen it.
//   2. MODE CLAMP. `ceiling` withholds prompt/caption/path for rows above the
//      active chat mode while KEEPING the tier label, matching the precedent in
//      render-journal.ts (counts are metadata; text and paths are not).
//
// Both are properties of the returned rows, not of the caller's discipline.

import Foundation
import SQLite3

public enum CatalogError: Error, LocalizedError {
    case prepareFailed(String)
    case stepFailed(String)
    case depthCapExceeded
    case notPermitted(String)
    case noSuchCollection(String)
    case noSuchAsset(String)

    public var errorDescription: String? {
        switch self {
        case let .prepareFailed(m): return "prepare failed: \(m)"
        case let .stepFailed(m): return "step failed: \(m)"
        case .depthCapExceeded: return "collections are two levels deep at most"
        case let .notPermitted(m): return "not permitted: \(m)"
        case let .noSuchCollection(id): return "no such collection: \(id)"
        case let .noSuchAsset(id): return "no such asset: \(id)"
        }
    }
}

public enum CatalogOrder: String, Sendable {
    case newest, oldest, rating
}

public struct CatalogQuery: Sendable, CustomStringConvertible {
    /// nil = every realm. Set by the SERVICE, never by client input.
    public var scope: CatalogRealm?
    /// The active chat-mode ceiling; nil = no clamp.
    public var ceiling: String?
    public var text: String?
    public var collectionID: String?
    public var lane: String?
    public var tier: String?
    public var character: String?
    public var source: String?
    public var stock: String?
    public var genre: String?
    public var arc: String?
    public var kind: String?
    public var mode: String?
    public var minDurationMs: Int?
    public var maxDurationMs: Int?
    public var minRating: Int?
    public var since: Date?
    public var until: Date?
    public var orderBy: CatalogOrder = .newest
    public var limit: Int = 50
    public var offset: Int = 0

    public init(scope: CatalogRealm? = nil, ceiling: String? = nil, text: String? = nil,
                collectionID: String? = nil, lane: String? = nil, tier: String? = nil,
                character: String? = nil, source: String? = nil, stock: String? = nil,
                genre: String? = nil, arc: String? = nil, kind: String? = nil,
                mode: String? = nil, minDurationMs: Int? = nil, maxDurationMs: Int? = nil,
                minRating: Int? = nil, since: Date? = nil, until: Date? = nil,
                orderBy: CatalogOrder = .newest, limit: Int = 50, offset: Int = 0) {
        self.scope = scope; self.ceiling = ceiling; self.text = text
        self.collectionID = collectionID; self.lane = lane; self.tier = tier
        self.character = character; self.source = source; self.stock = stock
        self.genre = genre; self.arc = arc; self.kind = kind; self.mode = mode
        self.minDurationMs = minDurationMs; self.maxDurationMs = maxDurationMs
        self.minRating = minRating; self.since = since; self.until = until
        self.orderBy = orderBy; self.limit = limit; self.offset = offset
    }

    public var description: String {
        "CatalogQuery(scope: \(scope?.rawValue ?? "all"), collection: \(collectionID ?? "-"), lane: \(lane ?? "-"))"
    }
}

public struct CatalogFacets: Sendable, Equatable {
    public var lane: [String: Int] = [:]
    public var tier: [String: Int] = [:]
    public var character: [String: Int] = [:]
    public var source: [String: Int] = [:]
    public var stock: [String: Int] = [:]
    public var genre: [String: Int] = [:]
    public var kind: [String: Int] = [:]
    public var mode: [String: Int] = [:]
    public var collection: [String: Int] = [:]
    public init() {}
}

/// The fruit tiers, least to most explicit. Used only for ceiling comparison.
///
/// `apple` is deliberately NOT a rung of its own. The daemon canonicalizes
/// `apple -> neutral` on write (content-mode-tools.ts) and nothing a
/// conversation can do produces a ceiling of `apple`, so ranking apple ABOVE
/// neutral made 278 rows — 164 of them the whole `col-kira-everyday` genre —
/// countable but permanently unopenable at every ceiling anyone could ask for.
/// The owner's call: collapse them here to match what the daemon already does
/// everywhere. Rows still STORE the literal string "apple"; only its RANK moved.
public let CATALOG_TIER_ORDER: [String] = ["neutral", "banana", "avocado"]

/// The least permissive ceiling: the lowest rung, admitting SFW content only.
///
/// Derived from the order rather than spelled "neutral" at each call site, so a
/// consumer asking "is this above SFW?" tracks the ladder instead of pinning a
/// word to it. This is what the desktop content gate clamps against.
public let CATALOG_STRICTEST_CEILING: String = CATALOG_TIER_ORDER[0]

/// Other vocabularies that mean the same thing. The desktop gate already sees
/// these values in `content_mode` (NSFWGate.swift: "explicit", "suggestive" and
/// "nsfw" are all treated as NSFW), so the catalog must rank them rather than
/// meet them as strangers.
public let CATALOG_TIER_ALIASES: [String: String] = [
    "apple": "neutral",
    "explicit": "avocado",
    "nsfw": "avocado",
    "suggestive": "banana",
]

/// Every spelling the ladder RECOGNISES, canonical rungs and aliases alike.
///
/// This is not the same set as `CATALOG_TIER_ORDER` and must never be confused
/// with it. The order is a RANKING (three rungs); this is a VOCABULARY (every
/// word that names a tier). Anywhere a tier is matched as a literal — a
/// directory name on disk, a filter enum, a sidecar's `content_mode` — it is
/// this set that applies, because `apple/` directories exist in the archive and
/// `apple` is still written into rows. Deriving those places from the three-rung
/// order instead would silently stop recognising apple.
public let CATALOG_TIER_SPELLINGS: Set<String> =
    Set(CATALOG_TIER_ORDER).union(CATALOG_TIER_ALIASES.keys)

/// How explicit an ASSET's tier is. Fails CLOSED: a non-nil tier this table does
/// not recognise ranks above every known tier, so an unrecognised vocabulary is
/// withheld from every ceiling rather than waved through as if it were neutral.
/// nil stays 0 — an asset that was never tiered is untiered, not secretly explicit.
func tierRank(_ tier: String?) -> Int {
    guard let raw = tier?.lowercased().trimmingCharacters(in: .whitespacesAndNewlines),
          !raw.isEmpty else { return 0 }
    if let i = CATALOG_TIER_ORDER.firstIndex(of: raw) { return i }
    if let alias = CATALOG_TIER_ALIASES[raw], let i = CATALOG_TIER_ORDER.firstIndex(of: alias) {
        return i
    }
    return CATALOG_TIER_ORDER.count
}

/// How permissive a CEILING is. Fails closed in the other direction: an
/// unrecognised ceiling admits the least, so a typo or an unknown chat mode
/// clamps everything above neutral instead of clamping nothing.
///
/// This is deliberately NOT `tierRank`. The two sides round opposite ways, and
/// one shared function would have to fail open on one of them.
func ceilingRank(_ ceiling: String) -> Int {
    let raw = ceiling.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    if let i = CATALOG_TIER_ORDER.firstIndex(of: raw) { return i }
    if let alias = CATALOG_TIER_ALIASES[raw], let i = CATALOG_TIER_ORDER.firstIndex(of: alias) {
        return i
    }
    return 0
}

/// THE ONE PUBLIC ANSWER to "is this asset above the caller's ceiling?".
///
/// The two ranks above are INTERNAL on purpose, and this is why. They were
/// `public func tierRank` beside an internal `ceilingRank`, which is an
/// asymmetry with teeth: a cross-module caller reaching for the obvious
/// `tierRank(asset) > tierRank(ceiling)` gets a comparison in which an
/// unrecognised ceiling ranks ABOVE every tier and therefore withholds nothing —
/// precisely the fail-open `ceilingRank` exists to prevent. Exposing the
/// decision rather than its two halves makes that spelling unavailable.
///
/// nil ceiling = no clamp.
public func isWithheld(tier: String?, ceiling: String?) -> Bool {
    guard let ceiling else { return false }
    return tierRank(tier) > ceilingRank(ceiling)
}

/// One bound parameter value, captured while the WHERE clause is assembled and
/// applied positionally once the statement is prepared.
///
/// This replaces the brief's `private var pending: OpaquePointer?` handle. The
/// values are collected in the same order the `?N` placeholders are emitted, so
/// the numbering is identical; carrying values rather than closures over a
/// mutable statement handle removes the shared-state hazard entirely (nothing
/// can bind into a stale statement, and no `[weak self]` capture can silently
/// no-op a bind and leave a parameter NULL — which, for the realm parameter,
/// would have been a leak).
private enum CatalogBind {
    case text(String)
    case int(Int64)
    case double(Double)
    case null

    /// `.text` for a value, `.null` for nil — so a nullable column is one case
    /// rather than a branch at every call site.
    static func opt(_ v: String?) -> CatalogBind { v.map { CatalogBind.text($0) } ?? .null }

    func apply(to stmt: OpaquePointer?, at index: Int32) {
        switch self {
        case let .text(v):
            sqlite3_bind_text(stmt, index, (v as NSString).utf8String, -1, CatalogSchema.transient)
        case let .int(v):
            sqlite3_bind_int64(stmt, index, v)
        case let .double(v):
            sqlite3_bind_double(stmt, index, v)
        case .null:
            sqlite3_bind_null(stmt, index)
        }
    }
}

public actor CatalogStore {
    private var db: OpaquePointer?
    public let dbPath: String

    /// True only when `open` resolved the default `~/.comfybox` location itself.
    /// The 0700 directory contract belongs to THAT directory, not to whatever
    /// directory a caller's `--db` happens to sit in.
    private let ownsDirectory: Bool

    private init(db: OpaquePointer, dbPath: String, ownsDirectory: Bool) {
        self.db = db
        self.dbPath = dbPath
        self.ownsDirectory = ownsDirectory
    }

    public static func open(path: String? = nil) async throws -> CatalogStore {
        let resolved: String
        if let p = path {
            resolved = p
        } else {
            let dir = NSString(string: "~/.comfybox").expandingTildeInPath
            try FileManager.default.createDirectory(
                atPath: dir, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            // `attributes:` applies only when createDirectory actually creates the
            // directory. ~/.comfybox already exists at 0755 in the field, so the
            // mode has to be set unconditionally or the 0700 half of the contract
            // never lands on any existing install.
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: dir)
            resolved = (dir as NSString).appendingPathComponent("dam.sqlite3")
        }

        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(resolved, &handle, flags, nil) == SQLITE_OK, let h = handle else {
            let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            if let x = handle { sqlite3_close(x) }
            throw CatalogError.prepareFailed(msg)
        }

        let store = CatalogStore(db: h, dbPath: resolved, ownsDirectory: path == nil)
        try await store.initialize()
        return store
    }

    private func initialize() throws {
        // sqlite3_open_v2 has ALREADY created the file by the time we get here, so
        // tighten before writing a byte to it — and again on the way out, whether
        // we leave by return or by throw, to catch the -wal and -shm that
        // journal_mode=WAL creates. A migration that throws must not leave raw
        // prompt text sitting at the umask default.
        tightenPermissions()
        defer { tightenPermissions() }

        // busy_timeout FIRST, and the order is load-bearing. `journal_mode=WAL`
        // takes an exclusive lock on the database; with no timeout yet set, a
        // busy file makes that very statement return SQLITE_BUSY, `open` throws,
        // and the gallery service exits(1) — the service refusing to start
        // precisely when the database is in use, which is when it is wanted.
        // Setting the timeout cannot itself be busy: it is a connection-local
        // setting that takes no lock.
        try CatalogSchema.exec(db, "PRAGMA busy_timeout = 5000")
        // Four consumers share this file under WAL and the desktop app writes to
        // it concurrently; without a busy timeout a collision is an immediate
        // SQLITE_BUSY rather than a short wait.
        try CatalogSchema.exec(db, "PRAGMA journal_mode=WAL")
        // `migrate` deliberately fails when `assets` is missing entirely (it must
        // never mask a real problem on the live database). A brand-new file has
        // no `assets` table at all, so create the base shape first; it is
        // CREATE TABLE IF NOT EXISTS, so an existing database is untouched.
        try CatalogSchema.ensureBaseSchema(db: db)
        try CatalogSchema.migrate(db: db)
    }

    /// Catalog holds raw prompt text under the extended 2026-07-07 provenance
    /// contract: 0600 file inside a 0700 directory. Applied on every open so a
    /// file created by another process is corrected too.
    private func tightenPermissions() {
        let fm = FileManager.default
        try? fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: dbPath)
        for suffix in ["-wal", "-shm"] {
            try? fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: dbPath + suffix)
        }
        // ONLY when this store chose the directory. Chmodding the parent of a
        // caller-supplied `--db` silently changes the mode of whatever directory
        // they named — `--db ~/Desktop/x.sqlite3` would make ~/Desktop 0700. The
        // dry runs escaped that only because /tmp is root-owned and the `try?`
        // swallowed the failure.
        guard ownsDirectory else { return }
        try? fm.setAttributes([.posixPermissions: 0o700],
                              ofItemAtPath: (dbPath as NSString).deletingLastPathComponent)
    }

    deinit { if let db = db { sqlite3_close(db) } }

    // MARK: - Write

    /// Insert or update an asset and (re-)apply its non-manual collection
    /// membership. Manual filings (`manual = 1`) are never removed here —
    /// precedence is manual > explicit > derived.
    /// One asset's row, its FTS entry and its filing are one fact, so they move
    /// together or not at all. Without this, a throw between the membership
    /// DELETE and the re-INSERT leaves an asset filed nowhere, and a throw
    /// between the FTS delete and insert leaves it permanently unsearchable —
    /// both silent, both only visible much later as an absence.
    public func upsert(_ asset: CatalogAsset, explicitCollectionIDs: [String]) throws {
        try CatalogSchema.exec(db, "SAVEPOINT catalog_upsert")
        do {
            try writeAssetRow(asset)
            try reindexFTS(asset)
            try applyDerivedFiling(asset, explicitCollectionIDs: explicitCollectionIDs)
            try CatalogSchema.exec(db, "RELEASE catalog_upsert")
        } catch {
            // ROLLBACK TO rewinds but LEAVES the savepoint on the stack; the
            // RELEASE pops it so a later upsert on this connection starts clean.
            _ = try? CatalogSchema.exec(db, "ROLLBACK TO catalog_upsert")
            _ = try? CatalogSchema.exec(db, "RELEASE catalog_upsert")
            throw error
        }
    }

    private func writeAssetRow(_ asset: CatalogAsset) throws {
        let sql = """
            INSERT INTO assets (
                id, kind, filename, absolute_path, file_size, sha256, width, height,
                created_at, modified_at, ingested_at, orphaned,
                prompt, negative_prompt, seed, steps, guidance, model_family,
                rating, favorite, content_mode, character_name, source,
                realm, sealed, lane, arc, theme, stock, genre, family, style,
                preset, loras, render_id, caption, caption_source, prompt_raw,
                mode, duration_ms, fps, frames, resolution, aspect_ratio,
                prompt_injected
            ) VALUES (
                ?1,?2,?3,?4,?5,?6,?7,?8,
                ?9,?9,?9,0,
                ?10,?11,?12,?13,?14,?15,
                ?16,?17,?18,?19,?20,
                ?21,?22,?23,?24,?25,?26,?27,?28,?29,
                ?30,?31,?32,?33,?34,?35,
                ?36,?37,?38,?39,?40,?41,
                ?42
            )
            ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind, filename=excluded.filename,
                absolute_path=excluded.absolute_path, file_size=excluded.file_size,
                sha256=COALESCE(excluded.sha256, assets.sha256),
                width=COALESCE(excluded.width, assets.width),
                height=COALESCE(excluded.height, assets.height),
                prompt=excluded.prompt, negative_prompt=excluded.negative_prompt,
                seed=COALESCE(excluded.seed, assets.seed),
                steps=COALESCE(excluded.steps, assets.steps),
                guidance=COALESCE(excluded.guidance, assets.guidance),
                model_family=COALESCE(excluded.model_family, assets.model_family),
                content_mode=COALESCE(excluded.content_mode, assets.content_mode),
                character_name=COALESCE(excluded.character_name, assets.character_name),
                source=COALESCE(excluded.source, assets.source),
                realm=excluded.realm, sealed=excluded.sealed,
                lane=COALESCE(excluded.lane, assets.lane),
                arc=COALESCE(excluded.arc, assets.arc),
                theme=COALESCE(excluded.theme, assets.theme),
                stock=COALESCE(excluded.stock, assets.stock),
                genre=COALESCE(excluded.genre, assets.genre),
                family=COALESCE(excluded.family, assets.family),
                style=COALESCE(excluded.style, assets.style),
                preset=COALESCE(excluded.preset, assets.preset),
                loras=COALESCE(excluded.loras, assets.loras),
                render_id=COALESCE(excluded.render_id, assets.render_id),
                caption=excluded.caption, caption_source=excluded.caption_source,
                prompt_raw=excluded.prompt_raw,
                mode=COALESCE(excluded.mode, assets.mode),
                duration_ms=COALESCE(excluded.duration_ms, assets.duration_ms),
                fps=COALESCE(excluded.fps, assets.fps),
                frames=COALESCE(excluded.frames, assets.frames),
                resolution=COALESCE(excluded.resolution, assets.resolution),
                aspect_ratio=COALESCE(excluded.aspect_ratio, assets.aspect_ratio),
                prompt_injected=excluded.prompt_injected
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }

        bindText(stmt, 1, asset.id)
        bindText(stmt, 2, asset.kind)
        bindText(stmt, 3, asset.filename)
        bindText(stmt, 4, asset.absolutePath)
        sqlite3_bind_int64(stmt, 5, asset.fileSize)
        bindText(stmt, 6, asset.sha256)
        bindInt(stmt, 7, asset.width)
        bindInt(stmt, 8, asset.height)
        sqlite3_bind_double(stmt, 9, asset.createdAt.timeIntervalSince1970)
        bindText(stmt, 10, asset.prompt)
        bindText(stmt, 11, asset.negativePrompt)
        bindInt(stmt, 12, asset.seed)
        bindInt(stmt, 13, asset.steps)
        bindDouble(stmt, 14, asset.guidance)
        bindText(stmt, 15, asset.modelFamily)
        sqlite3_bind_int(stmt, 16, Int32(clamping: asset.rating))
        sqlite3_bind_int(stmt, 17, asset.favorite ? 1 : 0)
        bindText(stmt, 18, asset.contentMode)
        bindText(stmt, 19, asset.characterName)
        bindText(stmt, 20, asset.source)
        bindText(stmt, 21, asset.realm.rawValue)
        sqlite3_bind_int(stmt, 22, asset.sealed ? 1 : 0)
        bindText(stmt, 23, asset.lane)
        bindText(stmt, 24, asset.arc)
        bindText(stmt, 25, asset.theme)
        bindText(stmt, 26, asset.stock)
        bindText(stmt, 27, asset.genre)
        bindText(stmt, 28, asset.family)
        bindText(stmt, 29, asset.style)
        bindText(stmt, 30, asset.preset)
        bindText(stmt, 31, asset.loras)
        bindText(stmt, 32, asset.renderID)
        bindText(stmt, 33, asset.caption)
        bindText(stmt, 34, asset.captionSource)
        bindText(stmt, 35, asset.promptRaw)
        bindText(stmt, 36, asset.mode)
        bindInt(stmt, 37, asset.durationMs)
        bindDouble(stmt, 38, asset.fps)
        bindInt(stmt, 39, asset.frames)
        bindText(stmt, 40, asset.resolution)
        bindText(stmt, 41, asset.aspectRatio)
        bindText(stmt, 42, asset.promptInjected)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw CatalogError.stepFailed(String(cString: sqlite3_errmsg(db)))
        }
    }

    /// A sealed row is never full-text indexed — that is what makes it
    /// unreachable by its own prompt.
    private func reindexFTS(_ asset: CatalogAsset) throws {
        try execBind("DELETE FROM assets_fts WHERE id = ?1", [.text(asset.id)])
        guard !asset.sealed else { return }
        // negativePrompt counts: a row whose only text is a negative prompt is
        // still findable by it, and omitting it here made that row invisible.
        let hasText = (asset.prompt ?? asset.caption ?? asset.promptRaw
                        ?? asset.promptInjected ?? asset.negativePrompt) != nil
        guard hasText else { return }
        // All three prompt spellings ride the prompt column so every phrasing is
        // findable: the sidecars carry prompt_optimized, prompt_raw AND
        // prompt_injected, and which one holds the words a search will use
        // varies by producer.
        let promptText = [asset.prompt, asset.promptRaw, asset.promptInjected]
            .compactMap { $0 }.joined(separator: " ")
        try execBind("""
            INSERT INTO assets_fts (id, prompt, negative_prompt, caption)
            VALUES (?1, ?2, ?3, ?4)
            """,
            [.text(asset.id), .text(promptText),
             .text(asset.negativePrompt ?? ""), .text(asset.caption ?? "")])
    }

    private func applyDerivedFiling(_ asset: CatalogAsset, explicitCollectionIDs: [String]) throws {
        // Replace only non-manual memberships; manual filings survive.
        try execBind("DELETE FROM asset_collections WHERE asset_id = ?1 AND manual = 0",
                     [.text(asset.id)])
        // Explicit filings ADD to the derived ones rather than replacing them: an
        // asset the caller also files by hand is still the tile / shoot / scene
        // its facets say it is, so it must stay in the body of work the rules
        // derive as well as the one the caller named.
        let derived = CollectionRules.defaultCollectionIDs(for: asset)
        let ids = Set(derived).union(explicitCollectionIDs)
        for cid in ids {
            // A shared asset can never enter a kira collection, whatever the caller says.
            if asset.realm == .shared, try collectionRealm(cid) == .kira { continue }
            try execBind("""
                INSERT OR IGNORE INTO asset_collections (asset_id, collection_id, manual)
                VALUES (?1, ?2, 0)
                """,
                [.text(asset.id), .text(cid)])
        }
    }

    /// Re-run derived filing for EVERY row, and report how many assets ended up
    /// in at least one collection.
    ///
    /// Derived filing is otherwise reachable only through `upsert`, and a
    /// re-sweep takes the duplicate branch for almost every asset — which
    /// upserts only when the folded row differs from the stored one. Nothing on
    /// disk changes between sweeps, so nothing differs, so no filing runs. The
    /// second live backfill demonstrated exactly that: `indexed: 0`,
    /// `merged: 4734`, and the only new memberships came from the 10 rows whose
    /// realm had changed. Without this, shipping new `CollectionRules` would
    /// report success and file nothing.
    ///
    /// `applyDerivedFiling` deletes only `manual = 0` rows, so hand-filings
    /// survive; that is the whole reason this is safe to re-run.
    @discardableResult
    public func refileAll() throws -> Int {
        var ids: [String] = []
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT id FROM assets", -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        // Count-critical and re-derives every asset's collection filing below —
        // a mid-iteration step failure (#357) must not silently pass a truncated
        // `ids` list on to a "refiled: N assets" report that names the wrong N.
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            if let c = sqlite3_column_text(stmt, 0) { ids.append(String(cString: c)) }
        }
        let stepError = rc == SQLITE_DONE ? nil : String(cString: sqlite3_errmsg(db))
        sqlite3_finalize(stmt)
        if let stepError { throw CatalogError.stepFailed(stepError) }

        // One transaction for the lot: a partial refile would leave the catalog
        // filed under a mix of the old and new rules with no way to tell which.
        try CatalogSchema.exec(db, "SAVEPOINT catalog_refile")
        do {
            for id in ids {
                guard let asset = try asset(id: id) else { continue }
                // No explicit ids: this re-derives, it does not invent filings the
                // caller never asked for.
                try applyDerivedFiling(asset, explicitCollectionIDs: [])
            }
            try CatalogSchema.exec(db, "RELEASE catalog_refile")
        } catch {
            _ = try? CatalogSchema.exec(db, "ROLLBACK TO catalog_refile")
            _ = try? CatalogSchema.exec(db, "RELEASE catalog_refile")
            throw error
        }
        return ids.count - (try unfiledAssetCount())
    }

    /// How many of an asset's memberships are flagged `manual = 1`.
    ///
    /// Membership alone cannot tell a preserved hand-filing from one that was
    /// deleted and re-derived: both leave a row. Only the flag distinguishes
    /// them, and the flag is what the next refile consults.
    public func manualFilingCount(assetID: String) throws -> Int {
        var stmt: OpaquePointer?
        let sql = "SELECT COUNT(*) FROM asset_collections WHERE asset_id = ?1 AND manual = 1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, assetID)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    // MARK: - Read

    /// The `WHERE` clause and its positional binds for `query`, shared by
    /// `search` (which adds `ORDER BY`/`LIMIT`/`OFFSET` on top) and
    /// `assetIDs(matching:)` (which does not page at all — see its doc).
    /// Extracted so the two can never drift into answering "which rows match"
    /// differently (#271): Select All asking the ids-only path for "everything
    /// matching this filter" must see exactly the same rows `search` would
    /// have paged through.
    private func whereClause(for query: CatalogQuery) -> (sql: String, binds: [CatalogBind]) {
        var wheres: [String] = []
        var binds: [CatalogBind] = []
        /// The 1-based placeholder index the NEXT appended bind will occupy.
        var next: Int32 { Int32(binds.count) + 1 }

        // THE REALM LOCK. Unconditional, first, and not reachable from client input.
        if let scope = query.scope {
            wheres.append("a.realm = ?\(next)")
            binds.append(.text(scope.rawValue))
        }
        if let t = query.text, !t.isEmpty {
            if let match = Self.ftsMatchExpression(t) {
                wheres.append("a.id IN (SELECT id FROM assets_fts WHERE assets_fts MATCH ?\(next))")
                binds.append(.text(match))
            } else {
                // The caller asked for text we cannot turn into a query. Return
                // nothing rather than silently dropping the filter and widening.
                wheres.append("0")
            }
        }
        if let c = query.collectionID {
            // Matches the collection AND its children — asking for Photography
            // returns Autocord Still Life too. One parameter, referenced twice.
            let i = next
            wheres.append("""
                a.id IN (SELECT asset_id FROM asset_collections
                         WHERE collection_id = ?\(i)
                            OR collection_id IN (SELECT id FROM collections WHERE parent_id = ?\(i)))
                """)
            binds.append(.text(c))
        }
        func eq(_ column: String, _ value: String?) {
            guard let v = value else { return }
            wheres.append("a.\(column) = ?\(next)")
            binds.append(.text(v))
        }
        eq("lane", query.lane); eq("content_mode", query.tier)
        eq("character_name", query.character); eq("source", query.source)
        eq("stock", query.stock); eq("genre", query.genre); eq("arc", query.arc)
        eq("kind", query.kind); eq("mode", query.mode)

        func cmp(_ column: String, _ op: String, _ value: Int?) {
            guard let v = value else { return }
            wheres.append("a.\(column) \(op) ?\(next)")
            binds.append(.int(Int64(v)))
        }
        cmp("duration_ms", ">=", query.minDurationMs)
        cmp("duration_ms", "<=", query.maxDurationMs)
        cmp("rating", ">=", query.minRating)

        func date(_ column: String, _ op: String, _ value: Date?) {
            guard let v = value else { return }
            wheres.append("a.\(column) \(op) ?\(next)")
            binds.append(.double(v.timeIntervalSince1970))
        }
        date("created_at", ">=", query.since)
        date("created_at", "<=", query.until)

        let whereSQL = wheres.isEmpty ? "" : "WHERE " + wheres.joined(separator: " AND ")
        return (whereSQL, binds)
    }

    public func search(_ query: CatalogQuery) throws -> [CatalogAsset] {
        let (whereSQL, binds) = whereClause(for: query)
        let order: String
        switch query.orderBy {
        case .newest: order = "a.created_at DESC"
        case .oldest: order = "a.created_at ASC"
        case .rating: order = "a.rating DESC, a.created_at DESC"
        }
        let limitIndex = Int32(binds.count) + 1
        let offsetIndex = limitIndex + 1

        let sql = """
            SELECT a.id, a.kind, a.filename, a.absolute_path, a.sha256, a.file_size,
                   a.width, a.height, a.created_at, a.realm, a.source, a.sealed,
                   a.prompt, a.negative_prompt, a.prompt_raw, a.caption, a.caption_source,
                   a.seed, a.steps, a.guidance, a.model_family, a.preset, a.loras,
                   a.render_id, a.content_mode, a.character_name,
                   a.lane, a.arc, a.theme, a.stock, a.genre, a.family, a.style,
                   a.mode, a.duration_ms, a.fps, a.frames, a.resolution, a.aspect_ratio,
                   a.rating, a.favorite, a.prompt_injected
            FROM assets a
            \(whereSQL)
            ORDER BY \(order)
            LIMIT ?\(limitIndex) OFFSET ?\(offsetIndex)
            """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }

        for (i, b) in binds.enumerated() { b.apply(to: stmt, at: Int32(i + 1)) }
        // int64, not Int32(...): limit and offset arrive from MCP tool arguments
        // and chat commands, and `Int32(someHugeInt)` TRAPS — crashing the process
        // that hosts the actor on what should be a bad-input no-op.
        sqlite3_bind_int64(stmt, limitIndex, Int64(query.limit))
        sqlite3_bind_int64(stmt, offsetIndex, Int64(query.offset))

        var rows: [CatalogAsset] = []
        var rc = sqlite3_step(stmt)
        while rc == SQLITE_ROW {
            rows.append(clamp(rowToAsset(stmt), to: query.ceiling))
            rc = sqlite3_step(stmt)
        }
        guard rc == SQLITE_DONE else {
            throw CatalogError.stepFailed(String(cString: sqlite3_errmsg(db)))
        }
        return rows
    }

    /// A hard ceiling on `assetIDs(matching:)`, independent of any HTTP-layer
    /// page clamp. Not the 500-row UI page size (`GalleryServer.maxLimit`) —
    /// this is "every id matching a filter", the shape Select All actually
    /// needs (#271), so it stays generous. It exists only so a filter that
    /// matches most of a huge catalog cannot return an unbounded array; a
    /// desktop library is nowhere near six figures today.
    public static let idsHardCap = 100_000

    /// The result of `assetIDs(matching:)` — ids AND whether the hard cap cut
    /// them off. R1 review correction: the cap used to truncate silently,
    /// which is exactly the bug #271 exists to fix, just moved one layer
    /// down — a caller cannot tell "this filter matched `idsHardCap` rows on
    /// the nose" from "this filter matched more and got cut", so `truncated`
    /// has to travel with the ids rather than being inferred from the count.
    public struct AssetIDsResult: Sendable, Equatable {
        public let ids: [String]
        /// True when more rows matched `query` than `idsHardCap` allows
        /// through. `ids` is a PREFIX in that case, not the whole filtered
        /// set, and callers (the HTTP route, Select All) must say so rather
        /// than presenting it as complete.
        public let truncated: Bool
    }

    /// Every asset id matching `query`, ignoring `query.limit`/`query.offset`
    /// entirely — deliberately, not merely widened. A "select all" that pages
    /// is the exact bug this exists to fix (#271: the gallery raised its page
    /// size to 20k and still came back ~500 at a time because something kept
    /// re-clamping it); the fix is a query shape with no page to clamp.
    ///
    /// Ids only, not full rows: selection is a set of ids, and fetching
    /// `idsHardCap`-many full `CatalogAsset` rows (prompts, captions, paths)
    /// just to read their primary keys is the wasted work a page size can
    /// never fix.
    ///
    /// Count-critical (#357): the caller uses this list to decide what "all"
    /// means, so a mid-iteration step failure throws rather than silently
    /// handing back a truncated selection that LOOKS like the whole filtered
    /// set.
    ///
    /// `cap` defaults to `idsHardCap` for every real caller; it is a
    /// parameter (not a hardcoded `Self.idsHardCap` in the SQL) only so a test
    /// can pin the truncation behaviour without inserting 100,001 rows.
    public func assetIDs(matching query: CatalogQuery, cap: Int = CatalogStore.idsHardCap) throws -> AssetIDsResult {
        let (whereSQL, binds) = whereClause(for: query)
        // Ask for one more than the cap: if it comes back, more than `cap`
        // rows matched and the result is genuinely truncated, not merely
        // exactly-the-cap-sized.
        let sql = """
            SELECT a.id FROM assets a
            \(whereSQL)
            ORDER BY a.created_at DESC
            LIMIT \(cap + 1)
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        for (i, b) in binds.enumerated() { b.apply(to: stmt, at: Int32(i + 1)) }

        var ids: [String] = []
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            if let id = text(stmt, 0) { ids.append(id) }
        }
        guard rc == SQLITE_DONE else {
            throw CatalogError.stepFailed(String(cString: sqlite3_errmsg(db)))
        }
        let truncated = ids.count > cap
        if truncated { ids.removeLast(ids.count - cap) }
        return AssetIDsResult(ids: ids, truncated: truncated)
    }

    /// Turn free text into an FTS5 MATCH expression that cannot be a syntax
    /// error. Each whitespace-separated term becomes a quoted phrase (so a
    /// hyphen, a colon or a stray quote in a user's words is data, not an
    /// operator) and the terms are space-joined, which FTS5 reads as AND.
    /// Returns nil when nothing searchable survives.
    /// Public because the DESKTOP search box needs the same sanitiser: it binds
    /// its own raw text into an FTS5 MATCH, where `sun-set` is a syntax error
    /// that reads to the user as "no results". One sanitiser, so the two search
    /// boxes cannot disagree about what a hyphen means.
    public static func ftsMatchExpression(_ raw: String) -> String? {
        let terms = raw
            .split(whereSeparator: { $0.isWhitespace })
            .filter { $0.contains(where: { $0 != "\"" }) }
            .map { "\"\($0.replacingOccurrences(of: "\"", with: "\"\""))\"" }
        return terms.isEmpty ? nil : terms.joined(separator: " ")
    }

    /// THE MODE CLAMP. Above the ceiling the tier LABEL survives — it is
    /// metadata — while text and the file path do not. Matches render-journal.ts.
    ///
    /// AN ALLOW-LIST, NOT A DENY-LIST. It was written the other way — copy every
    /// one of ~40 fields across and nil out the handful that looked like text —
    /// and it leaked exactly as that shape always does. `theme` was carried
    /// through as if it were a facet; it is not. `MetadataReader` reads it out of
    /// render-journal.jsonl, where it holds a free-text intent line, and 147 rows
    /// above neutral in the live catalog carry one over 60 characters. At ceiling
    /// `neutral` an avocado row correctly withheld its prompt and returned a full
    /// explicit sentence in `theme`.
    ///
    /// So the clamped row is BUILT from the fields named here and nothing else.
    /// A field added to `CatalogAsset` tomorrow is withheld by default; making it
    /// visible above the ceiling takes an edit to this list, which is a line a
    /// reviewer can see. The list holds only closed-vocabulary facets, numbers
    /// and structural identity — no field that any writer can put a sentence in.
    /// Withheld here and deliberately absent below: `filename` and
    /// `absolutePath` (in this library the filename IS the prompt), every
    /// prompt/caption spelling, `theme`, `arc`, `preset`, `loras` and `renderID`.
    private func clamp(_ a: CatalogAsset, to ceiling: String?) -> CatalogAsset {
        guard isWithheld(tier: a.contentMode, ceiling: ceiling) else { return a }
        return CatalogAsset(
            // Identity and shape — no free text in any of them.
            id: a.id, kind: a.kind, filename: "", absolutePath: "",
            sha256: a.sha256, fileSize: a.fileSize, width: a.width, height: a.height,
            createdAt: a.createdAt,
            // Provenance. `source` is an application name from a fixed set.
            realm: a.realm, source: a.source, sealed: a.sealed,
            // Generation numerics and the model's own name.
            seed: a.seed, steps: a.steps, guidance: a.guidance, modelFamily: a.modelFamily,
            // The tier LABEL survives the clamp by design; character is a facet.
            contentMode: a.contentMode, characterName: a.characterName,
            // Facets — every one of these is offered by /v1/catalog/facets and
            // filterable on /v1/catalog/search, i.e. already a closed vocabulary.
            lane: a.lane, stock: a.stock, genre: a.genre,
            // Video shape.
            mode: a.mode, durationMs: a.durationMs, fps: a.fps, frames: a.frames,
            resolution: a.resolution, aspectRatio: a.aspectRatio,
            // The viewer's own annotations.
            rating: a.rating, favorite: a.favorite)
    }

    // MARK: - Collections

    public func collections(visibleTo realm: CatalogRealm?) throws -> [CatalogCollection] {
        var out: [CatalogCollection] = []
        let sql = """
            SELECT id, slug, name, parent_id, realm, description FROM collections
            WHERE realm IS NULL OR ?1 IS NULL OR realm = ?1
            ORDER BY parent_id IS NOT NULL, name
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, realm?.rawValue)
        // Read-only listing for the collections rail (#357): a mid-iteration
        // step failure logs and returns whatever rows were seen rather than
        // throwing — a short rail is a visible degradation, not a corrupted
        // write, and the sidebar is not the place to turn a transient SQLITE_BUSY
        // into a hard failure of browsing.
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            out.append(CatalogCollection(
                id: text(stmt, 0) ?? "", slug: text(stmt, 1) ?? "", name: text(stmt, 2) ?? "",
                parentID: text(stmt, 3),
                realm: text(stmt, 4).flatMap { CatalogRealm(rawValue: $0) },
                description: text(stmt, 5)))
        }
        if rc != SQLITE_DONE {
            FileHandle.standardError.write(
                Data("CatalogStore.collections: step failed (code \(rc)): \(String(cString: sqlite3_errmsg(db)))\n".utf8))
        }
        return out
    }

    public func createCollection(_ c: CatalogCollection, by actor: CatalogRealm?) throws {
        try requireOwnership(of: c.realm, by: actor, action: "create")
        if let parent = c.parentID {
            guard let parentRow = try collectionRow(parent) else {
                throw CatalogError.noSuchCollection(parent)
            }
            guard parentRow.parentID == nil else { throw CatalogError.depthCapExceeded }
        }
        try execBind("""
            INSERT INTO collections (id, slug, name, parent_id, realm, description, created_at)
            VALUES (?1,?2,?3,?4,?5,?6,?7)
            """,
            [.text(c.id), .text(c.slug), .text(c.name),
             .opt(c.parentID), .opt(c.realm?.rawValue), .opt(c.description),
             .double(Date().timeIntervalSince1970)])
    }

    public func renameCollection(id: String, name: String, by actor: CatalogRealm?) throws {
        guard let row = try collectionRow(id) else { throw CatalogError.noSuchCollection(id) }
        try requireOwnership(of: row.realm, by: actor, action: "rename")
        try execBind("UPDATE collections SET name = ?2 WHERE id = ?1", [.text(id), .text(name)])
    }

    public func retireCollection(id: String, by actor: CatalogRealm?) throws {
        guard let row = try collectionRow(id) else { throw CatalogError.noSuchCollection(id) }
        try requireOwnership(of: row.realm, by: actor, action: "retire")
        try execBind("DELETE FROM asset_collections WHERE collection_id = ?1", [.text(id)])
        try execBind("DELETE FROM collections WHERE id = ?1", [.text(id)])
    }

    /// Manual filing. Wins over derived filing and survives re-ingest.
    public func file(assetID: String, into collectionID: String, by actor: CatalogRealm?) throws {
        guard let row = try collectionRow(collectionID) else {
            throw CatalogError.noSuchCollection(collectionID)
        }
        // The asset has to EXIST. `asset_collections` has no foreign keys, so a
        // filing of an unknown id inserts happily and leaves a membership row
        // pointing at nothing — which inflates every collection count and, for
        // the unscoped POST /v1/catalog/file, is reachable by typing an id.
        // Checked here rather than at the route because `asset_collections` is
        // this store's table and the invariant is the store's to keep.
        guard try assetExists(assetID) else { throw CatalogError.noSuchAsset(assetID) }
        // A shared asset can never enter a kira collection. This is a property of
        // the DATA, not of who is asking, so it holds for the nil actor — the
        // service, backfill, the desktop app — exactly as applyDerivedFiling
        // already enforces it. Previously only a non-nil actor was checked, which
        // left `file(assetID: <shared>, into: <kira collection>, by: nil)` open.
        let owner = try assetRealm(assetID)
        if owner == .shared, row.realm == .kira {
            throw CatalogError.notPermitted("a shared asset may not be filed into a kira collection")
        }
        if let actor, owner != actor {
            throw CatalogError.notPermitted("\(actor.rawValue) may not file an asset outside its realm")
        }
        // Contributing to a SHARED collection is allowed; restructuring it is not.
        if row.realm != nil { try requireOwnership(of: row.realm, by: actor, action: "file into") }
        try execBind("""
            INSERT INTO asset_collections (asset_id, collection_id, manual) VALUES (?1, ?2, 1)
            ON CONFLICT(asset_id, collection_id) DO UPDATE SET manual = 1
            """,
            [.text(assetID), .text(collectionID)])
    }

    /// nil actor = the service itself (backfill, desktop). A realm-scoped actor
    /// may only touch collections of its own realm.
    private func requireOwnership(of target: CatalogRealm?, by actor: CatalogRealm?,
                                  action: String) throws {
        guard let actor else { return }
        guard target == actor else {
            throw CatalogError.notPermitted("\(actor.rawValue) may not \(action) a \(target?.rawValue ?? "shared") collection")
        }
    }

    private func collectionRow(_ id: String) throws -> CatalogCollection? {
        var stmt: OpaquePointer?
        let sql = "SELECT id, slug, name, parent_id, realm, description FROM collections WHERE id = ?1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return CatalogCollection(
            id: text(stmt, 0) ?? "", slug: text(stmt, 1) ?? "", name: text(stmt, 2) ?? "",
            parentID: text(stmt, 3),
            realm: text(stmt, 4).flatMap { CatalogRealm(rawValue: $0) },
            description: text(stmt, 5))
    }

    private func collectionRealm(_ id: String) throws -> CatalogRealm? {
        try collectionRow(id)?.realm
    }

    /// Whether a row with this id exists at all. Distinct from `assetRealm`,
    /// which answers nil both for "no such asset" and for "realm is NULL".
    private func assetExists(_ id: String) throws -> Bool {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT 1 FROM assets WHERE id = ?1", -1, &stmt, nil) == SQLITE_OK
        else { throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db))) }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    private func assetRealm(_ id: String) throws -> CatalogRealm? {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT realm FROM assets WHERE id = ?1", -1, &stmt, nil) == SQLITE_OK
        else { throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db))) }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return text(stmt, 0).flatMap { CatalogRealm(rawValue: $0) }
    }

    // MARK: - Edges and locations

    public func addEdge(_ e: AssetEdge) throws {
        try execBind("""
            INSERT OR IGNORE INTO asset_edges (from_asset_id, to_asset_id, relation)
            VALUES (?1, ?2, ?3)
            """,
            [.text(e.fromAssetID), .text(e.toAssetID), .text(e.relation.rawValue)])
    }

    /// Every edge touching this asset, in either direction — so a still finds
    /// its clips and a clip finds its still with one call.
    ///
    /// `scope` carries the same realm lock `search` does, and for the same
    /// reason: without it a confined caller could take an id her scoped search
    /// legitimately returned, follow its i2v_source edge to a SHARED still, and
    /// read an id she was never allowed to see. BOTH endpoints must be in scope,
    /// so the graph cannot be walked out of the realm in either direction.
    ///
    /// `scope` has NO DEFAULT, deliberately. It used to default to nil, and the
    /// first route written against it simply omitted the argument — which
    /// compiled, and leaked. Backfill's `scope: nil` is correct and now says so
    /// at the call site, where a reviewer can see it.
    public func edges(for assetID: String, scope: CatalogRealm?) throws -> [AssetEdge] {
        var stmt: OpaquePointer?
        let sql = """
            SELECT e.from_asset_id, e.to_asset_id, e.relation FROM asset_edges e
            WHERE (e.from_asset_id = ?1 OR e.to_asset_id = ?1)
              AND (?2 IS NULL OR (
                    EXISTS (SELECT 1 FROM assets a WHERE a.id = e.from_asset_id AND a.realm = ?2)
                AND EXISTS (SELECT 1 FROM assets a WHERE a.id = e.to_asset_id   AND a.realm = ?2)))
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, assetID)
        bindText(stmt, 2, scope?.rawValue)
        var out: [AssetEdge] = []
        // Read-only listing (#357): a step failure logs and returns whatever
        // edges were already collected rather than throwing.
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            guard let r = text(stmt, 2).flatMap({ AssetRelation(rawValue: $0) }) else { return }
            out.append(AssetEdge(fromAssetID: text(stmt, 0) ?? "",
                                 toAssetID: text(stmt, 1) ?? "", relation: r))
        }
        if rc != SQLITE_DONE {
            FileHandle.standardError.write(
                Data("CatalogStore.edges: step failed (code \(rc)): \(String(cString: sqlite3_errmsg(db)))\n".utf8))
        }
        return out
    }

    public func addLocation(assetID: String, _ loc: AssetLocation) throws {
        try execBind("""
            INSERT OR REPLACE INTO asset_locations (asset_id, host, path, mtime)
            VALUES (?1, ?2, ?3, ?4)
            """,
            [.text(assetID), .text(loc.host), .text(loc.path),
             .double(loc.mtime.timeIntervalSince1970)])
    }

    /// Locations are real on-disk paths on real hosts — the most valuable thing
    /// in the schema to leak — so this takes the realm lock too. Out of scope
    /// returns empty rather than throwing: a confined caller learns nothing about
    /// whether the id exists.
    ///
    /// No default on `scope`, for the reason given on `edges(for:scope:)`.
    public func locations(of assetID: String, scope: CatalogRealm?) throws -> [AssetLocation] {
        var stmt: OpaquePointer?
        let sql = """
            SELECT l.host, l.path, l.mtime FROM asset_locations l
            WHERE l.asset_id = ?1
              AND (?2 IS NULL
                   OR EXISTS (SELECT 1 FROM assets a WHERE a.id = l.asset_id AND a.realm = ?2))
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, assetID)
        bindText(stmt, 2, scope?.rawValue)
        var out: [AssetLocation] = []
        // Read-only listing (#357): a step failure logs and returns whatever
        // locations were already collected rather than throwing.
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            out.append(AssetLocation(host: text(stmt, 0) ?? "", path: text(stmt, 1) ?? "",
                                     mtime: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))))
        }
        if rc != SQLITE_DONE {
            FileHandle.standardError.write(
                Data("CatalogStore.locations: step failed (code \(rc)): \(String(cString: sqlite3_errmsg(db)))\n".utf8))
        }
        return out
    }

    /// Asset id for a known file path — used by backfill to resolve
    /// `source_image` into an `i2v_source` edge.
    ///
    /// Scoped for the same reason as `locations`, with one extra edge: unscoped,
    /// this is an oracle. A confined caller who GUESSES a path gets back a real
    /// id when the guess is right, which both confirms the file exists and hands
    /// her a key for `edges`/`locations`. Out of scope is nil.
    ///
    /// `scope` has NO DEFAULT, for the reason given on `edges(for:scope:)`: a
    /// defaulted realm parameter is one a new call site can omit, and omitting
    /// it compiles and leaks. Backfill's `scope: nil` is correct and now says so
    /// where a reviewer can see it.
    public func assetID(forPath path: String, scope: CatalogRealm?) throws -> String? {
        var stmt: OpaquePointer?
        let sql = """
            SELECT id FROM assets WHERE absolute_path = ?1 AND (?2 IS NULL OR realm = ?2)
            UNION
            SELECT l.asset_id FROM asset_locations l
              JOIN assets a ON a.id = l.asset_id
             WHERE l.path = ?1 AND (?2 IS NULL OR a.realm = ?2)
            LIMIT 1
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, path)
        bindText(stmt, 2, scope?.rawValue)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return text(stmt, 0)
    }

    /// Same oracle problem as `assetID(forPath:)` — a hash is guessable when the
    /// file is one the caller already has a copy of.
    ///
    /// `scope` has NO DEFAULT, for the reason given on `edges(for:scope:)`: a
    /// defaulted realm parameter is one a new call site can omit, and omitting
    /// it compiles and leaks. Backfill's `scope: nil` is correct and now says so
    /// where a reviewer can see it.
    public func assetID(forSHA256 sha: String, scope: CatalogRealm?) throws -> String? {
        var stmt: OpaquePointer?
        let sql = "SELECT id FROM assets WHERE sha256 = ?1 AND (?2 IS NULL OR realm = ?2) LIMIT 1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK
        else { throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db))) }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, sha)
        bindText(stmt, 2, scope?.rawValue)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return text(stmt, 0)
    }

    /// Settle an asset's realm once a twin is found in a realm-bearing tree.
    /// Backfill-only: realm is otherwise stamped by the caller at render time.
    public func setRealm(_ realm: CatalogRealm, forAssetID id: String) throws {
        try execBind("UPDATE assets SET realm = ?2 WHERE id = ?1",
                     [.text(id), .text(realm.rawValue)])
        // Filing depends on realm, so re-derive it. Fetch the row by id —
        // a limit-1 search would almost never contain it.
        if let row = try asset(id: id) {
            try applyDerivedFiling(row, explicitCollectionIDs: [])
        }
    }

    /// Fetch one row by id, unscoped and unclamped.
    ///
    /// INTERNAL on purpose. This is backfill's helper and it obeys neither of
    /// the store's two rules; a doc comment is not enough to keep it out of a
    /// future route, so the module boundary does it instead. Service callers
    /// want `asset(id:visibleTo:ceiling:)` or `search`.
    func asset(id: String) throws -> CatalogAsset? {
        let sql = """
            SELECT a.id, a.kind, a.filename, a.absolute_path, a.sha256, a.file_size,
                   a.width, a.height, a.created_at, a.realm, a.source, a.sealed,
                   a.prompt, a.negative_prompt, a.prompt_raw, a.caption, a.caption_source,
                   a.seed, a.steps, a.guidance, a.model_family, a.preset, a.loras,
                   a.render_id, a.content_mode, a.character_name,
                   a.lane, a.arc, a.theme, a.stock, a.genre, a.family, a.style,
                   a.mode, a.duration_ms, a.fps, a.frames, a.resolution, a.aspect_ratio,
                   a.rating, a.favorite, a.prompt_injected
            FROM assets a WHERE a.id = ?1
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return rowToAsset(stmt)
    }

    /// Fetch one row by id for a SERVICE caller: BOTH rules apply, exactly as
    /// they do in `search`.
    ///
    /// The realm lock is why this exists rather than the route calling
    /// `asset(id:)` and checking `realm` itself — and the clamp is why the route
    /// cannot call `asset(id:)` at all. `asset(id:)` is unclamped by design
    /// (backfill needs the true row), so a detail route built on it would hand
    /// back the prompt and the on-disk path of a row that the very same
    /// caller's `search` had just withheld. One row by id is not a way around
    /// the ceiling.
    ///
    /// Out of realm returns nil, indistinguishable from "no such id" — the
    /// caller cannot tell absence from exclusion, which is the point.
    public func asset(id: String, visibleTo scope: CatalogRealm?,
                      ceiling: String?) throws -> CatalogAsset? {
        guard let row = try asset(id: id) else { return nil }
        if let scope, row.realm != scope { return nil }
        return clamp(row, to: ceiling)
    }

    /// The asset that OWNS this path — the one whose `absolute_path` it is.
    ///
    /// Deliberately NOT `assetID(forPath:)`. That one is a UNION over
    /// `absolute_path` and `asset_locations.path`; a compound UNION dedups
    /// through a temporary b-tree, so under `LIMIT 1` the row it hands back is
    /// not guaranteed to be the path's owner — with rows ('zzz','/P') and
    /// ('aaa','/Q') and a stale location ('aaa','/P') it returns 'aaa'.
    /// Backfill needs the owner EXACTLY: writing a second row with an existing
    /// `absolute_path` violates NOT NULL UNIQUE, and `ON CONFLICT(id)` does not
    /// absorb it, so the throw aborts the whole sweep.
    ///
    /// DELIBERATELY UNSCOPED, and the only lookup here that is. This answers a
    /// WRITE-path question — "does a row already own this path?" — whose true
    /// answer does not depend on who is asking. A scoped variant would return
    /// nil for an owner in the other realm, the caller would mint a new id, and
    /// the UNIQUE collision (and the aborted sweep) would be straight back. A
    /// caller that wants the anti-oracle behaviour wants
    /// `assetID(forPath:scope:)`, which is a different question.
    public func assetID(owningPath path: String) throws -> String? {
        var stmt: OpaquePointer?
        let sql = "SELECT id FROM assets WHERE absolute_path = ?1"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, path)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return text(stmt, 0)
    }

    /// Ids of the assets with this exact filename, capped at `limit`.
    ///
    /// Backfill's LAST-RESORT way to resolve a `source_image` recorded on a host
    /// whose paths it cannot translate. The cap defaults to 2 because the only
    /// question asked is "is this basename unambiguous?" — two hits is already
    /// the answer, and backfill skips rather than guesses.
    ///
    /// `scope` carries the same realm lock every other lookup here does. A
    /// filename is the most guessable key in the schema, so unscoped this would
    /// be the plainest oracle of the lot — and a confined caller resolving a
    /// basename across the boundary could mint an edge into a realm she cannot
    /// see.
    ///
    /// `scope` has NO DEFAULT, for the reason given on `edges(for:scope:)`: a
    /// defaulted realm parameter is one a new call site can omit, and omitting
    /// it compiles and leaks. Backfill's `scope: nil` is correct and now says so
    /// where a reviewer can see it.
    public func assetIDs(forFilename name: String, limit: Int = 2,
                         scope: CatalogRealm?) throws -> [String] {
        var stmt: OpaquePointer?
        let sql = """
            SELECT id FROM assets WHERE filename = ?1 AND (?3 IS NULL OR realm = ?3) LIMIT ?2
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, name)
        sqlite3_bind_int64(stmt, 2, Int64(limit))
        bindText(stmt, 3, scope?.rawValue)
        var out: [String] = []
        // THROWS on a non-DONE step (#357 R1 review correction — this was
        // classified log-and-partial, which is wrong). `CatalogBackfill
        // .resolveSource` decides "is this filename unambiguous?" from
        // `ids.count == 1`: a truncated drain that drops a second real match
        // makes an AMBIGUOUS filename look UNIQUE and creates a WRONG i2v
        // edge — the opposite of, and worse than, the missing edge a merely
        // partial result would cause. Only a caller that gets the true count
        // (or knows it could not) can tell "one match" from "one match found
        // before the step broke", so this cannot be log-and-partial.
        // `resolveSource` treats the thrown error the same as an ambiguous
        // filename: skip the edge rather than risk a wrong one.
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            if let id = text(stmt, 0) { out.append(id) }
        }
        guard rc == SQLITE_DONE else {
            throw CatalogError.stepFailed(String(cString: sqlite3_errmsg(db)))
        }
        return out
    }

    /// How many assets ended up in NO collection at all. Backfill reports this
    /// so a coverage gap is a number in the log rather than an absence nobody
    /// notices — most of the fleet has no `lane` in its sidecar, and lane is
    /// what most filing rules key on.
    public func unfiledAssetCount() throws -> Int {
        var stmt: OpaquePointer?
        let sql = """
            SELECT COUNT(*) FROM assets a
            WHERE NOT EXISTS (SELECT 1 FROM asset_collections ac WHERE ac.asset_id = a.id)
            """
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    /// Assets per collection counted the way `search(collectionID:)` returns
    /// them: DISTINCT, and including the collection's direct children.
    ///
    /// NOT derivable from `CatalogFacets.collection`, which holds DIRECT filings
    /// only. Summing a parent's facet with its children's double-counts every
    /// asset filed in both — in the live catalog all 44 `col-decoupage` assets
    /// are filed in the parent AND in `col-kira-decoupage`, so the sum says 88
    /// and opening it shows 44.
    public func collectionCounts(scope: CatalogRealm?) throws -> [String: Int] {
        let sql = """
            SELECT c.id, COUNT(DISTINCT ac.asset_id)
            FROM collections c
            LEFT JOIN asset_collections ac
              ON ac.collection_id = c.id
                 OR ac.collection_id IN (SELECT id FROM collections WHERE parent_id = c.id)
            LEFT JOIN assets a ON a.id = ac.asset_id
            WHERE ?1 IS NULL OR a.realm = ?1 OR ac.asset_id IS NULL
            GROUP BY c.id
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, scope?.rawValue)
        var out: [String: Int] = [:]
        // Read-only display counts (#357): a step failure logs and returns
        // whatever counts were already collected rather than throwing.
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            guard let k = text(stmt, 0) else { return }
            out[k] = Int(sqlite3_column_int(stmt, 1))
        }
        if rc != SQLITE_DONE {
            FileHandle.standardError.write(
                Data("CatalogStore.collectionCounts: step failed (code \(rc)): \(String(cString: sqlite3_errmsg(db)))\n".utf8))
        }
        return out
    }

    // MARK: - Facets

    public func facets(scope: CatalogRealm?) throws -> CatalogFacets {
        var f = CatalogFacets()
        func count(_ column: String, into keyPath: WritableKeyPath<CatalogFacets, [String: Int]>) throws {
            let sql = """
                SELECT \(column), COUNT(*) FROM assets
                WHERE \(column) IS NOT NULL AND (?1 IS NULL OR realm = ?1)
                GROUP BY \(column)
                """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }
            bindText(stmt, 1, scope?.rawValue)
            // Read-only facet counts feeding filter-UI chips (#357): a step
            // failure logs and returns whatever counts were already collected
            // rather than throwing — an incomplete facet list under-populates a
            // filter dropdown, it does not corrupt anything.
            let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
                guard let k = text(stmt, 0) else { return }
                f[keyPath: keyPath][k] = Int(sqlite3_column_int(stmt, 1))
            }
            if rc != SQLITE_DONE {
                FileHandle.standardError.write(
                    Data("CatalogStore.facets(\(column)): step failed (code \(rc)): \(String(cString: sqlite3_errmsg(db)))\n".utf8))
            }
        }
        try count("lane", into: \.lane)
        try count("content_mode", into: \.tier)
        try count("character_name", into: \.character)
        try count("source", into: \.source)
        try count("stock", into: \.stock)
        try count("genre", into: \.genre)
        try count("kind", into: \.kind)
        try count("mode", into: \.mode)

        // Collections need the join.
        let sql = """
            SELECT ac.collection_id, COUNT(*) FROM asset_collections ac
            JOIN assets a ON a.id = ac.asset_id
            WHERE ?1 IS NULL OR a.realm = ?1
            GROUP BY ac.collection_id
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, scope?.rawValue)
        // Read-only, same log-and-partial treatment as the facet loop above (#357).
        let rc = drainSQLiteRows(step: { sqlite3_step(stmt) }) {
            guard let k = text(stmt, 0) else { return }
            f.collection[k] = Int(sqlite3_column_int(stmt, 1))
        }
        if rc != SQLITE_DONE {
            FileHandle.standardError.write(
                Data("CatalogStore.facets(collection): step failed (code \(rc)): \(String(cString: sqlite3_errmsg(db)))\n".utf8))
        }
        return f
    }

    // MARK: - Row and bind helpers

    private func rowToAsset(_ s: OpaquePointer?) -> CatalogAsset {
        CatalogAsset(
            id: text(s, 0) ?? "", kind: text(s, 1) ?? "image",
            filename: text(s, 2) ?? "", absolutePath: text(s, 3) ?? "",
            sha256: text(s, 4), fileSize: sqlite3_column_int64(s, 5),
            width: optInt(s, 6), height: optInt(s, 7),
            createdAt: Date(timeIntervalSince1970: sqlite3_column_double(s, 8)),
            realm: CatalogRealm(rawValue: text(s, 9) ?? "shared") ?? .shared,
            source: text(s, 10), sealed: sqlite3_column_int(s, 11) != 0,
            prompt: text(s, 12), negativePrompt: text(s, 13), promptRaw: text(s, 14),
            promptInjected: text(s, 41),
            caption: text(s, 15), captionSource: text(s, 16),
            seed: optInt(s, 17), steps: optInt(s, 18), guidance: optDouble(s, 19),
            modelFamily: text(s, 20), preset: text(s, 21), loras: text(s, 22),
            renderID: text(s, 23), contentMode: text(s, 24), characterName: text(s, 25),
            lane: text(s, 26), arc: text(s, 27), theme: text(s, 28), stock: text(s, 29),
            genre: text(s, 30), family: text(s, 31), style: text(s, 32),
            mode: text(s, 33), durationMs: optInt(s, 34), fps: optDouble(s, 35),
            frames: optInt(s, 36), resolution: text(s, 37), aspectRatio: text(s, 38),
            rating: Int(sqlite3_column_int(s, 39)), favorite: sqlite3_column_int(s, 40) != 0)
    }

    /// Prepare, bind `values` to parameters 1...n positionally, step to
    /// completion.
    private func execBind(_ sql: String, _ values: [CatalogBind]) throws {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CatalogError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        for (i, v) in values.enumerated() { v.apply(to: stmt, at: Int32(i + 1)) }
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw CatalogError.stepFailed(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func bindText(_ s: OpaquePointer?, _ i: Int32, _ v: String?) {
        if let v { sqlite3_bind_text(s, i, (v as NSString).utf8String, -1, CatalogSchema.transient) }
        else { sqlite3_bind_null(s, i) }
    }
    private func bindInt(_ s: OpaquePointer?, _ i: Int32, _ v: Int?) {
        if let v { sqlite3_bind_int64(s, i, Int64(v)) } else { sqlite3_bind_null(s, i) }
    }
    private func bindDouble(_ s: OpaquePointer?, _ i: Int32, _ v: Double?) {
        if let v { sqlite3_bind_double(s, i, v) } else { sqlite3_bind_null(s, i) }
    }
    private func text(_ s: OpaquePointer?, _ i: Int32) -> String? {
        guard let c = sqlite3_column_text(s, i) else { return nil }
        return String(cString: c)
    }
    private func optInt(_ s: OpaquePointer?, _ i: Int32) -> Int? {
        sqlite3_column_type(s, i) == SQLITE_NULL ? nil : Int(sqlite3_column_int64(s, i))
    }
    private func optDouble(_ s: OpaquePointer?, _ i: Int32) -> Double? {
        sqlite3_column_type(s, i) == SQLITE_NULL ? nil : sqlite3_column_double(s, i)
    }
}
