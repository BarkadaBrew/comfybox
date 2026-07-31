// DAMCatalogColumnsTests.swift — the DAM must not blank the catalog's columns.
//
// The catalog migrated into ~/.comfybox/dam.sqlite3, so `assets` now carries 45
// columns of which DAMStore names 23. Every DAM write therefore has to be an
// UPDATE of the columns it owns, never a whole-row replacement — and the two
// columns that matter most are the two the DAM has no idea exist: `realm` (the
// isolation domain) and `sealed` (the row whose text must never reach FTS).

import ComfyBoxCatalog
import Foundation
import SQLite3
import Testing

@testable import ComfyBoxDesktop

@Suite("DAM writes preserve catalog columns")
struct DAMCatalogColumnsTests {

    /// Read one TEXT/INTEGER column straight out of the file, so the assertion
    /// does not depend on either store's idea of what a row contains.
    private func rawColumn(_ column: String, ofAssetID id: String, in dbPath: String) -> String? {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbPath, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_close(db) }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT \(column) FROM assets WHERE id = ?1", -1, &stmt, nil)
            == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1,
                          unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        guard sqlite3_column_type(stmt, 0) != SQLITE_NULL else { return nil }
        return String(cString: sqlite3_column_text(stmt, 0))
    }

    /// Run one statement against the file. Used to seal the seeded row — the
    /// `CatalogAsset` initialiser drops text on a sealed row at construction, so
    /// a row that is BOTH sealed and carrying `prompt_raw` (the state a live
    /// pre-seal row is in) cannot be built through the model.
    private func exec(_ sql: String, in dbPath: String) {
        var db: OpaquePointer?
        guard sqlite3_open_v2(dbPath, &db, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else { return }
        defer { sqlite3_close(db) }
        sqlite3_exec(db, sql, nil, nil, nil)
    }

    /// ONE ♥ CLICK. `toggleFavorite` reads the row the gallery is showing, flips
    /// one Bool and writes it back through `insertAsset`. Under the old
    /// `INSERT OR REPLACE` that was DELETE-then-INSERT: 22 catalog columns went
    /// to their defaults, `realm` to NULL (promoted to 'shared' on the next
    /// open — a Kira-private asset silently leaving her realm) and `sealed` to 0
    /// (the row unseals, and the FTS write at the end of `insertAsset` then
    /// indexes its prompt, which backfill cannot undo).
    @Test("favouriting a catalog row does not blank realm, sealed, lane or prompt_raw")
    func favouriteDoesNotWipeCatalogColumns() async throws {
        let dbPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("dam-catalog-\(UUID().uuidString).sqlite3")
        defer {
            for suffix in ["", "-wal", "-shm"] {
                try? FileManager.default.removeItem(atPath: dbPath + suffix)
            }
        }
        let assetPath = "/tmp/kira-\(UUID().uuidString).png"

        // Seed a row the way the catalog does: full 45-column schema, Kira's
        // realm, a lane, and the raw prompt spelling the DAM has no column for.
        do {
            let catalog = try await CatalogStore.open(path: dbPath)
            try await catalog.upsert(
                CatalogAsset(id: "k-seal", filename: "kira.png", absolutePath: assetPath,
                             realm: .kira, prompt: "a nightclub", promptRaw: "RAW-INTENT-LINE",
                             contentMode: "avocado", lane: "kira"),
                explicitCollectionIDs: [])
        }
        exec("UPDATE assets SET sealed = 1 WHERE id = 'k-seal'", in: dbPath)

        #expect(rawColumn("realm", ofAssetID: "k-seal", in: dbPath) == "kira")
        #expect(rawColumn("sealed", ofAssetID: "k-seal", in: dbPath) == "1")

        // The ♥: read it through the DAM, flip favorite, write it back.
        let dam = try await DAMStore.open(path: dbPath)
        let row = try #require(try await dam.fetchAsset(byPath: assetPath))
        #expect(row.id == "k-seal")
        try await dam.insertAsset(
            DAMAsset(id: row.id, kind: row.kind, filename: row.filename,
                     absolutePath: row.absolutePath, fileSize: row.fileSize,
                     sha256: row.sha256, width: row.width, height: row.height,
                     createdAt: row.createdAt, modifiedAt: row.modifiedAt,
                     ingestedAt: row.ingestedAt, orphaned: row.orphaned,
                     prompt: row.prompt, negativePrompt: row.negativePrompt,
                     seed: row.seed, steps: row.steps, guidance: row.guidance,
                     modelFamily: row.modelFamily, rating: row.rating,
                     favorite: true,
                     contentMode: row.contentMode, characterName: row.characterName,
                     source: row.source))

        // The change the user asked for landed…
        let after = try #require(try await dam.fetchAsset(byPath: assetPath))
        #expect(after.favorite)
        // …and nothing else moved.
        #expect(rawColumn("realm", ofAssetID: "k-seal", in: dbPath) == "kira",
                "realm was blanked — the row leaves Kira's realm on the next open")
        #expect(rawColumn("sealed", ofAssetID: "k-seal", in: dbPath) == "1",
                "the row unsealed — its prompt is now indexable")
        #expect(rawColumn("lane", ofAssetID: "k-seal", in: dbPath) == "kira")
        #expect(rawColumn("prompt_raw", ofAssetID: "k-seal", in: dbPath) == "RAW-INTENT-LINE")
    }

    /// The consequence the row-wipe had that backfill could never repair: a
    /// sealed row whose prompt reaches FTS is searchable by its own prompt
    /// forever after.
    @Test("a DAM write never puts a sealed row's text into FTS")
    func sealedRowStaysOutOfFTS() async throws {
        let dbPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("dam-fts-\(UUID().uuidString).sqlite3")
        defer {
            for suffix in ["", "-wal", "-shm"] {
                try? FileManager.default.removeItem(atPath: dbPath + suffix)
            }
        }
        let assetPath = "/tmp/sealed-\(UUID().uuidString).png"
        do {
            let catalog = try await CatalogStore.open(path: dbPath)
            try await catalog.upsert(
                CatalogAsset(id: "s-seal", filename: "s.png", absolutePath: assetPath,
                             realm: .kira, prompt: "unmistakablephrase",
                             contentMode: "avocado"),
                explicitCollectionIDs: [])
        }
        // The live shape of a sealed row: the flag set, the prompt column still
        // populated, and NO FTS entry. (Seeding it unsealed is the only way to
        // get text into the column at all — `CatalogAsset` drops text on a
        // sealed row at construction — so the index entry that seeding created
        // is cleared here, leaving the DAM write below as the only thing that
        // could put it back.)
        exec("UPDATE assets SET sealed = 1 WHERE id = 's-seal'", in: dbPath)
        exec("DELETE FROM assets_fts WHERE id = 's-seal'", in: dbPath)

        let dam = try await DAMStore.open(path: dbPath)
        let row = try #require(try await dam.fetchAsset(byPath: assetPath))
        try await dam.insertAsset(
            DAMAsset(id: row.id, kind: row.kind, filename: row.filename,
                     absolutePath: row.absolutePath, fileSize: row.fileSize,
                     createdAt: row.createdAt, modifiedAt: row.modifiedAt,
                     ingestedAt: row.ingestedAt, prompt: row.prompt,
                     rating: 5, favorite: false))

        #expect(rawColumn("sealed", ofAssetID: "s-seal", in: dbPath) == "1")
        let hits = try await dam.searchPrompts(query: "unmistakablephrase")
        #expect(hits.isEmpty, "a sealed row became reachable by its own prompt")
    }

    /// A hyphen in the desktop search box used to be an FTS5 syntax error, which
    /// `sqlite3_step` reports as a failed step and the gallery renders as "no
    /// results" — a search that silently answers nothing.
    @Test("desktop search survives FTS5 punctuation")
    func searchSanitisesPunctuation() async throws {
        let dbPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("dam-punct-\(UUID().uuidString).sqlite3")
        defer { try? FileManager.default.removeItem(atPath: dbPath) }
        let dam = try await DAMStore.open(path: dbPath)
        try await dam.insertAsset(DAMAsset(
            id: "p1", filename: "p.png", absolutePath: "/tmp/p1-\(UUID().uuidString).png",
            prompt: "a sun-drenched terrace"))

        #expect(try await dam.searchPrompts(query: "sun-drenched").count == 1)
        #expect(try await dam.searchPrompts(query: "terrace \"").count == 1)
        #expect(try await dam.searchPrompts(query: "   ").isEmpty)
    }
}
