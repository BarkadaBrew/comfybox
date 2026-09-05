// CatalogBrowser.swift — one gallery reader over the whole catalog.
//
// Replaces two disagreeing readers: GalleryView's local DAMStore fetch (Mac
// files only, no realm / lane / collection facets) and RemoteGalleryService's
// /v1/gallery/list (a bare directory listing with no metadata at all). Those
// two ARE the "Mac gallery and server gallery are different" problem, living
// inside one app.
//
// Todd's surface is deliberately UNSCOPED — he sees both realms. Only Kira's
// MCP tool carries a realm lock, and the store enforces it there. `scope` is
// therefore OVERWRITTEN here rather than merely defaulted: this class is the
// owner's browser by construction, so no caller can narrow it by accident.
//
// DAMStore is not replaced. AssetIngestor still writes through it, and ratings,
// favourites and the secure-vault moves still go there. Only browsing changes.
//
// IDENTITY. The catalog and the DAM share ONE `assets` table — dam.sqlite3 was
// migrated in place — so a catalog id IS a DAM id. Everything keyed by that id
// (ratings, favourites, folder filing, the secured set) keeps matching, and the
// display model below preserves it exactly.
//
// THE GATE. This is a data layer and holds no opinion about what is on screen.
// The app is Rated G until AppContentGate is revealed; the caller is responsible
// for keeping every surface that renders these rows behind that gate. What this
// class DOES own is `hiddenAssetIDs` — the vault carve-out, below.

import Foundation
import ComfyBoxCatalog

@Observable
@MainActor
public final class CatalogBrowser {

    /// The host name the backfill stamps on THIS Mac's copies of an asset.
    ///
    /// Kept in lockstep with the `home` tree in ComfyBoxGallery's backfill CLI
    /// (`GalleryServer.swift`, `host: "mac"`). A row with no `mac` location has
    /// never been copied here, so its bytes have to stream.
    public static let localHost = "mac"

    private let store: CatalogStore
    private let engineBaseURL: String

    public private(set) var items: [CatalogAsset] = []
    public private(set) var collections: [CatalogCollection] = []
    public private(set) var facets = CatalogFacets()
    /// Distinct per-collection counts (collection + its children), as the rail
    /// labels them. See `count(of:)`.
    public private(set) var collectionCounts: [String: Int] = [:]
    public private(set) var isLoading = false
    /// The filter the current page was actually produced by (post-override).
    public private(set) var activeFilter = CatalogQuery()
    public var error: String?

    /// How many rows one page holds. `load()` and the rail convenience methods
    /// use it; `apply(filter:)` honours whatever limit the caller set, so a
    /// deliberate small page stays small.
    public var pageSize = 500

    /// Assets the LOCAL library says are secured — moved into the vault at
    /// ~/.comfybox/secure.noindex and recorded in DAMStore's `secured_assets`.
    ///
    /// The catalog shares that database file but knows nothing about that table,
    /// and a secured row very often still has a streamable twin on a server. So
    /// without this, converging the gallery onto the catalog would quietly undo
    /// every vault move Todd has ever made. Set it before loading.
    public var hiddenAssetIDs: Set<String> = []

    /// Rows whose bytes are readable on this Mac: asset id -> the path to open.
    private var localPaths: [String: String] = [:]
    /// Rows with no copy here: asset id -> the path to ask the engine for.
    private var remotePaths: [String: String] = [:]

    public init(store: CatalogStore, engineBaseURL: String = "http://127.0.0.1:7870") {
        self.store = store
        self.engineBaseURL = engineBaseURL
    }

    // MARK: - Loading

    public func load() async {
        await apply(filter: CatalogQuery(limit: pageSize))
    }

    /// Run `filter` over the whole catalog and refresh the rail with it.
    ///
    /// `scope` is forced to nil (see the file header). `ceiling` is left exactly
    /// as the caller set it — the desktop passes nil, because on the owner's own
    /// machine it is the CONTENT GATE, not the chat-mode clamp, that decides what
    /// is on screen.
    public func apply(filter: CatalogQuery) async {
        isLoading = true
        defer { isLoading = false }
        error = nil
        var q = filter
        q.scope = nil
        activeFilter = q
        do {
            let rows = try await store.search(q)
            collections = try await store.collections(visibleTo: nil)
            facets = try await store.facets(scope: nil)
            collectionCounts = try await store.collectionCounts(scope: nil)
            await resolve(rows)
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Narrow to a collection (nil = the whole catalog again).
    public func apply(collectionID: String?) async {
        await apply(filter: CatalogQuery(collectionID: collectionID, limit: pageSize))
    }

    /// Narrow to a lane (nil = the whole catalog again).
    public func apply(lane: String?) async {
        await apply(filter: CatalogQuery(lane: lane, limit: pageSize))
    }

    /// Re-run the filter the current page was produced by — after an ingest, a
    /// delete, or a change to `hiddenAssetIDs`.
    public func reload() async {
        await apply(filter: activeFilter)
    }

    /// `allMatchingIDs()`'s answer: ids AND whether `CatalogStore.idsHardCap`
    /// cut them off. Mirrors `CatalogStore.AssetIDsResult` one layer up, after
    /// `hiddenAssetIDs` has been subtracted.
    public struct SelectAllResult: Sendable, Equatable {
        public let ids: Set<String>
        /// True when the filter matched more than `CatalogStore.idsHardCap`
        /// assets — `ids` is a PREFIX, not everything the filter matched, and
        /// the caller must tell the user rather than silently selecting less
        /// than "all" (R1 review correction: this used to be silent, which is
        /// the exact failure #271 exists to fix, one layer down).
        public let truncated: Bool
    }

    /// Every id matching the CURRENT filter, ignoring `pageSize`/paging
    /// entirely — what Select All actually needs (#271).
    ///
    /// Raising the page size and reloading (the old approach) fetches full
    /// `CatalogAsset` rows — prompts, captions, paths, every column — for
    /// however many thousand assets are in scope, just to read back their ids,
    /// AND kept re-truncating in practice at whatever page clamp sat in front
    /// of it. `CatalogStore.assetIDs(matching:)` is the dedicated ids-only
    /// query this calls instead: no page to clamp, and no row payload heavier
    /// than a `String` per asset.
    ///
    /// `hiddenAssetIDs` (the vault carve-out) is subtracted the same way
    /// `resolve(_:)` does for the visible page, so Select All cannot select a
    /// secured asset the grid never showed.
    public func allMatchingIDs() async -> SelectAllResult {
        do {
            let result = try await store.assetIDs(matching: activeFilter)
            let ids = Set(result.ids).subtracting(hiddenAssetIDs)
            return SelectAllResult(ids: ids, truncated: result.truncated)
        } catch {
            self.error = error.localizedDescription
            return SelectAllResult(ids: [], truncated: false)
        }
    }

    /// Classify one page ONCE. The grid asks "is this here or on the server?"
    /// for every visible cell; answering that per cell would be one actor
    /// round-trip and one stat(2) per cell per redraw.
    private func resolve(_ rows: [CatalogAsset]) async {
        let fm = FileManager.default
        var kept: [CatalogAsset] = []
        var local: [String: String] = [:]
        var remote: [String: String] = [:]

        for row in rows {
            guard !hiddenAssetIDs.contains(row.id) else { continue }
            let locations = (try? await store.locations(of: row.id, scope: nil)) ?? []
            // The row's own path first: it is the primary spelling and is what a
            // freshly ingested asset has before any location is recorded.
            let here = [row.absolutePath]
                + locations.filter { $0.host == Self.localHost }.map(\.path)
            if let path = here.first(where: { fm.fileExists(atPath: $0) }) {
                local[row.id] = path
            } else if let elsewhere = locations.first(where: { $0.host != Self.localHost })?.path {
                remote[row.id] = elsewhere
            } else {
                // Indexed, not here, and no other host claims it. Ask the engine
                // for the primary path anyway: on this Mac the engine and the
                // gallery share a filesystem, so a path the catalog knows and
                // `fileExists` denies is usually an unmounted share, not a lie.
                remote[row.id] = row.absolutePath
            }
            kept.append(row)
        }

        items = kept
        localPaths = local
        remotePaths = remote
    }

    // MARK: - Where an asset's bytes are

    /// The readable path on this Mac for a row on the CURRENT page, or nil when
    /// its bytes are only on a server.
    public func localPath(forID id: String) -> String? { localPaths[id] }

    /// The engine URL for a row on the CURRENT page whose bytes are not here.
    /// nil for a row that opens from disk — a local file is never streamed.
    public func resolvedStreamURL(forID id: String) -> URL? {
        guard let path = remotePaths[id] else { return nil }
        return streamURL(path: path)
    }

    /// True when no location for this asset exists on this Mac.
    ///
    /// Answered against the filesystem, not against the catalog's opinion: a
    /// `mac` location whose file has since been deleted must fall through to
    /// streaming rather than render as a broken local cell.
    public func isRemote(_ asset: CatalogAsset) async -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: asset.absolutePath) { return false }
        let locations = (try? await store.locations(of: asset.id, scope: nil)) ?? []
        return !locations
            .filter { $0.host == Self.localHost }
            .contains { fm.fileExists(atPath: $0.path) }
    }

    /// Stream URL for a server-side asset, via the engine route the spec
    /// deliberately left unchanged (`GET /v1/gallery/file?path=`).
    public func streamURL(for asset: CatalogAsset) async -> URL? {
        let locations = (try? await store.locations(of: asset.id, scope: nil)) ?? []
        let path = locations.first(where: { $0.host != Self.localHost })?.path ?? asset.absolutePath
        return streamURL(path: path)
    }

    private func streamURL(path: String) -> URL? {
        var components = URLComponents(string: engineBaseURL + "/v1/gallery/file")
        components?.queryItems = [URLQueryItem(name: "path", value: path)]
        return components?.url
    }

    // MARK: - Rendering

    /// The catalog row as the gallery's display model.
    ///
    /// Used only for rows the DAM half of the shared table has no richer record
    /// of; the id is preserved so everything keyed by it keeps matching. The
    /// three fields DAMAsset has and the catalog does not — modified, ingested,
    /// orphaned — fall back to the creation date rather than to `now`, so a
    /// server-only row does not present itself as freshly imported.
    public func damAsset(for asset: CatalogAsset) -> DAMAsset {
        DAMAsset(
            id: asset.id,
            kind: asset.kind,
            filename: asset.filename,
            absolutePath: asset.absolutePath,
            fileSize: asset.fileSize,
            sha256: asset.sha256,
            width: asset.width,
            height: asset.height,
            createdAt: asset.createdAt,
            modifiedAt: asset.createdAt,
            ingestedAt: asset.createdAt,
            orphaned: false,
            prompt: asset.prompt,
            negativePrompt: asset.negativePrompt,
            seed: asset.seed,
            steps: asset.steps,
            guidance: asset.guidance,
            modelFamily: asset.modelFamily,
            rating: asset.rating,
            favorite: asset.favorite,
            contentMode: asset.contentMode,
            characterName: asset.characterName,
            source: asset.source)
    }

    /// Root collections, then their children, with the counts the rail shows.
    public func rootCollections() -> [CatalogCollection] {
        collections.filter { $0.parentID == nil }
    }

    public func children(of root: CatalogCollection) -> [CatalogCollection] {
        collections.filter { $0.parentID == root.id }
    }

    /// A collection's count INCLUDING its children, matching what selecting it
    /// actually returns (`CatalogQuery.collectionID` matches a collection and its
    /// direct children). A parent labelled with only its own direct filings
    /// would read "Photography (0)" and then open 1,200 rows.
    ///
    /// Counted DISTINCTLY by the store, not summed from `facets.collection`.
    /// Those are direct filings, and an asset filed in both a parent and its
    /// child is counted twice by a sum — in the live catalog all 44
    /// `col-decoupage` assets are filed in both, so the sum said 88 and opening
    /// it showed 44, exactly the mismatch this count exists to prevent.
    public func count(of collection: CatalogCollection) -> Int {
        collectionCounts[collection.id] ?? 0
    }
}
