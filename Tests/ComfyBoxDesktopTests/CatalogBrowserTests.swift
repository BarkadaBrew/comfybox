// CatalogBrowserTests.swift — the one gallery reader, over the whole catalog.
//
// The fixtures write REAL files for the local row and none for the remote one,
// because "is this asset on this Mac" is answered against the filesystem, not
// against the catalog's opinion of it. A fixture that skipped the write would
// have made every row look remote and the local/remote split untestable.

import XCTest
import ComfyBoxCatalog
@testable import ComfyBoxDesktop

@MainActor
final class CatalogBrowserTests: XCTestCase {
    private var dir: String!
    private var path: String!
    private var localFile: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        dir = NSTemporaryDirectory() + "browser-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        path = (dir as NSString).appendingPathComponent("catalog.sqlite3")
        localFile = (dir as NSString).appendingPathComponent("l.png")
        FileManager.default.createFile(atPath: localFile, contents: Data([0x89, 0x50, 0x4E, 0x47]))

        store = try await CatalogStore.open(path: path)
        try await store.upsert(CatalogAsset(id: "local", filename: "l.png",
                                            absolutePath: localFile, realm: .shared,
                                            contentMode: "neutral"), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "local",
            AssetLocation(host: "mac", path: localFile, mtime: Date()))
        try await store.upsert(CatalogAsset(id: "remote", filename: "r.png",
                                            absolutePath: "/home/todd/.kira/studio/gallery/Kira/r.png",
                                            realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "remote",
            AssetLocation(host: "kira", path: "/home/todd/.kira/studio/gallery/Kira/r.png", mtime: Date()))
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: dir)
        try await super.tearDown()
    }

    func testDesktopSeesBothRealms() async throws {
        let b = CatalogBrowser(store: store)
        await b.load()
        XCTAssertEqual(Set(b.items.map(\.id)), ["local", "remote"],
                       "Todd's surface is unscoped — one gallery over everything")
    }

    func testARowOnlyOnTheServerIsMarkedRemoteAndGetsAStreamURL() async throws {
        let b = CatalogBrowser(store: store, engineBaseURL: "http://127.0.0.1:7870")
        await b.load()
        let remote = try XCTUnwrap(b.items.first { $0.id == "remote" })
        let local = try XCTUnwrap(b.items.first { $0.id == "local" })
        let isRemote = await b.isRemote(remote)
        let isLocal = await b.isRemote(local)
        XCTAssertTrue(isRemote)
        XCTAssertFalse(isLocal)
        let streamed = await b.streamURL(for: remote)
        let url = try XCTUnwrap(streamed)
        XCTAssertTrue(url.absoluteString.hasPrefix("http://127.0.0.1:7870/v1/gallery/file?path="),
                      "got \(url.absoluteString)")
    }

    /// The per-row `await` calls above are the answer to one question; the grid
    /// asks it 500 times per load, so the same answer is cached on the page.
    func testTheResolvedPageAgreesWithThePerRowAnswer() async throws {
        let b = CatalogBrowser(store: store, engineBaseURL: "http://127.0.0.1:7870")
        await b.load()
        XCTAssertEqual(b.localPath(forID: "local"), localFile)
        XCTAssertNil(b.localPath(forID: "remote"))
        let url = try XCTUnwrap(b.resolvedStreamURL(forID: "remote"))
        XCTAssertTrue(url.absoluteString.contains("gallery/Kira/r.png"))
        XCTAssertNil(b.resolvedStreamURL(forID: "local"),
                     "a row whose bytes are right here must not be streamed over HTTP")
    }

    func testCollectionFilterNarrowsTheGrid() async throws {
        let b = CatalogBrowser(store: store)
        await b.apply(filter: CatalogQuery(collectionID: "col-kira-autocord"))
        XCTAssertEqual(b.items.map(\.id), ["remote"])
    }

    func testFacetsPopulateForTheRail() async throws {
        let b = CatalogBrowser(store: store)
        await b.load()
        XCTAssertEqual(b.facets.lane["shoot"], 1)
        XCTAssertFalse(b.collections.isEmpty)
    }

    func testTheDesktopIsNotRealmScoped() async throws {
        let b = CatalogBrowser(store: store)
        await b.apply(filter: CatalogQuery(lane: "shoot"))
        XCTAssertEqual(b.items.first?.realm, .kira,
                       "a kira row must be reachable from Todd's surface")
    }

    /// A caller-supplied scope is OVERRIDDEN, not honoured. This class is the
    /// owner's browser by construction; if a future caller could narrow it, a
    /// stray `.kira` would silently empty half his gallery — and if it could
    /// widen someone else's, that would be the leak the store's lock exists to
    /// stop. Only this surface is unscoped, and only because it is his.
    func testACallerSuppliedScopeCannotNarrowTheOwnersSurface() async throws {
        let b = CatalogBrowser(store: store)
        await b.apply(filter: CatalogQuery(scope: .kira))
        XCTAssertEqual(Set(b.items.map(\.id)), ["local", "remote"])
        XCTAssertNil(b.activeFilter.scope)
    }

    // MARK: - The vault and the gate

    /// Securing an asset MOVES it into ~/.comfybox/secure.noindex and records the
    /// id in DAMStore's `secured_assets`. The catalog shares that database file
    /// but knows nothing about that table, and a secured row very often still has
    /// a streamable twin on a server — so without this filter, converging the
    /// gallery on the catalog would quietly undo every vault move Todd ever made.
    func testASecuredAssetIsWithheldEvenThoughTheCatalogStillHoldsItsRow() async throws {
        let b = CatalogBrowser(store: store)
        b.hiddenAssetIDs = ["remote"]
        await b.load()
        XCTAssertEqual(b.items.map(\.id), ["local"])
        XCTAssertNil(b.resolvedStreamURL(forID: "remote"),
                     "a withheld row must not keep a way to fetch its bytes")
    }

    /// The rail names bodies of work — "Adult", "Adult Scenes", "Erotic
    /// Portraiture", "Nightlife" are real collection names in the live catalog.
    /// The app is Rated G until revealed, so the rail is part of what the gate
    /// hides; a new reader must not become the way around it.
    func testTheCollectionRailIsHiddenWhileTheAppIsGRated() {
        XCTAssertFalse(GalleryView.showsCatalogRail(revealed: false, hasBrowser: true))
        XCTAssertFalse(GalleryView.showsCatalogRail(revealed: true, hasBrowser: false))
        XCTAssertTrue(GalleryView.showsCatalogRail(revealed: true, hasBrowser: true))
    }

    /// A fresh gate is hidden, so the rail is hidden on launch without anyone
    /// having to remember to hide it.
    func testAFreshLaunchHidesTheRail() {
        XCTAssertFalse(GalleryView.showsCatalogRail(revealed: AppContentGate().revealed,
                                                    hasBrowser: true))
    }

    // MARK: - Rendering

    /// The catalog and the DAM share ONE `assets` table (dam.sqlite3 was migrated
    /// in place), so ids are the same id space. The display model must preserve
    /// the id or ratings, favourites, folders and the secured set — all keyed by
    /// it — would stop matching the rows they belong to.
    func testTheDisplayModelKeepsTheCatalogId() async throws {
        let b = CatalogBrowser(store: store)
        await b.load()
        let row = try XCTUnwrap(b.items.first { $0.id == "remote" })
        XCTAssertEqual(b.damAsset(for: row).id, "remote")
        XCTAssertEqual(b.damAsset(for: row).filename, "r.png")
    }

    func testAPathWithASpaceStreamsAsAValidURL() async throws {
        try await store.upsert(CatalogAsset(id: "spacey", filename: "a b.png",
                                            absolutePath: "/home/todd/.kira/a b.png",
                                            realm: .kira), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "spacey",
            AssetLocation(host: "kira", path: "/home/todd/.kira/a b.png", mtime: Date()))
        let b = CatalogBrowser(store: store)
        await b.load()
        let url = try XCTUnwrap(b.resolvedStreamURL(forID: "spacey"))
        XCTAssertEqual(url.query(percentEncoded: false), "path=/home/todd/.kira/a b.png")
    }
}
