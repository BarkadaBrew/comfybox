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

    /// The rail's label must equal what clicking it opens. `facets.collection`
    /// holds DIRECT filings, so summing a parent with its children double-counts
    /// every asset filed in both — which, for decoupage, is all of them.
    func testARailCountEqualsWhatSelectingItOpens() async throws {
        // Kira's decoupage lane files into her genre AND the shared root.
        try await store.upsert(CatalogAsset(id: "tile", filename: "t.png",
                                            absolutePath: "/home/todd/.kira/t.png",
                                            realm: .kira, lane: "tile"), explicitCollectionIDs: [])
        let b = CatalogBrowser(store: store)
        await b.load()
        let root = try XCTUnwrap(b.collections.first { $0.id == "col-decoupage" })
        XCTAssertEqual(b.count(of: root), 1, "filed in parent AND child, but it is one asset")

        await b.apply(collectionID: "col-decoupage")
        XCTAssertEqual(b.items.count, b.count(of: root))
    }

    /// "Save to this Mac" writes into the output folder unattended; a server
    /// file sharing a basename with a local one must not clobber it.
    func testSavingARemoteFileNeverOverwritesALocalOne() throws {
        let existing = (dir as NSString).appendingPathComponent("l.png")
        XCTAssertTrue(FileManager.default.fileExists(atPath: existing))
        let first = GalleryView.uniqueDestination(inDirectory: dir, filename: "l.png")
        XCTAssertEqual((first as NSString).lastPathComponent, "l-1.png")

        FileManager.default.createFile(atPath: first, contents: Data([0x01]))
        let second = GalleryView.uniqueDestination(inDirectory: dir, filename: "l.png")
        XCTAssertEqual((second as NSString).lastPathComponent, "l-2.png")

        XCTAssertEqual((GalleryView.uniqueDestination(inDirectory: dir,
                                                      filename: "fresh.png") as NSString).lastPathComponent,
                       "fresh.png")
    }

    // MARK: - Select All (#271)

    /// The literal #271 acceptance test at the desktop's actual call site: a
    /// filter matching far more rows than the browser's own page size
    /// (`pageSize`, 500 by default) must still select every one of them.
    /// `allMatchingIDs` is what `GalleryView.selectAll()` calls instead of
    /// raising the page size and hoping the reload's limit does not get
    /// clamped again somewhere between here and the database.
    func testAllMatchingIDsSelectsAllOf1200SyntheticRowsBeyondThePageSize() async throws {
        for i in 0..<1200 {
            try await store.upsert(CatalogAsset(id: "select-all-\(i)", filename: "s\(i).png",
                                                absolutePath: "/tmp/s\(i).png", realm: .shared,
                                                lane: "select-all-fixture"),
                                   explicitCollectionIDs: [])
        }
        let b = CatalogBrowser(store: store)
        b.pageSize = 500   // the grid's normal page — deliberately smaller than the fixture
        await b.apply(filter: CatalogQuery(lane: "select-all-fixture", limit: b.pageSize))
        XCTAssertEqual(b.items.count, 500, "precondition: the loaded PAGE is still clamped")

        let allIDs = await b.allMatchingIDs()
        XCTAssertEqual(allIDs.count, 1200, "allMatchingIDs must not inherit the page's limit")
        XCTAssertEqual(allIDs, Set((0..<1200).map { "select-all-\($0)" }))
    }

    /// `allMatchingIDs` subtracts `hiddenAssetIDs` the same way the loaded
    /// page does (`resolve(_:)`) — Select All must not be a way to select a
    /// secured asset the grid never showed.
    func testAllMatchingIDsWithholdsHiddenAssetIDs() async throws {
        try await store.upsert(CatalogAsset(id: "visible", filename: "v.png",
                                            absolutePath: "/tmp/v.png", realm: .shared,
                                            lane: "vault-fixture"), explicitCollectionIDs: [])
        try await store.upsert(CatalogAsset(id: "vaulted", filename: "h.png",
                                            absolutePath: "/tmp/h.png", realm: .shared,
                                            lane: "vault-fixture"), explicitCollectionIDs: [])
        let b = CatalogBrowser(store: store)
        b.hiddenAssetIDs = ["vaulted"]
        await b.apply(filter: CatalogQuery(lane: "vault-fixture"))

        let allIDs = await b.allMatchingIDs()
        XCTAssertEqual(allIDs, ["visible"])
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
