import XCTest
@testable import ComfyBoxCatalog

final class CatalogStoreTests: XCTestCase {
    private var path: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        path = NSTemporaryDirectory() + "store-test-\(UUID().uuidString).sqlite3"
        store = try await CatalogStore.open(path: path)
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: path)
        try await super.tearDown()
    }

    private func make(_ id: String, realm: CatalogRealm, lane: String? = nil,
                      tier: String? = nil, kind: String = "image",
                      prompt: String? = nil, sealed: Bool = false,
                      collection: String? = nil, family: String? = nil,
                      mode: String? = nil, source: String? = nil) -> CatalogAsset {
        CatalogAsset(id: id, kind: kind, filename: "\(id).png",
                     absolutePath: "/tmp/\(id).png",
                     realm: realm, source: source, sealed: sealed,
                     prompt: prompt, contentMode: tier, lane: lane,
                     family: family, mode: mode)
    }

    // MARK: - Realm isolation (the invariant that matters most)

    func testKiraScopedSearchNeverReturnsASharedRow() async throws {
        try await store.upsert(make("k1", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared, lane: "shoot"), explicitCollectionIDs: [])

        // Every filter combination we expose must respect the lock.
        let variations: [CatalogQuery] = [
            CatalogQuery(scope: .kira),
            CatalogQuery(scope: .kira, lane: "shoot"),
            CatalogQuery(scope: .kira, collectionID: "col-photography"),
            CatalogQuery(scope: .kira, kind: "image"),
            CatalogQuery(scope: .kira, minRating: 0),
            CatalogQuery(scope: .kira, orderBy: .oldest),
        ]
        for q in variations {
            let rows = try await store.search(q)
            XCTAssertFalse(rows.contains { $0.realm != .kira },
                           "leaked a non-kira row for query \(q)")
        }
    }

    func testScopeNilSeesEverything() async throws {
        try await store.upsert(make("k1", realm: .kira), explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared), explicitCollectionIDs: [])
        let rows = try await store.search(CatalogQuery(scope: nil))
        XCTAssertEqual(Set(rows.map(\.id)), ["k1", "s1"])
    }

    // MARK: - Mode clamp

    func testTierCeilingHidesTextAndPathButKeepsCounts() async throws {
        try await store.upsert(make("a", realm: .kira, tier: "avocado", prompt: "explicit text"),
                               explicitCollectionIDs: [])
        try await store.upsert(make("n", realm: .kira, tier: "neutral", prompt: "a tulip"),
                               explicitCollectionIDs: [])

        let clamped = try await store.search(CatalogQuery(scope: .kira, ceiling: "apple"))
        let avocado = try XCTUnwrap(clamped.first { $0.id == "a" })
        XCTAssertNil(avocado.prompt, "text above the ceiling must not surface")
        XCTAssertEqual(avocado.absolutePath, "", "path above the ceiling must not surface")
        XCTAssertEqual(avocado.contentMode, "avocado", "the tier LABEL is metadata and stays")

        let neutral = try XCTUnwrap(clamped.first { $0.id == "n" })
        XCTAssertEqual(neutral.prompt, "a tulip")
        XCTAssertEqual(neutral.absolutePath, "/tmp/n.png")
    }

    func testNoCeilingMeansNoClamp() async throws {
        try await store.upsert(make("a", realm: .kira, tier: "avocado", prompt: "explicit text"),
                               explicitCollectionIDs: [])
        let rows = try await store.search(CatalogQuery(scope: .kira, ceiling: nil))
        XCTAssertEqual(rows.first?.prompt, "explicit text")
    }

    // MARK: - Sealed

    func testSealedRowStoresNoTextAndIsNotFullTextSearchable() async throws {
        try await store.upsert(make("sealed1", realm: .shared, prompt: "hunting-phrase", sealed: true),
                               explicitCollectionIDs: [])
        let byID = try await store.search(CatalogQuery(scope: nil))
        XCTAssertNil(byID.first { $0.id == "sealed1" }?.prompt)
        let byText = try await store.search(CatalogQuery(scope: nil, text: "hunting-phrase"))
        XCTAssertTrue(byText.isEmpty, "a sealed row must not be reachable by its text")
    }

    /// The negative case above only proves nothing came back, which a broken
    /// MATCH expression would also produce. These pin the positive side: text
    /// search does find an unsealed row, and punctuation in the query is treated
    /// as data rather than as FTS5 operator syntax (a bare `-` or `"` in a MATCH
    /// expression is a query error, and an errored query is silently empty).
    func testTextSearchFindsAnUnsealedRow() async throws {
        try await store.upsert(make("t", realm: .shared, prompt: "a hunting-phrase tulip"),
                               explicitCollectionIDs: [])
        let plain = try await store.search(CatalogQuery(scope: nil, text: "tulip"))
        XCTAssertEqual(plain.map(\.id), ["t"])
        let hyphenated = try await store.search(CatalogQuery(scope: nil, text: "hunting-phrase"))
        XCTAssertEqual(hyphenated.map(\.id), ["t"])
        let quoted = try await store.search(CatalogQuery(scope: nil, text: "\"tulip"))
        XCTAssertEqual(quoted.map(\.id), ["t"])
    }

    func testTextSearchIsRealmLockedToo() async throws {
        try await store.upsert(make("k", realm: .kira, prompt: "shared word"), explicitCollectionIDs: [])
        try await store.upsert(make("s", realm: .shared, prompt: "shared word"), explicitCollectionIDs: [])
        let rows = try await store.search(CatalogQuery(scope: .kira, text: "shared word"))
        XCTAssertEqual(rows.map(\.id), ["k"])
    }

    // MARK: - Collections

    func testDerivedFilingHappensOnUpsert() async throws {
        try await store.upsert(make("t1", realm: .kira, lane: "tile"), explicitCollectionIDs: [])
        let hers = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        let shared = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-decoupage"))
        XCTAssertEqual(hers.map(\.id), ["t1"])
        XCTAssertEqual(shared.map(\.id), ["t1"], "her tile work is in the shared body too")
    }

    func testAnAssetInTwoCollectionsIsReturnedByEitherAndCountedOnce() async throws {
        try await store.upsert(make("x", realm: .kira, lane: "shoot"),
                               explicitCollectionIDs: ["col-kira-erotic-portraiture"])
        let a = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-autocord"))
        let b = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-erotic-portraiture"))
        XCTAssertEqual(a.map(\.id), ["x"])
        XCTAssertEqual(b.map(\.id), ["x"])
        XCTAssertEqual(a.count, 1)
    }

    func testCollectionQueryMatchesRootAndChildren() async throws {
        try await store.upsert(make("auto", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        let roll = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-photography"))
        XCTAssertEqual(roll.map(\.id), ["auto"],
                       "asking for Photography must include Autocord Still Life")
    }

    func testDepthCapRejectsAThirdLevel() async throws {
        let child = CatalogCollection(id: "c-child", slug: "child", name: "Child",
                                      parentID: "col-kira-autocord", realm: .kira)
        await XCTAssertThrowsErrorAsync(try await store.createCollection(child, by: .kira))
    }

    func testKiraCannotRestructureASharedCollection() async throws {
        await XCTAssertThrowsErrorAsync(
            try await store.renameCollection(id: "col-decoupage", name: "Hers Now", by: .kira))
        await XCTAssertThrowsErrorAsync(
            try await store.retireCollection(id: "col-photography", by: .kira))
        await XCTAssertThrowsErrorAsync(
            try await store.createCollection(
                CatalogCollection(slug: "new-shared", name: "New Shared", realm: nil), by: .kira))
    }

    func testKiraCanCurateHerOwnRealm() async throws {
        let mine = CatalogCollection(id: "c-mine", slug: "kira-rainy-days",
                                     name: "Rainy Days", realm: .kira)
        try await store.createCollection(mine, by: .kira)
        try await store.renameCollection(id: "c-mine", name: "Rain", by: .kira)
        let visible = try await store.collections(visibleTo: .kira)
        XCTAssertEqual(visible.first { $0.id == "c-mine" }?.name, "Rain")
        try await store.retireCollection(id: "c-mine", by: .kira)
        // Hoisted out of the assertion: XCTAssert* takes a non-async autoclosure.
        let afterRetire = try await store.collections(visibleTo: .kira)
        XCTAssertNil(afterRetire.first { $0.id == "c-mine" })
    }

    func testKiraCannotFileASharedRow() async throws {
        try await store.upsert(make("s1", realm: .shared), explicitCollectionIDs: [])
        await XCTAssertThrowsErrorAsync(
            try await store.file(assetID: "s1", into: "col-kira-still-life", by: .kira))
    }

    func testCollectionsVisibleToKiraExcludeOtherRealmsPrivateOnes() async throws {
        try await store.createCollection(
            CatalogCollection(id: "c-secret", slug: "x", name: "X", realm: .kira), by: .kira)
        let shared = try await store.collections(visibleTo: .shared)
        XCTAssertNil(shared.first { $0.id == "c-secret" },
                     "her private collections are hers")
        XCTAssertNotNil(shared.first { $0.id == "col-decoupage" })
    }

    // MARK: - Edges

    func testI2VSourceResolvesInBothDirections() async throws {
        try await store.upsert(make("still", realm: .kira), explicitCollectionIDs: [])
        try await store.upsert(make("clip", realm: .kira, kind: "video", mode: "i2v"),
                               explicitCollectionIDs: [])
        try await store.addEdge(AssetEdge(fromAssetID: "clip", toAssetID: "still", relation: .i2vSource))

        let fromClip = try await store.edges(for: "clip")
        let fromStill = try await store.edges(for: "still")
        XCTAssertEqual(fromClip.map(\.toAssetID), ["still"])
        XCTAssertEqual(fromStill.map(\.fromAssetID), ["clip"],
                       "from the still, find its clips")
    }

    func testEdgesAreIdempotent() async throws {
        try await store.upsert(make("a", realm: .kira), explicitCollectionIDs: [])
        try await store.upsert(make("b", realm: .kira), explicitCollectionIDs: [])
        let e = AssetEdge(fromAssetID: "a", toAssetID: "b", relation: .memberOf)
        try await store.addEdge(e)
        try await store.addEdge(e)
        let edges = try await store.edges(for: "a")
        XCTAssertEqual(edges.count, 1)
    }

    // MARK: - Locations

    func testOneAssetManyLocations() async throws {
        try await store.upsert(make("a", realm: .kira), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "a",
            AssetLocation(host: "mac", path: "/Users/t/Pictures/ComfyBox/a.png", mtime: Date()))
        try await store.addLocation(assetID: "a",
            AssetLocation(host: "kira", path: "/home/todd/.kira/studio/gallery/Kira/a.png", mtime: Date()))
        let locations = try await store.locations(of: "a")
        XCTAssertEqual(locations.count, 2)
    }

    // MARK: - Facets

    func testFacetCountsAreRealmScoped() async throws {
        try await store.upsert(make("k1", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("k2", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared, lane: "shoot"), explicitCollectionIDs: [])
        let f = try await store.facets(scope: .kira)
        XCTAssertEqual(f.lane["shoot"], 2)
    }
}

/// XCTAssertThrowsError has no async form in this toolchain.
func XCTAssertThrowsErrorAsync(_ expression: @autoclosure () async throws -> Void,
                               file: StaticString = #filePath, line: UInt = #line) async {
    do {
        try await expression()
        XCTFail("expected an error", file: file, line: line)
    } catch {
        // expected
    }
}
