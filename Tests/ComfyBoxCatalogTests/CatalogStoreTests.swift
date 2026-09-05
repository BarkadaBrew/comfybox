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

    /// The 0700 contract belongs to `~/.comfybox`, not to whatever directory a
    /// caller's `--db` happens to name. Opening a store at an explicit path must
    /// leave that directory's mode alone — otherwise `--db ~/Desktop/x.sqlite3`
    /// silently makes ~/Desktop private.
    func testExplicitDBPathDoesNotChmodTheCallersDirectory() async throws {
        let dir = NSTemporaryDirectory() + "perm-test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o755])
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let dbFile = (dir as NSString).appendingPathComponent("x.sqlite3")
        let s = try await CatalogStore.open(path: dbFile)
        _ = try await s.unfiledAssetCount()

        let mode = try FileManager.default.attributesOfItem(atPath: dir)[.posixPermissions] as? NSNumber
        XCTAssertEqual(mode?.int16Value, 0o755, "opening a store rewrote the caller's directory mode")
        // The database file itself must still be locked down — that contract is
        // about the file's contents and applies wherever it lives.
        let fileMode = try FileManager.default
            .attributesOfItem(atPath: dbFile)[.posixPermissions] as? NSNumber
        XCTAssertEqual(fileMode?.int16Value, 0o600)
    }

    // MARK: - refileAll

    /// Derived filing is otherwise reachable only through `upsert`, which on a
    /// re-sweep skips every asset whose folded row is unchanged. Without
    /// `refileAll`, shipping new rules files nothing and still reports success.
    func testRefileAllAppliesRulesThatDidNotExistAtIngestTime() async throws {
        // Ingested with no lane and no tier — files nowhere.
        try await store.upsert(make("a1", realm: .kira), explicitCollectionIDs: [])
        let unfiledBefore = try await store.unfiledAssetCount()
        XCTAssertEqual(unfiledBefore, 1)

        // Simulate the row gaining a tier (as a re-sweep with better metadata
        // would), then refile.
        try await store.upsert(make("a1", realm: .kira, tier: "avocado"),
                               explicitCollectionIDs: [])
        let filed = try await store.refileAll()

        XCTAssertEqual(filed, 1)
        let unfiledAfter = try await store.unfiledAssetCount()
        XCTAssertEqual(unfiledAfter, 0)
        let inScenes = try await store.search(
            CatalogQuery(scope: .kira, collectionID: "col-kira-adult-scenes"))
        XCTAssertEqual(inScenes.map(\.id), ["a1"])
    }

    /// The whole reason `refileAll` is safe to re-run: it must never destroy a
    /// filing a human made by hand.
    func testRefileAllPreservesManualFilings() async throws {
        try await store.upsert(make("m1", realm: .kira, tier: "avocado"),
                               explicitCollectionIDs: [])
        // A human files it somewhere the rules would never derive.
        try await store.file(assetID: "m1", into: "col-kira-decoupage", by: nil)

        try await store.refileAll()

        let manual = try await store.search(
            CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        XCTAssertEqual(manual.map(\.id), ["m1"], "refileAll destroyed a manual filing")
        // …and the derived one is still applied alongside it.
        let derived = try await store.search(
            CatalogQuery(scope: .kira, collectionID: "col-kira-adult-scenes"))
        XCTAssertEqual(derived.map(\.id), ["m1"])
    }

    /// The COLLISION case: a manual filing into a collection the rules ALSO
    /// derive for that asset.
    ///
    /// The previous test files into a collection the rules would never pick, so
    /// it never exercises the conflicting INSERT. `applyDerivedFiling` deletes
    /// `manual = 0` and then re-inserts with `manual = 0`; only because that
    /// insert is `INSERT OR IGNORE` does the surviving `manual = 1` row keep its
    /// flag. Switch it to `INSERT OR REPLACE` and the row is demoted to 0, the
    /// NEXT refile deletes it, and the older test still passes — a hand-filing
    /// silently destroyed one run later. This is the assertion that catches that.
    /// (Verified by mutation: under OR REPLACE this test fails, the other passes.)
    func testRefileAllKeepsAManualFlagOnACollectionTheRulesAlsoDerive() async throws {
        // `avocado` derives col-kira-adult-scenes on its own.
        try await store.upsert(make("m2", realm: .kira, tier: "avocado"),
                               explicitCollectionIDs: [])
        try await store.file(assetID: "m2", into: "col-kira-adult-scenes", by: nil)
        let before = try await store.manualFilingCount(assetID: "m2")
        XCTAssertEqual(before, 1, "precondition: the filing is marked manual")

        try await store.refileAll()
        try await store.refileAll()   // the second run is where a demoted flag bites

        let after = try await store.manualFilingCount(assetID: "m2")
        XCTAssertEqual(after, 1, "a manual filing was demoted and would be deleted next refile")
        let stillThere = try await store.search(
            CatalogQuery(scope: .kira, collectionID: "col-kira-adult-scenes"))
        XCTAssertEqual(stillThere.map(\.id), ["m2"])
    }

    /// A shared asset must not acquire a kira collection through the refile path
    /// any more than through the upsert path.
    func testRefileAllRespectsTheRealmGuard() async throws {
        try await store.upsert(make("s9", realm: .shared, tier: "avocado"),
                               explicitCollectionIDs: [])
        try await store.refileAll()

        let kiraSide = try await store.search(
            CatalogQuery(scope: .kira, collectionID: "col-kira-adult-scenes"))
        XCTAssertTrue(kiraSide.isEmpty, "a shared asset reached a kira collection")
        let sharedSide = try await store.search(
            CatalogQuery(scope: .shared, collectionID: "col-adult"))
        XCTAssertEqual(sharedSide.map(\.id), ["s9"])
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
            // The negative assertion alone is trivially true of an EMPTY result,
            // so every variation must also still FIND her row. Without this, a
            // placeholder-numbering bug that made every query match nothing would
            // pass the leak check with flying colours.
            XCTAssertEqual(rows.map(\.id), ["k1"],
                           "the lock must narrow the result, not empty it, for \(q)")
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
        // EVERY text field is populated: a clamp that withheld `prompt` but
        // leaked `caption`, `promptRaw`, `negativePrompt`, `captionSource` or the
        // filename would pass a prompt-only test while still spilling the text.
        try await store.upsert(
            CatalogAsset(id: "a", filename: "a.png", absolutePath: "/tmp/a.png",
                         realm: .kira,
                         prompt: "explicit text", negativePrompt: "explicit negative",
                         promptRaw: "explicit raw", caption: "explicit caption",
                         captionSource: "explicit captioner",
                         contentMode: "avocado", characterName: "Kira", lane: "shoot",
                         rating: 4),
            explicitCollectionIDs: [])
        try await store.upsert(make("n", realm: .kira, tier: "neutral", prompt: "a tulip"),
                               explicitCollectionIDs: [])

        let clamped = try await store.search(CatalogQuery(scope: .kira, ceiling: "apple"))
        let avocado = try XCTUnwrap(clamped.first { $0.id == "a" })
        XCTAssertNil(avocado.prompt, "text above the ceiling must not surface")
        XCTAssertNil(avocado.negativePrompt, "negative prompt is text too")
        XCTAssertNil(avocado.promptRaw, "the raw phrasing is text too")
        XCTAssertNil(avocado.caption, "caption is text too")
        XCTAssertNil(avocado.captionSource, "caption source is text too")
        XCTAssertEqual(avocado.absolutePath, "", "path above the ceiling must not surface")
        XCTAssertEqual(avocado.filename, "", "the filename is a path fragment, not metadata")
        // ...while everything that is genuinely metadata survives, so counts and
        // facets over a clamped result set are still true.
        XCTAssertEqual(avocado.contentMode, "avocado", "the tier LABEL is metadata and stays")
        XCTAssertEqual(avocado.characterName, "Kira")
        XCTAssertEqual(avocado.lane, "shoot")
        XCTAssertEqual(avocado.rating, 4)
        XCTAssertEqual(avocado.realm, .kira)

        let neutral = try XCTUnwrap(clamped.first { $0.id == "n" })
        XCTAssertEqual(neutral.prompt, "a tulip")
        XCTAssertEqual(neutral.absolutePath, "/tmp/n.png")
    }

    /// tierRank used to return 0 for any tier it did not recognise, so an asset
    /// carrying a vocabulary the desktop gate already emits — NSFWGate treats
    /// "explicit", "nsfw" and "suggestive" as NSFW — ranked as NEUTRAL and walked
    /// straight through the clamp with its prompt and its absolute path.
    func testAnUnrecognisedTierVocabularyIsStillClamped() async throws {
        try await store.upsert(make("e", realm: .kira, tier: "explicit", prompt: "explicit text"),
                               explicitCollectionIDs: [])
        try await store.upsert(make("z", realm: .kira, tier: "wildly-unknown", prompt: "unknown text"),
                               explicitCollectionIDs: [])

        let rows = try await store.search(CatalogQuery(scope: .kira, ceiling: "neutral"))
        let explicit = try XCTUnwrap(rows.first { $0.id == "e" })
        XCTAssertNil(explicit.prompt, "'explicit' means avocado, not neutral")
        XCTAssertEqual(explicit.absolutePath, "")
        XCTAssertEqual(explicit.contentMode, "explicit", "the label still stays")

        let unknown = try XCTUnwrap(rows.first { $0.id == "z" })
        XCTAssertNil(unknown.prompt, "an unrecognised tier must fail CLOSED")
        XCTAssertEqual(unknown.absolutePath, "")
    }

    func testTierSynonymsRankWithTheirFruit() {
        XCTAssertEqual(tierRank("explicit"), tierRank("avocado"))
        XCTAssertEqual(tierRank("nsfw"), tierRank("avocado"))
        XCTAssertEqual(tierRank("suggestive"), tierRank("banana"))
        XCTAssertEqual(tierRank("EXPLICIT"), tierRank("avocado"), "case must not matter")
        XCTAssertEqual(tierRank(nil), 0, "untiered is not secretly explicit")
        XCTAssertEqual(tierRank("never-heard-of-it"), CATALOG_TIER_ORDER.count,
                       "an unknown tier outranks every known one")
        // The ceiling rounds the other way, so an unknown ceiling admits least.
        XCTAssertEqual(ceilingRank("never-heard-of-it"), 0)
        XCTAssertEqual(ceilingRank("explicit"), tierRank("avocado"))
    }

    // MARK: - apple IS neutral

    /// The daemon canonicalizes `apple -> neutral` on write and no conversation
    /// can produce a ceiling of `apple`, so while the catalog ranked apple ABOVE
    /// neutral there was no ceiling at all that admitted apple-tier content: 278
    /// live rows, 164 of them the whole `col-kira-everyday` genre, were countable
    /// and permanently unopenable. The owner's call was to collapse them here.
    func testAppleRanksAsNeutralOnBothOperands() {
        XCTAssertEqual(tierRank("apple"), tierRank("neutral"))
        XCTAssertEqual(tierRank("  Apple "), tierRank("neutral"), "case and space must not matter")
        XCTAssertEqual(ceilingRank("apple"), ceilingRank("neutral"))
        XCTAssertFalse(CATALOG_TIER_ORDER.contains("apple"), "apple is an alias, not a rung")
        XCTAssertEqual(CATALOG_TIER_ORDER, ["neutral", "banana", "avocado"])
    }

    /// The rungs still separate, in order — the collapse must not have flattened
    /// anything else on its way past.
    func testTheRemainingRungsStillRankStrictly() {
        XCTAssertEqual(CATALOG_TIER_ORDER.map { tierRank($0) }, [0, 1, 2])
        XCTAssertTrue(tierRank("banana") > ceilingRank("neutral"))
        XCTAssertTrue(tierRank("avocado") > ceilingRank("banana"))
    }

    /// The end-to-end version of the above: this is the 164-asset symptom.
    func testAnAppleAssetSurfacesAtANeutralCeilingAndABananaOneDoesNot() async throws {
        try await store.upsert(make("ap", realm: .kira, tier: "apple", prompt: "a bowl of pears"),
                               explicitCollectionIDs: [])
        try await store.upsert(make("ba", realm: .kira, tier: "banana", prompt: "a neon bar"),
                               explicitCollectionIDs: [])

        let rows = try await store.search(CatalogQuery(scope: .kira, ceiling: "neutral"))
        let apple = try XCTUnwrap(rows.first { $0.id == "ap" })
        XCTAssertEqual(apple.prompt, "a bowl of pears",
                       "an apple asset is unopenable at every ceiling anyone can ask for")
        XCTAssertEqual(apple.absolutePath, "/tmp/ap.png")
        XCTAssertEqual(apple.contentMode, "apple", "the stored LABEL is untouched; only its rank moved")

        let banana = try XCTUnwrap(rows.first { $0.id == "ba" })
        XCTAssertNil(banana.prompt, "the collapse must not have opened the tier above")
        XCTAssertEqual(banana.absolutePath, "")
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

    /// A filename is the most guessable key in the schema, so the lookup that
    /// takes one carries the same realm lock every other lookup here does.
    /// Without it, backfill's basename fallback could link a clip to a still in
    /// a realm the caller cannot see.
    func testFilenameLookupTakesTheRealmLock() async throws {
        try await store.upsert(CatalogAsset(id: "k1", filename: "still.png",
                                            absolutePath: "/tmp/hers/still.png", realm: .kira),
                               explicitCollectionIDs: [])
        try await store.upsert(CatalogAsset(id: "s1", filename: "still.png",
                                            absolutePath: "/tmp/theirs/still.png", realm: .shared),
                               explicitCollectionIDs: [])

        let hers = try await store.assetIDs(forFilename: "still.png", scope: .kira)
        XCTAssertEqual(hers, ["k1"], "her scope sees only hers")
        let theirs = try await store.assetIDs(forFilename: "still.png", scope: .shared)
        XCTAssertEqual(theirs, ["s1"])
        let unscoped = try await store.assetIDs(forFilename: "still.png", scope: nil)
        XCTAssertEqual(unscoped.count, 2, "the service still sees both")
    }

    /// `asset_collections` has no foreign keys, so filing an id that names no
    /// row inserts happily and leaves a membership pointing at nothing — which
    /// inflates every collection count. POST /v1/catalog/file is unauthenticated
    /// and unscoped, so the id is whatever a caller typed.
    func testFilingAnUnknownAssetIsRefusedRatherThanOrphaned() async throws {
        await XCTAssertThrowsErrorAsync(
            try await store.file(assetID: "no-such-asset", into: "col-kira-still-life", by: nil))
        let counts = try await store.facets(scope: nil).collection
        XCTAssertNil(counts["col-kira-still-life"])
    }

    /// The prohibition belongs to the data, not to the caller: the service actor
    /// (nil) used to bypass it, so backfill or the desktop app could put a shared
    /// row inside her realm.
    func testNotEvenTheServiceCanFileASharedRowIntoAKiraCollection() async throws {
        try await store.upsert(make("s1", realm: .shared), explicitCollectionIDs: [])
        await XCTAssertThrowsErrorAsync(
            try await store.file(assetID: "s1", into: "col-kira-still-life", by: nil))
        let inHers = try await store.search(
            CatalogQuery(scope: nil, collectionID: "col-kira-still-life"))
        XCTAssertTrue(inHers.isEmpty)
    }

    /// The same rule on the upsert path: naming a kira collection explicitly for
    /// a shared asset must not file it there.
    func testExplicitKiraCollectionIsIgnoredForASharedAsset() async throws {
        try await store.upsert(make("s1", realm: .shared),
                               explicitCollectionIDs: ["col-kira-still-life"])
        let rows = try await store.search(
            CatalogQuery(scope: nil, collectionID: "col-kira-still-life"))
        XCTAssertTrue(rows.isEmpty, "a shared asset cannot be filed into her realm by request")
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

        let fromClip = try await store.edges(for: "clip", scope: nil)
        let fromStill = try await store.edges(for: "still", scope: nil)
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
        let edges = try await store.edges(for: "a", scope: nil)
        XCTAssertEqual(edges.count, 1)
    }

    /// The escalation chain the realm lock exists to stop: a scoped search hands
    /// her one of her OWN ids, and the graph walk from it must not hand back a
    /// shared one.
    func testScopedEdgesDoNotRevealASharedEndpoint() async throws {
        try await store.upsert(make("shared-still", realm: .shared), explicitCollectionIDs: [])
        try await store.upsert(make("her-clip", realm: .kira, kind: "video", mode: "i2v"),
                               explicitCollectionIDs: [])
        try await store.addEdge(AssetEdge(fromAssetID: "her-clip", toAssetID: "shared-still",
                                          relation: .i2vSource))

        let unscoped = try await store.edges(for: "her-clip", scope: nil)
        XCTAssertEqual(unscoped.map(\.toAssetID), ["shared-still"], "the service still sees it")

        let scoped = try await store.edges(for: "her-clip", scope: .kira)
        XCTAssertTrue(scoped.isEmpty, "an edge to a shared asset must not survive her scope")
    }

    // MARK: - Locations

    func testScopedLookupsRefuseAnOutOfRealmAsset() async throws {
        try await store.upsert(
            CatalogAsset(id: "s1", filename: "s1.png", absolutePath: "/tmp/s1.png",
                         sha256: "deadbeef", realm: .shared),
            explicitCollectionIDs: [])
        try await store.addLocation(assetID: "s1",
            AssetLocation(host: "mac", path: "/Users/t/Pictures/ComfyBox/s1.png", mtime: Date()))

        // Unscoped — the service itself — resolves all three.
        let serviceLocations = try await store.locations(of: "s1", scope: nil)
        XCTAssertEqual(serviceLocations.count, 1)
        let serviceByPath = try await store.assetID(forPath: "/tmp/s1.png", scope: nil)
        XCTAssertEqual(serviceByPath, "s1")
        let serviceBySHA = try await store.assetID(forSHA256: "deadbeef", scope: nil)
        XCTAssertEqual(serviceBySHA, "s1")

        // Scoped to her realm, all three go quiet — including the two that would
        // otherwise confirm a GUESSED path or hash.
        let herLocations = try await store.locations(of: "s1", scope: .kira)
        XCTAssertTrue(herLocations.isEmpty, "on-disk paths of a shared asset are not hers")
        let herByPath = try await store.assetID(forPath: "/tmp/s1.png", scope: .kira)
        XCTAssertNil(herByPath, "a guessed path must not be confirmable")
        let herBySHA = try await store.assetID(forSHA256: "deadbeef", scope: .kira)
        XCTAssertNil(herBySHA, "a guessed hash must not be confirmable")
    }

    func testOneAssetManyLocations() async throws {
        try await store.upsert(make("a", realm: .kira), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "a",
            AssetLocation(host: "mac", path: "/Users/t/Pictures/ComfyBox/a.png", mtime: Date()))
        try await store.addLocation(assetID: "a",
            AssetLocation(host: "kira", path: "/home/todd/.kira/studio/gallery/Kira/a.png", mtime: Date()))
        let locations = try await store.locations(of: "a", scope: nil)
        XCTAssertEqual(locations.count, 2)
    }

    // MARK: - Paging

    /// limit and offset are the last two placeholders in the generated SQL — the
    /// part most likely to break if the parameter numbering ever drifts — and
    /// nothing else in the suite moves them off their defaults.
    func testLimitAndOffsetPageTheResult() async throws {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        for (i, id) in ["oldest", "middle", "newest"].enumerated() {
            try await store.upsert(
                CatalogAsset(id: id, filename: "\(id).png", absolutePath: "/tmp/\(id).png",
                             createdAt: base.addingTimeInterval(Double(i) * 60),
                             realm: .kira, lane: "shoot"),
                explicitCollectionIDs: [])
        }

        let all = try await store.search(CatalogQuery(scope: .kira))
        XCTAssertEqual(all.map(\.id), ["newest", "middle", "oldest"])

        let firstPage = try await store.search(CatalogQuery(scope: .kira, limit: 2))
        XCTAssertEqual(firstPage.map(\.id), ["newest", "middle"])

        let secondPage = try await store.search(CatalogQuery(scope: .kira, limit: 2, offset: 2))
        XCTAssertEqual(secondPage.map(\.id), ["oldest"])

        // Paging must not widen the scope either.
        try await store.upsert(make("s1", realm: .shared, lane: "shoot"), explicitCollectionIDs: [])
        let pagedWithOther = try await store.search(
            CatalogQuery(scope: .kira, limit: 10, offset: 0))
        XCTAssertFalse(pagedWithOther.contains { $0.realm != .kira })
    }

    // MARK: - Facets

    func testFacetCountsAreRealmScoped() async throws {
        try await store.upsert(make("k1", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("k2", realm: .kira, lane: "shoot"), explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared, lane: "shoot"), explicitCollectionIDs: [])
        let f = try await store.facets(scope: .kira)
        XCTAssertEqual(f.lane["shoot"], 2)
    }

    // MARK: - The service-side by-id fetch

    /// `asset(id:)` is the unscoped, unclamped backfill helper. The variant a
    /// SERVICE calls carries both rules, or fetching one row by id becomes the
    /// way around the two the store exists to enforce.
    func testScopedByIDFetchAppliesBothTheRealmLockAndTheClamp() async throws {
        try await store.upsert(make("k1", realm: .kira, tier: "avocado", prompt: "a nightclub"),
                               explicitCollectionIDs: [])
        try await store.upsert(make("s1", realm: .shared, prompt: "a tulip"),
                               explicitCollectionIDs: [])

        // Realm lock: out of scope is nil, not a row — and nil is also the
        // answer for an id that does not exist, so the two are indistinguishable.
        let shared = try await store.asset(id: "s1", visibleTo: .kira, ceiling: nil)
        XCTAssertNil(shared)
        let missing = try await store.asset(id: "no-such-id", visibleTo: .kira, ceiling: nil)
        XCTAssertNil(missing)

        // Clamp: above the ceiling the label survives, the text and path do not.
        let clamped = try await XCTUnwrapAsync(
            try await store.asset(id: "k1", visibleTo: .kira, ceiling: "apple"))
        XCTAssertNil(clamped.prompt)
        XCTAssertEqual(clamped.absolutePath, "")
        XCTAssertEqual(clamped.contentMode, "avocado")

        // No ceiling, in scope: the whole row.
        let full = try await XCTUnwrapAsync(
            try await store.asset(id: "k1", visibleTo: .kira, ceiling: nil))
        XCTAssertEqual(full.prompt, "a nightclub")
        XCTAssertEqual(full.absolutePath, "/tmp/k1.png")
    }

    // MARK: - assetIDs(matching:) — #271

    /// The literal #271 acceptance test: a filter matching far more rows than
    /// any page clamp in this codebase (500 for the HTTP listing, 20k for the
    /// desktop's old fixed "full scope" raise) must return every one of them,
    /// not a truncated prefix.
    func testAssetIDsMatchingSelectsAllOf1200SyntheticRows() async throws {
        for i in 0..<1200 {
            try await store.upsert(make("select-all-\(i)", realm: .shared, lane: "select-all-fixture"),
                                   explicitCollectionIDs: [])
        }
        let result = try await store.assetIDs(matching: CatalogQuery(lane: "select-all-fixture"))
        XCTAssertEqual(result.ids.count, 1200)
        XCTAssertEqual(Set(result.ids), Set((0..<1200).map { "select-all-\($0)" }))
        XCTAssertFalse(result.truncated, "1,200 rows is well under idsHardCap")
    }

    /// R1 review correction: the hard cap must not truncate silently. `cap:`
    /// lets this pin the behaviour without inserting `idsHardCap + 1` real
    /// rows.
    func testAssetIDsMatchingReportsTruncationWhenTheCapCutsRows() async throws {
        for i in 0..<10 {
            try await store.upsert(make("cap-\(i)", realm: .shared, lane: "cap-fixture"),
                                   explicitCollectionIDs: [])
        }
        let result = try await store.assetIDs(matching: CatalogQuery(lane: "cap-fixture"), cap: 5)
        XCTAssertTrue(result.truncated)
        XCTAssertEqual(result.ids.count, 5, "truncated to exactly the cap, not the cap-plus-lookahead row")
    }

    /// The negative case for the same knob: a filter that matches EXACTLY the
    /// cap is complete, not truncated — the "+1 lookahead row" must not turn
    /// an exact fit into a false truncation report.
    func testAssetIDsMatchingIsNotTruncatedWhenCountExactlyMeetsTheCap() async throws {
        for i in 0..<5 {
            try await store.upsert(make("exact-\(i)", realm: .shared, lane: "exact-fixture"),
                                   explicitCollectionIDs: [])
        }
        let result = try await store.assetIDs(matching: CatalogQuery(lane: "exact-fixture"), cap: 5)
        XCTAssertFalse(result.truncated)
        XCTAssertEqual(result.ids.count, 5)
    }

    /// Unlike `search`, which pages, `assetIDs(matching:)` ignores
    /// `query.limit`/`query.offset` entirely — that is the whole point (#271):
    /// a query built with a small limit must still get back every matching id.
    func testAssetIDsMatchingIgnoresQueryLimitAndOffset() async throws {
        for i in 0..<10 {
            try await store.upsert(make("ignore-limit-\(i)", realm: .shared, lane: "ignore-limit"),
                                   explicitCollectionIDs: [])
        }
        var q = CatalogQuery(lane: "ignore-limit")
        q.limit = 1
        q.offset = 5
        let result = try await store.assetIDs(matching: q)
        XCTAssertEqual(result.ids.count, 10, "limit/offset must not clamp the ids-only query")
    }

    /// The realm lock applies here exactly as it does to `search` — an
    /// ids-only query is still a way to enumerate the catalog, and a confined
    /// caller's select-all must not reach into another realm.
    func testAssetIDsMatchingHonoursRealmScope() async throws {
        try await store.upsert(make("kira-only", realm: .kira, lane: "mixed"), explicitCollectionIDs: [])
        try await store.upsert(make("shared-only", realm: .shared, lane: "mixed"), explicitCollectionIDs: [])
        let result = try await store.assetIDs(matching: CatalogQuery(scope: .kira, lane: "mixed"))
        XCTAssertEqual(result.ids, ["kira-only"])
    }

    /// Built from the same `whereClause` helper as `search` (#271), so a
    /// caller cannot get a different answer to "what matches" depending on
    /// which of the two it asks.
    func testAssetIDsMatchingAgreesWithSearchOnWhatMatches() async throws {
        try await store.upsert(make("agree-1", realm: .shared, tier: "neutral"), explicitCollectionIDs: [])
        try await store.upsert(make("agree-2", realm: .shared, tier: "avocado"), explicitCollectionIDs: [])
        let query = CatalogQuery(tier: "neutral")
        let searchIDs = Set(try await store.search(query).map(\.id))
        let idsOnly = Set(try await store.assetIDs(matching: query).ids)
        XCTAssertEqual(searchIDs, idsOnly)
    }

    /// The realm-scoped ORACLES must not carry a defaulted `scope`.
    ///
    /// `edges` and `locations` had their defaults dropped for this exact reason
    /// and these three are the same shape: they answer "does an asset with this
    /// path / this hash / this filename exist?", which lets a caller CONFIRM a
    /// guessed path or hash. A `scope: CatalogRealm? = nil` default means a new
    /// call site can omit the realm, and omitting it compiles clean and answers
    /// across every realm — a silent cross-realm leak with no diff to notice.
    /// With no default, omission is a compile error and the compiler enumerates
    /// every call site. (Backfill's explicit `scope: nil` is correct and stays.)
    ///
    /// This is a SOURCE pin because the guarantee is a compile-time one: a test
    /// that omits the argument cannot be written against the fixed code, so the
    /// only way to catch a re-added default is to read the signature.
    func testScopedOraclesHaveNoDefaultRealm() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // ComfyBoxCatalogTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("Sources/ComfyBoxCatalog/CatalogStore.swift")
        let text = try String(contentsOf: source, encoding: .utf8)

        for oracle in ["assetID(forPath path: String,",
                       "assetID(forSHA256 sha: String,",
                       "assetIDs(forFilename name: String,"] {
            guard let start = text.range(of: oracle) else {
                return XCTFail("CatalogStore no longer declares \(oracle) — update this pin")
            }
            // The signature runs to the opening brace of the body.
            guard let brace = text.range(of: "{", range: start.upperBound..<text.endIndex) else {
                return XCTFail("could not find the body of \(oracle)")
            }
            let signature = String(text[start.lowerBound..<brace.lowerBound])
            XCTAssertTrue(signature.contains("scope:"),
                          "\(oracle) must still take a realm")
            XCTAssertFalse(signature.contains("CatalogRealm? = nil"),
                           "\(oracle) must NOT default its realm: omitting it compiles and "
                           + "leaks across realms. Signature was: \(signature)")
        }
    }
}

/// `XCTUnwrap` on an async expression needs the value materialised first.
func XCTUnwrapAsync<T>(_ value: T?, file: StaticString = #filePath,
                       line: UInt = #line) throws -> T {
    try XCTUnwrap(value, file: file, line: line)
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
