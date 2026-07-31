import XCTest
@testable import ComfyBoxCatalog

final class CollectionRulesTests: XCTestCase {

    private func asset(realm: CatalogRealm = .kira, lane: String? = nil,
                       source: String? = nil, family: String? = nil,
                       mode: String? = nil, kind: String = "image",
                       contentMode: String? = nil) -> CatalogAsset {
        CatalogAsset(kind: kind, filename: "x", absolutePath: "/tmp/x",
                     realm: realm, source: source, contentMode: contentMode,
                     lane: lane, family: family, mode: mode)
    }

    func testKiraLanesFileIntoHerGenres() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "still")),
                       ["col-kira-still-life"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "shoot")),
                       ["col-kira-autocord"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "film-erotic")),
                       ["col-kira-erotic-portraiture"])
    }

    func testHerTileWorkJoinsBothHerGenreAndTheSharedBody() {
        let ids = CollectionRules.defaultCollectionIDs(for: asset(lane: "tile"))
        XCTAssertEqual(Set(ids), ["col-kira-decoupage", "col-decoupage"],
                       "her decoupage is hers AND part of the shared body of work")
    }

    func testNightlifeFamilySplitsOutOfTheKiraLane() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "kira", family: "nightlife")),
                       ["col-kira-nightlife"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "kira")),
                       ["col-kira-adult-scenes"])
    }

    func testDreamsAndMemoriesIsT2VOnly() {
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(lane: "video", mode: "t2v", kind: "video")),
            ["col-kira-dreams-memories"])
        // An i2v clip is not a dream. On a REAL lane it files by that lane, the
        // same as any other asset — this case never reaches the video branch.
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(lane: "kira", mode: "i2v", kind: "video")),
            ["col-kira-adult-scenes"])
    }

    /// The `lane == "video"` + i2v path itself, which nothing covered: the branch
    /// returns [] and therefore falls through to the content-tier fallback, so
    /// the clip is filed by its OWN tier.
    ///
    /// It does NOT inherit the genre of the still it animates — that would mean
    /// resolving the i2v edge, which this pure function cannot do and which 118
    /// clips in the live catalog could not do anyway, their source stills having
    /// been deleted.
    func testAnI2VClipOnTheVideoLaneFilesByItsOwnTier() {
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(
                for: asset(lane: "video", mode: "i2v", kind: "video", contentMode: "avocado")),
            ["col-kira-adult-scenes"])
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(
                for: asset(lane: "video", mode: "i2v", kind: "video", contentMode: "apple")),
            ["col-kira-everyday"])
        // With no tier to fall back on there is nothing to file it by, and it
        // stays unfiled rather than being guessed into a genre.
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(
                for: asset(lane: "video", mode: "i2v", kind: "video")),
            [])
    }

    func testSharedProducersFileIntoSharedBodiesOnly() {
        for src in ["desktop-decoupage", "tile-engine", "krita"] {
            XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, source: src)),
                           ["col-decoupage"], "\(src) contributes to the shared decoupage body")
        }
    }

    func testASharedAssetNeverFilesIntoAKiraCollection() {
        let kiraCollectionIDs = Set(CatalogSchema.seedCollections
            .filter { $0.realm == .kira }.map(\.id))
        for lane in ["still", "shoot", "tile", "kira", "film-erotic", "video"] {
            let ids = Set(CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, lane: lane)))
            XCTAssertTrue(ids.isDisjoint(with: kiraCollectionIDs),
                          "shared asset on lane \(lane) leaked into a kira collection")
        }
    }

    func testUnknownInputFilesNowhereRatherThanGuessing() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(lane: "no-such-lane")), [])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset()), [])
    }

    // MARK: - content_mode fallback

    func testKiraContentModeFallbackWhenLaneIsMissingOrUnmapped() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(contentMode: "avocado")),
                       ["col-kira-adult-scenes"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(contentMode: "banana")),
                       ["col-kira-nightlife"])
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(contentMode: "neutral")),
                       ["col-kira-still-life"])
        // `render` is the lane the field actually records, and it maps to nothing.
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(lane: "render", contentMode: "avocado")),
            ["col-kira-adult-scenes"],
            "an UNMAPPED lane must still fall through to the content tier")
    }

    /// The owner's later call: her SFW lifestyle work gets its own body of work.
    func testAppleInKiraRealmFilesIntoEveryday() {
        XCTAssertEqual(CollectionRules.defaultCollectionIDs(for: asset(contentMode: "apple")),
                       ["col-kira-everyday"])
    }

    /// The rules may only ever name a collection that is actually seeded — a
    /// typo'd id files nothing and reports nothing.
    func testEveryTierFallbackNamesASeededCollection() {
        let seeded = Set(CatalogSchema.seedCollections.map(\.id))
        // SPELLINGS, not the three-rung order: `apple` is an alias now but is
        // still what rows carry and still what `kiraModeToCollection` keys on,
        // so iterating the order would quietly stop covering the tier whose
        // collection the owner created this task for.
        for realm in [CatalogRealm.kira, .shared] {
            for tier in CATALOG_TIER_SPELLINGS.sorted() {
                for id in CollectionRules.defaultCollectionIDs(
                    for: asset(realm: realm, contentMode: tier)) {
                    XCTAssertTrue(seeded.contains(id), "\(realm)/\(tier) named unseeded \(id)")
                }
            }
        }
    }

    func testARealLaneAlwaysBeatsTheContentModeFallback() {
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(lane: "still", contentMode: "avocado")),
            ["col-kira-still-life"],
            "a real lane wins; the fallback is last-resort only")
        XCTAssertEqual(
            Set(CollectionRules.defaultCollectionIDs(for: asset(lane: "tile", contentMode: "banana"))),
            ["col-kira-decoupage", "col-decoupage"])
    }

    func testSharedContentModeFallsBackToSharedRootsOnly() {
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, contentMode: "neutral")),
            ["col-photography"])
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, contentMode: "apple")),
            ["col-photography"])
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, contentMode: "banana")),
            ["col-adult"])
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(realm: .shared, contentMode: "avocado")),
            ["col-adult"])
    }

    /// The shared branch must never name a kira collection. If it did,
    /// `applyDerivedFiling`'s realm guard would drop the row on the way in and
    /// the fallback would silently file NOTHING for every Mac-only asset.
    func testSharedContentModeFallbackNeverNamesAKiraCollection() {
        let kiraCollectionIDs = Set(CatalogSchema.seedCollections
            .filter { $0.realm == .kira }.map(\.id))
        for tier in ["apple", "banana", "avocado", "neutral"] {
            let ids = Set(CollectionRules.defaultCollectionIDs(
                for: asset(realm: .shared, contentMode: tier)))
            XCTAssertTrue(ids.isDisjoint(with: kiraCollectionIDs),
                          "shared asset in tier \(tier) named a kira collection")
        }
    }

    /// An explicit source still wins over the tier fallback.
    func testSharedSourceBeatsTheContentModeFallback() {
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(
                for: asset(realm: .shared, source: "krita", contentMode: "avocado")),
            ["col-decoupage"])
    }
}
