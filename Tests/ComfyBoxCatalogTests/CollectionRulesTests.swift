import XCTest
@testable import ComfyBoxCatalog

final class CollectionRulesTests: XCTestCase {

    private func asset(realm: CatalogRealm = .kira, lane: String? = nil,
                       source: String? = nil, family: String? = nil,
                       mode: String? = nil, kind: String = "image") -> CatalogAsset {
        CatalogAsset(kind: kind, filename: "x", absolutePath: "/tmp/x",
                     realm: realm, source: source, lane: lane, family: family, mode: mode)
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
        // An i2v clip is not a dream — it belongs to whatever genre its scene is.
        XCTAssertEqual(
            CollectionRules.defaultCollectionIDs(for: asset(lane: "kira", mode: "i2v", kind: "video")),
            ["col-kira-adult-scenes"])
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
}
