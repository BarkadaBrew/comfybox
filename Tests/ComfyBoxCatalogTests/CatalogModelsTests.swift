import XCTest
@testable import ComfyBoxCatalog

final class CatalogModelsTests: XCTestCase {
    func testRealmHasExactlyTwoValues() {
        XCTAssertEqual(CatalogRealm.allCases.map(\.rawValue).sorted(), ["kira", "shared"])
    }

    func testRealmDefaultsToSharedForUnknownInput() {
        XCTAssertEqual(CatalogRealm(rawValue: "todd") ?? .shared, .shared)
        XCTAssertEqual(CatalogRealm(rawValue: "kira"), .kira)
    }

    func testSealedAssetCarriesNoText() {
        let a = CatalogAsset(
            id: "a1", kind: "image", filename: "x.png", absolutePath: "/tmp/x.png",
            realm: .shared, sealed: true,
            prompt: "secret", negativePrompt: "neg", promptRaw: "raw",
            promptInjected: "inj", caption: "cap", captionSource: "capsrc",
            theme: "an intent line in full prose"
        )
        XCTAssertNil(a.prompt)
        XCTAssertNil(a.negativePrompt)
        XCTAssertNil(a.promptRaw)
        XCTAssertNil(a.promptInjected)
        XCTAssertNil(a.caption)
        XCTAssertNil(a.captionSource)
        // `theme` escaped this nulling for as long as it was filed with the
        // facets by position. MetadataReader fills it from the render journal's
        // free-text intent line, so it belongs with the text.
        XCTAssertNil(a.theme, "a sealed row kept its free-text theme")
    }

    /// The facets a sealed row DOES keep — sealing withholds the text, not the
    /// row, and a test that only checks nils would pass against a model that
    /// threw everything away.
    func testSealedAssetKeepsItsFacets() {
        let a = CatalogAsset(
            id: "a1", filename: "x.png", absolutePath: "/tmp/x.png",
            realm: .kira, sealed: true, prompt: "secret",
            contentMode: "avocado", lane: "kira", theme: "prose"
        )
        XCTAssertEqual(a.contentMode, "avocado")
        XCTAssertEqual(a.lane, "kira")
        XCTAssertEqual(a.realm, .kira)
        XCTAssertTrue(a.sealed)
    }
}
