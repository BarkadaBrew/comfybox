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
            prompt: "secret", negativePrompt: "neg", promptRaw: "raw", caption: "cap"
        )
        XCTAssertNil(a.prompt)
        XCTAssertNil(a.negativePrompt)
        XCTAssertNil(a.promptRaw)
        XCTAssertNil(a.caption)
    }
}
