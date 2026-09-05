// AssetDetailViewTests.swift — Tests for AssetDetailView.withEdits, the pure
// rebuild `saveChanges()` delegates to. The view itself is SwiftUI (no
// snapshot/unit harness in this repo, confirmed by every other *ViewTests.swift
// file here); this covers the field-preservation bug found alongside #268's
// two named source-dropping sites: saveChanges()'s DAMAsset rebuild also
// omitted `source`, so editing rating/favorite on a Kira/Bree render silently
// dropped it back into the main gallery.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("AssetDetailView.withEdits")
struct AssetDetailViewWithEditsTests {
    @Test("applies the edited rating and favorite, preserving source (regression: rebuild omitted source)")
    func preservesSource() {
        let asset = TestData.makeAsset(id: "kira-1", rating: 1, favorite: false, source: "kira")
        let updated = AssetDetailView.withEdits(asset, rating: 4, favorite: true)

        #expect(updated.rating == 4)
        #expect(updated.favorite == true)
        #expect(updated.source == "kira")
    }

    @Test("preserves a nil source (main-gallery asset is unaffected)")
    func preservesNilSource() {
        let asset = TestData.makeAsset(id: "main-1", source: nil)
        let updated = AssetDetailView.withEdits(asset, rating: 3, favorite: true)
        #expect(updated.source == nil)
    }

    @Test("every other field is carried over unchanged")
    func preservesOtherFields() {
        let asset = TestData.makeAsset(
            id: "asset-1", prompt: "a sunset", contentMode: "banana",
            characterName: "Alice", source: "bree"
        )
        let updated = AssetDetailView.withEdits(asset, rating: 5, favorite: true)
        #expect(updated.id == asset.id)
        #expect(updated.prompt == asset.prompt)
        #expect(updated.contentMode == asset.contentMode)
        #expect(updated.characterName == asset.characterName)
        #expect(updated.absolutePath == asset.absolutePath)
    }
}
