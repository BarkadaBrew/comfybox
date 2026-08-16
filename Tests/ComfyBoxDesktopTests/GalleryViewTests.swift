// GalleryViewTests.swift — Tests for GalleryView's pure `folderMembers`
// helper (I2). The view itself is SwiftUI (no snapshot/unit harness in this
// repo, confirmed by every other *ViewTests.swift file here); this covers
// the fix for "Archive Folder…" only archiving the loaded page: folder
// membership must be resolved against the full, unpaged asset set, not the
// view's 500-row page.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("GalleryView.folderMembers")
struct GalleryViewFolderMembersTests {
    @Test("returns exactly the assets whose id is in the folder's id set")
    func filtersByIdSet() {
        let assets = (0..<10).map { TestData.makeAsset(id: "asset-\($0)", filename: "img-\($0).png") }
        let ids: Set<String> = ["asset-2", "asset-5", "asset-9"]

        let members = GalleryView.folderMembers(ids: ids, from: assets)

        #expect(members.count == 3)
        #expect(Set(members.map(\.id)) == ids)
    }

    @Test("a folder larger than the 500-row gallery page still resolves completely")
    func survivesBeyondPageSize() {
        // Simulate a folder with more members than GalleryView's loaded
        // page (fetchAssets(limit: 500)) would ever contain — the exact
        // scenario I2 fixes: membership must come from the full asset list,
        // not the paged one.
        let total = 600
        let allAssets = (0..<total).map { TestData.makeAsset(id: "asset-\($0)", filename: "img-\($0).png") }
        let folderIds = Set(allAssets.map(\.id))   // every asset is in the folder

        let members = GalleryView.folderMembers(ids: folderIds, from: allAssets)

        #expect(members.count == total)
        // In particular, ids past index 500 (which a `limit: 500` page
        // would never have contained) are present.
        #expect(members.contains(where: { $0.id == "asset-599" }))
    }

    @Test("an empty id set returns no members")
    func emptyIdSet() {
        let assets = (0..<5).map { TestData.makeAsset(id: "asset-\($0)", filename: "img-\($0).png") }
        let members = GalleryView.folderMembers(ids: [], from: assets)
        #expect(members.isEmpty)
    }

    @Test("ids not present in the asset list are silently ignored")
    func unknownIdsIgnored() {
        let assets = (0..<3).map { TestData.makeAsset(id: "asset-\($0)", filename: "img-\($0).png") }
        let members = GalleryView.folderMembers(ids: ["asset-0", "does-not-exist"], from: assets)
        #expect(members.count == 1)
        #expect(members.first?.id == "asset-0")
    }
}
