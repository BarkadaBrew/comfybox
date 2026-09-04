// GalleryViewTests.swift — Tests for GalleryView's pure `folderMembers`
// helper (I2). The view itself is SwiftUI (no snapshot/unit harness in this
// repo, confirmed by every other *ViewTests.swift file here); this covers
// the fix for "Archive Folder…" only archiving the loaded page: folder
// membership must be resolved against the full, unpaged asset set, not the
// view's 500-row page.

import Testing
import Foundation
import SQLite3
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

// MARK: - C1: "desktop-edit" belongs in the main gallery

@Suite("GalleryView.isMainSource")
struct GalleryViewIsMainSourceTests {
    @Test("desktop-edit is a main source, not a persona section")
    func desktopEditIsMain() {
        #expect(GalleryView.isMainSource("desktop-edit"))
    }

    @Test("existing main sources are unaffected")
    func existingMainSources() {
        #expect(GalleryView.isMainSource(nil))
        #expect(GalleryView.isMainSource(""))
        #expect(GalleryView.isMainSource("desktop"))
        #expect(GalleryView.isMainSource("comfyui"))
        #expect(GalleryView.isMainSource("comfybox"))
    }

    @Test("a persona source is still excluded")
    func personaSourceExcluded() {
        #expect(!GalleryView.isMainSource("kira"))
        #expect(!GalleryView.isMainSource("bree"))
    }
}

// MARK: - X5 residual: "Edited from → Show" lands on the source's own persona section

@Suite("GalleryView.personaFilterKey")
struct GalleryViewPersonaFilterKeyTests {
    @Test("a main source resolves to nil (the main gallery)")
    func mainSourceResolvesToNil() {
        #expect(GalleryView.personaFilterKey(for: nil) == nil)
        #expect(GalleryView.personaFilterKey(for: "") == nil)
        #expect(GalleryView.personaFilterKey(for: "desktop") == nil)
        #expect(GalleryView.personaFilterKey(for: "desktop-edit") == nil)
        #expect(GalleryView.personaFilterKey(for: "comfybox") == nil)
    }

    @Test("a persona source resolves to its own lowercased key, not nil")
    func personaSourceResolvesToItsOwnKey() {
        #expect(GalleryView.personaFilterKey(for: "kira") == "kira")
        #expect(GalleryView.personaFilterKey(for: "Bree") == "bree")   // matches filteredAssets' own lowercasing
    }
}

// MARK: - X5: "Edited from → Show" resolves against the full asset list

@Suite("GalleryView.resolveSourceAsset")
struct GalleryViewResolveSourceAssetTests {
    @Test("resolves by source_asset_id even when the path doesn't match any known local path")
    func resolvesById() {
        let assets = [TestData.makeAsset(id: "root-1", filename: "root.png")]
        let match = GalleryView.resolveSourceAsset(sourceAssetId: "root-1", sourcePath: "/nowhere/gone.png",
                                                    in: assets, localPath: { _ in nil })
        #expect(match?.id == "root-1")
    }

    @Test("falls back to a normalized local-path match when there is no asset id")
    func fallsBackToPath() {
        let assets = [TestData.makeAsset(id: "root-1", filename: "root.png")]
        let match = GalleryView.resolveSourceAsset(sourceAssetId: nil, sourcePath: "/orig/root.png",
                                                    in: assets, localPath: { _ in "/orig/root.png" })
        #expect(match?.id == "root-1")
    }

    @Test("an id that isn't in the asset list falls through to the path match")
    func idMissFallsThroughToPath() {
        let assets = [TestData.makeAsset(id: "root-1", filename: "root.png")]
        let match = GalleryView.resolveSourceAsset(sourceAssetId: "does-not-exist", sourcePath: "/orig/root.png",
                                                    in: assets, localPath: { _ in "/orig/root.png" })
        #expect(match?.id == "root-1")
    }

    @Test("no match returns nil")
    func noMatch() {
        let assets = [TestData.makeAsset(id: "root-1", filename: "root.png")]
        let match = GalleryView.resolveSourceAsset(sourceAssetId: nil, sourcePath: "/orig/other.png",
                                                    in: assets, localPath: { _ in "/orig/root.png" })
        #expect(match == nil)
    }
}

// MARK: - PR #356 fix round 1: pruneOrphans() failures must never be
// silently swallowed in loadAssets()'s self-heal sweep.

/// `loadAssets()`'s `catch` around the unattended `pruneOrphans()` self-heal
/// used to special-case only `DAMStoreError.pruneRefused` into the
/// `pruneWarning` banner and silently discard everything else — including
/// #263's new `DAMStoreError.stepFailed`, which meant a truncated-read
/// failure during the sweep left the user with no indication the sweep
/// didn't run. `pruneSweepWarning(for:)` is the pure decision `loadAssets`
/// delegates to: every failure produces banner text (the sweep must never
/// look like it silently succeeded), while browsing itself is never
/// blocked — the view's `catch` sets `pruneWarning` and continues past it.
@Suite("GalleryView.pruneSweepWarning")
struct GalleryViewPruneSweepWarningTests {
    @Test("pruneRefused surfaces its own message verbatim, unchanged from before this fix")
    func pruneRefusedUnchanged() {
        let error = DAMStoreError.pruneRefused(candidates: 6, total: 100)
        let warning = GalleryView.pruneSweepWarning(for: error)
        #expect(warning == error.localizedDescription)
    }

    @Test("stepFailed (#263) is surfaced, not silently discarded")
    func stepFailedIsSurfaced() {
        let error = DAMStoreError.stepFailed(SQLITE_IOERR, "disk I/O error")
        let warning = GalleryView.pruneSweepWarning(for: error)
        #expect(warning.contains("disk I/O error"))
        #expect(warning.contains("Orphan cleanup skipped"))
    }

    @Test("every other DAMStoreError case is also surfaced, not just stepFailed")
    func otherDAMStoreErrorsAreSurfaced() {
        let error = DAMStoreError.prepareFailed("syntax error near SELECT")
        let warning = GalleryView.pruneSweepWarning(for: error)
        #expect(warning.contains("syntax error near SELECT"))
        #expect(warning.contains("Orphan cleanup skipped"))
    }

    @Test("a non-DAMStoreError failure is surfaced too, not just typed sqlite errors")
    func nonDAMStoreErrorIsSurfaced() {
        struct OtherError: LocalizedError {
            var errorDescription: String? { "thumbnail directory unreadable" }
        }
        let warning = GalleryView.pruneSweepWarning(for: OtherError())
        #expect(warning.contains("thumbnail directory unreadable"))
        #expect(warning.contains("Orphan cleanup skipped"))
    }
}
