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

// MARK: - #268: toggleFavorite must not drop `source`

@Suite("GalleryView.toggledFavorite")
struct GalleryViewToggledFavoriteTests {
    @Test("flips favorite and preserves source (regression: rebuild omitted source)")
    func preservesSource() {
        let asset = TestData.makeAsset(id: "kira-1", favorite: false, source: "kira")
        let updated = GalleryView.toggledFavorite(asset)

        #expect(updated.favorite == true)
        #expect(updated.source == "kira")
    }

    @Test("toggling twice restores the original favorite state, source intact throughout")
    func togglingTwiceRestores() {
        let asset = TestData.makeAsset(id: "bree-1", favorite: false, source: "bree")
        let once = GalleryView.toggledFavorite(asset)
        let twice = GalleryView.toggledFavorite(once)

        #expect(twice.favorite == false)
        #expect(twice.source == "bree")
    }

    @Test("preserves a nil source (main-gallery asset is unaffected)")
    func preservesNilSource() {
        let asset = TestData.makeAsset(id: "main-1", favorite: false, source: nil)
        let updated = GalleryView.toggledFavorite(asset)
        #expect(updated.source == nil)
        #expect(updated.favorite == true)
    }

    @Test("every other field is carried over unchanged")
    func preservesOtherFields() {
        let asset = TestData.makeAsset(
            id: "asset-1", prompt: "a sunset", rating: 4, contentMode: "banana",
            characterName: "Alice", source: "kira"
        )
        let updated = GalleryView.toggledFavorite(asset)
        #expect(updated.id == asset.id)
        #expect(updated.prompt == asset.prompt)
        #expect(updated.rating == asset.rating)
        #expect(updated.contentMode == asset.contentMode)
        #expect(updated.characterName == asset.characterName)
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

// MARK: - remoteLoadOutcome (#223 (a): optimistic client-side password lock)
//
// Review round 2: the earlier version of this function also decided whether
// to re-lock AppContentGate on `.unauthorized`. That's removed — every
// remote read here resolves to the engine's unauthenticated
// `/v1/gallery/file`, so a 401/403 is not actually reachable through this
// path, and re-locking the LOCAL content gate in response to the CATALOG
// service's realm lock (:7871, a route this view never calls) would have
// been the wrong prompt. What's left, and still worth pinning: no raw HTTP
// status code ever reaches the message shown to the user.

@Suite("GalleryView.remoteLoadOutcome")
struct GalleryViewRemoteLoadOutcomeTests {
    @Test("an unauthorized remote fetch never shows a raw status code")
    func unauthorizedIsFriendly() {
        let message = GalleryView.remoteLoadOutcome(for: RemoteMediaCache.FetchError.unauthorized)
        #expect(!message.contains("401"))
        #expect(!message.isEmpty)
        #expect(message == "Not authorized by the server.")
    }

    @Test("a 403 (classified the same as 401) is shown the same friendly way")
    func forbiddenIsFriendlyToo() {
        let message = GalleryView.remoteLoadOutcome(
            for: RemoteMediaCache.FetchError.classify(statusCode: 403)
        )
        #expect(message == "Not authorized by the server.")
    }

    @Test("a not-found or generic server failure is shown, never as a bare code")
    func nonAuthFailuresAreFriendlyToo() {
        let notFound = GalleryView.remoteLoadOutcome(for: RemoteMediaCache.FetchError.notFound)
        #expect(!notFound.isEmpty)

        let serverError = GalleryView.remoteLoadOutcome(for: RemoteMediaCache.FetchError.server(500))
        #expect(serverError != "500", "the 500 rendered as part of a sentence, not a bare code")
    }

    @Test("a non-FetchError (e.g. a network failure) is shown via its own description")
    func genericErrorUsesItsOwnDescription() {
        struct OtherError: LocalizedError {
            var errorDescription: String? { "The Internet connection appears to be offline." }
        }
        let message = GalleryView.remoteLoadOutcome(for: OtherError())
        #expect(message == "The Internet connection appears to be offline.")
    }
}

// MARK: - archiveAllowed (#223 (c): Archive Gallery is local-server-only)

@Suite("GalleryView.archiveAllowed")
struct GalleryViewArchiveAllowedTests {
    @Test("allowed when the connected engine is local")
    func allowedWhenLocal() {
        #expect(GalleryView.archiveAllowed(engineIsLocal: true))
    }

    @Test("refused — not a silent no-op — when the connected engine is remote")
    func refusedWhenRemote() {
        #expect(!GalleryView.archiveAllowed(engineIsLocal: false))
    }

    @Test("no EngineService attached (previews, tests) defaults to allowed")
    func defaultsToAllowedWithNoEngine() {
        #expect(GalleryView.archiveAllowed(engineIsLocal: nil))
    }
}
