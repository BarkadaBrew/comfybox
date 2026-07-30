import XCTest
@testable import ComfyBoxCatalog

final class CatalogBackfillTests: XCTestCase {
    private var root: String!
    private var dbPath: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        root = NSTemporaryDirectory() + "bf-\(UUID().uuidString)"
        dbPath = root + "/catalog.sqlite3"
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        store = try await CatalogStore.open(path: dbPath)
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: root)
        try await super.tearDown()
    }

    /// Write `bytes` to <tree>/<relative> and return the absolute path.
    @discardableResult
    private func write(_ relative: String, bytes: String, tree: String) throws -> String {
        let path = root + "/" + tree + "/" + relative
        try FileManager.default.createDirectory(
            atPath: (path as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true)
        try Data(bytes.utf8).write(to: URL(fileURLWithPath: path))
        return path
    }

    private func writeSidecar(_ relative: String, json: String, tree: String) throws {
        _ = try write(relative, bytes: json, tree: tree)
    }

    private func trees() -> [BackfillTree] {
        [
            BackfillTree(id: "home", realm: nil, host: "mac",
                         mediaRoot: root + "/home", metadataRoot: nil),
            BackfillTree(id: "kira", realm: .kira, host: "kira",
                         mediaRoot: root + "/kira/gallery", metadataRoot: root + "/kira/metadata"),
        ]
    }

    func testAssetsAreIndexedFromEveryTree() async throws {
        try write("a.png", bytes: "AAA", tree: "home")
        try write("Kira/generated/b.png", bytes: "BBB", tree: "kira/gallery")
        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.assetsIndexed, 2)
        // Awaits are hoisted out of the assertions throughout: XCTAssert* takes
        // autoclosures, which cannot carry `await`.
        let rows = try await store.search(CatalogQuery(scope: nil))
        XCTAssertEqual(rows.count, 2)
    }

    func testIdenticalBytesInTwoTreesBecomeOneAssetWithTwoLocations() async throws {
        try write("dup.png", bytes: "SAME BYTES", tree: "home")
        try write("Kira/generated/renamed.png", bytes: "SAME BYTES", tree: "kira/gallery")
        let report = try await CatalogBackfill.run(store: store, trees: trees())

        let rows = try await store.search(CatalogQuery(scope: nil))
        XCTAssertEqual(rows.count, 1, "same bytes = one asset")
        XCTAssertEqual(report.duplicatesMerged, 1)
        let locations = try await store.locations(of: rows[0].id)
        XCTAssertEqual(locations.count, 2)
    }

    func testRealmComesFromTheTreeThatHoldsTheCopy() async throws {
        try write("only-home.png", bytes: "H", tree: "home")
        try write("shared-bytes.png", bytes: "S", tree: "home")
        try write("Kira/generated/shared-bytes.png", bytes: "S", tree: "kira/gallery")
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let kira = try await store.search(CatalogQuery(scope: .kira))
        let shared = try await store.search(CatalogQuery(scope: .shared))
        XCTAssertEqual(kira.count, 1, "a Mac asset with a Kira twin is hers")
        XCTAssertEqual(shared.count, 1, "a Mac asset with no twin defaults to shared")
    }

    func testSidecarSuppliesFacetsAndFilesIntoACollection() async throws {
        try write("Kira/generated/tile1.png", bytes: "T", tree: "kira/gallery")
        try writeSidecar("Kira/generated/tile1.json", json: """
            {"character":"kira","content_mode":"neutral","lane":"tile","preset":"krea-kira","sealed":false}
            """, tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let hers = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        XCTAssertEqual(hers.count, 1)
        XCTAssertEqual(hers.first?.contentMode, "neutral")
        XCTAssertEqual(hers.first?.preset, "krea-kira")
    }

    /// The real shape of the fleet: rendered to the Mac gallery, which has NO
    /// sidecars, then copied to her server tree under a different name, which
    /// does. The Mac copy is swept first, so the row is created empty; if the
    /// merge only added a location, every asset that exists in both trees —
    /// which is most of them — would stay permanently unfiled and unfaceted.
    func testFacetsFromADownstreamCopyAreFoldedIntoTheMergedRow() async throws {
        try write("shot.png", bytes: "MERGED", tree: "home")
        try write("Kira/generated/1783_shot.png", bytes: "MERGED", tree: "kira/gallery")
        try writeSidecar("Kira/generated/1783_shot.json",
                         json: #"{"character":"kira","lane":"tile","content_mode":"neutral","preset":"krea-kira"}"#,
                         tree: "kira/metadata")
        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.duplicatesMerged, 1)

        let hers = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        XCTAssertEqual(hers.count, 1, "the downstream sidecar's facets must survive the merge")
        XCTAssertEqual(hers.first?.lane, "tile")
        XCTAssertEqual(hers.first?.characterName, "kira")
        XCTAssertEqual(hers.first?.preset, "krea-kira")
    }

    func testSealedSidecarProducesARowWithNoText() async throws {
        try write("Kira/generated/s.png", bytes: "S1", tree: "kira/gallery")
        try writeSidecar("Kira/generated/s.json", json: """
            {"character":"bree","sealed":true,"prompt":"must not be stored","content_mode":"apple"}
            """, tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let rows = try await store.search(CatalogQuery(scope: nil))
        XCTAssertNil(rows.first?.prompt)
        let byText = try await store.search(CatalogQuery(scope: nil, text: "must not be stored"))
        XCTAssertTrue(byText.isEmpty)
    }

    func testSourceImageBecomesAnI2VEdge() async throws {
        let still = try write("Kira/generated/still.png", bytes: "STILL", tree: "kira/gallery")
        try write("Kira/video/clip.mp4", bytes: "CLIP", tree: "kira/gallery")
        try writeSidecar("Kira/video/clip.json", json: """
            {"character":"kira","mode":"i2v","content_mode":"banana","source_image":"\(still)"}
            """, tree: "kira/metadata")
        let report = try await CatalogBackfill.run(store: store, trees: trees())

        XCTAssertEqual(report.edgesCreated, 1)
        let foundClip = try await store.assetID(forPath: root + "/kira/gallery/Kira/video/clip.mp4")
        let clipID = try XCTUnwrap(foundClip)
        let edges = try await store.edges(for: clipID)
        XCTAssertEqual(edges.first?.relation, .i2vSource)
    }

    func testAnUnresolvableSourceImageIsSkippedNotInvented() async throws {
        try write("Kira/video/orphan.mp4", bytes: "O", tree: "kira/gallery")
        try writeSidecar("Kira/video/orphan.json", json: """
            {"mode":"i2v","source_image":"/nowhere/gone.png"}
            """, tree: "kira/metadata")
        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.edgesCreated, 0)
        XCTAssertEqual(report.edgesUnresolved, 1, "a dropped edge must be visible, not silent")
        XCTAssertEqual(report.assetsIndexed, 1, "the clip is still indexed")
    }

    func testBackfillIsIdempotent() async throws {
        try write("a.png", bytes: "A", tree: "home")
        try write("Kira/generated/b.png", bytes: "B", tree: "kira/gallery")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let firstCount = try await store.search(CatalogQuery(scope: nil, limit: 500)).count
        let firstFound = try await store.assetID(forPath: root + "/home/a.png")
        let firstID = try XCTUnwrap(firstFound)
        let firstLocations = try await store.locations(of: firstID).count

        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let secondCount = try await store.search(CatalogQuery(scope: nil, limit: 500)).count
        XCTAssertEqual(secondCount, firstCount)
        let secondFound = try await store.assetID(forPath: root + "/home/a.png")
        let secondID = try XCTUnwrap(secondFound)
        let secondLocations = try await store.locations(of: secondID).count
        XCTAssertEqual(secondLocations, firstLocations)
    }

    func testRebuildFromScratchProducesTheSameRows() async throws {
        try write("Kira/generated/x.png", bytes: "X", tree: "kira/gallery")
        try writeSidecar("Kira/generated/x.json", json: #"{"character":"kira","lane":"shoot","content_mode":"neutral"}"#,
                         tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let before = try await store.search(CatalogQuery(scope: nil, limit: 500)).map(\.filename).sorted()

        // Delete the catalog entirely and rebuild from the media alone.
        store = nil
        try FileManager.default.removeItem(atPath: dbPath)
        store = try await CatalogStore.open(path: dbPath)
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let after = try await store.search(CatalogQuery(scope: nil, limit: 500)).map(\.filename).sorted()
        XCTAssertEqual(after, before, "the catalog must be reconstructible from the files")
    }

    /// Video cannot fall back on the file, so its rebuild is asserted separately.
    func testVideoRebuildsFromItsSidecar() async throws {
        try write("Kira/video/v.mp4", bytes: "V", tree: "kira/gallery")
        try writeSidecar("Kira/video/v.json", json: """
            {"character":"kira","mode":"t2v","lane":"video","content_mode":"banana",
             "resolution":"480p","aspect_ratio":"9:16","duration":null}
            """, tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        store = nil
        try FileManager.default.removeItem(atPath: dbPath)
        store = try await CatalogStore.open(path: dbPath)
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let rows = try await store.search(CatalogQuery(scope: .kira, kind: "video"))
        XCTAssertEqual(rows.first?.mode, "t2v")
        XCTAssertEqual(rows.first?.resolution, "480p")
        XCTAssertEqual(rows.first?.aspectRatio, "9:16")
        let dreams = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-dreams-memories"))
        XCTAssertEqual(dreams.count, 1, "t2v files into Dreams & Memories")
    }

    // MARK: - Re-sweeping a file whose bytes changed

    /// `SidecarService.embedArgs` runs exiftool with `-overwrite_original`, and
    /// a re-render reuses the filename, so an indexed path routinely has new
    /// bytes by the next sweep. A fresh UUID at that path collides with
    /// `absolute_path NOT NULL UNIQUE` — which `ON CONFLICT(id)` does NOT
    /// absorb — and the throw would abort the sweep partway through.
    func testChangedBytesUpdateTheRowInPlaceRatherThanAbortingTheSweep() async throws {
        try write("a.png", bytes: "V1", tree: "home")
        try write("Kira/generated/b.png", bytes: "B", tree: "kira/gallery")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let foundFirstID = try await store.assetID(forPath: root + "/home/a.png")
        let firstID = try XCTUnwrap(foundFirstID)

        try write("a.png", bytes: "V2 IS LONGER", tree: "home")
        let second = try await CatalogBackfill.run(store: store, trees: trees())

        let rows = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertEqual(rows.count, 2, "the changed file is the same asset, not a second one")
        XCTAssertEqual(second.assetsIndexed, 0, "an update in place is not a new asset")
        let foundAgain = try await store.assetID(forPath: root + "/home/a.png")
        let again = try XCTUnwrap(foundAgain)
        XCTAssertEqual(again, firstID, "the row keeps its identity")
        let foundUpdated = try await store.asset(id: firstID)
        let updated = try XCTUnwrap(foundUpdated)
        XCTAssertEqual(updated.fileSize, Int64("V2 IS LONGER".utf8.count))
        let locations = try await store.locations(of: firstID)
        XCTAssertEqual(locations.count, 1, "no duplicate location for the same path")
    }

    /// A DOWNSTREAM copy whose bytes diverge is a new asset, not an update to
    /// the row it used to duplicate. Reusing the row the path is merely a
    /// LOCATION of would drag that row's `absolute_path` across to the copy.
    func testAChangedDownstreamCopyBecomesItsOwnRow() async throws {
        let homePath = root + "/home/shot.png"
        let copyPath = root + "/kira/gallery/Kira/generated/copy.png"
        try write("shot.png", bytes: "SAME", tree: "home")
        try write("Kira/generated/copy.png", bytes: "SAME", tree: "kira/gallery")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let foundHomeID = try await store.assetID(forPath: homePath)
        let homeID = try XCTUnwrap(foundHomeID)

        // The copy diverges — a re-render into the same server filename.
        try write("Kira/generated/copy.png", bytes: "DIVERGED", tree: "kira/gallery")
        let second = try await CatalogBackfill.run(store: store, trees: trees())

        let rows = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertEqual(rows.count, 2, "the diverged copy is its own asset")
        XCTAssertEqual(second.assetsIndexed, 1)
        let foundOriginal = try await store.asset(id: homeID)
        let original = try XCTUnwrap(foundOriginal)
        XCTAssertEqual(original.absolutePath, homePath, "the original keeps its own path")
        let foundCopyID = try await store.assetID(owningPath: copyPath)
        let copyID = try XCTUnwrap(foundCopyID)
        XCTAssertNotEqual(copyID, homeID)
    }

    /// The state the test above CREATES is the dangerous one: a row owning P and
    /// a DIFFERENT row holding a stale location at P. `assetID(forPath:)` is a
    /// UNION over both, and a compound UNION dedups through a temp b-tree, so
    /// under LIMIT 1 it can hand back the row that merely has the location.
    /// Reuse then looks impossible, a fresh UUID is minted, and the INSERT
    /// violates `absolute_path NOT NULL UNIQUE` — aborting the whole sweep.
    ///
    /// The ids are seeded by hand rather than left to UUIDs precisely because
    /// that b-tree orders by id: "aaa-…" must come back first for the hazard to
    /// be reproduced deterministically instead of half the time.
    func testAStaleLocationCannotStealTheReuseLookupAndAbortTheSweep() async throws {
        let contested = root + "/home/contested.png"
        try write("contested.png", bytes: "ORIGINAL", tree: "home")

        // The row that OWNS the contested path — id sorts last.
        try await store.upsert(CatalogAsset(
            id: "zzz-owner", filename: "contested.png", absolutePath: contested,
            sha256: "stale-sha", realm: .shared), explicitCollectionIDs: [])
        // An unrelated row that merely has a stale LOCATION there — id sorts first.
        try await store.upsert(CatalogAsset(
            id: "aaa-other", filename: "other.png", absolutePath: root + "/home/other.png",
            sha256: "other-sha", realm: .shared), explicitCollectionIDs: [])
        try await store.addLocation(assetID: "aaa-other",
            AssetLocation(host: "kira", path: contested, mtime: Date()))

        // The contested file's bytes are not either recorded sha, so the sweep
        // takes the reuse path for it.
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let foundOwner = try await store.asset(id: "zzz-owner")
        let owner = try XCTUnwrap(foundOwner)
        XCTAssertEqual(owner.absolutePath, contested, "the owner still owns the path")
        XCTAssertNotEqual(owner.sha256, "stale-sha", "and was updated in place")
        let foundOther = try await store.asset(id: "aaa-other")
        let other = try XCTUnwrap(foundOther)
        XCTAssertEqual(other.absolutePath, root + "/home/other.png", "the other row did not move")
        let owners = try await store.assetIDs(forFilename: "contested.png", limit: 10)
        XCTAssertEqual(owners, ["zzz-owner"], "exactly one row for the contested path")
    }

    /// Swift cannot check that a field added to `FileMetadata` is also mapped in
    /// `CatalogBackfill.row(...)` — every `CatalogAsset.init` parameter is
    /// defaulted, so forgetting one compiles clean and silently drops the fact.
    ///
    /// What this CATCHES: a field added to (or removed from) `FileMetadata`
    /// without a matching decision in `row`.
    /// What it does NOT catch: a field that exists and is mapped but that no
    /// READER ever populates — which is what actually happened to `lane` (it was
    /// already a property here; `readSidecar` simply never set it). Only a test
    /// that feeds a real fixture through a reader catches that, and
    /// `MetadataReaderTests.testSidecarSuppliesTheFilingLane` is that test.
    func testEveryFileMetadataFieldIsMapped() {
        let labels = Mirror(reflecting: FileMetadata()).children.compactMap(\.label).sorted()
        XCTAssertEqual(labels, [
            "arc", "aspectRatio", "characterName", "contentMode", "durationMs",
            "family", "fps", "frames", "genre", "guidance", "height", "lane",
            "loras", "mode", "modelFamily", "negativePrompt", "preset", "prompt",
            "promptInjected", "promptRaw", "provider", "renderID", "resolution",
            "sealed", "seed", "software", "sourceImagePath", "steps", "stock",
            "style", "theme", "width",
        ], """
        A field was added to or removed from FileMetadata. Map it in \
        CatalogBackfill.row(file:existing:meta:) — or, if it deliberately has no \
        home on CatalogAsset (sourceImagePath, software, provider and sealed do \
        not), say so there — then update this list.
        """)
    }

    // MARK: - Server paths (source_image lives in the server's namespace)

    /// A tree that knows what it is called on its own host.
    private func translatingTrees() -> [BackfillTree] {
        [
            BackfillTree(id: "home", realm: nil, host: "mac",
                         mediaRoot: root + "/home", metadataRoot: nil),
            BackfillTree(id: "kira", realm: .kira, host: "kira",
                         mediaRoot: root + "/kira/gallery",
                         metadataRoot: root + "/kira/metadata",
                         remotePrefix: "/home/todd/.kira/studio/gallery"),
        ]
    }

    /// Real sidecars record `source_image` as a path on the SERVER
    /// (`/home/todd/.kira/studio/gallery/...`), which exists nowhere on this
    /// Mac. Without translation every i2v edge resolves zero times in
    /// production. A second file shares the basename here specifically so the
    /// last-resort basename match CANNOT be what resolves it.
    func testAServerSideSourceImageResolvesThroughTheTreesRemotePrefix() async throws {
        let still = try write("Kira/generated/still.png", bytes: "STILL", tree: "kira/gallery")
        try write("Kira/other/still.png", bytes: "DECOY", tree: "kira/gallery")
        try write("Kira/video/clip.mp4", bytes: "CLIP", tree: "kira/gallery")
        try writeSidecar("Kira/video/clip.json", json: """
            {"character":"kira","mode":"i2v",
             "source_image":"/home/todd/.kira/studio/gallery/Kira/generated/still.png"}
            """, tree: "kira/metadata")

        let report = try await CatalogBackfill.run(store: store, trees: translatingTrees())
        XCTAssertEqual(report.edgesCreated, 1)
        XCTAssertEqual(report.edgesUnresolved, 0)

        let foundClipID = try await store.assetID(forPath: root + "/kira/gallery/Kira/video/clip.mp4")
        let clipID = try XCTUnwrap(foundClipID)
        let foundStillID = try await store.assetID(forPath: still)
        let stillID = try XCTUnwrap(foundStillID)
        let edges = try await store.edges(for: clipID)
        XCTAssertEqual(edges.first?.toAssetID, stillID, "the edge must point at the translated still")
    }

    /// Same fixture, no `remotePrefix`, and the basename is ambiguous. A wrong
    /// edge is worse than a missing one, so this must skip rather than guess —
    /// and the skip must be VISIBLE in the report.
    func testAnAmbiguousBasenameIsSkippedRatherThanMisLinked() async throws {
        try write("Kira/generated/still.png", bytes: "STILL", tree: "kira/gallery")
        try write("Kira/other/still.png", bytes: "DECOY", tree: "kira/gallery")
        try write("Kira/video/clip.mp4", bytes: "CLIP", tree: "kira/gallery")
        try writeSidecar("Kira/video/clip.json", json: """
            {"mode":"i2v","source_image":"/home/todd/.kira/studio/gallery/Kira/generated/still.png"}
            """, tree: "kira/metadata")

        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.edgesCreated, 0)
        XCTAssertEqual(report.edgesUnresolved, 1)
    }

    /// The last resort itself: an untranslatable server path whose basename
    /// names exactly one asset is allowed to resolve.
    func testAnUnambiguousBasenameIsTheLastResort() async throws {
        let still = try write("Kira/generated/unique-still.png", bytes: "STILL", tree: "kira/gallery")
        try write("Kira/video/clip.mp4", bytes: "CLIP", tree: "kira/gallery")
        try writeSidecar("Kira/video/clip.json", json: """
            {"mode":"i2v","source_image":"/home/todd/.kira/studio/gallery/Deep/Elsewhere/unique-still.png"}
            """, tree: "kira/metadata")

        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.edgesCreated, 1)
        let foundClipID = try await store.assetID(forPath: root + "/kira/gallery/Kira/video/clip.mp4")
        let clipID = try XCTUnwrap(foundClipID)
        let foundStillID = try await store.assetID(forPath: still)
        let stillID = try XCTUnwrap(foundStillID)
        let edges = try await store.edges(for: clipID)
        XCTAssertEqual(edges.first?.toAssetID, stillID)
    }

    // MARK: - Provenance and precedence

    /// `EXIF:Software` is a human display string ("CoffeeShop Desktop
    /// (ComfyBox)") that matches no rule in CollectionRules; the sidecar's
    /// `provider` is the key that does. Taking software first turned `source`
    /// into a display name and unfiled every shared-realm asset.
    func testSidecarProviderWinsOverEmbeddedSoftware() async throws {
        var meta = FileMetadata()
        meta.software = "CoffeeShop Desktop (ComfyBox)"
        meta.provider = "Krita"
        XCTAssertEqual(CatalogBackfill.sourceLabel([meta]), "krita")

        // And end to end: a shared asset files by `source` alone.
        try write("tile.png", bytes: "K", tree: "home")
        try writeSidecar("tile.json", json: #"{"provider":"krita"}"#, tree: "home-meta")
        let homeWithSidecars = [
            BackfillTree(id: "home", realm: nil, host: "mac",
                         mediaRoot: root + "/home", metadataRoot: root + "/home-meta"),
        ]
        _ = try await CatalogBackfill.run(store: store, trees: homeWithSidecars)
        let shared = try await store.search(CatalogQuery(scope: .shared, collectionID: "col-decoupage"))
        XCTAssertEqual(shared.count, 1)
        XCTAssertEqual(shared.first?.source, "krita")
    }

    /// Sweeping a shared-realm tree AFTER hers must not take her asset out of
    /// her realm: her renders are mirrored onto shared hosts too, so a copy
    /// there is not evidence of shared ownership.
    func testASharedTreeSweptLaterDoesNotDemoteHerRealm() async throws {
        try write("Kira/generated/k.png", bytes: "SAME", tree: "kira/gallery")
        try write("Bree/generated/copy.png", bytes: "SAME", tree: "bree/gallery")
        let ordered = [
            BackfillTree(id: "kira", realm: .kira, host: "kira",
                         mediaRoot: root + "/kira/gallery", metadataRoot: root + "/kira/metadata"),
            BackfillTree(id: "bree", realm: .shared, host: "bree",
                         mediaRoot: root + "/bree/gallery", metadataRoot: nil),
        ]
        _ = try await CatalogBackfill.run(store: store, trees: ordered)

        let kira = try await store.search(CatalogQuery(scope: .kira, limit: 500))
        let shared = try await store.search(CatalogQuery(scope: .shared, limit: 500))
        XCTAssertEqual(kira.count, 1, "she keeps the asset")
        XCTAssertEqual(shared.count, 0, "a copy on a shared host is not a demotion")
    }

    /// Idempotence THROUGH the dedup + sidecar path — the one the earlier
    /// idempotence test never reaches, because it has no cross-tree duplicate.
    func testASecondSweepChangesNothingOnADeduplicatedRow() async throws {
        try write("shot.png", bytes: "SAME", tree: "home")
        try write("Kira/generated/renamed.png", bytes: "SAME", tree: "kira/gallery")
        try writeSidecar("Kira/generated/renamed.json",
                         json: #"{"character":"kira","lane":"tile","content_mode":"neutral"}"#,
                         tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())
        let before = try await store.search(CatalogQuery(scope: nil, limit: 500))
        let beforeLocations = try await store.locations(of: before[0].id).count
        let beforeFiled = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))

        let second = try await CatalogBackfill.run(store: store, trees: trees())
        let after = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertEqual(after, before, "a re-sweep must not churn the row")
        XCTAssertEqual(second.assetsIndexed, 0)
        let afterLocations = try await store.locations(of: after[0].id).count
        XCTAssertEqual(afterLocations, beforeLocations)
        let afterFiled = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        XCTAssertEqual(afterFiled.count, beforeFiled.count)
    }

    /// First non-nil wins, forever: a later tree's copy fills gaps but never
    /// overwrites a fact an earlier tree already supplied.
    func testALaterTreeCannotOverwriteAnEarlierTreesFacts() async throws {
        try write("shot.png", bytes: "SAME", tree: "home")
        try writeSidecar("shot.json", json: #"{"preset":"home-preset"}"#, tree: "home-meta")
        try write("Kira/generated/renamed.png", bytes: "SAME", tree: "kira/gallery")
        try writeSidecar("Kira/generated/renamed.json",
                         json: #"{"preset":"kira-preset","lane":"tile"}"#, tree: "kira/metadata")

        let ordered = [
            BackfillTree(id: "home", realm: nil, host: "mac",
                         mediaRoot: root + "/home", metadataRoot: root + "/home-meta"),
            BackfillTree(id: "kira", realm: .kira, host: "kira",
                         mediaRoot: root + "/kira/gallery", metadataRoot: root + "/kira/metadata"),
        ]
        _ = try await CatalogBackfill.run(store: store, trees: ordered)

        let rows = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].preset, "home-preset", "the first tree's fact survives")
        XCTAssertEqual(rows[0].lane, "tile", "but a gap is still filled")
        XCTAssertEqual(rows[0].realm, .kira, "and the realm still settles to hers")
    }

    // MARK: - Journals, the third source

    private func journalTrees(journal: String? = nil, history: String? = nil) -> [BackfillTree] {
        [
            BackfillTree(id: "kira", realm: .kira, host: "kira",
                         mediaRoot: root + "/kira/gallery",
                         metadataRoot: root + "/kira/metadata",
                         remotePrefix: "/home/todd/.kira/studio/gallery",
                         journalPath: journal, historyPath: history),
        ]
    }

    /// `lane` is in only 131/400 image sidecars, so for most of the fleet the
    /// journal is the only thing that knows where an asset belongs.
    func testTheJournalSuppliesTheLaneWhenNoSidecarDoes() async throws {
        try write("Kira/generated/j.png", bytes: "J", tree: "kira/gallery")
        let journal = try write("render-journal.jsonl", bytes: """
            {"ts":1,"tier":"neutral","lane":"tile","theme":"balcony","path":"/home/todd/.kira/studio/gallery/Kira/generated/j.png","seed":{"stock":"portra","style":"soft","genre":"portrait","family":"nightlife"}}
            """, tree: "kira")
        let report = try await CatalogBackfill.run(store: store, trees: journalTrees(journal: journal))
        XCTAssertEqual(report.journalEntriesRead, 1)

        let filed = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        XCTAssertEqual(filed.count, 1, "the journal's lane files the asset")
        XCTAssertEqual(filed.first?.theme, "balcony")
        XCTAssertEqual(filed.first?.stock, "portra")
        XCTAssertEqual(filed.first?.contentMode, "neutral")
    }

    /// The journal is the WEAKEST source: it never overrides the sidecar.
    func testTheSidecarBeatsTheJournal() async throws {
        try write("Kira/generated/j.png", bytes: "J", tree: "kira/gallery")
        try writeSidecar("Kira/generated/j.json", json: #"{"lane":"shoot","content_mode":"apple"}"#,
                         tree: "kira/metadata")
        let journal = try write("render-journal.jsonl", bytes: """
            {"ts":1,"tier":"banana","lane":"tile","path":"/home/todd/.kira/studio/gallery/Kira/generated/j.png"}
            """, tree: "kira")
        _ = try await CatalogBackfill.run(store: store, trees: journalTrees(journal: journal))

        let rows = try await store.search(CatalogQuery(scope: .kira, limit: 500))
        XCTAssertEqual(rows.first?.lane, "shoot")
        XCTAssertEqual(rows.first?.contentMode, "apple")
    }

    /// "tier" is overloaded: the journal means the fruit tier, but sidecars use
    /// the same word for a QUALITY tier. An unrecognised content_mode ranks
    /// above every ceiling (tierRank fails closed), so it must not be adopted.
    func testAJournalTierThatIsNotAFruitTierIsIgnored() async throws {
        try write("Kira/generated/j.png", bytes: "J", tree: "kira/gallery")
        let journal = try write("render-journal.jsonl", bytes: """
            {"ts":1,"tier":"standard","lane":"tile","path":"/home/todd/.kira/studio/gallery/Kira/generated/j.png"}
            """, tree: "kira")
        _ = try await CatalogBackfill.run(store: store, trees: journalTrees(journal: journal))
        let rows = try await store.search(CatalogQuery(scope: .kira, limit: 500))
        XCTAssertEqual(rows.first?.lane, "tile", "the lane is still taken")
        XCTAssertNil(rows.first?.contentMode, "a quality tier is not a content mode")
    }

    func testHistorySuppliesCharacterPromptAndProvider() async throws {
        try write("Kira/generated/h.png", bytes: "H", tree: "kira/gallery")
        let history = try write("history.json", bytes: """
            {"version":1,"records":[
              {"id":"r-1","prompt":"a quiet balcony","character":"kira","contentMode":"neutral",
               "width":576,"height":1024,"steps":9,"seed":42,"provider":"comfybox","durationMs":84944,
               "outputPath":"/home/todd/.kira/studio/gallery/Kira/generated/h.png"}]}
            """, tree: "kira")
        _ = try await CatalogBackfill.run(store: store, trees: journalTrees(history: history))

        let rows = try await store.search(CatalogQuery(scope: .kira, limit: 500))
        XCTAssertEqual(rows.first?.characterName, "kira")
        XCTAssertEqual(rows.first?.prompt, "a quiet balcony")
        XCTAssertEqual(rows.first?.source, "comfybox")
        XCTAssertEqual(rows.first?.seed, 42)
        XCTAssertNil(rows.first?.durationMs,
                     "history's durationMs is render time, not clip length — it must not become a duration")
    }

    /// The journal and history know DISJOINT things about the same render — the
    /// journal has the lane, history has the character — so an output named in
    /// both must end up with both. Letting one stand in for the other drops the
    /// filing key of whichever lost.
    func testAJournalLineAndAHistoryRecordForOneOutputBothSurvive() async throws {
        try write("Kira/generated/both.png", bytes: "B", tree: "kira/gallery")
        let journal = try write("render-journal.jsonl", bytes: """
            {"ts":1,"tier":"neutral","lane":"tile","theme":"balcony","path":"/home/todd/.kira/studio/gallery/Kira/generated/both.png"}
            """, tree: "kira")
        let history = try write("history.json", bytes: """
            {"version":1,"records":[
              {"id":"r-9","prompt":"a quiet balcony","character":"kira","provider":"comfybox",
               "seed":7,"outputPath":"/home/todd/.kira/studio/gallery/Kira/generated/both.png"}]}
            """, tree: "kira")
        let report = try await CatalogBackfill.run(
            store: store, trees: journalTrees(journal: journal, history: history))
        XCTAssertEqual(report.journalEntriesRead, 2)

        let rows = try await store.search(CatalogQuery(scope: .kira, limit: 500))
        XCTAssertEqual(rows.first?.lane, "tile", "the journal's lane")
        XCTAssertEqual(rows.first?.theme, "balcony")
        XCTAssertEqual(rows.first?.characterName, "kira", "AND history's character")
        XCTAssertEqual(rows.first?.prompt, "a quiet balcony")
        XCTAssertEqual(rows.first?.seed, 7)
        let filed = try await store.search(CatalogQuery(scope: .kira, collectionID: "col-kira-decoupage"))
        XCTAssertEqual(filed.count, 1)
    }

    /// `provider` and `software` never live in the same source: only embedded
    /// EXIF sets software, only the sidecar and history set provider. So the
    /// label has to be resolved across ALL of them — otherwise the first source
    /// carrying EITHER wins, an EXIF display string lands in `source`, and the
    /// asset files nowhere, since `source` is a shared asset's only filing input.
    func testAProviderFromAnySourceBeatsSoftwareFromAnySource() async throws {
        var embedded = FileMetadata()
        embedded.software = "CoffeeShop Desktop (ComfyBox)"
        var fromHistory = FileMetadata()
        fromHistory.provider = "Krita"
        XCTAssertEqual(CatalogBackfill.sourceLabel([embedded, fromHistory]), "krita",
                       "embedded is FIRST and still loses — provider is the filing key")

        // And through the fold, which is where resolving per source would put
        // the EXIF display string in the column: embedded carries software and
        // is applied first, history carries provider and is applied last.
        let identity = CatalogBackfill.FileFacts(
            id: "a1", kind: "image", filename: "shot.png", absolutePath: "/tmp/shot.png",
            sha256: "sha", fileSize: 3, createdAt: Date(), realm: .shared)
        let folded = CatalogBackfill.fold(identity: identity, existing: nil,
                                          sources: [embedded, fromHistory])
        XCTAssertEqual(folded.source, "krita",
                       "the label is resolved across ALL sources, not folded per source")

        // End to end, with the tree that has no sidecars at all (the Mac home
        // gallery) and a history record that names the producer.
        try write("shot.png", bytes: "S", tree: "home")
        let history = try write("history.json", bytes: """
            {"version":1,"records":[
              {"id":"r-1","provider":"krita","outputPath":"\(root!)/home/shot.png"}]}
            """, tree: "home-journals")
        let homeTree = [
            BackfillTree(id: "home", realm: nil, host: "mac",
                         mediaRoot: root + "/home", metadataRoot: nil,
                         historyPath: history),
        ]
        _ = try await CatalogBackfill.run(store: store, trees: homeTree)
        let filed = try await store.search(CatalogQuery(scope: .shared, collectionID: "col-decoupage"))
        XCTAssertEqual(filed.count, 1, "history's provider files the asset")
        XCTAssertEqual(filed.first?.source, "krita")
    }

    /// The basename fallback is scoped to the CLIP's realm. Here the only asset
    /// with that basename lives in the other realm, so the guess must fail —
    /// unscoped it would mint an edge across the boundary.
    func testTheBasenameFallbackWillNotReachIntoAnotherRealm() async throws {
        try write("shared-still.png", bytes: "SHARED", tree: "home")
        try write("Kira/video/clip.mp4", bytes: "CLIP", tree: "kira/gallery")
        try writeSidecar("Kira/video/clip.json", json: """
            {"mode":"i2v","source_image":"/home/todd/.kira/studio/gallery/Kira/generated/shared-still.png"}
            """, tree: "kira/metadata")

        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.edgesCreated, 0, "her clip may not be linked to a shared still by a guess")
        XCTAssertEqual(report.edgesUnresolved, 1)
    }

    /// `contentMode` from history gets the same fruit-tier gate the journal's
    /// `tier` does: it is a weak source, and an unrecognised value ranks above
    /// EVERY ceiling (tierRank fails closed), withholding the asset from
    /// everyone rather than from nobody.
    func testAHistoryContentModeThatIsNotAFruitTierIsIgnored() async throws {
        try write("Kira/generated/h.png", bytes: "H", tree: "kira/gallery")
        let history = try write("history.json", bytes: """
            {"version":1,"records":[
              {"id":"r-1","character":"kira","contentMode":"premium",
               "outputPath":"/home/todd/.kira/studio/gallery/Kira/generated/h.png"}]}
            """, tree: "kira")
        _ = try await CatalogBackfill.run(store: store, trees: journalTrees(history: history))

        let rows = try await store.search(CatalogQuery(scope: .kira, limit: 500))
        XCTAssertEqual(rows.first?.characterName, "kira", "the rest of the record is still taken")
        XCTAssertNil(rows.first?.contentMode, "an unrecognised tier is not a content mode")
    }

    // MARK: - Prompts, coverage and junk files

    /// Image sidecars have NO `prompt` key: they carry prompt_optimized,
    /// prompt_raw and prompt_injected. All three are indexed.
    func testAllThreePromptSpellingsAreCapturedAndSearchable() async throws {
        try write("Kira/generated/p.png", bytes: "P", tree: "kira/gallery")
        try writeSidecar("Kira/generated/p.json", json: """
            {"prompt_optimized":"golden hour balcony","prompt_raw":"balcony raw words",
             "prompt_injected":"balcony with injected trigger"}
            """, tree: "kira/metadata")
        _ = try await CatalogBackfill.run(store: store, trees: trees())

        let rows = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertEqual(rows.first?.prompt, "golden hour balcony")
        XCTAssertEqual(rows.first?.promptRaw, "balcony raw words")
        XCTAssertEqual(rows.first?.promptInjected, "balcony with injected trigger")

        for phrase in ["golden hour", "raw words", "injected trigger"] {
            let hits = try await store.search(CatalogQuery(scope: nil, text: phrase))
            XCTAssertEqual(hits.count, 1, "\(phrase) must be searchable")
        }
    }

    /// Every truncated render hashes to the SAME sha256, so indexing them would
    /// collapse the lot into one bogus asset with a location per failure.
    func testZeroByteFilesAreSkipped() async throws {
        try write("good.png", bytes: "G", tree: "home")
        try write("truncated.png", bytes: "", tree: "home")
        let report = try await CatalogBackfill.run(store: store, trees: trees())
        let rows = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(report.skipped, 1)
    }

    /// The coverage gap has to be a number in the report, not an absence.
    func testUnfiledAssetsAreCounted() async throws {
        try write("nowhere.png", bytes: "N", tree: "home")
        try write("Kira/generated/filed.png", bytes: "F", tree: "kira/gallery")
        try writeSidecar("Kira/generated/filed.json", json: #"{"lane":"tile"}"#, tree: "kira/metadata")
        let report = try await CatalogBackfill.run(store: store, trees: trees())
        XCTAssertEqual(report.assetsIndexed, 2)
        XCTAssertEqual(report.assetsUnfiled, 1)
    }

    /// The vault is out of scope. Assert BEHAVIOUR, not the fixture: a tree
    /// pointing into a vault must be refused, and nothing under it read.
    func testBackfillRefusesAVaultTree() async throws {
        let vaultDir = root + "/Documents/Vaults/BarkadaAI"
        try FileManager.default.createDirectory(atPath: vaultDir, withIntermediateDirectories: true)
        let victim = vaultDir + "/private.png"
        try Data("SECRET".utf8).write(to: URL(fileURLWithPath: victim))
        let before = try FileManager.default.attributesOfItem(atPath: victim)[.modificationDate] as? Date

        let bad = BackfillTree(id: "vault", realm: .shared, host: "kira",
                               mediaRoot: vaultDir, metadataRoot: nil)
        await XCTAssertThrowsErrorAsync(
            _ = try await CatalogBackfill.run(store: store, trees: trees() + [bad]))

        // Nothing indexed from it, and the file itself untouched.
        let rows = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertFalse(rows.contains { $0.absolutePath.contains("Vaults") })
        let after = try FileManager.default.attributesOfItem(atPath: victim)[.modificationDate] as? Date
        XCTAssertEqual(before, after)
    }

    /// Checking only the ROOT is not enough. `/Volumes/todd` is a real mount of
    /// the server home on this machine and `/Volumes/todd/Documents/Vaults/
    /// BarkadaAI` sits underneath it, so a tree with a perfectly innocent root
    /// can still contain the vault. Every path is checked, not just the root.
    func testAVaultSubtreeUnderACleanRootIsNotIndexed() async throws {
        try write("clean.png", bytes: "C", tree: "home")
        try write("Documents/Vaults/BarkadaAI/private.png", bytes: "SECRET", tree: "home")

        let report = try await CatalogBackfill.run(store: store, trees: trees())
        let rows = try await store.search(CatalogQuery(scope: nil, limit: 500))
        XCTAssertEqual(rows.count, 1, "only the clean file")
        XCTAssertFalse(rows.contains { $0.absolutePath.contains("Vaults") })
        XCTAssertEqual(report.skipped, 1, "the vault file is skipped, and visibly so")
    }
}
