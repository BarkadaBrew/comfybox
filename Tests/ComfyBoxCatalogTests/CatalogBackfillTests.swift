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
}
